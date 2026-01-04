#!/usr/bin/env python3
"""
SmolVLM2-256M Visual Question Answering (VQA) Fine-tuning
Based on vision/finetuning/Smol_VLM_FT.ipynb

Trains on VQAv2 dataset with proper image handling
Perfect for RTX A6000 - optimized for fast training!
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import wandb

print("=" * 60)
print("SmolVLM2-256M VQA Fine-tuning")
print("=" * 60)

# Initialize wandb (optional)
wandb.init(project="smolvlm2-finetuning", name="vqa-finetuning-256m", mode="offline")

# Model setup
model_path = "HuggingFaceTB/SmolVLM2-256M-Instruct"  # Using base Instruct model for VQA
print(f"\nLoading model: {model_path}")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Use BF16 for better stability
    _attn_implementation="flash_attention_2",
).to("cuda")  # Explicit GPU placement

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

print(f"✅ Model loaded on: {next(model.parameters()).device}")
print(f"📏 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load VQAv2 dataset (small version for testing)
print("\nLoading VQAv2 dataset...")
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)

# Split dataset
split_ds = ds["validation"].train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]

print(f"✅ Training samples: {len(train_ds)}")
print(f"✅ Eval samples: {len(eval_ds)}")
print(f"Example - Q: {train_ds[0]['question']}, A: {train_ds[0]['multiple_choice_answer']}")

# Get image token ID for masking
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

def collate_fn(examples):
    """
    Collate function using chat template format (matches inference!)
    """
    texts = []
    images = []

    for ex in examples:
        q = ex["question"]
        a = ex["multiple_choice_answer"]

        # Use chat template format - SAME AS INFERENCE
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": q}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": a}
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False)
        texts.append(text)

        # Get image
        img = ex["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append([img])

    # Process batch
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Create labels - mask everything except assistant's response
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100

    # Mask the user's prompt (only train on assistant's answer)
    # Find "Assistant:" token in the actual token sequence
    assistant_token = "Assistant:"
    for i, input_ids in enumerate(batch["input_ids"]):
        # Find where "Assistant:" appears in the token sequence
        tokens_list = input_ids.tolist()

        # Decode to find the split point
        full_text = processor.tokenizer.decode(input_ids, skip_special_tokens=False)

        if "Assistant:" in full_text:
            # Find the token index where Assistant response starts
            # We want to mask everything BEFORE the assistant's actual answer
            split_idx = full_text.index("Assistant:") + len("Assistant:")

            # Binary search to find the token position
            for j in range(len(tokens_list)):
                decoded_so_far = processor.tokenizer.decode(tokens_list[:j+1], skip_special_tokens=False)
                if len(decoded_so_far) >= split_idx:
                    labels[i, :j+1] = -100
                    break

    batch["labels"] = labels

    return batch

# Training arguments - optimized for A6000
training_args = TrainingArguments(
    output_dir="./smolvlm2-256m-vqa",

    # Training schedule
    num_train_epochs=3,  # Increased from 1 to 3
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16

    # Optimization
    learning_rate=5e-6,  # MUCH lower learning rate (was 1e-4, catastrophic!)
    weight_decay=0.01,
    warmup_ratio=0.1,  # Warmup 10% of steps
    max_grad_norm=1.0,

    # Precision - MUST match model dtype!
    fp16=False,
    bf16=True,  # Enable BF16 to match model

    # Optimizer
    optim="adamw_torch",

    # Logging & saving
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,  # Must be multiple of eval_steps when load_best_model_at_end=True
    save_total_limit=2,
    metric_for_best_model="loss",  # Save based on lowest loss
    greater_is_better=False,

    # Performance
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,  # Enable for memory efficiency

    # Misc
    remove_unused_columns=False,
    report_to="wandb",
    load_best_model_at_end=True,
)

print("\nTraining configuration:")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Batch size: {training_args.per_device_train_batch_size}")
print(f"  • Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  • Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  • Learning rate: {training_args.learning_rate}")
print(f"  • Precision: bfloat16")
print(f"  • Gradient checkpointing: enabled")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# Print memory usage
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"\nGPU memory allocated: {peak_mem / 1024**3:.2f} GB")

# Start training
print("\n🚀 Starting VQA training...")
print("=" * 60)
train_result = trainer.train()

# Save the model
print("\n💾 Saving model...")
trainer.save_model("./smolvlm2-256m-vqa-final")
processor.save_pretrained("./smolvlm2-256m-vqa-final")

# Print metrics
print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
print(f"Model saved to: ./smolvlm2-256m-vqa-final")
print("="*60)

# Test inference on a sample
print("\n🧪 Testing VQA inference...")
test_example = eval_ds[0]
test_image = test_example["image"]
if test_image.mode != 'RGB':
    test_image = test_image.convert('RGB')
test_question = test_example["question"]
test_answer = test_example["multiple_choice_answer"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Answer briefly."},
            {"type": "image"},
            {"type": "text", "text": test_question}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[test_image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)

# Decode only new tokens
input_length = inputs['input_ids'].shape[1]
generated_tokens = generated_ids[:, input_length:]
generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(f"\n📝 Test Results:")
print(f"Question: {test_question}")
print(f"Ground Truth: {test_answer}")
print(f"Generated: {generated_text}")

print("\n✅ All done! VQA model is ready to use.")
