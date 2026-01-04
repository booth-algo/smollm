#!/usr/bin/env python3
"""
SmolVLM2-256M VQA Fine-tuning with QLoRA
Direct port of vision/finetuning/Smol_VLM_FT.ipynb to standalone script

Uses QLoRA (4-bit quantization + LoRA adapters) for memory-efficient training.
This approach is more stable and less likely to cause model collapse.
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import wandb

print("=" * 80)
print("SmolVLM2-256M VQA Fine-tuning with QLoRA")
print("=" * 80)

# Initialize wandb (optional)
wandb.init(project="smolvlm2-finetuning", name="vqa-qlora-256m", mode="offline")

# ============================================================
# Configuration
# ============================================================
USE_QLORA = True  # Set to False for full fine-tuning
model_path = "HuggingFaceTB/SmolVLM2-256M-Instruct"

print(f"\nConfiguration:")
print(f"  • Model: {model_path}")
print(f"  • Method: {'QLoRA (4-bit + LoRA)' if USE_QLORA else 'Full fine-tuning'}")

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# ============================================================
# Load Processor
# ============================================================
print(f"\nLoading processor from: {model_path}")
processor = AutoProcessor.from_pretrained(model_path)

# ============================================================
# Load Model with QLoRA
# ============================================================
print(f"\nLoading model with QLoRA...")

if USE_QLORA:
    # QLoRA configuration - exactly from notebook
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        use_dora=False,  # DoRA disabled for QLoRA
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # Add LoRA adapters
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = model.get_nb_trainable_parameters()
    print(f"✅ Trainable parameters: {trainable_params[0]:,} / {trainable_params[1]:,}")
    print(f"   Trainable: {trainable_params[0]/trainable_params[1]*100:.2f}%")
else:
    # Full fine-tuning
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")

    # Optionally freeze vision encoder
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    print(f"✅ Model loaded on: {next(model.parameters()).device}")

print(f"📏 Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ============================================================
# Load Dataset
# ============================================================
print("\nLoading VQAv2 dataset...")
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)

# Split exactly like the notebook
split_ds = ds["validation"].train_test_split(test_size=0.5, seed=42)
train_ds = split_ds["train"]

print(f"✅ Training samples: {len(train_ds)}")
print(f"Example - Q: {train_ds[0]['question']}, A: {train_ds[0]['multiple_choice_answer']}")

# ============================================================
# Collate Function - EXACT COPY from notebook
# ============================================================
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

def collate_fn(examples):
    """
    Collate function - EXACTLY from the notebook
    Simple masking: only mask padding and image tokens
    """
    texts = []
    images = []

    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        question = example["question"]
        answer = example["multiple_choice_answer"]

        # Use chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    # Process batch
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Create labels - SIMPLE masking like notebook
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# ============================================================
# Training Arguments - EXACT COPY from notebook
# ============================================================
model_name = model_path.split("/")[-1]

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Increased from 4 to 8
    gradient_accumulation_steps=2,  # Reduced from 4 to 2 (same effective batch=16)
    warmup_steps=50,
    learning_rate=1e-4,  # Same as notebook (works with QLoRA!)
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    bf16=True,
    output_dir=f"./smolvlm2-256m-vqa-qlora",
    report_to="wandb",
    remove_unused_columns=False,
    gradient_checkpointing=True,

    # Performance optimizations
    logging_first_step=True,
    eval_strategy="no",  # Notebook doesn't use eval
    dataloader_num_workers=0,  # Use main process for data loading (avoid GPU contention)
    dataloader_pin_memory=False,  # Disable to reduce overhead
    dataloader_prefetch_factor=None,  # Disable prefetch when num_workers=0
)

print("\nTraining configuration:")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Batch size: {training_args.per_device_train_batch_size}")
print(f"  • Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  • Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  • Learning rate: {training_args.learning_rate}")
print(f"  • Optimizer: {training_args.optim}")
print(f"  • Precision: bfloat16")
print(f"  • Gradient checkpointing: enabled")

# ============================================================
# Initialize Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

# Print memory usage before training
if torch.cuda.is_available():
    print(f"\nGPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ============================================================
# Start Training
# ============================================================
print("\n" + "=" * 80)
print("🚀 Starting VQA training with QLoRA...")
print("=" * 80)
print("\n⚠️  NOTE: First iteration will be SLOW (60-90s) due to CUDA kernel compilation.")
print("    Subsequent iterations should be much faster (5-15s each).\n")
train_result = trainer.train()

# ============================================================
# Save Model
# ============================================================
print("\n💾 Saving model...")
output_dir = "./smolvlm2-256m-vqa-qlora-final"

if USE_QLORA:
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ LoRA adapters saved to: {output_dir}")
    print("   To use: Load base model + load adapter from this directory")
else:
    # Save full model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Full model saved to: {output_dir}")

# ============================================================
# Print Training Results
# ============================================================
print("\n" + "=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
print(f"Model saved to: {output_dir}")
print("=" * 80)

# ============================================================
# Test Inference
# ============================================================
print("\n🧪 Testing VQA inference...")
test_example = train_ds[0]
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

print("\n🔮 Generating answer...")
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

if generated_text.strip() == "":
    print("\n⚠️  WARNING: Model generated empty response!")
    print("   This might indicate training issues.")
else:
    print("\n✅ Model is generating text! Training successful.")

print("\n✅ All done! VQA model with QLoRA is ready to use.")

# ============================================================
# Usage Instructions
# ============================================================
print("\n" + "=" * 80)
print("📖 How to use the trained model:")
print("=" * 80)
if USE_QLORA:
    print(f"""
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Load base model
base_model = AutoModelForImageTextToText.from_pretrained(
    "{model_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{output_dir}")
processor = AutoProcessor.from_pretrained("{output_dir}")

# Now use model for inference!
""")
else:
    print(f"""
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("{output_dir}")
model = AutoModelForImageTextToText.from_pretrained(
    "{output_dir}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Now use model for inference!
""")
print("=" * 80)
