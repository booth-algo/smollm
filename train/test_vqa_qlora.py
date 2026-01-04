#!/usr/bin/env python3
"""
Test script for QLoRA-trained SmolVLM2-256M VQA model
Loads base model + LoRA adapter and evaluates on VQAv2
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import time
import statistics
import sys

print("=" * 80)
print("SmolVLM2-256M VQA Testing (QLoRA Model)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    ADAPTER_PATH = sys.argv[1]
else:
    ADAPTER_PATH = "./smolvlm2-256m-vqa-qlora-final"

BASE_MODEL = "HuggingFaceTB/SmolVLM2-256M-Instruct"
NUM_TEST_SAMPLES = 100

print(f"🎯 Base model: {BASE_MODEL}")
print(f"🎯 Adapter path: {ADAPTER_PATH}")

# ============================================================
# Load Model & Processor
# ============================================================
print(f"\n📦 Loading base model: {BASE_MODEL}")
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
    device_map="auto"
)

print(f"📦 Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print(f"📦 Loading processor from: {ADAPTER_PATH}")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

print(f"✅ Model loaded on: {next(model.parameters()).device}")
print(f"📏 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ============================================================
# Load Test Dataset
# ============================================================
print("\n📊 Loading VQAv2 test dataset...")
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
split_ds = ds["validation"].train_test_split(test_size=0.1, seed=42)
test_ds = split_ds["test"]

print(f"✅ Test samples available: {len(test_ds)}")
print(f"🎯 Testing on: {min(NUM_TEST_SAMPLES, len(test_ds))} samples")

# ============================================================
# Helper Functions
# ============================================================
def prepare_input(example):
    """Prepare input for the model"""
    image = example["image"]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    question = example["question"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer briefly."},
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    return inputs, question, example["multiple_choice_answer"]

def generate_answer(model, processor, inputs, max_new_tokens=50):
    """Generate answer from model"""
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = generated_ids[:, input_length:]
    return generated_tokens

# ============================================================
# Warm-up GPU
# ============================================================
print("\n🔥 Warming up GPU...")
warmup_inputs, _, _ = prepare_input(test_ds[0])
with torch.no_grad():
    for _ in range(3):
        _ = model.generate(**warmup_inputs, max_new_tokens=50)
torch.cuda.synchronize()
print("✅ GPU warmed up")

# ============================================================
# Run Inference Tests
# ============================================================
print("\n" + "=" * 80)
print("RUNNING INFERENCE TESTS")
print("=" * 80)

results = []
correct = 0
total = 0

print(f"\n🧪 Running inference on {min(NUM_TEST_SAMPLES, len(test_ds))} samples...")

start_time = time.time()
for i in tqdm(range(min(NUM_TEST_SAMPLES, len(test_ds)))):
    example = test_ds[i]
    inputs, question, ground_truth = prepare_input(example)

    # Generate answer
    generated_tokens = generate_answer(model, processor, inputs, max_new_tokens=50)

    # Decode
    generated_text_with_tokens = processor.batch_decode(generated_tokens, skip_special_tokens=False)[0]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    answer = generated_text.strip()

    # Debug first few
    if i < 3:
        print(f"\nDebug sample {i}:")
        print(f"  Question: {question}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Generated: {repr(answer)}")

    # Check accuracy
    if answer.strip() == "":
        is_correct = False
    else:
        is_correct = ground_truth.lower() in answer.lower() or answer.lower() in ground_truth.lower()

    if is_correct:
        correct += 1
    total += 1

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "prediction": answer,
        "correct": is_correct
    })

end_time = time.time()
total_time = end_time - start_time

# ============================================================
# Print Results
# ============================================================
print("\n" + "=" * 80)
print("INFERENCE RESULTS")
print("=" * 80)
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%")
print(f"Total time: {total_time:.2f}s")
print(f"Average time per sample: {total_time/total*1000:.2f}ms")
print(f"Throughput: {total/total_time:.2f} samples/sec")

# Show examples
print("\n📝 Sample Predictions:")
for i in range(min(10, len(results))):
    r = results[i]
    status = "✅" if r["correct"] else "❌"
    print(f"\n{status} Question: {r['question']}")
    print(f"   Ground Truth: {r['ground_truth']}")
    print(f"   Prediction: {r['prediction']}")

# Save results
results_file = "test_vqa_qlora_results.json"
with open(results_file, 'w') as f:
    json.dump({
        "accuracy": correct/total*100,
        "total_samples": total,
        "correct": correct,
        "results": results
    }, f, indent=2)
print(f"\n💾 Results saved to: {results_file}")

# ============================================================
# Latency Benchmark
# ============================================================
print("\n" + "=" * 80)
print("LATENCY BENCHMARK")
print("=" * 80)

torch.cuda.synchronize()
num_runs = 50
times = []

print(f"\n⏱️  Running {num_runs} inference passes...")

for i in range(num_runs):
    example = test_ds[i % len(test_ds)]
    inputs, _, _ = prepare_input(example)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))

print(f"\n{'Metric':<20} {'Value':<15}")
print("-" * 40)
print(f"{'Mean':<20} {statistics.mean(times):>12.2f}ms")
print(f"{'Median':<20} {statistics.median(times):>12.2f}ms")
print(f"{'Min':<20} {min(times):>12.2f}ms")
print(f"{'Max':<20} {max(times):>12.2f}ms")
print(f"{'Std Dev':<20} {statistics.stdev(times):>12.2f}ms")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)
print(f"\n📊 Accuracy: {correct/total*100:.2f}%")
print(f"⚡ Average Latency: {statistics.mean(times):.2f}ms")
print(f"🎯 Throughput: {1000/statistics.mean(times):.2f} samples/sec")
print(f"\n📁 Results saved to: {results_file}")
print("\n" + "=" * 80)
