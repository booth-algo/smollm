#!/usr/bin/env python3
"""
SmolVLM2-256M VQA Inference Testing & Profiling
Tests fine-tuned model on VQAv2 test set with detailed layer-level profiling
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import json
import time
from collections import defaultdict
import re

print("=" * 80)
print("SmolVLM2-256M VQA Inference Testing & Profiling")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
import sys

# Allow command-line override of model path
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    MODEL_PATH = sys.argv[1]
else:
    MODEL_PATH = "./smolvlm2-256m-vqa-final"

NUM_TEST_SAMPLES = 100  # Number of samples to evaluate
MAX_NEW_TOKENS = 50
PROFILE_SAMPLES = 10  # Number of samples to profile in detail

print(f"🎯 Using model: {MODEL_PATH}")

# ============================================================
# Load Model & Processor
# ============================================================
print(f"\n📦 Loading model from: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Try to load with bfloat16 first (for models trained with bf16)
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")
    print("✅ Loaded model in bfloat16")
except Exception as e:
    print(f"⚠️ Could not load as bfloat16, trying float16: {e}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")
    print("✅ Loaded model in float16")

model.eval()  # Set to evaluation mode

print(f"✅ Model loaded on: {next(model.parameters()).device}")
print(f"📏 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Print model architecture summary
print("\n🏗️  Model Architecture:")
for name, module in model.named_children():
    print(f"  • {name}: {module.__class__.__name__}")

# ============================================================
# Load Test Dataset
# ============================================================
print("\n📊 Loading VQAv2 test dataset...")
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)

# Use validation split and further split for testing
split_ds = ds["validation"].train_test_split(test_size=0.1, seed=42)
test_ds = split_ds["test"]  # This is our test set

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
    """Generate answer from model and return only the new tokens"""
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Get only the newly generated tokens (exclude input)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = generated_ids[:, input_length:]

    return generated_tokens

def extract_answer(generated_text):
    """Extract answer from generated text"""
    # Try to extract after "Assistant:" marker
    if 'Assistant:' in generated_text:
        answer = generated_text.split('Assistant:')[-1].strip()
        # Remove end tokens
        answer = answer.replace('<end_of_utterance>', '').strip()
        return answer

    # If no Assistant marker, return the whole text (cleaned)
    answer = generated_text.strip()
    answer = answer.replace('<end_of_utterance>', '').strip()
    return answer

# ============================================================
# Warm-up GPU
# ============================================================
print("\n🔥 Warming up GPU...")
warmup_example = test_ds[0]
warmup_inputs, _, _ = prepare_input(warmup_example)

with torch.no_grad():
    for _ in range(3):
        _ = model.generate(**warmup_inputs, max_new_tokens=MAX_NEW_TOKENS)
torch.cuda.synchronize()
print("✅ GPU warmed up")

# ============================================================
# Part 1: Standard Inference Testing
# ============================================================
print("\n" + "=" * 80)
print("PART 1: STANDARD INFERENCE TESTING")
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
    generated_tokens = generate_answer(model, processor, inputs, MAX_NEW_TOKENS)

    # Decode the new tokens only
    generated_text_with_tokens = processor.batch_decode(generated_tokens, skip_special_tokens=False)[0]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    answer = extract_answer(generated_text)

    # Debug: print first few to see what's happening
    if i < 3:
        print(f"\nDebug sample {i}:")
        print(f"  Question: {question}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  With tokens: {repr(generated_text_with_tokens[:300])}")
        print(f"  Skip tokens: {repr(generated_text[:200])}")
        print(f"  Extracted answer: {repr(answer)}")

    # Simple accuracy check (case-insensitive contains)
    # Handle empty predictions as incorrect
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
        "raw_output": generated_text,  # Store raw output for debugging
        "correct": is_correct
    })

end_time = time.time()
total_time = end_time - start_time

# Print results
print("\n" + "=" * 80)
print("INFERENCE RESULTS")
print("=" * 80)
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%")
print(f"Total time: {total_time:.2f}s")
print(f"Average time per sample: {total_time/total*1000:.2f}ms")
print(f"Throughput: {total/total_time:.2f} samples/sec")

# Show some examples
print("\n📝 Sample Predictions:")
for i in range(min(5, len(results))):
    r = results[i]
    status = "✅" if r["correct"] else "❌"
    print(f"\n{status} Question: {r['question']}")
    print(f"   Ground Truth: {r['ground_truth']}")
    print(f"   Prediction: {r['prediction']}")

# Save results
results_file = "test_vqa_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Full results saved to: {results_file}")

# ============================================================
# Part 2: Detailed Layer-Level Profiling
# ============================================================
print("\n" + "=" * 80)
print("PART 2: DETAILED LAYER-LEVEL PROFILING")
print("=" * 80)

print(f"\n🔍 Profiling {PROFILE_SAMPLES} inference runs...")

# Profile with detailed layer information
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    with torch.no_grad():
        for i in range(PROFILE_SAMPLES):
            example = test_ds[i]
            inputs, _, _ = prepare_input(example)
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

# ============================================================
# Analyze Profiling Results by Layer Type
# ============================================================
print("\n" + "=" * 80)
print("TOP OPERATIONS BY CUDA TIME")
print("=" * 80)
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

print("\n" + "=" * 80)
print("TOP OPERATIONS BY CPU TIME")
print("=" * 80)
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=20
))

print("\n" + "=" * 80)
print("TOP OPERATIONS BY MEMORY USAGE")
print("=" * 80)
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=20
))

# ============================================================
# Aggregate by Layer Type
# ============================================================
print("\n" + "=" * 80)
print("BREAKDOWN BY LAYER/OPERATION TYPE")
print("=" * 80)

layer_stats = defaultdict(lambda: {"cpu_time": 0, "cuda_time": 0, "count": 0, "memory": 0})

for event in prof.key_averages():
    key = event.key

    # Categorize operations
    category = "Other"

    if "layernorm" in key.lower() or "layer_norm" in key.lower():
        category = "LayerNorm"
    elif "linear" in key.lower() or "addmm" in key.lower() or "matmul" in key.lower() or "gemm" in key.lower():
        category = "Linear/MatMul"
    elif "attention" in key.lower() or "softmax" in key.lower():
        category = "Attention"
    elif "gelu" in key.lower() or "silu" in key.lower() or "relu" in key.lower():
        category = "Activation"
    elif "conv" in key.lower():
        category = "Convolution"
    elif "embedding" in key.lower() or "embed" in key.lower():
        category = "Embedding"
    elif "dropout" in key.lower():
        category = "Dropout"
    elif "batch_norm" in key.lower() or "batchnorm" in key.lower():
        category = "BatchNorm"
    elif "vision" in key.lower() or "image" in key.lower():
        category = "Vision Encoder"
    elif "pool" in key.lower():
        category = "Pooling"

    layer_stats[category]["cpu_time"] += event.cpu_time_total
    layer_stats[category]["cuda_time"] += event.cuda_time_total
    layer_stats[category]["count"] += event.count
    layer_stats[category]["memory"] += event.cuda_memory_usage

# Sort by CUDA time
sorted_stats = sorted(layer_stats.items(), key=lambda x: x[1]["cuda_time"], reverse=True)

print(f"\n{'Category':<20} {'CUDA Time':<15} {'CPU Time':<15} {'Memory':<15} {'Count':<10} {'%Time':<10}")
print("-" * 100)

total_cuda_time = sum(stats["cuda_time"] for _, stats in sorted_stats)

for category, stats in sorted_stats:
    cuda_time_ms = stats["cuda_time"] / 1000  # Convert to ms
    cpu_time_ms = stats["cpu_time"] / 1000
    memory_mb = stats["memory"] / (1024 * 1024)  # Convert to MB
    percentage = (stats["cuda_time"] / total_cuda_time * 100) if total_cuda_time > 0 else 0

    print(f"{category:<20} {cuda_time_ms:>12.2f}ms {cpu_time_ms:>12.2f}ms {memory_mb:>12.2f}MB {stats['count']:>8} {percentage:>8.1f}%")

# ============================================================
# Export Traces for Visualization
# ============================================================
print("\n" + "=" * 80)
print("EXPORTING TRACE FILES")
print("=" * 80)

# Chrome trace
chrome_trace_file = "test_vqa_trace.json"
prof.export_chrome_trace(chrome_trace_file)
print(f"✅ Chrome trace saved to: {chrome_trace_file}")
print(f"   View at: chrome://tracing")

# ============================================================
# Additional Profiling with Custom Contexts
# ============================================================
print("\n" + "=" * 80)
print("DETAILED INFERENCE PIPELINE BREAKDOWN")
print("=" * 80)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        example = test_ds[0]
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        question = example["question"]

        with record_function("1_preprocessing"):
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

        with record_function("2_model_generation"):
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        with record_function("3_postprocessing"):
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(f"\n{'Stage':<25} {'CPU Time':<15} {'CUDA Time':<15}")
print("-" * 60)

for event in prof.key_averages():
    if event.key.startswith(("1_", "2_", "3_")):
        stage_name = event.key.split("_", 1)[1] if "_" in event.key else event.key
        cpu_time_ms = event.cpu_time_total / 1000
        cuda_time_ms = event.cuda_time_total / 1000
        print(f"{stage_name:<25} {cpu_time_ms:>12.2f}ms {cuda_time_ms:>12.2f}ms")

# ============================================================
# Simple Benchmark for Latency Statistics
# ============================================================
print("\n" + "=" * 80)
print("LATENCY BENCHMARK")
print("=" * 80)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

num_runs = 50
times = []

print(f"\n⏱️  Running {num_runs} inference passes for timing statistics...")

for i in range(num_runs):
    example = test_ds[i % len(test_ds)]
    inputs, _, _ = prepare_input(example)

    start_event.record()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))

import statistics

print(f"\n{'Metric':<20} {'Value':<15}")
print("-" * 40)
print(f"{'Runs':<20} {num_runs:<15}")
print(f"{'Mean':<20} {statistics.mean(times):>12.2f}ms")
print(f"{'Median':<20} {statistics.median(times):>12.2f}ms")
print(f"{'Min':<20} {min(times):>12.2f}ms")
print(f"{'Max':<20} {max(times):>12.2f}ms")
print(f"{'Std Dev':<20} {statistics.stdev(times):>12.2f}ms")
print(f"{'P95':<20} {sorted(times)[int(0.95*len(times))]:>12.2f}ms")
print(f"{'P99':<20} {sorted(times)[int(0.99*len(times))]:>12.2f}ms")

# Save timing stats
timing_file = "test_vqa_timing.json"
with open(timing_file, 'w') as f:
    json.dump({
        "num_runs": num_runs,
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times),
        "p95_ms": sorted(times)[int(0.95*len(times))],
        "p99_ms": sorted(times)[int(0.99*len(times))],
    }, f, indent=2)
print(f"\n💾 Timing statistics saved to: {timing_file}")

# ============================================================
# GPU Memory Analysis
# ============================================================
print("\n" + "=" * 80)
print("GPU MEMORY ANALYSIS")
print("=" * 80)

if torch.cuda.is_available():
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 55)
    print(f"{'Allocated Memory':<30} {torch.cuda.memory_allocated() / 1024**3:>17.2f} GB")
    print(f"{'Reserved Memory':<30} {torch.cuda.memory_reserved() / 1024**3:>17.2f} GB")
    print(f"{'Max Allocated Memory':<30} {torch.cuda.max_memory_allocated() / 1024**3:>17.2f} GB")
    print(f"{'Max Reserved Memory':<30} {torch.cuda.max_memory_reserved() / 1024**3:>17.2f} GB")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✅ TESTING & PROFILING COMPLETE!")
print("=" * 80)
print(f"\n📊 Accuracy: {correct/total*100:.2f}%")
print(f"⚡ Average Latency: {statistics.mean(times):.2f}ms")
print(f"🎯 Throughput: {1000/statistics.mean(times):.2f} samples/sec")
print(f"\n📁 Generated Files:")
print(f"   • {results_file} - Full test results")
print(f"   • {chrome_trace_file} - Chrome trace (view at chrome://tracing)")
print(f"   • {timing_file} - Latency statistics")
print("\n" + "=" * 80)
