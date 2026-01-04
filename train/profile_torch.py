#!/usr/bin/env python3
"""
Detailed Vision Encoder Profiling for SmolVLM2
Uses forward hooks to measure module-level timings

SmolVLM2 Vision Encoder Architecture:
- Patch Embedding: Conv2d(3→768, kernel=16x16, stride=16)
  Input: [B, 3, 512, 512] → Output: [B, 768, 32, 32]

- Position Embedding: Two-part operation
  1. Bucketization: Python for-loop computing 2D→1D position IDs (~48ms)
  2. Embedding lookup: nn.Embedding(1024, 768) table lookup (~1.4ms)
  NOT RoPE! Uses adaptive 2D→1D position bucketization (NaViT paper)

- Vision Transformer: 12 layers of:
  - LayerNorm (pre-attention)
  - Multi-head Attention (Flash Attention 2)
  - LayerNorm (pre-MLP)
  - MLP with GELU activation

This profiler categorizes operations relevant for ASIC acceleration.
Key insight: Position bucketization is a hidden bottleneck (~48ms)!
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import time
from collections import defaultdict
import json

print("=" * 80)
print("SmolVLM2-256M Vision Encoder Detailed Profiling")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Instruct"
NUM_SAMPLES = 10

print(f"\n🎯 Model: {MODEL_PATH}")
print(f"📊 Profiling samples: {NUM_SAMPLES}")

# ============================================================
# Load Model & Processor
# ============================================================
print(f"\n📦 Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2",
).to("cuda")

model.eval()

print(f"✅ Model loaded on: {next(model.parameters()).device}")

# ============================================================
# Explore Model Architecture
# ============================================================
print("\n🏗️  Model Architecture:")
for name, module in model.named_modules():
    if len(name.split('.')) <= 2:  # Only show top-level modules
        print(f"  • {name}: {module.__class__.__name__}")

# ============================================================
# Setup Profiling Hooks
# ============================================================
module_times = defaultdict(list)
module_forward_times = {}

def create_forward_hook(module_name):
    """Create a forward hook that measures execution time"""
    def hook(module, input, output):
        if module_name in module_forward_times:
            # Record end time
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = (end_time - module_forward_times[module_name]) * 1000  # Convert to ms
            module_times[module_name].append(elapsed)
            del module_forward_times[module_name]
    return hook

def create_pre_forward_hook(module_name):
    """Create a pre-forward hook that records start time"""
    def hook(module, input):
        torch.cuda.synchronize()
        module_forward_times[module_name] = time.time()
    return hook

# Register hooks for key modules
hooks = []
for name, module in model.named_modules():
    # Track specific vision and LLM components
    if any(keyword in name for keyword in [
        'vision_model',
        'patch_embedding',
        'embeddings',
        'encoder.layers',
        'text_model',
        'lm_head'
    ]):
        # Only track leaf modules and important containers
        if len(list(module.children())) == 0 or any(x in name for x in ['vision_model', 'language_model']):
            hooks.append(module.register_forward_pre_hook(create_pre_forward_hook(name)))
            hooks.append(module.register_forward_hook(create_forward_hook(name)))

print(f"\n✅ Registered hooks on {len(hooks)//2} modules")

# ============================================================
# Load Test Data
# ============================================================
print("\n📊 Loading test dataset...")
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
split_ds = ds["validation"].train_test_split(test_size=0.1, seed=42)
test_ds = split_ds["test"]

print(f"✅ Dataset loaded")

# ============================================================
# Run Profiling
# ============================================================
print(f"\n🔍 Profiling {NUM_SAMPLES} forward passes...")

for i in range(NUM_SAMPLES):
    example = test_ds[i]
    image = example["image"]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    question = example["question"]

    # Prepare input
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

    # Run forward pass (single token generation for profiling)
    with torch.no_grad():
        # _ = model(**inputs)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,  # Generate 50 tokens
            do_sample=False,    # Greedy decoding for consistency
        )

    if (i + 1) % 5 == 0:
        print(f"  Completed {i + 1}/{NUM_SAMPLES} samples")

# Remove hooks
for hook in hooks:
    hook.remove()

print("✅ Profiling complete")

# ============================================================
# Analyze Results
# ============================================================
print("\n" + "=" * 80)
print("PROFILING RESULTS")
print("=" * 80)

# Calculate statistics
results = {}
for module_name, times in module_times.items():
    if times:  # Only include modules that were actually called
        results[module_name] = {
            'mean_ms': sum(times) / len(times),
            'total_ms': sum(times),
            'count': len(times),
            'min_ms': min(times),
            'max_ms': max(times),
        }

# Sort by total time
sorted_results = sorted(results.items(), key=lambda x: x[1]['total_ms'], reverse=True)

# ============================================================
# Print Vision Encoder Breakdown
# ============================================================
print("\n📊 VISION ENCODER COMPONENTS:")
print("-" * 80)
print(f"{'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
print("-" * 80)

vision_total = 0
for name, stats in sorted_results:
    if 'vision' in name.lower() or 'patch' in name.lower():
        print(f"{name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f} {stats['count']:>6}")
        vision_total += stats['total_ms']

print("-" * 80)
print(f"{'VISION TOTAL':<50} {'':<12} {vision_total:>10.2f}ms")

# ============================================================
# Print ASIC-Relevant Operations
# ============================================================
print("\n📊 ASIC-RELEVANT OPERATIONS:")
print("-" * 80)
print(f"{'Operation Type':<30} {'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12}")
print("-" * 80)

asic_ops = {
    'Conv2d (Patch Embedding)': [],
    'Position Embedding + Bucketization': [],
    'LayerNorm': [],
    'GELU Activation': [],
}

# Categorize ASIC-relevant operations based on SmolVLM architecture
for name, stats in sorted_results:
    name_lower = name.lower()

    # Patch embedding: Conv2d(3, 768, kernel=16, stride=16)
    # Match the actual Conv2d layer, not the parent embeddings container
    if 'patch_embedding' in name_lower and name_lower.endswith('patch_embedding'):
        asic_ops['Conv2d (Patch Embedding)'].append((name, stats))

    # Position embedding + bucketization:
    # The position_embedding module is just the lookup (1.36ms)
    # But we want the TOTAL time which includes bucketization (~48ms)
    # So we capture the parent 'embeddings' module and subtract patch_embedding time
    elif name_lower.endswith('embeddings') and 'vision' in name_lower and 'patch' not in name_lower:
        # This captures the full embeddings.forward() including bucketization
        asic_ops['Position Embedding + Bucketization'].append((name, stats))

    # LayerNorm: Both pre/post layernorm in vision encoder
    elif 'layernorm' in name_lower or 'layer_norm' in name_lower or 'post_layernorm' in name_lower:
        asic_ops['LayerNorm'].append((name, stats))

    # GELU activation: Used in MLP blocks
    elif 'gelu' in name_lower or 'activation' in name_lower:
        asic_ops['GELU Activation'].append((name, stats))

# Print categorized results
asic_total = 0
for op_type, modules in asic_ops.items():
    if modules:
        for name, stats in sorted(modules, key=lambda x: x[1]['total_ms'], reverse=True):
            display_name = name if len(name) <= 50 else '...' + name[-47:]
            print(f"{op_type:<30} {display_name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f}")
            asic_total += stats['total_ms']
            op_type = ''  # Only show operation type once per category

print("-" * 80)
print(f"{'ASIC OPS TOTAL':<30} {'':<50} {'':<12} {asic_total:>10.2f}ms")

# Print summary by operation type
print("\n📊 ASIC OPERATIONS BREAKDOWN:")
print("-" * 80)
print(f"{'Operation Type':<30} {'Count':<10} {'Total Time (ms)':<20} {'% of ASIC Ops':<15}")
print("-" * 80)
for op_type, modules in asic_ops.items():
    if modules:
        op_total = sum(stats['total_ms'] for _, stats in modules)
        count = len(modules)
        pct = (op_total / asic_total * 100) if asic_total > 0 else 0
        print(f"{op_type:<30} {count:<10} {op_total:<20.2f} {pct:>14.1f}%")

# ============================================================
# Print Other Vision Operations (Non-ASIC specific)
# ============================================================
print("\n📊 OTHER VISION OPERATIONS (likely already have hardware support):")
print("-" * 80)
print(f"{'Operation Type':<30} {'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12}")
print("-" * 80)

other_ops = {
    'Self-Attention (Q/K/V)': [],
    'MLP Linear (fc1/fc2)': [],
    'Connector/Projection': [],
    'Other': [],
}

# Track what we've already categorized
already_categorized = set()
for modules in asic_ops.values():
    for name, _ in modules:
        already_categorized.add(name)

# Categorize remaining vision operations
for name, stats in sorted_results:
    if name in already_categorized:
        continue
    if 'vision' not in name.lower():
        continue

    name_lower = name.lower()

    # Attention projections (Q, K, V, output)
    if 'self_attn' in name_lower and any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
        other_ops['Self-Attention (Q/K/V)'].append((name, stats))

    # MLP linear layers
    elif 'mlp' in name_lower and ('fc1' in name_lower or 'fc2' in name_lower):
        other_ops['MLP Linear (fc1/fc2)'].append((name, stats))

    # Connector/modality projection
    elif 'connector' in name_lower or 'projection' in name_lower:
        other_ops['Connector/Projection'].append((name, stats))

    # Everything else
    else:
        other_ops['Other'].append((name, stats))

# Print categorized results
other_total = 0
for op_type, modules in other_ops.items():
    if modules:
        # Sort by total time
        for name, stats in sorted(modules, key=lambda x: x[1]['total_ms'], reverse=True)[:10]:  # Top 10 per category
            display_name = name if len(name) <= 50 else '...' + name[-47:]
            print(f"{op_type:<30} {display_name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f}")
            other_total += stats['total_ms']
            op_type = ''  # Only show operation type once per category

print("-" * 80)
print(f"{'OTHER OPS TOTAL':<30} {'':<50} {'':<12} {other_total:>10.2f}ms")

# Print summary by operation type
print("\n📊 OTHER OPERATIONS BREAKDOWN:")
print("-" * 80)
print(f"{'Operation Type':<30} {'Count':<10} {'Total Time (ms)':<20} {'% of Other Ops':<15}")
print("-" * 80)
for op_type, modules in other_ops.items():
    if modules:
        op_total = sum(stats['total_ms'] for _, stats in modules)
        count = len(modules)
        pct = (op_total / other_total * 100) if other_total > 0 else 0
        print(f"{op_type:<30} {count:<10} {op_total:<20.2f} {pct:>14.1f}%")

# ============================================================
# Print Language Model ASIC-Relevant Operations
# ============================================================
print("\n📊 LANGUAGE MODEL ASIC-RELEVANT OPERATIONS:")
print("-" * 80)
print(f"{'Operation Type':<30} {'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12}")
print("-" * 80)

llm_asic_ops = {
    '1D RoPE (Rotary Position Embedding)': [],
    'RMSNorm': [],
    'SiLU/SwiGLU Activation': [],
}

# Categorize LM ASIC-relevant operations
for name, stats in sorted_results:
    # Skip vision operations
    if 'vision' in name.lower():
        continue

    name_lower = name.lower()

    # 1D RoPE: Rotary position embeddings used in Llama
    if 'rotary' in name_lower or 'rope' in name_lower:
        llm_asic_ops['1D RoPE (Rotary Position Embedding)'].append((name, stats))

    # RMSNorm: Llama uses RMSNorm instead of LayerNorm
    elif 'rmsnorm' in name_lower or 'rms_norm' in name_lower:
        llm_asic_ops['RMSNorm'].append((name, stats))

    # SiLU/SwiGLU: Llama uses SiLU activation (x * sigmoid(x))
    elif 'silu' in name_lower or 'swiglu' in name_lower or ('act_fn' in name_lower and 'mlp' in name_lower):
        llm_asic_ops['SiLU/SwiGLU Activation'].append((name, stats))

# Print categorized results
llm_asic_total = 0
for op_type, modules in llm_asic_ops.items():
    if modules:
        for name, stats in sorted(modules, key=lambda x: x[1]['total_ms'], reverse=True):
            display_name = name if len(name) <= 50 else '...' + name[-47:]
            print(f"{op_type:<30} {display_name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f}")
            llm_asic_total += stats['total_ms']
            op_type = ''  # Only show operation type once per category

print("-" * 80)
print(f"{'LM ASIC OPS TOTAL':<30} {'':<50} {'':<12} {llm_asic_total:>10.2f}ms")

# Print summary by operation type
if llm_asic_total > 0:
    print("\n📊 LM ASIC OPERATIONS BREAKDOWN:")
    print("-" * 80)
    print(f"{'Operation Type':<30} {'Count':<10} {'Total Time (ms)':<20} {'% of LM ASIC Ops':<15}")
    print("-" * 80)
    for op_type, modules in llm_asic_ops.items():
        if modules:
            op_total = sum(stats['total_ms'] for _, stats in modules)
            count = len(modules)
            pct = (op_total / llm_asic_total * 100) if llm_asic_total > 0 else 0
            print(f"{op_type:<30} {count:<10} {op_total:<20.2f} {pct:>14.1f}%")

# ============================================================
# Print Other Language Model Operations
# ============================================================
print("\n📊 OTHER LANGUAGE MODEL OPERATIONS:")
print("-" * 80)
print(f"{'Operation Type':<30} {'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12}")
print("-" * 80)

llm_other_ops = {
    'Self-Attention (Q/K/V)': [],
    'MLP Linear (gate/up/down)': [],
    'Embedding/LM Head': [],
    'Other': [],
}

# Track what we've already categorized for LM
llm_already_categorized = set()
for modules in llm_asic_ops.values():
    for name, _ in modules:
        llm_already_categorized.add(name)

# Categorize remaining LM operations
for name, stats in sorted_results:
    if name in llm_already_categorized:
        continue
    if 'vision' in name.lower() or 'connector' in name.lower():
        continue

    # Only process text_model or lm_head operations
    if not ('text_model' in name.lower() or 'lm_head' in name.lower() or 'language' in name.lower()):
        continue

    name_lower = name.lower()

    # Attention projections (Q, K, V, output)
    if 'self_attn' in name_lower and any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        llm_other_ops['Self-Attention (Q/K/V)'].append((name, stats))

    # MLP linear layers (Llama uses gate_proj, up_proj, down_proj)
    elif 'mlp' in name_lower and any(x in name_lower for x in ['gate_proj', 'up_proj', 'down_proj']):
        llm_other_ops['MLP Linear (gate/up/down)'].append((name, stats))

    # Embedding and LM head
    elif 'embed' in name_lower or 'lm_head' in name_lower:
        llm_other_ops['Embedding/LM Head'].append((name, stats))

    # Everything else
    else:
        llm_other_ops['Other'].append((name, stats))

# Print categorized results
llm_other_total = 0
for op_type, modules in llm_other_ops.items():
    if modules:
        # Sort by total time, show top 10 per category
        for name, stats in sorted(modules, key=lambda x: x[1]['total_ms'], reverse=True)[:10]:
            display_name = name if len(name) <= 50 else '...' + name[-47:]
            print(f"{op_type:<30} {display_name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f}")
            llm_other_total += stats['total_ms']
            op_type = ''  # Only show operation type once per category

print("-" * 80)
print(f"{'LM OTHER OPS TOTAL':<30} {'':<50} {'':<12} {llm_other_total:>10.2f}ms")

# Calculate total LM time
llm_total = llm_asic_total + llm_other_total

# Print summary by operation type
if llm_other_total > 0:
    print("\n📊 LM OTHER OPERATIONS BREAKDOWN:")
    print("-" * 80)
    print(f"{'Operation Type':<30} {'Count':<10} {'Total Time (ms)':<20} {'% of LM Other Ops':<15}")
    print("-" * 80)
    for op_type, modules in llm_other_ops.items():
        if modules:
            op_total = sum(stats['total_ms'] for _, stats in modules)
            count = len(modules)
            pct = (op_total / llm_other_total * 100) if llm_other_total > 0 else 0
            print(f"{op_type:<30} {count:<10} {op_total:<20.2f} {pct:>14.1f}%")

# ============================================================
# Print Top 20 Components Overall
# ============================================================
print("\n📊 TOP 20 SLOWEST COMPONENTS:")
print("-" * 80)
print(f"{'Module':<50} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
print("-" * 80)

for name, stats in sorted_results[:20]:
    # Shorten name if too long
    display_name = name if len(name) <= 50 else '...' + name[-47:]
    print(f"{display_name:<50} {stats['mean_ms']:>10.2f} {stats['total_ms']:>10.2f} {stats['count']:>6}")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_time = sum(stats['total_ms'] for stats in results.values())
print(f"\n{'Metric':<30} {'Value':<20}")
print("-" * 55)
print(f"{'Total Profiled Time':<30} {total_time:>17.2f}ms")
print(f"{'Vision Encoder Time':<30} {vision_total:>17.2f}ms ({vision_total/total_time*100:>5.1f}%)")
print(f"{'ASIC Operations Time':<30} {asic_total:>17.2f}ms ({asic_total/total_time*100:>5.1f}%)")
print(f"{'Language Model Time':<30} {llm_total:>17.2f}ms ({llm_total/total_time*100:>5.1f}%)")
print(f"{'Avg Time per Sample':<30} {total_time/NUM_SAMPLES:>17.2f}ms")

# ============================================================
# Save Results
# ============================================================
output_file = "vision_profiling_detailed.json"

# Prepare ASIC ops breakdown for JSON
asic_ops_summary = {}
for op_type, modules in asic_ops.items():
    if modules:
        asic_ops_summary[op_type] = {
            'count': len(modules),
            'total_ms': sum(stats['total_ms'] for _, stats in modules),
            'modules': {name: stats for name, stats in modules}
        }

# Prepare other ops breakdown for JSON
other_ops_summary = {}
for op_type, modules in other_ops.items():
    if modules:
        other_ops_summary[op_type] = {
            'count': len(modules),
            'total_ms': sum(stats['total_ms'] for _, stats in modules),
            'modules': {name: stats for name, stats in modules}
        }

# Prepare LM ASIC ops breakdown for JSON
llm_asic_ops_summary = {}
for op_type, modules in llm_asic_ops.items():
    if modules:
        llm_asic_ops_summary[op_type] = {
            'count': len(modules),
            'total_ms': sum(stats['total_ms'] for _, stats in modules),
            'modules': {name: stats for name, stats in modules}
        }

# Prepare LM other ops breakdown for JSON
llm_other_ops_summary = {}
for op_type, modules in llm_other_ops.items():
    if modules:
        llm_other_ops_summary[op_type] = {
            'count': len(modules),
            'total_ms': sum(stats['total_ms'] for _, stats in modules),
            'modules': {name: stats for name, stats in modules}
        }

output_data = {
    "model": MODEL_PATH,
    "num_samples": NUM_SAMPLES,
    "total_time_ms": total_time,
    "vision_time_ms": vision_total,
    "asic_ops_time_ms": asic_total,
    "other_ops_time_ms": other_total,
    "llm_time_ms": llm_total,
    "llm_asic_ops_time_ms": llm_asic_total,
    "llm_other_ops_time_ms": llm_other_total,
    "asic_ops_breakdown": asic_ops_summary,
    "other_ops_breakdown": other_ops_summary,
    "llm_asic_ops_breakdown": llm_asic_ops_summary,
    "llm_other_ops_breakdown": llm_other_ops_summary,
    "module_times": {name: stats for name, stats in results.items()}
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n💾 Detailed results saved to: {output_file}")

# ============================================================
# Hardware Acceleration Recommendations
# ============================================================
print("\n" + "=" * 80)
print("HARDWARE ACCELERATION RECOMMENDATIONS")
print("=" * 80)

print("\n🔧 PRIORITY 1: Vision Encoder Custom Hardware")
print("-" * 80)

# Calculate vision times
conv2d_time = sum(stats['total_ms'] for _, stats in asic_ops.get('Conv2d (Patch Embedding)', []))
pos_emb_time = sum(stats['total_ms'] for _, stats in asic_ops.get('Position Embedding + Bucketization', []))
layernorm_time = sum(stats['total_ms'] for _, stats in asic_ops.get('LayerNorm', []))

# Position embedding time includes Conv2d, so subtract it to get bucketization overhead
bucketization_time = pos_emb_time - conv2d_time if pos_emb_time > conv2d_time else 0

print(f"1. Conv2d (Patch Embedding): {conv2d_time:.2f}ms")
print(f"   - Spec: Conv2d(in=3, out=768, kernel=16x16, stride=16)")
print(f"   - Input: [B, 3, 512, 512] → Output: [B, 768, 32, 32]")
print(f"   - Why: High compute density, predictable access patterns")
print(f"   - Expected speedup: 3-5x with dedicated hardware")
print()
print(f"2. Position Bucketization (2D→1D): {bucketization_time:.2f}ms")
print(f"   - Python for-loop computing 2D→1D position IDs")
print(f"   - Why: CPU-bound, can be pre-computed for fixed resolution")
print(f"   - Expected speedup: 10-100x (eliminate with lookup table)")
print()
print(f"3. LayerNorm (Vision): {layernorm_time:.2f}ms ({len(asic_ops.get('LayerNorm', []))} instances)")
print(f"   - Fused mean/variance/scale operation")
print(f"   - Why: Memory-bound on GPU, benefits from custom datapath")
print(f"   - Expected speedup: 2-3x with fused implementation")

print("\n🔧 PRIORITY 2: Language Model Custom Hardware")
print("-" * 80)

# Calculate LM times
rope_time = sum(stats['total_ms'] for _, stats in llm_asic_ops.get('1D RoPE (Rotary Position Embedding)', []))
rmsnorm_time = sum(stats['total_ms'] for _, stats in llm_asic_ops.get('RMSNorm', []))
silu_time = sum(stats['total_ms'] for _, stats in llm_asic_ops.get('SiLU/SwiGLU Activation', []))

if rope_time > 0:
    print(f"1. 1D RoPE (Rotary Position Embedding): {rope_time:.2f}ms")
    print(f"   - Applies rotation matrix for position encoding")
    print(f"   - Why: Complex operation (sin/cos + rotation), used in every layer")
    print(f"   - Expected speedup: 2-4x with fused sin/cos lookup + multiply")
    print()
else:
    print("1. 1D RoPE: Not observed (likely cached or model not generating)")
    print()

if rmsnorm_time > 0:
    print(f"2. RMSNorm: {rmsnorm_time:.2f}ms ({len(llm_asic_ops.get('RMSNorm', []))} instances)")
    print(f"   - Root mean square normalization (simpler than LayerNorm)")
    print(f"   - Why: Used 2x per layer, memory-bound operation")
    print(f"   - Expected speedup: 2-3x with fused implementation")
    print()
else:
    print("2. RMSNorm: Not observed (likely cached or model not generating)")
    print()

if silu_time > 0:
    print(f"3. SiLU/SwiGLU Activation: {silu_time:.2f}ms")
    print(f"   - SiLU: x * sigmoid(x), used in SwiGLU MLP")
    print(f"   - Why: More complex than GELU, used frequently")
    print(f"   - Expected speedup: 2-3x with custom implementation")
else:
    print("3. SiLU/SwiGLU: Not observed (likely cached or model not generating)")

print("\n🔧 PRIORITY 3: Software/Assembly Implementation")
print("-" * 80)

gelu_time = sum(stats['total_ms'] for _, stats in asic_ops.get('GELU Activation', []))

print(f"1. GELU Activation (Vision): {gelu_time:.2f}ms")
print(f"   - Simple element-wise operation")
print(f"   - Implement with FMA instructions or polynomial approximation")
print(f"   - Not worth dedicated hardware")
print()
print(f"2. Position Embedding Lookup: ~1.4ms (included in bucketization above)")
print(f"   - Just a memory lookup (embedding table)")
print(f"   - Use standard SRAM/cache (1.5MB table: 1024×768×2 bytes)")
print(f"   - Negligible time, already counted in bucketization")

print("\n🔧 EXISTING HARDWARE (assume you already have these)")
print("-" * 80)

attn_time = sum(stats['total_ms'] for _, stats in other_ops.get('Self-Attention (Q/K/V)', []))
mlp_time = sum(stats['total_ms'] for _, stats in other_ops.get('MLP Linear (fc1/fc2)', []))

print(f"1. Matrix Multiply (GEMM): {attn_time + mlp_time:.2f}ms")
print(f"   - Self-attention Q/K/V/O projections: {attn_time:.2f}ms")
print(f"   - MLP fc1/fc2 layers: {mlp_time:.2f}ms")
print(f"   - Assume you already have efficient GEMM units")
print()
print(f"2. Attention Computation: (included in self-attention timing)")
print(f"   - Likely using Flash Attention 2 or similar")
print(f"   - Already optimized, no custom hardware needed")

print("\n📊 EXPECTED TOTAL SPEEDUP")
print("-" * 80)
potential_savings = conv2d_time + bucketization_time + layernorm_time
current_vision_time = vision_total
print(f"Current vision encoder time: {current_vision_time:.2f}ms")
print(f"Acceleratable operations: {potential_savings:.2f}ms ({potential_savings/current_vision_time*100:.1f}%)")
print(f"  - Conv2d: {conv2d_time:.2f}ms → {conv2d_time/3:.2f}ms (3x speedup)")
print(f"  - Bucketization: {bucketization_time:.2f}ms → 0.5ms (eliminate via LUT)")
print(f"  - LayerNorm: {layernorm_time:.2f}ms → {layernorm_time/2.5:.2f}ms (2.5x speedup)")
savings = (conv2d_time - conv2d_time/3) + (bucketization_time - 0.5) + (layernorm_time - layernorm_time/2.5)
new_vision_time = current_vision_time - savings
speedup = current_vision_time / new_vision_time if new_vision_time > 0 else 1.0
print(f"\nEstimated new vision time: {new_vision_time:.2f}ms")
print(f"Estimated speedup: {speedup:.2f}x on vision encoder")

print("\n" + "=" * 80)
print("✅ PROFILING COMPLETE!")
print("=" * 80)
