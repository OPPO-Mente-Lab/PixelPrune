# PixelPrune Training

PixelPrune visual token pruning training for Qwen VLMs (Qwen3-VL / Qwen3.5).

## Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# PixelPrune environment variables
export PIXELPRUNE_ENABLED=true          # Enable visual token pruning
export PIXELPRUNE_THRESHOLD=0.0

# Launch training
deepspeed --num_gpus 8 train.py \
    --deepspeed_config_path    configs/deepspeed_config.json \
    --training_config_path     configs/training_config.json \
    --task_info_config_path    configs/task_info.json \
    --task_weight_config_path  configs/task_weight.json
```

Set `PIXELPRUNE_ENABLED=false` to disable pruning (standard training).

## Data Format

JSONL files in ShareGPT format. Two image reference styles are supported:

```jsonl
{"conversations": [{"from": "human", "value": "<image>Describe this image."}, {"from": "gpt", "value": "A photo of..."}], "images": ["/path/to/image.jpg"]}
```

Or with inline absolute image paths:

```jsonl
{"conversations": [{"from": "human", "value": "<img>/path/to/image.jpg</img>\nDescribe this image."}, {"from": "gpt", "value": "A photo of..."}]}
```

Configure data paths in `task_info.json` and sampling weights in `task_weight.json`.

## Key Config Options (`training_config.json`)

| Option | Description |
|--------|-------------|
| `model_path` | Path to Qwen3-VL or Qwen3.5 model |
| `max_length` | Max packed sequence length |
| `max_size` | Max image size (max_pixels capped at max_size²) |
| `kd_config.alpha` | KD weight (>0 enables KD; self-distillation if `ref_model_path` unset) |

## Training Optimizations

- **PixelPrune**: Visual tokens pruned at configurable ViT layer, reducing ViT and LLM input length
- **Data packing** with varlen flash attention (`cu_seqlens`) — no padding waste
- **Flash Attention** (`flash_attn_varlen_func`) for Qwen3-VL and Qwen3.5 full_attention layers
- **Flash Linear Attention** (fla `causal_conv1d` + `chunk_gated_delta_rule`) for Qwen3.5 GatedDeltaNet layers
- **Fused kernels** (Liger): RMSNorm, SwiGLU MLP, Linear CrossEntropy (logit-free), Linear JSD for KD
- **Selective loss**: Only compute `lm_head` on valid label positions, skip all -100 tokens
- **Global loss correction**: All-reduce valid token counts across ranks and gradient accumulation steps for correct loss scaling
- **Pre-computation on CPU**: 3D RoPE position IDs and prune indices computed during data loading
- **Memory**: BFloat16, DeepSpeed ZeRO, gradient checkpointing for LLM + ViT

## File Structure

```
training/
├── train.py              # Unified training (CE + KD)
├── utils.py              # Model loading, monkey-patches, fused kernels, checkpoint saving
├── data/
│   ├── dataset.py        # MultiTurn / BufferMultiTurn dataset with packing
│   └── data_generator.py # DataLoader with distributed sampling
└── configs/              # Default configs
```
