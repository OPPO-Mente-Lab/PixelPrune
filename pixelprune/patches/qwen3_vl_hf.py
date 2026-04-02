"""
HuggingFace Qwen3-VL monkey-patch：在不动 modeling 源码的前提下注入 PatchSelector 选择器与 dedup 逻辑。

环境变量：PIXELPRUNE_ENABLED, PIXELPRUNE_METHOD, PIXELPRUNE_METRIC, PIXELPRUNE_THRESHOLD, PIXELPRUNE_VIT_LAYER
在加载模型前调用 apply_patches()，然后照常使用 AutoModelForImageTextToText.from_pretrained(...)。
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pixelprune.core import (
    compute_merged_keep_indices,
    merged_indices_to_patch_indices,
)


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------

def normalize_pixel_values_for_selector(pixel_values: torch.Tensor) -> torch.Tensor:
    """将 PixelValues 归一化到 [0,1] 以供 PatchSelector 使用。

    Qwen3-VL 的 pixel_values shape 为 (total_patches, C*T*H*W)，
    每个 patch 包含 T=2 帧，两帧内容相同（静态图像）。
    取第 0 帧并做逆标准化 x*0.5+0.5 还原到 [0,1]。
    """
    C, T, H, W = 3, 2, 16, 16
    pv_reshaped = pixel_values.view(-1, C, T, H, W)
    frame_0 = pv_reshaped[:, :, 0, :, :]
    normalized = (frame_0 * 0.5) + 0.5
    return normalized.reshape(pixel_values.shape[0], -1)


def _get_log_paths() -> Tuple[str, str, str]:
    """从 PIXELPRUNE_LOG_FILE 派生 (dedup_log, vit_log, e2e_log) 三条路径。

    若未设置，三者均返回空字符串。
    示例：PIXELPRUNE_LOG_FILE=run.jsonl → run.jsonl, run.vit.jsonl, run.e2e.jsonl
    """
    base_path = os.environ.get("PIXELPRUNE_LOG_FILE", "")
    if not base_path:
        return "", "", ""
    base, ext = os.path.splitext(base_path)
    return base_path, f"{base}.vit{ext}", f"{base}.e2e{ext}"


def _get_rank() -> int:
    """统一获取当前进程 rank，优先使用 dist，fallback 到 cuda device。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def _write_jsonl(record: Dict, filepath: str) -> None:
    """向 filepath 追加一条 JSON Lines 记录。

    使用 per-rank 独立文件（filepath 已由调用方含 rank 后缀），
    单进程单文件写入，无需跨进程锁，对 NFS 也安全。
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(filepath, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _rank_path(filepath: str, rank: int) -> str:
    """将 filepath 插入 rank 后缀，使每个进程写独立文件。

    例：run.vit.jsonl → run.vit.rank0.jsonl
    这样多卡场景下各 rank 写自己的文件，完全避免并发冲突。
    """
    base, ext = os.path.splitext(filepath)
    return f"{base}.rank{rank}{ext}"


# rank → 尚未 flush 的 VIT profile 记录（在 _model_get_image_features 中补充 timing 后 flush）
_pending_vit_record: Dict[int, Dict] = {}

# rank → 最近一次 ViT forward 的 FLOPs 缓存（供 model.py 写 e2e jsonl 时读取）
_last_vit_flops: Dict[int, Dict] = {}  # {"theory": float, "theory_full": float}


def _store_dedup_stats(
    original_merged_lengths: List[int],
    new_merged_lengths: List[int],
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    stats_only: bool = False,
    selector_latency_ms: Optional[float] = None,
) -> None:
    """构造 VIT profile 的基础字段并存入 _pending_vit_record[rank]。

    VERBOSE=true 时同时向 stdout 打印可读摘要。
    当 stats_only=True（不运行模型）时，调用方负责调用 flush_pending_vit_record()。
    当 stats_only=False 时，_model_get_image_features 会在 ViT 运行结束后补充
    timing/VRAM 字段并自动 flush。

    Args:
        selector_latency_ms: compute_merged_keep_indices 的耗时（ms），
            由 _cg_forward 计时后传入，写入 vit profile 的 selector_latency_ms 字段。
    """
    VERBOSE = os.environ.get("PIXELPRUNE_VERBOSE", "false").lower() in ("true", "1", "yes")
    _, vit_log_file, _ = _get_log_paths()
    if not VERBOSE and not vit_log_file:
        return

    assert input_ids.shape[0] == 1, "请确保 batch_size == 1"
    rank = _get_rank()

    merge_size_sq = 4  # spatial_merge_size=2 → 2²=4（Qwen3-VL 固定）
    org_input_len = input_ids.shape[1]
    org_vit_lens = [l * merge_size_sq for l in original_merged_lengths]
    new_vit_lens = [l * merge_size_sq for l in new_merged_lengths]
    retain_ratios = [
        round(nv / ov, 6) if ov > 0 else 1.0
        for nv, ov in zip(new_merged_lengths, original_merged_lengths)
    ]
    llm_prune_len = sum(original_merged_lengths) - sum(new_merged_lengths)
    new_input_len = org_input_len - llm_prune_len

    grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else []
    patch_size = 16
    img_h_pixels = [int(g[1]) * patch_size for g in grid_thw_list]
    img_w_pixels = [int(g[2]) * patch_size for g in grid_thw_list]

    record: Dict = {
        "type": "vit",
        "rank": rank,
        "sample_idx": int(os.environ.get("CURRENT_SAMPLE_IDX", "-1")),
        "num_images": len(original_merged_lengths),
        "image_grid_thw": grid_thw_list,
        "image_h_pixels": img_h_pixels,
        "image_w_pixels": img_w_pixels,
        "org_vit_lens": org_vit_lens,
        "new_vit_lens": new_vit_lens,
        "org_merged_lens": original_merged_lengths,
        "new_merged_lens": new_merged_lengths,
        "retain_ratios": retain_ratios,
        "org_input_len": org_input_len,
        "new_input_len": new_input_len,
        "llm_prune_len": llm_prune_len,
        "method": os.environ.get("PIXELPRUNE_METHOD", "pred_2d"),
        "metric": os.environ.get("PIXELPRUNE_METRIC", "max"),
        "threshold": float(os.environ.get("PIXELPRUNE_THRESHOLD", "0.0") or "0.0"),
        "vit_prune_layer": int(os.environ.get("PIXELPRUNE_VIT_LAYER", "0") or "0"),
        "stats_only": stats_only,
        # selector 计时：在 _cg_forward 中测量 compute_merged_keep_indices 耗时
        "selector_latency_ms": round(selector_latency_ms, 3) if selector_latency_ms is not None else None,
        # timing/VRAM 字段在 _model_get_image_features 中补充（stats_only=False 时）
        "vit_latency_ms": None,
        "vram_vit_before_gb": None,
        "vram_vit_after_gb": None,
        "vram_vit_peak_gb": None,
    }
    _pending_vit_record[rank] = record

    if VERBOSE:
        sys.stdout.write(
            f"[DEDUP STATS][rank={rank}] "
            f"Org Input Len: {org_input_len}, "
            f"Org Vit Lens: {org_vit_lens}, "
            f"New Vit Lens: {new_vit_lens}, "
            f"Retain Ratios: {[f'{r:.3f}' for r in retain_ratios]}, "
            f"LLM Prune Len: {llm_prune_len}, "
            f"New Input Len: {new_input_len}\n"
        )
        sys.stdout.flush()


def flush_pending_vit_record(rank: int) -> None:
    """将 _pending_vit_record[rank] 写入 per-rank vit log 文件（如果已配置）。

    写入 {base}.vit.rank{rank}{ext}，每个进程写自己的文件，无跨进程竞争。
    """
    _, vit_log_file, _ = _get_log_paths()
    if vit_log_file and rank in _pending_vit_record:
        _write_jsonl(_pending_vit_record.pop(rank), _rank_path(vit_log_file, rank))

# -----------------------------------------------------------------------------
# 辅助：ViT 内按 keep_indices 做 token 选择
# -----------------------------------------------------------------------------


def _select_packed_by_indices(
    tensor: torch.Tensor,
    grid_thw: torch.Tensor,
    keep_indices: List[torch.Tensor],
) -> torch.Tensor:
    """从 packed tensor 中按每张图的 keep_indices 选择 token。"""
    selected = []
    offset = 0
    grid_list = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    for seq_idx, (t, h, w) in enumerate(grid_list):
        t, h, w = int(t), int(h), int(w)
        seq_length = t * h * w
        indices = keep_indices[seq_idx].to(tensor.device)
        selected.append(tensor[offset : offset + seq_length][indices])
        offset += seq_length
    return torch.cat(selected)


def _adjust_inputs_for_dedup(
    self: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    image_embeds: List[torch.Tensor],
    image_grid_thw: torch.Tensor,
    padding_side: str = "left",
    merged_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    """根据 dedup 后的 image_embeds 长度，裁剪 inputs_embeds / input_ids 中多余的 image token 位置。

    当提供 merged_indices 时，按实际保留索引的空间位置标记 keep_mask，
    确保后续 position_ids 保留正确的空间编码。
    """
    batch_size, seq_len = input_ids.shape
    image_token_id = self.config.image_token_id
    vision_start_token_id = self.config.vision_start_token_id
    spatial_merge_size = getattr(
        self.config.vision_config, "spatial_merge_size", 2
    )
    keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cpu")
    new_inputs_embeds_list = []
    new_input_ids_list = []

    for batch_idx in range(batch_size):
        sample_input_ids = input_ids[batch_idx]
        sample_inputs_embeds = inputs_embeds[batch_idx]
        vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
        if len(vision_start_indices) == 0 or len(
            image_positions := vision_start_indices[
                (vision_tokens := sample_input_ids[vision_start_indices + 1]) == image_token_id
            ]
        ) == 0:
            new_inputs_embeds_list.append(sample_inputs_embeds)
            new_input_ids_list.append(sample_input_ids)
            continue
        sample_keep_mask = torch.ones(len(sample_input_ids), dtype=torch.bool, device="cpu")
        for img_idx, img_pos in enumerate(image_positions):
            t, h, w = (int(x) for x in image_grid_thw[img_idx])
            llm_grid_h, llm_grid_w = h // spatial_merge_size, w // spatial_merge_size
            num_original_llm_tokens = t * llm_grid_h * llm_grid_w
            num_new_tokens = image_embeds[img_idx].shape[0]
            visual_start = (img_pos + 1).item()
            sample_keep_mask[visual_start : visual_start + num_original_llm_tokens] = False
            if merged_indices is not None:
                sample_keep_mask[visual_start + merged_indices[img_idx].cpu().long()] = True
            else:
                sample_keep_mask[visual_start : visual_start + num_new_tokens] = True
        keep_mask[batch_idx] = sample_keep_mask
        new_inputs_embeds_list.append(sample_inputs_embeds[sample_keep_mask])
        new_input_ids_list.append(sample_input_ids[sample_keep_mask])

    max_len = max(ids.shape[0] for ids in new_input_ids_list)
    padded_inputs_embeds = []
    padded_input_ids = []
    pad_token_id = getattr(self.config, "pad_token_id", 0)
    for embeds, ids in zip(new_inputs_embeds_list, new_input_ids_list):
        if embeds.shape[0] < max_len:
            pad_len = max_len - embeds.shape[0]
            if padding_side == "left":
                embeds = F.pad(embeds, (0, 0, pad_len, 0), value=0)
                ids = F.pad(ids, (pad_len, 0), value=pad_token_id)
            else:
                embeds = F.pad(embeds, (0, 0, 0, pad_len), value=0)
                ids = F.pad(ids, (0, pad_len), value=pad_token_id)
        padded_inputs_embeds.append(embeds)
        padded_input_ids.append(ids)
    return (
        torch.stack(padded_inputs_embeds),
        torch.stack(padded_input_ids),
        keep_mask,
    )


# -----------------------------------------------------------------------------
# Patch 实现
# -----------------------------------------------------------------------------


def _vision_forward(
    self: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    keep_indices: Optional[List[torch.Tensor]] = None,
    **kwargs: Any,
) -> Any:
    """Patched Qwen3VLVisionModel.forward：支持可选的 keep_indices 和 VIT_PRUNE_LAYER 中间层裁剪。"""
    if keep_indices is None:
        return _orig_vision_forward(self, hidden_states, grid_thw=grid_thw, **kwargs)

    vit_prune_layer = int(os.environ.get("PIXELPRUNE_VIT_LAYER", "") or "0")
    num_blocks = len(self.blocks)
    assert 0 <= vit_prune_layer <= num_blocks, (
        f"PIXELPRUNE_VIT_LAYER={vit_prune_layer} out of range [0, {num_blocks}]"
    )

    ds_indexes = getattr(self, "deepstack_visual_indexes", [])
    if ds_indexes:
        min_ds, max_ds = min(ds_indexes), max(ds_indexes)
        assert vit_prune_layer <= min_ds or vit_prune_layer > max_ds, (
            f"PIXELPRUNE_VIT_LAYER={vit_prune_layer} must be <= {min_ds} "
            f"(before all deepstack) or > {max_ds} (after all deepstack), "
            f"deepstack_visual_indexes={ds_indexes}"
        )

    hidden_states_full = self.patch_embed(hidden_states)
    pos_embeds_full = self.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb_full = self.rot_pos_emb(grid_thw)
    rotary_pos_emb_full = rotary_pos_emb_full.reshape(rotary_pos_emb_full.shape[0], -1)

    if vit_prune_layer == 0:
        hidden_states = _select_packed_by_indices(hidden_states_full, grid_thw, keep_indices)
        pos_embeds = _select_packed_by_indices(pos_embeds_full, grid_thw, keep_indices)
        rotary_pos_emb = _select_packed_by_indices(rotary_pos_emb_full, grid_thw, keep_indices)
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seq_lengths = [len(idx) for idx in keep_indices]
        cu_seqlens = F.pad(
            torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0),
            (1, 0), value=0,
        )
    else:
        hidden_states = hidden_states_full + pos_embeds_full
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        emb_full = torch.cat((rotary_pos_emb_full, rotary_pos_emb_full), dim=-1)
        position_embeddings = (emb_full.cos(), emb_full.sin())

        cu_seqlens = F.pad(
            torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(0, dtype=torch.int32),
            (1, 0), value=0,
        ).to(hidden_states.device)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens.int(),
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[
                self.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

        if layer_num + 1 == vit_prune_layer:
            hidden_states = _select_packed_by_indices(hidden_states, grid_thw, keep_indices)
            cos_pruned, sin_pruned = (_select_packed_by_indices(pos, grid_thw, keep_indices) for pos in position_embeddings)
            position_embeddings = (cos_pruned, sin_pruned)

            seq_lengths = [len(idx) for idx in keep_indices]
            cu_seqlens = F.pad(
                torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0),
                (1, 0), value=0,
            )

    hidden_states = self.merger(hidden_states)

    if deepstack_feature_lists and ds_indexes and vit_prune_layer > max(ds_indexes):
        spatial_merge_size = getattr(self.config, "spatial_merge_size", 2)
        block_size = spatial_merge_size ** 2
        merged_indices = [idx[::block_size] // block_size for idx in keep_indices]
        split_sizes = (grid_thw.prod(-1) // block_size).tolist()
        for i, ds_feat in enumerate(deepstack_feature_lists):
            ds_per_img = torch.split(ds_feat, split_sizes)
            deepstack_feature_lists[i] = torch.cat([
                ds_per_img[j][merged_indices[j].to(ds_per_img[j].device)]
                for j in range(len(ds_per_img))
            ])

    return hidden_states, deepstack_feature_lists


def _model_get_image_features(
    self: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    keep_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[Any, Any]:
    """Patched get_image_features：将 keep_indices 传入 visual。"""
    pixel_values_typed = pixel_values.type(self.visual.dtype)

    VERBOSE = os.environ.get("PIXELPRUNE_VERBOSE", "false").lower() in ("true", "1", "yes")
    _, vit_log_file, _ = _get_log_paths()
    SHOULD_PROFILE = VERBOSE or bool(vit_log_file)

    if SHOULD_PROFILE:
        from torch.profiler import profile as _torch_profile, ProfilerActivity as _PA
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_vit_before = torch.cuda.memory_allocated() / 1024**3
        t_vit_start = time.perf_counter()
        with _torch_profile(activities=[_PA.CUDA], with_flops=False, record_shapes=False) as _prof:
            image_embeds, deepstack_embeds = self.visual(
                pixel_values_typed,
                grid_thw=image_grid_thw,
                keep_indices=keep_indices,
            )
        torch.cuda.synchronize()
        t_vit_end = time.perf_counter()
    else:
        image_embeds, deepstack_embeds = self.visual(
            pixel_values_typed,
            grid_thw=image_grid_thw,
            keep_indices=keep_indices,
        )

    if SHOULD_PROFILE:
        mem_vit_after = torch.cuda.memory_allocated() / 1024**3
        mem_vit_peak = torch.cuda.max_memory_allocated() / 1024**3
        # 重置 peak，让 model.py 侧捕获的是 LLM 阶段的 peak
        torch.cuda.reset_peak_memory_stats()

        rank = _get_rank()
        vit_input_lens = image_grid_thw.prod(dim=-1).tolist() if image_grid_thw is not None else []
        vit_output_lens = [len(idx) for idx in keep_indices] if keep_indices is not None else vit_input_lens
        vit_latency_ms = (t_vit_end - t_vit_start) * 1000

        # ── TFLOPs 理论值（线性层 + Flash Attention）───────────────────────
        # 注：torch.profiler 无法捕获 Flash Attention（custom CUDA kernel），
        # 因此不使用 profiler 实测 FLOPs，仅保留理论值计算。
        # 线性层：每层 FLOPs = N × (4H² + 2H×FFN) × 2（MAC→FLOPs）
        #   QKV+OutProj = 4H²（MACs），FFN 两层（fc1+fc2，标准 GELU MLP）= 2H×FFN（MACs）
        # Flash Attention：全序列自注意力（cu_seqlens），每层 FLOPs = 4 × N_i² × H
        #   包括 QK^T（2N²H）和 AV（2N²H），按每张图分开计算（图间无跨注意力）
        _vc  = getattr(self.visual, 'config', None)
        _H   = getattr(_vc, 'hidden_size', 1024)
        _FFN = getattr(_vc, 'intermediate_size', 4096)
        _D   = getattr(_vc, 'depth', 24)

        _linear_per_tok = (4 * _H * _H + 2 * _H * _FFN) * 2  # FLOPs per token per layer (linear only)

        # 当前（pruning 后）实际经过 ViT 所有层的 token 数（vit_prune_layer=0 时全部层都是 pruning 后）
        # 注意：attention 按每张图独立计算（O(N_i²)），须分图求和
        _attn_tflops     = _D * sum(4 * n * n * _H for n in vit_output_lens) / 1e12
        _attn_full_tflops = _D * sum(4 * n * n * _H for n in vit_input_lens)  / 1e12

        vit_flops_theory_tflops      = _D * sum(vit_output_lens) * _linear_per_tok / 1e12 + _attn_tflops
        vit_flops_theory_full_tflops = _D * sum(vit_input_lens)  * _linear_per_tok / 1e12 + _attn_full_tflops

        if VERBOSE:
            print(
                f"[ViT Profile] Rank {rank} | "
                f"VitInput: {vit_input_lens} | VitOutput: {vit_output_lens} | "
                f"Latency: {vit_latency_ms:.2f} ms | "
                f"TFLOPs: theory={vit_flops_theory_tflops:.4f} theory_full={vit_flops_theory_full_tflops:.4f} | "
                f"VRAM: {mem_vit_before:.2f} -> {mem_vit_after:.2f} (peak {mem_vit_peak:.2f}) GB"
            )

        # 缓存本次 ViT FLOPs 供 model.py 写 e2e jsonl 时读取
        # 同时缓存 new_input_len（loco 压缩后的实际 LLM 输入长度），供 model.py 正确计算 LLM prefill FLOPs
        _last_vit_flops[rank] = {
            "theory": vit_flops_theory_tflops,
            "theory_full": vit_flops_theory_full_tflops,
            "new_input_len": _pending_vit_record.get(rank, {}).get("new_input_len", None),
        }

        # 补充 timing/VRAM/TFLOPs 到 pending record 并 flush 到 per-rank vit log
        if vit_log_file and rank in _pending_vit_record:
            _pending_vit_record[rank].update({
                "vit_input_lens": vit_input_lens,
                "vit_output_lens": vit_output_lens,
                "vit_latency_ms": round(vit_latency_ms, 3),
                "vram_vit_before_gb": round(mem_vit_before, 4),
                "vram_vit_after_gb": round(mem_vit_after, 4),
                "vram_vit_peak_gb": round(mem_vit_peak, 4),
                "vit_flops_theory_tflops": round(vit_flops_theory_tflops, 6),
                "vit_flops_theory_full_tflops": round(vit_flops_theory_full_tflops, 6),
            })
            _write_jsonl(_pending_vit_record.pop(rank), _rank_path(vit_log_file, rank))
    spatial_merge_size = getattr(
        self.visual.config, "spatial_merge_size", 2
    )
    merge_size_sq = spatial_merge_size ** 2
    if keep_indices is not None:
        new_lengths = [len(idx) // merge_size_sq for idx in keep_indices]
        image_embeds = torch.split(image_embeds, new_lengths)
    else:
        split_sizes = (
            image_grid_thw.prod(-1) // merge_size_sq
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_embeds


def _model_forward(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    keep_indices: Optional[List[torch.Tensor]] = None,
    **kwargs: Any,
) -> Any:
    """Patched Qwen3VLModel.forward：支持 keep_indices，并做 inputs_embeds/position_ids 裁剪。"""
    from transformers.utils import is_torchdynamo_compiling

    if keep_indices is None:
        return _orig_model_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

    spatial_merge_size = getattr(self.visual.config, "spatial_merge_size", 2)
    block_size = spatial_merge_size ** 2
    merged_keep_indices = [idx[::block_size] // block_size for idx in keep_indices]

    vit_prune_layer = int(os.environ.get("PIXELPRUNE_VIT_LAYER", "0"))
    vit_keep_indices, merged_indices_list = (None, merged_keep_indices) if vit_prune_layer == -1 else (keep_indices, None)

    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw, vit_keep_indices)

    if merged_indices_list is not None:
        image_embeds = [seq_embeds[merged_indices_list[seq_idx].to(seq_embeds.device)] for seq_idx, seq_embeds in enumerate(image_embeds)]
        if deepstack_image_embeds and len(deepstack_image_embeds) > 0:
            merge_size_sq = spatial_merge_size ** 2
            split_sizes = (image_grid_thw.prod(-1) // merge_size_sq).tolist()
            deepstack_image_embeds = [
                torch.cat([torch.split(ds_layer_embeds, split_sizes)[i][merged_indices_list[i].to(ds_layer_embeds.device)]
                          for i in range(len(split_sizes))])
                for ds_layer_embeds in deepstack_image_embeds
            ]

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

    keep_mask = None
    original_input_ids = None
    original_attention_mask = None
    if pixel_values is not None and keep_indices is not None:
        original_input_ids = input_ids.clone()
        if attention_mask is not None:
            original_attention_mask = (
                attention_mask.clone()
                if isinstance(attention_mask, torch.Tensor)
                else {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in attention_mask.items()}
            )
        inputs_embeds, input_ids, keep_mask = _adjust_inputs_for_dedup(
            self, inputs_embeds, input_ids, image_embeds, image_grid_thw, "left",
            merged_indices=merged_keep_indices,
        )
        if cache_position is not None:
            cache_position = torch.arange(
                inputs_embeds.shape[1],
                device=cache_position.device,
                dtype=cache_position.dtype,
            )

    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
    deepstack_visual_embeds = deepstack_image_embeds

    mask_for_rope = original_attention_mask if original_attention_mask is not None else attention_mask
    attention_mask_tensor = (
        mask_for_rope if not isinstance(mask_for_rope, dict) else mask_for_rope.get("full_attention")
    )
    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
        attention_mask_tensor = torch.diagonal(
            attention_mask_tensor[:, 0], dim1=1, dim2=2
        )
        if attention_mask_tensor.dtype.is_floating_point:
            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

    prefill_compiled = is_torchdynamo_compiling() and (
        (input_ids is not None and input_ids.shape[1] != 1)
        or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
    )
    prefill_noncompiled = not is_torchdynamo_compiling() and (
        (cache_position is not None and cache_position[0] == 0)
        or past_key_values is None
        or past_key_values.get_seq_length() == 0
    )

    if (prefill_compiled or prefill_noncompiled) or self.rope_deltas is None:
        ids_for_rope = original_input_ids if original_input_ids is not None else input_ids
        position_ids, rope_deltas = self.get_rope_index(
            ids_for_rope,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        if keep_mask is not None:
            batch_size = position_ids.shape[1]
            adjusted_pos_ids = [position_ids[:, b, keep_mask[b]] for b in range(batch_size)]
            max_len = max(p.shape[1] for p in adjusted_pos_ids)
            padded_pos_ids = [
                F.pad(pos_ids, (max_len - pos_ids.shape[1], 0))
                for pos_ids in adjusted_pos_ids
            ]
            position_ids = torch.stack(padded_pos_ids, dim=1)
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                adjusted_attn = [
                    attention_mask[b, keep_mask[b]] for b in range(batch_size)
                ]
                padded_attn = [
                    F.pad(m, (max_len - m.shape[0], 0), value=0) for m in adjusted_attn
                ]
                attention_mask = torch.stack(padded_attn, dim=0)
        self.rope_deltas = rope_deltas
    else:
        batch_size, seq_length, _ = inputs_embeds.shape
        delta = (
            (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            if cache_position is not None
            else 0
        )
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
        if cache_position is not None:
            delta = delta.repeat_interleave(
                batch_size // delta.shape[0], dim=0
            )
        position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def _cg_forward(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
    **kwargs: Any,
) -> Any:
    """Patched Qwen3VLForConditionalGeneration.forward：在需要时计算 keep_indices 并传入 model。"""
    keep_indices = None

    _, vit_log_file, _ = _get_log_paths()
    SHOULD_PROFILE_VIT = bool(vit_log_file) or os.environ.get("PIXELPRUNE_VERBOSE", "false").lower() in ("true", "1", "yes")

    if (
        os.environ.get("PIXELPRUNE_ENABLED", "true").lower() in ("true", "1", "yes")
        and pixel_values is not None
        and image_grid_thw is not None
    ):
        spatial_merge_size = getattr(
            self.config.vision_config, "spatial_merge_size", 2
        )
        merge_size_sq = spatial_merge_size ** 2

        original_merged_lengths = [
            int((t * h * w) // merge_size_sq)
            for t, h, w in image_grid_thw.tolist()
        ]

        pixel_values_norm = normalize_pixel_values_for_selector(pixel_values)

        _t0_selector = time.perf_counter()
        merged_keep_indices = compute_merged_keep_indices(
            pixel_values_norm,
            image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )
        selector_latency_ms = (time.perf_counter() - _t0_selector) * 1000

        block_size = spatial_merge_size ** 2
        keep_indices = merged_indices_to_patch_indices(
            merged_keep_indices, block_size, pixel_values.device
        )

        new_merged_lengths = [len(idx) for idx in merged_keep_indices]
        _store_dedup_stats(
            original_merged_lengths, new_merged_lengths, input_ids, image_grid_thw,
            selector_latency_ms=selector_latency_ms,
        )
    elif SHOULD_PROFILE_VIT and pixel_values is not None and image_grid_thw is not None:
        # full 模式（PIXELPRUNE_ENABLED=false）：retain 100%，退化为 keep 所有 patch；
        # 仍需创建 pending vit record，使 _model_get_image_features 能写入 ViT timing/VRAM。
        spatial_merge_size = getattr(
            self.config.vision_config, "spatial_merge_size", 2
        )
        merge_size_sq = spatial_merge_size ** 2
        original_merged_lengths = [
            int((t * h * w) // merge_size_sq)
            for t, h, w in image_grid_thw.tolist()
        ]
        _store_dedup_stats(
            original_merged_lengths, original_merged_lengths, input_ids, image_grid_thw,
            selector_latency_ms=0.0,
        )

    return _orig_cg_forward(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        keep_indices=keep_indices,
        **kwargs,
    )


# 保存原始方法引用（在 apply 时写入）
_orig_vision_forward: Any = None
_orig_model_get_image_features: Any = None
_orig_model_forward: Any = None
_orig_cg_forward: Any = None


def apply_patches() -> None:
    """对 HuggingFace Qwen3-VL 应用 monkey-patch。"""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as m

    global _orig_vision_forward, _orig_model_get_image_features, _orig_model_forward, _orig_cg_forward

    if getattr(m.Qwen3VLVisionModel, "__patch_selector_patched__", False):
        return

    _orig_vision_forward = m.Qwen3VLVisionModel.forward
    m.Qwen3VLVisionModel.forward = _vision_forward

    _orig_model_get_image_features = m.Qwen3VLModel.get_image_features
    m.Qwen3VLModel.get_image_features = _model_get_image_features

    _orig_model_forward = m.Qwen3VLModel.forward
    m.Qwen3VLModel.forward = _model_forward

    _orig_cg_forward = m.Qwen3VLForConditionalGeneration.forward
    m.Qwen3VLForConditionalGeneration.forward = _cg_forward

    m.Qwen3VLVisionModel.__patch_selector_patched__ = True
    m.Qwen3VLModel.__patch_selector_patched__ = True
    m.Qwen3VLForConditionalGeneration.__patch_selector_patched__ = True
