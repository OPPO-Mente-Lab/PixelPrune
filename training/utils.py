"""
PixelPrune Training Utilities.

Contains model loading, dataset creation, flash attention monkey-patching,
optimized loss computation, and various training helpers.
"""

import os
import sys
import copy
import random
import argparse
from collections import OrderedDict
import json
import time
import math
import torch
import torch.distributed as dist
import deepspeed
import numpy as np
import glob

import transformers
try:
    #Priority: flash_attn_interface > flash_attn
    from flash_attn_interface import flash_attn_varlen_func
except ImportError:
    from flash_attn import flash_attn_varlen_func
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

# KD imports
from liger_kernel.transformers import LigerFusedLinearJSD


# ---------------------------------------------------------------------------
# Seed & Logging
# ---------------------------------------------------------------------------

class Logger:
    """Lightweight logger: console + file + tensorboard. Replaces runx.logx."""

    def __init__(self, logdir, hparams=None, global_rank=0):
        self.rank = global_rank
        self.writer = None
        if self.rank == 0:
            os.makedirs(logdir, exist_ok=True)
            self._log_file = open(os.path.join(logdir, "log.txt"), "a")
            if hparams:
                with open(os.path.join(logdir, "hparams.json"), "w") as f:
                    json.dump(hparams, f, indent=2, ensure_ascii=False, default=str)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=logdir)

    def msg(self, text):
        if self.rank == 0:
            print(text, flush=True)
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def metric(self, tag, metrics: dict, step):
        if self.rank == 0 and self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{tag}/{k}", v, step)
            self.writer.flush()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def log_version(save_dir, rank):
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    year_month = time.strftime("%Y%m", time.localtime(time.time()))
    if os.environ.get('HOSTNAME', "").startswith('dlc'):
        if 'TASK_ID' in os.environ:
            task_id = os.environ['TASK_ID'].split('-')
            save_dir = os.path.join(save_dir, year_month, task_id[1] + "-task" + task_id[2])
        else:
            save_dir = os.path.join(save_dir, year_month, cur_time + "-" + os.environ['HOSTNAME'].split('-')[0])
    else:
        save_dir = os.path.join(save_dir, year_month, cur_time)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            objects = [save_dir]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=0, device=torch.device("cuda"))
        save_dir = objects[0]
    return save_dir

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--deepspeed_config_path", type=str, default="./configs/deepspeed_config.json"
    )
    parser.add_argument(
        "--training_config_path", type=str, default="./configs/training_config.json"
    )
    parser.add_argument(
        "--task_info_config_path", type=str, default="./configs/task_info.json"
    )
    parser.add_argument(
        "--task_weight_config_path", type=str, default="./configs/task_weight.json"
    )
    args = parser.parse_args()
    hparam = OrderedDict()
    hparam["local_rank"] = args.local_rank
    with open(args.deepspeed_config_path, "r", encoding="utf-8") as f:
        hparam["deepspeed_config"] = json.load(f)
    with open(args.training_config_path, "r", encoding="utf-8") as f:
        hparam["training_config"] = json.load(f)
    with open(args.task_info_config_path, "r", encoding="utf-8") as f:
        hparam["task_info"] = json.load(f)
    with open(args.task_weight_config_path, "r", encoding="utf-8") as f:
        hparam["task_weight"] = json.load(f)
    hparam["args"] = args
    return hparam


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

def sanity_check(hparam):
    """Validate model paths and dataset file patterns before training starts."""
    training_config = hparam["training_config"]
    task_info = hparam["task_info"]
    task_weight = hparam["task_weight"]

    model_path = training_config["model_path"]
    assert os.path.exists(model_path), f"ERROR: model_path {model_path} does not exist!"

    for task, weight in task_weight.items():
        if weight == 0:
            continue
        assert task in task_info, f"ERROR: Task {task} in task_weight but not found in task_info!"

        file_pattern = task_info[task]['dataset']['file_pattern']
        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]

        task_has_files = False
        for pattern in file_pattern:
            n = len(glob.glob(pattern, recursive=True))
            if n > 0:
                task_has_files = True
                break
        assert task_has_files, f"ERROR: Task {task} has no matching files for any pattern!"


# ---------------------------------------------------------------------------
# Sequence Length Utilities
# ---------------------------------------------------------------------------

def get_seq_lens(position_ids):
    """
    Given position_ids of shape (1, total_seq_len) containing packed sequences
    (each starting from 0), return cu_seqlens and max_seqlen.
    """
    total_seq_len = position_ids.size(1)
    diffs = (position_ids[0, 1:] == 0).nonzero(as_tuple=True)[0] + 1
    seq_lens = torch.diff(
        torch.cat((
            torch.tensor([0], device=position_ids.device),
            diffs,
            torch.tensor([total_seq_len], device=position_ids.device),
        ))
    )
    max_seqlen = seq_lens.max().item()
    cu_seqlens = torch.cat((
        torch.tensor([0], device=position_ids.device),
        torch.cumsum(seq_lens, dim=0),
    ), dim=0).to(torch.int32)
    return cu_seqlens, max_seqlen


# ---------------------------------------------------------------------------
# Flash Attention Monkey-Patches
# ---------------------------------------------------------------------------

from liger_kernel.transformers.functional import liger_rms_norm


def rms_norm_forward(self, hidden_states):
    return liger_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def rms_norm_forward_zero_centered(self, hidden_states):
    """Zero-centered RMSNorm: output = norm(x) * (1 + weight), offset=1.0."""
    return liger_rms_norm(hidden_states, self.weight, self.eps, offset=1.0)


def qwen3_vl_flat_flash_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    """Flat flash attention forward for Qwen3-VL text layers (varlen, no padding)."""
    position_ids = kwargs['position_ids']
    assert position_ids is not None and position_ids.shape[0] == 1
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cu_seqlens, max_seqlen = get_seq_lens(position_ids)

    attn_output = flash_attn_varlen_func(
        query_states.squeeze(0),
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def qwen3_5_flat_flash_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    position_ids=None,
    **kwargs,
):
    """Flat flash attention forward for Qwen3.5 full_attention layers (gated, partial RoPE, varlen)."""
    from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # q_proj outputs 2x head_dim: [query | gate]
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Compute cu_seqlens from position_ids (gradient-checkpointing safe, no globals)
    # position_ids is [3, bs, seq_len] for mrope; take temporal dim [0]
    pos_for_seqlens = position_ids[0] if position_ids.ndim == 3 else position_ids
    cu_seqlens, max_seqlen = get_seq_lens(pos_for_seqlens)

    attn_output = flash_attn_varlen_func(
        query_states.squeeze(0),
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Apply gated output
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Qwen3-VL Prune-Aware Forward
# ---------------------------------------------------------------------------

def _extract_single_frame(pixel_values: torch.Tensor, patch_embed) -> torch.Tensor:
    """Extract single frame from temporal-duplicated pixel_values.

    pixel_values layout is (N, C*T*P*P), flattened as (C, T, P, P).
    We reshape and take t=0 to get correct channel ordering.
    """
    weight_shape = patch_embed.proj.weight.shape
    in_channels = weight_shape[1]
    temporal_patch_size = weight_shape[2]
    patch_h = weight_shape[3]
    patch_w = weight_shape[4]

    N = pixel_values.shape[0]
    x = pixel_values.reshape(N, in_channels, temporal_patch_size, patch_h, patch_w)
    x_single = x[:, :, 0, :, :]
    return x_single.reshape(N, -1)


def _get_2d_weight_qwen3vl(patch_embed):
    """Get reduced 2D weights from 3D Conv3D weights for single-frame processing."""
    weight_3d = patch_embed.proj.weight
    weight_2d = weight_3d.sum(dim=2)
    _2d_weight = weight_2d.reshape(patch_embed.embed_dim, -1)
    return _2d_weight


def patch_embed_forward_qwen3vl(patch_embed, x: torch.Tensor) -> torch.Tensor:
    weight = _get_2d_weight_qwen3vl(patch_embed)
    bias = patch_embed.proj.bias if hasattr(patch_embed.proj, 'bias') else None
    return torch.nn.functional.linear(x, weight, bias)


def _adjust_inputs_for_prune(
    config,
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    image_embeds: torch.Tensor,
    new_lengths: list,
    original_lengths: list,
    image_grid_thw: torch.Tensor,
    merged_indices_list: list,
):
    """Adjust inputs_embeds and input_ids by removing pruned image token positions.

    Training uses packing (varlen): all sequences are concatenated, no padding needed.
    """
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    assert batch_size == 1, "Training should use packing strategy with batch_size=1"

    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id

    input_ids_flat = input_ids[0]
    inputs_embeds_flat = inputs_embeds[0]

    vision_start_indices = torch.where(input_ids_flat == vision_start_token_id)[0]

    if len(vision_start_indices) == 0:
        return inputs_embeds, input_ids, torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)

    vision_tokens = input_ids_flat[vision_start_indices + 1]
    image_positions = vision_start_indices[vision_tokens == image_token_id]

    if len(image_positions) == 0:
        return inputs_embeds, input_ids, torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)

    keep_mask_flat = torch.ones(seq_len, dtype=torch.bool, device=inputs_embeds.device)

    for img_idx, img_pos in enumerate(image_positions):
        num_original_tokens = original_lengths[img_idx]
        kept_indices = merged_indices_list[img_idx]

        assert kept_indices.max() < num_original_tokens, (
            f"[Prune Bug] img {img_idx}: max index {kept_indices.max()} "
            f">= original length {num_original_tokens}"
        )

        visual_start = img_pos + 1
        visual_end = visual_start + num_original_tokens

        keep_mask_flat[visual_start:visual_end] = False
        absolute_kept_indices = visual_start + kept_indices
        keep_mask_flat[absolute_kept_indices] = True

    new_inputs_embeds_flat = inputs_embeds_flat[keep_mask_flat]
    new_input_ids_flat = input_ids_flat[keep_mask_flat]

    return (
        new_inputs_embeds_flat.unsqueeze(0),
        new_input_ids_flat.unsqueeze(0),
        keep_mask_flat.unsqueeze(0),
    )


def qwen3_vl_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    image_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    hidden_only=False,
    no_prune=False,
    merged_indices_list=None,
    **kwargs,
):
    """Prune-aware forward for Qwen3-VL.

    Handles ViT encoding with optional pixel pruning at input/middle/output layers,
    LLM input adjustment, and per-token loss computation using Liger fused kernels.
    Returns sum-reduced loss (compatible with gradient accumulation correction in train.py).

    Now expects:
    - pixel_values: native processor output tensor (num_patches_total, patch_dim)
    - image_grid_thw: (num_images, 3)
    - position_ids: 3D tensor (3, batch, seq_len) pre-computed from data pipeline
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # PixelPrune config from env vars
    enable_prune = os.environ.get('PIXELPRUNE_ENABLED', 'false').lower() in ('true', '1', 'yes')
    vit_prune_layer = int(os.environ.get('PIXELPRUNE_VIT_LAYER', '0'))
    if no_prune:
        enable_prune = False

    prune_new_lengths = None
    prune_original_lengths = None
    keep_mask = None

    if pixel_values is not None:
        pixel_values = pixel_values.type(self.model.visual.patch_embed.proj.weight.dtype)
        hidden_states = pixel_values
        grid_thw = image_grid_thw

        # PixelPrune: validate PIXELPRUNE_VIT_LAYER
        if enable_prune:
            num_vit_blocks = len(self.model.visual.blocks)
            assert vit_prune_layer == -1 or 0 <= vit_prune_layer <= num_vit_blocks

        # Compute or use precomputed prune indices
        prune_indices = None
        precomputed = merged_indices_list is not None and len(merged_indices_list) > 0
        if not precomputed:
            merged_indices_list = None

        if enable_prune:
            spatial_merge_size = self.model.visual.spatial_merge_size
            block_size = spatial_merge_size * spatial_merge_size
            merged_lengths = [int(t * h * w) // block_size for t, h, w in image_grid_thw]

            if precomputed:
                merged_indices_list = [idx.to(hidden_states.device) for idx in merged_indices_list]
            else:
                if os.environ.get('PIXELPRUNE_VERBOSE', '').lower() in ('true', '1', 'yes'):
                    print("[PixelPrune] computing prune indices on-the-fly (not precomputed)")
                from pixelprune import compute_merged_keep_indices
                half_pixel_values = _extract_single_frame(hidden_states, self.model.visual.patch_embed) * 0.5 + 0.5
                merged_indices_list = compute_merged_keep_indices(
                    pixel_values=half_pixel_values,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )

            new_lengths = [len(indices) for indices in merged_indices_list]
            prune_original_lengths = merged_lengths
            prune_new_lengths = new_lengths

            if os.environ.get('PIXELPRUNE_VERBOSE', '').lower() in ('true', '1', 'yes'):
                total_original = sum(merged_lengths)
                total_new = sum(new_lengths)
                print(f"[PixelPrune] ViT tokens: {total_original} -> {total_new} "
                      f"(keep {total_new * 100 / total_original:.1f}%)"
                      f"{' [precomputed]' if precomputed else ''}")

            if vit_prune_layer >= 0:
                from pixelprune import merged_indices_to_patch_indices
                prune_indices = merged_indices_to_patch_indices(
                    merged_indices_list, block_size, hidden_states.device
                )
            else:
                prune_indices = merged_indices_list

        # Patch embedding: Conv3D -> Linear (single frame)
        hidden_states = patch_embed_forward_qwen3vl(
            self.model.visual.patch_embed,
            _extract_single_frame(hidden_states, self.model.visual.patch_embed),
        )

        pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
        seq_len_full = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len_full, -1)

        # Prune at ViT input layer
        if enable_prune and vit_prune_layer == 0 and prune_indices is not None:
            selected_hidden, selected_pos, selected_rope = [], [], []
            offset = 0
            for seq_idx, (t, h, w) in enumerate(grid_thw):
                seq_length = int(t * h * w)
                indices = prune_indices[seq_idx].to(hidden_states.device)
                selected_hidden.append(hidden_states[offset:offset + seq_length][indices])
                selected_pos.append(pos_embeds[offset:offset + seq_length][indices])
                selected_rope.append(rotary_pos_emb[offset:offset + seq_length][indices])
                offset += seq_length
            hidden_states = torch.cat(selected_hidden)
            pos_embeds = torch.cat(selected_pos)
            rotary_pos_emb = torch.cat(selected_rope)

        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens for ViT attention
        if enable_prune and vit_prune_layer == 0 and prune_indices is not None:
            seq_lengths = [len(indices) for indices in prune_indices]
            cu_seqlens = torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()
        else:
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()

        # ViT blocks
        cu_seqlens = cu_seqlens.int()
        for layer_num, blk in enumerate(self.model.visual.blocks):
            if self.training:
                hidden_states = self.model.visual._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk.__call__(hidden_states, cu_seqlens, None, position_embeddings)

            # Prune at intermediate layer
            if (enable_prune and vit_prune_layer > 0
                    and layer_num + 1 == vit_prune_layer and prune_indices is not None):
                selected_hidden, selected_cos, selected_sin = [], [], []
                offset = 0
                for seq_idx, (t, h, w) in enumerate(grid_thw):
                    seq_length = int(t * h * w)
                    indices = prune_indices[seq_idx].to(hidden_states.device)
                    selected_hidden.append(hidden_states[offset:offset + seq_length][indices])
                    selected_cos.append(position_embeddings[0][offset:offset + seq_length][indices])
                    selected_sin.append(position_embeddings[1][offset:offset + seq_length][indices])
                    offset += seq_length
                hidden_states = torch.cat(selected_hidden)
                position_embeddings = (torch.cat(selected_cos), torch.cat(selected_sin))
                seq_lengths = [len(indices) for indices in prune_indices]
                cu_seqlens = torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0)
                cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()

        # Merger
        image_embeds = self.model.visual.merger(hidden_states)

        # Prune: post-ViT compression (vit_prune_layer == -1)
        if enable_prune and vit_prune_layer == -1 and prune_indices is not None:
            spatial_merge_size = self.model.visual.spatial_merge_size
            split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size ** 2).tolist()
            image_embeds_list = list(torch.split(image_embeds, split_sizes))
            filtered_embeds = []
            for seq_idx, seq_embeds in enumerate(image_embeds_list):
                indices = prune_indices[seq_idx].to(seq_embeds.device)
                filtered_embeds.append(seq_embeds[indices])
            image_embeds = torch.cat(filtered_embeds, dim=0)
        elif not (enable_prune and vit_prune_layer >= 0 and prune_indices is not None):
            split_sizes = (image_grid_thw.prod(-1) // self.model.visual.spatial_merge_size ** 2).tolist()
            image_embeds_list = torch.split(image_embeds, split_sizes)
            image_embeds = torch.cat(image_embeds_list, dim=0)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # Adjust LLM inputs for prune
        if enable_prune and prune_new_lengths is not None:
            inputs_embeds, input_ids, keep_mask = _adjust_inputs_for_prune(
                self.config, inputs_embeds, input_ids, image_embeds,
                prune_new_lengths, prune_original_lengths,
                image_grid_thw, merged_indices_list,
            )
            if labels is not None:
                labels = labels[:, keep_mask[0]]
            if position_ids is not None:
                position_ids = position_ids[:, :, keep_mask[0]]

            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Ensure position_ids is 3D for Qwen3-VL
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    if hidden_only:
        return hidden_states, position_ids, labels

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    loss = None
    if labels is not None:
        cu_seqlens, _ = get_seq_lens(position_ids[0] if position_ids.ndim == 3 else position_ids)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        valid_seq_mask = seqlens > 1
        valid_starts = cu_seqlens[:-1][valid_seq_mask]
        valid_ends = cu_seqlens[1:][valid_seq_mask]

        indices = torch.cat([
            torch.arange(start, end - 1, device=labels.device, dtype=torch.int32)
            for start, end in zip(valid_starts, valid_ends)
        ])
        label_indices = indices + 1
        shift_labels = labels[:, label_indices].view(-1)
        valid_positions = shift_labels != -100
        final_indices = indices[valid_positions]
        valid_labels = shift_labels[valid_positions]

        selected_hidden = hidden_states[:, final_indices, :].view(-1, self.config.text_config.hidden_size)
        loss_fct = LigerFusedLinearCrossEntropyLoss(reduction='sum')
        loss = loss_fct(self.lm_head.weight, selected_hidden, valid_labels)

    if not return_dict:
        return (loss,) + (None,) + outputs[1:]

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def qwen3_5_gated_deltanet_flat_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    position_ids=None,
    **kwargs,
):
    """
    Varlen forward for Qwen3_5GatedDeltaNet.
    Uses fla causal_conv1d + chunk_gated_delta_rule with cu_seqlens for packing-aware training.
    Only replaces forward; __init__ and weights are unchanged.
    """
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    batch_size, seq_len, _ = hidden_states.shape

    # Compute cu_seqlens from position_ids (gradient-checkpointing safe)
    pos_for_seqlens = position_ids[0] if position_ids.ndim == 3 else position_ids
    cu_seqlens, _ = get_seq_lens(pos_for_seqlens)

    # ==================== Projections ====================
    qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    a = self.in_proj_a(hidden_states)
    b = self.in_proj_b(hidden_states)

    # ==================== Conv1d ====================
    # fla causal_conv1d: native varlen support, x shape [B, T, D]
    conv_weight = self.conv1d.weight.squeeze(1)  # [D, 1, K] -> [D, K]
    conv_out = causal_conv1d(
        x=qkv,
        weight=conv_weight,
        bias=self.conv1d.bias,
        cu_seqlens=cu_seqlens,
        activation=self.activation,
        output_final_state=False,
    )
    qkv = conv_out if not isinstance(conv_out, tuple) else conv_out[0]

    # ==================== Reshape ====================
    query = qkv[..., :self.key_dim].reshape(
        batch_size, seq_len, self.num_k_heads, self.head_k_dim
    )
    key = qkv[..., self.key_dim:self.key_dim * 2].reshape(
        batch_size, seq_len, self.num_k_heads, self.head_k_dim
    )
    value = qkv[..., self.key_dim * 2:].reshape(
        batch_size, seq_len, self.num_v_heads, self.head_v_dim
    )
    z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

    # ==================== Gates ====================
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)

    # GQA expansion
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # ==================== Core Attention ====================
    core_attn_out, _ = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    # ==================== Output ====================
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


def qwen3_5_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    image_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    hidden_only=False,
    no_prune=False,
    merged_indices_list=None,
    **kwargs,
):
    """Prune-aware forward for Qwen3.5.

    Same structure as qwen3_vl_lm_forward but adapted for Qwen3.5:
    - Passes 3D position_ids to Qwen3_5TextModel (mrope dims only; text_position_ids=None)
    - GatedDeltaNet (linear_attention) layers process packed sequence as-is (no isolation)
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # PixelPrune config from env vars
    enable_prune = os.environ.get('PIXELPRUNE_ENABLED', 'false').lower() in ('true', '1', 'yes')
    vit_prune_layer = int(os.environ.get('PIXELPRUNE_VIT_LAYER', '0'))
    if no_prune:
        enable_prune = False

    prune_new_lengths = None
    prune_original_lengths = None
    keep_mask = None

    if pixel_values is not None:
        pixel_values = pixel_values.type(self.model.visual.patch_embed.proj.weight.dtype)
        hidden_states = pixel_values
        grid_thw = image_grid_thw

        if enable_prune:
            num_vit_blocks = len(self.model.visual.blocks)
            assert vit_prune_layer == -1 or 0 <= vit_prune_layer <= num_vit_blocks

        prune_indices = None
        precomputed = merged_indices_list is not None and len(merged_indices_list) > 0
        if not precomputed:
            merged_indices_list = None

        if enable_prune:
            spatial_merge_size = self.model.visual.spatial_merge_size
            block_size = spatial_merge_size * spatial_merge_size
            merged_lengths = [int(t * h * w) // block_size for t, h, w in image_grid_thw]

            if precomputed:
                merged_indices_list = [idx.to(hidden_states.device) for idx in merged_indices_list]
            else:
                if os.environ.get('PIXELPRUNE_VERBOSE', '').lower() in ('true', '1', 'yes'):
                    print("[PixelPrune] computing prune indices on-the-fly (not precomputed)")
                from pixelprune import compute_merged_keep_indices
                half_pixel_values = _extract_single_frame(hidden_states, self.model.visual.patch_embed) * 0.5 + 0.5
                merged_indices_list = compute_merged_keep_indices(
                    pixel_values=half_pixel_values,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )

            new_lengths = [len(indices) for indices in merged_indices_list]
            prune_original_lengths = merged_lengths
            prune_new_lengths = new_lengths

            if os.environ.get('PIXELPRUNE_VERBOSE', '').lower() in ('true', '1', 'yes'):
                total_original = sum(merged_lengths)
                total_new = sum(new_lengths)
                print(f"[PixelPrune] ViT tokens: {total_original} -> {total_new} "
                      f"(keep {total_new * 100 / total_original:.1f}%)"
                      f"{' [precomputed]' if precomputed else ''}")

            if vit_prune_layer >= 0:
                from pixelprune import merged_indices_to_patch_indices
                prune_indices = merged_indices_to_patch_indices(
                    merged_indices_list, block_size, hidden_states.device
                )
            else:
                prune_indices = merged_indices_list

        # Patch embedding: Conv3D -> Linear (single frame)
        hidden_states = patch_embed_forward_qwen3vl(
            self.model.visual.patch_embed,
            _extract_single_frame(hidden_states, self.model.visual.patch_embed),
        )

        pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
        seq_len_full = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len_full, -1)

        # Prune at ViT input layer
        if enable_prune and vit_prune_layer == 0 and prune_indices is not None:
            selected_hidden, selected_pos, selected_rope = [], [], []
            offset = 0
            for seq_idx, (t, h, w) in enumerate(grid_thw):
                seq_length = int(t * h * w)
                indices = prune_indices[seq_idx].to(hidden_states.device)
                selected_hidden.append(hidden_states[offset:offset + seq_length][indices])
                selected_pos.append(pos_embeds[offset:offset + seq_length][indices])
                selected_rope.append(rotary_pos_emb[offset:offset + seq_length][indices])
                offset += seq_length
            hidden_states = torch.cat(selected_hidden)
            pos_embeds = torch.cat(selected_pos)
            rotary_pos_emb = torch.cat(selected_rope)

        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens for ViT attention
        if enable_prune and vit_prune_layer == 0 and prune_indices is not None:
            seq_lengths = [len(indices) for indices in prune_indices]
            cu_seqlens = torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()
        else:
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()

        # ViT blocks
        cu_seqlens = cu_seqlens.int()
        for layer_num, blk in enumerate(self.model.visual.blocks):
            if self.training:
                hidden_states = self.model.visual._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk.__call__(hidden_states, cu_seqlens, None, position_embeddings)

            if (enable_prune and vit_prune_layer > 0
                    and layer_num + 1 == vit_prune_layer and prune_indices is not None):
                selected_hidden, selected_cos, selected_sin = [], [], []
                offset = 0
                for seq_idx, (t, h, w) in enumerate(grid_thw):
                    seq_length = int(t * h * w)
                    indices = prune_indices[seq_idx].to(hidden_states.device)
                    selected_hidden.append(hidden_states[offset:offset + seq_length][indices])
                    selected_cos.append(position_embeddings[0][offset:offset + seq_length][indices])
                    selected_sin.append(position_embeddings[1][offset:offset + seq_length][indices])
                    offset += seq_length
                hidden_states = torch.cat(selected_hidden)
                position_embeddings = (torch.cat(selected_cos), torch.cat(selected_sin))
                seq_lengths = [len(indices) for indices in prune_indices]
                cu_seqlens = torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0)
                cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).int()

        # Merger
        image_embeds = self.model.visual.merger(hidden_states)

        # Prune: post-ViT compression (vit_prune_layer == -1)
        if enable_prune and vit_prune_layer == -1 and prune_indices is not None:
            spatial_merge_size = self.model.visual.spatial_merge_size
            split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size ** 2).tolist()
            image_embeds_list = list(torch.split(image_embeds, split_sizes))
            filtered_embeds = []
            for seq_idx, seq_embeds in enumerate(image_embeds_list):
                indices = prune_indices[seq_idx].to(seq_embeds.device)
                filtered_embeds.append(seq_embeds[indices])
            image_embeds = torch.cat(filtered_embeds, dim=0)
        elif not (enable_prune and vit_prune_layer >= 0 and prune_indices is not None):
            split_sizes = (image_grid_thw.prod(-1) // self.model.visual.spatial_merge_size ** 2).tolist()
            image_embeds_list = torch.split(image_embeds, split_sizes)
            image_embeds = torch.cat(image_embeds_list, dim=0)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # Adjust LLM inputs for prune
        if enable_prune and prune_new_lengths is not None:
            inputs_embeds, input_ids, keep_mask = _adjust_inputs_for_prune(
                self.config, inputs_embeds, input_ids, image_embeds,
                prune_new_lengths, prune_original_lengths,
                image_grid_thw, merged_indices_list,
            )
            if labels is not None:
                labels = labels[:, keep_mask[0]]
            if position_ids is not None:
                position_ids = position_ids[:, :, keep_mask[0]]

            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Ensure position_ids is 3D for Qwen3.5 mrope (temporal, height, width)
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    if hidden_only:
        return hidden_states, position_ids, labels

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    loss = None
    if labels is not None:
        cu_seqlens, _ = get_seq_lens(position_ids[0] if position_ids.ndim == 3 else position_ids)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        valid_seq_mask = seqlens > 1
        valid_starts = cu_seqlens[:-1][valid_seq_mask]
        valid_ends = cu_seqlens[1:][valid_seq_mask]

        indices = torch.cat([
            torch.arange(start, end - 1, device=labels.device, dtype=torch.int32)
            for start, end in zip(valid_starts, valid_ends)
        ])
        label_indices = indices + 1
        shift_labels = labels[:, label_indices].view(-1)
        valid_positions = shift_labels != -100
        final_indices = indices[valid_positions]
        valid_labels = shift_labels[valid_positions]

        selected_hidden = hidden_states[:, final_indices, :].view(-1, self.config.text_config.hidden_size)
        loss_fct = LigerFusedLinearCrossEntropyLoss(reduction='sum')
        loss = loss_fct(self.lm_head.weight, selected_hidden, valid_labels)

    if not return_dict:
        return (loss,) + (None,) + outputs[1:]

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ---------------------------------------------------------------------------
# Fuse (Monkey-Patch) for Training
# ---------------------------------------------------------------------------

def fuse():
    """Apply monkey-patches for efficient training: flash attention, RMS norm, fused MLP."""
    from liger_kernel.transformers import LigerSwiGLUMLP

    # Qwen3-VL patches
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = rms_norm_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3_vl_flat_flash_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lm_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP

    # Qwen3.5 patches (if transformers version supports it)
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
        # Zero-centered RMSNorm via liger_rms_norm(offset=1.0)
        modeling_qwen3_5.Qwen3_5RMSNorm.forward = rms_norm_forward_zero_centered
        from transformers.models.qwen3_next import modeling_qwen3_next
        modeling_qwen3_next.Qwen3NextRMSNorm.forward = rms_norm_forward_zero_centered
        # Gated flash attention for full_attention layers only
        modeling_qwen3_5.Qwen3_5Attention.forward = qwen3_5_flat_flash_forward
        # Prune-aware LM forward
        modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = qwen3_5_lm_forward
        # Fused SwiGLU MLP (wrapper to match Qwen3_5MLP(config, intermediate_size) signature)
        _OrigLigerSwiGLUMLP = LigerSwiGLUMLP

        class Qwen3_5LigerSwiGLUMLP(_OrigLigerSwiGLUMLP):
            def __init__(self, config, intermediate_size=None):
                if intermediate_size is not None:
                    config = copy.copy(config)
                    config.intermediate_size = intermediate_size
                super().__init__(config)

        modeling_qwen3_5.Qwen3_5MLP = Qwen3_5LigerSwiGLUMLP
        # GatedDeltaNet: varlen forward with fla causal_conv1d + chunk_gated_delta_rule
        modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward = qwen3_5_gated_deltanet_flat_forward

        # Patch decoder layer to pass position_ids to linear_attn
        # (needed so GatedDeltaNet can compute cu_seqlens from position_ids
        #  instead of globals, which breaks gradient checkpointing in KD mode)
        _orig_decoder_forward = modeling_qwen3_5.Qwen3_5DecoderLayer.forward

        def _patched_decoder_forward(self, hidden_states, position_embeddings=None,
                                     attention_mask=None, position_ids=None,
                                     past_key_values=None, cache_position=None, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            if self.layer_type == "linear_attention":
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            elif self.layer_type == "full_attention":
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        modeling_qwen3_5.Qwen3_5DecoderLayer.forward = _patched_decoder_forward
        print("[fuse] Qwen3.5 monkey-patches applied successfully.")
    except (ImportError, AttributeError) as e:
        print(f"[fuse] Qwen3.5 patches skipped (not available): {e}")


# ---------------------------------------------------------------------------
# Save HuggingFace Checkpoint
# ---------------------------------------------------------------------------

def save_hf_checkpoint(engine, save_path, hparam):
    """Save a complete HuggingFace checkpoint (config, tokenizer, safetensors).

    Uses DeepSpeed ZeRO's 16-bit param gathering to consolidate weights on rank 0.
    Saves processor first (tokenizer + image_processor), then model (weights + config.json),
    so model's config.json correctly overwrites the processor's version.
    """
    from transformers import AutoProcessor
    rank = dist.get_rank()

    state_dict = {}
    for name, param in engine.module.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            if rank == 0:
                state_dict[name] = param.data.clone().cpu()

    if rank == 0:
        model_path = hparam["training_config"]["model_path"]

        # Save processor first (tokenizer, image_processor, chat_template)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(save_path)

        # Save model weights + config.json (overwrites processor's config.json)
        model = engine.module
        original_device = next(model.parameters()).device
        model.to("cpu")
        model.load_state_dict(state_dict, strict=False)
        model.save_pretrained(
            save_path,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        model.to(original_device)

        print(f"[save_hf_checkpoint] Saved HF checkpoint to {save_path}")

    dist.barrier()


# ---------------------------------------------------------------------------
# Model Loading (using AutoModelForImageTextToText)
# ---------------------------------------------------------------------------

def get_model(hparam):
    """Load VLM model (Qwen3-VL or Qwen3.5) with gradient checkpointing."""
    from transformers import AutoModelForImageTextToText

    rank = torch.distributed.get_rank()
    model_path = hparam["training_config"]["model_path"]

    print(f"> rank {rank} loading model from {model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing
    model.model.language_model.gradient_checkpointing_enable()
    visual = getattr(model, 'visual', None) or model.model.visual
    visual.gradient_checkpointing_enable()

    print(f"> rank {rank} model loaded, type: {getattr(model.config, 'model_type', 'unknown')}")
    return model


def get_ref_model(hparam):
    """Load frozen reference model for knowledge distillation.

    If ref_model_path is not set, uses model_path (self-distillation).
    """
    from transformers import AutoModelForImageTextToText

    rank = torch.distributed.get_rank()
    ref_model_path = hparam["training_config"].get(
        "ref_model_path", hparam["training_config"]["model_path"]
    )

    print(f"> rank {rank} loading ref_model from {ref_model_path} ...")
    ref_model = AutoModelForImageTextToText.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    print(f"> rank {rank} ref_model loaded and frozen")
    return ref_model


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def get_dataset(hparam):
    """Create DataGenerator for training."""
    from data.data_generator import DataGenerator

    dataloader = DataGenerator(
        model_path=hparam["training_config"]["model_path"],
        tasks_info=hparam["task_info"],
        train_micro_batch_size_per_gpu=hparam["deepspeed_config"]["train_micro_batch_size_per_gpu"],
        gradient_accumulation_steps=hparam["deepspeed_config"]["gradient_accumulation_steps"],
        task_weight_config_path=hparam["args"].task_weight_config_path,
        max_length=hparam["training_config"]["max_length"],
        max_size=hparam["training_config"].get("max_size"),
        packing=hparam["training_config"]["packing"],
        random_seed=hparam["training_config"]["random_seed"],
    )
    return dataloader


# ---------------------------------------------------------------------------
# KD Loss Computation
# ---------------------------------------------------------------------------

def compute_kd_loss(
    student_hidden_states,
    teacher_hidden_states,
    student_lm_head_weight,
    teacher_lm_head_weight,
    student_labels,
    student_position_ids,
    teacher_labels,
    teacher_position_ids,
    temperature=1.0,
    alpha=0.,
):
    """
    Compute KD loss: alpha * CE + (1 - alpha) * KL.

    Student and Teacher may have different sequence lengths (Student does prune),
    so we extract valid tokens separately and align by label values.

    Loss = alpha * L_ce + (1 - alpha) * L_kl

    Returns:
        total_loss, ce_loss, kl_loss, num_valid_tokens
    """
    def _extract_valid_hidden_and_labels(hidden_states, labels, position_ids):
        pos_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
        cu_seqlens, _ = get_seq_lens(pos_ids)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        valid_seq_mask = seqlens > 1
        valid_starts = cu_seqlens[:-1][valid_seq_mask]
        valid_ends = cu_seqlens[1:][valid_seq_mask]

        indices = torch.cat([
            torch.arange(start, end - 1, device=labels.device, dtype=torch.int32)
            for start, end in zip(valid_starts, valid_ends)
        ])
        label_indices = indices + 1
        shift_labels = labels[:, label_indices].view(-1)
        valid_positions = shift_labels != -100
        final_indices = indices[valid_positions]
        valid_labels = shift_labels[valid_positions]
        hidden_size = hidden_states.shape[-1]
        selected_hidden = hidden_states[:, final_indices, :].view(-1, hidden_size)
        return selected_hidden, valid_labels

    student_hidden, student_valid_labels = _extract_valid_hidden_and_labels(
        student_hidden_states, student_labels, student_position_ids
    )
    teacher_hidden, teacher_valid_labels = _extract_valid_hidden_and_labels(
        teacher_hidden_states, teacher_labels, teacher_position_ids
    )
    teacher_hidden = teacher_hidden.detach()

    # Alignment verification
    if student_valid_labels.shape != teacher_valid_labels.shape:
        raise RuntimeError(
            f"[KD] Label count mismatch: student={student_valid_labels.shape[0]}, "
            f"teacher={teacher_valid_labels.shape[0]}."
        )
    if not torch.equal(student_valid_labels, teacher_valid_labels):
        mismatch_mask = student_valid_labels != teacher_valid_labels
        first_mismatch = mismatch_mask.nonzero(as_tuple=True)[0][0].item()
        raise RuntimeError(
            f"[KD] Label value mismatch at position {first_mismatch}: "
            f"student={student_valid_labels[first_mismatch].item()}, "
            f"teacher={teacher_valid_labels[first_mismatch].item()}."
        )

    num_valid_tokens = student_valid_labels.numel()
    valid_labels = student_valid_labels

    ce_loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    ce_loss = ce_loss_fct(student_lm_head_weight, student_hidden, valid_labels)

    distill_loss_fct = LigerFusedLinearJSD(
        jsd_beta=0,
        temperature=temperature,
        ignore_index=-100,
    )
    kl_loss = distill_loss_fct(
        student_hidden,
        student_lm_head_weight,
        teacher_hidden,
        teacher_lm_head_weight,
        valid_labels,
    )
    kl_loss = kl_loss * num_valid_tokens * (temperature ** 2)

    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return total_loss, ce_loss, kl_loss, num_valid_tokens
