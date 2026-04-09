"""
MultiTurn dataset and Buffer packing for Qwen VLM training (Qwen3-VL / Qwen3.5).

- MultiTurn: conversation dataset using AutoProcessor (auto-detects model type)
- Buffer: accumulates samples into packed sequences up to max_length
- BufferMultiTurn: iterable dataset combining MultiTurn + Buffer for data packing
"""

import re
import os
import json
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from glob import glob
from torch.utils.data import IterableDataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

_SPATIAL_MERGE_SIZE = 2  # Qwen3-VL / Qwen3.5 fixed value


# ---------------------------------------------------------------------------
# MultiTurn Dataset
# ---------------------------------------------------------------------------

class MultiTurn:
    """MultiTurn conversation dataset with Qwen3-VL native processor.

    Each sample is a multi-turn conversation (user/assistant/system) with optional images.
    Images are processed using Qwen3-VL's AutoProcessor which handles resizing,
    patch extraction, and token generation natively.
    """

    def __init__(
        self,
        file_pattern,
        task_name,
        model_path,
        max_length=2048,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.task_name = task_name
        self.file_pattern = file_pattern
        if isinstance(self.file_pattern, str):
            self.filepaths = sorted(glob(self.file_pattern, recursive=True))
        else:
            assert isinstance(self.file_pattern, list) and isinstance(self.file_pattern[0], str)
            self.filepaths = sorted(sum([glob(fp, recursive=True) for fp in self.file_pattern], []))

        # Use Qwen3-VL's native processor
        max_size = kwargs.pop('max_size', None)
        processor_kwargs = {}
        if max_size is not None:
            processor_kwargs['max_pixels'] = max_size * max_size
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, **processor_kwargs
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.eos_token != "<|im_end|>":
            self.tokenizer.eos_token = "<|im_end|>"

        self.assistant_start = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_end = self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])

        self.max_length = max_length
        self.role_map = {
            "user": "user",
            "human": "user",
            "assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
        }
        # Load data into memory (only current rank's portion)
        self.load_data()
        self.num_examples = len(self.example)

    def load_data(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if 'rank' in self.kwargs:
            rank = self.kwargs['rank']
        if 'world_size' in self.kwargs:
            world_size = self.kwargs['world_size']

        self.example = []
        example_id = 0
        for fp in self.filepaths:
            try:
                with open(fp, 'r', encoding="utf-8") as f:
                    for line in f:
                        if example_id % world_size == rank:
                            self.example.append(line)
                        example_id += 1
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                continue
        print(f'> rank-{rank}, task {self.task_name}, nums {len(self.example)}')

    def __len__(self):
        return self.num_examples

    def _build_messages(self, d):
        """Build structured messages supporting text/image multi-media interleaving.

        Supports two data formats:
        1. Inline format: <img>/path/to/image.jpg</img> tags in value
        2. <image> placeholder format: <image> placeholders in value + top-level "images" list
        """
        conversations = d.get("conversations")
        assert conversations and isinstance(conversations, list)

        # Detect format: if top-level "images" list exists, use <image> placeholder format
        image_list = d.get("images", None)
        use_image_list = image_list is not None and len(image_list) > 0

        media_pattern = re.compile(r"<img>(.+?)</img>")
        image_placeholder_pattern = re.compile(r"<image>")
        image_idx = 0  # tracks which image in the list to use next
        messages = []

        for conv in conversations:
            role = self.role_map.get(conv.get("from"))
            value = conv.get("value", "")

            if role is None:
                continue

            if role in ("system", "assistant"):
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": value.strip()}]}
                )
                continue

            # user: parse image references
            content = []
            last_end = 0

            if use_image_list:
                # <image> placeholder format: <image> placeholder + images list
                for m in image_placeholder_pattern.finditer(value):
                    if text := value[last_end:m.start()].strip():
                        content.append({"type": "text", "text": text})

                    assert image_idx < len(image_list), (
                        f"<image> placeholder count exceeds images list length "
                        f"({len(image_list)})"
                    )
                    img_path = image_list[image_idx].strip()
                    image_idx += 1
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    content.append({"type": "image", "image": img_path})
                    last_end = m.end()
            else:
                # Inline format: <img>path</img>
                for m in media_pattern.finditer(value):
                    if text := value[last_end:m.start()].strip():
                        content.append({"type": "text", "text": text})

                    img_path = m.group(1).strip()
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    content.append({"type": "image", "image": img_path})
                    last_end = m.end()

            if text := value[last_end:].strip():
                content.append({"type": "text", "text": text})

            if content:
                messages.append({"role": "user", "content": content})

        assert messages, "messages is empty, data format error"
        return messages

    def _create_labels(self, input_ids):
        """Create labels: only keep assistant response tokens, others set to -100."""
        labels = torch.full_like(input_ids, -100)
        ids = input_ids[0].tolist()
        assistant_start_len = len(self.assistant_start)
        assistant_end_len = len(self.assistant_end)

        i = 0
        while i <= len(ids) - assistant_start_len:
            if ids[i:i + assistant_start_len] == self.assistant_start:
                start = i + assistant_start_len
                end = start
                while end <= len(ids) - assistant_end_len:
                    if ids[end:end + assistant_end_len] == self.assistant_end:
                        labels[0, start:end + assistant_end_len] = input_ids[
                            0, start:end + assistant_end_len
                        ]
                        i = end + assistant_end_len - 1
                        break
                    end += 1
            i += 1

        return labels

    def process_conversation(self, d):
        """Process conversation: build messages, run processor, create labels and position_ids."""
        messages = self._build_messages(d)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=text,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        labels = self._create_labels(input_ids)

        # Compute 3D position_ids (mrope: temporal, height, width)
        # Shape: (3, batch, seq_len) — works for both Qwen3-VL and Qwen3.5
        attention_mask = torch.ones_like(input_ids)
        spatial_merge_size = self.processor.image_processor.merge_size
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        position_ids = _compute_qwen3vl_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
        )

        # Precompute prune indices on CPU if enabled
        merged_indices_list = self._compute_merged_keep_indices(pixel_values, image_grid_thw)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if merged_indices_list:
            result["merged_indices_list"] = merged_indices_list
        return result

    def _compute_merged_keep_indices(self, pixel_values, image_grid_thw):
        """Precompute merged token keep indices on CPU for each image.

        Uses pixelprune.compute_merged_keep_indices to determine which merged tokens
        to keep based on pixel redundancy. The processor's native pixel_values
        (total_patches, C*T*P*P) are converted to the required format.

        Returns empty list if pruning is not enabled or no images present.
        """
        if pixel_values is None or image_grid_thw is None:
            return []

        enable_prune = (
            os.environ.get('PIXELPRUNE_ENABLED', 'false').lower() in ('true', '1', 'yes')
        )
        if not enable_prune:
            return []

        prune_mode = os.environ.get('PIXELPRUNE_METHOD', 'pred_2d').lower()
        if prune_mode == 'random':
            return []

        from pixelprune import compute_merged_keep_indices

        # Processor output pixel_values: (total_patches, C * temporal_patch_size * patch_size * patch_size)
        # For single images, temporal_patch_size=2 but frames are duplicated.
        # We need to extract a single frame and normalize to [0,1] range.
        #
        # The pixel_values are already organized per-patch. We group them by
        # spatial_merge_size^2 to form merged tokens for the pruning algorithm.
        # compute_merged_keep_indices expects (total_merged_tokens, feat_dim)
        # with image_grid_thw containing original patch counts.

        spatial_merge_size = _SPATIAL_MERGE_SIZE
        block_size = spatial_merge_size * spatial_merge_size

        # Reshape pixel_values to extract single-frame pixel representation
        # pixel_values shape: (total_patches, C * T * P * P)
        # For Qwen3-VL: C=3, T=2 (temporal_patch_size), P=16 (patch_size)
        C, T, P = 3, 2, 16
        N = pixel_values.shape[0]
        pv_reshaped = pixel_values.reshape(N, C, T, P, P)
        # Take first temporal frame and flatten: (N, C*P*P)
        pv_single = pv_reshaped[:, :, 0, :, :].reshape(N, -1)
        # Normalize to [0,1] range (processor normalizes with mean/std)
        pv_single = pv_single * 0.5 + 0.5

        return compute_merged_keep_indices(
            pixel_values=pv_single,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )

    def __getitem__(self, idx):
        """Index-based access to the dataset."""
        example = self.example[idx]
        d = json.loads(example)
        result = self.process_conversation(d)

        if result["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f"WARN: Rank{dist.get_rank() if dist.is_initialized() else 0}, "
                f"task {self.task_name} sample {idx} length {result['input_ids'].shape[1]} "
                f"exceeds max_length {self.max_length}, skipping"
            )

        # Only keep non-None fields
        out = {}
        for k, v in result.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                out[k] = v.clone().contiguous()
            else:
                out[k] = v  # merged_indices_list is a list of tensors
        return out


# ---------------------------------------------------------------------------
# Buffer + BufferMultiTurn (packing)
# ---------------------------------------------------------------------------

class Buffer:
    """Accumulates samples until max_length is reached, then pops a packed batch."""

    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []
        self._len = 0

    def __len__(self):
        return self._len

    def add(self, d):
        if self._len + d['input_ids'].shape[1] > self.max_length:
            raise ValueError("the length of input_ids is larger than max_length")
        self._len = self._len + d['input_ids'].shape[1]
        self.data.append(d)

    def pop(self):
        if not self.data:
            return {}

        # Collect all keys that have non-None values
        all_keys = set()
        for sample in self.data:
            for k, v in sample.items():
                if v is not None:
                    all_keys.add(k)

        result = {}

        # pixel_values and image_grid_thw: cat on dim=0
        cat_dim0_keys = ['pixel_values', 'image_grid_thw']

        # merged_indices_list: collect as flat list (not tensor concat)
        list_concat_keys = ['merged_indices_list']

        for key in all_keys:
            valid_data = [d[key] for d in self.data if key in d and d[key] is not None]
            if not valid_data:
                continue

            if key in cat_dim0_keys:
                result[key] = torch.cat(valid_data, dim=0)
            elif key == 'position_ids':
                # 3D mrope position_ids: shape (3, 1, seq_len) -> cat on last dim
                squeezed = [d.squeeze(1) for d in valid_data]  # (3, seq_len)
                result[key] = torch.cat(squeezed, dim=-1).unsqueeze(1)  # (3, 1, total_seq_len)
            elif key in list_concat_keys:
                # List of tensors: just extend
                combined = []
                for d in valid_data:
                    if isinstance(d, list):
                        combined.extend(d)
                    else:
                        combined.append(d)
                result[key] = combined
            else:
                # 1D sequence tensors (input_ids, labels): squeeze, concat, unsqueeze
                concat_list = []
                for d in valid_data:
                    concat_list.extend(d.squeeze(0).tolist())
                result[key] = torch.tensor(concat_list).unsqueeze(0)

        assert result['input_ids'].shape[1] == self._len
        self._len = 0
        self.data = []
        return result


class _IterableDatasetBase(IterableDataset):
    """Base class for iterable datasets with epoch/worker management."""

    def __init__(
        self,
        file_pattern,
        task_name,
        model_path,
        max_length: int = 2048,
        random_seed=0,
        **kwargs,
    ):
        self.task_name = task_name
        self.max_length = max_length
        self.file_pattern = file_pattern
        if isinstance(self.file_pattern, str):
            self.filepaths = sorted(glob(self.file_pattern, recursive=True))
        else:
            assert isinstance(self.file_pattern, list) and isinstance(self.file_pattern[0], str)
            self.filepaths = sorted(sum([glob(fp, recursive=True) for fp in self.file_pattern], []))
        self.model_path = model_path
        self.num_examples = -1
        self.epoch = 0
        self.random_seed = random_seed
        self.kwargs = kwargs

    def set_epoch_seed(self, epoch):
        np.random.seed(self.random_seed + self.epoch)


class BufferMultiTurn(_IterableDatasetBase):
    """Iterable dataset that packs multiple MultiTurn samples into sequences of max_length."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = MultiTurn(**kwargs)
        self.num_examples = self.dataset.num_examples
        self.buffer = Buffer(self.max_length)

    def get_skip_size(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if 'rank' in self.kwargs:
            rank = self.kwargs['rank']
        if 'world_size' in self.kwargs:
            world_size = self.kwargs['world_size']

        worker = torch.utils.data.get_worker_info()
        num_workers = 1 if worker is None else worker.num_workers
        worker_id = 0 if worker is None else worker.id
        self.world_size = world_size
        self.rank = rank
        self.worker_id = worker_id
        self.num_workers = num_workers
        return num_workers, worker_id

    def get_worker_idxs(self):
        idxs = np.random.permutation(len(self.dataset))
        return idxs[self.worker_id::self.num_workers]

    def __iter__(self):
        self.get_skip_size()

        while True:
            self.set_epoch_seed(self.epoch)
            idxs = self.get_worker_idxs()
            for idx in idxs:
                try:
                    d = self.dataset[idx]
                    assert d['input_ids'].shape[1] <= self.max_length
                except Exception as e:
                    print(f"rank {self.rank} task {self.task_name} worker {self.worker_id} skip error: {e}")
                    continue
                try:
                    self.buffer.add(d)
                except:
                    yield self.buffer.pop()
                    self.buffer.add(d)
            if len(self.buffer) > 0:
                yield self.buffer.pop()
            self.epoch += 1


# ---------------------------------------------------------------------------
# 3D RoPE position IDs (Qwen3-VL)
# ---------------------------------------------------------------------------

def _compute_qwen3vl_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    spatial_merge_size=2,
    image_token_id=None,
    video_token_id=None,
    vision_start_token_id=None,
):
    """Compute 3D RoPE position IDs (mrope: temporal, height, width).

    Returns position_ids of shape (3, batch, seq_len).
    Compatible with both Qwen3-VL and Qwen3.5 (which accepts 3D mrope pos_ids).
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0

        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(
                input_ids_i == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum() if video_token_id is not None else 0

            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = len(input_tokens) + 1
                ed_video = len(input_tokens) + 1
                if image_token_id in input_tokens[st:] and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                if video_token_id is not None and video_token_id in input_tokens[st:] and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index].tolist()
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index].tolist()
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = int(t)
                llm_grid_h = int(h) // spatial_merge_size
                llm_grid_w = int(w) // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t).view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w).flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h).view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w).flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w).view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1).flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)

        return position_ids

    else:
        # Pure text
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            )
        return position_ids
