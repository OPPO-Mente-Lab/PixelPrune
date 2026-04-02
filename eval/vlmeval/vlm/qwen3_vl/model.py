from __future__ import annotations

import logging
import os
import time
import warnings

import torch

import math
import numpy as np
from PIL import Image as PILImage

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr

_PIXELPRUNE_APPLIED_BACKENDS = set()


# ---------------------------------------------------------------------------
# PixelPrune integration
# ---------------------------------------------------------------------------

def _ensure_pixelprune_importable() -> None:
    """确保 pixelprune 包可导入。优先尝试直接 import，失败则将项目根目录加入 sys.path。"""
    import sys
    try:
        import pixelprune  # noqa: F401
        return
    except ImportError:
        pass
    # 从环境变量 PIXELPRUNE_PATH 或上级目录查找
    pp_path = os.environ.get('PIXELPRUNE_PATH', '')
    if pp_path and pp_path not in sys.path:
        sys.path.insert(0, pp_path)
        return
    # fallback: 假设 PixelPrune 项目在 VLMEvalKit 的同级目录
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    candidates = [
        repo_dir,  # VLMEvalKit 根目录本身（如果 pixelprune 装在这里）
        os.path.join(os.path.dirname(repo_dir), 'PixelPrune'),  # 同级 PixelPrune 目录
    ]
    for p in candidates:
        if p not in sys.path and os.path.isdir(os.path.join(p, 'pixelprune')):
            sys.path.insert(0, p)
            return


def _compute_pixelprune_retain_ratio(
    image_path: str,
    min_pixels: int,
    max_pixels: int,
    dedup_method: str = "exact",
    dedup_threshold: float = 0.0,
) -> float:
    """对一张图片模拟 PixelPrune (Pred2DLoco)，返回 retain ratio (0, 1]。

    在像素空间按 Qwen3-VL 的 merged token grid (32x32 像素/token) 切分图片，
    然后跑 LOCO-I 预测编码，计算保留比例。

    NOTE: exact match (threshold=0) 下结果与实际 PixelPrune 完全一致，
    因为像素相同性不受归一化方式影响。near-exact 下可能有微小差异。
    """
    import torch
    from qwen_vl_utils import smart_resize

    path = image_path[7:] if image_path.startswith('file://') else image_path
    img = PILImage.open(path).convert('RGB')
    orig_w, orig_h = img.size

    # Qwen3-VL merged token 的空间尺寸: patch_size(16) * spatial_merge_size(2) = 32
    factor = 32
    resized_h, resized_w = smart_resize(
        orig_h, orig_w, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    img_resized = img.resize((resized_w, resized_h), PILImage.BICUBIC)
    img.close()

    # 切为 merged token grid: 每个 token = factor x factor x 3 像素
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, 3]
    merged_h = resized_h // factor
    merged_w = resized_w // factor
    tokens = arr.reshape(merged_h, factor, merged_w, factor, 3)
    tokens = tokens.transpose(0, 2, 1, 3, 4)  # [merged_h, merged_w, factor, factor, 3]
    tokens = tokens.reshape(merged_h * merged_w, -1)  # [N, D]

    tokens_t = torch.from_numpy(tokens)

    _ensure_pixelprune_importable()
    from pixelprune.methods.pred_2d import Pred2DSelector

    selector = Pred2DSelector(method=dedup_method, threshold=dedup_threshold)
    keep_indices = selector._select_2d_loco(tokens_t, merged_h, merged_w, tokens_t.device)
    return len(keep_indices) / (merged_h * merged_w)


def _apply_resize_baseline(
    item: dict,
    min_pixels: int,
    max_pixels: int,
) -> None:
    """对一个 image content item，计算 PixelPrune retain ratio 并设置缩小后的 resized_height/width。

    修改 item in-place，添加 resized_height 和 resized_width 字段。
    """
    from qwen_vl_utils import smart_resize

    image_url = item.get('image', '')
    path = image_url[7:] if image_url.startswith('file://') else image_url
    if not os.path.exists(path):
        return

    img = PILImage.open(path)
    orig_w, orig_h = img.size
    img.close()

    factor = 32  # Qwen3-VL: patch_size=16 * spatial_merge_size=2
    item_min = item.get('min_pixels', min_pixels)
    item_max = item.get('max_pixels', max_pixels)

    # 原始 Qwen3-VL 分辨率
    resized_h, resized_w = smart_resize(
        orig_h, orig_w, factor=factor,
        min_pixels=item_min, max_pixels=item_max,
    )

    dedup_method = os.environ.get('DEDUP_METHOD', '') or 'exact'
    dedup_threshold = float(os.environ.get('DEDUP_THRESHOLD', '') or '0.0')

    retain_ratio = _compute_pixelprune_retain_ratio(
        image_url, item_min, item_max,
        dedup_method=dedup_method,
        dedup_threshold=dedup_threshold,
    )

    # 按 sqrt(retain_ratio) 缩小宽高
    scale = math.sqrt(retain_ratio)
    new_h = max(factor, int(resized_h * scale))
    new_w = max(factor, int(resized_w * scale))

    # smart_resize 对齐到 factor 的倍数，但不要让 min_pixels 把它拉回去
    # 用较小的 min_pixels 确保 resize 后确实缩小了
    new_h, new_w = smart_resize(
        new_h, new_w, factor=factor,
        min_pixels=factor * factor,  # 最小 1 个 merged token
        max_pixels=item_max,
    )

    VERBOSE = os.environ.get('VERBOSE', 'false').lower() in ('true', '1', 'yes')
    if VERBOSE:
        orig_tokens = (resized_h // factor) * (resized_w // factor)
        new_tokens = (new_h // factor) * (new_w // factor)
        print(
            f"[Resize Baseline] {orig_w}x{orig_h} -> {resized_w}x{resized_h} ({orig_tokens} tokens)"
            f" -> {new_w}x{new_h} ({new_tokens} tokens)"
            f" | PixelPrune retain={retain_ratio:.3f}, resize retain={new_tokens/orig_tokens:.3f}"
        )

    item['resized_height'] = new_h
    item['resized_width'] = new_w


def _resolve_patch_backend(model_path: str) -> str:
    """Return 'qwen3_vl' or 'qwen3_5' for apply_pixelprune(model=...)."""
    model_type = os.environ.get('MODEL_TYPE', '').strip().lower()
    if model_type in ('qwen3_5', 'qwen3.5', 'qwen35'):
        return 'qwen3_5'
    if model_type in ('qwen3_vl', 'qwen3-vl', 'qwen3vl'):
        return 'qwen3_vl'
    if listinstr(['qwen3.5', 'qwen3_5'], model_path.lower()):
        return 'qwen3_5'
    return 'qwen3_vl'


def _maybe_apply_pixelprune(model_path: str) -> None:
    """Apply PixelPrune monkey-patch if enabled via env vars."""
    enabled = (
        os.environ.get('PIXELPRUNE_ENABLED', os.environ.get('ENABLE_DEDUP', 'false')).lower()
        in ('true', '1', 'yes')
    )
    if not enabled and not os.environ.get('DEDUP_LOG_FILE', ''):
        return

    model_type = _resolve_patch_backend(model_path)
    if model_type in _PIXELPRUNE_APPLIED_BACKENDS:
        return

    _ensure_pixelprune_importable()
    try:
        from pixelprune import apply_pixelprune
        apply_pixelprune(model=model_type)
        _PIXELPRUNE_APPLIED_BACKENDS.add(model_type)
        print(f"PixelPrune applied successfully (model={model_type})")
    except Exception as e:
        print(f"Error applying PixelPrune: {e}")
        raise e

_QWEN35_FA2_PATCHED = False


def _fix_qwen35_fa2_position_ids():
    """Fix transformers bug: Qwen3.5 passes 3D mrope position_ids to decoder layers,
    which leaks through **kwargs into flash_attention_forward. The 3D tensor causes
    _is_packed_sequence to misinterpret it as packed sequences, generating wrong
    cu_seqlens and crashing with 'CUDA illegal memory access'.

    Fix: monkey-patch Qwen3_5DecoderLayer.forward to drop position_ids for
    full_attention layers (rotary embedding is already pre-computed as position_embeddings).
    """
    global _QWEN35_FA2_PATCHED
    if _QWEN35_FA2_PATCHED:
        return
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    except ImportError:
        return

    _orig_forward = Qwen3_5DecoderLayer.forward

    def _patched_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                         position_ids=None, past_key_values=None, cache_position=None, **kwargs):
        if self.layer_type == "full_attention":
            position_ids = None
        return _orig_forward(
            self, hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    Qwen3_5DecoderLayer.forward = _patched_forward
    _QWEN35_FA2_PATCHED = True
    print("[Fix] Patched Qwen3.5 FA2 position_ids leak (3D mrope → _is_packed_sequence crash)")


VLLM_MAX_IMAGE_INPUT_NUM = 24


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model by looking for active-param suffixes like A3B, A17B."""
    import re
    if re.search(r'-A\d+B', model_path):
        return True
    return False


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path

        _maybe_apply_pixelprune(self.model_path)

        # Fix Qwen3.5 + flash_attention_2 incompatibility in transformers
        if listinstr(['qwen3.5', 'qwen3_5'], model_path.lower()):
            _fix_qwen35_fa2_position_ids()

        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        # 预存 assistant start token ids 用于可靠解码
        # Qwen3 格式: "<|im_start|>assistant\n"
        self._assistant_prefix_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False
        )
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=8,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]

                # Resize baseline: 用 PixelPrune 的 per-image retain ratio 缩小图像分辨率
                if os.environ.get('RESIZE_BASELINE', 'false').lower() in ('true', '1', 'yes'):
                    _default_min = 4 * 32 * 32
                    _default_max = 16384 * 32 * 32
                    _apply_resize_baseline(
                        item,
                        min_pixels=item.get('min_pixels', self.min_pixels or _default_min),
                        max_pixels=item.get('max_pixels', self.max_pixels or _default_max),
                    )
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        system_msgs = [s for s in message if s.get('role') == 'system']
        user_msgs = [s for s in message if s.get('role') != 'system']

        messages = []
        if system_msgs:
            system_content = [{'type': 'text', 'text': s['value']} for s in system_msgs]
            messages.append({'role': 'system', 'content': system_content})
        elif self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(user_msgs, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        # STATS_ONLY：仅计算压缩率统计后写入 vit log，跳过模型前向传播
        if (
            not is_omni
            and os.environ.get('PIXELPRUNE_ENABLED', os.environ.get('ENABLE_DEDUP', 'false')).lower() in ('true', '1', 'yes')
            and os.environ.get('STATS_ONLY', 'false').lower() in ('true', '1', 'yes')
        ):
            pixel_values = inputs.get('pixel_values')
            image_grid_thw = inputs.get('image_grid_thw')
            if pixel_values is not None and image_grid_thw is not None:
                import torch.distributed as _dist
                from pixelprune.core import compute_merged_keep_indices
                _is_qwen3_5 = 'qwen3.5' in self.model_path.lower() or 'qwen3_5' in self.model_path.lower()
                if _is_qwen3_5:
                    from pixelprune.patches.qwen3_5_hf import (
                        normalize_pixel_values_for_selector,
                        _store_dedup_stats,
                        flush_pending_vit_record,
                    )
                else:
                    from pixelprune.patches.qwen3_vl_hf import (
                        normalize_pixel_values_for_selector,
                        _store_dedup_stats,
                        flush_pending_vit_record,
                    )
                _rank = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
                spatial_merge_size = getattr(
                    getattr(self.model.config, 'vision_config', self.model.config),
                    'spatial_merge_size', 2
                )
                merge_size_sq = spatial_merge_size ** 2
                original_merged_lengths = [
                    int((t * h * w) // merge_size_sq)
                    for t, h, w in image_grid_thw.tolist()
                ]
                pixel_values_norm = normalize_pixel_values_for_selector(pixel_values)
                merged_keep_indices = compute_merged_keep_indices(
                    pixel_values_norm, image_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )
                new_merged_lengths = [len(idx) for idx in merged_keep_indices]
                _store_dedup_stats(
                    original_merged_lengths, new_merged_lengths,
                    inputs['input_ids'], image_grid_thw,
                    stats_only=True,
                )
                flush_pending_vit_record(_rank)
            return ''

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            VERBOSE = os.environ.get('VERBOSE', 'false').lower() in ('true', '1', 'yes')
            if VERBOSE:
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**3
                # Use TextIteratorStreamer to measure TTFT, and capture generate output for accurate token count
                from transformers import TextIteratorStreamer
                import threading

                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                num_input_tokens = inputs.input_ids.shape[-1]
                generate_kwargs_with_streamer = dict(
                    **inputs,
                    **self.generate_kwargs,
                    streamer=streamer,
                )

                # Container for generate result to get accurate token ids
                generate_result = [None]
                t_start = time.perf_counter()
                ttft = None
                first_token_received = threading.Event()

                def generate_in_thread():
                    result = self.model.generate(**generate_kwargs_with_streamer)
                    generate_result[0] = result

                thread = threading.Thread(target=generate_in_thread)
                thread.start()

                # Wait for first token to measure TTFT
                for _ in streamer:
                    if not first_token_received.is_set():
                        ttft = time.perf_counter() - t_start
                        first_token_received.set()
                    # Continue consuming streamer to avoid blocking

                thread.join()
                t_end = time.perf_counter()

                # Get accurate token ids from generate result
                generated_ids_full = generate_result[0]
                full_sequence = generated_ids_full[0]
                prefix_len = len(self._assistant_prefix_ids)

                # 从后往前查找 prefix (处理可能的 input 被修剪的情况)
                response_start = None
                for i in range(len(full_sequence) - prefix_len, -1, -1):
                    if full_sequence[i:i + prefix_len].tolist() == self._assistant_prefix_ids:
                        response_start = i + prefix_len
                        break

                generated_ids = full_sequence[response_start:]

                # Accurate token count (includes eos and all special tokens)
                num_generated_tokens = len(generated_ids)

                # Decode response
                response = self.processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                total_time = t_end - t_start

                # Decoding speed: exclude first token, calculate speed for remaining tokens
                if num_generated_tokens > 1 and ttft is not None:
                    decoding_time = total_time - ttft
                    decoding_tokens_per_sec = (num_generated_tokens - 1) / decoding_time if decoding_time > 0 else 0
                else:
                    decoding_time = 0
                    decoding_tokens_per_sec = 0

                mem_after = torch.cuda.memory_allocated() / 1024**3
                mem_peak = torch.cuda.max_memory_allocated() / 1024**3
                import torch.distributed as _dist
                rank = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else torch.cuda.current_device()
                num_input_tokens = inputs.input_ids.shape[-1]
                vit_input_lens = inputs['image_grid_thw'].prod(dim=-1).tolist() if 'image_grid_thw' in inputs else []

                print(f"[Profile] Rank {rank} | "
                        f"Input: {num_input_tokens} | VitInput: {vit_input_lens} | "
                        f"Gen: {num_generated_tokens} | TTFT: {ttft*1000:.2f} ms | Decoding Speed: {decoding_tokens_per_sec:.2f} tok/s "
                        f"({num_generated_tokens-1} tokens / {decoding_time:.3f}s) | Total: {total_time:.2f}s | VRAM: {mem_before:.2f} -> {mem_after:.2f} (LLM-phase peak {mem_peak:.2f}) GB")

                # 写 E2E profile 到 {base}.e2e.rank{rank}{ext}
                # 只要设置了 DEDUP_LOG_FILE 即生效，不依赖 ENABLE_DEDUP
                _dedup_log_base = os.environ.get('DEDUP_LOG_FILE', '')
                if _dedup_log_base:
                    try:
                        import json as _json, fcntl as _fcntl
                        _b, _e = os.path.splitext(_dedup_log_base)
                        _e2e_path = f"{_b}.e2e.rank{rank}{_e}"
                        _igt = inputs.get('image_grid_thw')
                        _vc = getattr(getattr(self.model, 'config', None), 'vision_config', None)
                        _sms = getattr(_vc, 'spatial_merge_size', 2) if _vc is not None else 2
                        _vto = int(_igt.prod(-1).sum() // (_sms ** 2)) if _igt is not None else 0

                        # ── 先读 patch 模块缓存（含 new_input_len）────────────────────
                        try:
                            _is_q35 = 'qwen3.5' in self.model_path.lower() or 'qwen3_5' in self.model_path.lower()
                            if _is_q35:
                                from pixelprune.patches.qwen3_5_hf import _last_vit_flops as _lvf
                            else:
                                from pixelprune.patches.qwen3_vl_hf import _last_vit_flops as _lvf
                            _vit_cache = _lvf.get(rank, {})
                        except Exception:
                            _vit_cache = {}
                        _vit_theory          = _vit_cache.get("theory", 0.0)
                        _vit_theory_full     = _vit_cache.get("theory_full", 0.0)

                        # ── LLM Prefill 理论 TFLOPs ──────────────────────────────────
                        # 从 text_config 读取 LLM 结构参数
                        _tc = getattr(getattr(self.model, 'config', None), 'text_config', None)
                        _L   = getattr(_tc, 'num_hidden_layers', 28)
                        _H   = getattr(_tc, 'hidden_size', 2048)
                        _FFN = getattr(_tc, 'intermediate_size', 6144)
                        _nh  = getattr(_tc, 'num_attention_heads', 16)
                        _nkv = getattr(_tc, 'num_key_value_heads', 8)
                        _dh  = getattr(_tc, 'head_dim', 128)
                        # 优先使用 loco 压缩后的实际 LLM 输入长度（new_input_len），
                        # 压缩在模型 forward 内部发生，inputs.input_ids 仍是原始长度
                        _new_input_len = _vit_cache.get("new_input_len", None)
                        _S = _new_input_len if _new_input_len is not None else num_input_tokens

                        # Q/K/V/Out 投影（GQA）
                        _flops_qkvo = 2 * _S * (_H * _nh * _dh          # Q proj
                                                 + _H * _nkv * _dh * 2  # K+V proj
                                                 + _nh * _dh * _H)      # Out proj
                        # Causal attention score + weighted sum（prefill 为 O(S²)）
                        _flops_attn = 4 * _S * _S * _nh * _dh
                        # FFN：gate + up + down（SwiGLU，3 个线性层）
                        _flops_ffn  = 2 * _S * _H * _FFN * 3
                        _llm_prefill_tflops = _L * (_flops_qkvo + _flops_attn + _flops_ffn) / 1e12

                        _line = _json.dumps({
                            "type": "e2e",
                            "rank": rank,
                            "sample_idx": int(os.environ.get("CURRENT_SAMPLE_IDX", "-1")),
                            "num_input_tokens": num_input_tokens,
                            "text_tokens": num_input_tokens - _vto,
                            "vision_tokens_org": _vto,
                            "num_generated_tokens": num_generated_tokens,
                            "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
                            "decoding_speed_tok_per_s": round(decoding_tokens_per_sec, 3),
                            "decoding_time_s": round(decoding_time, 4),
                            "total_time_s": round(total_time, 4),
                            "vram_before_gb": round(mem_before, 4),
                            "vram_after_gb": round(mem_after, 4),
                            "vram_llm_peak_gb": round(mem_peak, 4),
                            # ViT FLOPs 理论值（线性层 + Flash Attention，来自 patch 模块缓存）
                            "vit_flops_theory_tflops": round(_vit_theory, 6),
                            "vit_flops_theory_full_tflops": round(_vit_theory_full, 6),
                            # LLM prefill 理论 TFLOPs（使用压缩后实际输入长度）
                            "llm_prefill_tflops": round(_llm_prefill_tflops, 6),
                            # E2E 理论 TFLOPs（ViT theory + LLM prefill）
                            "e2e_flops_theory_tflops": round(_vit_theory + _llm_prefill_tflops, 6),
                            "e2e_flops_theory_full_tflops": round(_vit_theory_full + _llm_prefill_tflops, 6),
                        }, ensure_ascii=False) + "\n"
                        with open(_e2e_path, 'a', encoding='utf-8') as _f:
                            _fcntl.flock(_f, _fcntl.LOCK_EX)
                            try: _f.write(_line); _f.flush()
                            finally: _fcntl.flock(_f, _fcntl.LOCK_UN)
                    except Exception:
                        pass
            else:
                num_input_tokens = inputs.input_ids.shape[-1]
                _max_new_tokens_env = os.environ.get('QWEN3VL_MAX_NEW_TOKENS')
                generated_ids_full = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                    **({"max_new_tokens": int(_max_new_tokens_env)} if _max_new_tokens_env else {}),
                )
                generated_ids_full = generated_ids_full[0]  # 先取第一个 batch，变成 1D tensor

                response_start = None
                prefix_len = len(self._assistant_prefix_ids)
                for i in range(len(generated_ids_full) - prefix_len, -1, -1):
                    if generated_ids_full[i:i + prefix_len].tolist() == self._assistant_prefix_ids:
                        response_start = i + prefix_len
                        break

                generated_ids = generated_ids_full[response_start:]
                response = self.processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )


        # Qwen3.5: enable_thinking=False injects empty <think>\n\n</think> in template;
        # strip it, then assert no real thinking content leaked through.
        import re
        response = re.sub(r'<think>\s*</think>\s*', '', response, flags=re.DOTALL).strip()
        assert '<think>' not in response, (
            f'Unexpected non-empty <think> block (enable_thinking=False should prevent this): {response[:300]}'
        )

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        system_msgs = [s for s in message if s.get('role') == 'system']
        user_msgs = [s for s in message if s.get('role') != 'system']

        messages = []
        if system_msgs:
            system_content = [{'type': 'text', 'text': s['value']} for s in system_msgs]
            messages.append({'role': 'system', 'content': system_content})
        elif self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(user_msgs, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        # Strip <think>...</think> blocks (Qwen3.5 thinking mode)
        import re
        generated_text = re.sub(r'<think>.*?</think>\s*', '', generated_text, flags=re.DOTALL).strip()

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
