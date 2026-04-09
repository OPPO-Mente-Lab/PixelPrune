"""
DataGenerator: task weighting, data loading,
and cross-rank padding for balanced computation.
"""

import os
import json
from collections import defaultdict
from typing import TypeVar, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler as SamplerBase, Dataset

from .dataset import MultiTurn, BufferMultiTurn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

T_co = TypeVar('T_co', covariant=True)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler(SamplerBase[T_co]):
    """Simple deterministic sampler for reproducible training."""

    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0) -> None:
        self.epoch = 0
        self.num_samples = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DataGenerator(object):
    def __init__(
        self,
        tasks_info,
        task_weight_config_path='',
        train_micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=1,
        max_length=2048,
        max_size=None,
        model_path="",
        packing=False,
        task_weight={},
        random_seed=0,
        **kwargs,
    ):
        self.tasks_info = tasks_info
        self.task_weight_config_path = task_weight_config_path
        self.task_weight = task_weight
        self.train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps
        self.max_length = max_length
        self.max_size = max_size
        self.model_path = model_path
        self.packing = packing
        self.random_seed = random_seed
        self.kwargs = kwargs

        self.setup()

    def setup(self):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if 'rank' in self.kwargs:
            self.rank = self.kwargs['rank']
        if 'world_size' in self.kwargs:
            self.world_size = self.kwargs['world_size']

        self.set_task_weight()

        self.task_to_loader = {}
        self.task_dataset = {}
        for task_name, task_info in self.tasks_info.items():
            task_info['dataset']['task_name'] = task_name
            task_info['dataset']['rank'] = self.rank
            task_info['dataset']['world_size'] = self.world_size
            task_info['epoch'] = 0

            assert task_info['task_type'] == "MultiTurn", (
                f"Only MultiTurn task type is supported, got {task_info['task_type']}"
            )

            if self.packing:
                task_dataset = BufferMultiTurn(
                    **task_info['dataset'],
                    model_path=self.model_path,
                    max_length=self.max_length,
                    max_size=self.max_size,
                    random_seed=self.random_seed,
                )
            else:
                task_dataset = MultiTurn(
                    **task_info['dataset'],
                    model_path=self.model_path,
                    max_length=self.max_length,
                    max_size=self.max_size,
                )

            task_info['num_examples'] = task_dataset.num_examples
            self.task_dataset[task_name] = task_dataset

        for task_name in self.tasks_info.keys():
            self.create_loader(task_name)

        # Get image_token_id from the last dataset's processor
        self.processor = task_dataset.processor if hasattr(task_dataset, 'processor') else task_dataset.dataset.processor
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Detect Qwen3.5 (has GatedDeltaNet linear attention that needs fixed-length input)
        self._is_qwen3_5 = self._detect_qwen3_5()
        if self._is_qwen3_5:
            self.pad_length = self.max_length * self.train_micro_batch_size_per_gpu
            print(f" > Qwen3.5 detected, padding to fixed length {self.pad_length}")
        else:
            self.pad_length = None

        task_to_size = {tn: ti['num_examples'] for tn, ti in self.tasks_info.items()}
        task_to_weight = {tn: ti['task_weight'] for tn, ti in self.tasks_info.items()}
        print(f" > rank-{self.rank}, num tasks {len(self.task_to_loader)}, "
              f"task_to_size {task_to_size}, task_to_weight {task_to_weight}")

    def _detect_qwen3_5(self):
        """Detect if model is Qwen3.5 by checking config file in model_path."""
        import json as _json
        config_path = os.path.join(self.model_path, "config.json")
        try:
            with open(config_path, 'r') as f:
                config = _json.load(f)
            model_type = config.get("model_type", "")
            return "qwen3_5" in model_type or "qwen3_next" in model_type
        except Exception:
            return False

    def set_task_weight(self):
        if self.task_weight:
            tasks_weight = self.task_weight
        else:
            try:
                with open(self.task_weight_config_path, 'r') as f:
                    tasks_weight = json.load(f)
                    self.task_weight = tasks_weight
            except (IOError, json.JSONDecodeError) as e:
                raise Exception(f"Error reading task weight config: {e}")

        tasks_weight_sum = sum(v for v in tasks_weight.values())
        if tasks_weight_sum == 0:
            raise ValueError("Sum of task weights cannot be zero")
        tasks_weight = {k: v / tasks_weight_sum for k, v in tasks_weight.items()}
        print(f"real task weight {tasks_weight}")

        empty_tasks = set()
        for task in self.tasks_info:
            weight = tasks_weight.get(task, 0)
            self.tasks_info[task]['task_weight'] = weight
            if weight == 0:
                empty_tasks.add(task)

        for t in empty_tasks:
            del self.tasks_info[t]

    def seed_to_task(self, seed):
        total = 0
        for task_name, task_info in self.tasks_info.items():
            total += task_info['task_weight']
            if total > seed:
                return task_name
        return task_name

    def create_loader(self, task_name):
        if self.task_to_loader.get(task_name, None) is not None:
            del self.task_to_loader[task_name]

        task_dataset = self.task_dataset[task_name]
        task_info = self.tasks_info[task_name]

        if hasattr(task_dataset, '__len__'):
            sampler = Sampler(task_dataset, seed=self.random_seed)
            sampler.set_epoch(task_info['epoch'])
            task_info['epoch'] += 1
        else:
            sampler = None
            task_dataset.epoch = task_info['epoch']
            task_info['epoch'] += 1

        task_loader = DataLoader(
            dataset=task_dataset,
            batch_size=1,
            shuffle=task_info.get('shuffle', False),
            num_workers=task_info.get('num_workers', 0),
            prefetch_factor=(
                task_info.get('prefetch_factor', 2)
                if task_info.get('num_workers', 0) != 0
                else None
            ),
            pin_memory=task_info.get("pin_memory", True),
            collate_fn=lambda x: x,
            sampler=sampler,
            drop_last=task_info.get('drop_last', False),
        )
        self.task_to_loader[task_name] = iter(task_loader)

    def __iter__(self):
        batch_id = 0
        while True:
            batch_id += 1
            batch = []
            while len(batch) < self.batch_size:
                seed = np.random.random()
                task_name = self.seed_to_task(seed)
                task_loader = self.task_to_loader[task_name]
                try:
                    feature = next(task_loader)
                except StopIteration:
                    print(f' > rank {self.rank} task name {task_name} Finished one epoch ...')
                    self.create_loader(task_name)
                    task_loader = self.task_to_loader[task_name]
                    continue
                except Exception as e:
                    print(f"| Error!!! ---> task name: {task_name} batch_id {batch_id} "
                          f"rank {self.rank} | error: {e} |")
                    continue
                batch.extend(feature)
            yield batch_seq_collate_fn(batch, self.image_token_id, pad_length=self.pad_length)


def pad_image(batch, image_token_id):
    """Pad with a dummy image when no pixel_values exist (pure text batch)."""
    batch['pixel_values'] = torch.zeros(4, 16 * 16 * 3 * 2)
    batch['image_grid_thw'] = torch.tensor([[1, 2, 2]])
    batch['input_ids'] = torch.cat([
        batch['input_ids'],
        torch.zeros(1, 1).type_as(batch['input_ids']) + image_token_id
    ], dim=-1)
    batch['position_ids'] = torch.cat([
        batch['position_ids'],
        torch.zeros(batch['position_ids'].shape[0], 1, 1).type_as(batch['position_ids'])
    ], dim=-1)
    batch['labels'] = torch.cat([
        batch['labels'],
        torch.zeros(1, 1).type_as(batch['labels']) - 100
    ], dim=-1)


def batch_seq_collate_fn(features: list, image_token_id, pad_length=None):
    """Collate multiple samples into a single packed batch.

    Handles:
    - Concatenation of input_ids, labels (removing batch dim)
    - 3D position_ids concatenation on last dim
    - pixel_values and image_grid_thw concatenation on dim=0
    - Padding pure-text batches with dummy images
    - Padding to fixed length for Qwen3.5 (avoids fla recompilation)
    """
    batch = defaultdict(list)
    for feature in features:
        for k in feature.keys():
            if k in ['input_ids', 'labels']:
                batch[k].extend(feature[k])
            else:
                batch[k].append(feature[k])

    batch['input_ids'] = torch.cat(batch['input_ids'], dim=0).unsqueeze(0)
    batch['labels'] = torch.cat(batch['labels'], dim=0).unsqueeze(0)

    # 3D position_ids: concat from each feature
    batch['position_ids'] = torch.cat([f['position_ids'] for f in features], dim=-1)

    # pixel_values and image_grid_thw
    if 'pixel_values' in batch:
        batch['pixel_values'] = torch.cat(batch['pixel_values'], dim=0)
        batch['image_grid_thw'] = torch.cat(batch['image_grid_thw'], dim=0)
    else:
        pad_image(batch, image_token_id)

    # merged_indices_list: flatten list of lists
    if 'merged_indices_list' in batch:
        combined = []
        for item in batch['merged_indices_list']:
            if isinstance(item, list):
                combined.extend(item)
            else:
                combined.append(item)
        batch['merged_indices_list'] = combined if combined else None
    else:
        batch['merged_indices_list'] = None

    # Pad to fixed length (Qwen3.5: avoids fla recompilation on variable lengths)
    if pad_length is not None:
        current_length = batch['input_ids'].size(-1)
        if current_length < pad_length:
            pad_size = pad_length - current_length

            batch['input_ids'] = torch.cat([
                batch['input_ids'],
                torch.zeros(1, pad_size, dtype=batch['input_ids'].dtype),
            ], dim=-1)

            batch['labels'] = torch.cat([
                batch['labels'],
                torch.full((1, pad_size), -100, dtype=batch['labels'].dtype),
            ], dim=-1)

            # position_ids: pad as a fake extra sample (arange from 0)
            # shape: (3, 1, seq_len) for mrope
            pos_pad = torch.arange(pad_size, dtype=batch['position_ids'].dtype)
            pos_pad = pos_pad.view(1, 1, -1).expand(
                batch['position_ids'].size(0), 1, -1
            )
            batch['position_ids'] = torch.cat([
                batch['position_ids'], pos_pad,
            ], dim=-1)

    return batch
