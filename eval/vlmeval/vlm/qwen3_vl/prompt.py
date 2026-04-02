from __future__ import annotations

import string
import pandas as pd


class Qwen3VLPromptMixin:
    """
    Mixin class for Qwen3VLChat to build prompts consistent with Qwen3-VL README.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        # Follow Qwen3-VL convention: apply concise, task-specified prompts for MCQ/YN/VQA
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        # Dataset-specific prompts for OCR and chart understanding benchmarks
        if dataset and any(dataset.startswith(prefix) for prefix in [
            'DocVQA', 'InfoVQA', 'ChartQA', 'AI2D',
            'OCRBench', 'OCRBench_v2', 'CC-OCR', 'CharXiv'
        ]):
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N':
            return True
        if dataset_type == 'VQA':
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Build prompt for different datasets with optimized formats."""
        from vlmeval.dataset import DATASET_TYPE

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        hint = line.get('hint')
        if hint is not None and pd.isna(hint):
            hint = None

        # Helper function to build options prompt
        def build_options_prompt():
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            if not options:
                return '', options
            options_str = 'Options:\n' + '\n'.join(f'{k}. {v}' for k, v in options.items())
            return options_str, options

        # Dataset-specific prompts
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            options_prompt, options = build_options_prompt()
            prompt_parts = []
            if hint is not None:
                prompt_parts.append(f'Hint: {hint}')
            prompt_parts.append(f'Question: {question}')
            if options:
                prompt_parts.append(options_prompt)
                prompt_parts.append('Please select the correct answer from the options above.')
            prompt = '\n'.join(prompt_parts).rstrip()

        # DocVQA / InfoVQA / ChartQA: answer with a single word or phrase
        elif dataset and any(dataset.startswith(prefix) for prefix in ['DocVQA', 'InfoVQA', 'ChartQA']):
            prompt = f'{question}\nAnswer the question using a single word or phrase.'

        # AI2D: select correct answer from options
        elif dataset and dataset.startswith('AI2D'):
            options_prompt, _ = build_options_prompt()
            prompt = f'Question: {question}\n{options_prompt}Please select the correct answer from the options above.'
            print(f"[DEBUG AI2D] prompt repr: {repr(prompt)}")

        # OCRBench / OCRBench_v2 / CC-OCR / CharXiv: plain question without suffix
        elif dataset and any(dataset.startswith(prefix) for prefix in ['OCRBench', 'OCRBench_v2', 'CC-OCR', 'CharXiv']):
            prompt = question

        # Type-based prompts
        else:
            dataset_type = DATASET_TYPE(dataset, default=None)

            if dataset_type == 'MCQ':
                options_prompt, options = build_options_prompt()
                prompt_parts = []
                if hint is not None:
                    prompt_parts.append(f'Hint: {hint}')
                prompt_parts.append(f'Question: {question}')
                if options:
                    prompt_parts.append(options_prompt)
                    prompt_parts.append('Answer with the option letter only.')
                prompt = '\n'.join(prompt_parts).rstrip()

            elif dataset_type == 'Y/N':
                prompt = f'{question} Please answer yes or no.'

            elif dataset_type == 'VQA':
                prompt = f'{question}\nPlease answer concisely with short words or phrases when possible.'

            else:
                raise ValueError(f'Unsupported dataset: {dataset}')

        # Build message with images first (Qwen3-VL convention)
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))
        msgs.append(dict(type='text', value=prompt))

        return msgs
