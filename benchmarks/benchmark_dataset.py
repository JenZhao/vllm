# SPDX-License-Identifier: Apache-2.0
"""
benchmark_dataset.py

This module defines a framework for
sampling benchmark requests from various datasets.
Each dataset subclass of BenchmarkDataset must implement sample generation.
Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: Dict[int, AnyTokenizer] = {}

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """Represents a single inference request for benchmarking."""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 enable_lora_tokenizer: bool = False,
                 lora_path: Optional[str] = None,
                 max_loras: Optional[int] = None,
                 num_requests: int = 0,
                 input_len: Optional[int] = None,
                 output_len: Optional[int] = None,
                 dataset_path: Optional[str] = None,
                 model: Optional[str] = None,
                 data: Optional[List] = None) -> None:
        self.tokenizer = tokenizer
        self.data = data  # For datasets that require pre-loading
        self.dataset_path = dataset_path

        # lora related
        self.enable_lora_tokenizer = enable_lora_tokenizer
        self.lora_path = lora_path
        self.max_loras = max_loras

        self.num_requests = num_requests
        self.input_len = input_len
        self.output_len = output_len
        if self.num_requests is None:
            raise ValueError("num_requests must be provided for sampling.")

        self.model = model

        if self.enable_lora_tokenizer and not self.lora_path:
            raise ValueError("LoRA is enabled but no lora_path provided.")

    @abstractmethod
    def load_data(self) -> None:
        """Load data from the specified dataset path."""
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
            self) -> Tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Return a tuple (lora_request, tokenizer) for tokenizing requests.
        If LoRA is enabled, returns the LoRA-specific tokenizer;
        otherwise, the base tokenizer.
        """
        if not self.enable_lora_tokenizer:
            return None, self.tokenizer

        if self.max_loras is None:
            raise ValueError(
                "max_lora must be set when enabling LoRA tokenizer.")

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, self.max_loras)
        lora_request = LoRARequest(lora_name=str(lora_id),
                                   lora_int_id=lora_id,
                                   lora_path=lora_path_on_disk(self.lora_path))
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        return lora_request, lora_tokenizer_cache[lora_id]

    @abstractmethod
    def sample(self) -> List:
        """Generate sample requests from the dataset."""
        pass


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 num_requests: int = 0,
                 input_len: Optional[int] = None,
                 output_len: Optional[int] = None,
                 prefix_len: Optional[int] = None,
                 range_ratio: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(tokenizer,
                         num_requests=num_requests,
                         input_len=input_len,
                         output_len=output_len,
                         **kwargs)
        assert self.input_len is not None \
            and self.output_len is not None, \
                "input_len and output_len must be set for RandomDataset"
        self.prefix_len = prefix_len
        self.range_ratio = range_ratio

    def load_data(self) -> None:
        # No data loading needed for RandomDataset.
        pass

    def sample_serving(self) -> List:
        assert self.range_ratio is not None \
         and self.prefix_len is not None, \
            "range_ratio and prefix_len must be \
                set for RandomDataset when returning tuple."

        vocab_size = self.tokenizer.vocab_size

        prefix_token_ids = np.random.randint(
            0, vocab_size,
            size=self.prefix_len).tolist() if self.prefix_len > 0 else []

        input_low = int(self.input_len * self.range_ratio)
        output_low = int(self.output_len * self.range_ratio)

        input_lens = np.random.randint(input_low,
                                       self.input_len + 1,
                                       size=self.num_requests)
        output_lens = np.random.randint(output_low,
                                        self.output_len + 1,
                                        size=self.num_requests)
        offsets = np.random.randint(0, vocab_size, size=self.num_requests)

        requests = []
        for i in range(self.num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                         vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = self.tokenizer.decode(token_sequence)
            total_input_len = self.prefix_len + int(input_lens[i])
            requests.append(
                (prompt, total_input_len, int(output_lens[i]), None))
        return requests

    def sample(self, return_tuple=False) -> List:
        if return_tuple:
            return self.sample_serving()
        vocab_size = self.tokenizer.vocab_size
        requests = []
        for _ in range(self.num_requests):
            lora_request, tokenizer = self.get_random_lora_request()

            candidate_ids = [
                random.randint(0, vocab_size - 1)
                for _ in range(self.input_len)
            ]
            candidate_prompt = tokenizer.decode(candidate_ids)

            for _ in range(5):
                tokenized_len = len(tokenizer.encode(candidate_prompt))
                if tokenized_len == self.input_len:
                    break
                diff = self.input_len - tokenized_len
                if diff > 0:
                    candidate_ids += [
                        random.randint(100, vocab_size - 100)
                        for _ in range(diff)
                    ]
                else:
                    candidate_ids = candidate_ids[:diff]
                candidate_prompt = tokenizer.decode(candidate_ids)

            requests.append(
                SampleRequest(prompt=candidate_prompt,
                              prompt_len=self.input_len,
                              expected_output_len=self.output_len,
                              lora_request=lora_request))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.
    Loads data from a JSON file and generates sample requests 
    based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.data is None:
            self.load_data()

    def _get_prompt_for_image_model(self, question: str) -> str:
        """Prepend and append special tokens around the question
        to form a prompt."""
        model = self.model.lower() if self.model else ""
        if "pixtral" in model:
            return f"<s>[INST]{question}\n[IMG][/INST]"
        raise ValueError(f"Unsupported model {model}")

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.shuffle(self.data)

    def sample(self, return_tuple=False) -> List:

        if self.num_requests is None:
            raise ValueError("num_requests must be provided for sampling.")

        samples: List[SampleRequest] = []
        for entry in self.data:
            if len(samples) >= self.num_requests:
                break
            prompt = entry["conversations"][0]["value"]
            completion = entry["conversations"][1]["value"]
            multi_modal_data: Optional[MultiModalDataDict] = None

            # Process image input if available.
            if "image" in entry:
                multi_modal_data = {}
                image_path = entry["image"]
                if not isinstance(image_path, str):
                    raise ValueError(
                        "Only support single image input as a string")
                try:
                    multi_modal_data["image"] = Image.open(image_path).convert(
                        "RGB")
                except FileNotFoundError:
                    continue
                prompt = self._get_prompt_for_image_model(prompt)

            lora_request, tokenizer = self.get_random_lora_request()
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            output_len = len(
                completion_ids) if self.output_len is None else self.output_len
            if prompt_len < 4 or output_len < 4 or prompt_len > 1024 or (
                    prompt_len + output_len) > 2048:
                continue
            if return_tuple:
                samples.append(
                    (prompt, prompt_len, output_len, multi_modal_data))
            else:
                samples.append(
                    SampleRequest(prompt=prompt,
                                  prompt_len=prompt_len,
                                  expected_output_len=output_len,
                                  multi_modal_data=multi_modal_data,
                                  lora_request=lora_request))
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.
    Loads poem lines from a text file and generates sample requests.
    """

    def __init__(self, prefix_len: int = 200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix_len = prefix_len or 200
        self.input_len = self.input_len or 550
        self.output_len = self.output_len or 150
        if self.input_len <= self.prefix_len:
            raise ValueError("'input_len' must be greater than 'prefix_len'")
        if self.enable_lora_tokenizer:
            raise NotImplementedError
        if self.data is None:
            self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(self,
               return_tuple: bool = False,
               return_prompt_formatted: bool = False) -> List:
        # Calculate average token length for a poem line.
        tokenized_lines = [
            self.tokenizer(line).input_ids for line in self.data
        ]
        avg_len = sum(len(tokens)
                      for tokens in tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = self.tokenizer.apply_chat_template(
            base_msg, add_generation_prompt=True, tokenize=False)
        base_offset = len(self.tokenizer(base_fmt).input_ids)
        if self.input_len <= base_offset:
            raise ValueError(f"'input_len' must be higher than the \
                base prompt length ({base_offset}).")

        # Determine how many poem lines to use.
        num_input_lines = round((self.input_len - base_offset) / avg_len)
        num_prefix_lines = round((self.prefix_len - base_offset) / avg_len)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        for _ in range(self.num_requests):
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            if return_prompt_formatted:
                prompt = self.tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(self.tokenizer(prompt).input_ids)
            if return_tuple:
                samples.append((prompt, prompt_len, self.output_len, None))
            else:
                samples.append(
                    SampleRequest(prompt=prompt,
                                  prompt_len=prompt_len,
                                  expected_output_len=self.output_len))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.
    Loads data from a CSV file and generates sample 
    requests based on synthetic prompt generation.
    Only rows with Model "GPT-4" and positive response tokens are used.
    """

    def __init__(self, random_seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.random_seed = random_seed
        if self.data is None:
            self.load_data()

    def load_data(self):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        if self.num_requests <= len(gpt4_df):
            gpt4_df = gpt4_df.sample(n=self.num_requests,
                                     random_state=self.random_seed)
        else:
            gpt4_df = gpt4_df.sample(n=self.num_requests,
                                     random_state=self.random_seed,
                                     replace=True)
        # Convert the dataframe to a list of lists.
        self.data = gpt4_df.values.tolist()

    def sample(self, return_tuple: bool = False) -> List:
        samples = []
        for i in range(self.num_requests):
            input_len = int(self.data[i][2])
            output_len = int(self.data[i][3])
            # Use the LoRA-specific tokenizer if enabled.
            lora_req, tokenizer = self.get_random_lora_request()
            # Generate a synthetic prompt:
            # a list of token IDs computed as (i + j) modulo vocab_size.
            token_ids = [(i + j) % tokenizer.vocab_size
                         for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)

            if return_tuple:
                samples.append((prompt, input_len, output_len, None))
            else:
                samples.append(
                    SampleRequest(prompt=prompt,
                                  prompt_len=input_len,
                                  expected_output_len=output_len,
                                  multi_modal_data=None,
                                  lora_request=lora_req))
        return samples
