#!/usr/bin/env python3 SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""
benchmark_dataset.py

This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""

import base64
import io
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import ImageItem
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------

VISION_ARENA_DATASET_PATH = "lmarena-ai/vision-arena-bench-v0.1"


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(
    image: Any,
    return_multi_modal_data_dict: bool = True
) -> Union[MultiModalDataDict, Mapping[str, Any]]:
    """
    Process an single image input and return a multimedia content dictionary.

    Modes:
      1. MultiModal Data Dictionary (return_multi_modal_data_dict=True): - If
         the input is an ImageItem, it is assigned directly under the "image"
         key.  - If the input is a string (file path), the image is opened with
         PIL and converted to RGB.  - Raises a ValueError if the input type is
         invalid - Return empty data if the file is not found.

      2. Base64-Encoded Image URL (return_multi_modal_data_dict=False): - If the
         input is a PIL.Image.Image, it is converted to RGB, saved as a JPEG
         in-memory,
           encoded in base64, and returned as a data URL.
         - If the input is a string, it is treated as a URL or file path. If it
           lacks a valid prefix, "file://" is prepended.
         - Raises a ValueError if the input type is invalid.
    """
    if return_multi_modal_data_dict:
        mm_data: MultiModalDataDict = {}  # ensure type hinting
        if isinstance(image, ImageItem):
            mm_data["image"] = image
        elif isinstance(image, str):
            try:
                mm_data["image"] = Image.open(image).convert("RGB")
            except FileNotFoundError:
                print(f"Image not found: {image}")
                return mm_data
        else:
            # TODO(vllm-project/vllm/issues/9778): Support multiple images.
            raise ValueError(
                f"Invalid image input {image}. Must be an ImageItem or str.")
        return mm_data

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image or str.")


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_NUM_REQUESTS = 1000
    DEFAULT_SEED = 0

    # num_requests has default 1000 in both the benchmark_serving.py and
    # benchmark_throughput.py

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        enable_lora_tokenizer: bool = False,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        num_requests: int = DEFAULT_NUM_REQUESTS,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        dataset_path: Optional[str] = None,
        model: Optional[str] = None,
        data: Optional[list] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.tokenizer = tokenizer
        self.data = data  # For datasets that require pre-loading
        self.dataset_path = dataset_path
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)

        # LoRA related
        self.enable_lora_tokenizer = enable_lora_tokenizer
        self.lora_path = lora_path
        self.max_loras = max_loras

        self.num_requests = (num_requests if num_requests is not None else
                             self.DEFAULT_NUM_REQUESTS)
        self.input_len = input_len
        self.output_len = output_len
        if self.num_requests is None:
            raise ValueError("num_requests must be provided for sampling.")

        self.model = model

        if self.enable_lora_tokenizer and not self.lora_path:
            raise ValueError("LoRA is enabled but no lora_path provided.")

    def _get_prompt_for_image_model(self, text_prompt: str) -> str:
        """
        Prepend and append special tokens around the text prompt.
        """
        model = self.model.lower() if self.model else ""
        if "pixtral" in model:
            return f"<s>[INST]{text_prompt}\n[IMG][/INST]"
        raise ValueError(f"Unsupported model {model}")

    def load_data(self) -> None:
        """
        Load data from the specified dataset path.  RandomDataset does not need
        to implement this method.
        """
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self, ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Return a tuple (lora_request, tokenizer) for tokenizing requests.  If
        LoRA is enabled, returns the LoRA-specific tokenizer; otherwise, the
        base tokenizer.
        """
        if not self.enable_lora_tokenizer:
            return None, self.tokenizer

        if self.max_loras is None:
            raise ValueError(
                "max_lora must be set when enabling LoRA tokenizer.")

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, self.max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(self.lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        return lora_request, lora_tokenizer_cache[lora_id]

    def create_sample(
        self,
        prompt: str,
        prompt_len: int,
        output_len: int,
        mm_content: Optional[MultiModalDataDict] = None,
        lora_request: Optional[LoRARequest] = None,
        for_online_benchmark: bool = False,
    ) -> Union[
            SampleRequest,
            tuple[str, int, int, Optional[MultiModalDataDict]],
    ]:
        """
        Helper to build a sample in either tuple or SampleRequest format.
        """
        if for_online_benchmark:
            return (prompt, prompt_len, output_len, mm_content)
        return SampleRequest(
            prompt=prompt,
            prompt_len=prompt_len,
            expected_output_len=output_len,
            multi_modal_data=mm_content,
            lora_request=lora_request,
        )

    @abstractmethod
    def sample(
        self,
        for_online_benchmark: bool = False
    ) -> list[Union[
            SampleRequest,
            tuple[str, int, int, Optional[MultiModalDataDict]],
    ]]:
        """
        Generate sample requests from the dataset.
        """
        pass


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 1.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prefix_len = (prefix_len if prefix_len is not None else
                           self.DEFAULT_PREFIX_LEN)
        self.range_ratio = (range_ratio if range_ratio is not None else
                            self.DEFAULT_RANGE_RATIO)
        self.input_len = (input_len
                          if input_len is not None else self.DEFAULT_INPUT_LEN)
        self.output_len = (output_len if output_len is not None else
                           self.DEFAULT_OUTPUT_LEN)

    def sample(self, for_online_benchmark: bool = False) -> list:
        vocab_size = self.tokenizer.vocab_size
        prefix_token_ids = (np.random.randint(
            0, vocab_size, size=self.prefix_len).tolist()
                            if self.prefix_len > 0 else [])

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
                self.create_sample(
                    prompt,
                    total_input_len,
                    int(output_lens[i]),
                    mm_content=None,
                    lora_request=None,
                    for_online_benchmark=for_online_benchmark,
                ))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.data is None:
            self.load_data()

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

    def sample(self, for_online_benchmark: bool = False) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= self.num_requests:
                break
            prompt = entry["conversations"][0]["value"]
            completion = entry["conversations"][1]["value"]
            mm_content: Optional[MultiModalDataDict] = None

            # Process image input if available.
            if "image" in entry:
                mm_content = process_image(entry["image"])
                if not mm_content:
                    continue
                prompt = self._get_prompt_for_image_model(prompt)

            lora_request, tokenizer = self.get_random_lora_request()
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            output_len = (len(completion_ids)
                          if self.output_len is None else self.output_len)
            if not is_valid_sequence(
                    prompt_len,
                    output_len,
                    skip_min_output_len_check=self.output_len is not None
                    and self.output_len > 0,
            ):
                continue
            samples.append(
                self.create_sample(
                    prompt,
                    prompt_len,
                    output_len,
                    mm_content,
                    lora_request,
                    for_online_benchmark,
                ))
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prefix_len = (self.DEFAULT_PREFIX_LEN
                           if prefix_len is None else prefix_len)
        self.input_len = (self.DEFAULT_INPUT_LEN
                          if input_len is None else input_len)
        self.output_len = (self.DEFAULT_OUTPUT_LEN
                           if output_len is None else output_len)
        if self.enable_lora_tokenizer:
            raise NotImplementedError("LoRA is not supported in SonnetDataset")
        if self.data is None:
            self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        for_online_benchmark: bool = False,
        return_prompt_formatted: bool = False,
    ) -> list:
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
        print(base_offset, self.input_len)
        if self.input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset}).")

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
            prompt_formatted = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(self.tokenizer(prompt_formatted).input_ids)
            samples.append(
                self.create_sample(
                    prompt_formatted if return_prompt_formatted else prompt,
                    prompt_len,
                    self.output_len,
                    mm_content=None,
                    lora_request=None,
                    for_online_benchmark=for_online_benchmark,
                ))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
            gpt4_df = gpt4_df.sample(
                n=self.num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        self.data = gpt4_df.values.tolist()

    def sample(self, for_online_benchmark: bool = False) -> list:
        samples = []
        for i in range(self.num_requests):
            input_len = int(self.data[i][2])
            output_len = int(self.data[i][3])
            lora_req, tokenizer = self.get_random_lora_request()
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                self.create_sample(
                    prompt,
                    input_len,
                    output_len,
                    mm_content=None,
                    lora_request=lora_req,
                    for_online_benchmark=for_online_benchmark,
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Implementation
# -----------------------------------------------------------------------------


class HuggingFaceDataset(BenchmarkDataset):
    """
    Dataset class for processing a HuggingFace dataset with conversation data
    and optional images.
    """

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset

        if self.data is None:
            self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided for loading data.")

        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )

        if "conversations" not in self.data.features:
            raise ValueError("HF Dataset must have a 'conversations' column.")

        # Shuffle and filter examples with at least 2 conversations.
        self.data = self.data.shuffle(seed=self.random_seed).filter(
            lambda x: len(x["conversations"]) >= 2)

    def sample(self, for_online_benchmark: bool = False) -> list:
        sampled_requests = []
        dynamic_output = self.output_len is None

        for item in self.data:
            if len(sampled_requests) >= self.num_requests:
                break

            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            lora_request, tokenizer = self.get_random_lora_request()

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else self.output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                    prompt_len, completion_len):
                continue

            mm_content = process_image(
                item["image"],
                return_multi_modal_data_dict=not for_online_benchmark,
            ) if "image" in item else None
            sampled_requests.append(
                self.create_sample(
                    prompt,
                    prompt_len,
                    output_len,
                    mm_content,
                    lora_request=lora_request,
                    for_online_benchmark=for_online_benchmark,
                ))
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(BenchmarkDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.output_len = (output_len if output_len is not None else
                           self.DEFAULT_OUTPUT_LEN)

        if self.dataset_path != VISION_ARENA_DATASET_PATH:
            raise ValueError(f"Only support Vision Arena dataset.\
                    This data path {self.dataset_path} is not valid.")
        if self.dataset_subset is None and self.dataset_split != "train":
            raise ValueError("Dataset split must be 'train'.")

        if self.data is None:
            self.load_data()

        if self.enable_lora_tokenizer:
            raise NotImplementedError(
                "LoRA is not supported in VisionArenaDataset")

    def load_data(self) -> None:
        dataset = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = dataset.shuffle(seed=self.random_seed)

    def sample(self, for_online_benchmark: bool = False) -> list:
        # TODO (jenniferzhao): Add support for offline benchmark sampling
        assert for_online_benchmark, (
            "VisionArenaDataset only support online benchmark sampling "
            "for now")
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= self.num_requests:
                break
            prompt = item["turns"][0][0]["content"]
            prompt_len = len(self.tokenizer(prompt).input_ids)
            mm_content = process_image(
                item["images"][0],
                return_multi_modal_data_dict=not for_online_benchmark,
            )
            sampled_requests.append(
                self.create_sample(
                    prompt,
                    prompt_len,
                    self.output_len,
                    mm_content,
                    for_online_benchmark=for_online_benchmark,
                ))
        return sampled_requests
