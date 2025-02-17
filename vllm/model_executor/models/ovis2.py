# copied from https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/modeling_ovis.py
# draft ... 

from typing import Iterable, List, Set, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ovis2 import ConversationFormatter, \
    QwenConversationFormatter, OvisConfig, IMAGE_TOKEN_ID

from .utils import (AutoWeightsLoader, maybe_prefix,
                    init_vllm_registered_model, merge_multimodal_embeddings)


class VisualEmbedding(torch.nn.Embedding):
    def forward(self, input: Tensor) -> Tensor:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64,
                           torch.long]:
            return super().forward(input)
        return torch.matmul(input, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class Ovis2Processor:
    def __init__(self, config: OvisConfig, visual_tokenizer: PreTrainedModel,
                 text_tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.config = config
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = text_tokenizer

    def __call__(
            self,
            text: Optional[Union[TextInput, list[TextInput]]] = None,
            images: Optional[Union[ImageInput, list[ImageInput]]] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> None:
        # ?? https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/modeling_ovis.py#L483
        pass


class Ovis2ProcessingInfo(BaseProcessingInfo):
    pass


class Qvis2MultiModalProcessor(BaseMultiModalProcessor[Ovis2ProcessingInfo]):
    pass


class Qvis2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size,
            device=self.visual_tokenizer.device,
            dtype=self.visual_tokenizer.dtype
        )

        def _merge_modules(modules_list: tuple):
            merged_modules = []
            for modules in modules_list:
                merged_modules.extend(modules if modules else [])
            return merged_modules

        self._no_split_modules = _merge_modules(
            (self.llm._no_split_modules, self.visual_tokenizer._no_split_modules))
        self._skip_keys_device_placement = self.llm._skip_keys_device_placement
        self._keep_in_fp32_modules = _merge_modules(
            (self.llm._keep_in_fp32_modules,
             self.visual_tokenizer._keep_in_fp32_modules))
        self.is_parallelizable = all(
            (self.llm.is_parallelizable, self.visual_tokenizer.is_parallelizable))
        self.supports_gradient_checkpointing = True
        self._supports_flash_attn_2 = True

    def _validate_pixel_values(self,
                               pixel_values: Union[torch.Tensor, List[torch.Tensor]]) -> \
    Union[torch.Tensor, List[torch.Tensor]]:
        h = w = self.config.visual_tokenizer.backbone_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(p: torch.Tensor):
            actual_dims = tuple(p.shape[1:])
            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(p.shape)}.")

        for p in pixel_values:
            _validate_shape(p)

        return pixel_values

    def tie_weights(self):
        if not self.config.disable_tie_weight:
            self.get_llm().tie_weights()

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def get_conversation_formatter(self) -> ConversationFormatter:
        if getattr(self, 'conversation_formatter', None) is None:
            self.conversation_formatter = getattr(QwenConversationFormatter,
                                                  self.config.conversation_formatter_class)(
                self.text_tokenizer)
        return self.conversation_formatter

    def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[NestedTensors] = None
    ) -> torch.Tensor:
        input_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            input_embeds = merge_multimodal_embeddings(
                input_ids, input_embeds, multimodal_embeddings,
                IMAGE_TOKEN_ID)
        return input_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):

        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.llm.model(input_ids,
                                       positions,
                                       kv_caches,
                                       attn_metadata,
                                       intermediate_tensors,
                                       inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.llm.compute_logits(hidden_states, sampling_metadata)

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.llm.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights) 
