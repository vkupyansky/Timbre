"""Narrow Transformers 5.16 signatures used by local reference transcription."""

from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from torch import Tensor, nn

class ProcessorOutput(Protocol):
    input_features: Tensor
    attention_mask: Tensor

class WhisperProcessor:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Path,
        *,
        local_files_only: bool,
        clean_up_tokenization_spaces: bool,
    ) -> WhisperProcessor: ...
    def __call__(
        self,
        audio: npt.NDArray[np.float32],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> ProcessorOutput: ...
    def batch_decode(
        self,
        sequences: Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...

class WhisperGenerationConfig:
    language: str
    task: str
    return_dict_in_generate: bool
    output_scores: bool

class WhisperForConditionalGeneration(nn.Module):
    generation_config: WhisperGenerationConfig
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Path,
        *,
        local_files_only: bool,
    ) -> WhisperForConditionalGeneration: ...
    def detect_language(self, *, input_features: Tensor) -> Tensor: ...
    def generate(
        self,
        input_features: Tensor,
        *,
        attention_mask: Tensor,
        generation_config: WhisperGenerationConfig,
        return_dict_in_generate: Literal[True],
    ) -> WhisperGenerationOutput: ...
    def compute_transition_scores(
        self, sequences: Tensor, scores: tuple[Tensor, ...], *, normalize_logits: bool
    ) -> Tensor: ...

class WhisperGenerationOutput(Protocol):
    sequences: Tensor
    scores: tuple[Tensor, ...]
