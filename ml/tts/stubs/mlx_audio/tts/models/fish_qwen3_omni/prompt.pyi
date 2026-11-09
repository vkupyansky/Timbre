"""Checked prompt API from the installed mlx-audio0.5.3 wheel."""

from typing import Literal

import mlx.core as mx

class TextPart:
    text: str
    def __init__(self, text: str) -> None: ...

class VQPart:
    codes: mx.array
    def __init__(self, codes: mx.array) -> None: ...

class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[TextPart | VQPart]
    modality: Literal["text", "voice", "interleave"] | None
    add_im_start: bool
    add_im_end: bool
    def __init__(
        self,
        *,
        role: Literal["system", "user", "assistant"],
        parts: list[TextPart | VQPart],
        modality: Literal["text", "voice", "interleave"] | None = ...,
        add_im_start: bool = ...,
        add_im_end: bool = ...,
    ) -> None: ...

class Conversation:
    messages: list[Message]
    def __init__(self, messages: list[Message] = ...) -> None: ...
    def append(self, message: Message) -> None: ...
    def encode_for_inference(
        self, tokenizer: object, num_codebooks: int
    ) -> mx.array: ...
