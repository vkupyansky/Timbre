"""Used soundfile 0.14 API; ndarray reads are explicitly float32 mono."""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

class Info(Protocol):
    samplerate: int
    channels: int
    duration: float
    frames: int
    format: str
    subtype: str

def info(file: Path | str) -> Info: ...
def read(
    file: Path | str, *, dtype: str = ..., always_2d: bool = ...
) -> tuple[npt.NDArray[np.float32], int]: ...
def write(
    file: Path | str,
    data: npt.NDArray[np.float32],
    samplerate: int,
    *,
    subtype: str = ...,
    format: str = ...,
) -> None: ...
