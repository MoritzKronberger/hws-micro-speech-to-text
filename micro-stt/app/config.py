"""Transcription configuration."""

import numpy as np
import numpy.typing as npt
from typing import Literal

SAMPLE_RATE = 16_000
BLOCK_SIZE = int(SAMPLE_RATE * 0.01)
CHANNELS = 1
MIN_TRANSCRPTION_INPUT_DURATION_S = 5
NP_BUFFER = npt.NDArray[np.float32]
LANGUAGE = 'en'
MODEL: Literal['silero', 'hubert'] = 'silero'
PASSTHROUGH = True
