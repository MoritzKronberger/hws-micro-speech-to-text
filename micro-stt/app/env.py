"""Transcription configuration."""

import os
import numpy as np
import numpy.typing as npt
from typing import Literal, get_args, cast
from dotenv import load_dotenv

def __from_dotenv(varname: str) -> str:
    """Load .env variable."""
    load_dotenv()
    val = os.getenv(varname)
    if val is None:
        raise Exception(
            f'{varname} not found in .env'
        )
    return val


DEVICE_SAMPLE_RATE = int(__from_dotenv('DEVICE_SAMPLE_RATE'))
TARGET_SAMPLE_RATE = int(__from_dotenv('TARGET_SAMPLE_RATE'))
BLOCK_SIZE = int(__from_dotenv('BLOCK_SIZE'))
CHANNELS = int(__from_dotenv('CHANNELS'))
LANGUAGE = __from_dotenv('LANGUAGE')
models = Literal['silero', 'hubert']
model = __from_dotenv('MODEL')
assert model in get_args(models)
MODEL: Literal['silero', 'hubert'] = cast(models, model)
PASSTHROUGH = __from_dotenv('PASSTHROUGH') == 'True'
NOISE_FLOOR_DURATION_S = float(__from_dotenv('NOISE_FLOOR_DURATION_S'))

NP_BUFFER = npt.NDArray[np.float32]
