"""Transcription configuration."""

import os
import numpy as np
import numpy.typing as npt
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
NOISE_FLOOR_DURATION_S = float(__from_dotenv('NOISE_FLOOR_DURATION_S'))
CPU_SPEED_GHZ = float(__from_dotenv('CPU_SPEED_GHZ'))

NP_BUFFER = npt.NDArray[np.float32]
OUT_PATH = 'out'
IN_PATH = 'in'
RECORDINGS_PATH = f'{OUT_PATH}/recordings'
BENCHMARK_PATH = f'{OUT_PATH}/benchmarks'
