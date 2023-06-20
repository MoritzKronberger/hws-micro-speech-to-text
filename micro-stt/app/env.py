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


# Audio recording
DEVICE_SAMPLE_RATE = int(__from_dotenv('DEVICE_SAMPLE_RATE'))
TARGET_SAMPLE_RATE = int(__from_dotenv('TARGET_SAMPLE_RATE'))
BLOCK_SIZE = int(__from_dotenv('BLOCK_SIZE'))
CHANNELS = int(__from_dotenv('CHANNELS'))

# Model configuration
LANGUAGE = __from_dotenv('LANGUAGE')

# Audio preprocessing
# Bandpass filter
LOW_CUTOFF_FREQ = int(__from_dotenv('LOW_CUTOFF_FREQ'))
HIGH_CUTOFF_FREQ = int(__from_dotenv('HIGH_CUTOFF_FREQ'))
BANDPASS_Q = float(__from_dotenv('BANDPASS_Q'))
# Noise reduction
NOISE_REDUCE_STATIONARY = __from_dotenv('NOISE_REDUCE_STATIONARY') == 'True'
NOISE_REDUCE_PROP_DECREASE = float(__from_dotenv('NOISE_REDUCE_PROP_DECREASE'))
NOISE_FLOOR_DURATION_S = float(__from_dotenv('NOISE_FLOOR_DURATION_S'))
# Scaling
LEVEL_SCALE = float(__from_dotenv('LEVEL_SCALE'))
# Overall configuration
ENABLE_BANDPASS = __from_dotenv('ENABLE_BANDPASS') == 'True'
ENABLE_NOISE_REDUCE = __from_dotenv('ENABLE_NOISE_REDUCE') == 'True'
ENABLE_SCALE = __from_dotenv('ENABLE_SCALE') == 'True'

# Performance benchmark configuration
CPU_SPEED_GHZ = float(__from_dotenv('CPU_SPEED_GHZ'))

# Helper constants
NP_BUFFER = npt.NDArray[np.float32]
OUT_PATH = 'out'
IN_PATH = 'in'
RECORDINGS_PATH = f'{OUT_PATH}/recordings'
BENCHMARK_PATH = f'{OUT_PATH}/benchmarks'
