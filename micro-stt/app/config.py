"""Global configuration for Micro STT app."""

from app.models.pocket_sphinx import PocketSphinx
from app.models.silero import Silero, SileroQuantized
from app.models.whisper import (
    WhisperSmall,
    WhisperTiny,
    WhisperCPPBase,
    WhisperCPPSmall,
    WhisperCPPTiny,
    WhisperSmallQuantized,
    WhisperTinyQuantized
)
from app.env import NOISE_FLOOR_DURATION_S
from app.preprocessing import load_noise_floor, bandpass_opts, noise_reduce_opts, preprocessing_opts

# Register available transcription models
models = {
    'pocketsphinx': PocketSphinx,
    'silero': Silero,
    'silero-quantized': SileroQuantized,
    'whisper-tiny': WhisperTiny,
    'whisper-small': WhisperSmall,
    'whisper-tiny-quantized': WhisperTinyQuantized,
    'whisper-small-quantized': WhisperSmallQuantized,
    'whisper-cpp-tiny': WhisperCPPTiny,
    'whisper-cpp-small': WhisperCPPSmall,
    'whisper-cpp-base': WhisperCPPBase,
}

# Configure bandpass filter
# Roughly use human vocal range:
# Reference: https://en.wikipedia.org/wiki/Voice_frequency
bandpass_options: bandpass_opts = {
    'low_cutoff_freq': 50,
    'high_cutoff_freq': 3000,
    'Q': 0.707
}

# Configure noise reduction
try:
    noise_floor = load_noise_floor()
    noise_reduction_options: noise_reduce_opts = {
        'stationary': False,
        'y_noise': noise_floor,
        'prop_decrease': 1,
        'time_constant_s': NOISE_FLOOR_DURATION_S,
    }
except Exception:
    noise_reduction_options: noise_reduce_opts = {
        'stationary': True,
        'prop_decrease': 1,
    }

# Configure audio preprocessing
preprocessing_options: preprocessing_opts = {
    'bandpass': bandpass_options,
    'noise_reduce': None,
    'scale': 1
}
