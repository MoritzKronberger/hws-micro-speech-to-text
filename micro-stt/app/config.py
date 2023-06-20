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
from app.env import (
    BANDPASS_Q,
    ENABLE_BANDPASS,
    ENABLE_NOISE_REDUCE,
    ENABLE_SCALE,
    HIGH_CUTOFF_FREQ,
    LEVEL_SCALE,
    LOW_CUTOFF_FREQ,
    NOISE_FLOOR_DURATION_S,
    NOISE_REDUCE_PROP_DECREASE,
    NOISE_REDUCE_STATIONARY
)
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
    'low_cutoff_freq': LOW_CUTOFF_FREQ,
    'high_cutoff_freq': HIGH_CUTOFF_FREQ,
    'Q': BANDPASS_Q
}

# Configure noise reduction
if NOISE_REDUCE_STATIONARY:
    noise_reduction_options: noise_reduce_opts = {
        'stationary': True,
        'prop_decrease': NOISE_REDUCE_PROP_DECREASE,
    }
else:
    noise_floor = load_noise_floor()
    noise_reduction_options: noise_reduce_opts = {
        'stationary': False,
        'y_noise': noise_floor,
        'prop_decrease': NOISE_REDUCE_PROP_DECREASE,
        'time_constant_s': NOISE_FLOOR_DURATION_S,
    }

# Configure audio preprocessing
preprocessing_options: preprocessing_opts = {
    'bandpass': bandpass_options if ENABLE_BANDPASS else None,
    'noise_reduce': noise_reduction_options if ENABLE_NOISE_REDUCE else None,
    'scale': LEVEL_SCALE if ENABLE_SCALE else None,
}
