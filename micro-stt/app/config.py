"""Global configuration for Micro STT app."""

from app.models.pocket_sphinx import PocketSphinx
from app.models.silero import Silero, SileroQuantized, SileroGerman
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
    NOISE_REDUCE_N_STD_THRESH_STATIONARY,
    NOISE_REDUCE_NFFT,
    NOISE_REDUCE_WIN_LENGTH,
    NOISE_TIME_CONSTANT_S,
    NOISE_REDUCE_PROP_DECREASE,
    NOISE_REDUCE_STATIONARY
)
from app.preprocessing import (
    load_noise_floor,
    bandpass_opts,
    base_noise_reduce_opts,
    noise_reduce_opts,
    preprocessing_opts
)

# Register available transcription models
models = {
    'pocketsphinx': PocketSphinx,
    'silero': Silero,
    'silero-quantized': SileroQuantized,
    'silero-german': SileroGerman,
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
base_nr_opts: base_noise_reduce_opts = {
    'prop_decrease': NOISE_REDUCE_PROP_DECREASE,
    'n_fft': NOISE_REDUCE_NFFT,
    'win_length': NOISE_REDUCE_WIN_LENGTH,
    'n_std_thresh_stationary': NOISE_REDUCE_N_STD_THRESH_STATIONARY,
}
if NOISE_REDUCE_STATIONARY:
    noise_reduction_options: noise_reduce_opts = {
        **base_nr_opts,
        'stationary': True,
    }
else:
    noise_floor = load_noise_floor()
    noise_reduction_options: noise_reduce_opts = {
        **base_nr_opts,
        'stationary': False,
        'y_noise': noise_floor,
        'time_constant_s': NOISE_TIME_CONSTANT_S,
    }

# Configure audio preprocessing
preprocessing_options: preprocessing_opts = {
    'bandpass': bandpass_options if ENABLE_BANDPASS else None,
    'noise_reduce': noise_reduction_options if ENABLE_NOISE_REDUCE else None,
    'scale': LEVEL_SCALE if ENABLE_SCALE else None,
}
