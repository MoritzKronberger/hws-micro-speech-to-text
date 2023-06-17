"""Global configuration for Micro STT app."""

from app.models.silero import Silero
from app.models.hubert import HuBERT
from app.env import NOISE_FLOOR_DURATION_S
from app.preprocessing import load_noise_floor, bandpass_opts, noise_reduce_opts, preprocessing_opts

# Register available transcription models
models = {
    'silero': Silero,
    'hubert': HuBERT,
}

# Configure bandpass filter
# Roughly use human vocal range:
# Reference: https://en.wikipedia.org/wiki/Voice_frequency
bandpass_options: bandpass_opts = {
    'low_cutoff_freq': 50,
    'high_cutoff_freq': 300,
    'Q': 0.85
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
    'noise_reduce': noise_reduction_options,
    'scale':75
}
