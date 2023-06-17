"""Global configuration for Micro STT app."""

from app.models.silero import Silero
from app.models.hubert import HuBERT
from app.env import NOISE_FLOOR_DURATION_S
from app.preprocessing import load_noise_floor, lowpass_opts, noise_reduce_opts, preprocessing_opts

# Register available transcription models
models = {
    'silero': Silero,
    'hubert': HuBERT,
}

# Configure lowpass filter
lowpass_options: lowpass_opts = {
    'cutoff_freq': 4000,
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
    'lowpass': lowpass_options,
    'noise_reduce': noise_reduction_options,
}
