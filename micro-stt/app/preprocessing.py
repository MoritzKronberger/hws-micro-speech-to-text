"""Preprocess audio."""

import torch
import numpy as np
import noisereduce as nr
from typing import TypedDict, Literal
from torchaudio.functional import lowpass_biquad
from scipy.io.wavfile import write as write_wav
from app.env import NP_BUFFER
from app.utils import create_dir_if_not_exists


class lowpass_opts(TypedDict):
    """Options for PyTorch `LOWPASS_BIQUAD`.

    Reference:
    https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    """
    cutoff_freq: float
    Q: float


class base_noise_reduce_opts(TypedDict):
    """Base options for noiseredurce with TorchGate.

    Reference:
    https://github.com/timsainb/noisereduce#arguments-to-reduce_noise
    """
    prop_decrease: float


class stationary_noise_reduce_opts(base_noise_reduce_opts):
    """Options for stationary noise reduction with TorchGate.

    Reference:
    https://github.com/timsainb/noisereduce#arguments-to-reduce_noise
    """
    stationary: Literal[True]


class non_stationary_noise_reduce_opts(base_noise_reduce_opts):
    """Options for non stationary noise reduction with TorchGate.

    Reference:
    https://github.com/timsainb/noisereduce#arguments-to-reduce_noise
    """
    stationary: Literal[False]
    y_noise: NP_BUFFER
    time_constant_s: float


noise_reduce_opts = stationary_noise_reduce_opts | non_stationary_noise_reduce_opts


class preprocessing_opts(TypedDict):
    lowpass: lowpass_opts | None
    noise_reduce: noise_reduce_opts | None


def preprocess_tensor(input: torch.Tensor, sample_rate: int, opts: preprocessing_opts) -> torch.Tensor:
    """Preprocess waveform tensor.

    - Apply PyTorch `LOWPASS_BIQUAD`
    - Apply noisereduce with `TorchGate`

    References:
    - https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    - https://github.com/timsainb/noisereduce#simplest-usage
    """
    lowpass = opts['lowpass']
    noise_reduce = opts['noise_reduce']
    out = input

    # Apply noise reduction:
    # Reference:
    # https://github.com/timsainb/noisereduce#simplest-usage-1
    if noise_reduce is not None:
        in_np = input.numpy()
        out_np = nr.reduce_noise(
            y=in_np,
            sr=sample_rate,
            **noise_reduce
        )
        out = torch.from_numpy(out_np)

    # Apply lowpass filter:
    # Reference:
    # https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    if lowpass is not None:
        out = lowpass_biquad(
            waveform=input,
            sample_rate=sample_rate,
            **lowpass
        )

    return out


def save_noise_floor(input: NP_BUFFER, sample_rate: int, temp_dir: str = 'tmp', filename: str = 'noise-floor', save_wav: bool = True) -> None:
    """Save Numpy buffer of noise floor to disk."""
    create_dir_if_not_exists(temp_dir)
    filename = f'{temp_dir}/{filename}'
    np.save(filename, input)
    if save_wav:
        write_wav(f'{filename}.wav', sample_rate, input)



def load_noise_floor(temp_dir: str = 'tmp', filename: str = 'noise-floor') -> NP_BUFFER:
    """Load Numpy buffer of noise floor from disk."""
    return np.load(f'{temp_dir}/{filename}')
