"""Preprocess audio."""

import torch
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
from typing import TypedDict, Literal
from torchaudio.functional import lowpass_biquad, highpass_biquad
from matplotlib.figure import Figure
from app.env import NP_BUFFER
from app.utils import create_dir_if_not_exists


class bandpass_opts(TypedDict):
    """Options for bandpass using PyTorch `LOWPASS_BIQUAD` and `HIGHPASS_BIQUAD`.

    References:
    - https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    - https://pytorch.org/audio/stable/generated/torchaudio.functional.highpass_biquad.html
    """
    high_cutoff_freq: float
    low_cutoff_freq: float
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
    bandpass: bandpass_opts | None
    noise_reduce: noise_reduce_opts | None


def preprocess_tensor(input: torch.Tensor, sample_rate: int, opts: preprocessing_opts) -> torch.Tensor:
    """Preprocess waveform tensor.

    - Apply PyTorch `LOWPASS_BIQUAD`
    - Apply PyTorch `HGHPASS_BIQUAD`
    - Apply noisereduce with `TorchGate`

    References:
    - https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    - https://pytorch.org/audio/main/generated/torchaudio.functional.highpass_biquad.html
    - https://github.com/timsainb/noisereduce#simplest-usage
    """
    bandpass = opts['bandpass']
    noise_reduce = opts['noise_reduce']
    out = input

    # Apply lowpass filter:
    # Reference:
    # https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    if bandpass is not None:
        Q = bandpass['Q']
        # Cut off high frequencies using lowpass filter
        out = lowpass_biquad(
            waveform=out,
            sample_rate=sample_rate,
            cutoff_freq=bandpass['high_cutoff_freq'],
            Q=Q
        )
        # Cut off low frequencies using highpass filter
        out = highpass_biquad(
            waveform=out,
            sample_rate=sample_rate,
            cutoff_freq=bandpass['low_cutoff_freq'],
            Q=Q
        )

    # Apply noise reduction:
    # Reference:
    # https://github.com/timsainb/noisereduce#simplest-usage-1
    if noise_reduce is not None:
        in_np = out.numpy()
        out_np = nr.reduce_noise(
            y=in_np,
            sr=sample_rate,
            **noise_reduce
        )
        out = torch.from_numpy(out_np)

    return out


def __plot_audio(input: NP_BUFFER, sample_rate: int, title: str) -> Figure:
    """Plot spectrogram and waveform.

    Reference:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html
    """
    # Configure plot
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)

    # Plot waveform
    ax1.plot(input)
    ax1.set(
        title='Waveform',
        xlabel='Time [s]'
    )
    ax1.set_xlim(left=0, right=input.shape[0])

    # Plot spectrogram
    ax2.specgram(input, Fs=sample_rate)
    ax2.set(
        title='Spectrogram',
        xlabel='Sample',
        ylabel='Frequency [Hz]'
    )

    fig.tight_layout()

    return fig


def visualize_preprocessing(input: torch.Tensor, sample_rate: int, opts: preprocessing_opts, sub_dirname: str, out_dir: str = 'out') -> None:
    """Visualize preprocessing steps in spectrograms.

    Reference:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html
    """
    bandpass = opts['bandpass']
    noise_reduce = opts['noise_reduce']

    # Create output directory
    viz_dirname = f'{out_dir}/viz'
    job_dirname = f'{viz_dirname}/{sub_dirname}'
    create_dir_if_not_exists(job_dirname)

    np_input = input.numpy()

    # Visualize raw input
    raw_figure = __plot_audio(np_input, sample_rate, 'Raw signal')
    raw_figure.savefig(f'{job_dirname}/raw.svg')

    out = np_input

    # Visualize bandpass filter
    if bandpass is not None:
        out_t = input
        Q = bandpass['Q']
        # Cut off high frequencies using lowpass filter
        out_t = lowpass_biquad(
            waveform=out_t,
            sample_rate=sample_rate,
            cutoff_freq=bandpass['high_cutoff_freq'],
            Q=Q
        )
        # Cut off low frequencies using highpass filter
        out_t = highpass_biquad(
            waveform=out_t,
            sample_rate=sample_rate,
            cutoff_freq=bandpass['low_cutoff_freq'],
            Q=Q
        )
        out = out_t.numpy()
        lowpass_figure = __plot_audio(
            out,
            sample_rate,
            f'Bandpass filter ({bandpass["low_cutoff_freq"]} Hz - {bandpass["high_cutoff_freq"]} Hz)'
        )
        lowpass_figure.savefig(f'{job_dirname}/bandpass.svg')

    # Visualize noise reduction (& lowpass)
    if noise_reduce is not None:
        out = nr.reduce_noise(
            y=out,
            sr=sample_rate,
            **noise_reduce
        )
        noise_reduce_figure = __plot_audio(
            out,
            sample_rate,
            'Noise reduction '
            f'({"stationary" if noise_reduce["stationary"] else "non stationary"}) '
            f'& Bandpass filter ({bandpass["low_cutoff_freq"]} Hz # {bandpass["high_cutoff_freq"]} Hz)' if bandpass is not None else ''
        )
        noise_reduce_figure.savefig(f'{job_dirname}/noise_reduction.svg')


def save_noise_floor(input: NP_BUFFER, temp_dir: str = 'tmp', filename: str = 'noise-floor') -> None:
    """Save Numpy buffer of noise floor to disk."""
    create_dir_if_not_exists(temp_dir)
    filename = f'{temp_dir}/{filename}'
    np.save(filename, input)


def load_noise_floor(temp_dir: str = 'tmp', filename: str = 'noise-floor') -> NP_BUFFER:
    """Load Numpy buffer of noise floor from disk."""
    return np.load(f'{temp_dir}/{filename}.npy')
