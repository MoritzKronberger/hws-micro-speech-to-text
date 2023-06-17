"""Preprocess audio."""

import torch
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
from typing import TypedDict, Literal
from uuid import uuid4
from torchaudio.functional import lowpass_biquad
from scipy.io.wavfile import write as write_wav
from matplotlib.figure import Figure
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

    # Apply lowpass filter:
    # Reference:
    # https://pytorch.org/audio/main/generated/torchaudio.functional.lowpass_biquad.html
    if lowpass is not None:
        out = lowpass_biquad(
            waveform=input,
            sample_rate=sample_rate,
            **lowpass
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


def visualize_preprocessing(input: torch.Tensor, sample_rate: int, opts: preprocessing_opts, out_dir: str = 'out') -> None:
    """Visualize preprocessing steps in spectrograms.

    Reference:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html
    """
    lowpass = opts['lowpass']
    noise_reduce = opts['noise_reduce']

    # Create output directory
    viz_dirname = f'{out_dir}/viz'
    job_dirname = f'{viz_dirname}/{uuid4()}'
    create_dir_if_not_exists(job_dirname)

    np_input = input.numpy()

    # Visualize raw input
    raw_figure = __plot_audio(np_input, sample_rate, 'Raw signal')
    raw_figure.savefig(f'{job_dirname}/raw.svg')

    out = np_input

    # Visualize lowpass filter
    if lowpass is not None:
        out_t = lowpass_biquad(
            waveform=input,
            sample_rate=sample_rate,
            **lowpass
        )
        out = out_t.numpy()
        lowpass_figure = __plot_audio(
            out,
            sample_rate,
            f'Lowpass filter ({lowpass["cutoff_freq"]} Hz)'
        )
        lowpass_figure.savefig(f'{job_dirname}/lowpass.svg')

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
            f'& Lowpass filter ({lowpass["cutoff_freq"]} Hz)' if lowpass is not None else ''
        )
        noise_reduce_figure.savefig(f'{job_dirname}/noise_reduction.svg')


def save_noise_floor(input: NP_BUFFER, sample_rate: int, temp_dir: str = 'tmp', filename: str = 'noise-floor', save_wav: bool = True) -> None:
    """Save Numpy buffer of noise floor to disk."""
    create_dir_if_not_exists(temp_dir)
    filename = f'{temp_dir}/{filename}'
    np.save(filename, input)
    if save_wav:
        write_wav(f'{filename}.wav', sample_rate, input)


def load_noise_floor(temp_dir: str = 'tmp', filename: str = 'noise-floor') -> NP_BUFFER:
    """Load Numpy buffer of noise floor from disk."""
    return np.load(f'{temp_dir}/{filename}.npy')
