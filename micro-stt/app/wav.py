"""Utilities to record, save and load wav files."""

import torch
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from app.utils import get_file_paths, print_countdown
from app.env import CHANNELS


def save_tensor_as_wav(input: torch.Tensor, sample_rate: int, filename: str, path: str) -> None:
    """Save Torch tensor to wav file.

    (Use Scipy wavfile utils since torchaudio can be buggy.)

    References:
    - https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#saving-audio-to-file
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    """
    write_wav(f'{path}/{filename}.wav', sample_rate, input.numpy())


def load_tensor_from_wav(filepath: str, target_sample_rate: int | None) -> tuple[torch.Tensor, int]:
    """Load Torch tensor from wav file.

    Optionally resample audio.

    References:
    - https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data
    - https://github.com/snakers4/silero-models/blob/master/src/silero/utils.py
    """
    tensor, sample_rate = torchaudio.load(filepath)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        tensor = resample(tensor)
        sample_rate = target_sample_rate
    return (tensor[0], sample_rate)


def record_blocking(duration_s: float, sample_rate: int, countdown_s: int | None = 5) -> torch.Tensor:
    """Record audio using sounddevice.
    
    Reference:
    https://python-sounddevice.readthedocs.io/en/0.3.7/#recording
    """
    devices = sd.query_devices()
    print(
        'Available devices:\n',
        devices
    )
    print(
        '\n'
        f'Recording for {duration_s} seconds'
        ' in...' if countdown_s is not None else '...'
    )
    if countdown_s:
        print_countdown(countdown_s)
        print('Started recording...')
    recording = sd.rec(
        int(duration_s * sample_rate),
        sample_rate,
        channels=CHANNELS,
        dtype='float32',
        blocking=True
    )
    print('Recording sucessful')
    return torch.from_numpy(recording.flatten())


def get_wav_files(dirpath: str) -> list[str]:
    """Get filepaths of all wav files in directory."""
    return get_file_paths(dirpath, '.wav')
