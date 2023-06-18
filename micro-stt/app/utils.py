"""Miscellaneous utility functions."""

import os
import torch
from pathlib import Path
from time import sleep
from glob import glob


def create_dir_if_not_exists(dirpath: str) -> None:
    """Create directory if not exists."""
    Path(dirpath).mkdir(
        parents=True,
        exist_ok=True
    )


def get_hash_comment(content: str) -> str:
    """Print hash comment.
    
    I.e.:
    ```
    ###########
    # content #
    ###########
    ```
    """
    content_str = f'# {content} #'
    num_hashes = len(content_str)
    hashes = '#' * num_hashes
    return (
        f'{hashes}\n'
        f'{content_str}\n'
        f'{hashes}\n'
    )


def print_countdown(seconds: int) -> None:
    """Print countdown seconds (blocking)."""
    for i in reversed(range(seconds)):
        print(i+1)
        sleep(1)


def get_immidiate_sub_dirs(dirpath: str) -> list[str]:
    """Get all immediate subdirectories.
    
    Reference:
    https://stackoverflow.com/a/40347279/14906871
    """
    return [f.path for f in os.scandir(dirpath) if f.is_dir()]


def get_file_paths(dirpath: str, file_ending: str) -> list[str]:
    """Get filepaths of files with specified ending in specified directory."""
    return [f'{dirpath}/{filename}' for filename in glob(f'*{file_ending}', root_dir=dirpath)]


def byte_to_mb(byte: float) -> float:
    "Convert byte to MB."
    return byte / (1024 ** 2)


def get_audio_duration_ms(inputs: list[torch.Tensor], sample_rate: int) -> float:
    """Calculate audio duration in ms."""
    waveform = torch.stack(inputs).flatten()
    num_samples = waveform.size(dim=0)
    return (num_samples / sample_rate) * 1000
