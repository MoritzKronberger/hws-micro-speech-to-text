"""Miscellaneous utility functions."""

from pathlib import Path
from time import sleep


def create_dir_if_not_exists(dirpath: str) -> None:
    """Create directory if not exists."""
    Path(dirpath).mkdir(
        parents=True,
        exist_ok=True
    )


def print_hash_comment(content: str):
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
    print(
        '\n'
        f'{hashes}\n'
        f'{content_str}\n'
        f'{hashes}\n'
        '\n'
    )


def print_countdown(seconds: int) -> None:
    """Print countdown seconds (blocking)."""
    for i in reversed(range(seconds)):
        print(i+1)
        sleep(1)
