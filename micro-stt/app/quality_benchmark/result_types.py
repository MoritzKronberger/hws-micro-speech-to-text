"""Result types for full benchmark.

(Prevents circular imports.)
"""

from typing import TypedDict


class full_results(TypedDict):
    """Full results dict."""
    model_name: str
    word_error_count: int
