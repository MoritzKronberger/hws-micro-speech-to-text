"""Result types for full benchmark.

(Prevents circular imports.)
"""

from typing import TypedDict


class model_results(TypedDict):
    """Model results dict."""
    model_name: str
    mean_wer: float
    std_wer: float
    mean_mer: float
    std_mer: float
    mean_wil: float
    std_wil: float
    mean_wip: float
    std_wip: float
    mean_cer: float
    std_cer: float


class full_results(TypedDict):
    """Full results dict."""
    mean_audio_duration_ms: float
    num_samples: int
    model_results: list[model_results]
