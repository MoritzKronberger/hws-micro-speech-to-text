"""Result types for foll benchmark.

(Prevents circular imports.)
"""

from typing import TypedDict
from app.performance_benchmark.microcontroller_compatibility import micro_controller_compatibility_results
from app.performance_benchmark.torch_bench import torch_bench_result
from app.performance_benchmark.universal_bench import universal_bench_result


class sys_info(TypedDict):
    """System info dict."""
    machine: str
    system: str
    version: str
    processor: str
    memory_byte: int
    cpu_speed_ghz: float
    cpu_cores: int


class universal_model_results(TypedDict):
    """Universal benchmark results dict."""
    model_name: str
    results: universal_bench_result
    micro_controllers_compats: list[micro_controller_compatibility_results]


class torch_model_results(universal_model_results):
    """Torch benchmark results dict."""
    torch_results: torch_bench_result


class full_results(TypedDict):
    """Full results dict."""
    system_info: sys_info
    audio_duration_ms: float
    max_memory_usage_prop: float
    iterations: int
    model_results: list[universal_model_results | torch_model_results]
