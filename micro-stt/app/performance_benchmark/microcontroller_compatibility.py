"""Scale performance benchmark results to micro controller and determine model compatibility.

A model is deemed compatible if:
- At least 1 second of audio can be processed per second per core
- The memory usage does not exceed the micro controller's memory
"""

from typing import TypedDict
from app.performance_benchmark.universal_bench import universal_bench_result
from app.utils import byte_to_mb


class micro_controller(TypedDict):
    name: str
    architecture: str
    memory_mb: float
    cpu_speed_ghz: float


class micro_controller_compatibility_results(universal_bench_result):
    compatible: bool


def micro_controller_compatibility(micro_ctr: micro_controller, results: universal_bench_result, system_cpu_speed_ghz: float) -> micro_controller_compatibility_results:
    """Scale benchmark results to micro controller and determine model compatibility."""
    # Scale CPU results to micro controller
    cpu_speed_factor = micro_ctr['cpu_speed_ghz'] / system_cpu_speed_ghz
    inference_time_ms = results['inference_time_ms'] * cpu_speed_factor
    per_core_1_over_rtf = results['per_core_1_over_rtf'] * cpu_speed_factor
    # Determine compatibility
    compatible = (
        per_core_1_over_rtf >= 1
        and
        byte_to_mb(results['memory_rss_byte']) <= micro_ctr['memory_mb']
    )

    return {
        'audio_duration_ms': results['audio_duration_ms'],
        'memory_rss_byte': results['memory_rss_byte'],
        'inference_time_ms': inference_time_ms,
        'per_core_1_over_rtf': per_core_1_over_rtf,
        'compatible': compatible,
    }
