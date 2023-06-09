"""Scale performance benchmark results to micro controller and determine model compatibility.

A model is deemed compatible if:
- At least 1 second of audio can be processed per second per core
- The memory usage does not exceed the micro controller's memory
"""

from typing import TypedDict
from app.performance_benchmark.universal_bench import universal_bench_result
from app.utils import byte_to_mb


class micro_controller(TypedDict):
    """Micro controller dict."""
    name: str
    architecture: str
    memory_mb: float
    cpu_speed_ghz: float
    cpu_cores: int


class micro_controller_compatibility_results(TypedDict):
    """Compatibility list dict."""
    compatible: bool
    micro_controller_info: micro_controller
    audio_duration_ms: float
    memory_rss_byte: float
    inference_time_ms: float
    rtf: float


def micro_controller_compatibility(
        micro_ctr: micro_controller,
        results: universal_bench_result,
        system_cpu_speed_ghz: float,
        system_cpu_cores: int,
        max_memory_usage_prop: float) -> micro_controller_compatibility_results:
    """Scale benchmark results to micro controller and determine model compatibility."""
    # Scale CPU results to micro controller
    cpu_factor = (system_cpu_speed_ghz * system_cpu_cores) / (micro_ctr['cpu_speed_ghz'] * micro_ctr['cpu_cores'])
    inference_time_ms = results['inference_time_ms'] * cpu_factor
    rtf = results['rtf'] * cpu_factor
    memory_usage = byte_to_mb(results['memory_rss_byte']) / micro_ctr['memory_mb']
    # Determine compatibility:
    # - RTF <= 1 -> processing can be done in real time
    # - Memory usage doesn't exceed memory usage limit
    compatible = (
        rtf <= 1
        and
        memory_usage <= max_memory_usage_prop
    )

    return {
        'audio_duration_ms': results['audio_duration_ms'],
        'memory_rss_byte': results['memory_rss_byte'],
        'inference_time_ms': inference_time_ms,
        'rtf': rtf,
        'compatible': compatible,
        'micro_controller_info': micro_ctr,
    }
