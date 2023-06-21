"""Universal performance benchmark.

Works for all types of models, since it relies on measuring inference time and process memory usage.

Reference:
https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#v1
"""

import gc
import os
import psutil
import torch
from typing import TypedDict
from timeit import default_timer
from app.models import IModel
from app.utils import get_audio_duration_ms


class universal_bench_result(TypedDict):
    """Universal benchmark results dict."""
    memory_rss_byte: float
    inference_time_ms: float
    rtf: float
    audio_duration_ms: float


def benchmark(model: IModel, inputs: list[torch.Tensor], sample_rate: int) -> universal_bench_result:
    """Run universal performance benchmark."""
    # Manually run garbage collection
    # (Just to be sure)
    gc.collect()

    ##########################
    # Benchmark memory usage #
    ##########################

    # Get current process
    # References:
    # - https://psutil.readthedocs.io/en/latest/index.html?highlight=status#process-class
    # - https://docs.python.org/3/library/os.html#os.getpid
    # - https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_info
    process = psutil.Process(os.getpid())
    _ = model.transcribe_tensor_batches(inputs, sample_rate)
    memory_info = process.memory_full_info()
    memory_rss_byte = memory_info.rss

    ############################
    # Benchmark inference time #
    ############################

    # Get inference time
    # References:
    # - https://docs.python.org/3/library/timeit.html#timeit.default_timer
    start = default_timer()
    _ = model.transcribe_tensor_batches(inputs, sample_rate)
    end = default_timer()
    inference_time_ms = (end - start) * 1000

    ##############################
    # Calculate 1 / RTF per core #
    ##############################

    # Calculate realtime factor
    # References:
    # - https://openvoice-tech.net/index.php/Real-time-factor
    # - https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#ce-speed-benchmarks
    audio_duration_ms = get_audio_duration_ms(inputs, sample_rate)
    rtf = inference_time_ms / audio_duration_ms

    return {
        'memory_rss_byte': memory_rss_byte,
        'inference_time_ms': inference_time_ms,
        'rtf': rtf,
        'audio_duration_ms': audio_duration_ms,
    }
