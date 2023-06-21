"""Universal performance benchmark.

Works for all types of models, since it relies on measuring inference time and process memory usage.

Reference:
https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#v1
"""

import gc
import os
import psutil
import torch
import numpy as np
from typing import TypedDict
from timeit import default_timer
from app.models import IModel
from app.utils import get_audio_duration_ms


class universal_bench_result(TypedDict):
    """Universal benchmark results dict."""
    memory_rss_byte: float
    std_memory_rss_byte: float
    inference_time_ms: float
    std_inference_time_ms: float
    rtf: float
    rtf_at_1ghz: float
    audio_duration_ms: float


def benchmark(
        model: IModel,
        inputs: list[torch.Tensor],
        sample_rate: int,
        system_cpu_speed_ghz: float,
        iterations: int) -> universal_bench_result:
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
    memory_rss_byte_its: list[float] = []
    for _ in range(iterations):
        process = psutil.Process(os.getpid())
        _ = model.transcribe_tensor_batches(inputs, sample_rate)
        memory_info = process.memory_full_info()
        memory_rss_byte_its.append(memory_info.rss)
        gc.collect()  # (Just to be sure)
    memory_rss_byte = float(np.mean(np.array(memory_rss_byte_its)))
    std_memory_rss_byte = float(np.std(np.array(memory_rss_byte_its)))

    ############################
    # Benchmark inference time #
    ############################

    # Get inference time
    # References:
    # - https://docs.python.org/3/library/timeit.html#timeit.default_timer
    inference_time_ms_its: list[float] = []
    for _ in range(iterations):
        start = default_timer()
        _ = model.transcribe_tensor_batches(inputs, sample_rate)
        end = default_timer()
        inference_time_ms_its.append((end - start) * 1000)
    print(inference_time_ms_its)
    inference_time_ms = float(np.mean(np.array(inference_time_ms_its)))
    std_inference_time_ms = float(np.std(np.array(inference_time_ms_its)))

    ##############################
    # Calculate 1 / RTF per core #
    ##############################

    # Calculate realtime factor
    # References:
    # - https://openvoice-tech.net/index.php/Real-time-factor
    # - https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#ce-speed-benchmarks
    audio_duration_ms = get_audio_duration_ms(inputs, sample_rate)
    rtf = inference_time_ms / audio_duration_ms
    rtf_at_1ghz = rtf * system_cpu_speed_ghz

    return {
        'memory_rss_byte': memory_rss_byte,
        'std_memory_rss_byte': std_memory_rss_byte,
        'inference_time_ms': inference_time_ms,
        'std_inference_time_ms': std_inference_time_ms,
        'rtf': rtf,
        'rtf_at_1ghz': rtf_at_1ghz,
        'audio_duration_ms': audio_duration_ms,
    }
