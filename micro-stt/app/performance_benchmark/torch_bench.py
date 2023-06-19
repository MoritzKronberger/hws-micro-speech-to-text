"""PyTorch performance benchmark.

Only works for PyTorch models, since it relies on PyTorch profiler.

Reference:
https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#v1
"""

import gc
import torch
from typing import TypedDict
from torch import profiler
from app.models import IModel


class torch_bench_result(TypedDict):
    """Torch benchmark results dict."""
    cpu_time_ms: float
    self_cpu_time_ms: float
    cpu_memory_usage_byte: int
    flops: int


def benchmark(
        model: IModel,
        inputs: list[torch.Tensor],
        sample_rate: int,
        verbose: bool = False,
        row_limit: int = 5) -> torch_bench_result:
    """Run universal performance benchmark."""
    # Manually run garbage collection
    # (Just to be sure)
    gc.collect()

    ########################
    # Run PyTorch Profiler #
    ########################

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        profile_memory=True,
        with_flops=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        with profiler.record_function('model_inference'):
            _ = model.transcribe_tensor(inputs, sample_rate)

    key_averages = prof.key_averages()
    total_averages = key_averages.total_average()

    if verbose:
        print(
            f'\n PyTorch Profile {model.name}\n'
            f'CPU time top {row_limit}\n'
            f'{key_averages.table(sort_by="cpu_time_total", row_limit=row_limit)}\n'
            f'CPU memory usage top {row_limit}\n'
            f'{key_averages.table(sort_by="cpu_memory_usage", row_limit=row_limit)}\n'
            f'Flops top {row_limit}\n'
            f'{key_averages.table(sort_by="flops", row_limit=row_limit)}\n'
        )

    return {
        'cpu_time_ms': total_averages.cpu_time_total * 0.001,
        'self_cpu_time_ms': total_averages.self_cpu_time_total * 0.001,
        'cpu_memory_usage_byte': total_averages.cpu_memory_usage,
        'flops': total_averages.flops,
    }
