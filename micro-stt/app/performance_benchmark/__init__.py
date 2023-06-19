"""Run performance benchmark."""

import gc
import json
import inquirer
import psutil
import platform
import torch
from app.config import models
from app.env import BENCHMARK_PATH, CPU_SPEED_GHZ, IN_PATH, TARGET_SAMPLE_RATE
from app.performance_benchmark.microcontroller_compatibility import micro_controller, micro_controller_compatibility
from app.performance_benchmark.prettify import prettify_results
from app.performance_benchmark.result_types import full_results, sys_info, torch_model_results, universal_model_results
from app.performance_benchmark.torch_bench import benchmark as torch_benchmark
from app.performance_benchmark.universal_bench import benchmark as universal_benchmark
from app.utils import create_dir_if_not_exists, get_audio_duration_ms_flexible_length, get_file_paths, get_immidiate_sub_dirs
from app.wav import get_wav_files, load_tensor_from_wav


# Use different batch sizes
# Reference:
# https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#v1
BATCH_SIZES = [1, 5, 10, 25]


def benchmark(
        inputs: list[torch.Tensor],
        sample_rate: int,
        model_names: list[str],
        micro_controllers: list[micro_controller],
        system_cpu_speed_ghz: float) -> full_results:
    """Run performance benchmark."""
    model_results: list[universal_model_results | torch_model_results] = []

    for model_name in model_names:
        # Instantiate model
        model = models[model_name]()

        # Run universal benchmark
        universal_results = universal_benchmark(model, inputs, sample_rate)
        print("uni: " + universal_results)
        # Run torch benchmark for torch models
        if model.is_pytorch:
            torch_results = torch_benchmark(model, inputs, sample_rate)
        else:
            torch_results = None

        # Calculate micro controller compatibilities
        micro_controller_compats = [
            micro_controller_compatibility(micro_ctr, universal_results, system_cpu_speed_ghz)
            for micro_ctr in micro_controllers
        ]

        # DELETE MODEL TO FREE MEMORY
        model_name = model.name
        del model
        # Manually run garbage collection
        # (Just in case)
        gc.collect()

        # Append results
        if torch_results is not None:
            model_results.append(
                {
                    'model_name': model_name,
                    'results': universal_results,
                    'torch_results': torch_results,
                    'micro_controllers_compats': micro_controller_compats,
                }
            )
        else:
            model_results.append(
                {
                    'model_name': model_name,
                    'results': universal_results,
                    'micro_controllers_compats': micro_controller_compats,
                }
            )

    # Get system info
    system_info: sys_info = {
        'machine': platform.machine(),
        'system': platform.system(),
        'version': platform.version(),
        'processor': platform.processor(),
        'memory_byte': psutil.virtual_memory().total,
        'cpu_speed_ghz': system_cpu_speed_ghz,
    }

    # Calculate audio duration
    audio_duration_ms = get_audio_duration_ms_flexible_length(inputs, sample_rate)


    return {
        'system_info': system_info,
        'audio_duration_ms': audio_duration_ms,
        'model_results': model_results,
    }


def main():
    """Run main performance benchmark."""
    input_dirs = get_immidiate_sub_dirs(IN_PATH)
    micro_controller_filepaths = get_file_paths(IN_PATH, '.json')

    # Prompt user for audio file, models and micro controller definitions and benchmark config
    prompts = [
        inquirer.List(
            'audio_in_dir',
            message='Audio input directory',
            choices=input_dirs
        ),
        inquirer.Checkbox(
            'models',
            message='Models',
            choices=models.keys()
        ),
        inquirer.List(
            'micro_controller',
            message='Micro controllers',
            choices=micro_controller_filepaths
        ),
        inquirer.Text(
            'name',
            message='Benchmark name'
        )
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No models or micro controllers selected')

    # Benchmark configuration
    input_audio_filepaths = get_wav_files(answers['audio_in_dir'])
    waveform_inputs = [load_tensor_from_wav(path, TARGET_SAMPLE_RATE) for path in input_audio_filepaths]

    if len(waveform_inputs) == 0:
        raise Exception('Input audio directory must contain at least one wav file')

    batches = [
        waveform_inputs[:size] for size in BATCH_SIZES if len(waveform_inputs) >= size
    ]

    model_names = answers['models']
    with open(answers['micro_controller'], 'r') as f:
        micro_controllers: list[micro_controller] = json.load(f)
    benchmark_name = answers['name']

    results_dirpath = f'{BENCHMARK_PATH}/{benchmark_name}'
    create_dir_if_not_exists(results_dirpath)

    # Run benchmark for batches and write results to disk
    for batch in batches:
        inputs = [_[0] for _ in batch]
        results = benchmark(
            inputs,
            TARGET_SAMPLE_RATE,
            model_names,
            micro_controllers,
            CPU_SPEED_GHZ
        )
        pretty_results = prettify_results(results)
        results_json_filepath = f'{results_dirpath}/results_batch_{len(batch)}.json'
        with open(results_json_filepath, 'w') as f:
            json.dump(results, f)
        results_pretty_filepath = f'{results_dirpath}/results_batch_{len(batch)}.txt'
        with open(results_pretty_filepath, 'w') as f:
            f.write(pretty_results)
