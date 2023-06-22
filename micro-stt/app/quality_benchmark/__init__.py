"""Run quality benchmark."""

import csv
import inquirer
import torch
import numpy as np
import numpy.typing as npt
from app.config import models
from app.env import IN_PATH, QUALITY_BENCHMARK_PATH, TARGET_SAMPLE_RATE
from app.quality_benchmark.normalization import normalize_transcriptions
from app.quality_benchmark.prettify import prettify_results
from app.quality_benchmark.result_types import full_results, model_results
from app.utils import (
    create_dir_if_not_exists,
    get_audio_duration_ms,
    get_file_paths,
    get_immidiate_sub_dirs
)
from app.wav import get_wav_files, load_tensor_from_wav
from jiwer import wer, mer, wil, wip, cer


def benchmark(
        inputs: list[torch.Tensor],
        sample_rate: int,
        model_names: list[str],
        norm_target_transcriptions: list[str]) -> list[model_results]:
    """Run quality benchmark.

    Calculates:
    - Word Error Rate (WER)
    - Match Error Rate (MER)
    - Word Information Lost (WIL)
    - Word Information Preserved (WIP)
    - Character Error Rate (CER)

    Reference:
    https://github.com/jitsi/jiwer
    """
    results: list[model_results] = []

    for model_name in model_names:
        # Instantiate model
        model = models[model_name]()

        # Transcribe inputs
        print(f'Transcribing using {model_name}...')
        transcriptions = model.transcribe_tensor_batches(inputs, sample_rate)
        print('Calculating metrics...')
        # Normalize transcriptions
        norm_transcriptions = normalize_transcriptions(transcriptions)
        # Calculate metrics
        word_error_rate = np.array(
            [wer(target, trans) for target, trans in zip(norm_target_transcriptions, norm_transcriptions)]
        )
        match_error_rate = np.array(
            [mer(target, trans) for target, trans in zip(norm_target_transcriptions, norm_transcriptions)]
        )
        word_information_lost = np.array(
            [wil(target, trans) for target, trans in zip(norm_target_transcriptions, norm_transcriptions)]
        )
        word_information_preverved = np.array(
            [wip(target, trans) for target, trans in zip(norm_target_transcriptions, norm_transcriptions)]
        )
        character_error_rate = np.array(
            [cer(target, trans) for target, trans in zip(norm_target_transcriptions, norm_transcriptions)]
        )

        def __mean_std(metrics: npt.NDArray[np.float32]) -> tuple[float, float]:
            return (
                float(np.mean(metrics)),
                float(np.std(metrics))
            )

        mean_wer, std_wer = __mean_std(word_error_rate)
        mean_mer, std_mer = __mean_std(match_error_rate)
        mean_wil, std_wil = __mean_std(word_information_lost)
        mean_wip, std_wip = __mean_std(word_information_preverved)
        mean_cer, std_cer = __mean_std(character_error_rate)

        results.append({
            'model_name': model_name,
            'mean_wer': mean_wer,
            'std_wer': std_wer,
            'mean_mer': mean_mer,
            'std_mer': std_mer,
            'mean_wil': mean_wil,
            'std_wil': std_wil,
            'mean_wip': mean_wip,
            'std_wip': std_wip,
            'mean_cer': mean_cer,
            'std_cer': std_cer,
        })

    return results


def main():
    """Run main quality benchmark."""
    input_dirs = get_immidiate_sub_dirs(IN_PATH)
    transcription_filepaths = get_file_paths(IN_PATH, '.csv')

    # Prompt user for audio file, models, and microcontroller definitions and benchmark config
    prompts = [
        inquirer.List(
            'audio_in_dir',
            message='Audio input directory',
            choices=input_dirs
        ),
        inquirer.List(
            'transcriptions',
            message='Select transcriptions',
            choices=transcription_filepaths
        ),
        inquirer.Checkbox(
            'models',
            message='Models',
            choices=models.keys()
        ),
        inquirer.Text(
            'name',
            message='Benchmark name'
        )
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No benchmark config provided')

    # Benchmark configuration
    input_audio_filepaths = get_wav_files(answers['audio_in_dir'])
    waveform_inputs = [load_tensor_from_wav(path, TARGET_SAMPLE_RATE) for path in input_audio_filepaths]
    if len(waveform_inputs) == 0:
        raise Exception('Input audio directory must contain at least one WAV file')

    model_names = answers['models']
    benchmark_name = answers['name']
    results_dirpath = f'{QUALITY_BENCHMARK_PATH}/{benchmark_name}'
    create_dir_if_not_exists(results_dirpath)

    # Load transcriptions from metadata.csv file
    metadata_path = answers['transcriptions']
    target_transcriptions = []
    with open(metadata_path, encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        for row in reader:
            transcription = row[2]  # Use pre-normalized transcriptions in the third column
            target_transcriptions.append(transcription)

    # Normalize transcriptions (fully)
    norm_target_transcriptions = normalize_transcriptions(target_transcriptions)

    # Run benchmark
    inputs = [_[0] for _ in waveform_inputs]
    mdl_results = benchmark(
        inputs,
        TARGET_SAMPLE_RATE,
        model_names,
        norm_target_transcriptions
    )

    # Calculate benchmark information
    num_samples = len(inputs)
    total_audio_durations_ms = get_audio_duration_ms(inputs, TARGET_SAMPLE_RATE)
    mean_audio_duration_ms = total_audio_durations_ms / num_samples

    results: full_results = {
        'num_samples': num_samples,
        'mean_audio_duration_ms': mean_audio_duration_ms,
        'model_results': mdl_results,
    }

    # Write results to disk
    print('Writing results to disk...')
    pretty_results = prettify_results(results)
    results_pretty_filepath = f'{results_dirpath}/results.txt'
    with open(results_pretty_filepath, 'w') as f:
        f.write(pretty_results)

    print("Benchmark completed successfully.")
