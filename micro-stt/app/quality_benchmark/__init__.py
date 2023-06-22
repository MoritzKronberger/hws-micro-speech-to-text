"""Run quality benchmark."""

import csv
import inquirer
import torch
from app.config import models
from app.env import BENCHMARK_PATH, IN_PATH, TARGET_SAMPLE_RATE
from app.quality_benchmark.prettify import prettify_results
from app.quality_benchmark.result_types import full_results
from app.utils import (
    create_dir_if_not_exists,
    get_file_paths,
    get_immidiate_sub_dirs,
    get_audio_duration_ms_flexible_length
)
from app.wav import get_wav_files, load_tensor_from_wav
from jiwer import wer

# Use different batch sizes
# Reference:
# https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#v1
BATCH_SIZES = [1, 5, 10, 25]


def benchmark(
        inputs: list[torch.Tensor],
        sample_rate: int,
        model_names: list[str],
        target_transcripts: list[str]) -> list[full_results]:
    """Run quality benchmark."""
    model_results: list[full_results] = []

    for model_name in model_names:
        # Instantiate model
        model = models[model_name]()

        # Transcribe inputs
        transcriptions = model.transcribe_tensor_batches(inputs, sample_rate)
        # Calculate word error count
        word_error_count = [wer(target, trans) for target, trans in zip(target_transcripts, transcriptions)]
        # Calculate audio duration
        audio_duration_ms = get_audio_duration_ms_flexible_length(inputs, sample_rate)

        model_results.append(
            {
                'model_name': model_name,
                'word_error_count': word_error_count,
                'audio_durations': audio_duration_ms,
                'transcriptions': {
                    'reference': target_transcripts,
                    'transcription': transcriptions
                }
            }
        )

    return model_results


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
        raise Exception('No models or microcontrollers selected')

    # Benchmark configuration
    input_audio_filepaths = get_wav_files(answers['audio_in_dir'])
    input_audio_filepaths = [path.replace('/', '\\') for path in input_audio_filepaths]  # Update this line
    # print("Input audio file paths:", input_audio_filepaths)  # Add this line
    waveform_inputs = [load_tensor_from_wav(path, TARGET_SAMPLE_RATE) for path in input_audio_filepaths]
    if len(waveform_inputs) == 0:
        raise Exception('Input audio directory must contain at least one WAV file')

    model_names = answers['models']
    benchmark_name = answers['name']

    results_dirpath = f'{BENCHMARK_PATH}/{benchmark_name}'
    create_dir_if_not_exists(results_dirpath)

    # Load transcriptions from metadata.csv file
    metadata_path = answers['transcriptions']
    target_transcripts = []
    with open(metadata_path, encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        next(reader)  # Skip the header row
        for row in reader:
            transcription = row[1]  # Assuming the transcriptions are in the third column
            target_transcripts.append(transcription)

    # Run benchmark for batches and write results to disk
    inputs = [_[0] for _ in waveform_inputs]
    # print([inp.shape for inp in inputs])  # Add this line to check the shape of the tensors
    results = benchmark(
        inputs,
        TARGET_SAMPLE_RATE,
        model_names,
        target_transcripts
    )

    pretty_results = prettify_results(results)

    results_pretty_filepath = f'{results_dirpath}/results.txt'
    with open(results_pretty_filepath, 'w') as f:
        f.write(pretty_results)

    print("Benchmark completed successfully.")


if __name__ == '__main__':
    main()
