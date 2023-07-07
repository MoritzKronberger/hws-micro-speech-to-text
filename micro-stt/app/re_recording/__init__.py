"""Define main wav re-recording function."""

import inquirer
from app.utils import create_dir_if_not_exists, get_immidiate_sub_dirs
from app.wav import get_wav_files, load_tensor_from_wav, playrec_blocking, save_tensor_as_wav
from app.env import IN_PATH, RECORDINGS_PATH, TARGET_SAMPLE_RATE


def main() -> None:
    """Re-record batch of wav files."""
    input_dirs = get_immidiate_sub_dirs(IN_PATH)

    # Prompt user for input and output directory
    prompts = [
        inquirer.List(
            'audio_in_dir',
            message='Audio input directory',
            choices=input_dirs
        ),
        inquirer.Text(
            'out_dirname',
            message='Output directory name'
        ),
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No configuration set')

    input_audio_filepaths = get_wav_files(answers['audio_in_dir'])
    waveform_inputs = [load_tensor_from_wav(path, TARGET_SAMPLE_RATE) for path in input_audio_filepaths]
    out_dirpath = f'{RECORDINGS_PATH}/{answers["out_dirname"]}'
    create_dir_if_not_exists(out_dirpath)

    # Re-record wav files
    for [waveform, sample_rate], orig_filepath in zip(waveform_inputs, input_audio_filepaths):
        # Re-record audio
        recording = playrec_blocking(
            waveform,
            sample_rate,
            countdown_s=3
        )
        # Save recording to disk
        print('Saving recording to disk...')
        filename = orig_filepath.split('/')[-1].replace('.wav', '')
        save_tensor_as_wav(recording, sample_rate, filename, out_dirpath)
        print('Successfully saved recording')

    print('Successfully completed re-recording')
