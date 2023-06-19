"""Define main wav recording function."""

import inquirer
from app.utils import create_dir_if_not_exists
from app.wav import record_blocking, save_tensor_as_wav
from app.env import DEVICE_SAMPLE_RATE, RECORDINGS_PATH


def main():
    """Run main wav recording function."""
    # Prompt user for recording duration and filename
    prompts = [
        inquirer.Text(
            'duration',
            message='Duration [s]'
        ),
        inquirer.Text(
            'filename',
            message='Filename'
        ),
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No configuration set')

    duration_s = float(answers['duration'])
    filename: str = answers['filename']
    filename = filename.replace('.wav', '')  # File extension not needed

    # Start recording
    recording = record_blocking(
        duration_s,
        DEVICE_SAMPLE_RATE
    )

    # Save recording to disk
    print('Saving recording to disk...')
    create_dir_if_not_exists(RECORDINGS_PATH)
    save_tensor_as_wav(recording, DEVICE_SAMPLE_RATE, filename, RECORDINGS_PATH)
    print('Successfully saved recording')
