"""Define main preprocessing visualization function."""

import inquirer
from app.env import RECORDINGS_PATH, TARGET_SAMPLE_RATE
from app.preprocessing import preprocess_tensor, visualize_preprocessing
from app.config import preprocessing_options
from app.wav import get_wav_files, load_tensor_from_wav, save_tensor_as_wav


def main():
    """Run main processing visualization."""
    # Prompt user for recording to use as noise floor
    recordings = get_wav_files(RECORDINGS_PATH)
    prompts = [
        inquirer.List(
            'recording_filepath',
            message='Recording to use as noise floor',
            choices=recordings
        ),
        inquirer.Confirm(
            'save_wav',
            message='Save preprocessed audio to recordings'
        ),
        inquirer.Text(
            'sub_dirname',
            message='Visualization sub-directory name'
        )
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No recording selected')

    # Load recording
    filepath = answers['recording_filepath']
    save_wav: bool = answers['save_wav']
    sub_dirname = answers['sub_dirname']
    input_tensor, sample_rate = load_tensor_from_wav(filepath, target_sample_rate=TARGET_SAMPLE_RATE)
    print('Successfully loaded recording')

    visualize_preprocessing(input_tensor, sample_rate, preprocessing_options, sub_dirname)

    # Preprocess audio and save as wav
    if save_wav:
        print('Saving preprocessed audio to recordings...')
        preprocessed_tensor = preprocess_tensor(input_tensor, sample_rate, preprocessing_options)
        save_tensor_as_wav(preprocessed_tensor, sample_rate, f'{sub_dirname}_preprocessed', RECORDINGS_PATH)
        print('Successfully saved preprocessed audio')
