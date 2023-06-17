"""Define main noise floor calibration function."""

import inquirer
from app.preprocessing import save_noise_floor, preprocess_tensor, preprocessing_opts
from app.config import preprocessing_options
from app.env import RECORDINGS_PATH
from app.wav import get_wav_files, load_tensor_from_wav


def main():
    """Run main noise floor calibration."""
    # Prompt user for recording to use as noise floor
    recordings = get_wav_files(RECORDINGS_PATH)
    recordings_list = inquirer.List(
        'recording_filepath',
        message='Recording to use as noise floor',
        choices=recordings
    )
    answers = inquirer.prompt([recordings_list])
    if answers is None:
        raise Exception('No recording selected')
    
    # Load recording
    filepath = answers['recording_filepath']
    noise_floor_tensor, sample_rate = load_tensor_from_wav(filepath)
    print(f'Successfully loaded recording')

    # Apply all other preprocessing steps
    print('Preprocessing recording...')
    prep_options: preprocessing_opts = {
        'noise_reduce': None,
        **preprocessing_options,
    }
    preprocessed_noise_tensor = preprocess_tensor(
        noise_floor_tensor,
        sample_rate,
        prep_options
    )

    # Save noise flor as numpy file
    # (For use with noisereduce library)
    print('Saving noise floor to disk...')
    preprocessed_noise_floor_np = preprocessed_noise_tensor.numpy()
    save_noise_floor(preprocessed_noise_floor_np)
    print('Successfully saved noise floor to disk')
