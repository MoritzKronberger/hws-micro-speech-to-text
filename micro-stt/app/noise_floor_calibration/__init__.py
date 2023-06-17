"""Define main noise floor calibration function."""

import torch
import sounddevice as sd
from app.preprocessing import save_noise_floor, preprocess_tensor, preprocessing_opts
from app.config import preprocessing_options
from app.env import DEVICE_SAMPLE_RATE, NOISE_FLOOR_DURATION_S, CHANNELS
from app.utils import print_countdown


def main():
    """Run main noise floor calibration."""
    # Record noise floor
    print(sd.query_devices())
    print(f'Recording {NOISE_FLOOR_DURATION_S} s noise floor in...')
    print_countdown(5)
    print('Started recording...')
    noise_floor = sd.rec(
        int(NOISE_FLOOR_DURATION_S * DEVICE_SAMPLE_RATE),
        DEVICE_SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocking=True
    )
    print('Successfully recorded noise floor')
    # Apply all other preprocessing steps
    prep_options: preprocessing_opts = {
        'noise_reduce': None,
        **preprocessing_options,
    }
    print('Saving noise floor to disk...')
    noise_floor_tensor = torch.from_numpy(noise_floor.flatten())
    preprocessed_noise_tensor = preprocess_tensor(
        noise_floor_tensor,
        DEVICE_SAMPLE_RATE,
        prep_options
    )
    preprocessed_noise_floor_np = preprocessed_noise_tensor.numpy()
    # Save noise floor to disk
    save_noise_floor(preprocessed_noise_floor_np, DEVICE_SAMPLE_RATE)
    print('Successfully saved noise floor to disk')
