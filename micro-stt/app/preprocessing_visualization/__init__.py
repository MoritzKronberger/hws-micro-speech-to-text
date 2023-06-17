"""Define main preprocessing visualization function."""

import torch
import sounddevice as sd
from app.env import DEVICE_SAMPLE_RATE, NOISE_FLOOR_DURATION_S, CHANNELS
from app.preprocessing import visualize_preprocessing
from app.utils import print_countdown
from app.config import preprocessing_options


def main():
    """Run main processing visualization."""
    # Record input to visualize
    print(sd.query_devices())
    print(f'Recording {NOISE_FLOOR_DURATION_S}s input to visualize in...')
    print_countdown(5)
    print('Started recording...')
    input = sd.rec(
        int(NOISE_FLOOR_DURATION_S * DEVICE_SAMPLE_RATE),
        DEVICE_SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocking=True
    )
    print('Successfully recorded noise floor')
    print('Visualizing preprocessing...')
    input_tensor = torch.from_numpy(input.flatten())
    visualize_preprocessing(input_tensor, DEVICE_SAMPLE_RATE, preprocessing_options)
