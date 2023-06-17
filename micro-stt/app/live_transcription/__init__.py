"""Define main transcription function.

Why separate the callback and transcription worker?
- Transcription is handled in separate thread
- Audio passthrough can operate on different latency than transcription
"""

import asyncio
import threading
import inquirer
import torch
import torchaudio
import sounddevice as sd
from typing import Callable
from queue import Queue, Empty
from app.models import IModel
from app.preprocessing import preprocess_tensor
from app.config import models, preprocessing_options
from app.env import BLOCK_SIZE, CHANNELS, NP_BUFFER, DEVICE_SAMPLE_RATE, TARGET_SAMPLE_RATE

# Store audio buffers in queue
buffer_queue = Queue()

# Transcription settings
preprocess = True
model_name = 'silero'
passthrough = False


def __create_model() -> IModel:
    """Create the model that is specified in globals."""
    return models[model_name]()


def callback(in_data: NP_BUFFER, out_data: NP_BUFFER, _frame: int, _time: int, status: str):
    """Add audio buffer to buffer queue."""
    # Check for errors
    if status:
        print(status)

    # Transform audio input
    flat_data = in_data.flatten()
    waveform_tensor = torch.from_numpy(flat_data)

    if preprocess:
        waveform_tensor = preprocess_tensor(
            input=waveform_tensor,
            sample_rate=DEVICE_SAMPLE_RATE,
            opts=preprocessing_options
        )

    # Pass audio to output (for monitoring)
    if passthrough:
        preprocessed_np = waveform_tensor.numpy()
        out_data[:] = preprocessed_np.reshape(-1, 1)

    # Put data in queue for transcription
    buffer_queue.put(waveform_tensor)


def transcription_worker(create_model: Callable[[], IModel]) -> None:
    """Transcribe audio in buffer queue."""
    # Create transcription model
    model: IModel = create_model()
    print(
        'Starting transcription worker'
        f' with model {model.name}'
    )
    # Until stop event is set
    # Reference: https://superfastpython.com/stop-daemon-thread
    stop = asyncio.Event()

    while not stop.is_set():
        try:
            # Get queue data
            waveform_tensor: torch.Tensor = buffer_queue.get()
            # Resample audio input
            if DEVICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                resample = torchaudio.transforms.Resample(
                    orig_freq=DEVICE_SAMPLE_RATE,
                    new_freq=TARGET_SAMPLE_RATE
                )
                waveform_tensor = resample(waveform_tensor)
            # Transcribe audio
            transcription = model.transcribe_tensor(waveform_tensor, TARGET_SAMPLE_RATE)
            print(transcription)
        except Empty:
            continue
    print('Transcription worker received stop event and exited')


async def main() -> None:
    """Run main transcription function."""
    # Prompt user for transcription settings
    prompts = [
        inquirer.List(
            'model',
            message='Model',
            choices=models.keys()
        ),
        inquirer.Confirm(
            'preprocess',
            message='Preprocess recording'
        ),
        inquirer.Confirm(
            'passthrough',
            message='Enable audio passthrough'
        )
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No recording selected')

    # Apply transcription settings
    global model_name, preprocess, passthrough
    model_name = answers['model']
    preprocess = answers['preprocess']
    passthrough = answers['passthrough']

    # Print available audio devices
    audio_devices = sd.query_devices()
    print(f'{audio_devices}\n')

    # Start transcription worker in new thread
    # Reference: https://superfastpython.com/daemon-threads-in-python
    threading.Thread(
        target=transcription_worker,
        args=[__create_model],
        daemon=True
    ).start()

    # Create audio input stream
    # Add input audio buffer to transcription queue using on-data callback
    input_stream = sd.Stream(
        samplerate=DEVICE_SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype='float32',
        callback=callback
    )

    # Run stream until stop event
    # Reference:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#using-a-stream-in-an-asyncio-coroutine
    stop = asyncio.Event()
    with input_stream:
        await stop.wait()
