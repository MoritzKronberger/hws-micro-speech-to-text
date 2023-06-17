"""Define main transcription function.

Why separate the callback and transcription worker?
- Transcription is handled in separate thread
- Audio passthrough can operate on different latency than transcription
"""

import asyncio
import threading
import torch
import sounddevice as sd
from typing import Callable
from queue import Queue, Empty
from torchaudio.functional import resample
from app.models import IModel
from app.preprocessing import preprocess_tensor
from app.config import models, preprocessing_options
from app.env import BLOCK_SIZE, CHANNELS, NP_BUFFER, DEVICE_SAMPLE_RATE, TARGET_SAMPLE_RATE, MODEL, PASSTHROUGH

# Store audio buffers in queue
buffer_queue = Queue()


def __create_model() -> IModel:
    """Create the model that is specified in config."""
    return models[MODEL]()


def callback(in_data: NP_BUFFER, out_data: NP_BUFFER, _frame: int, _time: int, status: str):
    """Add audio buffer to buffer queue."""
    # Check for errors
    if status:
        print(status)

    # Preprocess audio input
    flat_data = in_data.flatten()
    flat_tensor = torch.from_numpy(flat_data)
    preprocessed_tensor = preprocess_tensor(
        input=flat_tensor,
        sample_rate=DEVICE_SAMPLE_RATE,
        opts=preprocessing_options
    )

    # Pass audio to output (for monitoring)
    if PASSTHROUGH:
        preprocessed_np = preprocessed_tensor.numpy()
        out_data[:] = preprocessed_np.reshape(-1, 1)

    # Put data in queue for transcription
    buffer_queue.put(preprocessed_tensor)


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
            in_tensor: torch.Tensor = buffer_queue.get()
            # Resample audio to match valid sampling rate for model
            if DEVICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                pass
                in_tensor = resample(in_tensor, DEVICE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            # Transcribe audio
            transcription = model.transcribe_tensor(in_tensor, TARGET_SAMPLE_RATE)
            print(transcription)
        except Empty:
            continue
    print('Transcription worker received stop event and exited')


async def main() -> None:
    """Run main transcription function."""
    # Print available audio devices
    audio_devices = sd.query_devices()
    print(audio_devices)

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
