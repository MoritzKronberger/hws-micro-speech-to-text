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
from app.config import BLOCK_SIZE, CHANNELS, NP_BUFFER, DEVICE_SAMPLE_RATE, TARGET_SAMPLE_RATE, MODEL, PASSTHROUGH
from app.models import IModel
from app.models.silero import Silero
from app.models.hubert import HuBERT

# Store audio buffers in queue
buffer_queue = Queue()

# Register available transcription models
models = {
    'silero': Silero,
    'hubert': HuBERT,
}


def __create_model() -> IModel:
    """Create the model that is specified in config."""
    return models[MODEL]()


def callback(in_data: NP_BUFFER, out_data: NP_BUFFER, _frame: int, _time: int, status: str):
    """Add audio buffer to buffer queue."""
    if status:
        print(status)
    if PASSTHROUGH:
        out_data[:] = in_data
    buffer_queue.put(in_data)


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
            buffer_data: NP_BUFFER = buffer_queue.get()
            # And flatten dimensions
            full_buffer = buffer_data.flatten()
            in_tensor = torch.from_numpy(full_buffer)
            # Resample audio to match valid sampling rate for model
            if DEVICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                in_tensor = resample(in_tensor, DEVICE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            # Normalize input data between -1 and 1
            # References:
            # - https://pytorch.org/hub/snakers4_silero-models_stt
            # - https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            in_tensor = torch.nn.functional.normalize(in_tensor, p=2, dim=0)
            # TODO: This is where VAD could be performed
            # Transcribe audio
            transcription = model.transcribe_live(in_tensor)
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
