"""Define main transcription function.

Why separate the callback and transcription worker?
- Transcription is handled in separate thread
- Audio passthrough can operate on different latency than transcription
"""

import asyncio
import threading
import torch
import numpy as np
import sounddevice as sd
from typing import Callable
from queue import Queue, Empty
from time import sleep
from app.config import BLOCK_SIZE, CHANNELS, NP_BUFFER, SAMPLE_RATE, MODEL, PASSTHROUGH, MIN_TRANSCRPTION_INPUT_DURATION_S
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
    # Keep data that could not be processed
    unprocessed: NP_BUFFER = np.empty(0, dtype=np.float32)
    while not stop.is_set():
        try:
            # Get entire queue data
            buffer_list: list[NP_BUFFER] = []
            while not buffer_queue.empty():
                buffer_data: NP_BUFFER = buffer_queue.get()
                buffer_list.append(buffer_data)
                buffer_queue.task_done()
            # Convert queue data to numpy array
            full_buffer = np.array(buffer_list)
            # And flatten dimensions
            full_buffer = full_buffer.flatten()
            # Prepend unprocessed data from previous run
            full_buffer = np.append(unprocessed, full_buffer)
            # Wait until enough data has been gathered
            if full_buffer.shape[0] > SAMPLE_RATE * MIN_TRANSCRPTION_INPUT_DURATION_S:
                in_tensor = torch.from_numpy(full_buffer)
                # TODO: This is where VAD could be performed
                # Transcribe audio
                transcription = model.transcribe_live(in_tensor)
                print(transcription)
                # Reset unprocessed data
                unprocessed = np.empty(0, dtype=np.float32)
            # Otherwise set current data as unprocessed
            else:
                unprocessed = full_buffer
            sleep(BLOCK_SIZE / SAMPLE_RATE)
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
        samplerate=SAMPLE_RATE,
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
