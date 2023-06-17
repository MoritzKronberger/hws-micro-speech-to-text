"""Micro Text-to-Speech application."""

import asyncio
import sys
from . import main

if __name__ == "__main__":
    # Run transcription in async coroutine
    # Reference:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#using-a-stream-in-an-asyncio-coroutine
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Gracefully stop worker threads
        # Reference: https://superfastpython.com/stop-daemon-thread
        stop = asyncio.Event()
        stop.set()
        sys.exit('\nInterrupted by user')
