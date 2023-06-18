"""Micro Text-to-Speech Transcription."""

import asyncio
from . import main

if __name__ == "__main__":
    # Run transcription in async coroutine
    # Reference:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#using-a-stream-in-an-asyncio-coroutine
    # (Error handled by root module)
    asyncio.run(main())
