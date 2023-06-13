"""Micro Text-To_Speech models."""

from torch import Tensor


class IModel():
    """Interface for transcription model."""

    name: str
    is_pytorch: bool

    def transcribe_live(self, in_tensor: Tensor) -> str:
        """Transcribe live audio."""
        raise NotImplementedError
