"""Micro Text-To_Speech models."""

from torch import Tensor


class IModel():
    """Interface for transcription model."""

    name: str
    is_pytorch: bool

    def transcribe_tensor(self, waveform_tensor: Tensor, sample_rate: int) -> str:
        """Transcribe waveform tensor."""
        raise NotImplementedError
