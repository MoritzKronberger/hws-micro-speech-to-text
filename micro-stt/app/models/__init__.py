"""Micro Text-To_Speech models."""

from torch import Tensor

model_inputs = list[Tensor]


class IModel():
    """Interface for transcription model."""

    name: str
    is_pytorch: bool

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe input batches."""
        raise NotImplementedError
