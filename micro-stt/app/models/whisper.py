"""PyTorch Whisper model."""

from typing import Literal
import torch
from app.models import IModel, model_inputs
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class __GenericWhisper(IModel):
    """Generic PyTorch Whisper model.

    Used to instantiate small and tiny versions.
    """

    is_pytorch = True

    def __init__(self, size: Literal['tiny', 'small']) -> None:
        """Create new generic Whisper model.

        Reference:
        https://huggingface.co/openai/whisper-small#transcription
        """
        model_path = f'openai/whisper-{size}'
        self.name = f'Whisper ({size})'
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_path)
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            model_path
        )
        self.model.config.forced_decoder_ids = None

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe input batches.

        Reference:
        https://huggingface.co/openai/whisper-small#transcription
        """
        # Flatten inputs into single array
        flat_inputs = torch.stack(inputs).flatten()
        input_features = self.processor(flat_inputs, sampling_rate=sample_rate, return_tensors='pt').input_features
        predicted_ids = self.model.generate(input_features)
        output = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return ';'.join(output)


class WhisperSmall(__GenericWhisper):
    """Small PyTorch Whisper model."""

    def __init__(self) -> None:
        """Create new small Whisper model."""
        super().__init__('small')


class WhisperTiny(__GenericWhisper):
    """Tiny PyTorch Whisper model."""

    def __init__(self) -> None:
        """Create new tiny Whisper model."""
        super().__init__('tiny')
