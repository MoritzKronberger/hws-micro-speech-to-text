"""PyTorch Whisper model."""

import torch
from app.models import IModel, model_inputs
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class Whisper(IModel):
    """PyTorch Whisper model."""

    name = 'Whisper'
    is_pytorch = True

    def __init__(self) -> None:
        """Create new Whisper model.

        Reference:
        https://huggingface.co/openai/whisper-small#transcription
        """
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
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
