"""PyTorch Whisper model."""

import torch
from typing import Literal
from whispercpp import Whisper as WhisperCPP
from app.models import IModel, model_inputs
from transformers import WhisperProcessor, WhisperForConditionalGeneration

WhisperSize = Literal['tiny', 'small', 'base']


class __GenericWhisper(IModel):
    """Generic PyTorch Whisper model.

    Used to instantiate small and tiny versions.
    """

    is_pytorch = True

    def __init__(self, size: WhisperSize) -> None:
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


class __GenericWhisperCPP(IModel):
    """Use Python bindings for C++ port of the Whisper model.

    Generic wrapper to instantiate small or tiny model.

    References:
    - https://github.com/stlukey/whispercpp.py
    - https://github.com/ggerganov/whisper.cpp
    """

    is_pytorch = False

    def __init__(self, size: WhisperSize) -> None:
        """Create new generic Whisper C++ model.

        Reference:
        https://github.com/stlukey/whispercpp.py
        """
        self.name = f'Whisper C++ ({size})'
        self.model = WhisperCPP(size)

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe input batches.

        Reference:
        https://github.com/stlukey/whispercpp.py
        """
        # Flatten inputs into single array
        flat_inputs = torch.stack(inputs).flatten()
        # Convert to Numpy array
        np_inputs = flat_inputs.numpy()
        segments = self.model.transcribe(np_inputs)
        outputs = self.model.extract_text(segments)
        return ';'.join(outputs)


class WhisperCPPBase(__GenericWhisperCPP):
    """Use Python bindings for C++ port of the base Whisper model.

    References:
    - https://github.com/stlukey/whispercpp.py
    - https://github.com/ggerganov/whisper.cpp
    """

    def __init__(self) -> None:
        """Create new base Whisper C++ model."""
        super().__init__('base')


class WhisperCPPSmall(__GenericWhisperCPP):
    """Use Python bindings for C++ port of the small Whisper model.

    References:
    - https://github.com/stlukey/whispercpp.py
    - https://github.com/ggerganov/whisper.cpp
    """

    def __init__(self) -> None:
        """Create new small Whisper C++ model."""
        super().__init__('small')


class WhisperCPPTiny(__GenericWhisperCPP):
    """Use Python bindings for C++ port of the tiny Whisper model.

    References:
    - https://github.com/stlukey/whispercpp.py
    - https://github.com/ggerganov/whisper.cpp
    """

    def __init__(self) -> None:
        """Create new tiny Whisper C++ model."""
        super().__init__('tiny')
