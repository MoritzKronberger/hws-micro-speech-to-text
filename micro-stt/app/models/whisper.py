"""PyTorch Whisper model."""

import torch
import whisper
from typing import Literal
from whispercpp import Whisper as WhisperCPP
from app.models import IModel, model_inputs

WhisperSize = Literal['tiny', 'small', 'base']


class __GenericWhisper(IModel):
    """Generic PyTorch Whisper model.

    Used to instantiate small, tiny  and quantized versions.

    Reference:
    https://github.com/MiscellaneousStuff/openai-whisper-cpu/blob/main/script/custom_whisper.py
    """

    is_pytorch = True

    def __init__(self, size: WhisperSize, quantized: bool = False) -> None:
        """Create new generic Whisper model.

        Reference:
        https://github.com/MiscellaneousStuff/openai-whisper-cpu/blob/main/script/custom_whisper.py
        """
        self.name = f'Whisper ({size}{", quantized" if quantized else ""})'
        self.model = whisper.load_model(size)
        if quantized:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe input batches.

        Reference:
        https://github.com/MiscellaneousStuff/openai-whisper-cpu/blob/main/script/custom_whisper.py
        """
        # Process batch-wise (faster than single flattened tensor)
        outputs: list[str] = []
        for input in inputs:
            output = whisper.transcribe(self.model, input)
            outputs.append(output['text'])  # type: ignore
        return ' '.join(outputs)


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


class WhisperSmallQuantized(__GenericWhisper):
    """Small PyTorch Whisper model."""

    def __init__(self) -> None:
        """Create new small, quantized Whisper model."""
        super().__init__('small', quantized=True)


class WhisperTinyQuantized(__GenericWhisper):
    """Tiny PyTorch Whisper model."""

    def __init__(self) -> None:
        """Create new tiny, quantized Whisper model."""
        super().__init__('tiny', quantized=True)


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
        return ''.join(outputs)


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
