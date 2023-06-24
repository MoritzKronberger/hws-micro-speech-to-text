"""PyTorch Silero model."""

import torch
from typing import Literal
from silero import silero_stt
from app.models import IModel, model_inputs


class __GenericSilero(IModel):
    """Generic PyTorch Silero model.

    Used to instantiate quantized and non-quantized versions.

    Reference:
    https://pytorch.org/hub/snakers4_silero-models_stt
    """

    is_pytorch = True

    def __init__(self, quantized: bool = False, language: Literal['en', 'de'] = 'en') -> None:
        """Create new Silero model.

        Model is optionally quantized.

        References:
        - https://pypi.org/project/silero
        - https://pytorch.org/hub/snakers4_silero-models_stt
        - https://github.com/MiscellaneousStuff/openai-whisper-cpu/blob/main/script/custom_whisper.py
        """
        self.name = f'Silero{" (quantized)" if quantized else ""}{" GERMAN " if language=="de" else ""}'
        self.device = torch.device('cpu')
        self.model, self.decoder, self.utils = silero_stt(
            # Set version as v4 for german model,
            # since latest version of german model is currently pinned to deprecated v1.
            # Reference:
            # https://github.com/snakers4/silero-models/blob/21ed251aa28d023db96a8fdaaf5b22877bc8c0af/models.yml#L83
            version='v4' if language == 'de' else 'lastest',
            # Needs different jit model name for german model v4
            # Reference:
            # https://github.com/snakers4/silero-models/blob/21ed251aa28d023db96a8fdaaf5b22877bc8c0af/models.yml#L100
            jit_model='jit_large' if language == 'de' else 'latest',
            language=language,
            device=self.device
        )
        if quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

    def transcribe_tensor_batches(self, inputs: model_inputs, sample_rate: int) -> list[str]:
        """Transcribe input batches.

        Reference:
        https://pytorch.org/hub/snakers4_silero-models_stt
        """
        # Transcribe inputs
        prepare_model_input = self.utils[-1]
        input = prepare_model_input(inputs, self.device)
        output = self.model(input)
        outputs = [self.decoder(example.cpu()) for example in output]
        return outputs


class Silero(__GenericSilero):
    """Non-quantized Silero model."""

    def __init__(self) -> None:
        """Create new non-quantized Silero model."""
        super().__init__(quantized=False)


class SileroQuantized(__GenericSilero):
    """Quantized Silero model."""

    def __init__(self) -> None:
        """Create new quantized Silero model."""
        super().__init__(quantized=True)


class SileroGerman(__GenericSilero):
    """German Silero model."""

    def __init__(self) -> None:
        """Create new german Silero model."""
        super().__init__(quantized=False, language='de')
