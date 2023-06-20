"""PyTorch Silero model."""

import torch
from app.models import IModel, model_inputs
from app.env import LANGUAGE


class __GenericSilero(IModel):
    """Generic PyTorch Silero model.

    Used to instantiate quantized and non-quantized versions.

    Reference:
    https://pytorch.org/hub/snakers4_silero-models_stt
    """

    is_pytorch = True

    def __init__(self, quantized: bool = False) -> None:
        """Create new Silero model.

        Model is optionally quantized.

        References:
        - https://pytorch.org/hub/snakers4_silero-models_stt
        - https://github.com/MiscellaneousStuff/openai-whisper-cpu/blob/main/script/custom_whisper.py
        """
        self.name = f'Silero{" (quantized)" if quantized else ""}'
        self.device = torch.device('cpu')
        self.model, self.decoder, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=LANGUAGE,
            device=self.device
        )
        if quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe input batches.

        Reference:
        https://pytorch.org/hub/snakers4_silero-models_stt
        """
        # Transcribe inputs
        prepare_model_input = self.utils[-1]
        input = prepare_model_input(inputs, self.device)
        output = self.model(input)
        return ';'.join(
            [self.decoder(example.cpu()) for example in output]
        )


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
