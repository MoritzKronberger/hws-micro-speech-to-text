"""PyTorch Silero model."""

import torch
from app.models import IModel, model_inputs
from app.env import LANGUAGE


class Silero(IModel):
    """PyTorch Silero model.

    Reference:
    https://pytorch.org/hub/snakers4_silero-models_stt
    """

    name = 'Silero'
    is_pytorch = True

    def __init__(self) -> None:
        """Create new Silero model.

        Reference:
        https://pytorch.org/hub/snakers4_silero-models_stt
        """
        self.device = torch.device('cpu')
        self.model, self.decoder, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=LANGUAGE,
            device=self.device
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
