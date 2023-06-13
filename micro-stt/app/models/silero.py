"""PyTorch Silero model."""

import torch
from app.models import IModel
from app.config import LANGUAGE


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

    def transcribe_live(self, in_tensor: torch.Tensor) -> str:
        """Transcribe live data.
        
        Reference:
        https://pytorch.org/hub/snakers4_silero-models_stt
        """
        in_tensor_batches = [in_tensor]
        prepare_model_input = self.utils[-1]
        input = prepare_model_input(in_tensor_batches, self.device)
        output = self.model(input)
        return ';'.join(
            [self.decoder(example.cpu()) for example in output]
        )
