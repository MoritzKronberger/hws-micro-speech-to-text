"""PyTorch HuBERT model."""

import torch
from transformers import AutoProcessor, HubertForCTC
from app.models import IModel


class HuBERT(IModel):
    """PyTorch HuBERT model.

    Reference:
    https://huggingface.co/docs/transformers/model_doc/hubert
    """

    name = 'HuBERT'
    is_pytorch = True

    def __init__(self) -> None:
        """Create new HuBERT model.
        
        Reference:
        https://huggingface.co/docs/transformers/model_doc/hubert
        """
        model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        if not isinstance(model, HubertForCTC):
            raise Exception('Downloaded model as incorrect type.')
        self.model = model
        # Set model to evaluation mode
        # (Deactivates dropout)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    def transcribe_tensor(self, waveform_tensor: torch.Tensor, sample_rate: int) -> str:
        """Transcribe live data.
        
        Reference:
        https://huggingface.co/docs/transformers/model_doc/hubert
        """
        input = self.processor(waveform_tensor, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**input).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids)
        return ';'.join(output)
