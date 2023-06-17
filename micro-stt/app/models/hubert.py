"""PyTorch HuBERT model."""

import torch
from transformers import AutoProcessor, HubertForCTC
from app.models import IModel
from app.env import TARGET_SAMPLE_RATE


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

    def transcribe_live(self, in_tensor: torch.Tensor) -> str:
        """Transcribe live data.
        
        Reference:
        https://huggingface.co/docs/transformers/model_doc/hubert
        """
        input = self.processor(in_tensor, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**input).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids)
        return ';'.join(output)
