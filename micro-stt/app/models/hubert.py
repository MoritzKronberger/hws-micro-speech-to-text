"""PyTorch HuBERT model."""

import torch
from transformers import AutoProcessor, HubertForCTC, PreTrainedModel
from app.models import IModel, model_inputs


class HuBERT(IModel):
    """PyTorch HuBERT model.

    Reference:
    https://huggingface.co/docs/transformers/model_doc/hubert
    """

    name = 'HuBERT'
    is_pytorch = True
    max_input_length = 1

    def __init__(self) -> None:
        """Create new HuBERT model.
        
        Reference:
        https://huggingface.co/docs/transformers/model_doc/hubert
        """
        model: PreTrainedModel = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        if not isinstance(model, HubertForCTC):
            raise Exception('Downloaded model as incorrect type.')
        self.model = model
        # Set model to evaluation mode
        # (Deactivates dropout)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    def transcribe_tensor(self, inputs: model_inputs, sample_rate: int) -> str:
        """Transcribe live data.
        
        Reference:
        https://huggingface.co/docs/transformers/model_doc/hubert
        """
        # Flatten inputs into single array
        flat_inputs = torch.stack(inputs).flatten()
        # Transcribe inputs
        input = self.processor(flat_inputs, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**input).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids)
        return ';'.join(output)
