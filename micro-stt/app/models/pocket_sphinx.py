"""PocketSphinx Hidden Markov Model."""

import struct
from pocketsphinx import Decoder
from app.env import TARGET_SAMPLE_RATE
from app.models import IModel, model_inputs


class PocketSphinx(IModel):
    """PocketSphinx Hidden Markov Model."""

    name = 'PocketSphinx'
    is_pytorch = False

    def __init__(self) -> None:
        """Create new PocketSphinx model.

        References:
        - https://pypi.org/project/pocketsphinx
        - https://github.com/cmusphinx/pocketsphinx/blob/master/examples/simple.py
        """
        self.decoder = Decoder(samprate=TARGET_SAMPLE_RATE)

    def transcribe_tensor_batches(self, inputs: model_inputs, sample_rate: int) -> list[str]:
        """Transcribe input batches.

        References:
        - https://pypi.org/project/pocketsphinx
        - https://github.com/cmusphinx/pocketsphinx/blob/master/examples/simple.py
        """
        # Process inputs individually
        outputs: list[str] = []
        for input in inputs:
            # Convert float32 torch tensor (32-bit PCM) to int16 bytes (16-bit-PCM)
            # References:
            # - https://github.com/scipy/scipy/blob/37b650e04f3bf49bc11cfcd18f2848ad4f957a0d/scipy/io/wavfile.py#L699
            # - https://stackoverflow.com/a/43882434/14906871
            floats = input.tolist()
            ints = [int(sample * 32767) for sample in floats]
            int16_bytes = struct.pack('<%dh' % len(ints), *ints)
            # Transcribe utterance
            self.decoder.start_utt()
            self.decoder.process_raw(int16_bytes, full_utt=True)
            self.decoder.end_utt()
            output = self.decoder.hyp().hypstr
            outputs.append(output)
        return outputs
