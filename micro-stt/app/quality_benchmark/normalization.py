"""Normalize transcriptions for comparison."""


def normalize_transcriptions(transcriptions: list[str]) -> list[str]:
    """Normalize transcriptions for comparison.

    - Remove punctuation and quotation marks
    - To lower case
    """
    invalid_characters = ',;.:-–—\'"!?'
    norm_transcriptions: list[str] = []
    for transcription in transcriptions:
        for char in invalid_characters:
            transcription = transcription.replace(char, '')
        norm_transcriptions.append(transcription.lower())
    return norm_transcriptions
