"""Define main file transcription function."""

import inquirer
from app.env import RECORDINGS_PATH, TARGET_SAMPLE_RATE
from app.models import IModel
from app.preprocessing import preprocess_tensor
from app.wav import get_wav_files, load_tensor_from_wav
from app.config import models, preprocessing_options


def transcribe_file(filepath: str, model: IModel, preprocess: bool) -> None:
    """Transcribe wav file using STT model."""
    print('Loading recording...')
    waveform, sample_rate = load_tensor_from_wav(filepath, TARGET_SAMPLE_RATE)
    if preprocess:
        print('Preprocessing recording...')
        waveform = preprocess_tensor(waveform, sample_rate, preprocessing_options)
    print('Transcribing...')
    transcription = model.transcribe_tensor([waveform], sample_rate)
    print(f'\n{transcription}\n')


def main():
    """Run main file transcription."""
    # Prompt user for recording to transcribe
    recordings = get_wav_files(RECORDINGS_PATH)
    prompts = [
        inquirer.List(
            'recording_filepath',
            message='Recording to transcribe',
            choices=recordings
        ),
        inquirer.List(
            'model',
            message='Model',
            choices=models.keys()
        ),
        inquirer.Confirm(
            'preprocess',
            message='Preprocess recording'
        )
    ]
    answers = inquirer.prompt(prompts)
    if answers is None:
        raise Exception('No recording selected')

    # Transcribe recording
    filepath = answers['recording_filepath']
    model_name = answers['model']
    model = models[model_name]()
    preprocess = answers['preprocess']
    transcribe_file(filepath, model, preprocess)
