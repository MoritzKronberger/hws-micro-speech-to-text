"""Run selected app action."""

import inquirer
from inspect import iscoroutinefunction
from app import recording
from app import re_recording
from app import noise_floor_calibration
from app import preprocessing_visualization
from app import file_transcription
from app import live_transcription
from app import performance_benchmark
from app import quality_benchmark
from app.utils import get_hash_comment

actions = {
    'Record audio (save as WAV)': recording,
    'Re-record audio files (from WAV)': re_recording,
    'Calibrate noise floor (from WAV)': noise_floor_calibration,
    'Visualize preprocessing (from WAV)': preprocessing_visualization,
    'Transcribe recording (from WAV)': file_transcription,
    'Live transcription (audio stream)': live_transcription,
    'Performance benchmark': performance_benchmark,
    'Quality benchmark':  quality_benchmark
}


async def main() -> None:
    """Prompt and run app action."""
    # Define selection for app actions using inquirer
    # Reference: https://github.com/magmax/python-inquirer
    action_list = inquirer.List(
        'action',
        message='Launch Micro STT action',
        choices=actions.keys()
    )
    # Prompt actions
    print(
        '\n'
        f'{get_hash_comment("Micro Text-to-Speech (Micro STT)")}'
        '\n'
    )
    answers = inquirer.prompt([action_list])
    if answers is not None:
        action = answers['action']
        main = actions[action].main
        if iscoroutinefunction(main):
            await main()
        else:
            main()
