"""Run selected app action."""

import inquirer
from inspect import iscoroutinefunction
from app import recording
from app import noise_floor_calibration
from app import preprocessing_visualization
from app import file_transcription
from app import live_transcription
from app import performance_benchmark
from app.utils import get_hash_comment

actions = {
    'Record audio (save as wav)': recording,
    'Calibrate noise floor (from wav)': noise_floor_calibration,
    'Visualize preprocessing (from wav)': preprocessing_visualization,
    'Transcribe recording (from wav)': file_transcription,
    'Start live transcription': live_transcription,
    'Run performance benchmark': performance_benchmark,
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
