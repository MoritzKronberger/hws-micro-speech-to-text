"""Run selected app action."""

import inquirer
from inspect import iscoroutinefunction
from app import noise_floor_calibration
from app import preprocessing_visualization
from app import live_transcription
from app.utils import print_hash_comment

actions = {
    'Calibrate noise floor': noise_floor_calibration,
    'Visualize preprocessing': preprocessing_visualization,
    'Start live transcription': live_transcription,
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
    print_hash_comment('Micro Text-to-Speech (Micro STT)')
    answers = inquirer.prompt([action_list])
    if answers is not None:
        action = answers['action']
        main = actions[action].main
        if iscoroutinefunction(main):
            await main()
        else:
            main()
