"""Format benchmark results as a pretty string."""
from prettytable import PrettyTable
from app.quality_benchmark.result_types import full_results
from app.utils import get_hash_comment


def prettify_results(results: full_results) -> str:
    """Format benchmark results as pretty string."""
    pretty_string = ''

    # Show benchmark information
    benchmark_info_header = f'{get_hash_comment("Benchmark info")}\n'
    pretty_string += benchmark_info_header
    benchmark_info_str = (
        f'Number of samples: {results["num_samples"]}\n'
        f'Mean audio duration [ms]: {results["mean_audio_duration_ms"]}\n'
        '\n'
    )
    pretty_string += benchmark_info_str

    model_results = results["model_results"]

    table_headers = [
        'Model',
        'Mean WER',
        'Mean MER',
        'Mean WIL',
        'Mean WIP',
        'Mean CER'
    ]
    table_rows = [
        [
            res['model_name'],
            res['mean_wer'],
            res['mean_mer'],
            res['mean_wil'],
            res['mean_wip'],
            res['mean_cer'],
        ] for res in model_results
    ]
    model_results_table = PrettyTable()
    model_results_table.field_names = table_headers
    model_results_table.add_rows(table_rows)
    model_results_str = model_results_table.get_string()
    pretty_string += model_results_str

    return pretty_string
