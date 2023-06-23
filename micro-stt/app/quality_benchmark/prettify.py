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

    # Show preprocessing information
    preprocessing = results['preprocessing']
    if preprocessing is not None:
        preprocessing_header = f'{get_hash_comment("Preprocessing info")}\n'
        pretty_string += preprocessing_header
        preprocessing_info_str = ''
        bandpass = preprocessing['bandpass']
        noise_reduce = preprocessing['noise_reduce']
        scale = preprocessing['scale']
        if bandpass is not None:
            preprocessing_info_str += (
                'Bandpass\n'
                f'High cutoff frequency [Hz]: {bandpass["high_cutoff_freq"]}\n'
                f'Low cutoff frequency [Hz]: {bandpass["low_cutoff_freq"]}\n'
                f'Q: {bandpass["Q"]}\n'
                '\n'
            )
        if noise_reduce is not None:
            preprocessing_info_str += (
                'Noise reduction\n'
                f'Stationary: {noise_reduce["stationary"]}\n'
                f'Noise decrease (proportional): {noise_reduce["prop_decrease"]}\n'
                f'Time constant [s]: {noise_reduce["time_constant_s"]}\n'  # type: ignore
                '\n'
            )
        if scale is not None:
            preprocessing_info_str += (
                'Scale amplitude\n'
                f'Factor: {scale}\n'
                '\n'
            )
        preprocessing_info_str += '\n'
        pretty_string += preprocessing_info_str

    # Show benchmark results
    results_header = f'{get_hash_comment("Results")}\n'
    pretty_string += results_header

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
    pretty_string += '\n'

    return pretty_string
