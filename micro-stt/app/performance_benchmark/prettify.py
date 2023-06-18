"""Format benchmark results as pretty string."""

from prettytable import PrettyTable
from app.performance_benchmark.result_types import full_results
from app.utils import byte_to_mb, get_hash_comment


def prettify_results(results: full_results) -> str:
    """Format benchmark results as pretty string."""
    pretty_string = ''
    # Format system info
    system_info = results['system_info']
    system_info_header = f'{get_hash_comment("System info")}\n'
    pretty_string += system_info_header
    system_info_str = (
        f'Arch: {system_info["machine"]}\n'
        f'Platform: {system_info["system"]} {system_info["version"]}\n'
        f'CPU: {system_info["processor"]}, {system_info["cpu_speed_ghz"]} GHz\n'
        f'Memory: {byte_to_mb(system_info["memory_byte"])} MB\n'
        '\n'
    )
    pretty_string += system_info_str

    # Format benchmark info
    model_results = results["model_results"]
    audio_duration_s = results["audio_duration_ms"] * 0.001
    model_benchmark_info = (
        f'Benchmarked {len(model_results)} models '
        f'for {audio_duration_s} seconds of audio\n'
    )
    pretty_string += model_benchmark_info

    for model_result in model_results:
        # Format model header
        model_header = f'\n{get_hash_comment(model_result["model_name"])}\n'
        pretty_string += model_header
        # Format model results
        result = model_result['results']
        torch_result = model_result.get('torch_results')
        result_str = (
            'Universal benchmark results:\n'
            f'Memory usage RSS [mb]: {byte_to_mb(result["memory_rss_byte"])}\n'
            f'Inference time [ms]: {result["inference_time_ms"]}\n'
            f'Per core 1 / RTF: {result["per_core_1_over_rtf"]}\n'
            '\n'
        )
        pretty_string += result_str
        if torch_result is not None:
            torch_result_str = (
                'Torch benchmark results:\n'
                f'Memory usage [MB]: {byte_to_mb(torch_result["cpu_memory_usage_byte"])}\n'
                f'CPU time [ms]: {torch_result["cpu_time_ms"]}\n'
                f'Self CPU time [ms]: {torch_result["self_cpu_time_ms"]}\n'
                f'MFlops: {torch_result["flops"] * 0.000001}\n'
                '\n'
            )
            pretty_string += torch_result_str
        # Format micro controller compatibility
        micro_controller_compats = model_result['micro_controllers_compats']
        table_headers = [
            'Micro Controller',
            'CPU Speed [GHz]',
            'Memory [MB]',
            'Memory RSS [MB]',
            'Estimated Inference Time [ms]',
            'Estimated per core 1 / RTF',
            'Compatible'
        ]
        table_rows: list[list[str | float | bool]] = []
        for micro_ctr_compat in micro_controller_compats:
            row = [
                micro_ctr_compat['micro_controller_info']['name'],
                micro_ctr_compat['micro_controller_info']['cpu_speed_ghz'],
                micro_ctr_compat['micro_controller_info']['memory_mb'],
                byte_to_mb(micro_ctr_compat['memory_rss_byte']),
                micro_ctr_compat['inference_time_ms'],
                micro_ctr_compat['per_core_1_over_rtf'],
                micro_ctr_compat['compatible']
            ]
            table_rows.append(row)
        micro_controller_table = PrettyTable()
        micro_controller_table.field_names = table_headers
        micro_controller_table.add_rows(table_rows)
        micro_controller_str = micro_controller_table.get_string()
        pretty_string += micro_controller_str
        pretty_string += '\n'
    
    return pretty_string
