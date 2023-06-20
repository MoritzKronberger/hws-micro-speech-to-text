"""Convert results to LaTEX tables."""


from app.performance_benchmark.result_types import full_results
from app.utils import byte_to_mb


def to_tex_table(results: full_results) -> str:
    """Convert results to LaTEX tables.

    Reference:
    https://www.overleaf.com/learn/latex/Tables
    """
    tex_string = ''
    model_results_table_str = [
        r'\begin{table}[ht]',
        r'\begin{tabularx}{\textwidth} {',
        r'| >{\raggedright\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X | }',
        r'\hline',
        r'\textbf{Model} & \textbf{Memory usage (RSS) [MB]} & \textbf{Inference time [ms]}' +
        r' & \textbf{1 / RTF per core} \\',
        r'\hline',
    ]
    microctr_compat_table_str = [
        r'\begin{table}[ht]',
        r'\begin{tabularx}{\textwidth} {',
        r'| >{\raggedright\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedright\arraybackslash}X | }',
        r'\hline',
    ]

    # Decimal formatter
    df = '{:.2f}'

    model_results = results["model_results"]

    # Format model results:
    for model_result in model_results:
        result = model_result['results']
        memory_usage_mb = df.format(byte_to_mb(result['memory_rss_byte']))
        inf_time = df.format(result['inference_time_ms'])
        one_over_rtf = df.format(result['per_core_1_over_rtf'])
        model_results_table_str.extend([
            model_result['model_name'] + r' & ' + memory_usage_mb + r' & ' + inf_time + r' & ' + one_over_rtf + r' \\',
            r'\hline'
        ])

    # End table
    model_results_table_str.extend(
        [
            r'\end{tabularx}',
            r'\caption{INSERT-CAPTION}',
            r'\label{table:INSERT-LABEL}',
            r'\end{table}',
        ]
    )

    # Format micro controller compatibility
    for model_result in model_results:
        compats = model_result['micro_controllers_compats']

        # Add model name row
        model_name = model_result['model_name']
        microctr_compat_table_str.extend(
            [
                (r'\multicolumn{5}{|c|}{\textbf{' + model_name + r'}} \\'),
                r'\hline'
            ]
        )

        # Add header row
        microctr_compat_table_str.extend(
            [
                r'\textbf{Micro controller} & \textbf{Memory usage [\%]} & \textbf{Inference time [ms]} &' +
                r'\textbf{1 / RTF per core} & \textbf{Compatible} \\',
                r'\hline'
            ]
        )

        # Add microcontroller compatibility rows
        for compat in compats:
            controller_info = compat['micro_controller_info']
            memory_usage = byte_to_mb(compat['memory_rss_byte']) / controller_info['memory_mb']
            memory_usage_percent = df.format(memory_usage * 100)
            inf_time = df.format(compat['inference_time_ms'])
            one_over_rtf = df.format(compat['per_core_1_over_rtf'])
            compatible = 'Yes' if memory_usage <= 1 and compat['per_core_1_over_rtf'] >= 1 else 'No'
            microctr_compat_table_str.extend(
                [
                    compat['micro_controller_info']['name'] + r' & ' + memory_usage_percent +
                    r' \% & ' + inf_time + r' & ' + one_over_rtf + r' & ' + compatible + r' \\',
                    r'\hline'
                ]
            )

    # End table
    microctr_compat_table_str.extend(
        [
            r'\end{tabularx}',
            r'\caption{INSERT-CAPTION}',
            r'\label{table:INSERT-LABEL}',
            r'\end{table}',
        ]
    )

    tex_string += '\n'.join(model_results_table_str)
    tex_string += '\n\n'
    tex_string += '\n'.join(microctr_compat_table_str)

    return tex_string