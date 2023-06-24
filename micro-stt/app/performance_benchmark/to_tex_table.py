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
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X | }',
        r'\hline',
        r'\textbf{Model} & $\overline{\textbf{Memory usage}}$ & $\overline{\textbf{Inference time}}$' +
        r' & $\overline{\textbf{RTF}}$ & $\overline{\textbf{RTF@1GHz/Core}}$ \\',
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
        std_memory_usage_mb = df.format(byte_to_mb(result['std_memory_rss_byte']))
        inf_time_ms = df.format(result['inference_time_ms'])
        std_inf_time_ms = df.format(result['std_inference_time_ms'])
        rtf = df.format(result['rtf'])
        rtf_at_1ghz = df.format(result['rtf_at_1ghz_per_core'])
        model_results_table_str.extend([
            model_result['model_name'] + r' & ' + memory_usage_mb + r' MB (\sigma=' + std_memory_usage_mb +
            r') & ' + inf_time_ms + r' ms (\sigma=' + std_inf_time_ms + r') & ' + rtf + r' & ' + rtf_at_1ghz + r' \\',
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
                r'\textbf{Micro controller} & \textbf{Memory usage} & \textbf{Inference time} &' +
                r'\textbf{RTF} & \textbf{Compatible} \\',
                r'\hline'
            ]
        )

        # Add microcontroller compatibility rows
        for compat in compats:
            controller_info = compat['micro_controller_info']
            memory_usage = byte_to_mb(compat['memory_rss_byte']) / controller_info['memory_mb']
            memory_usage_percent = df.format(memory_usage * 100)
            inf_time_ms = df.format(compat['inference_time_ms'])
            rtf = df.format(compat['rtf'])
            compatible = 'Yes' if compat['compatible'] else 'No'
            microctr_compat_table_str.extend(
                [
                    compat['micro_controller_info']['name'] + r' & ' + memory_usage_percent +
                    r' \% & ' + inf_time_ms + r' ms & ' + rtf + r' & ' + compatible + r' \\',
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
