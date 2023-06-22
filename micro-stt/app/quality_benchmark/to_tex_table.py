"""Convert results to LaTEX tables."""


from app.quality_benchmark.result_types import full_results


def to_tex_table(results: full_results) -> str:
    """Convert results to LaTEX tables.

    Reference:
    https://www.overleaf.com/learn/latex/Tables
    """
    results_table_str = [
        r'\begin{table}[ht]',
        r'\begin{tabularx}{\textwidth} {',
        r'| >{\raggedright\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X',
        r'| >{\raggedleft\arraybackslash}X | }',
        r'\hline',
        r'\textbf{Model} & \textbf{WER} & \textbf{MER}' +
        r' & \textbf{WIL} & \textbf{WIP} & \textbf{CER} \\',
        r'\hline',
    ]

    # Decimal formatter
    df2 = '{:.2f}'
    df4 = '{:.4f}'

    model_results = results["model_results"]

    # Format model results:
    for res in model_results:
        results_table_str.extend([
            res['model_name'] + r' & '
            + df4.format(res['mean_wer']) + r' (\sigma=' + df2.format(res['std_wer']) + r') & '
            + df4.format(res['mean_mer']) + r' (\sigma=' + df2.format(res['std_mer']) + r') & '
            + df4.format(res['mean_wil']) + r' (\sigma=' + df2.format(res['std_wil']) + r') & '
            + df4.format(res['mean_wip']) + r' (\sigma=' + df2.format(res['std_wip']) + r') & '
            + df4.format(res['mean_cer']) + r' (\sigma=' + df2.format(res['std_cer']) + r') '
            + r' \\',
            r'\hline'
        ])

    # End table
    results_table_str.extend(
        [
            r'\end{tabularx}',
            r'\caption{INSERT-CAPTION}',
            r'\label{table:INSERT-LABEL}',
            r'\end{table}',
        ]
    )

    return '\n'.join(results_table_str) + '\n'
