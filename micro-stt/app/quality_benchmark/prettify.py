"""Format benchmark results as a pretty string."""
import statistics
from prettytable import PrettyTable
from app.quality_benchmark.result_types import full_results
from app.utils import get_hash_comment


def prettify_results(results: list[full_results]) -> str:
    total_word_error_count = 0
    total_samples = 0
    error_count_distribution = [0, 0, 0, 0, 0]

    for result in results:
        word_error_count = result["word_error_count"]
        total_word_error_count += sum(word_error_count)
        total_samples += len(word_error_count)

        for error_count in word_error_count:
            if error_count < 1:
                error_count_distribution[0] += 1
            elif error_count < 2:
                error_count_distribution[1] += 1
            elif error_count < 3:
                error_count_distribution[2] += 1
            elif error_count < 4:
                error_count_distribution[3] += 1
            else:
                error_count_distribution[4] += 1

    average_word_error_count = total_word_error_count / total_samples
    sample_percentage = [round(count / total_samples * 100, 2) for count in error_count_distribution]

    pretty_string = f"Benchmark Results - LJ Speech\n\n"
    pretty_string += f"Average Word Error Count: {average_word_error_count:.2f}\n\n"
    pretty_string += "Word Error Count Distribution:\n"

    ranges = ["0-1", "1-2", "2-3", "3-4", "4 =<"]
    for i, count in enumerate(error_count_distribution):
        pretty_string += f"- {ranges[i]}: {sample_percentage[i]}% of samples\n"

    return pretty_string








