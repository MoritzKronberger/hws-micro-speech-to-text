"""Format benchmark results as a pretty string."""
import statistics
from prettytable import PrettyTable
from app.quality_benchmark.result_types import full_results
from app.utils import get_hash_comment


def prettify_results(results: list[full_results]) -> str:
    total_word_error_count = 0
    total_samples = 0
    error_count_distribution = [0, 0, 0, 0, 0]
    audio_duration_distribution = [0, 0, 0, 0, 0]

    for result in results:
        word_error_count = result["word_error_count"]
        total_word_error_count += sum(word_error_count)
        total_samples += len(word_error_count)

        audio_durations = result["audio_durations"]

        for error_count, audio_duration in zip(word_error_count, audio_durations):
            if error_count < 1:
                error_count_distribution[0] += 1
                audio_duration_distribution[0] += audio_duration
            elif error_count < 2:
                error_count_distribution[1] += 1
                audio_duration_distribution[1] += audio_duration
            elif error_count < 3:
                error_count_distribution[2] += 1
                audio_duration_distribution[2] += audio_duration
            elif error_count < 4:
                error_count_distribution[3] += 1
                audio_duration_distribution[3] += audio_duration
            else:
                error_count_distribution[4] += 1
                audio_duration_distribution[4] += audio_duration

    average_word_error_count = total_word_error_count / total_samples
    average_audio_duration = [audio_duration / count if count > 0 else 0 for audio_duration, count in zip(audio_duration_distribution, error_count_distribution)]

    sample_percentage = [round(count / total_samples * 100, 2) for count in error_count_distribution]

    pretty_string = f"Benchmark Results - LJ Speech\n\n"
    pretty_string += f"Average Word Error Count: {average_word_error_count:.2f}\n"
    pretty_string += "Average Audio Duration:\n"
    ranges = ["0-1", "1-2", "2-3", "3-4", "4 =<"]
    for i, duration in enumerate(average_audio_duration):
        pretty_string += f"- {ranges[i]}: {duration:.2f} seconds\n"
    pretty_string += "\nWord Error Count Distribution:\n"
    for i, count in enumerate(error_count_distribution):
        pretty_string += f"- {ranges[i]}: {sample_percentage[i]}% of samples\n"

    return pretty_string










