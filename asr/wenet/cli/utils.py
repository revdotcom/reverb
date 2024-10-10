from typing import Any, Generator


def hyps_to_ctm(
    audio_name: str,
    path: list[dict[str, Any]]
) -> Generator[str, None, None]:
    """Convert a given set of decode results for a single audio file into CTM lines."""
    for line in path:
        start_seconds = line['start_time_ms'] / 1000
        duration_seconds = line['end_time_ms'] / 1000 - start_seconds
        ctm_line = f"{audio_name} 0 {start_seconds:.2f} {duration_seconds:.2f} {line['word']} {line['confidence']:.2f}"
        yield ctm_line


def hyps_to_txt(
    path: list[dict[str, Any]]
) -> Generator[str, None, None]:
    """Convert a given set of decode results for a single audio file into CTM lines."""
    for line in path:
        yield line['word']
