import time
from typing import Type, Any
from types import TracebackType


def clock() -> float:
    """
    Return the number of fractional seconds elapsed since some point of reference.
    """
    return time.perf_counter()


class Timer:
    def __init__(self) -> None:
        """Constructor"""
        self.start_time = clock()
        self.end_time = None

    def __enter__(self) -> Any:
        """Begin of `with` block"""
        self.start_time = clock()
        self.end_time = None
        return self

    def __exit__(self, exc_type: Type, exc_value: BaseException, tb: TracebackType) -> None:
        """End of `with` block"""
        self.end_time = clock()  # type: ignore

    def elapsed_time(self) -> float:
        """Return elapsed time in seconds"""
        if self.end_time is None:
            # still running
            return clock() - self.start_time
        else:
            return self.end_time - self.start_time  # type: ignore
