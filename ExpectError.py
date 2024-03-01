#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from types import FrameType, TracebackType
from typing import Union, Optional, Callable, Any
import traceback
import sys
import sys

import sys, importlib
from pathlib import Path


def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]
    sys.path.append(str(top))
    # try:
    #     sys.path.remove(str(parent))
    # except ValueError:  # already removed
    #     pass
    __package__ = ".".join(parent.parts[len(top.parts) :])
    importlib.import_module(__package__)  # won't be needed after that


import_parents(level=1)

from .Timeout import Timeout
import bookutils.setup


class ExpectError:
    """Execute a code block expecting (and catching) an error."""

    def __init__(self, exc_type: Optional[type] = None, print_traceback: bool = True, mute: bool = False):
        """
        Constructor. Expect an exception of type `exc_type` (`None`: any exception).
        If `print_traceback` is set (default), print a traceback to stderr.
        If `mute` is set (default: False), do not print anything.
        """
        self.print_traceback = print_traceback
        self.mute = mute
        self.expected_exc_type = exc_type

    def __enter__(self) -> Any:
        """Begin of `with` block"""
        return self

    def __exit__(self, exc_type: type, exc_value: BaseException, tb: TracebackType) -> Optional[bool]:
        """End of `with` block"""
        if exc_type is None:
            # No exception
            return

        if self.expected_exc_type is not None and exc_type != self.expected_exc_type:
            raise  # Unexpected exception

        # An exception occurred
        if self.print_traceback:
            lines = "".join(traceback.format_exception(exc_type, exc_value, tb)).strip()
        else:
            lines = traceback.format_exception_only(exc_type, exc_value)[-1].strip()

        if not self.mute:
            print(lines, file=sys.stderr)
        return True  # Ignore it


class ExpectTimeout(Timeout):  # type: ignore
    """Execute a code block expecting (and catching) a timeout."""

    def __init__(self, timeout: Union[int, float], print_traceback: bool = True, mute: bool = False):
        """
        Constructor. Interrupt execution after `seconds` seconds.
        If `print_traceback` is set (default), print a traceback to stderr.
        If `mute` is set (default: False), do not print anything.
        """
        super().__init__(timeout)

        self.print_traceback = print_traceback
        self.mute = mute

    def __exit__(self, exc_type: type, exc_value: BaseException, tb: TracebackType) -> Optional[bool]:
        """End of `with` block"""

        super().__exit__(exc_type, exc_value, tb)

        if exc_type is None:
            return

        # An exception occurred
        if self.print_traceback:
            lines = "".join(traceback.format_exception(exc_type, exc_value, tb)).strip()
        else:
            lines = traceback.format_exception_only(exc_type, exc_value)[-1].strip()

        if not self.mute:
            print(lines, "(expected)", file=sys.stderr)

        return True  # Ignore exception


if __name__ == "__main__":

    def fail_test() -> None:
        # Trigger an exception
        x = 1 / 0

    with ExpectError(ZeroDivisionError):
        fail_test()

    import time

    def long_running_test() -> None:
        print("Start")
        for i in range(10):
            time.sleep(1)
            print(i, "seconds have passed")
        print("End")

    with ExpectTimeout(5):
        long_running_test()
