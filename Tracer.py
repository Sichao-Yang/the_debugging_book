from types import FrameType, TracebackType
from typing import Any, Optional, Callable, Dict, List, Type, TextIO, cast
import inspect
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

from .bookutils import print_content
from .StackInspector import StackInspector


class Tracer(StackInspector):
    """A class for tracing a piece of code. Use as `with Tracer(): block()`"""

    def __init__(self, file: TextIO = sys.stdout) -> None:
        """
        Create a new tracer.
        If `file` is given, output to `file` instead of stdout.
        """
        self.last_vars: Dict[str, Any] = {}
        self.original_trace_function: Optional[Callable] = None
        self.file = file

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        """Internal tracing function."""
        if self.our_frame(frame):
            # Do not trace our own methods
            pass
        else:
            self.traceit(frame, event, arg)
        return self._traceit

    def log(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = True) -> None:
        """
        Like `print()`, but always sending to `file` given at initialization,
        and flushing by default.
        """
        print(*objects, sep=sep, end=end, file=self.file, flush=flush)

    def __enter__(self) -> Any:
        """Called at begin of `with` block. Turn tracing on."""
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)

        # This extra line also enables tracing for the current block
        # inspect.currentframe().f_back.f_trace = self._traceit
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> Optional[bool]:
        """
        Called at end of `with` block. Turn tracing off.
        Return `None` if ok, not `None` if internal error.
        """
        sys.settrace(self.original_trace_function)

        # Note: we must return a non-True value here,
        # such that we re-raise all exceptions
        if self.is_internal_error(exc_tp, exc_value, exc_traceback):
            return False  # internal error
        else:
            return None  # all ok

    def changed_vars(self, new_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Track changed variables, based on `new_vars` observed."""
        changed = {}
        for var_name, var_value in new_vars.items():
            if var_name not in self.last_vars or self.last_vars[var_name] != var_value:
                changed[var_name] = var_value
        self.last_vars = new_vars.copy()
        return changed

    def print_debugger_status(self, frame: FrameType, event: str, arg: Any) -> None:
        """Show current source line and changed vars"""
        changes = self.changed_vars(frame.f_locals)
        changes_s = ", ".join([var + " = " + repr(changes[var]) for var in changes])

        if event == "call":
            self.log("Calling " + frame.f_code.co_name + "(" + changes_s + ")")
        elif changes:
            self.log(" " * 40, "#", changes_s)

        if event == "line":
            try:
                module = inspect.getmodule(frame.f_code)
                if module is None:
                    source = inspect.getsource(frame.f_code)
                else:
                    source = inspect.getsource(module)
                current_line = source.split("\n")[frame.f_lineno - 1]

            except OSError as err:
                self.log(f"{err.__class__.__name__}: {err}")
                current_line = ""

            self.log(repr(frame.f_lineno) + " " + current_line)

        if event == "return":
            self.log(frame.f_code.co_name + "()" + " returns " + repr(arg))
            self.last_vars = {}  # Delete 'last' variables

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.print_debugger_status(frame, event, arg)


class ConditionalTracer(Tracer):
    def __init__(self, *, condition: Optional[str] = None, file: TextIO = sys.stdout) -> None:
        """Constructor. Trace all events for which `condition` (a Python expr) holds."""

        if condition is None:
            condition = "False"

        self.condition: str = condition
        self.last_report: Optional[bool] = None
        super().__init__(file=file)

    def eval_in_context(self, expr: str, frame: FrameType) -> Optional[bool]:
        try:
            cond = eval(expr, None, frame.f_locals)
        except NameError:  # (yet) undefined variable
            cond = None
        return cond

    def do_report(self, frame: FrameType, event: str, arg: Any) -> Optional[bool]:
        return self.eval_in_context(self.condition, frame)

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        report = self.do_report(frame, event, arg)
        if report != self.last_report:
            if report:
                self.log("...")
            self.last_report = report

        if report:
            self.print_debugger_status(frame, event, arg)


class EventTracer(ConditionalTracer):
    """Log when a given event expression changes its value"""

    def __init__(self, *, condition: Optional[str] = None, events: List[str] = [], file: TextIO = sys.stdout) -> None:
        """Constructor. `events` is a list of expressions to watch."""
        self.events = events
        self.last_event_values: Dict[str, Any] = {}
        super().__init__(file=file, condition=condition)

    def events_changed(self, events: List[str], frame: FrameType) -> bool:
        """Return True if any of the observed `events` has changed"""
        change = False
        for event in events:
            value = self.eval_in_context(event, frame)

            if event not in self.last_event_values or value != self.last_event_values[event]:
                self.last_event_values[event] = value
                change = True

        return change

    def do_report(self, frame: FrameType, event: str, arg: Any) -> bool:
        """Return True if a line should be shown"""
        return self.eval_in_context(self.condition, frame) or self.events_changed(self.events, frame)


if __name__ == "__main__":
    from .demo_func import remove_html_markup

    with Tracer():
        remove_html_markup("abc")
