#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Set, List, TypeVar, Union
from types import FrameType, TracebackType
import inspect
import sys, importlib
from pathlib import Path
import html
import math
import random
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import graphviz

# from IPython.display import Markdown
from rich.markdown import Markdown
from rich.console import Console


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


from .Tracer import Tracer
from .StackInspector import StackInspector
from .bookutils import getsourcelines  # like inspect.getsourcelines(), but in color
from .ExpectError import ExpectError

Coverage = Set[Tuple[Callable, int]]


class Collector(Tracer):
    def __init__(self) -> None:
        """Constructor."""
        self._function: Optional[Callable] = None
        self._args: Optional[Dict[str, Any]] = None
        self._argstring: Optional[str] = None
        self._exception: Optional[Type] = None
        self.items_to_ignore: List[Union[Type, Callable]] = [self.__class__]

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Tracing function.
        Saves the first function and calls collect().
        """
        for item in self.items_to_ignore:
            if isinstance(item, type) and "self" in frame.f_locals and isinstance(frame.f_locals["self"], item):
                # Ignore this class
                return
            if item.__name__ == frame.f_code.co_name:
                # Ignore this function
                return

        if self._function is None and event == "call":
            # Save function
            self._function = self.create_function(frame)
            self._args = frame.f_locals.copy()
            self._argstring = ", ".join([f"{var}={repr(self._args[var])}" for var in self._args])

        self.collect(frame, event, arg)

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """Collector function. To be overloaded in subclasses."""
        pass

    def events(self) -> Set:
        """Return a collection of events. To be overridden in subclasses."""
        return set()

    def id(self) -> str:
        """Return an identifier for the collector,
        created from the first call"""
        return f"{self.function().__name__}({self.argstring()})"

    def function(self) -> Callable:
        """Return the function from the first call, as a function object"""
        if not self._function:
            raise ValueError("No call collected")
        return self._function

    def argstring(self) -> str:
        """
        Return the list of arguments from the first call,
        as a printable string
        """
        if not self._argstring:
            raise ValueError("No call collected")
        return self._argstring

    def args(self) -> Dict[str, Any]:
        """Return a dict of argument names and values from the first call"""
        if not self._args:
            raise ValueError("No call collected")
        return self._args

    def exception(self) -> Optional[Type]:
        """Return the exception class from the first call,
        or None if no exception was raised."""
        return self._exception

    def __repr__(self) -> str:
        """Return a string representation of the collector"""
        # We use the ID as default representation when printed
        return self.id()

    def covered_functions(self) -> Set[Callable]:
        """Set of covered functions. To be overloaded in subclasses."""
        return set()

    def coverage(self) -> Coverage:
        """
        Return a set (function, lineno) with locations covered.
        To be overloaded in subclasses.
        """
        return set()

    def add_items_to_ignore(self, items_to_ignore: List[Union[Type, Callable]]) -> None:
        """
        Define additional classes and functions to ignore during collection
        (typically `Debugger` classes using these collectors).
        """
        self.items_to_ignore += items_to_ignore

    def __exit__(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> Optional[bool]:
        """Exit the `with` block."""
        ret = super().__exit__(exc_tp, exc_value, exc_traceback)
        if not self._function:
            if exc_tp:
                return False  # re-raise exception
            else:
                raise ValueError("No call collected")
        return ret


class CoverageCollector(Collector, StackInspector):
    """A class to record covered locations during execution."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self._coverage: Coverage = set()

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Save coverage for an observed event.
        """
        name = frame.f_code.co_name
        function = self.search_func(name, frame)

        if function is None:
            function = self.create_function(frame)

        location = (function, frame.f_lineno)
        self._coverage.add(location)

    def events(self) -> Set[Tuple[str, int]]:
        """
        Return the set of locations covered.
        Each location comes as a pair (`function_name`, `lineno`).
        """
        return {(func.__name__, lineno) for func, lineno in self._coverage}

    def covered_functions(self) -> Set[Callable]:
        """Return a set with all functions covered."""
        return {func for func, lineno in self._coverage}

    def coverage(self) -> Coverage:
        """Return a set (function, lineno) with all locations covered."""
        return self._coverage


class StatisticalDebugger:
    """A class to collect events for multiple outcomes."""

    def __init__(self, collector_class: Type = CoverageCollector, log: bool = False):
        """Constructor. Use instances of `collector_class` to collect events."""
        self.collector_class = collector_class
        self.collectors: Dict[str, List[Collector]] = {}
        self.log = log

    def collect(self, outcome: str, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for the given outcome.
        Additional args are passed to the collector."""
        collector = self.collector_class(*args, **kwargs)
        collector.add_items_to_ignore([self.__class__])
        return self.add_collector(outcome, collector)

    def add_collector(self, outcome: str, collector: Collector) -> Collector:
        if outcome not in self.collectors:
            self.collectors[outcome] = []
        self.collectors[outcome].append(collector)
        return collector

    def all_events(self, outcome: Optional[str] = None) -> Set[Any]:
        """Return a set of all events observed."""
        all_events = set()

        if outcome:
            if outcome in self.collectors:
                for collector in self.collectors[outcome]:
                    all_events.update(collector.events())
        else:
            for outcome in self.collectors:
                for collector in self.collectors[outcome]:
                    all_events.update(collector.events())

        return all_events

    def function(self) -> Optional[Callable]:
        """
        Return the entry function from the events observed,
        or None if ambiguous.
        """
        names_seen = set()
        functions = []
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                # We may have multiple copies of the function,
                # but sharing the same name
                func = collector.function()
                if func.__name__ not in names_seen:
                    functions.append(func)
                    names_seen.add(func.__name__)

        if len(functions) != 1:
            return None  # ambiguous
        return functions[0]

    def covered_functions(self) -> Set[Callable]:
        """Return a set of all functions observed."""
        functions = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                functions |= collector.covered_functions()
        return functions

    def coverage(self) -> Coverage:
        """Return a set of all (functions, line_numbers) observed"""
        coverage = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                coverage |= collector.coverage()
        return coverage

    def color(self, event: Any) -> Optional[str]:
        """
        Return a color for the given event, or None.
        To be overloaded in subclasses.
        """
        return None

    def tooltip(self, event: Any) -> Optional[str]:
        """
        Return a tooltip string for the given event, or None.
        To be overloaded in subclasses.
        """
        return None

    def event_str(self, event: Any) -> str:
        """Format the given event. To be overloaded in subclasses."""
        if isinstance(event, str):
            return event
        if isinstance(event, tuple):
            return ":".join(self.event_str(elem) for elem in event)
        return str(event)

    def event_table_text(self, *, args: bool = False, color: bool = False) -> str:
        """
        Print out a table of events observed.
        If `args` is True, use arguments as headers.
        If `color` is True, use colors.
        """
        sep = " | "
        all_events = self.all_events()
        longest_event = max(len(f"{self.event_str(event)}") for event in all_events)
        out = ""

        # Header
        if args:
            out += "| "
            func = self.function()
            if func:
                out += "`" + func.__name__ + "`"
            out += sep
            for name in self.collectors:
                for collector in self.collectors[name]:
                    out += "`" + collector.argstring() + "`" + sep
            out += "\n"
        else:
            out += "| " + " " * longest_event + sep
            for name in self.collectors:
                for i in range(len(self.collectors[name])):
                    out += name + sep
            out += "\n"

        out += "| " + "-" * longest_event + sep
        for name in self.collectors:
            for i in range(len(self.collectors[name])):
                out += "-" * len(name) + sep
        out += "\n"

        # Data
        for event in sorted(all_events):
            event_name = self.event_str(event).rjust(longest_event)

            tooltip = self.tooltip(event)
            if tooltip:
                title = f' title="{tooltip}"'
            else:
                title = ""

            if color:
                color_name = self.color(event)
                if color_name:
                    event_name = (
                        f'<samp style="background-color: {color_name}"{title}>' f"{html.escape(event_name)}" f"</samp>"
                    )

            out += f"| {event_name}" + sep
            for name in self.collectors:
                for collector in self.collectors[name]:
                    out += " " * (len(name) - 1)
                    if event in collector.events():
                        out += "X"
                    else:
                        out += "-"
                    out += sep
            out += "\n"

        return out

    def event_table(self, **_args: Any) -> Any:
        """Print out event table in Markdown format."""
        return Console().print(Markdown(self.event_table_text(**_args)))

    def __repr__(self) -> str:
        return self.event_table_text()

    def _repr_markdown_(self) -> str:
        return self.event_table_text(args=True, color=True)


class DifferenceDebugger(StatisticalDebugger):
    """A class to collect events for passing and failing outcomes."""

    PASS = "PASS"
    FAIL = "FAIL"

    def collect_pass(self, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for passing runs."""
        return self.collect(self.PASS, *args, **kwargs)

    def collect_fail(self, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for failing runs."""
        return self.collect(self.FAIL, *args, **kwargs)

    def pass_collectors(self) -> List[Collector]:
        return self.collectors[self.PASS]

    def fail_collectors(self) -> List[Collector]:
        return self.collectors[self.FAIL]

    def all_fail_events(self) -> Set[Any]:
        """Return all events observed in failing runs."""
        return self.all_events(self.FAIL)

    def all_pass_events(self) -> Set[Any]:
        """Return all events observed in passing runs."""
        return self.all_events(self.PASS)

    def only_fail_events(self) -> Set[Any]:
        """Return all events observed only in failing runs."""
        return self.all_fail_events() - self.all_pass_events()

    def only_pass_events(self) -> Set[Any]:
        """Return all events observed only in passing runs."""
        return self.all_pass_events() - self.all_fail_events()

    def __enter__(self) -> Any:
        """Enter a `with` block. Collect coverage and outcome;
        classify as FAIL if the block raises an exception,
        and PASS if it does not.
        """
        self.collector = self.collector_class()
        self.collector.add_items_to_ignore([self.__class__])
        self.collector.__enter__()
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> Optional[bool]:
        """Exit the `with` block."""
        status = self.collector.__exit__(exc_tp, exc_value, exc_traceback)

        if status is None:
            pass
        else:
            return False  # Internal error; re-raise exception

        if exc_tp is None:
            outcome = self.PASS
        else:
            outcome = self.FAIL

        self.add_collector(outcome, self.collector)
        return True  # Ignore exception, if any


class SpectrumDebugger(DifferenceDebugger):
    def suspiciousness(self, event: Any) -> Optional[float]:
        """
        Return a suspiciousness value in the range [0, 1.0]
        for the given event, or `None` if unknown.
        To be overloaded in subclasses.
        """
        return None

    def tooltip(self, event: Any) -> str:
        """
        Return a tooltip for the given event (default: percentage).
        To be overloaded in subclasses.
        """
        return self.percentage(event)

    def percentage(self, event: Any) -> str:
        """
        Return the suspiciousness for the given event as percentage string.
        """
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is not None:
            return str(int(suspiciousness * 100)).rjust(3) + "%"
        else:
            return " " * len("100%")

    def code(
        self,
        functions: Optional[Set[Callable]] = None,
        *,
        color: bool = False,
        suspiciousness: bool = False,
        line_numbers: bool = True,
    ) -> str:
        """
        Return a listing of `functions` (default: covered functions).
        If `color` is True, render as HTML, using suspiciousness colors.
        If `suspiciousness` is True, include suspiciousness values.
        If `line_numbers` is True (default), include line numbers.
        """

        if not functions:
            functions = self.covered_functions()

        out = ""
        seen = set()
        for function in functions:
            source_lines, starting_line_number = inspect.getsourcelines(function)

            if (function.__name__, starting_line_number) in seen:
                continue
            seen.add((function.__name__, starting_line_number))

            if out:
                out += "\n"
                if color:
                    out += "<p/>"

            line_number = starting_line_number
            for line in source_lines:
                if color:
                    line = html.escape(line)
                    if line.strip() == "":
                        line = "&nbsp;"

                location = (function.__name__, line_number)
                location_suspiciousness = self.suspiciousness(location)
                if location_suspiciousness is not None:
                    tooltip = f"Line {line_number}: {self.tooltip(location)}"
                else:
                    tooltip = f"Line {line_number}: not executed"
                if suspiciousness:
                    line = self.percentage(location) + " " + line
                if line_numbers:
                    line = str(line_number).rjust(4) + " " + line
                line_color = self.color(location)

                if color and line_color:
                    line = f"""<pre style="background-color:{line_color}"
                    title="{tooltip}">{line.rstrip()}</pre>"""
                elif color:
                    line = f'<pre title="{tooltip}">{line}</pre>'
                else:
                    line = line.rstrip()
                out += line + "\n"
                line_number += 1

        return out

    def _repr_html_(self) -> str:
        """When output in Jupyter, visualize as HTML"""
        return self.code(color=True)

    def __str__(self) -> str:
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

    def __repr__(self) -> str:
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

    def save_html(self, file_path):
        with open(file_path, "w") as fp:
            fp.write(self.code(color=True))


class DiscreteSpectrumDebugger(SpectrumDebugger):
    """Visualize differences between executions using three discrete colors"""

    def suspiciousness(self, event: Any) -> Optional[float]:
        """
        Return a suspiciousness value [0, 1.0]
        for the given event, or `None` if unknown.
        """
        passing = self.all_pass_events()
        failing = self.all_fail_events()
        if event in passing and event in failing:
            return 0.5
        elif event in failing:
            return 1.0
        elif event in passing:
            return 0.0
        else:
            return None

    def color(self, event: Any) -> Optional[str]:
        """
        Return a HTML color for the given event.
        """
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None
        if suspiciousness > 0.8:
            return "mistyrose"
        if suspiciousness >= 0.5:
            return "lightyellow"
        return "honeydew"

    def tooltip(self, event: Any) -> str:
        """Return a tooltip for the given event."""
        passing = self.all_pass_events()
        failing = self.all_fail_events()
        if event in passing and event in failing:
            return "in passing and failing runs"
        elif event in failing:
            return "only in failing runs"
        elif event in passing:
            return "only in passing runs"
        else:
            return "never"


class ContinuousSpectrumDebugger(DiscreteSpectrumDebugger):
    """Visualize differences between executions using a color spectrum"""

    def collectors_with_event(self, event: Any, category: str) -> Set[Collector]:
        """
        Return all collectors in a category
        that observed the given event.
        """
        all_runs = self.collectors[category]
        collectors_with_event = set(collector for collector in all_runs if event in collector.events())
        return collectors_with_event

    def collectors_without_event(self, event: Any, category: str) -> Set[Collector]:
        """
        Return all collectors in a category
        that did not observe the given event.
        """
        all_runs = self.collectors[category]
        collectors_without_event = set(collector for collector in all_runs if event not in collector.events())
        return collectors_without_event

    def event_fraction(self, event: Any, category: str) -> float:
        if category not in self.collectors:
            return 0.0

        all_collectors = self.collectors[category]
        collectors_with_event = self.collectors_with_event(event, category)
        fraction = len(collectors_with_event) / len(all_collectors)
        # print(f"%{category}({event}) = {fraction}")
        return fraction

    def passed_fraction(self, event: Any) -> float:
        return self.event_fraction(event, self.PASS)

    def failed_fraction(self, event: Any) -> float:
        return self.event_fraction(event, self.FAIL)

    def hue(self, event: Any) -> Optional[float]:
        """Return a color hue from 0.0 (red) to 1.0 (green)."""
        passed = self.passed_fraction(event)
        failed = self.failed_fraction(event)
        if passed + failed > 0:
            return passed / (passed + failed)
        else:
            return None

    def suspiciousness(self, event: Any) -> Optional[float]:
        hue = self.hue(event)
        if hue is None:
            return None
        return 1 - hue

    def tooltip(self, event: Any) -> str:
        return self.percentage(event)

    def brightness(self, event: Any) -> float:
        return max(self.passed_fraction(event), self.failed_fraction(event))

    def color(self, event: Any) -> Optional[str]:
        hue = self.hue(event)
        if hue is None:
            return None
        saturation = self.brightness(event)

        # HSL color values are specified with:
        # hsl(hue, saturation, lightness).
        return f"hsl({hue * 120}, {saturation * 100}%, 80%)"


class RankingDebugger(DiscreteSpectrumDebugger):
    """Rank events by their suspiciousness"""

    def rank(self) -> List[Any]:
        """Return a list of events, sorted by suspiciousness, highest first."""

        def susp(event: Any) -> float:
            suspiciousness = self.suspiciousness(event)
            assert suspiciousness is not None
            return suspiciousness

        events = list(self.all_events())
        events.sort(key=susp, reverse=True)
        return events

    def __repr__(self) -> str:
        return repr(self.rank())


class TarantulaDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    """Spectrum-based Debugger using the Tarantula metric for suspiciousness"""

    pass


class OchiaiDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    """Spectrum-based Debugger using the Ochiai metric for suspiciousness"""

    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = len(self.collectors_with_event(event, self.FAIL))
        not_in_failed = len(self.collectors_without_event(event, self.FAIL))
        passed = len(self.collectors_with_event(event, self.PASS))

        try:
            return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
        except ZeroDivisionError:
            return None

    def hue(self, event: Any) -> Optional[float]:
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None
        return 1 - suspiciousness


class ValueCollector(Collector):
    """ "A class to collect local variables and their values."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.vars: Set[str] = set()

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        local_vars = frame.f_locals
        for var in local_vars:
            value = local_vars[var]
            self.vars.add(f"{var} = {repr(value)}")

    def events(self) -> Set[str]:
        """A set of (variable, value) pairs observed"""
        return self.vars


class ClassifyingDebugger(DifferenceDebugger):
    """A debugger implementing a decision tree for events"""

    PASS_VALUE = +1.0
    FAIL_VALUE = -1.0

    def samples(self) -> Dict[str, float]:
        samples = {}
        for collector in self.pass_collectors():
            samples[collector.id()] = self.PASS_VALUE
        for collector in debugger.fail_collectors():
            samples[collector.id()] = self.FAIL_VALUE
        return samples

    def features(self) -> Dict[str, Any]:
        features = {}
        for collector in debugger.pass_collectors():
            features[collector.id()] = collector.events()
        for collector in debugger.fail_collectors():
            features[collector.id()] = collector.events()
        return features

    def feature_names(self) -> List[str]:
        return [repr(feature) for feature in self.all_events()]

    def shape(self, sample: str) -> List[float]:
        x = []
        features = self.features()
        for f in self.all_events():
            if f in features[sample]:
                x += [+1.0]
            else:
                x += [-1.0]
        return x

    def X(self) -> List[List[float]]:
        X = []
        samples = self.samples()
        for key in samples:
            X += [self.shape(key)]
        return X

    def Y(self) -> List[float]:
        Y = []
        samples = self.samples()
        for key in samples:
            Y += [samples[key]]
        return Y

    def classifier(self) -> DecisionTreeClassifier:
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(self.X(), self.Y())
        return classifier

    def show_classifier(self, classifier: DecisionTreeClassifier) -> Any:
        dot_data = export_graphviz(
            classifier,
            out_file=None,
            filled=False,
            rounded=True,
            feature_names=self.feature_names(),
            class_names=["FAIL", "PASS"],
            label="none",
            node_ids=False,
            impurity=False,
            proportion=True,
            special_characters=True,
        )

        return graphviz.Source(dot_data)


def code_with_coverage(function: Callable, coverage: Coverage) -> None:
    def _check(function, line_number, coverage):
        e = next(iter(coverage))
        if isinstance(e[0], str):
            return (function.__name__, line_number) in coverage
        else:
            return (function, line_number) in coverage

    source_lines, starting_line_number = getsourcelines(function)

    line_number = starting_line_number
    for line in source_lines:
        marker = "*" if _check(function, line_number, coverage) else " "
        print(f"{line_number:4} {marker} {line}", end="")
        line_number += 1


if __name__ == "__main__":
    from pprint import pprint
    from .test_func import remove_html_markup, middle

    with CoverageCollector() as c:
        remove_html_markup('"abc"')
    code_with_coverage(remove_html_markup, c.coverage())

    s = StatisticalDebugger()
    with s.collect("PASS"):
        remove_html_markup("abc")
    with s.collect("PASS"):
        remove_html_markup("<b>abc</b>")
    with s.collect("FAIL"):
        remove_html_markup('"abc"')

    pprint(s.all_events())
    pprint(s.all_events("FAIL"))

    s.event_table(args=True)

    def test_debugger_html(debugger: DifferenceDebugger) -> DifferenceDebugger:
        with debugger:
            remove_html_markup("abc")
        with debugger:
            remove_html_markup("<b>abc</b>")
        with debugger:
            remove_html_markup('"abc"')
            assert False  # Mark test as failing
        return debugger

    debugger = test_debugger_html(DifferenceDebugger())
    debugger.event_table(args=True)

    code_with_coverage(remove_html_markup, debugger.only_fail_events())

    debugger = test_debugger_html(DiscreteSpectrumDebugger())
    print(debugger)
    debugger.save_html("testd.html")
    debugger = test_debugger_html(ContinuousSpectrumDebugger())
    debugger.save_html("testc1.html")
    with debugger.collect_pass():
        out = remove_html_markup('<b link="blue"></b>')
    debugger.save_html("testc2.html")

    def test_debugger_middle(debugger: DifferenceDebugger) -> DifferenceDebugger:
        with debugger.collect_pass():
            middle(3, 3, 5)
        with debugger.collect_pass():
            middle(1, 2, 3)
        with debugger.collect_pass():
            middle(3, 2, 1)
        with debugger.collect_pass():
            middle(5, 5, 5)
        with debugger.collect_pass():
            middle(5, 3, 4)
        with debugger.collect_fail():
            middle(2, 1, 3)
        return debugger

    debugger = test_debugger_middle(ContinuousSpectrumDebugger())
    debugger.event_table(args=True)
    debugger.save_html("testc_middle.html")

    tarantula_middle = test_debugger_middle(TarantulaDebugger())
    pprint(tarantula_middle.rank())
    print(tarantula_middle.suspiciousness(tarantula_middle.rank()[0]))

    def middle_testcase() -> Tuple[int, int, int]:
        x = random.randrange(10)
        y = random.randrange(10)
        z = random.randrange(10)
        return x, y, z

    def middle_test(x: int, y: int, z: int) -> None:
        m = middle(x, y, z)
        assert m == sorted([x, y, z])[1]

    def middle_passing_testcase() -> Tuple[int, int, int]:
        while True:
            try:
                x, y, z = middle_testcase()
                middle_test(x, y, z)
                return x, y, z
            except AssertionError:
                pass

    def middle_failing_testcase() -> Tuple[int, int, int]:
        while True:
            try:
                x, y, z = middle_testcase()
                middle_test(x, y, z)
            except AssertionError:
                return x, y, z

    ochiai_middle = OchiaiDebugger()

    MIDDLE_TESTS = 100
    MIDDLE_PASSING_TESTCASES = [middle_passing_testcase() for i in range(MIDDLE_TESTS)]
    MIDDLE_FAILING_TESTCASES = [middle_failing_testcase() for i in range(MIDDLE_TESTS)]
    for x, y, z in MIDDLE_PASSING_TESTCASES:
        with ochiai_middle.collect_pass():
            middle(x, y, z)
    for x, y, z in MIDDLE_FAILING_TESTCASES:
        with ochiai_middle.collect_fail():
            middle(x, y, z)
    ochiai_middle.save_html("testc_middle_large_testcases.html")

    debugger = test_debugger_html(ContinuousSpectrumDebugger(ValueCollector))
    debugger.event_table(color=True, args=True)
    with open("test_colored.md", "w") as fp:
        fp.write(debugger._repr_markdown_())
