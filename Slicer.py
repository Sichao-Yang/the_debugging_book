#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Set, List, Tuple, Any, Callable, Dict, Optional
from typing import Union, Type, Generator, cast
import inspect
import warnings
from graphviz import Digraph
import html
import sys, importlib
from pathlib import Path
from ast import NodeTransformer, NodeVisitor, Name, AST
import typing
import ast
import itertools
import copy
import re
from ast import (
    Module,
    Load,
    Store,
    Attribute,
    With,
    withitem,
    keyword,
    Call,
    Expr,
    Assign,
    AugAssign,
    AnnAssign,
    Assert,
)
from types import TracebackType


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

from .bookutils import next_inputs, print_content
from .StackInspector import StackInspector
from .demo_func import remove_html_markup


Location = Tuple[Callable, int]
Node = Tuple[str, Location]
Dependency = Dict[Node, Set[Node]]
Criterion = Union[str, Location, Node]
DATA_TRACKER = "_data"


class Dependencies(StackInspector):
    """A dependency graph"""

    NODE_COLOR = "peachpuff"
    FONT_NAME = "Courier"  # 'Fira Mono' may produce warnings in 'dot'

    def __init__(self, data: Optional[Dependency] = None, control: Optional[Dependency] = None) -> None:
        """
        Create a dependency graph from `data` and `control`.
        Both `data` and `control` are dictionaries
        holding _nodes_ as keys and sets of nodes as values.
        Each node comes as a tuple (variable_name, location)
        where `variable_name` is a string
        and `location` is a pair (function, lineno)
        where `function` is a callable and `lineno` is a line number
        denoting a unique location in the code.
        """

        if data is None:
            data = {}
        if control is None:
            control = {}

        self.data = data
        self.control = control

        for var in self.data:
            self.control.setdefault(var, set())
        for var in self.control:
            self.data.setdefault(var, set())

        self.validate()

    def validate(self) -> None:
        """Check dependency structure."""
        assert isinstance(self.data, dict)
        assert isinstance(self.control, dict)

        for node in set(self.data.keys()) | set(self.control.keys()):
            var_name, location = node
            assert isinstance(var_name, str)
            func, lineno = location
            assert callable(func)
            assert isinstance(lineno, int)

    def _source(self, node: Node) -> str:
        # Return source line, or ''
        (name, location) = node
        func, lineno = location
        if not func:  # type: ignore
            # No source
            return ""

        try:
            source_lines, first_lineno = inspect.getsourcelines(func)
        except OSError:
            warnings.warn(f"Couldn't find source " f"for {func} ({func.__name__})")
            return ""

        try:
            line = source_lines[lineno - first_lineno].strip()
        except IndexError:
            return ""

        return line

    def source(self, node: Node) -> str:
        """Return the source code for a given node."""
        line = self._source(node)
        if line:
            return line

        (name, location) = node
        func, lineno = location
        code_name = func.__name__

        if code_name.startswith("<"):
            return code_name
        else:
            return f"<{code_name}()>"

    def make_graph(self, name: str = "dependencies", comment: str = "Dependencies") -> Digraph:
        return Digraph(
            name=name,
            comment=comment,
            graph_attr={},
            node_attr={"style": "filled", "shape": "box", "fillcolor": self.NODE_COLOR, "fontname": self.FONT_NAME},
            edge_attr={"fontname": self.FONT_NAME},
        )

    def graph(self, *, mode: str = "flow") -> Digraph:
        """
        Draw dependencies. `mode` is either
        * `'flow'`: arrows indicate information flow (from A to B); or
        * `'depend'`: arrows indicate dependencies (B depends on A)
        """
        self.validate()

        g = self.make_graph()
        self.draw_dependencies(g, mode)
        self.add_hierarchy(g)
        return g

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """If the object is output in Jupyter, render dependencies as a SVG graph"""
        return self.graph()._repr_mimebundle_(include, exclude)

    def all_vars(self) -> Set[Node]:
        """Return a set of all variables (as `var_name`, `location`) in the dependencies"""
        all_vars = set()
        for var in self.data:
            all_vars.add(var)
            for source in self.data[var]:
                all_vars.add(source)

        for var in self.control:
            all_vars.add(var)
            for source in self.control[var]:
                all_vars.add(source)

        return all_vars

    def draw_edge(self, g: Digraph, mode: str, node_from: str, node_to: str, **kwargs: Any) -> None:
        if mode == "flow":
            g.edge(node_from, node_to, **kwargs)
        elif mode == "depend":
            g.edge(node_from, node_to, dir="back", **kwargs)
        else:
            raise ValueError("`mode` must be 'flow' or 'depend'")

    def draw_dependencies(self, g: Digraph, mode: str) -> None:
        for var in self.all_vars():
            g.node(self.id(var), label=self.label(var), tooltip=self.tooltip(var))

            if var in self.data:
                for source in self.data[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var))

            if var in self.control:
                for source in self.control[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var), style="dashed", color="grey")

    def id(self, var: Node) -> str:
        """Return a unique ID for `var`."""
        id = ""
        # Avoid non-identifier characters
        for c in repr(var):
            if c.isalnum() or c == "_":
                id += c
            if c == ":" or c == ",":
                id += "_"
        return id

    def label(self, var: Node) -> str:
        """Render node `var` using HTML style."""
        (name, location) = var
        source = self.source(var)

        title = html.escape(name)
        if name.startswith("<"):
            title = f"<I>{title}</I>"

        label = f"<B>{title}</B>"
        if source:
            label += f'<FONT POINT-SIZE="9.0"><BR/><BR/>' f"{html.escape(source)}" f"</FONT>"
        label = f"<{label}>"
        return label

    def tooltip(self, var: Node) -> str:
        """Return a tooltip for node `var`."""
        (name, location) = var
        func, lineno = location
        return f"{func.__name__}:{lineno}"

    def add_hierarchy(self, g: Digraph) -> Digraph:
        """Add invisible edges for a proper hierarchy."""
        functions = self.all_functions()
        for func in functions:
            last_var = None
            last_lineno = 0
            for lineno, var in functions[func]:
                if last_var is not None and lineno > last_lineno:
                    g.edge(self.id(last_var), self.id(var), style="invis")

                last_var = var
                last_lineno = lineno

        return g

    def all_functions(self) -> Dict[Callable, List[Tuple[int, Node]]]:
        """
        Return mapping
        {`function`: [(`lineno`, `var`), (`lineno`, `var`), ...], ...}
        for all functions in the dependencies.
        """
        functions: Dict[Callable, List[Tuple[int, Node]]] = {}
        for var in self.all_vars():
            (name, location) = var
            func, lineno = location
            if func not in functions:
                functions[func] = []
            functions[func].append((lineno, var))

        for func in functions:
            functions[func].sort()

        return functions

    def expand_criteria(self, criteria: List[Criterion]) -> List[Node]:
        """Return list of vars matched by `criteria`."""
        all_vars = []
        for criterion in criteria:
            criterion_var = None
            criterion_func = None
            criterion_lineno = None

            if isinstance(criterion, str):
                criterion_var = criterion
            elif len(criterion) == 2 and callable(criterion[0]):
                criterion_func, criterion_lineno = criterion
            elif len(criterion) == 2 and isinstance(criterion[0], str):
                criterion_var = criterion[0]
                criterion_func, criterion_lineno = criterion[1]
            else:
                raise ValueError("Invalid argument")

            for var in self.all_vars():
                (var_name, location) = var
                func, lineno = location

                name_matches = (
                    criterion_func is None or criterion_func == func or criterion_func.__name__ == func.__name__
                )

                location_matches = criterion_lineno is None or criterion_lineno == lineno

                var_matches = criterion_var is None or criterion_var == var_name

                if name_matches and location_matches and var_matches:
                    all_vars.append(var)

        return all_vars

    def backward_slice(self, *criteria: Criterion, mode: str = "cd", depth: int = -1):
        """
        Create a backward slice from nodes `criteria`.
        `mode` can contain 'c' (draw control dependencies)
        and 'd' (draw data dependencies) (default: 'cd')
        """
        data = {}
        control = {}
        queue = self.expand_criteria(criteria)  # type: ignore
        seen = set()

        while len(queue) > 0 and depth != 0:
            var = queue[0]
            queue = queue[1:]
            seen.add(var)

            if "d" in mode:
                # Follow data dependencies
                data[var] = self.data[var]
                for next_var in data[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                data[var] = set()

            if "c" in mode:
                # Follow control dependencies
                control[var] = self.control[var]
                for next_var in control[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                control[var] = set()

            depth -= 1

        return Dependencies(data, control)

    def format_var(self, var: Node, current_func: Optional[Callable] = None) -> str:
        """Return string for `var` in `current_func`."""
        name, location = var
        func, lineno = location
        if current_func and (func == current_func or func.__name__ == current_func.__name__):
            return f"{name} ({lineno})"
        else:
            return f"{name} ({func.__name__}:{lineno})"

    def __str__(self) -> str:
        """Return string representation of dependencies"""
        self.validate()

        out = ""
        for func in self.all_functions():
            code_name = func.__name__

            if out != "":
                out += "\n"
            out += f"{code_name}():\n"

            all_vars = list(set(self.data.keys()) | set(self.control.keys()))
            all_vars.sort(key=lambda var: var[1][1])

            for var in all_vars:
                (name, location) = var
                var_func, var_lineno = location
                var_code_name = var_func.__name__

                if var_code_name != code_name:
                    continue

                all_deps = ""
                for source, arrow in [(self.data, "<="), (self.control, "<-")]:
                    deps = ""
                    for data_dep in source[var]:
                        if deps == "":
                            deps = f" {arrow} "
                        else:
                            deps += ", "
                        deps += self.format_var(data_dep, func)

                    if deps != "":
                        if all_deps != "":
                            all_deps += ";"
                        all_deps += deps

                if all_deps == "":
                    continue

                out += "    " + self.format_var(var, func) + all_deps + "\n"

        return out

    def repr_var(self, var: Node) -> str:
        name, location = var
        func, lineno = location
        return f"({repr(name)}, ({func.__name__}, {lineno}))"

    def repr_deps(self, var_set: Set[Node]) -> str:
        if len(var_set) == 0:
            return "set()"

        return "{" + ", ".join(f"{self.repr_var(var)}" for var in var_set) + "}"

    def repr_dependencies(self, vars: Dependency) -> str:
        return (
            "{\n        "
            + ",\n        ".join(f"{self.repr_var(var)}: {self.repr_deps(vars[var])}" for var in vars)
            + "}"
        )

    def __repr__(self) -> str:
        """Represent dependencies as a Python expression"""
        # Useful for saving and restoring values
        return (
            f"Dependencies(\n"
            + f"    data={self.repr_dependencies(self.data)},\n"
            + f" control={self.repr_dependencies(self.control)})"
        )

    def code(self, *items: Callable, mode: str = "cd") -> None:
        """
        List `items` on standard output, including dependencies as comments.
        If `items` is empty, all included functions are listed.
        `mode` can contain 'c' (draw control dependencies) and 'd' (draw data dependencies)
        (default: 'cd').
        """

        if len(items) == 0:
            items = cast(Tuple[Callable], self.all_functions().keys())

        for i, item in enumerate(items):
            if i > 0:
                print()
            self._code(item, mode)

    def _code(self, item: Callable, mode: str) -> None:
        # The functions in dependencies may be (instrumented) copies
        # of the original function. Find the function with the same name.
        func = item
        for fn in self.all_functions():
            if fn == item or fn.__name__ == item.__name__:
                func = fn
                break

        all_vars = self.all_vars()
        slice_locations = set(location for (name, location) in all_vars)

        source_lines, first_lineno = inspect.getsourcelines(func)

        n = first_lineno
        for line in source_lines:
            line_location = (func, n)
            if line_location in slice_locations:
                prefix = "* "
            else:
                prefix = "  "

            print(f"{prefix}{n:4} ", end="")

            comment = ""
            for mode_control, source, arrow in [("d", self.data, "<="), ("c", self.control, "<-")]:
                if mode_control not in mode:
                    continue

                deps = ""
                for var in source:
                    name, location = var
                    if location == line_location:
                        for dep_var in source[var]:
                            if deps == "":
                                deps = arrow + " "
                            else:
                                deps += ", "
                            deps += self.format_var(dep_var, item)

                if deps != "":
                    if comment != "":
                        comment += "; "
                    comment += deps

            if comment != "":
                line = line.rstrip() + "  # " + comment

            print_content(line.rstrip(), ".py")
            print()
            n += 1


class Dependencies(Dependencies):
    def validate(self) -> None:
        """Perform a simple syntactic validation of dependencies"""
        super().validate()

        for var in self.all_vars():
            source = self.source(var)
            if not source:
                continue
            if source.startswith("<"):
                continue  # no source

            for dep_var in self.data[var] | self.control[var]:
                dep_name, dep_location = dep_var

                if dep_name == DependencyTracker.TEST:
                    continue  # dependency on <test>

                if dep_name.endswith(" value>"):
                    if source.find("(") < 0:
                        warnings.warn(
                            f"Warning: {self.format_var(var)} "
                            f"depends on {self.format_var(dep_var)}, "
                            f"but {repr(source)} does not "
                            f"seem to have a call"
                        )
                    continue

                if source.startswith("def"):
                    continue  # function call

                rx = re.compile(r"\b" + dep_name + r"\b")
                if rx.search(source) is None:
                    warnings.warn(
                        f"{self.format_var(var)} "
                        f"depends on {self.format_var(dep_var)}, "
                        f"but {repr(dep_name)} does not occur "
                        f"in {repr(source)}"
                    )


class DataTracker(StackInspector):
    """Track data accesses during execution"""

    def __init__(self, log: bool = False) -> None:
        """Constructor. If `log` is set, turn on logging."""
        self.log = log

    def set(self, name: str, value: Any, loads: Optional[Set[str]] = None) -> Any:
        """Track setting `name` to `value`."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: setting {name}")

        return value

    def get(self, name: str, value: Any) -> Any:
        """Track getting `value` from `name`."""

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: getting {name}")

        return value

    def augment(self, name: str, value: Any) -> Any:
        """
        Track augmenting `name` with `value`.
        To be overloaded in subclasses.
        """
        self.set(name, self.get(name, value))
        return value

    def test(self, cond: AST) -> AST:
        """Test condition `cond`. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: testing condition")

        return cond

    def __enter__(self) -> Any:
        """Enter `with` block. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: entering block")
        return self

    def __exit__(self, exc_type: Type, exc_value: BaseException, traceback: TracebackType) -> Optional[bool]:
        """Exit `with` block. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: exiting block")
        return None

    def arg(self, value: Any, pos: Optional[int] = None, kw: Optional[str] = None) -> Any:
        """
        Track `value` being passed as argument.
        `pos` (if given) is the argument position (starting with 1).
        `kw` (if given) is the argument keyword.
        """

        if self.log:
            caller_func, lineno = self.caller_location()
            info = ""
            if pos:
                info += f" #{pos}"
            if kw:
                info += f" {repr(kw)}"

            print(f"{caller_func.__name__}:{lineno}: pushing arg{info}")

        return value

    def ret(self, value: Any) -> Any:
        """Track `value` being used as return value."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: returned from call")

        return value

    def instrument_call(self, func: Callable) -> Callable:
        """Instrument a call to `func`. To be implemented in subclasses."""
        return func

    def call(self, func: Callable) -> Callable:
        """Track a call to `func`."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: calling {func}")

        return self.instrument_call(func)

    def param(self, name: str, value: Any, pos: Optional[int] = None, vararg: str = "", last: bool = False) -> Any:
        """
        At the beginning of a function, track parameter `name` being set to `value`.
        `pos` is the position of the argument (starting with 1).
        `vararg` is "*" if `name` is a vararg parameter (as in *args),
        and "**" is `name` is a kwargs parameter (as in *kwargs).
        `last` is True if `name` is the last parameter.
        """
        if self.log:
            caller_func, lineno = self.caller_location()
            info = ""
            if pos is not None:
                info += f" #{pos}"

            print(f"{caller_func.__name__}:{lineno}: initializing {vararg}{name}{info}")

        return self.set(name, value)


def is_internal(id: str) -> bool:
    """Return True if `id` is a built-in function or type"""
    return id in dir(__builtins__) or id in dir(typing)


class TrackGetTransformer(NodeTransformer):
    def visit_Name(self, node: Name) -> AST:
        self.generic_visit(node)

        if is_internal(node.id):
            # Do not change built-in names and types
            return node

        if node.id == DATA_TRACKER:
            # Do not change own accesses
            return node

        if not isinstance(node.ctx, Load):
            # Only change loads (not stores, not deletions)
            return node

        new_node = make_get_data(node.id)
        ast.copy_location(new_node, node)
        return new_node


def make_get_data(id: str, method: str = "get") -> Call:
    return Call(
        func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), attr=method, ctx=Load()),
        args=[ast.Str(s=id), Name(id=id, ctx=Load())],
        keywords=[],
    )


def dump_tree(tree: AST) -> None:
    print_content(ast.unparse(tree), ".py")
    ast.fix_missing_locations(tree)  # Must run this before compiling
    _ = compile(cast(ast.Module, tree), "<dump_tree>", "exec")


class DataTrackerTester:
    def __init__(self, tree: AST, func: Callable, log: bool = True) -> None:
        """Constructor. Execute the code in `tree` while instrumenting `func`."""
        # We pass the source file of `func` such that we can retrieve it
        # when accessing the location of the new compiled code
        source = cast(str, inspect.getsourcefile(func))
        self.code = compile(cast(ast.Module, tree), source, "exec")
        self.func = func
        self.log = log

    def make_data_tracker(self) -> Any:
        return DataTracker(log=self.log)

    def __enter__(self) -> Any:
        """Rewrite function"""
        tracker = self.make_data_tracker()
        globals()[DATA_TRACKER] = tracker
        exec(self.code, globals())
        return tracker

    def __exit__(self, exc_type: Type, exc_value: BaseException, traceback: TracebackType) -> Optional[bool]:
        """Restore function"""
        globals()[self.func.__name__] = self.func
        del globals()[DATA_TRACKER]
        return None


def make_set_data(id: str, value: Any, loads: Optional[Set[str]] = None, method: str = "set") -> Call:
    """
    Construct a subtree _data.`method`('`id`', `value`).
    If `loads` is set to [X1, X2, ...], make it
    _data.`method`('`id`', `value`, loads=(X1, X2, ...))
    """

    keywords = []

    if loads:
        keywords = [
            keyword(arg="loads", value=ast.Tuple(elts=[Name(id=load, ctx=Load()) for load in loads], ctx=Load()))
        ]

    new_node = Call(
        func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), attr=method, ctx=Load()),
        args=[ast.Str(s=id), value],
        keywords=keywords,
    )

    ast.copy_location(new_node, value)

    return new_node


class LeftmostNameVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.leftmost_name: Optional[str] = None

    def visit_Name(self, node: Name) -> None:
        if self.leftmost_name is None:
            self.leftmost_name = node.id
        self.generic_visit(node)


def leftmost_name(tree: AST) -> Optional[str]:
    visitor = LeftmostNameVisitor()
    visitor.visit(tree)
    return visitor.leftmost_name


class StoreVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()

    def visit(self, node: AST) -> None:
        if hasattr(node, "ctx") and isinstance(node.ctx, Store):  # type: ignore
            name = leftmost_name(node)
            if name:
                self.names.add(name)

        self.generic_visit(node)


def store_names(tree: AST) -> Set[str]:
    visitor = StoreVisitor()
    visitor.visit(tree)
    return visitor.names


class LoadVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()

    def visit(self, node: AST) -> None:
        if hasattr(node, "ctx") and isinstance(node.ctx, Load):  # type: ignore
            name = leftmost_name(node)
            if name is not None:
                self.names.add(name)

        self.generic_visit(node)


def load_names(tree: AST) -> Set[str]:
    visitor = LoadVisitor()
    visitor.visit(tree)
    return visitor.names


class TrackSetTransformer(NodeTransformer):
    def visit_Assign(self, node: Assign) -> Assign:
        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER + ".set"):
            return node  # Do not apply twice

        for target in node.targets:
            loads = load_names(target)
            for store_name in store_names(target):
                node.value = make_set_data(store_name, node.value, loads=loads)
                loads = set()

        return node

    def visit_AugAssign(self, node: AugAssign) -> AugAssign:
        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER):
            return node  # Do not apply twice

        id = cast(str, leftmost_name(node.target))
        node.value = make_set_data(id, node.value, method="augment")

        return node

    def visit_AnnAssign(self, node: AnnAssign) -> AnnAssign:
        if node.value is None:
            return node  # just <var>: <type> without value

        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER + ".set"):
            return node  # Do not apply twice

        loads = load_names(node.target)
        for store_name in store_names(node.target):
            node.value = make_set_data(store_name, node.value, loads=loads)
            loads = set()

        return node

    def visit_Assert(self, node: Assert) -> Assert:
        value = ast.unparse(node.test)
        if value.startswith(DATA_TRACKER + ".set"):
            return node  # Do not apply twice

        loads = load_names(node.test)
        node.test = make_set_data("<assertion>", node.test, loads=loads)
        return node


class TrackReturnTransformer(NodeTransformer):

    def __init__(self) -> None:
        self.function_name: Optional[str] = None
        super().__init__()

    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> AST:
        outer_name = self.function_name
        self.function_name = node.name  # Save current name
        self.generic_visit(node)
        self.function_name = outer_name
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> AST:
        return self.visit_FunctionDef(node)

    def return_value(self, tp: str = "return") -> str:
        if self.function_name is None:
            return f"<{tp} value>"
        else:
            return f"<{self.function_name}() {tp} value>"

    def visit_return_or_yield(self, node: Union[ast.Return, ast.Yield, ast.YieldFrom], tp: str = "return") -> AST:

        if node.value is not None:
            value = ast.unparse(node.value)
            if not value.startswith(DATA_TRACKER + ".set"):
                node.value = make_set_data(self.return_value(tp), node.value)

        return node

    def visit_Return(self, node: ast.Return) -> AST:
        return self.visit_return_or_yield(node, tp="return")

    def visit_Yield(self, node: ast.Yield) -> AST:
        return self.visit_return_or_yield(node, tp="yield")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> AST:
        return self.visit_return_or_yield(node, tp="yield")


class TrackControlTransformer(NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

    def make_with(self, block: List[ast.stmt]) -> List[ast.stmt]:
        """Create a subtree 'with _data: `block`'"""
        if len(block) == 0:
            return []

        block_as_text = ast.unparse(block[0])
        if block_as_text.startswith("with " + DATA_TRACKER):
            return block  # Do not apply twice

        new_node = With(
            items=[withitem(context_expr=Name(id=DATA_TRACKER, ctx=Load()), optional_vars=None)], body=block
        )
        ast.copy_location(new_node, block[0])
        return [new_node]

    def make_test(self, test: ast.expr) -> ast.expr:
        test_as_text = ast.unparse(test)
        if test_as_text.startswith(DATA_TRACKER + ".test"):
            return test  # Do not apply twice

        new_test = Call(
            func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), attr="test", ctx=Load()), args=[test], keywords=[]
        )
        ast.copy_location(new_test, test)
        return new_test

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

    # regular `for` loop
    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> AST:
        self.generic_visit(node)
        id = ast.unparse(node.target).strip()
        node.iter = make_set_data(id, node.iter)

        # Uncomment if you want iterators to control their bodies
        # node.body = self.make_with(node.body)
        # node.orelse = self.make_with(node.orelse)
        return node

    # `for` loops in async functions
    def visit_AsyncFor(self, node: ast.AsyncFor) -> AST:
        return self.visit_For(node)

    # `for` clause in comprehensions
    def visit_comprehension(self, node: ast.comprehension) -> AST:
        self.generic_visit(node)
        id = ast.unparse(node.target).strip()
        node.iter = make_set_data(id, node.iter)
        return node


class TrackCallTransformer(NodeTransformer):
    def make_call(self, node: AST, func: str, pos: Optional[int] = None, kw: Optional[str] = None) -> Call:
        """Return _data.call(`func`)(`node`)"""
        keywords = []

        # `Num()` and `Str()` are deprecated in favor of `Constant()`
        if pos:
            keywords.append(keyword(arg="pos", value=ast.Num(pos)))
        if kw:
            keywords.append(keyword(arg="kw", value=ast.Str(kw)))

        return Call(
            func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), attr=func, ctx=Load()),
            args=[node],
            keywords=keywords,
        )

    def visit_Call(self, node: Call) -> Call:
        self.generic_visit(node)

        call_as_text = ast.unparse(node)
        if call_as_text.startswith(DATA_TRACKER + ".ret"):
            return node  # Already applied

        func_as_text = ast.unparse(node)
        if func_as_text.startswith(DATA_TRACKER + "."):
            return node  # Own function

        new_args = []
        for n, arg in enumerate(node.args):
            new_args.append(self.make_call(arg, "arg", pos=n + 1))
        node.args = cast(List[ast.expr], new_args)

        for kw in node.keywords:
            id = kw.arg if hasattr(kw, "arg") else None
            kw.value = self.make_call(kw.value, "arg", kw=id)

        node.func = self.make_call(node.func, "call")
        return self.make_call(node, "ret")


class TrackParamsTransformer(NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)

        named_args = []
        for child in ast.iter_child_nodes(node.args):
            if isinstance(child, ast.arg):
                named_args.append(child)

        create_stmts = []
        for n, child in enumerate(named_args):
            keywords = [keyword(arg="pos", value=ast.Num(n=n + 1))]
            if child is node.args.vararg:
                keywords.append(keyword(arg="vararg", value=ast.Str(s="*")))
            if child is node.args.kwarg:
                keywords.append(keyword(arg="vararg", value=ast.Str(s="**")))
            if n == len(named_args) - 1:
                keywords.append(keyword(arg="last", value=ast.NameConstant(value=True)))

            create_stmt = Expr(
                value=Call(
                    func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), attr="param", ctx=Load()),
                    args=[ast.Str(s=child.arg), Name(id=child.arg, ctx=Load())],
                    keywords=keywords,
                )
            )
            ast.copy_location(create_stmt, node)
            create_stmts.append(create_stmt)

        node.body = cast(List[ast.stmt], create_stmts) + node.body
        return node


class DependencyTracker(DataTracker):
    """Track dependencies during execution"""

    TEST = "<test>"  # Name of pseudo-variables for testing conditions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor. Arguments are passed to DataTracker.__init__()"""
        super().__init__(*args, **kwargs)

        self.origins: Dict[str, Location] = {}  # Where current variables were last set
        self.data_dependencies: Dependency = {}  # As with Dependencies, above
        self.control_dependencies: Dependency = {}

        self.last_read: List[str] = []  # List of last read variables
        self.last_checked_location = (StackInspector.unknown, 1)
        self._ignore_location_change = False

        self.data: List[List[str]] = [[]]  # Data stack
        self.control: List[List[str]] = [[]]  # Control stack

        self.frames: List[Dict[Union[int, str], Any]] = [{}]  # Argument stack
        self.args: Dict[Union[int, str], Any] = {}  # Current args

    def get(self, name: str, value: Any) -> Any:
        """Track a read access for variable `name` with value `value`"""
        self.check_location()
        self.last_read.append(name)
        return super().get(name, value)

    def clear_read(self) -> None:
        """Clear set of read variables"""
        if self.log:
            direct_caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore
            caller_func, lineno = self.caller_location()
            print(
                f"{caller_func.__name__}:{lineno}: "
                f"clearing read variables {self.last_read} "
                f"(from {direct_caller})"
            )

        self.last_read = []

    def check_location(self) -> None:
        """If we are in a new location, clear set of read variables"""
        location = self.caller_location()
        func, lineno = location
        last_func, last_lineno = self.last_checked_location

        if self.last_checked_location != location:
            if self._ignore_location_change:
                self._ignore_location_change = False
            elif func.__name__.startswith("<"):
                # Entering list comprehension, eval(), exec(), ...
                pass
            elif last_func.__name__.startswith("<"):
                # Exiting list comprehension, eval(), exec(), ...
                pass
            else:
                # Standard case
                self.clear_read()

        self.last_checked_location = location

    def ignore_next_location_change(self) -> None:
        self._ignore_location_change = True

    def ignore_location_change(self) -> None:
        self.last_checked_location = self.caller_location()

    def set(self, name: str, value: Any, loads: Optional[Set[str]] = None) -> Any:
        """Add a dependency for `name` = `value`"""

        def add_dependencies(dependencies: Set[Node], vars_read: List[str], tp: str) -> None:
            """Add origins of `vars_read` to `dependencies`."""
            for var_read in vars_read:
                if var_read in self.origins:
                    if var_read == self.TEST and tp == "data":
                        # Can't have data dependencies on conditions
                        continue

                    origin = self.origins[var_read]
                    dependencies.add((var_read, origin))

                    if self.log:
                        origin_func, origin_lineno = origin
                        caller_func, lineno = self.caller_location()
                        print(
                            f"{caller_func.__name__}:{lineno}: "
                            f"new {tp} dependency: "
                            f"{name} <= {var_read} "
                            f"({origin_func.__name__}:{origin_lineno})"
                        )

        self.check_location()
        ret = super().set(name, value)
        location = self.caller_location()

        add_dependencies(self.data_dependencies.setdefault((name, location), set()), self.last_read, tp="data")
        add_dependencies(
            self.control_dependencies.setdefault((name, location), set()),
            cast(List[str], itertools.chain.from_iterable(self.control)),
            tp="control",
        )

        self.origins[name] = location

        # Reset read info for next line
        self.last_read = [name]

        # Next line is a new location
        self._ignore_location_change = False

        return ret

    def dependencies(self) -> Dependencies:
        """Return dependencies"""
        return Dependencies(self.data_dependencies, self.control_dependencies)

    def test(self, value: Any) -> Any:
        """Track a test for condition `value`"""
        self.set(self.TEST, value)
        return super().test(value)

    def __enter__(self) -> Any:
        """Track entering an if/while/for block"""
        self.control.append(self.last_read)
        self.clear_read()
        return super().__enter__()

    def __exit__(self, exc_type: Type, exc_value: BaseException, traceback: TracebackType) -> Optional[bool]:
        """Track exiting an if/while/for block"""
        self.clear_read()
        self.last_read = self.control.pop()
        self.ignore_next_location_change()
        return super().__exit__(exc_type, exc_value, traceback)

    def call(self, func: Callable) -> Callable:
        """Track a call of function `func`"""
        func = super().call(func)

        if inspect.isgeneratorfunction(func):
            return self.call_generator(func)

        # Save context
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: " f"saving read variables {self.last_read}")

        self.data.append(self.last_read)
        self.clear_read()
        self.ignore_next_location_change()

        self.frames.append(self.args)
        self.args = {}

        return func

    def ret(self, value: Any) -> Any:
        """Track a function return"""
        value = super().ret(value)

        if self.in_generator():
            return self.ret_generator(value)

        # Restore old context and add return value
        ret_name = None
        for var in self.last_read:
            if var.startswith("<"):  # "<return value>"
                ret_name = var

        self.last_read = self.data.pop()
        if ret_name:
            self.last_read.append(ret_name)

        if self.args:
            # We return from an uninstrumented function:
            # Make return value depend on all args
            for key, deps in self.args.items():
                self.last_read += deps

        self.ignore_location_change()

        self.args = self.frames.pop()

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: " f"restored read variables {self.last_read}")

        return value

    def in_generator(self) -> bool:
        """True if we are calling a generator function"""
        return len(self.data) > 0 and self.data[-1] is None

    def call_generator(self, func: Callable) -> Callable:
        """Track a call of a generator function"""
        # Mark the fact that we're in a generator with `None` values
        self.data.append(None)  # type: ignore
        self.frames.append(None)  # type: ignore
        assert self.in_generator()

        self.clear_read()
        return func

    def ret_generator(self, generator: Any) -> Any:
        """Track the return of a generator function"""
        # Pop the two 'None' values pushed earlier
        self.data.pop()
        self.frames.pop()

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: " f"wrapping generator {generator} (args={self.args})")

        # At this point, we already have collected the args.
        # The returned generator depends on all of them.
        for arg in self.args:
            self.last_read += self.args[arg]

        # Wrap the generator such that the args are restored
        # when it is actually invoked, such that we can map them
        # to parameters.
        saved_args = copy.deepcopy(self.args)

        def wrapper() -> Generator[Any, None, None]:
            self.args = saved_args
            if self.log:
                caller_func, lineno = self.caller_location()
                print(f"{caller_func.__name__}:{lineno}: " f"calling generator (args={self.args})")

            self.ignore_next_location_change()
            yield from generator

        return wrapper()

    def arg(self, value: Any, pos: Optional[int] = None, kw: Optional[str] = None) -> Any:
        """
        Track passing an argument `value`
        (with given position `pos` 1..n or keyword `kw`)
        """
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: " f"saving args read {self.last_read}")

        if pos:
            self.args[pos] = self.last_read
        if kw:
            self.args[kw] = self.last_read

        self.clear_read()
        return super().arg(value, pos, kw)

    def param(self, name: str, value: Any, pos: Optional[int] = None, vararg: str = "", last: bool = False) -> Any:
        """
        Track getting a parameter `name` with value `value`
        (with given position `pos`).
        vararg parameters are indicated by setting `varargs` to
        '*' (*args) or '**' (**kwargs)
        """
        self.clear_read()

        if vararg == "*":
            # We over-approximate by setting `args` to _all_ positional args
            for index in self.args:
                if isinstance(index, int) and pos is not None and index >= pos:
                    self.last_read += self.args[index]
        elif vararg == "**":
            # We over-approximate by setting `kwargs` to _all_ passed keyword args
            for index in self.args:
                if isinstance(index, str):
                    self.last_read += self.args[index]
        elif name in self.args:
            self.last_read = self.args[name]
        elif pos in self.args:
            self.last_read = self.args[pos]

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: " f"restored params read {self.last_read}")

        self.ignore_location_change()
        ret = super().param(name, value, pos)

        if last:
            self.clear_read()
            self.args = {}  # Mark `args` as processed

        return ret


def call_test() -> int:
    c = 47

    def sq(n: int) -> int:
        return n * n

    def gen(e: int) -> Generator[int, None, None]:
        yield e * c

    def just_x(x: Any, y: Any) -> Any:
        return x

    a = 42
    b = gen(a)
    d = list(b)[0]

    xs = [1, 2, 3, 4]
    ys = [sq(elem) for elem in xs if elem > 2]

    return just_x(just_x(d, y=b), ys[0])


class DependencyTrackerTester(DataTrackerTester):
    def make_data_tracker(self):
        return DependencyTracker(log=self.log)


class Instrumenter(StackInspector):
    """Instrument functions for dynamic tracking"""

    def __init__(
        self, *items_to_instrument: Callable, globals: Optional[Dict[str, Any]] = None, log: Union[bool, int] = False
    ) -> None:
        """
        Create an instrumenter.
        `items_to_instrument` is a list of items to instrument.
        `globals` is a namespace to use (default: caller's globals())
        """

        self.log = log
        self.items_to_instrument: List[Callable] = list(items_to_instrument)
        self.instrumented_items: Set[Any] = set()

        if globals is None:
            globals = self.caller_globals()
        self.globals = globals

    def __enter__(self) -> Any:
        """Instrument sources"""
        items = self.items_to_instrument
        if not items:
            items = self.default_items_to_instrument()

        for item in items:
            self.instrument(item)

        return self

    def default_items_to_instrument(self) -> List[Callable]:
        return []

    def instrument(self, item: Any) -> Any:
        """Instrument `item`. To be overloaded in subclasses."""
        if self.log:
            print("Instrumenting", item)
        self.instrumented_items.add(item)
        return item

    def __exit__(self, exc_type: Type, exc_value: BaseException, traceback: TracebackType) -> Optional[bool]:
        """Restore sources"""
        self.restore()
        return None

    def restore(self) -> None:
        for item in self.instrumented_items:
            self.globals[item.__name__] = item


class WithVisitor(NodeVisitor):
    def __init__(self) -> None:
        self.withs: List[ast.With] = []

    def visit_With(self, node: ast.With) -> AST:
        self.withs.append(node)
        return self.generic_visit(node)


class CallCollector(NodeVisitor):
    def __init__(self) -> None:
        self.calls: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> AST:
        caller_id = ast.unparse(node.func).strip()
        self.calls.add(caller_id)
        return self.generic_visit(node)


class Slicer(Instrumenter):
    """Track dependencies in an execution"""

    def __init__(
        self,
        *items_to_instrument: Any,
        dependency_tracker: Optional[DependencyTracker] = None,
        globals: Optional[Dict[str, Any]] = None,
        log: Union[bool, int] = False,
    ):
        """Create a slicer.
        `items_to_instrument` are Python functions or modules with source code.
        `dependency_tracker` is the tracker to be used (default: DependencyTracker).
        `globals` is the namespace to be used(default: caller's `globals()`)
        `log`=True or `log` > 0 turns on logging
        """
        super().__init__(*items_to_instrument, globals=globals, log=log)

        if dependency_tracker is None:
            dependency_tracker = DependencyTracker(log=(log > 1))
        self.dependency_tracker = dependency_tracker

        self.saved_dependencies = None

    def default_items_to_instrument(self) -> List[Callable]:
        raise ValueError("Need one or more items to instrument")

    def parse(self, item: Any) -> AST:
        """Parse `item`, returning its AST"""
        source_lines, lineno = inspect.getsourcelines(item)
        source = "".join(source_lines)

        if self.log >= 2:
            print_content(source, ".py", start_line_number=lineno)
            print()
            print()

        tree = ast.parse(source)
        ast.increment_lineno(tree, lineno - 1)
        return tree

    def transformers(self) -> List[NodeTransformer]:
        """List of transformers to apply. To be extended in subclasses."""
        return [
            TrackCallTransformer(),
            TrackSetTransformer(),
            TrackGetTransformer(),
            TrackControlTransformer(),
            TrackReturnTransformer(),
            TrackParamsTransformer(),
        ]

    def transform(self, tree: AST) -> AST:
        """Apply transformers on `tree`. May be extended in subclasses."""
        # Apply transformers
        for transformer in self.transformers():
            if self.log >= 3:
                print(transformer.__class__.__name__ + ":")

            transformer.visit(tree)
            ast.fix_missing_locations(tree)
            if self.log >= 3:
                print_content(ast.unparse(tree), ".py")
                print()
                print()

        if 0 < self.log < 3:
            print_content(ast.unparse(tree), ".py")
            print()
            print()

        return tree

    def execute(self, tree: AST, item: Any) -> None:
        """Compile and execute `tree`. May be extended in subclasses."""

        # We pass the source file of `item` such that we can retrieve it
        # when accessing the location of the new compiled code
        source = cast(str, inspect.getsourcefile(item))
        code = compile(cast(ast.Module, tree), source, "exec")

        # Enable dependency tracker
        self.globals[DATA_TRACKER] = self.dependency_tracker

        # Execute the code, resulting in a redefinition of item
        exec(code, self.globals)

    def instrument(self, item: Any) -> Any:
        """Instrument `item`, transforming its source code, and re-defining it."""
        if is_internal(item.__name__):
            return item  # Do not instrument `print()` and the like

        if inspect.isbuiltin(item):
            return item  # No source code

        item = super().instrument(item)
        tree = self.parse(item)
        tree = self.transform(tree)
        self.execute(tree, item)

        new_item = self.globals[item.__name__]
        return new_item

    def restore(self) -> None:
        """Restore original code."""
        if DATA_TRACKER in self.globals:
            self.saved_dependencies = self.globals[DATA_TRACKER]
            del self.globals[DATA_TRACKER]

        super().restore()

    def dependencies(self) -> Dependencies:
        """Return collected dependencies."""
        if self.saved_dependencies is None:
            return Dependencies({}, {})
        return self.saved_dependencies.dependencies()

    def code(self, *args: Any, **kwargs: Any) -> None:
        """Show code of instrumented items, annotated with dependencies."""
        first = True
        for item in self.instrumented_items:
            if not first:
                print()
            self.dependencies().code(item, *args, **kwargs)  # type: ignore
            first = False

    def graph(self, *args: Any, **kwargs: Any) -> Digraph:
        """Show dependency graph."""
        return self.dependencies().graph(*args, **kwargs)  # type: ignore

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """If the object is output in Jupyter, render dependencies as a SVG graph"""
        return self.graph()._repr_mimebundle_(include, exclude)

    def our_with_block(self) -> ast.With:
        """Return the currently active `with` block."""
        frame = self.caller_frame()
        source_lines, starting_lineno = inspect.getsourcelines(frame)
        starting_lineno = max(starting_lineno, 1)
        if len(source_lines) == 1:
            # We only get one `with` line, rather than the full block
            # This happens in Jupyter notebooks with iPython 8.1.0 and later.
            # Here's a hacky workaround to get the cell contents:
            # https://stackoverflow.com/questions/51566497/getting-the-source-of-an-object-defined-in-a-jupyter-notebook
            source_lines = inspect.linecache.getlines(inspect.getfile(frame))  # type: ignore
            starting_lineno = 1

        source_ast = ast.parse("".join(source_lines))
        wv = WithVisitor()
        wv.visit(source_ast)

        for with_ast in wv.withs:
            if starting_lineno + (with_ast.lineno - 1) == frame.f_lineno:
                return with_ast

        raise ValueError("Cannot find 'with' block")

    def calls_in_our_with_block(self) -> Set[str]:
        """Return a set of function names called in the `with` block."""
        block_ast = self.our_with_block()
        cc = CallCollector()
        for stmt in block_ast.body:
            cc.visit(stmt)
        return cc.calls

    def funcs_in_our_with_block(self) -> List[Callable]:
        funcs = []
        for id in self.calls_in_our_with_block():
            func = self.search_func(id)
            if func:
                funcs.append(func)

        return funcs

    def default_items_to_instrument(self) -> List[Callable]:
        # In _data.call(), return instrumented function
        self.dependency_tracker.instrument_call = self.instrument  # type: ignore

        # Start instrumenting the functions in our `with` block
        return self.funcs_in_our_with_block()


def fun_1(x: int) -> int:
    return x


def fun_2(x: int) -> int:
    return fun_1(x)


if __name__ == "__main__":
    from demo_func import middle, call_test

    middle_tree = ast.parse(inspect.getsource(middle))
    TrackGetTransformer().visit(middle_tree)
    TrackSetTransformer().visit(middle_tree)
    TrackReturnTransformer().visit(middle_tree)
    TrackControlTransformer().visit(middle_tree)
    TrackCallTransformer().visit(middle_tree)
    TrackParamsTransformer().visit(middle_tree)
    dump_tree(middle_tree)
    print()
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)

    _test_data = DependencyTracker()
    x = _test_data.set("x", 1)
    y = _test_data.set("y", _test_data.get("x", x))

    if _test_data.test(_test_data.get("x", x) >= _test_data.get("y", y)):
        with _test_data:
            z = _test_data.set("z", _test_data.get("x", x) + _test_data.get("y", y))

    print(_test_data.control_dependencies)

    print(call_test())

    call_tree = ast.parse(inspect.getsource(call_test))
    TrackCallTransformer().visit(call_tree)
    TrackSetTransformer().visit(call_tree)
    TrackGetTransformer().visit(call_tree)
    TrackControlTransformer().visit(call_tree)
    TrackReturnTransformer().visit(call_tree)
    TrackParamsTransformer().visit(call_tree)
    dump_tree(call_tree)

    with DependencyTrackerTester(call_tree, call_test, log=False) as call_deps:
        call_test()

    print(call_deps.dependencies())

    with Slicer(middle) as slicer:
        m = middle(2, 1, 3)
    print(slicer.dependencies())
    print(slicer.code())

    with Slicer(log=True) as slicer:
        fun_2(10)
