#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence, Any, Callable, Tuple
from typing import Dict, Union, Set, List, cast, Optional
import sys, importlib
from pathlib import Path
import tempfile
import subprocess
from types import FrameType
import ast
import inspect
import itertools
import functools


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
from .ExpectError import ExpectError, ExpectTimeout
from .demo_func import square_root, remove_html_markup, middle
from .StatisticalDebugger import MIDDLE_FAILING_TESTCASES  # minor dependency
from .StatisticalDebugger import MIDDLE_PASSING_TESTCASES  # minor dependency

from .bookutils import print_content
from .bookutils import show_ast
from .StackInspector import StackInspector

Arguments = List[Tuple[str, Any]]


def square_root_with_type_annotations(x: float) -> float:
    """Computes the square root of x, using the Newton-Raphson method"""
    return square_root(x)


def get_arguments(frame: FrameType) -> Arguments:
    """Return call arguments in the given frame"""
    # When called, all arguments are local variables
    local_variables = dict(frame.f_locals)  # explicit copy
    arguments = [(var, frame.f_locals[var]) for var in local_variables]

    # FIXME: This may be needed for Python < 3.10
    # arguments.reverse()  # Want same order as call

    return arguments


def simple_call_string(function_name: str, argument_list: Arguments, return_value: Any = None) -> str:
    """Return function_name(arg[0], arg[1], ...) as a string"""
    call = function_name + "(" + ", ".join([var + "=" + repr(value) for (var, value) in argument_list]) + ")"

    if return_value is not None:
        call += " = " + repr(return_value)

    return call


class CallTracer(Tracer):
    def __init__(self, log: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._log = log
        self.reset()

    def reset(self) -> None:
        self._calls: Dict[str, List[Tuple[Arguments, Any]]] = {}
        self._stack: List[Tuple[str, Arguments]] = []

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracking function: Record all calls and all args"""
        if event == "call":
            self.trace_call(frame, event, arg)
        elif event == "return":
            self.trace_return(frame, event, arg)

    def trace_call(self, frame: FrameType, event: str, arg: Any) -> None:
        """Save current function name and args on the stack"""
        code = frame.f_code
        function_name = code.co_name
        arguments = get_arguments(frame)
        self._stack.append((function_name, arguments))

        if self._log:
            print(simple_call_string(function_name, arguments))

    def trace_return(self, frame: FrameType, event: str, arg: Any) -> None:
        """Get return value and store complete call with arguments and return value"""
        code = frame.f_code
        function_name = code.co_name
        return_value = arg
        # TODO: Could call get_arguments() here
        # to also retrieve _final_ values of argument variables

        called_function_name, called_arguments = self._stack.pop()
        assert function_name == called_function_name

        if self._log:
            print(simple_call_string(function_name, called_arguments), "returns", return_value)

        self.add_call(function_name, called_arguments, return_value)

    def add_call(self, function_name: str, arguments: Arguments, return_value: Any = None) -> None:
        """Add given call to list of calls"""
        if function_name not in self._calls:
            self._calls[function_name] = []

        self._calls[function_name].append((arguments, return_value))

    def calls(self, function_name: str) -> List[Tuple[Arguments, Any]]:
        """Return list of calls for `function_name`."""
        return self._calls[function_name]

    def all_calls(self) -> Dict[str, List[Tuple[Arguments, Any]]]:
        """
        Return list of calls for function_name,
        or a mapping function_name -> calls for all functions tracked
        """
        return self._calls


def hello(name: str) -> None:
    print("Hello,", name)


def parse_type(name: str) -> ast.expr:
    class ValueVisitor(ast.NodeVisitor):
        def visit_Expr(self, node: ast.Expr) -> None:
            self.value_node = node.value

    tree = ast.parse(name)
    name_visitor = ValueVisitor()
    name_visitor.visit(tree)
    return name_visitor.value_node


class TypeTransformer(ast.NodeTransformer):
    def __init__(self, argument_types: Dict[str, str], return_type: Optional[str] = None):
        self.argument_types = argument_types
        self.return_type = return_type
        super().__init__()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Add annotation to function"""
        # Set argument types
        new_args = []
        for arg in node.args.args:
            new_args.append(self.annotate_arg(arg))

        new_arguments = ast.arguments(
            node.args.posonlyargs,
            new_args,
            node.args.vararg,
            node.args.kwonlyargs,
            node.args.kw_defaults,
            node.args.kwarg,
            node.args.defaults,
        )

        # Set return type
        if self.return_type is not None:
            node.returns = parse_type(self.return_type)

        return ast.copy_location(
            ast.FunctionDef(node.name, new_arguments, node.body, node.decorator_list, node.returns), node
        )

    def annotate_arg(self, arg: ast.arg) -> ast.arg:
        """Add annotation to single function argument"""
        arg_name = arg.arg
        if arg_name in self.argument_types:
            arg.annotation = parse_type(self.argument_types[arg_name])
        return arg


def type_string(value: Any) -> str:
    return type(value).__name__


def annotate_types(calls: Dict[str, List[Tuple[Arguments, Any]]]) -> Dict[str, ast.AST]:
    annotated_functions = {}
    stack_inspector = StackInspector()

    for function_name in calls:
        function = stack_inspector.search_func(function_name)
        if function:
            annotated_functions[function_name] = annotate_function_with_types(function, calls[function_name])

    return annotated_functions


def annotate_function_with_types(function: Callable, function_calls: List[Tuple[Arguments, Any]]) -> ast.AST:
    function_code = inspect.getsource(function)
    function_ast = ast.parse(function_code)
    return annotate_function_ast_with_types(function_ast, function_calls)


def annotate_function_ast_with_types(function_ast: ast.AST, function_calls: List[Tuple[Arguments, Any]]) -> ast.AST:
    parameter_types: Dict[str, str] = {}
    return_type = None

    for calls_seen in function_calls:
        args, return_value = calls_seen
        if return_value:
            if return_type and return_type != type_string(return_value):
                return_type = "Any"
            else:
                return_type = type_string(return_value)

        for parameter, value in args:
            try:
                different_type = parameter_types[parameter] != type_string(value)
            except KeyError:
                different_type = False

            if different_type:
                parameter_types[parameter] = "Any"
            else:
                parameter_types[parameter] = type_string(value)

    annotated_function_ast = TypeTransformer(parameter_types, return_type).visit(function_ast)

    return annotated_function_ast


class TypeAnnotator(CallTracer):
    def typed_functions_ast(self) -> Dict[str, ast.AST]:
        """Return a dict name -> AST for all functions observed, annotated with types"""
        return annotate_types(self.all_calls())

    def typed_function_ast(self, function_name: str) -> Optional[ast.AST]:
        """Return an AST for all calls of `function_name` observed, annotated with types"""
        function = self.search_func(function_name)
        if not function:
            return None
        return annotate_function_with_types(function, self.calls(function_name))

    def typed_functions(self) -> str:
        """Return the code for all functions observed, annotated with types"""
        functions = ""
        for f_name in self.all_calls():
            f_ast = self.typed_function_ast(f_name)
            if f_ast:
                functions += ast.unparse(f_ast)
            else:
                functions += "# Could not find function " + repr(f_name)

        return functions

    def typed_function(self, function_name: str) -> str:
        """Return the code for all calls of `function_name` observed, annotated with types"""
        function_ast = self.typed_function_ast(function_name)
        if not function_ast:
            raise KeyError
        return ast.unparse(function_ast)

    def __repr__(self) -> str:
        """String representation, like `typed_functions()`"""
        return self.typed_functions()


def square_root_with_invariants(x):  # type: ignore
    assert x >= 0  # Precondition

    ...

    assert result * result == x  # Postcondition
    return result


def condition(precondition: Optional[Callable] = None, postcondition: Optional[Callable] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # preserves name, docstring, etc
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if precondition is not None:
                assert precondition(*args, **kwargs), "Precondition violated"

            # Call original function or method
            retval = func(*args, **kwargs)
            if postcondition is not None:
                assert postcondition(retval, *args, **kwargs), "Postcondition violated"

            return retval

        return wrapper

    return decorator


def precondition(check: Callable) -> Callable:
    return condition(precondition=check)


def postcondition(check: Callable) -> Callable:
    return condition(postcondition=check)


@precondition(lambda x: x > 0)
def square_root_with_precondition(x):  # type: ignore
    return square_root(x)


import math


@postcondition(lambda ret, x: math.isclose(ret * ret, x))
def square_root_with_postcondition(x):  # type: ignore
    return square_root(x)


@postcondition(lambda ret, x: math.isclose(ret * ret, x))
def buggy_square_root_with_postcondition(x):  # type: ignore
    return square_root(x) + 0.1


INVARIANT_PROPERTIES = [
    "X < 0",
    "X <= 0",
    "X > 0",
    "X >= 0",
    # "X == 0",  # implied by "X", below
    # "X != 0",  # implied by "not X", below
]

INVARIANT_PROPERTIES += [
    "X == Y",
    "X > Y",
    "X < Y",
    "X >= Y",
    "X <= Y",
]

INVARIANT_PROPERTIES += [
    "isinstance(X, bool)",
    "isinstance(X, int)",
    "isinstance(X, float)",
    "isinstance(X, list)",
    "isinstance(X, dict)",
]

INVARIANT_PROPERTIES += [
    "X == Y + Z",
    "X == Y * Z",
    "X == Y - Z",
    "X == Y / Z",
]

INVARIANT_PROPERTIES += [
    "X < Y < Z",
    "X <= Y <= Z",
    "X > Y > Z",
    "X >= Y >= Z",
]

INVARIANT_PROPERTIES += ["X", "not X"]

INVARIANT_PROPERTIES += [
    "X == len(Y)",
    "X == sum(Y)",
    "X in Y",
    "X.startswith(Y)",
    "X.endswith(Y)",
]


def metavars(prop: str) -> List[str]:
    metavar_list = []

    class ArgVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id.isupper():
                metavar_list.append(node.id)

    ArgVisitor().visit(ast.parse(prop))
    return metavar_list


def instantiate_prop_ast(prop: str, var_names: Sequence[str]) -> ast.AST:
    class NameTransformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id not in mapping:
                return node
            return ast.Name(id=mapping[node.id], ctx=ast.Load())

    meta_variables = metavars(prop)
    assert len(meta_variables) == len(var_names)

    mapping = {}
    for i in range(0, len(meta_variables)):
        mapping[meta_variables[i]] = var_names[i]

    prop_ast = ast.parse(prop, mode="eval")
    new_ast = NameTransformer().visit(prop_ast)

    return new_ast


def instantiate_prop(prop: str, var_names: Sequence[str]) -> str:
    prop_ast = instantiate_prop_ast(prop, var_names)
    prop_text = ast.unparse(prop_ast).strip()
    while prop_text.startswith("(") and prop_text.endswith(")"):
        prop_text = prop_text[1:-1]
    return prop_text


def prop_function_text(prop: str) -> str:
    return "lambda " + ", ".join(metavars(prop)) + ": " + prop


def prop_function(prop: str) -> Callable:
    return eval(prop_function_text(prop))


Invariants = Set[Tuple[str, Tuple[str, ...]]]

RETURN_VALUE = "return_value"


def true_property_instantiations(prop: str, vars_and_values: Arguments, log: bool = False) -> Invariants:
    instantiations = set()
    p = prop_function(prop)

    len_metavars = len(metavars(prop))
    for combination in itertools.permutations(vars_and_values, len_metavars):
        args = [value for var_name, value in combination]
        var_names = [var_name for var_name, value in combination]

        try:
            result = p(*args)
        except:
            result = None

        if log:
            print(prop, combination, result)
        if result:
            instantiations.add((prop, tuple(var_names)))

    return instantiations


def pretty_invariants(invariants: Invariants) -> List[str]:
    props = []
    for prop, var_names in invariants:
        props.append(instantiate_prop(prop, var_names))
    return sorted(props)


class InvariantTracer(CallTracer):
    def __init__(self, props: Optional[List[str]] = None, **kwargs: Any) -> None:
        if props is None:
            props = INVARIANT_PROPERTIES

        self.props = props
        super().__init__(**kwargs)

    def all_invariants(self) -> Dict[str, Invariants]:
        return {function_name: self.invariants(function_name) for function_name in self.all_calls()}

    def invariants(self, function_name: str) -> Invariants:
        invariants = None
        for variables, return_value in self.calls(function_name):
            vars_and_values = variables + [(RETURN_VALUE, return_value)]

            s = set()
            for prop in self.props:
                s |= true_property_instantiations(prop, vars_and_values, self._log)
            if invariants is None:
                invariants = s
            else:
                invariants &= s

        assert invariants is not None
        return invariants


class InvariantAnnotator(InvariantTracer):
    def params(self, function_name: str) -> str:
        arguments, return_value = self.calls(function_name)[0]
        return ", ".join(arg_name for (arg_name, arg_value) in arguments)

    def preconditions(self, function_name: str) -> List[str]:
        """Return a list of mined preconditions for `function_name`"""
        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) >= 0:
                continue  # Postcondition

            cond = "@precondition(lambda " + self.params(function_name) + ": " + inv + ")"
            conditions.append(cond)

        return conditions

    def postconditions(self, function_name: str) -> List[str]:
        """Return a list of mined postconditions for `function_name`"""

        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) < 0:
                continue  # Precondition

            cond = f"@postcondition(lambda {RETURN_VALUE}," f" {self.params(function_name)}: {inv})"
            conditions.append(cond)

        return conditions

    def functions_with_invariants(self) -> str:
        """Return the code of all observed functions, annotated with invariants"""

        functions = ""
        for function_name in self.all_invariants():
            try:
                function = self.function_with_invariants(function_name)
            except KeyError:
                function = "# Could not find function " + repr(function_name)

            functions += function
        return functions

    def function_with_invariants(self, function_name: str) -> str:
        """Return the code of `function_name`, annotated with invariants"""
        function = self.search_func(function_name)
        if not function:
            raise KeyError
        source = inspect.getsource(function)
        return "\n".join(self.preconditions(function_name) + self.postconditions(function_name)) + "\n" + source

    def __repr__(self) -> str:
        """String representation, like `functions_with_invariants()`"""
        return self.functions_with_invariants()

@precondition(lambda s: len(s) > 0)
def remove_first_char(s: str) -> str:
    return s[1:]


def verbose_condition(
    precondition: Optional[Callable] = None, postcondition: Optional[Callable] = None, doc: str = "Unknown"
) -> Callable:
    def decorator(func: Callable) -> Callable:
        # Use `functools` to preserve name, docstring, etc
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if precondition is not None:
                assert precondition(*args, **kwargs), "Precondition violated: " + doc

            # call original function or method
            retval = func(*args, **kwargs)

            if postcondition is not None:
                assert postcondition(retval, *args, **kwargs), "Postcondition violated: " + doc

            return retval

        return wrapper

    return decorator


def verbose_precondition(check: Callable, **kwargs: Any) -> Callable:
    return verbose_condition(precondition=check, doc=kwargs.get("doc", "Unknown"))


def verbose_postcondition(check: Callable, **kwargs: Any) -> Callable:
    return verbose_condition(postcondition=check, doc=kwargs.get("doc", "Unknown"))


@verbose_precondition(lambda s: len(s) > 0, doc="len(s) > 0")  # type: ignore
def remove_first_char(s: str) -> str:
    return s[1:]


class VerboseInvariantAnnotator(InvariantAnnotator):
    def preconditions(self, function_name: str) -> List[str]:
        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) >= 0:
                continue  # Postcondition

            cond = (
                "@verbose_precondition(lambda " + self.params(function_name) + ": " + inv + ", doc=" + repr(inv) + ")"
            )
            conditions.append(cond)

        return conditions

    def postconditions(self, function_name: str) -> List[str]:
        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) < 0:
                continue  # Precondition

            cond = (
                "@verbose_postcondition(lambda "
                + RETURN_VALUE
                + ", "
                + self.params(function_name)
                + ": "
                + inv
                + ", doc="
                + repr(inv)
                + ")"
            )
            conditions.append(cond)

        return conditions


class EmbeddedInvariantAnnotator(InvariantTracer):
    def function_with_invariants_ast(self, function_name: str) -> ast.AST:
        return annotate_function_with_invariants(function_name, self.invariants(function_name))

    def function_with_invariants(self, function_name: str) -> str:
        return ast.unparse(self.function_with_invariants_ast(function_name))


def annotate_invariants(invariants: Dict[str, Invariants]) -> Dict[str, ast.AST]:
    annotated_functions = {}

    for function_name in invariants:
        try:
            annotated_functions[function_name] = annotate_function_with_invariants(
                function_name, invariants[function_name]
            )
        except KeyError:
            continue

    return annotated_functions


def annotate_function_with_invariants(function_name: str, function_invariants: Invariants) -> ast.AST:
    stack_inspector = StackInspector()
    function = stack_inspector.search_func(function_name)
    if function is None:
        raise KeyError

    function_code = inspect.getsource(function)
    function_ast = ast.parse(function_code)
    return annotate_function_ast_with_invariants(function_ast, function_invariants)


def annotate_function_ast_with_invariants(function_ast: ast.AST, function_invariants: Invariants) -> ast.AST:
    annotated_function_ast = EmbeddedInvariantTransformer(function_invariants).visit(function_ast)
    return annotated_function_ast


class PreconditionTransformer(ast.NodeTransformer):
    def __init__(self, invariants: Invariants) -> None:
        self.invariants = invariants
        super().__init__()

    def preconditions(self) -> List[ast.stmt]:
        preconditions = []
        for prop, var_names in self.invariants:
            assertion = "assert " + instantiate_prop(prop, var_names) + ', "violated precondition"'
            assertion_ast = ast.parse(assertion)

            if assertion.find(RETURN_VALUE) < 0:
                preconditions += assertion_ast.body

        return preconditions

    def insert_assertions(self, body: List[ast.stmt]) -> List[ast.stmt]:
        preconditions = self.preconditions()
        try:
            docstring = cast(ast.Constant, body[0]).value.s
        except:
            docstring = None

        if docstring:
            return [body[0]] + preconditions + body[1:]
        else:
            return preconditions + body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Add invariants to function"""
        # print(ast.dump(node))
        node.body = self.insert_assertions(node.body)
        return node


class EmbeddedInvariantTransformer(PreconditionTransformer):
    pass


class EmbeddedInvariantTransformer(EmbeddedInvariantTransformer):
    def postconditions(self) -> List[ast.stmt]:
        postconditions = []

        for prop, var_names in self.invariants:
            assertion = "assert " + instantiate_prop(prop, var_names) + ', "violated postcondition"'
            assertion_ast = ast.parse(assertion)

            if assertion.find(RETURN_VALUE) >= 0:
                postconditions += assertion_ast.body

        return postconditions

    def insert_assertions(self, body: List[ast.stmt]) -> List[ast.stmt]:
        new_body = super().insert_assertions(body)
        postconditions = self.postconditions()

        body_ends_with_return = isinstance(new_body[-1], ast.Return)
        if body_ends_with_return:
            ret_val = cast(ast.Return, new_body[-1]).value
            saver = RETURN_VALUE + " = " + ast.unparse(cast(ast.AST, ret_val))
        else:
            saver = RETURN_VALUE + " = None"

        saver_ast = cast(ast.stmt, ast.parse(saver))
        postconditions = [saver_ast] + postconditions

        if body_ends_with_return:
            return new_body[:-1] + postconditions + [new_body[-1]]
        else:
            return new_body + postconditions


if __name__ == "__main__":
    with CallTracer(log=True) as tracer:
        y = square_root(25)
        y = square_root(2.0)

    # calls = tracer.calls("square_root")
    # print(calls)

    # square_root_argument_list, square_root_return_value = calls[0]
    # simple_call_string("square_root", square_root_argument_list, square_root_return_value)

    square_root_source = inspect.getsource(square_root)
    square_root_ast = ast.parse(square_root_source)
    new_ast = TypeTransformer({"x": "int"}, "float").visit(square_root_ast)
    print_content(ast.unparse(new_ast), ".py")

    print("/n")
    print_content(ast.unparse(annotate_types(tracer.all_calls())["square_root"]), ".py")
