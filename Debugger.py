#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import sys, importlib
from pathlib import Path
from types import FrameType
from typing import Any, Optional, Callable, Dict, List, Tuple, Set, TextIO
import inspect


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
from .bookutils import getsourcelines  # like inspect.getsourcelines(), but in color


class Debugger(Tracer):
    """Interactive Debugger"""

    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Create a new interactive debugger."""
        self.stepping: bool = True
        self.breakpoints: Set[int] = set()
        self.interact: bool = True

        self.frame: FrameType
        self.event: Optional[str] = None
        self.arg: Any = None

        self.local_vars: Dict[str, Any] = {}
        super().__init__(file=file)

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.frame = frame
        self.local_vars = frame.f_locals  # Dereference exactly once
        self.event = event
        self.arg = arg

        if self.stop_here():
            self.interaction_loop()

    def stop_here(self) -> bool:
        """Return True if we should stop"""
        return self.stepping or self.frame.f_lineno in self.breakpoints

    def interaction_loop(self) -> None:
        """Interact with the user"""
        self.print_debugger_status(self.frame, self.event, self.arg)  # type: ignore

        self.interact = True
        while self.interact:
            command = input("(debugger) ")
            self.execute(command)  # type: ignore

    def step_command(self, arg: str = "") -> None:
        """Execute up to the next line"""

        self.stepping = True
        self.interact = False

    def continue_command(self, arg: str = "") -> None:
        """Resume execution"""

        self.stepping = False
        self.interact = False

    def commands(self) -> List[str]:
        """Return a list of commands"""

        cmds = [method.replace("_command", "") for method in dir(self.__class__) if method.endswith("_command")]
        cmds.sort()
        return cmds

    def command_method(self, command: str) -> Optional[Callable[[str], None]]:
        """Convert `command` into the method to be called.
        If the method is not found, return `None` instead."""

        if command.startswith("#"):
            return None  # Comment

        possible_cmds = [possible_cmd for possible_cmd in self.commands() if possible_cmd.startswith(command)]
        if len(possible_cmds) != 1:
            self.help_command(command)
            return None

        cmd = possible_cmds[0]
        return getattr(self, cmd + "_command")

    def execute(self, command: str) -> None:
        """Execute `command`"""

        sep = command.find(" ")
        if sep > 0:
            cmd = command[:sep].strip()
            arg = command[sep + 1 :].strip()
        else:
            cmd = command.strip()
            arg = ""

        method = self.command_method(cmd)
        if method:
            method(arg)

    def help_command(self, command: str = "") -> None:
        """Give help on given `command`. If no command is given, give help on all"""

        if command:
            possible_cmds = [possible_cmd for possible_cmd in self.commands() if possible_cmd.startswith(command)]

            if len(possible_cmds) == 0:
                self.log(f"Unknown command {repr(command)}. Possible commands are:")
                possible_cmds = self.commands()
            elif len(possible_cmds) > 1:
                self.log(f"Ambiguous command {repr(command)}. Possible expansions are:")
        else:
            possible_cmds = self.commands()

        for cmd in possible_cmds:
            method = self.command_method(cmd)
            self.log(f"{cmd:10} -- {method.__doc__}")

    def print_command(self, arg: str = "") -> None:
        """Print an expression. If no expression is given, print all variables"""

        vars = self.local_vars

        if not arg:
            self.log("\n".join([f"{var} = {repr(value)}" for var, value in vars.items()]))
        else:
            try:
                self.log(f"{arg} = {repr(eval(arg, globals(), vars))}")
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")

    def break_command(self, arg: str = "") -> None:
        """Set a breakoint in given line. If no line is given, list all breakpoints"""

        if arg:
            self.breakpoints.add(int(arg))
        self.log("Breakpoints:", self.breakpoints)

    def delete_command(self, arg: str = "") -> None:
        """Delete breakoint in line given by `arg`.
        Without given line, clear all breakpoints"""

        if arg:
            try:
                self.breakpoints.remove(int(arg))
            except KeyError:
                self.log(f"No such breakpoint: {arg}")
        else:
            self.breakpoints = set()
        self.log("Breakpoints:", self.breakpoints)

    def list_command(self, arg: str = "") -> None:
        """Show current function. If `arg` is given, show its source code."""

        try:
            if arg:
                obj = eval(arg)
                source_lines, line_number = inspect.getsourcelines(obj)
                current_line = -1
            else:
                source_lines, line_number = getsourcelines(self.frame.f_code)
                current_line = self.frame.f_lineno
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")
            source_lines = []
            line_number = 0

        for line in source_lines:
            spacer = " "
            if line_number == current_line:
                spacer = ">"
            elif line_number in self.breakpoints:
                spacer = "#"
            self.log(f"{line_number:4}{spacer} {line}", end="")
            line_number += 1

    def quit_command(self, arg: str = "") -> None:
        """Finish execution"""

        self.breakpoints = set()
        self.stepping = False
        self.interact = False
