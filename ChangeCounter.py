from pydriller import Repository  # https://pydriller.readthedocs.io/
from pydriller.domain.commit import Commit
from pydriller.domain.commit import ModifiedFile
import os

from typing import Callable, Optional, Type, Tuple, Any
from typing import Dict, Union, Set, List


import easyplotly as ep
import plotly.graph_objects as go
import math
import magic
import re
from typing import Pattern
from datetime import datetime
from collections import defaultdict
import warnings
from git.exc import GitCommandError  # type: ignore
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


def current_repo(path=os.path.abspath(".")) -> Optional[str]:
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return os.path.normpath(path)

        # Go one level up
        new_path = os.path.normpath(os.path.join(path, ".."))
        if new_path != path:
            path = new_path
        else:
            return None
    return None


DEBUGGINGBOOK_START_DATE: datetime = datetime(2021, 3, 1)
NUM_WORKERS = 4  # Number of threads to be run in parallel
Node = Tuple


class ChangeCounter:
    """Count the number of changes for a repository."""

    def __init__(
        self, repo: str, *, filter: Optional[Callable[[Commit], bool]] = None, log: bool = False, **kwargs: Any
    ) -> None:
        """
        Constructor.
        `repo` is a git repository (as URL or directory).
        `filter` is a predicate that takes a modification and returns True
        if it should be considered (default: consider all).
        `log` turns on logging if set.
        `kwargs` are passed to the `Repository()` constructor.
        """
        self.repo = repo
        self.log = log

        if filter is None:

            def filter(m: ModifiedFile) -> bool:
                return True

        assert filter is not None

        self.filter = filter

        # A node is an tuple (f_1, f_2, f_3, ..., f_n) denoting
        # a folder f_1 holding a folder f_2 ... holding a file f_n.

        # Mapping node -> #of changes
        self.changes: Dict[Node, int] = defaultdict(int)

        # Mapping node -> list of commit messages
        self.messages: Dict[Node, List[str]] = defaultdict(list)

        # Mapping node -> last size seen
        self.sizes: Dict[Node, Union[int, float]] = {}

        self.mine(**kwargs)

    def mine(self, **kwargs: Any) -> None:
        """Gather data from repository. To be extended in subclasses."""
        miner = Repository(self.repo, **kwargs)

        for commit in miner.traverse_commits():
            try:
                self.mine_commit(commit)
            except GitCommandError as err:
                # Warn about failing git commands, but continue
                warnings.warn("Cannot mine commit " + repr(commit.hash) + "\n" + str(err))
            except (ValueError, TypeError) as err:
                warnings.warn("Cannot mine commit " + repr(commit.hash) + "\n" + str(err))
                raise err

    def mine_commit(self, commit: Commit) -> None:
        for m in commit.modified_files:
            m.committer = commit.committer
            m.committer_date = commit.committer_date
            m.msg = commit.msg

            if self.include(m):
                self.update_stats(m)

    def include(self, m: ModifiedFile) -> bool:
        """
        Return True if the modification `m` should be included
        (default: the `filter` predicate given to the constructor).
        To be overloaded in subclasses.
        """
        return self.filter(m)

    def update_stats(self, m: ModifiedFile) -> None:
        """
        Update counters with modification `m`.
        Can be extended in subclasses.
        """
        if not m.new_path:
            return

        node = tuple(m.new_path.split("/"))

        self.update_size(node, len(m.content) if m.content else 0)
        self.update_changes(node, m.msg)

        self.update_elems(node, m)

    def update_size(self, node: Tuple, size: int) -> None:
        """
        Update counters for `node` with `size`.
        Can be extended in subclasses.
        """
        self.sizes[node] = size

    def update_changes(self, node: Tuple, commit_msg: str) -> None:
        """
        Update stats for `node` changed with `commit_msg`.
        Can be extended in subclasses.
        """
        self.changes[node] += 1

        self.messages[node].append(commit_msg)

    def update_elems(self, node: Tuple, m: ModifiedFile) -> None:
        """
        Update counters for subelements of `node` with modification `m`.
        To be defined in subclasses.
        """
        pass


class ChangeCounter(ChangeCounter):
    def map_node_sizes(self, scale: str = "log") -> Dict[Node, Union[int, float]]:
        """
        Return a mapping of nodes to sizes.
        Can be overloaded in subclasses.
        """

        if scale == "log":
            # Default: use log scale
            return {node: math.log(size + 1) for node, size in self.sizes.items()}

        elif scale == "sqrt":
            # Alternative: use sqrt size
            return {node: math.sqrt(size) for node, size in self.sizes.items()}

        elif scale == "abs":
            # Alternative: use absolute size
            return self.sizes

        else:
            raise ValueError(f"Unknown scale: {scale}; " f"use one of [log, sqrt, abs]")

    def map_node_color(self, node: Node) -> Optional[int]:
        """
        Return a color of the node, as a number.
        Can be overloaded in subclasses.
        """
        return self.changes.get(node)

    def map_node_text(self, node: Node) -> Optional[str]:
        """
        Return the text to be shown for the node (default: #changes).
        Can be overloaded in subclasses.
        """
        change = self.changes.get(node)
        return str(change) if change is not None else None

    def map_hoverinfo(self) -> str:
        """
        Return the text to be shown when hovering over a node.
        To be overloaded in subclasses.
        """
        return "label+text"

    def map_colorscale(self) -> str:
        """
        Return the colorscale for the map. To be overloaded in subclasses.
        """
        return "YlOrRd"

    def map(self) -> go.Figure:
        """Produce an interactive tree map of the repository."""
        treemap = ep.Treemap(
            self.map_node_sizes(),
            text=self.map_node_text,
            hoverinfo=self.map_hoverinfo(),
            marker_colors=self.map_node_color,
            marker_colorscale=self.map_colorscale(),
            root_label=self.repo,
            branchvalues="total",
        )

        fig = go.Figure(treemap)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        return fig


class FixCounter(ChangeCounter):
    """
    Count the fixes for files in the repository.
    Fixes are all commits whose message starts with the word 'Fix: '
    """

    def include(self, m: ModifiedFile) -> bool:
        """Include all modifications whose commit messages start with 'Fix:'"""
        return super().include(m) and m and m.msg.startswith("Fix:")

    def map_node_text(self, node: Node) -> str:
        return "<br>".join(self.messages.get(node, []))

    def map_hoverinfo(self) -> str:
        return "label"


Mapping = List[Optional[str]]

DELIMITERS: List[Tuple[Pattern, Pattern, Pattern]] = [
    (
        # Python
        re.compile(r"^python.*"),
        # Beginning of element
        re.compile(r"^(async\s+)?(def|class)\s+(?P<name>\w+)\W.*"),
        # End of element
        re.compile(r"^[^#\s]"),
    ),
    (
        # Jupyter Notebooks
        re.compile(r"^(json|exported sgml|jupyter).*"),
        re.compile(r'^\s+"(async\s+)?(def|class)\s+(?P<name>\w+)\W'),
        re.compile(r'^(\s+"[^#\s\\]|\s+\])'),
    ),
    (
        # C source code (actually, any { }-delimited language)
        re.compile(r"^(c |c\+\+|c#|java|perl|php).*"),
        re.compile(r"^[^\s].*\s+(?P<name>\w+)\s*[({].*"),
        re.compile(r"^[}]"),
    ),
]


def rxdelim(content: str) -> Tuple[Optional[Pattern], Optional[Pattern]]:
    """
    Return suitable begin and end delimiters for the content `content`.
    If no matching delimiters are found, return `None, None`.
    """
    tp = magic.from_buffer(content).lower()
    for rxtp, rxbegin, rxend in DELIMITERS:
        if rxtp.match(tp):
            return rxbegin, rxend

    return None, None


def elem_mapping(content: str, log: bool = False) -> Mapping:
    """Return a list of the elements in `content`, indexed by line number."""
    rxbegin, rxend = rxdelim(content)
    if rxbegin is None:
        return []
    if rxend is None:
        return []

    mapping: List[Optional[str]] = [None]
    current_elem = None
    lineno = 0

    for line in content.split("\n"):
        lineno += 1

        match = rxbegin.match(line)
        if match:
            current_elem = match.group("name")
        elif rxend.match(line):
            current_elem = None

        mapping.append(current_elem)

        if log:
            print(f"{lineno:3} {str(current_elem):15} {line}")

    return mapping


def changed_elems_by_mapping(mapping: Mapping, start: int, length: int = 0) -> Set[str]:
    """
    Within `mapping`, return the set of elements affected by a change
    starting in line `start` and extending over `length` additional lines.
    """
    elems = set()
    for line in range(start, start + length + 1):
        if line < len(mapping) and mapping[line]:
            elem = mapping[line]
            assert elem is not None
            elems.add(elem)

    return elems


def elem_size(elem: str, source: str) -> int:
    """Within `source`, return the size of `elem`"""
    source_lines = [""] + source.split("\n")
    size = 0
    mapping = elem_mapping(source)

    for line_no in range(len(mapping)):
        if mapping[line_no] == elem or mapping[line_no] is elem:
            size += len(source_lines[line_no] + "\n")

    return size


from .ChangeDebugger import diff  # minor dependency

from diff_match_patch import diff_match_patch


def changed_elems(old_source: str, new_source: str) -> Set[str]:
    """Determine the elements affected by the change from `old_source` to `new_source`"""
    patches = diff(old_source, new_source)

    old_mapping = elem_mapping(old_source)
    new_mapping = elem_mapping(new_source)

    elems = set()

    for patch in patches:
        old_start_line = patch.start1 + 1
        new_start_line = patch.start2 + 1

        for op, data in patch.diffs:
            length = data.count("\n")

            if op == diff_match_patch.DIFF_INSERT:
                elems |= changed_elems_by_mapping(old_mapping, old_start_line)
                elems |= changed_elems_by_mapping(new_mapping, new_start_line, length)
            elif op == diff_match_patch.DIFF_DELETE:
                elems |= changed_elems_by_mapping(old_mapping, old_start_line, length)
                elems |= changed_elems_by_mapping(new_mapping, new_start_line)

            old_start_line += length
            new_start_line += length

    return elems
