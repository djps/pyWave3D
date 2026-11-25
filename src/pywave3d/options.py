"""Read-only access to the JSON options file (``woptions.json``).

The rules:

* keys are matched case-insensitively and dotted paths walk nested objects
  (``opts.get("Model.Geometry.lx", float)``); an empty path is the node itself;
* a missing key gives the default (``get``) or an empty node (``child``), never an error;
* a value of the wrong type raises :class:`TypeError` naming the key;
* accepted values, by requested type: ``str`` (trimmed), ``bool`` (``true``/``false`` or the
  strings yes/no, y/n, true/false, t/f), ``int`` (an integer, not a bool), ``float`` (any
  number), and ``list[float]``/``list[str]``/... (a single value is a list of one).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, TypeVar, overload

T = TypeVar("T")

_TRUE = ("true", "yes", "y", "t")
_FALSE = ("false", "no", "n", "f")


def _find(node: Any, key: str) -> Any:
    """Return the value of ``key`` in the dict ``node`` (case-insensitive), or None."""
    if not isinstance(node, dict):
        return None
    if key in node:
        return node[key]
    lower = key.lower()
    for k, v in node.items():
        if k.lower() == lower:
            return v
    return None


class Options:
    """One node (JSON object or value) of the options file.

    Args:
        node: the parsed JSON value wrapped by this node (None for an empty node).
        path: the dotted path of the node in the document, used in error messages.
        base_dir: the directory of the options file, against which relative file names are
            resolved (see :meth:`resolve`).
    """

    __slots__ = ("_base_dir", "_node", "_path")

    def __init__(self, node: Any = None, path: str = "", base_dir: Path | None = None) -> None:
        self._node = node
        self._path = path
        self._base_dir = base_dir

    @classmethod
    def load(cls, filename: str | Path) -> Options:
        """Parse a JSON file and return its top-level node.

        Raises:
            OSError: if the file cannot be read.
            json.JSONDecodeError: if it is not valid JSON.
        """
        path = Path(filename)
        with path.open(encoding="utf-8") as f:
            return cls(json.load(f), "", path.resolve().parent)

    @classmethod
    def from_string(cls, text: str) -> Options:
        """Parse JSON text and return its top-level node."""
        return cls(json.loads(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> Options:
        """Wrap an already parsed JSON object (it is not copied)."""
        return cls(data, "", base_dir)

    # -- navigation ----------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if this node is present in the document."""
        return self._node is not None

    @property
    def path(self) -> str:
        """Dotted path of this node, as used in error messages."""
        return self._path

    @property
    def base_dir(self) -> Path | None:
        """Directory of the options file, or None for options not read from a file."""
        return self._base_dir

    def _join(self, path: str) -> str:
        return f"{self._path}.{path}" if self._path and path else (self._path or path)

    def _locate(self, path: str) -> Any:
        node = self._node
        if not path:
            return node
        for key in path.split("."):
            node = _find(node, key)
            if node is None:
                return None
        return node

    def child(self, path: str) -> Options:
        """Return the node at ``path``; an empty node (``exists()`` False) if there is none."""
        return Options(self._locate(path), self._join(path), self._base_dir)

    def children(self, tag: str) -> list[Options]:
        """Return the elements of the array at ``tag``.

        A single object there is a list of one, and nothing there an empty list.
        """
        node = self._locate(tag)
        if node is None:
            return []
        path = self._join(tag)
        if isinstance(node, list):
            return [Options(e, f"{path}[{i}]", self._base_dir) for i, e in enumerate(node)]
        return [Options(node, path, self._base_dir)]

    # names kept from the first Python port
    def get_option(self, path: str) -> Options | None:
        """Return the child at ``path``, or None if there is none (see :meth:`child`)."""
        node = self.child(path)
        return node if node.exists() else None

    def get_options(self, tag: str) -> list[Options]:
        """Alias of :meth:`children`."""
        return self.children(tag)

    # -- values --------------------------------------------------------------------------

    @overload
    def get(self, path: str, type: type[T], default: T) -> T: ...
    @overload
    def get(self, path: str, type: type[T], default: None = None) -> T | None: ...
    @overload
    def get(self, path: str, type: Any, default: Any = None) -> Any: ...

    def get(self, path: str = "", type: Any = str, default: Any = None) -> Any:
        """Return the value at ``path`` converted to ``type``, or ``default`` if it is missing.

        Args:
            path: dotted, case-insensitive path below this node ("" for the node itself).
            type: ``str``, ``bool``, ``int``, ``float`` or ``list[...]`` of one of these.
            default: returned when there is no such key.

        Raises:
            TypeError: if the value cannot be read as ``type``.
            ValueError: if ``type`` is not one of the supported types.
        """
        value = self._locate(path)
        if value is None:
            return default
        where = self._join(path) or "<root>"

        if type is str:
            if not isinstance(value, str):
                raise TypeError(f"options: '{where}' should be a string")
            return value.strip()
        if type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in _TRUE + _FALSE:
                return value.strip().lower() in _TRUE
            raise TypeError(f"options: '{where}' should be true or false")
        if type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"options: '{where}' should be an integer")
            return value
        if type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"options: '{where}' should be a number")
            return float(value)

        origin = getattr(type, "__origin__", None)
        args = getattr(type, "__args__", None)
        if origin is list and args:
            items = value if isinstance(value, list) else [value]
            if not items:
                raise TypeError(f"options: '{where}' should not be empty")
            node = Options(None, where, self._base_dir)
            out = []
            for item in items:
                node._node = item
                out.append(node.get("", args[0]))
            return out

        raise ValueError(f"options: unsupported type {type!r} for '{where}'")

    def resolve(self, filename: str, search: tuple[str, ...] = ("", "../data")) -> Path:
        """Find a file named in the options.

        An absolute name is returned as is. A relative one is looked for, in order, in each
        ``search`` directory relative to the options file, then in the current directory.

        Raises:
            FileNotFoundError: if it is found nowhere.
        """
        p = Path(filename)
        if p.is_absolute():
            return p
        candidates = []
        if self._base_dir is not None:
            candidates += [self._base_dir / d / p for d in search]
        candidates.append(Path.cwd() / p)
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"options: cannot find '{filename}' (looked in {', '.join(str(c.parent) for c in candidates)})"
        )

    def to_dict(self) -> Any:
        """Return a deep copy of the wrapped JSON value, as plain dicts, lists and values."""
        return copy.deepcopy(self._node)

    def __repr__(self) -> str:
        return f"Options(path={self._path!r}, exists={self.exists()})"


def load_options(filename: str | Path) -> Options:
    """Load a JSON options file and return its top-level node."""
    return Options.load(filename)


def options_to_dict(opt: Options) -> Any:
    """Return a deep copy of the JSON value wrapped by ``opt``."""
    return opt.to_dict()
