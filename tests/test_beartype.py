"""beartype must be active on the whole package: every annotated function is checked at run time.

Ruff's ANN rules (pyproject.toml) make every function in src/ annotated; this test makes sure the
beartype import hook in ``pywave3d/__init__.py`` then wraps all of them, so no module escapes it.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import pywave3d


def package_functions() -> Iterator[tuple[str, Callable[..., Any]]]:
    """Every function and method defined in a pywave3d module (not ``__main__``, which runs a simulation)."""
    for info in pkgutil.walk_packages(pywave3d.__path__, "pywave3d."):
        if info.name.endswith("__main__"):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{name}", obj
            elif inspect.isclass(obj):
                for attr, value in vars(obj).items():
                    func = value.fget if isinstance(value, property) else getattr(value, "__func__", value)
                    if inspect.isfunction(func) and func.__qualname__.startswith(obj.__qualname__ + "."):
                        yield f"{module.__name__}.{name}.{attr}", func


def checkable(func: Callable[..., Any]) -> bool:
    """beartype has something to check: an annotation other than Any (dataclass-generated methods have none)."""
    return any(a not in (Any, "Any") for a in func.__annotations__.values())


def test_beartype_is_on() -> None:
    assert not os.environ.get("PYWAVE3D_NO_BEARTYPE"), "unset PYWAVE3D_NO_BEARTYPE: the tests need beartype on"


def test_every_annotated_function_is_beartyped() -> None:
    functions = list(package_functions())
    assert len(functions) > 100
    unchecked = [name for name, func in functions if checkable(func) and not hasattr(func, "__beartype_wrapper")]
    assert not unchecked, f"not checked by beartype: {unchecked}"


def test_beartype_rejects_a_wrong_type() -> None:
    from beartype.roar import BeartypeCallHintParamViolation

    with pytest.raises(BeartypeCallHintParamViolation):
        pywave3d.Grid.from_lengths("1", 1.0, 1.0, wavelength=1e-3)  # type: ignore[arg-type]
