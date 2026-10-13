from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

_PID_RE = re.compile(r"^[0-9a-f]{8}$")


def load_pid(pid: str) -> ModuleType:
    """
    Load an ARC-2 solver module by puzzle id (hex string), even though the underlying filename
    is not a valid Python identifier (e.g. `332f06d7.py`).
    """
    pid = str(pid).strip().lower()
    if not _PID_RE.match(pid):
        raise ValueError(f"load_pid: invalid pid {pid!r}")

    here = Path(__file__).resolve().parent
    path = here / f"{pid}.py"
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(f"solvers.arc2.pid_{pid}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_input(pid: str, suffix: str = ""):
    """Convenience wrapper: load pid module and call its generate_input()."""
    mod = load_pid(pid)
    fn = getattr(mod, "generate_input" + suffix, None)
    if not callable(fn):
        raise AttributeError(f"{pid}: module has no generate_input()")
    return fn()


def main(pid: str, grid):
    """Convenience wrapper: load pid module and call its main(grid)."""
    mod = load_pid(pid)
    fn = getattr(mod, "main", None)
    if not callable(fn):
        raise AttributeError(f"{pid}: module has no main()")
    return fn(grid)

