# Combo variant of 7b0280bc (index 15)
# combo_spec: D1/0+D6/0+D9/0
# Name: 树直径路径提取

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import colors_present, get_bg
from _shared import load_variant


_D1 = load_variant("D1", 0)

_SIBLING_6 = HERE.parent.parent / "6" / "code.py"
_SPEC_6 = importlib.util.spec_from_file_location("combo_7b0280bc_6", _SIBLING_6)
assert _SPEC_6 is not None and _SPEC_6.loader is not None
_COMBO6 = importlib.util.module_from_spec(_SPEC_6)
_SPEC_6.loader.exec_module(_COMBO6)


def _infer_station_and_track_colors(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    square_cands = _D1.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    counts: dict[int, int] = {}
    for d in square_cands:
        counts[int(d["color"])] = int(counts.get(int(d["color"]), 0) + 1)
    terminal_color = next(int(col) for col, cnt in counts.items() if int(cnt) == 2)
    station_color = next(int(col) for col, cnt in counts.items() if int(col) != int(terminal_color))
    track_color = next(
        int(c) for c in colors_present(g, exclude={bg})
        if int(c) not in {int(terminal_color), int(station_color)}
    )
    return int(station_color), int(track_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract only the tree diameter path.

    Rule:
    - Use the tree-diameter logic from combo 6.
    - Then erase every station and track pixel that is not on that diameter path.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    station_color, track_color = _infer_station_and_track_colors(g)
    solved = np.asarray(_COMBO6.main(g), dtype=int)
    out = solved.copy()
    out[out == int(station_color)] = int(bg)
    out[out == int(track_color)] = int(bg)
    return out


def generate_input() -> np.ndarray:
    return np.asarray(_COMBO6.generate_input(), dtype=int)
