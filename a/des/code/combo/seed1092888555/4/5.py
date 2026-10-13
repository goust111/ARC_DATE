import random
from typing import Dict, List, Tuple

import numpy as np


_BG = 0
_COLORS = tuple(range(1, 10))
_GEN_RETRIES = 400


# Each arrow occupies a 2x2 box with one missing corner.
# Direction is derived from the missing-corner position.
_ARROW_SPECS = {
    "right": {
        "missing": (0, 1),
        "cells": ((0, 0), (1, 0), (1, 1)),
        "vertex": (1, 0),
        "axis": "h",
    },
    "down": {
        "missing": (1, 0),
        "cells": ((0, 0), (0, 1), (1, 1)),
        "vertex": (0, 1),
        "axis": "v",
    },
    "left": {
        "missing": (0, 0),
        "cells": ((0, 1), (1, 0), (1, 1)),
        "vertex": (1, 1),
        "axis": "h",
    },
    "up": {
        "missing": (1, 1),
        "cells": ((0, 0), (0, 1), (1, 0)),
        "vertex": (0, 0),
        "axis": "v",
    },
}
_MISSING_TO_DIR = {spec["missing"]: name for name, spec in _ARROW_SPECS.items()}


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_arrows(grid: np.ndarray) -> List[Dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    arrows: List[Dict[str, int]] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) != _BG):
        for comp in _components(g == color):
            if len(comp) != 3:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue

            present = {(r - r0, c - c0) for r, c in comp}
            all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
            missing = tuple((all_cells - present).pop())
            if missing not in _MISSING_TO_DIR:
                continue

            direction = _MISSING_TO_DIR[missing]
            vr, vc = _ARROW_SPECS[direction]["vertex"]
            arrows.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "direction": direction,
                    "line": int(r0 + vr if _ARROW_SPECS[direction]["axis"] == "h" else c0 + vc),
                }
            )
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify 3-cell L-shaped arrows by their missing corner, then convert them into full-edge stripes.

    Rule:
    - Missing top-right -> right  -> horizontal stripe on the arrow vertex row.
    - Missing bottom-left -> down -> vertical stripe on the arrow vertex column.
    - Missing top-left -> left    -> horizontal stripe on the arrow vertex row.
    - Missing bottom-right -> up  -> vertical stripe on the arrow vertex column.
    - Vertical stripes are drawn first; horizontal stripes overwrite them.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), _BG, dtype=int)
    arrows = _extract_arrows(g)

    vertical = [a for a in arrows if _ARROW_SPECS[a["direction"]]["axis"] == "v"]
    horizontal = [a for a in arrows if _ARROW_SPECS[a["direction"]]["axis"] == "h"]

    vertical.sort(key=lambda a: (a["line"], a["r0"], a["c0"], a["color"]))
    horizontal.sort(key=lambda a: (a["line"], a["r0"], a["c0"], a["color"]))

    for arrow in vertical:
        out[:, arrow["line"]] = arrow["color"]
    for arrow in horizontal:
        out[arrow["line"], :] = arrow["color"]
    return out


def _can_place(occ: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + 3)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + 3)
    return not occ[rr0:rr1, cc0:cc1].any()


def _place_arrow(grid: np.ndarray, occ: np.ndarray, r0: int, c0: int, direction: str, color: int) -> None:
    for dr, dc in _ARROW_SPECS[direction]["cells"]:
        grid[r0 + dr, c0 + dc] = color
        occ[r0 + dr, c0 + dc] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(8, 16)
    w = random.randint(8, 16)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    max_arrows = min(7, len(_COLORS), max(2, (h * w) // 18))
    n_arrows = random.randint(2, max_arrows)

    directions = [random.choice(tuple(_ARROW_SPECS)) for _ in range(n_arrows)]
    if len(set(directions)) < min(2, n_arrows):
        return None

    colors = random.sample(_COLORS, n_arrows)
    used_h_rows = set()
    used_v_cols = set()

    order = list(range(n_arrows))
    random.shuffle(order)
    for idx in order:
        direction = directions[idx]
        axis = _ARROW_SPECS[direction]["axis"]
        placed = False
        for _ in range(120):
            r0 = random.randint(0, h - 2)
            c0 = random.randint(0, w - 2)
            vertex_r, vertex_c = _ARROW_SPECS[direction]["vertex"]
            line = r0 + vertex_r if axis == "h" else c0 + vertex_c
            if axis == "h" and line in used_h_rows:
                continue
            if axis == "v" and line in used_v_cols:
                continue
            if not _can_place(occ, r0, c0):
                continue
            _place_arrow(grid, occ, r0, c0, direction, colors[idx])
            if axis == "h":
                used_h_rows.add(line)
            else:
                used_v_cols.add(line)
            placed = True
            break
        if not placed:
            return None

    arrows = _extract_arrows(grid)
    if len(arrows) != n_arrows:
        return None

    axis_set = {_ARROW_SPECS[a["direction"]]["axis"] for a in arrows}
    if axis_set != {"h", "v"}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-stripe input")
