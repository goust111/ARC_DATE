from __future__ import annotations

import random

import numpy as np

import os
import sys
from pathlib import Path


_arc_utils_path = None
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    for _cand_dir in (_p / "new" / "solvers", _p / "solvers"):
        _cand = _cand_dir / "arc_utils.py"
        if _cand.exists():
            _arc_utils_path = str(_cand_dir)
            break
    if _arc_utils_path is not None:
        break
if _arc_utils_path and _arc_utils_path not in sys.path:
    sys.path.insert(0, _arc_utils_path)

from arc_utils import get_bg



def _infer_period(states: list[tuple[int, ...]]) -> int:
    if not states:
        return 1
    K = len(states)
    for p in range(1, K + 1):
        ok = True
        for i in range(K):
            if states[i] != states[i % p]:
                ok = False
                break
        if ok:
            return p
    return K


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    bg = int(get_bg(g))
    baseline_row = H - 1
    baseline_color = int(g[baseline_row, 0])

    col_has_fg: list[bool] = []
    for c in range(W):
        col = g[:baseline_row, c]
        col_has_fg.append(np.any((col != bg) & (col != baseline_color)))
    if not any(col_has_fg):
        return g.copy()

    K = max(i for i, v in enumerate(col_has_fg) if v) + 1
    states = []
    for c in range(K):
        rows = tuple(r for r in range(baseline_row) if g[r, c] not in (bg, baseline_color))
        states.append(rows)

    p = _infer_period(states)
    if p <= 0:
        return g.copy()

    out = g.copy()
    for c in range(K, W):
        state_rows = states[c % p]
        for r in state_rows:
            out[r, c] = 1

    # gravity settle per column above baseline
    for c in range(W):
        pixels = [int(v) for v in out[:baseline_row, c] if v not in (bg, baseline_color)]
        out[:baseline_row, c] = bg
        if not pixels:
            continue
        start = baseline_row - len(pixels)
        for i, color in enumerate(pixels):
            out[start + i, c] = color

    return out


def generate_input() -> np.ndarray:
    H = random.randint(12, 12)
    bg = 0
    baseline_color = 5

    period = random.randint(2, 4)
    states = []
    for _ in range(period):
        rows_count = random.randint(1, 3)
        rows = sorted(random.sample(range(H - 1), rows_count))
        states.append(rows)

    # enforce actual period (not degenerated)
    if all(s == states[0] for s in states):
        states[0] = tuple(sorted((np.array(states[0]) + 1) % (H - 1))
)
    state_colors: list[list[tuple[int, int]]] = []
    for idx, state in enumerate(states):
        row_colors = []
        for r in state:
            row_colors.append((int(r), 1))
        state_colors.append(row_colors)

    pattern_len = period * random.randint(2, 3)
    colors_for_cells = [1, 4]
    W = random.randint(pattern_len + 2, min(15, pattern_len + 4))

    g = np.full((H, W), bg, dtype=int)
    g[H - 1, :] = baseline_color

    for c in range(pattern_len):
        for r, _ in state_colors[c % period]:
            g[r, c] = random.choice(colors_for_cells)

    try:
        _ = main(g)
    except Exception:
        raise RuntimeError("failed to generate valid puzzle-7 input")

    return g
