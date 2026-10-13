from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _add_bar(bar_colors: dict[int, int], index: int, color: int) -> None:
    previous = bar_colors.setdefault(int(index), int(color))
    assert previous == int(color), "bar axis cannot receive two different colors"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored domino marker into a full row or column bar, with horizontal bars covering vertical bars at crossings.

    Input structure:
    - The background is the dominant color.
    - Every foreground object is a 4-connected same-color domino of exactly two pixels.
    - A horizontal domino marks a row to paint; a vertical domino marks a column to paint.

    Rule:
    - Paint each marked column entirely with the color of its vertical domino marker.
    - Paint each marked row entirely with the color of its horizontal domino marker.
    - Rows are painted after columns, so row bars overwrite column bars at intersections.

    Key insight:
    - The two-pixel orientation is the instruction: horizontal markers emit row bars, vertical markers emit column bars, and the only layering cue is row-over-column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected at least one foreground domino"

    for comp in components:
        color = int(comp["color"])
        coords = comp["coords"]
        assert len(coords) == 2, "each foreground object must be a two-pixel domino"
        r0, c0, r1, c1 = comp["bbox"]

        if r0 == r1 and c1 == c0 + 1:
            _add_bar(row_colors, r0, color)
        elif c0 == c1 and r1 == r0 + 1:
            _add_bar(col_colors, c0, color)
        else:
            raise AssertionError("domino marker must be horizontal or vertical")

    out = np.full_like(g, int(bg))
    for c, color in col_colors.items():
        out[:, c] = int(color)
    for r, color in row_colors.items():
        out[r, :] = int(color)
    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
_GEN_BG = 7
_GEN_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
_GEN_SIZE_MIN = 7
_GEN_SIZE_MAX = 10

# Bar counts and color diversity.
_GEN_H_COUNT_MIN = 2
_GEN_H_COUNT_MAX = 3
_GEN_V_COUNT_MIN = 1
_GEN_V_COUNT_MAX = 4
_GEN_TOTAL_COUNT_MIN = 4
_GEN_TOTAL_COUNT_MAX = 7
_GEN_UNIQUE_COLOR_MIN = 3
_GEN_UNIQUE_COLOR_MAX = 5

# Marker placement.
_GEN_ROW_MARGIN = 1
_GEN_DOMINO_LEN = 2
_GEN_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_MAX_ATTEMPTS = 300

def _gen_counts() -> tuple[int, int]:
    h_count = random.randint(_GEN_H_COUNT_MIN, _GEN_H_COUNT_MAX)
    min_v = max(_GEN_V_COUNT_MIN, _GEN_TOTAL_COUNT_MIN - h_count)
    max_v = min(_GEN_V_COUNT_MAX, _GEN_TOTAL_COUNT_MAX - h_count)
    return h_count, random.randint(min_v, max_v)

def _gen_bar_colors(n_bars: int) -> list[int]:
    n_unique = random.randint(_GEN_UNIQUE_COLOR_MIN, min(_GEN_UNIQUE_COLOR_MAX, n_bars))
    chosen = random.sample(_GEN_COLORS, n_unique)
    colors = chosen + [random.choice(chosen) for _ in range(n_bars - n_unique)]
    random.shuffle(colors)
    return colors

def _gen_bars(n: int) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    h_count, v_count = _gen_counts()
    rows = random.sample(range(_GEN_ROW_MARGIN, n - _GEN_ROW_MARGIN), h_count)
    cols = random.sample(range(n), v_count)
    colors = _gen_bar_colors(h_count + v_count)
    h_bars = [("h", r, colors[i]) for i, r in enumerate(rows)]
    v_bars = [("v", c, colors[h_count + i]) for i, c in enumerate(cols)]
    return h_bars, v_bars

def _gen_marker_options(n: int, orient: str, index: int, color: int):
    if orient == "h":
        opts = [
            (index, c, [(index, c), (index, c + 1)], np.full((1, _GEN_DOMINO_LEN), color, dtype=int))
            for c in range(n - _GEN_DOMINO_LEN + 1)
        ]
    else:
        opts = [
            (r, index, [(r, index), (r + 1, index)], np.full((_GEN_DOMINO_LEN, 1), color, dtype=int))
            for r in range(n - _GEN_DOMINO_LEN + 1)
        ]
    random.shuffle(opts)
    return opts

def _gen_can_place(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if grid[r, c] != _GEN_BG:
            return False
        for dr, dc in _GEN_DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set and grid[rr, cc] == color:
                return False
    return True

def _gen_render_output(n: int, h_bars: list[tuple[str, int, int]], v_bars: list[tuple[str, int, int]]) -> np.ndarray:
    out = np.full((n, n), _GEN_BG, dtype=int)
    for _, c, color in v_bars:
        out[:, c] = color
    for _, r, color in h_bars:
        out[r, :] = color
    return out

def _gen_place_markers(n: int, bars: list[tuple[str, int, int]]) -> np.ndarray | None:
    grid = np.full((n, n), _GEN_BG, dtype=int)
    ordered = list(bars)
    random.shuffle(ordered)

    def backtrack(i: int) -> bool:
        if i == len(ordered):
            return True
        orient, index, color = ordered[i]
        for r0, c0, cells, patch in _gen_marker_options(n, orient, index, color):
            if not _gen_can_place(grid, cells, color):
                continue
            paste(grid, patch, r0, c0)
            if backtrack(i + 1):
                return True
            for r, c in cells:
                grid[r, c] = _GEN_BG
        return False

    return grid if backtrack(0) else None

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(_GEN_SIZE_MIN, _GEN_SIZE_MAX)
    h_bars, v_bars = _gen_bars(n)
    out = _gen_render_output(n, h_bars, v_bars)
    inp = _gen_place_markers(n, h_bars + v_bars)
    if inp is None:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 9344f635 input")
