from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


PALETTE = tuple(Color.ALL_COLORS)
MAX_GENERATION_ATTEMPTS = 400


def _parse_bars(grid: np.ndarray) -> tuple[int, list[dict[str, object]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    H, W = map(int, g.shape)
    bg = int(get_bg(g))

    bars: list[dict[str, object]] = []
    cap_color: int | None = None
    start_index: int | None = None

    for c in range(W):
        rows = np.flatnonzero(g[:, c] != bg)
        if len(rows) == 0:
            continue

        assert int(rows[-1]) == H - 1, "every bar must stand on the bottom edge"
        expected_rows = np.arange(int(rows[0]), H)
        assert np.array_equal(rows, expected_rows), "bar columns must be vertically contiguous"

        values = g[rows, c].astype(int)
        top_row = int(rows[0])
        cap_row: int | None = None

        if np.all(values == int(values[0])):
            body_color = int(values[0])
            body_rows = rows
            height = len(rows)
        else:
            assert len(values) >= 2, "a capped bar needs at least one body cell"
            assert int(values[0]) != int(values[1]), "the cap color must differ from the bar color"
            assert np.all(values[1:] == int(values[1])), "the bar body must be monochrome"
            body_color = int(values[1])
            body_rows = rows[1:]
            height = len(body_rows)
            cap_row = top_row
            if cap_color is None:
                cap_color = int(values[0])
                start_index = len(bars)
            else:
                assert cap_color == int(values[0]), "all cap markers must use one color"
                raise AssertionError("expected exactly one capped bar")

        assert height >= 1
        bars.append(
            {
                "col": int(c),
                "body_rows": body_rows.copy(),
                "height": int(height),
                "body_color": int(body_color),
                "cap_row": cap_row,
            }
        )

    assert bars, "expected at least one bar"
    assert cap_color is not None and start_index is not None, "expected exactly one capped bar"

    heights = [int(bar["height"]) for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be distinct"

    body_colors = [int(bar["body_color"]) for bar in bars]
    assert len(set(body_colors)) == len(body_colors), "bar colors must be distinct"

    return bg, bars, int(cap_color), int(start_index)


def _terminal_index(bars: list[dict[str, object]], start_index: int) -> int:
    order = sorted(range(len(bars)), key=lambda idx: int(bars[idx]["height"]), reverse=True)
    rank_of = {idx: rank for rank, idx in enumerate(order, start=1)}

    current = int(start_index)
    visited = {current}
    while True:
        next_index = int(rank_of[current]) - 1
        if next_index in visited:
            return current
        visited.add(next_index)
        current = next_index


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Follow the hidden rank-to-position permutation from the capped bar and recolor the terminal bar.

    Input structure:
    - Non-background columns are 1-cell-wide bottom-aligned vertical bars.
    - Bar heights are all distinct, and bar colors are all distinct.
    - Exactly one bar has one extra cap cell directly above its body in a special marker color.

    Rule:
    - Rank bars by height, with rank 1 meaning tallest.
    - A bar points to the bar whose left-to-right position equals its height rank.
    - Start from the capped bar and follow these jumps until the next jump would revisit a bar.
    - Recolor the last visited bar body to the cap color; everything else stays unchanged.

    Key insight:
    - The height order is not the answer itself. It encodes a permutation over bar positions, and the terminal state is only visible after tracing that permutation from the marked start.
    """
    grid = np.asarray(input_grid, dtype=int)
    _, bars, cap_color, start_index = _parse_bars(grid)
    terminal = _terminal_index(bars, start_index)

    out = grid.copy()
    bar = bars[terminal]
    out[np.asarray(bar["body_rows"], dtype=int), int(bar["col"])] = int(cap_color)
    return out


def _trace_length(permutation: list[int], start_index: int) -> int:
    current = int(start_index)
    visited = {current}
    length = 1
    while True:
        nxt = int(permutation[current])
        if nxt in visited:
            return length
        visited.add(nxt)
        current = nxt
        length += 1


def _build_case() -> np.ndarray | None:
    n = random.randint(3, 8)
    permutation = random.sample(range(n), k=n)
    start_index = random.randrange(n)
    if n >= 5 and _trace_length(permutation, start_index) == 1 and random.random() < 0.75:
        return None

    max_height = random.randint(max(n, 4), 12)
    top_margin = random.randint(1, 4)
    H = max_height + top_margin + 1
    if H > 30:
        return None

    heights_desc = sorted(random.sample(range(1, max_height + 1), k=n), reverse=True)
    heights = [int(heights_desc[permutation[idx]]) for idx in range(n)]

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gaps = [random.randint(0, 2) for _ in range(n - 1)]
    W = left_margin + right_margin + n + sum(gaps)
    if W > 30:
        return None

    colors = random.sample(PALETTE, k=n + 2)
    bg = int(colors[0])
    cap_color = int(colors[1])
    bar_colors = [int(color) for color in colors[2:]]

    grid = np.full((H, W), bg, dtype=int)

    cols: list[int] = []
    c = left_margin
    for idx in range(n):
        cols.append(c)
        c += 1
        if idx < n - 1:
            c += gaps[idx]

    for idx, col in enumerate(cols):
        height = int(heights[idx])
        body_top = H - height
        grid[body_top:H, col] = int(bar_colors[idx])
        if idx == start_index:
            grid[body_top - 1, col] = cap_color

    assert grid.shape[0] <= 30 and grid.shape[1] <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ranked-bar permutation instance")
