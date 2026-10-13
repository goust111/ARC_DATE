from __future__ import annotations

import random

import numpy as np


BLACK = 0
MIN_TOWERS = 3
MAX_TOWERS = 6
HEIGHT_RANGE = (2, 8)
GAP_RANGE = (2, 6)
EDGE_MARGIN_RANGE = (1, 4)
TOP_PADDING_RANGE = (0, 4)
GEN_RETRIES = 300


def _detect_towers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return towers as (column, color, height), ordered left-to-right."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    towers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for c in range(w):
        column = g[:, c]
        nz_rows = np.flatnonzero(column != BLACK)
        if nz_rows.size == 0:
            continue

        colors = np.unique(column[nz_rows])
        assert colors.size == 1, "each tower column must use a single color"
        color = int(colors[0])
        assert color not in seen_colors, "tower colors must be unique"
        seen_colors.add(color)

        height = int(nz_rows.size)
        start = h - height
        assert int(nz_rows[0]) == start, "tower must be a bottom-aligned contiguous bar"
        assert np.all(column[:start] == BLACK), "cells above tower must be black"
        assert np.all(column[start:] == color), "tower body must be solid"

        towers.append((int(c), color, height))

    assert len(towers) >= 2, "expected at least two towers"
    return towers


def _paint_signal(out: np.ndarray, left: int, right: int, color: int, height: int) -> None:
    """Paint a horizontal signal band on columns [left, right)."""
    if left >= right or height <= 0:
        return
    out[-height:, left:right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast horizontal signals from bottom-aligned single-column towers.

    Input structure:
    - Black background.
    - Several one-column towers, each a solid vertical bar growing upward from the bottom.
    - Every tower has a unique color and a distinct height.

    Rule:
    - Each tower emits horizontally across its own height range.
    - Between adjacent towers, both signals cover the full gap; the taller tower's color wins.
    - Outside the outermost towers, the nearest tower fills to the grid edge.
    - Tower pixels themselves stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    towers = _detect_towers(g)

    out = g.copy()

    left_col, left_color, left_height = towers[0]
    _paint_signal(out, 0, left_col, left_color, left_height)

    for (c0, color0, height0), (c1, color1, height1) in zip(towers, towers[1:]):
        if height0 >= height1:
            winner_color, winner_height = color0, height0
        else:
            winner_color, winner_height = color1, height1
        _paint_signal(out, c0 + 1, c1, winner_color, winner_height)

    right_col, right_color, right_height = towers[-1]
    _paint_signal(out, right_col + 1, g.shape[1], right_color, right_height)

    return out


def _arrange_heights(rng: random.Random, heights: list[int]) -> list[int]:
    mode = rng.choice(["random", "monotone", "zigzag", "valley"])
    values = sorted(heights)

    if mode == "random":
        rng.shuffle(values)
        return values

    if mode == "monotone":
        if rng.random() < 0.5:
            return values
        return list(reversed(values))

    arranged: list[int] = []
    lo, hi = 0, len(values) - 1
    if mode == "zigzag":
        take_hi = True
        while lo <= hi:
            if take_hi:
                arranged.append(values[hi])
                hi -= 1
            else:
                arranged.append(values[lo])
                lo += 1
            take_hi = not take_hi
        return arranged

    take_hi = False
    while lo <= hi:
        if take_hi:
            arranged.append(values[hi])
            hi -= 1
        else:
            arranged.append(values[lo])
            lo += 1
        take_hi = not take_hi
    return arranged


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tower_count = rng.randint(MIN_TOWERS, MAX_TOWERS)
    colors = rng.sample(range(1, 10), tower_count)
    heights = _arrange_heights(
        rng,
        rng.sample(range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1), tower_count),
    )

    left_margin = rng.randint(*EDGE_MARGIN_RANGE)
    right_margin = rng.randint(*EDGE_MARGIN_RANGE)
    gaps = [rng.randint(*GAP_RANGE) for _ in range(tower_count - 1)]
    top_padding = rng.randint(*TOP_PADDING_RANGE)

    grid_h = max(heights) + top_padding
    grid_w = left_margin + right_margin + tower_count + sum(gaps)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    columns: list[int] = []
    col = left_margin
    for i in range(tower_count):
        columns.append(col)
        if i + 1 < tower_count:
            col += 1 + gaps[i]

    for c, color, height in zip(columns, colors, heights):
        grid[-height:, c] = color

    try:
        towers = _detect_towers(grid)
    except AssertionError:
        return None

    if len(towers) != tower_count:
        return None

    if not all(towers[i + 1][0] - towers[i][0] >= 3 for i in range(len(towers) - 1)):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
