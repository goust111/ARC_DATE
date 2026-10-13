from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
BAR_COLORS = (1, 2, 3, 4)
SEED_COLORS = (6, 8, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_seed(grid: np.ndarray, divider_c: int) -> np.ndarray:
    left = grid[:, :divider_c]
    box = _bbox(left != BG)
    assert box is not None, "expected a seed pattern in the left region"
    r0, c0, r1, c1 = box
    assert r1 == grid.shape[0] - 1, "seed pattern must be bottom-aligned"
    seed = left[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert seed.shape[0] >= 5 and seed.shape[1] >= 4, "seed pattern is too small"
    return seed


def _extract_bars(grid: np.ndarray, divider_c: int) -> list[tuple[int, int, int]]:
    right = grid[:, divider_c + 1 :]
    assert right.size > 0, "expected a bar region to the right of the divider"

    base_rows = np.flatnonzero(np.all(right == ORANGE, axis=1))
    assert len(base_rows) == 1, "expected exactly one orange baseline row"
    base_r = int(base_rows[0])

    bars: list[tuple[int, int, int]] = []
    for local_c in range(right.shape[1]):
        col = right[:base_r, local_c]
        nz = np.flatnonzero(col != BG)
        if len(nz) == 0:
            continue
        top_r = int(nz[0])
        color = int(col[top_r])
        assert color in BAR_COLORS, "bars must use the designated visible palette"
        assert np.all(col[nz] == color), "each bar must be monochrome"
        assert np.array_equal(nz, np.arange(top_r, base_r)), "each bar must be a contiguous column touching the baseline"
        bars.append((int(local_c), int(base_r - top_r), color))

    assert 3 <= len(bars) <= 4, "expected 3 or 4 bars"
    heights = [h for _x, h, _c in bars]
    colors = [c for _x, _h, c in bars]
    assert len(set(heights)) == len(heights), "bar heights must be distinct"
    assert len(set(colors)) == len(colors), "bar colors must be distinct"
    return sorted(bars, key=lambda item: item[0])


def _render_output(
    seed: np.ndarray,
    bars: Iterable[tuple[int, int, int]],
    *,
    row_mode: str = "bottom",
    sort_mode: str = "x",
    recolor: bool = True,
) -> np.ndarray:
    bar_list = list(bars)
    if sort_mode == "x":
        ordered = sorted(bar_list, key=lambda item: item[0])
    elif sort_mode == "height_asc":
        ordered = sorted(bar_list, key=lambda item: item[1])
    elif sort_mode == "height_desc":
        ordered = sorted(bar_list, key=lambda item: -item[1])
    else:
        raise ValueError(f"unsupported sort_mode: {sort_mode}")

    out_rows = []
    height = seed.shape[0]
    for _x, h, color in ordered:
        assert 1 <= h <= height, "bar height must index a valid seed row"
        if row_mode == "bottom":
            row = seed[-h].copy()
        elif row_mode == "top":
            row = seed[h - 1].copy()
        else:
            raise ValueError(f"unsupported row_mode: {row_mode}")
        if recolor:
            row[row != BG] = int(color)
        out_rows.append(row)
    return np.stack(out_rows, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """Stack seed rows selected by bar heights, ordered by bar positions, and recolored by bar colors.

    Input structure:
    - A black background grid is split by one full-height grey divider column.
    - The left region contains one bottom-aligned rectangular seed pattern whose interior may mix black and non-black pixels.
    - The right region contains one full-width orange baseline and 3-4 distinct single-column bars rising upward from it.

    Rule:
    - Read each bar's height, left-to-right position, and color from the right region.
    - Sort bars by position from left to right; this determines the output row order.
    - For each bar, select the seed row indexed by its height from the bottom of the seed, then recolor every non-black cell in that row to the bar's color.
    - Crop away everything else and stack the recolored rows into the output.

    Key insight:
    - The same bars simultaneously encode which seed rows to extract, in what vertical order to place them, and which visible color each extracted row should become.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(divider_cols) == 1, "expected exactly one full-height grey divider"
    divider_c = int(divider_cols[0])

    seed = _extract_seed(g, divider_c)
    bars = _extract_bars(g, divider_c)
    return _render_output(seed, bars, row_mode="bottom", sort_mode="x", recolor=True)


def _sample_seed(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    codes = np.arange(1, 1 << width, dtype=int)
    for _ in range(100):
        picked = rng.choice(codes, size=height, replace=False)
        masks = ((picked[:, None] >> np.arange(width)) & 1).astype(bool)
        if not masks[:, 0].any() or not masks[:, -1].any():
            continue
        if not np.any(masks.sum(axis=1) >= 2):
            continue
        seed = np.zeros((height, width), dtype=int)
        for r in range(height):
            fg_cols = np.flatnonzero(masks[r])
            seed[r, fg_cols] = rng.choice(SEED_COLORS, size=len(fg_cols), replace=True)
        return seed
    return None


def _sample_bar_columns(rng: np.random.Generator, count: int) -> tuple[list[int], int]:
    left_margin = int(rng.integers(0, 3))
    cols = [left_margin]
    cur = left_margin
    for _ in range(count - 1):
        cur += int(rng.integers(2, 4))
        cols.append(cur)
    right_width = int(cols[-1] + 1 + rng.integers(1, 4))
    return cols, right_width


def _sample_heights(rng: np.random.Generator, seed_h: int, count: int) -> list[int] | None:
    for _ in range(100):
        heights = [int(v) for v in rng.choice(np.arange(1, seed_h + 1), size=count, replace=False)]
        if heights == sorted(heights) or heights == sorted(heights, reverse=True):
            continue
        return heights
    return None


def _semantic_ok(seed: np.ndarray, bars: list[tuple[int, int, int]]) -> bool:
    correct = _render_output(seed, bars, row_mode="bottom", sort_mode="x", recolor=True)
    alternatives = [
        _render_output(seed, bars, row_mode="top", sort_mode="x", recolor=True),
        _render_output(seed, bars, row_mode="bottom", sort_mode="height_asc", recolor=True),
        _render_output(seed, bars, row_mode="bottom", sort_mode="height_desc", recolor=True),
        _render_output(seed, bars, row_mode="bottom", sort_mode="x", recolor=False),
    ]
    return all(not np.array_equal(correct, alt) for alt in alternatives)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    seed_h = int(rng.integers(5, 8))
    seed_w = int(rng.integers(4, 7))
    bar_count = int(rng.choice([3, 4]))
    if bar_count >= seed_h:
        return None

    seed = _sample_seed(rng, seed_h, seed_w)
    if seed is None:
        return None

    heights = _sample_heights(rng, seed_h, bar_count)
    if heights is None:
        return None

    colors = [int(v) for v in rng.choice(BAR_COLORS, size=bar_count, replace=False)]
    bar_cols, right_width = _sample_bar_columns(rng, bar_count)
    bars = [(x, h, c) for x, h, c in zip(bar_cols, heights, colors)]
    if not _semantic_ok(seed, bars):
        return None

    top_margin = int(rng.integers(1, 5))
    grid_h = int(seed_h + top_margin)
    seed_left = int(rng.integers(0, 3))
    gap_before_divider = int(rng.integers(1, 3))
    divider_c = int(seed_left + seed_w + gap_before_divider)
    grid_w = int(divider_c + 1 + right_width)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    seed_top = grid_h - seed_h
    grid[seed_top:, seed_left : seed_left + seed_w] = seed
    grid[:, divider_c] = GREY
    grid[-1, divider_c + 1 :] = ORANGE

    for local_c, height, color in bars:
        global_c = divider_c + 1 + local_c
        grid[grid_h - 1 - height : grid_h - 1, global_c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _render_output(seed, bars, row_mode="bottom", sort_mode="x", recolor=True)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
