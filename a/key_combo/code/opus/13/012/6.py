from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
BAR = 3
FILL = 9

HEIGHT_POOL = (4, 6, 8, 10, 12, 14, 16)
COUNT_RANGE = (5, 7)
GAP_RANGE = (2, 4)
MAX_ATTEMPTS = 600


def _contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for value in rows[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    nonempty_cols = [int(c) for c in np.flatnonzero(np.any(g != BG, axis=0))]
    assert 5 <= len(nonempty_cols) <= 7, "expected 5-7 vertical bars"
    assert all(nonempty_cols[i + 1] - nonempty_cols[i] >= 3 for i in range(len(nonempty_cols) - 1))

    bars: list[dict] = []
    seen_heights: set[int] = set()
    for c in nonempty_cols:
        rows = np.flatnonzero(g[:, c] != BG)
        assert rows.size > 0
        runs = _contiguous_runs(rows)
        assert len(runs) == 1, "each bar must be one contiguous vertical segment"
        r0, r1 = runs[0]
        height = r1 - r0 + 1
        assert height % 2 == 0 and height >= 4

        colors = set(int(v) for v in g[r0 : r1 + 1, c])
        assert colors <= {BAR, MARKER}
        assert int(np.count_nonzero(g[:, max(0, c - 1) : c] != BG)) == 0
        assert int(np.count_nonzero(g[:, c + 1 : min(g.shape[1], c + 2)] != BG)) == 0
        assert height not in seen_heights, "bar heights must be distinct"
        seen_heights.add(height)

        bars.append({"c": c, "r0": r0, "r1": r1, "h": height})
    return bars


def _marker_info(grid: np.ndarray, bars: list[dict]) -> tuple[int, str]:
    g = np.asarray(grid, dtype=int)
    sources = [i for i, bar in enumerate(bars) if np.any(g[bar["r0"] : bar["r1"] + 1, bar["c"]] == MARKER)]
    assert len(sources) == 1, "expected exactly one marked bar"
    source_idx = int(sources[0])
    bar = bars[source_idx]
    red_rows = np.flatnonzero(g[:, bar["c"]] == MARKER)
    assert red_rows.size == bar["h"] // 2
    assert np.all(np.diff(red_rows) == 1)
    half = bar["h"] // 2
    if int(red_rows[0]) == bar["r0"] and int(red_rows[-1]) == bar["r0"] + half - 1:
        return source_idx, "top"
    if int(red_rows[0]) == bar["r1"] - half + 1 and int(red_rows[-1]) == bar["r1"]:
        return source_idx, "bottom"
    raise AssertionError("marker must occupy exactly one half of its bar")


def _paint_half(grid: np.ndarray, bar: dict, half: str, color: int) -> None:
    mid = bar["h"] // 2
    if half == "top":
        grid[bar["r0"] : bar["r0"] + mid, bar["c"]] = int(color)
    else:
        grid[bar["r1"] - mid + 1 : bar["r1"] + 1, bar["c"]] = int(color)


def _migrate_marker(grid: np.ndarray, bars: list[dict]) -> tuple[np.ndarray, int, int, str]:
    out = np.asarray(grid, dtype=int).copy()
    source_idx, half = _marker_info(out, bars)
    source = bars[source_idx]
    _paint_half(out, source, half, BAR)

    target_height = source["h"] + 2
    targets = [i for i, bar in enumerate(bars) if bar["h"] == target_height]
    assert len(targets) == 1, "expected a unique next-taller target bar"
    target_idx = int(targets[0])
    _paint_half(out, bars[target_idx], half, MARKER)
    return out, source_idx, target_idx, half


def _regions_from_grid(grid: np.ndarray, bars: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        colors = g[bar["r0"] : bar["r1"] + 1, bar["c"]]
        start = bar["r0"]
        current = int(colors[0])
        for offset, value in enumerate(colors[1:], start=1):
            value = int(value)
            if value != current:
                regions.append(
                    {
                        "bar": bar_idx,
                        "c": bar["c"],
                        "r0": start,
                        "r1": bar["r0"] + offset - 1,
                        "color": current,
                    }
                )
                start = bar["r0"] + offset
                current = value
        regions.append({"bar": bar_idx, "c": bar["c"], "r0": start, "r1": bar["r1"], "color": current})
    return regions


def _fill_mask(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    occupied = g != BG
    mask = np.zeros_like(g, dtype=bool)
    regions = _regions_from_grid(g, bars)

    for color in (BAR, MARKER):
        layer = sorted(
            [region for region in regions if region["color"] == color],
            key=lambda region: (region["c"], region["r0"], region["r1"]),
        )
        for i, left in enumerate(layer):
            gap0_min = left["c"] + 1
            for j in range(i + 1, len(layer)):
                right = layer[j]
                if right["c"] <= gap0_min:
                    continue
                row0 = max(left["r0"], right["r0"])
                row1 = min(left["r1"], right["r1"])
                if row0 > row1:
                    continue

                gap0 = left["c"] + 1
                gap1 = right["c"] - 1
                blocked = False
                for k, blocker in enumerate(layer):
                    if k in (i, j):
                        continue
                    if gap0 <= blocker["c"] <= gap1 and max(row0, blocker["r0"]) <= min(row1, blocker["r1"]):
                        blocked = True
                        break
                if blocked:
                    continue
                mask[row0 : row1 + 1, gap0 : gap1 + 1] = True

    mask &= ~occupied
    return mask


def _apply_fill(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[_fill_mask(out, bars)] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the half-bar marker to the next taller column, then bridge same-color row overlaps with maroon.

    Input structure:
    - The grid is a black background containing 5-7 one-cell-wide vertical bars in green, each with a distinct even height.
    - Exactly one bar has either its top half or its bottom half recolored red, marking that half-bar segment.
    - Bars are separated by empty columns and are not ordered by height from left to right.

    Rule:
    - Restore the marked red half of its source bar to green, find the unique bar whose height is exactly two cells larger, and recolor the same half of that target bar red.
    - After the move, split every bar into maximal same-color vertical segments and treat each segment as an independent rectangular region.
    - For green regions and red regions separately, fill maroon across the horizontal gap between any two regions whose row spans overlap, unless another region of that same color lies between their columns and overlaps the same rows.
    - Keep all original bar pixels unchanged; only background cells inside eligible gap rectangles become maroon.

    Key insight:
    - The marker move changes which rows belong to each color-layer segment, so the green-layer gap bridges are recomputed on a different region graph after the target column is split and the source column is restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(int(v) for v in np.unique(g)) <= {BG, BAR, MARKER}

    bars = _extract_bars(g)
    moved, _source_idx, _target_idx, _half = _migrate_marker(g, bars)
    return _apply_fill(moved, bars)


def _sample_heights(count: int) -> tuple[list[int], int]:
    while True:
        heights = sorted(random.sample(HEIGHT_POOL, count))
        candidates = [h for h in heights if h + 2 in heights]
        if candidates:
            return heights, int(random.choice(candidates))


def _sample_columns(count: int) -> tuple[list[int], int] | None:
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    used = count + sum(gaps)
    if used > 28:
        return None
    width_lo = max(18, used + 2)
    width_hi = min(30, used + 6)
    if width_lo > width_hi:
        return None
    width = random.randint(width_lo, width_hi)
    start = random.randint(1, width - used - 1)

    cols: list[int] = []
    c = start
    for i in range(count):
        cols.append(c)
        c += 1
        if i < count - 1:
            c += gaps[i]
    return cols, width


def _build_state(
    bars: list[dict],
    shape: tuple[int, int],
    marked_idx: int,
    half: str,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx, bar in enumerate(bars):
        grid[bar["r0"] : bar["r1"] + 1, bar["c"]] = BAR
        if idx == marked_idx:
            _paint_half(grid, bar, half, MARKER)
    return grid


def _attempt_generate() -> np.ndarray | None:
    count = random.randint(*COUNT_RANGE)
    heights_sorted, source_height = _sample_heights(count)
    target_height = source_height + 2

    heights = heights_sorted[:]
    random.shuffle(heights)
    if heights == sorted(heights) or heights == sorted(heights, reverse=True):
        return None

    col_info = _sample_columns(count)
    if col_info is None:
        return None
    cols, width = col_info

    max_height = max(heights)
    height_lo = max(16, max_height + 2)
    height_hi = min(30, max_height + 8)
    if height_lo > height_hi:
        return None
    grid_h = random.randint(height_lo, height_hi)
    baseline = random.randint(max_height, grid_h - 2)

    bars = []
    for h, c in zip(heights, cols):
        r0 = baseline - h + 1
        if r0 < 1:
            return None
        bars.append({"c": c, "r0": r0, "r1": baseline, "h": h})

    by_height = {bar["h"]: idx for idx, bar in enumerate(bars)}
    source_idx = by_height[source_height]
    target_idx = by_height[target_height]
    half = random.choice(("top", "bottom"))

    output_core = _build_state(bars, (grid_h, width), target_idx, half)
    input_grid = _build_state(bars, (grid_h, width), source_idx, half)

    pre_mask = _fill_mask(input_grid, bars)
    post_mask = _fill_mask(output_core, bars)
    if not np.any(post_mask):
        return None
    if not np.any(pre_mask != post_mask):
        return None

    expected = _apply_fill(output_core, bars)
    try:
        actual = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
