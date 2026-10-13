from __future__ import annotations

import numpy as np

from arc_utils import Color


def _terrain_starts(g: np.ndarray) -> np.ndarray:
    empty = int(g[0, 0])
    terrain = int(g[-1, 0])
    upper = int(g[1, 0])
    assert len({empty, terrain, upper}) == 3, "expected distinct empty, upper, and terrain colors"
    assert np.all(g[0, :] == empty), "top row must be a solid empty separator"
    assert np.all(g[1, :] == upper), "first field row must be the upper fill color"
    assert np.all(g[-1, :] == terrain), "bottom row must be a solid terrain color"
    assert set(map(int, np.unique(g))) == {empty, upper, terrain}, "expected exactly three input colors"

    starts = []
    for col in range(g.shape[1]):
        terrain_rows = np.flatnonzero(g[:, col] == terrain)
        assert terrain_rows.size > 0, "each column must contain terrain"
        start = int(terrain_rows[0])
        assert start > 1, "marked terrain columns must leave upper cells above them"
        assert np.all(g[1:start, col] == upper), "cells above terrain must be upper fill"
        assert np.all(g[start:, col] == terrain), "terrain must be a contiguous column suffix"
        starts.append(start)
    return np.array(starts, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

    Input structure:
    - The first row is a solid empty separator.
    - Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
    - The bottom row is all terrain, so every column has a well-defined first terrain row.

    Rule:
    - Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
    - Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
    - Preserve the empty separator, terrain cells, and all other upper fill cells.

    Key insight:
    - The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 2, "expected at least two rows and columns"

    starts = _terrain_starts(g)

    deepest_start = int(starts.max())
    deepest_cols = np.flatnonzero(starts == deepest_start)
    assert deepest_cols.size == 1, "expected a unique deepest upper run"
    maroon_col = int(deepest_cols[0])

    highest_start = int(starts.min())
    yellow_col = int(np.flatnonzero(starts == highest_start)[-1])
    assert maroon_col != yellow_col, "highest and deepest columns must be different"

    out = g.copy()
    out[1:deepest_start, maroon_col] = Color.MAROON
    out[1:highest_start, yellow_col] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------

# Canvas/profile bounds.
CANVAS_HEIGHT_RANGE = (10, 15)  # np.random.randint high bound is exclusive.
CANVAS_WIDTH_RANGE = (10, 15)
MIN_TERRAIN_HEIGHT_CHOICES = (1, 2)
TERRAIN_SPAN_CHOICES = (4, 5)
UPPER_ROW_MARGIN = 2
# Terrain-walk controls.
MAX_ADJACENT_STEP = 2
STEP_DELTA_WEIGHTS = {0: 3, 1: 6, 2: 1}
PEAK_TIE_PROBABILITY = 0.5
PROFILE_INTERIOR_MIN = 1
# Colors and retries.
CANONICAL_COLOR_PROBABILITY = 0.55
CANONICAL_COLORS = (0, 1, 6)
ALT_COLOR_PALETTE = (0, 1, 2, 3, 5, 6, 7, 8)
VALLEY_MARK_COLOR = Color.MAROON
PEAK_MARK_COLOR = Color.YELLOW
GENERATION_RETRIES = 300


def _gen_weighted_choice(values, weights):
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(values, p=probs))


def _gen_segment(start, end, steps, lo, hi):
    if steps == 0:
        return [int(start)] if start == end else None

    vals = [int(start)]
    cur = int(start)
    end = int(end)
    for pos in range(1, steps):
        remaining = steps - pos
        candidates = []
        weights = []
        for nxt in range(lo, hi + 1):
            delta = abs(nxt - cur)
            if delta > MAX_ADJACENT_STEP:
                continue
            if abs(end - nxt) > MAX_ADJACENT_STEP * remaining:
                continue
            candidates.append(nxt)
            weights.append(STEP_DELTA_WEIGHTS[delta])
        if not candidates:
            return None
        cur = _gen_weighted_choice(candidates, weights)
        vals.append(cur)

    if abs(end - cur) > MAX_ADJACENT_STEP:
        return None
    vals.append(end)
    return vals


def _gen_endpoint_value(anchor_value, distance, span):
    candidates = [
        value
        for value in range(PROFILE_INTERIOR_MIN, span)
        if abs(value - anchor_value) <= MAX_ADJACENT_STEP * distance
    ]
    if not candidates:
        return None
    return int(np.random.choice(candidates))


def _valid_normalized_profile(profile, width, span, peak_count):
    return (
        profile.size == width
        and np.count_nonzero(profile == 0) == 1
        and np.count_nonzero(profile == span) == peak_count
        and np.max(np.abs(np.diff(profile))) <= MAX_ADJACENT_STEP
    )


def _gen_profile(width, min_height, span):
    """Sample a height profile with one minimum and one or two maximum peaks."""
    valley = int(np.random.randint(width))
    min_peak_dist = (span + MAX_ADJACENT_STEP - 1) // MAX_ADJACENT_STEP
    peak_candidates = [
        col
        for col in range(width)
        if col != valley and abs(col - valley) >= min_peak_dist
    ]
    if not peak_candidates:
        return None

    can_tie = len(peak_candidates) >= 2 and np.random.random() < PEAK_TIE_PROBABILITY
    peak_count = 2 if can_tie else 1
    peak_cols = np.random.choice(peak_candidates, size=peak_count, replace=False)

    controls = {valley: 0}
    for col in peak_cols:
        controls[int(col)] = span

    if 0 not in controls:
        right = min(col for col in controls if col > 0)
        value = _gen_endpoint_value(controls[right], right, span)
        if value is None:
            return None
        controls[0] = value

    last = width - 1
    if last not in controls:
        left = max(col for col in controls if col < last)
        value = _gen_endpoint_value(controls[left], last - left, span)
        if value is None:
            return None
        controls[last] = value

    cols = sorted(controls)
    normalized = []
    for left, right in zip(cols, cols[1:]):
        segment = _gen_segment(
            controls[left], controls[right], right - left, PROFILE_INTERIOR_MIN, span - 1
        )
        if segment is None:
            return None
        if normalized:
            normalized.extend(segment[1:])
        else:
            normalized.extend(segment)

    arr = np.array(normalized, dtype=int)
    if not _valid_normalized_profile(arr, width, span, peak_count):
        return None
    return arr + min_height


def _gen_colors():
    if np.random.random() < CANONICAL_COLOR_PROBABILITY:
        return CANONICAL_COLORS
    colors = np.random.choice(ALT_COLOR_PALETTE, size=3, replace=False)
    return tuple(int(color) for color in colors)


def _render_terrain_grid(height, width, terrain_heights, colors):
    empty, upper, terrain = colors
    grid = np.full((height, width), upper, dtype=int)
    grid[0, :] = empty
    for col, terrain_height in enumerate(terrain_heights):
        grid[height - int(terrain_height) :, col] = terrain
    return grid


def _expected_output(grid, terrain_heights, min_height, max_height):
    valley_cols = np.flatnonzero(terrain_heights == min_height)
    peak_cols = np.flatnonzero(terrain_heights == max_height)
    if valley_cols.size != 1 or peak_cols.size not in (1, 2):
        return None
    valley_col = int(valley_cols[0])
    yellow_col = int(peak_cols[-1])
    if valley_col == yellow_col:
        return None
    height = grid.shape[0]
    expected = grid.copy()
    expected[1 : height - min_height, valley_col] = VALLEY_MARK_COLOR
    expected[1 : height - max_height, yellow_col] = PEAK_MARK_COLOR
    return expected


def _oracle_accepts(grid, expected):
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    height = int(np.random.randint(*CANVAS_HEIGHT_RANGE))
    width = int(np.random.randint(*CANVAS_WIDTH_RANGE))
    min_height = int(np.random.choice(MIN_TERRAIN_HEIGHT_CHOICES))
    span = int(np.random.choice(TERRAIN_SPAN_CHOICES))

    terrain_heights = _gen_profile(width, min_height, span)
    if terrain_heights is None:
        return None
    max_height = min_height + span
    if max_height > height - UPPER_ROW_MARGIN:
        return None

    colors = _gen_colors()
    grid = _render_terrain_grid(height, width, terrain_heights, colors)
    expected = _expected_output(grid, terrain_heights, min_height, max_height)
    if expected is None or not _oracle_accepts(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e45ef808 input")
