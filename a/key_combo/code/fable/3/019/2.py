from __future__ import annotations

from collections import Counter
import random

import numpy as np


Segment = dict[str, int]

BG = 0
COLOR_POOL = tuple(range(1, 10))

# Generator bounds.
COUNT_MIN, COUNT_MAX = 2, 4
GRID_MIN_SIZE, GRID_MAX_SIZE = 10, 26
ROWCOL_BORDER = 2
ENDPOINT_BORDER = 1
MIN_ROWCOL_GAP = 2
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color, matching arc_utils semantics."""
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).reshape(-1)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _parse_segments(grid: np.ndarray, bg: int) -> tuple[list[Segment], list[Segment]]:
    horizontals: list[Segment] = []
    verticals: list[Segment] = []

    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == int(color))
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"

        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1:
            c_start, c_end = sorted((int(c0), int(c1)))
            horizontals.append({"color": int(color), "row": int(r0), "c0": c_start, "c1": c_end})
        else:
            assert c0 == c1, "same-color markers must share a row or a column"
            r_start, r_end = sorted((int(r0), int(r1)))
            verticals.append({"color": int(color), "col": int(c0), "r0": r_start, "r1": r_end})

    assert horizontals, "expected at least one horizontal segment"
    assert verticals, "expected at least one vertical segment"

    h_rows = [seg["row"] for seg in horizontals]
    v_cols = [seg["col"] for seg in verticals]
    assert len(set(h_rows)) == len(h_rows), "horizontal segments must occupy distinct rows"
    assert len(set(v_cols)) == len(v_cols), "vertical segments must occupy distinct columns"

    horizontals.sort(key=lambda seg: seg["row"])
    verticals.sort(key=lambda seg: seg["col"])
    return horizontals, verticals


def _relation_matrix(horizontals: list[Segment], verticals: list[Segment]) -> np.ndarray:
    out = np.zeros((len(horizontals), len(verticals)), dtype=int)
    for i, h_seg in enumerate(horizontals):
        for j, v_seg in enumerate(verticals):
            if (
                h_seg["c0"] <= v_seg["col"] <= h_seg["c1"]
                and v_seg["r0"] <= h_seg["row"] <= v_seg["r1"]
            ):
                out[i, j] = v_seg["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the hidden horizontal/vertical segments into a vertical-colored intersection matrix.

    Input structure:
    - Black background with sparse colored marker pixels, with each foreground color appearing exactly twice.
    - Each same-color pair is aligned either on one row or on one column, defining an implicit straight segment.
    - Different horizontal segments occupy different rows, and different vertical segments occupy different columns.

    Rule:
    - Reconstruct every implicit horizontal and vertical segment from its endpoint pair.
    - Sort horizontal segments by row from top to bottom and vertical segments by column from left to right.
    - Build a new grid whose cell (i,j) is the vertical segment color when those two segments intersect, otherwise black.

    Key insight:
    - The output is not a crop of the scene but an incidence summary: endpoints define hidden segments, and the matrix records only which horizontal/vertical pairs cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    horizontals, verticals = _parse_segments(g, bg)
    return _relation_matrix(horizontals, verticals)


def _sample_spaced_positions(limit: int, count: int, border: int = ROWCOL_BORDER) -> list[int]:
    candidates = list(range(border, limit - border))
    if len(candidates) < count:
        raise ValueError("not enough candidate positions")

    for _ in range(200):
        chosen = sorted(random.sample(candidates, count))
        if all(chosen[i + 1] - chosen[i] >= MIN_ROWCOL_GAP for i in range(count - 1)):
            return chosen
    raise ValueError("failed to sample spaced positions")


def _sample_index_interval(count: int) -> tuple[int, int]:
    length = random.randint(1, count - 1)
    start = random.randint(0, count - length)
    return start, start + length - 1


def _sample_horizontal_segments(
    width: int, colors: list[int], h_rows: list[int], v_cols: list[int]
) -> list[Segment]:
    horizontals: list[Segment] = []
    for row, color in zip(h_rows, colors):
        left_idx, right_idx = _sample_index_interval(len(v_cols))
        left_lo = ENDPOINT_BORDER if left_idx == 0 else v_cols[left_idx - 1] + 1
        left_hi = v_cols[left_idx] - 1
        right_lo = v_cols[right_idx] + 1
        right_hi = width - ENDPOINT_BORDER - 1 if right_idx == len(v_cols) - 1 else v_cols[right_idx + 1] - 1
        if left_lo > left_hi or right_lo > right_hi:
            raise ValueError("invalid horizontal endpoint range")
        c0 = random.randint(left_lo, left_hi)
        c1 = random.randint(right_lo, right_hi)
        horizontals.append({"color": int(color), "row": int(row), "c0": int(c0), "c1": int(c1)})
    horizontals.sort(key=lambda seg: seg["row"])
    return horizontals


def _sample_vertical_segments(height: int, width: int, colors: list[int], h_rows: list[int], v_cols: list[int]) -> list[Segment]:
    verticals: list[Segment] = []
    for col, color in zip(v_cols, colors):
        top_idx, bottom_idx = _sample_index_interval(len(h_rows))
        top_lo = ENDPOINT_BORDER if top_idx == 0 else h_rows[top_idx - 1] + 1
        top_hi = h_rows[top_idx] - 1
        bottom_lo = h_rows[bottom_idx] + 1
        bottom_hi = height - ENDPOINT_BORDER - 1 if bottom_idx == len(h_rows) - 1 else h_rows[bottom_idx + 1] - 1
        if top_lo > top_hi or bottom_lo > bottom_hi:
            raise ValueError("invalid vertical endpoint range")
        r0 = random.randint(top_lo, top_hi)
        r1 = random.randint(bottom_lo, bottom_hi)
        verticals.append({"color": int(color), "col": int(col), "r0": int(r0), "r1": int(r1)})
    verticals.sort(key=lambda seg: seg["col"])
    return verticals


def _render_input(height: int, width: int, horizontals: list[Segment], verticals: list[Segment]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    for seg in horizontals:
        for col in (seg["c0"], seg["c1"]):
            if grid[seg["row"], col] != BG:
                return None
            grid[seg["row"], col] = seg["color"]
    for seg in verticals:
        for row in (seg["r0"], seg["r1"]):
            if grid[row, seg["col"]] != BG:
                return None
            grid[row, seg["col"]] = seg["color"]
    return grid


def _matrix_is_informative(matrix: np.ndarray) -> bool:
    n_h, n_v = map(int, matrix.shape)
    present = matrix != BG
    row_sums = present.sum(axis=1)
    col_sums = present.sum(axis=0)

    if np.any(row_sums == 0) or np.any(row_sums == n_v):
        return False
    if np.any(col_sums == 0) or np.any(col_sums == n_h):
        return False
    if row_sums.min() != 1 or row_sums.max() < 2:
        return False
    if col_sums.min() != 1 or col_sums.max() < 2:
        return False

    row_patterns = {tuple(row.tolist()) for row in present}
    col_patterns = {tuple(col.tolist()) for col in present.T}
    return len(row_patterns) >= 2 and len(col_patterns) >= 2


def _specs_match_parsed_segments(
    grid: np.ndarray, horizontals: list[Segment], verticals: list[Segment]
) -> bool:
    parsed_h, parsed_v = _parse_segments(grid, BG)
    return parsed_h == horizontals and parsed_v == verticals


def _gen_attempt() -> np.ndarray | None:
    n_h = random.randint(COUNT_MIN, COUNT_MAX)
    n_v = random.randint(COUNT_MIN, COUNT_MAX)
    if n_h == 2 and n_v == 2:
        return None

    height = random.randint(max(GRID_MIN_SIZE, 2 * n_h + 5), GRID_MAX_SIZE)
    width = random.randint(max(GRID_MIN_SIZE, 2 * n_v + 5), GRID_MAX_SIZE)
    if max(height, width) > 30:
        return None

    try:
        h_rows = _sample_spaced_positions(height, n_h)
        v_cols = _sample_spaced_positions(width, n_v)
    except ValueError:
        return None

    colors = random.sample(COLOR_POOL, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]

    try:
        horizontals = _sample_horizontal_segments(width, h_colors, h_rows, v_cols)
        verticals = _sample_vertical_segments(height, width, v_colors, h_rows, v_cols)
    except ValueError:
        return None

    expected = _relation_matrix(horizontals, verticals)
    if not _matrix_is_informative(expected):
        return None

    grid = _render_input(height, width, horizontals, verticals)
    if grid is None:
        return None
    if not _specs_match_parsed_segments(grid, horizontals, verticals):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid intersection-matrix input")
