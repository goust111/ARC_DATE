from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

BLOCK_SIZE = 2
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _true_intervals(bits: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(bits.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(bits) - 1))
    return intervals


def _parse_lattice(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    block_mask = grid == GREY
    bbox = _bbox(block_mask)
    assert bbox is not None, "expected a grey lattice"

    row_bands = _true_intervals(block_mask.any(axis=1))
    col_bands = _true_intervals(block_mask.any(axis=0))
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a rectangular array of blocks"
    assert all(r1 - r0 + 1 == BLOCK_SIZE for r0, r1 in row_bands), "expected 2-row block bands"
    assert all(c1 - c0 + 1 == BLOCK_SIZE for c0, c1 in col_bands), "expected 2-col block bands"

    row_mask = np.zeros(grid.shape[0], dtype=bool)
    col_mask = np.zeros(grid.shape[1], dtype=bool)
    for r0, r1 in row_bands:
        row_mask[r0 : r1 + 1] = True
    for c0, c1 in col_bands:
        col_mask[c0 : c1 + 1] = True
    assert np.array_equal(block_mask, row_mask[:, None] & col_mask[None, :]), "grey cells must form a complete block lattice"
    return bbox, row_bands, col_bands


def _landing_row(base_grid: np.ndarray, src_r: int, col: int, bg: int) -> int:
    rows = np.flatnonzero(base_grid[:, col] == bg)
    assert rows.size > 0, "expected at least one landing cell in the column"
    return int(rows[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop orange pixels to the lowest empty cell in their column and recolor them by whether the landing site lies inside the grey lattice bbox.

    Input structure:
    - The background is black, and a full rectangular lattice of grey 2x2 blocks occupies the lower part of the grid.
    - The block lattice is separated by 1-pixel-wide black gutters, with extra black gutter space allowed around the lattice.
    - Several orange single pixels float above the lattice in distinct columns that contain no grey blockers.

    Rule:
    - Keep the grey lattice unchanged and treat each orange pixel independently.
    - Remove the orange pixels, then for each one find the lowest black cell anywhere in the same column and move the pixel there.
    - If the landing coordinate lies inside the grey lattice bounding box, recolor that settled pixel red; otherwise recolor it blue.
    - If an orange pixel is already at the lowest black cell in its column, leave it orange in place.

    Key insight:
    - The final color is not determined by the source position alone: the orange pixel must first settle under gravity, and only that landing coordinate can be tested against the lattice bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    bbox, _row_bands, _col_bands = _parse_lattice(g)
    r0, c0, r1, c1 = bbox

    orange_coords = np.argwhere(g == ORANGE)
    cols = [int(c) for _r, c in orange_coords]
    assert len(cols) == len(set(cols)), "orange pixels must occupy distinct columns"

    out = g.copy()
    out[g == ORANGE] = bg

    for rr, cc in orange_coords:
        src_r = int(rr)
        col = int(cc)
        dest_r = _landing_row(out, src_r, col, bg)
        if dest_r <= src_r:
            out[src_r, col] = ORANGE
            continue
        inside_bbox = r0 <= dest_r <= r1 and c0 <= col <= c1
        out[dest_r, col] = RED if inside_bbox else BLUE
    return out


def _axis_starts(start_margin: int, gaps: list[int]) -> list[int]:
    starts = [int(start_margin)]
    for gap in gaps:
        starts.append(int(starts[-1] + BLOCK_SIZE + gap))
    return starts


def _render_lattice(
    h: int,
    w: int,
    row_starts: list[int],
    col_starts: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = GREY
    return grid


def _expected_from_input(grid: np.ndarray) -> np.ndarray:
    bbox = _bbox(grid == GREY)
    assert bbox is not None
    r0, c0, r1, c1 = bbox

    expected = grid.copy()
    expected[expected == ORANGE] = BG
    for rr, cc in np.argwhere(grid == ORANGE):
        src_r = int(rr)
        col = int(cc)
        dest_r = _landing_row(expected, src_r, col, BG)
        if dest_r <= src_r:
            expected[src_r, col] = ORANGE
            continue
        expected[dest_r, col] = RED if (r0 <= dest_r <= r1 and c0 <= col <= c1) else BLUE
    return expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_row_bands = int(rng.integers(2, 5))
    n_col_bands = int(rng.integers(3, 6))

    row_gaps = [int(rng.integers(1, 3)) for _ in range(n_row_bands - 1)]
    col_gaps = [int(rng.integers(1, 3)) for _ in range(n_col_bands - 1)]

    top_margin = int(rng.integers(4, 10))
    bottom_margin = 0 if bool(rng.integers(0, 2)) else int(rng.integers(1, 4))
    left_margin = int(rng.integers(2, 5))
    right_margin = int(rng.integers(2, 5))

    lattice_h = n_row_bands * BLOCK_SIZE + sum(row_gaps)
    lattice_w = n_col_bands * BLOCK_SIZE + sum(col_gaps)
    h = top_margin + lattice_h + bottom_margin
    w = left_margin + lattice_w + right_margin
    if not (15 <= h <= 30 and 15 <= w <= 30):
        return None

    row_starts = _axis_starts(top_margin, row_gaps)
    col_starts = _axis_starts(left_margin, col_gaps)
    grid = _render_lattice(h, w, row_starts, col_starts)

    bbox = (
        row_starts[0],
        col_starts[0],
        row_starts[-1] + BLOCK_SIZE - 1,
        col_starts[-1] + BLOCK_SIZE - 1,
    )
    bbox_r0, bbox_c0, bbox_r1, bbox_c1 = bbox

    internal_cols: list[int] = []
    for left_start, right_start in zip(col_starts, col_starts[1:]):
        internal_cols.extend(range(left_start + BLOCK_SIZE, right_start))
    exterior_cols = list(range(0, bbox_c0)) + list(range(bbox_c1 + 1, w))

    if len(internal_cols) < 2 or len(exterior_cols) < 2:
        return None

    n_orange = int(rng.integers(4, 7))
    max_internal = min(len(internal_cols), n_orange - 1)
    if max_internal < 2:
        return None
    n_internal = int(rng.integers(2, max_internal + 1))
    n_exterior = n_orange - n_internal
    if n_exterior > len(exterior_cols):
        return None

    chosen_internal = rng.choice(np.array(internal_cols, dtype=int), size=n_internal, replace=False).tolist()
    chosen_exterior = rng.choice(np.array(exterior_cols, dtype=int), size=n_exterior, replace=False).tolist()
    chosen_cols = [int(c) for c in chosen_internal + chosen_exterior]
    rng.shuffle(chosen_cols)

    top_rows = np.arange(0, min(9, top_margin), dtype=int)
    if top_rows.size == 0:
        return None
    for col in chosen_cols:
        row = int(rng.choice(top_rows))
        grid[row, col] = ORANGE

    expected = _expected_from_input(grid)
    if not np.array_equal(main(grid), expected):
        return None

    orange_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == ORANGE)]

    # Semantic branch checks:
    # - internal columns should flip between red and blue depending on whether the lattice touches the bottom.
    # - exterior columns should always land blue.
    for col in chosen_cols:
        if np.any(grid[:, col] == GREY):
            return None
    for col in chosen_internal:
        dest_r = h - 1
        landed_color = int(expected[dest_r, col])
        if bottom_margin == 0:
            if landed_color != RED:
                return None
        else:
            if landed_color != BLUE:
                return None
    for col in chosen_exterior:
        if int(expected[h - 1, col]) != BLUE:
            return None

    if bottom_margin == 0:
        if RED not in expected or BLUE not in expected:
            return None
    else:
        if RED in expected:
            return None
        if sum(col in chosen_internal for col in chosen_cols) < 2:
            return None

    # Keep the lattice clearly in the lower portion and the orange pixels in the upper portion.
    if bbox_r0 < h // 3:
        return None
    if any(r >= bbox_r0 for r, _c in orange_coords):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-lattice input")
