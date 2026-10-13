from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TOP_FG = 3
BOTTOM_FG = 7
HEAVY = 2
LIGHT = 4
HALF = 12
GRID_SHAPE = (HALF, HALF)
GEN_RETRIES = 800


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _grow_connected(
    height: int,
    width: int,
    target_size: int,
    required: set[tuple[int, int]] | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    required = set() if required is None else set(required)
    forbidden = set() if forbidden is None else set(forbidden)
    if required & forbidden:
        return None
    if target_size < len(required):
        return None

    if required:
        cells = set(required)
    else:
        cells = {(random.randrange(height), random.randrange(width))}

    while len(cells) < target_size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(height, width, nr, nc):
                    continue
                if (nr, nc) in cells or (nr, nc) in forbidden:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _generate_local_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        height = random.choice((3, 4))
        width = random.choice((3, 4))
        top_size = random.randint(4, min(7, height * width - 1))
        bottom_size = random.randint(4, min(7, height * width - 1))
        overlap_min = max(1, top_size + bottom_size - height * width)
        overlap_max = min(3, top_size - 1, bottom_size - 1)
        if overlap_min > overlap_max:
            continue
        overlap_size = random.randint(overlap_min, overlap_max)

        overlap = _grow_connected(height, width, overlap_size)
        if overlap is None:
            continue
        top_cells = _grow_connected(height, width, top_size, required=overlap)
        if top_cells is None:
            continue
        top_only = top_cells - overlap
        bottom_cells = _grow_connected(
            height,
            width,
            bottom_size,
            required=overlap,
            forbidden=top_only,
        )
        if bottom_cells is None:
            continue

        top = np.zeros((height, width), dtype=int)
        bottom = np.zeros((height, width), dtype=int)
        for r, c in top_cells:
            top[r, c] = 1
        for r, c in bottom_cells:
            bottom[r, c] = 1

        if _component_sizes(top.astype(bool)) != [int(top.sum())]:
            continue
        if _component_sizes(bottom.astype(bool)) != [int(bottom.sum())]:
            continue
        return top, bottom
    return None


def _fits_with_gap(occupied: np.ndarray, sprite: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    sh, sw = sprite.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + sh + gap)
    c1 = min(w, left + sw + gap)
    return not occupied[r0:r1, c0:c1].any()


def _stacked_compare(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(top, dtype=int)
    both = (top == TOP_FG) & (bottom == BOTTOM_FG)
    exactly_one = (top == TOP_FG) ^ (bottom == BOTTOM_FG)
    out[both] = HEAVY
    out[exactly_one] = LIGHT
    return out


def _drop_once(state: np.ndarray) -> tuple[np.ndarray, bool]:
    out = state.copy()
    h, w = out.shape
    moved = False
    for r in range(h - 2, -1, -1):
        for c in range(w):
            color = int(out[r, c])
            if color == HEAVY:
                if out[r + 1, c] == BG:
                    out[r, c] = BG
                    out[r + 1, c] = HEAVY
                    moved = True
            elif color == LIGHT:
                if r + 2 >= h:
                    continue
                if out[r + 1, c] == BG and out[r + 2, c] == BG:
                    out[r, c] = BG
                    out[r + 1, c] = LIGHT
                    moved = True
    return out, moved


def _drop_weighted(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    while True:
        out, moved = _drop_once(out)
        if not moved:
            return out


def _drop_uniform(state: np.ndarray, color: int) -> np.ndarray:
    work = np.where(state != BG, color, BG)
    return _drop_weighted(work)


def _drop_independent_overlay(state: np.ndarray) -> np.ndarray:
    heavy_only = _drop_weighted(np.where(state == HEAVY, HEAVY, BG))
    light_only = _drop_weighted(np.where(state == LIGHT, LIGHT, BG))
    out = np.zeros_like(state)
    out[light_only == LIGHT] = LIGHT
    out[heavy_only == HEAVY] = HEAVY
    return out


def _column_has_interaction(state: np.ndarray) -> bool:
    for c in range(state.shape[1]):
        rows = np.flatnonzero(state[:, c] != BG)
        if len(rows) < 2:
            continue
        colors = {int(state[r, c]) for r in rows}
        if colors == {HEAVY, LIGHT}:
            return True
    return False


def _horizontally_interleaved(state: np.ndarray) -> bool:
    heavy_cols = sorted({int(c) for c in np.flatnonzero(np.any(state == HEAVY, axis=0))})
    light_cols = sorted({int(c) for c in np.flatnonzero(np.any(state == LIGHT, axis=0))})
    if not heavy_cols or not light_cols:
        return False
    return min(heavy_cols) < max(light_cols) and min(light_cols) < max(heavy_cols)


def _validate_case(top: np.ndarray, bottom: np.ndarray) -> bool:
    if top.shape != GRID_SHAPE or bottom.shape != GRID_SHAPE:
        return False

    top_sizes = sorted(_component_sizes(top == TOP_FG))
    bottom_sizes = sorted(_component_sizes(bottom == BOTTOM_FG))
    if len(top_sizes) not in (2, 3) or len(bottom_sizes) not in (2, 3):
        return False
    if any(size < 4 or size > 8 for size in top_sizes + bottom_sizes):
        return False

    compare = _stacked_compare(top, bottom)
    heavy_count = int(np.sum(compare == HEAVY))
    top_only_count = int(np.sum((top == TOP_FG) & (bottom == BG)))
    bottom_only_count = int(np.sum((top == BG) & (bottom == BOTTOM_FG)))
    if heavy_count < 3 or top_only_count < 3 or bottom_only_count < 3:
        return False
    if not _column_has_interaction(compare):
        return False
    if not _horizontally_interleaved(compare):
        return False

    settled = _drop_weighted(compare)
    if np.array_equal(settled, compare):
        return False
    if np.array_equal(settled, _drop_uniform(compare, HEAVY)):
        return False
    if np.array_equal(settled, _drop_uniform(compare, LIGHT)):
        return False
    if np.array_equal(settled, _drop_independent_overlay(compare)):
        return False

    return True


def _place_components() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice((2, 3))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    heights: list[int] = []
    widths: list[int] = []

    for _ in range(count):
        pair = _generate_local_pair()
        if pair is None:
            return None
        top_mask, _bottom_mask = pair
        pairs.append(pair)
        heights.append(int(top_mask.shape[0]))
        widths.append(int(top_mask.shape[1]))

    min_total_height = sum(heights) + (count - 1)
    if min_total_height > HALF:
        return None

    remaining = HALF - min_total_height
    extra_before = random.randint(0, remaining)
    remaining -= extra_before
    extra_between = [1] * (count - 1)
    for i in range(count - 1):
        add = random.randint(0, remaining)
        extra_between[i] += add
        remaining -= add
    extra_after = remaining

    row_starts: list[int] = []
    row = extra_before
    for i, height in enumerate(heights):
        row_starts.append(row)
        row += height
        if i < count - 1:
            row += extra_between[i]
    if row + extra_after != HALF:
        return None

    col_starts: list[int] = []
    first_col = random.randint(0, HALF - widths[0])
    col_starts.append(first_col)
    for i in range(1, count):
        prev_left = col_starts[i - 1]
        prev_right = prev_left + widths[i - 1] - 1
        cur_width = widths[i]
        lo = max(0, prev_left - cur_width + 1)
        hi = min(HALF - cur_width, prev_right)
        if lo > hi:
            return None
        candidate_cols = list(range(lo, hi + 1))
        random.shuffle(candidate_cols)
        picked = None
        for col in candidate_cols:
            cur_right = col + cur_width - 1
            overlap = min(prev_right, cur_right) - max(prev_left, col) + 1
            if overlap >= 1:
                picked = col
                break
        if picked is None:
            return None
        col_starts.append(picked)

    top = np.zeros(GRID_SHAPE, dtype=int)
    bottom = np.zeros(GRID_SHAPE, dtype=int)
    top_occ = np.zeros(GRID_SHAPE, dtype=bool)
    bottom_occ = np.zeros(GRID_SHAPE, dtype=bool)

    for (top_mask, bottom_mask), row, col in zip(pairs, row_starts, col_starts):
        if not _fits_with_gap(top_occ, top_mask.astype(bool), row, col):
            return None
        if not _fits_with_gap(bottom_occ, bottom_mask.astype(bool), row, col):
            return None

        sh, sw = top_mask.shape
        top_slice = top[row : row + sh, col : col + sw]
        bottom_slice = bottom[row : row + sh, col : col + sw]
        top_slice[top_mask.astype(bool)] = TOP_FG
        bottom_slice[bottom_mask.astype(bool)] = BOTTOM_FG
        top_occ[row : row + sh, col : col + sw] |= top_mask.astype(bool)
        bottom_occ[row : row + sh, col : col + sw] |= bottom_mask.astype(bool)

    return top, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two stacked masks cell by cell, convert overlap into heavy pixels and exclusives into light pixels, then let the two weights fall under different gravity rules.

    Input structure:
    - The grid is split into two equally sized square halves stacked vertically.
    - The top half uses black background with green foreground pixels; the bottom half uses black background with orange foreground pixels.
    - Matching relative positions across the two halves encode overlap, while non-matching foreground pixels encode single-layer cells.

    Rule:
    - Compare the two halves cell by cell to build one square output canvas of the same half-size.
    - Put red at positions where both halves have foreground, yellow where exactly one half has foreground, and black elsewhere.
    - Repeatedly scan the square canvas from bottom row to top row and left to right within each row.
    - A red heavy pixel moves down one cell whenever the cell directly below is empty.
    - A yellow light pixel moves down one cell only when both the next cell and the cell two rows below are empty, so it preserves a one-row black buffer above the floor or any settled colored pixel.
    - Keep repeating sweeps until no pixel can move.

    Key insight:
    - The stacked-mask comparison does not directly produce the final picture; it assigns each resulting pixel a hidden weight, and the final arrangement emerges from the interaction between heavy direct settling and light buffered settling in the same columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "expected two stacked halves"
    half = height // 2
    assert half == width, "expected square halves"

    top = g[:half, :]
    bottom = g[half:, :]
    assert set(map(int, np.unique(top))) <= {BG, TOP_FG}, "unexpected colors in top half"
    assert set(map(int, np.unique(bottom))) <= {BG, BOTTOM_FG}, "unexpected colors in bottom half"

    compare = _stacked_compare(top, bottom)
    return _drop_weighted(compare)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        placed = _place_components()
        if placed is None:
            continue
        top, bottom = placed
        if not _validate_case(top, bottom):
            continue
        grid = np.vstack([top, bottom]).astype(int)
        solved = main(grid)
        if solved.shape != GRID_SHAPE:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
