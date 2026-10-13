from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
EDGE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (9, 15)
MARKER_HEIGHT_RANGE = (2, 4)
ACTIVE_COUNT_RANGE = (3, 6)
MAX_ATTEMPTS = 500


def _marker_info(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey marker segment"

    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "grey marker must be one column wide"
    col = int(cols[0])
    h, w = grid.shape
    assert 0 < col < w - 1, "grey marker cannot be on the boundary"

    rows = np.sort(coords[:, 0])
    run_len = len(rows)
    assert np.array_equal(rows, np.arange(run_len)), "grey marker must be top-aligned"
    assert run_len == int(coords.shape[0]), "grey marker must be contiguous and one-pixel wide"
    assert np.all(grid[run_len:, col] == BG), "cells below the grey segment must be black"
    return run_len, col


def _active_rows(grid: np.ndarray) -> list[int]:
    h, _w = grid.shape
    rows: list[int] = []
    for r in range(h):
        left = int(grid[r, 0])
        right = int(grid[r, -1])
        if left != BG and left == right:
            rows.append(r)
    return rows


def _apply_rule(grid: np.ndarray, fill_len: int, *, match_required: bool = True, full_row: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == GREY] = BG
    h, w = g.shape

    for r in range(h):
        left = int(g[r, 0])
        right = int(g[r, w - 1])
        has_pair = left != BG and right != BG
        is_active = has_pair and (left == right if match_required else True)
        if not is_active:
            continue

        color = left if left == right else left
        if full_row:
            out[r, 1 : w - 1] = color
            continue

        left_stop = min(w - 1, 1 + int(fill_len))
        right_start = max(1, w - 1 - int(fill_len))
        out[r, 1:left_stop] = color
        out[r, right_start : w - 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill matched boundary rows inward from both sides, with the fill radius set by a grey marker height.

    Input structure:
    - The grid has black background.
    - Some rows have colored pixels on both horizontal boundaries; some boundary pairs match, some differ, and some rows are blank.
    - One interior column contains a top-aligned vertical grey segment whose height is the global length marker.

    Rule:
    - Measure the grey segment height h.
    - A row is active iff its leftmost and rightmost boundary pixels are the same non-black color.
    - On each active row, keep the boundary markers and fill inward up to h cells from the left boundary and up to h cells from the right boundary.
    - Non-active rows stay unchanged, and the grey marker segment is erased to black in the output.

    Key insight:
    - Endpoint color equality chooses which rows can change, while the separate grey marker supplies the exact fill distance, so both clues are jointly necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fill_len, _marker_col = _marker_info(g)
    return _apply_rule(g, fill_len)


def _sample_width(fill_len: int) -> int:
    min_width = max(8, 2 * int(fill_len) + 3)
    max_width = min(15, 2 * int(fill_len) + 8)
    return random.randint(min_width, max_width)


def _assign_active_colors(count: int) -> list[int]:
    n_colors = random.randint(2, min(4, count))
    palette = random.sample(EDGE_COLORS, n_colors)
    colors = palette.copy()
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return [int(color) for color in colors]


def _pick_rows(h: int, fill_len: int) -> tuple[list[int], list[int], list[int]] | None:
    active_max = min(ACTIVE_COUNT_RANGE[1], h - 2)
    if active_max < ACTIVE_COUNT_RANGE[0]:
        return None

    active_count = random.randint(ACTIVE_COUNT_RANGE[0], active_max)
    remaining = h - active_count
    mismatch_max = min(4, remaining - 1)
    if mismatch_max < 1:
        return None
    mismatch_count = random.randint(1, mismatch_max)
    blank_count = h - active_count - mismatch_count
    if blank_count < 1:
        return None

    top_rows = list(range(fill_len))
    lower_rows = list(range(fill_len, h))
    if not top_rows or not lower_rows:
        return None

    active_rows = {
        random.choice(top_rows),
        random.choice(lower_rows),
    }
    pool = [r for r in range(h) if r not in active_rows]
    need = active_count - len(active_rows)
    if need > len(pool):
        return None
    active_rows.update(random.sample(pool, need))

    leftover = [r for r in range(h) if r not in active_rows]
    mismatch_rows = set(random.sample(leftover, mismatch_count))
    blank_rows = [r for r in leftover if r not in mismatch_rows]

    return sorted(active_rows), sorted(mismatch_rows), sorted(blank_rows)


def _mismatch_pair(active_palette: list[int]) -> tuple[int, int]:
    left = random.choice(active_palette if random.random() < 0.7 else EDGE_COLORS)
    right_choices = [color for color in EDGE_COLORS if color != left]
    if random.random() < 0.7:
        preferred = [color for color in right_choices if color in active_palette]
        if preferred:
            right_choices = preferred
    right = random.choice(right_choices)
    return int(left), int(right)


def _is_valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    h, w = map(int, g.shape)
    try:
        fill_len, marker_col = _marker_info(g)
    except AssertionError:
        return False

    if not (2 <= fill_len <= 4):
        return False
    if w < 2 * fill_len + 3:
        return False
    if not (fill_len < marker_col < w - 1 - fill_len):
        return False

    inner = g[:, 1:-1]
    non_black_inner = np.argwhere(inner != BG)
    if non_black_inner.size:
        for rr, cc in non_black_inner:
            if int(inner[rr, cc]) != GREY or int(cc + 1) != marker_col or int(rr) >= fill_len:
                return False

    active_rows: list[int] = []
    mismatch_rows: list[int] = []
    blank_rows: list[int] = []
    for r in range(h):
        left = int(g[r, 0])
        right = int(g[r, -1])
        if left == BG and right == BG:
            blank_rows.append(r)
        elif left != BG and right != BG:
            if left == right:
                active_rows.append(r)
            else:
                mismatch_rows.append(r)
        else:
            return False

    if len(active_rows) < 3 or not mismatch_rows or not blank_rows:
        return False
    if len({int(g[r, 0]) for r in active_rows}) < 2:
        return False
    if not any(r < fill_len for r in active_rows):
        return False
    if not any(r >= fill_len for r in active_rows):
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if np.any(out == GREY):
        return False
    if not np.array_equal(out[:, [0, -1]], g[:, [0, -1]]):
        return False

    gap_slice = slice(fill_len + 1, w - 1 - fill_len)
    for r in active_rows:
        if not np.all(out[r, 1 : 1 + fill_len] == int(g[r, 0])):
            return False
        if not np.all(out[r, w - 1 - fill_len : w - 1] == int(g[r, 0])):
            return False
        if gap_slice.start < gap_slice.stop and not np.all(out[r, gap_slice] == BG):
            return False

    shorter = _apply_rule(g, fill_len - 1)
    longer = _apply_rule(g, fill_len, full_row=True)
    mismatch_fill = _apply_rule(g, fill_len, match_required=False)
    left_only = _apply_rule(g, fill_len)
    for r in _active_rows(g):
        left_only[r, w - 1 - fill_len : w - 1] = BG

    if np.array_equal(out, shorter):
        return False
    if np.array_equal(out, longer):
        return False
    if np.array_equal(out, mismatch_fill):
        return False
    if np.array_equal(out, left_only):
        return False

    return True


def _attempt() -> np.ndarray | None:
    fill_len = random.randint(*MARKER_HEIGHT_RANGE)
    width = _sample_width(fill_len)
    height = random.randint(max(HEIGHT_RANGE[0], fill_len + 4), HEIGHT_RANGE[1])

    row_groups = _pick_rows(height, fill_len)
    if row_groups is None:
        return None
    active_rows, mismatch_rows, _blank_rows = row_groups

    marker_col = random.randint(fill_len + 1, width - fill_len - 2)
    grid = np.zeros((height, width), dtype=int)
    grid[:fill_len, marker_col] = GREY

    active_colors = _assign_active_colors(len(active_rows))
    active_palette = sorted(set(active_colors))
    for r, color in zip(active_rows, active_colors):
        grid[r, 0] = int(color)
        grid[r, -1] = int(color)

    for r in mismatch_rows:
        left, right = _mismatch_pair(active_palette)
        grid[r, 0] = left
        grid[r, -1] = right

    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid same-edge-limited-fill input")
