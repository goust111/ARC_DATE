from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_RETRIES = 500
MARKER_RETRIES = 300

GRID_H_RANGE = (11, 17)
GRID_W_RANGE = (11, 17)


def _cross_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return (
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    )


def _parse_input(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small for this task"
    assert set(map(int, np.unique(g))) <= {BG, BLUE, GREEN}, "input uses unsupported colors"

    top_cols = [int(c) for c in np.where(g[0] == BLUE)[0]]
    right_rows = [int(r) for r in np.where(g[:, w - 1] == BLUE)[0] if int(r) != 0]
    assert 2 <= len(top_cols) <= 4, "expected 2-4 top-row blue markers"
    assert 2 <= len(right_rows) <= 4, "expected 2-4 right-column blue markers"

    blue_coords = {tuple(map(int, coord)) for coord in np.argwhere(g == BLUE)}
    allowed_blue = {(0, c) for c in top_cols} | {(r, w - 1) for r in right_rows}
    assert blue_coords == allowed_blue, "blue pixels must lie on the top row or right edge markers"

    green_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == GREEN)]
    assert len(green_coords) == 2, "expected exactly two green pixels"
    for r, c in green_coords:
        assert r != 0 and c != w - 1, "green pixels must not lie on the top row or rightmost column"

    (r1, c1), (r2, c2) = sorted(green_coords)
    assert r1 == r2 or c1 == c2, "green pixels must be horizontally or vertically aligned"
    assert (r1 + r2) % 2 == 0 and (c1 + c2) % 2 == 0, "green midpoint must be integral"
    midpoint = ((r1 + r2) // 2, (c1 + c2) // 2)
    mr, mc = midpoint
    assert 2 <= mr <= h - 3 and 2 <= mc <= w - 3, "midpoint must be at least two cells from every edge"

    return top_cols, right_rows, green_coords, midpoint


def main(grid: np.ndarray) -> np.ndarray:
    """Place a midpoint red cross, stop border rays on its red cells, and keep only the surviving row-column intersections.

    Input structure:
    - The background is black, with 2-4 blue markers on the top row and 2-4 blue markers on the rightmost column.
    - Exactly two interior green pixels are aligned horizontally or vertically and determine an integer midpoint.
    - No other foreground objects are present in the input.

    Rule:
    - Compute the midpoint of the two green pixels and place a red plus-shaped cross there, but only on black cells, so overlapping green cells stay green.
    - From each top-row blue marker, extend a vertical ray downward until just before the first red cross cell in that column, or to the bottom edge if none exists.
    - From each right-edge blue marker, extend a horizontal ray leftward until just before the first red cross cell in that row, or to the left edge if none exists.
    - Paint non-overlapping ray cells gray and ray intersections yellow, while keeping all original blue markers, green pixels, and newly placed red cross cells unchanged.

    Key insight:
    - The green pair does not mark the output directly; it fixes the cross location, and the cross's occupied red cells selectively truncate the Cartesian product formed by the border rays.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    top_cols, right_rows, green_coords, midpoint = _parse_input(g)

    out = g.copy()
    green_set = set(green_coords)
    for r, c in _cross_cells(midpoint):
        if 0 <= r < h and 0 <= c < w and int(out[r, c]) == BG:
            out[r, c] = RED

    vertical_cells: set[tuple[int, int]] = set()
    for c in top_cols:
        for r in range(1, h):
            if int(out[r, c]) == RED:
                break
            vertical_cells.add((r, c))

    horizontal_cells: set[tuple[int, int]] = set()
    for r in right_rows:
        for c in range(w - 2, -1, -1):
            if int(out[r, c]) == RED:
                break
            horizontal_cells.add((r, c))

    both = vertical_cells & horizontal_cells
    for r, c in vertical_cells | horizontal_cells:
        if (r, c) in green_set or int(out[r, c]) in {BLUE, RED, GREEN}:
            continue
        out[r, c] = YELLOW if (r, c) in both else GRAY

    return out


def _independent_expected(
    shape: tuple[int, int],
    top_cols: tuple[int, ...],
    right_rows: tuple[int, ...],
    greens: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for c in top_cols:
        grid[0, c] = BLUE
    for r in right_rows:
        grid[r, w - 1] = BLUE
    for r, c in greens:
        grid[r, c] = GREEN

    (r1, c1), (r2, c2) = greens
    midpoint = ((r1 + r2) // 2, (c1 + c2) // 2)
    green_set = set(greens)
    red_cells = {cell for cell in _cross_cells(midpoint) if cell not in green_set}

    out = grid.copy()
    for r, c in red_cells:
        out[r, c] = RED

    vertical_cells: set[tuple[int, int]] = set()
    for c in top_cols:
        for r in range(1, h):
            if (r, c) in red_cells:
                break
            vertical_cells.add((r, c))

    horizontal_cells: set[tuple[int, int]] = set()
    for r in right_rows:
        for c in range(w - 2, -1, -1):
            if (r, c) in red_cells:
                break
            horizontal_cells.add((r, c))

    both = vertical_cells & horizontal_cells
    for r, c in vertical_cells | horizontal_cells:
        if (r, c) in green_set or (r, c) in red_cells or out[r, c] == BLUE:
            continue
        out[r, c] = YELLOW if (r, c) in both else GRAY

    return out


def _trace_profile(
    shape: tuple[int, int],
    top_cols: tuple[int, ...],
    right_rows: tuple[int, ...],
    greens: tuple[tuple[int, int], tuple[int, int]],
) -> dict[str, object]:
    h, w = shape
    (r1, c1), (r2, c2) = greens
    midpoint = ((r1 + r2) // 2, (c1 + c2) // 2)
    green_set = set(greens)
    red_cells = {cell for cell in _cross_cells(midpoint) if cell not in green_set}
    red_rows = {r for r, _c in red_cells}
    red_cols = {c for _r, c in red_cells}

    vertical_by_col: dict[int, set[tuple[int, int]]] = {}
    for c in top_cols:
        cells: set[tuple[int, int]] = set()
        for r in range(1, h):
            if (r, c) in red_cells:
                break
            cells.add((r, c))
        vertical_by_col[c] = cells

    horizontal_by_row: dict[int, set[tuple[int, int]]] = {}
    for r in right_rows:
        cells: set[tuple[int, int]] = set()
        for c in range(w - 2, -1, -1):
            if (r, c) in red_cells:
                break
            cells.add((r, c))
        horizontal_by_row[r] = cells

    blocked_top = {c for c in top_cols if c in red_cols}
    blocked_right = {r for r in right_rows if r in red_rows}

    pair_cells = {(r, c) for r in right_rows for c in top_cols}
    yellow_pairs = {
        (r, c)
        for r in right_rows
        for c in top_cols
        if (r, c) in vertical_by_col[c] and (r, c) in horizontal_by_row[r]
    }
    missing_pairs = pair_cells - yellow_pairs

    return {
        "red_cells": red_cells,
        "green_set": green_set,
        "vertical_by_col": vertical_by_col,
        "horizontal_by_row": horizontal_by_row,
        "blocked_top": blocked_top,
        "blocked_right": blocked_right,
        "pair_cells": pair_cells,
        "yellow_pairs": yellow_pairs,
        "missing_pairs": missing_pairs,
    }


def _pick_green_pair(
    h: int,
    w: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    for _ in range(100):
        orientation = random.choice(("horizontal", "vertical"))
        mr = random.randint(2, h - 3)
        mc = random.randint(2, w - 3)

        if orientation == "horizontal":
            max_half_gap = min(mc - 1, w - 2 - mc)
        else:
            max_half_gap = min(mr - 1, h - 2 - mr)
        if max_half_gap < 1:
            continue

        if max_half_gap >= 2 and random.random() < 0.65:
            half_gap = random.randint(2, max_half_gap)
        else:
            half_gap = 1

        if orientation == "horizontal":
            greens = ((mr, mc - half_gap), (mr, mc + half_gap))
        else:
            greens = ((mr - half_gap, mc), (mr + half_gap, mc))
        return tuple(sorted(greens)), (mr, mc)
    raise RuntimeError("failed to place a valid green pair")


def _marker_semantics_ok(
    shape: tuple[int, int],
    top_cols: tuple[int, ...],
    right_rows: tuple[int, ...],
    greens: tuple[tuple[int, int], tuple[int, int]],
    mode: str,
) -> bool:
    profile = _trace_profile(shape, top_cols, right_rows, greens)
    green_set = profile["green_set"]
    pair_cells = profile["pair_cells"]
    yellow_pairs = profile["yellow_pairs"]
    missing_pairs = profile["missing_pairs"]
    blocked_top = profile["blocked_top"]
    blocked_right = profile["blocked_right"]

    if pair_cells & green_set:
        return False
    if not yellow_pairs:
        return False
    if not missing_pairs:
        return False
    if len(yellow_pairs) == len(pair_cells):
        return False

    if mode == "vertical_only" and not blocked_top:
        return False
    if mode == "horizontal_only" and not blocked_right:
        return False
    if mode == "both" and (not blocked_top or not blocked_right):
        return False

    if mode in {"vertical_only", "both"}:
        witnessed = any(
            any((r, c) in yellow_pairs for r in right_rows)
            and any((r, c) in missing_pairs for r in right_rows)
            for c in blocked_top
        )
        if not witnessed:
            return False

    if mode in {"horizontal_only", "both"}:
        witnessed = any(
            any((r, c) in yellow_pairs for c in top_cols)
            and any((r, c) in missing_pairs for c in top_cols)
            for r in blocked_right
        )
        if not witnessed:
            return False

    return True


def _sample_markers(
    shape: tuple[int, int],
    greens: tuple[tuple[int, int], tuple[int, int]],
    mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    h, w = shape
    profile = _trace_profile(shape, tuple(), tuple(), greens)
    red_cells = profile["red_cells"]

    blocked_cols = sorted({c for _r, c in red_cells})
    blocked_rows = sorted({r for r, _c in red_cells})

    top_domain = list(range(0, w - 1))
    right_domain = list(range(1, h - 1))
    top_unblocked = [c for c in top_domain if c not in blocked_cols]
    right_unblocked = [r for r in right_domain if r not in blocked_rows]

    need_top_block = mode in {"vertical_only", "both"}
    need_right_block = mode in {"horizontal_only", "both"}

    if need_top_block and not blocked_cols:
        return None
    if need_right_block and not blocked_rows:
        return None

    for _ in range(MARKER_RETRIES):
        top_count = random.randint(2, 4)
        right_count = random.randint(2, 4)

        if need_top_block:
            top_block_count = random.randint(1, min(len(blocked_cols), top_count - 1))
        else:
            top_block_count = 0
        top_unblocked_count = top_count - top_block_count
        if top_unblocked_count > len(top_unblocked):
            continue

        if need_right_block:
            right_block_count = random.randint(1, min(len(blocked_rows), right_count - 1))
        else:
            right_block_count = 0
        right_unblocked_count = right_count - right_block_count
        if right_unblocked_count > len(right_unblocked):
            continue

        top_cols = tuple(
            sorted(
                random.sample(blocked_cols, top_block_count)
                + random.sample(top_unblocked, top_unblocked_count)
            )
        )
        right_rows = tuple(
            sorted(
                random.sample(blocked_rows, right_block_count)
                + random.sample(right_unblocked, right_unblocked_count)
            )
        )

        if _marker_semantics_ok(shape, top_cols, right_rows, greens, mode):
            return top_cols, right_rows

    return None


def _build_input(
    shape: tuple[int, int],
    top_cols: tuple[int, ...],
    right_rows: tuple[int, ...],
    greens: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for c in top_cols:
        grid[0, c] = BLUE
    for r in right_rows:
        grid[r, w - 1] = BLUE
    for r, c in greens:
        grid[r, c] = GREEN
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    greens, _ = _pick_green_pair(h, w)
    mode = random.choice(("vertical_only", "horizontal_only", "both"))
    markers = _sample_markers((h, w), greens, mode)
    if markers is None:
        return None

    top_cols, right_rows = markers
    grid = _build_input((h, w), top_cols, right_rows, greens)
    expected = _independent_expected((h, w), top_cols, right_rows, greens)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    profile = _trace_profile((h, w), top_cols, right_rows, greens)
    if not profile["yellow_pairs"] or not profile["missing_pairs"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid midpoint-cross ray-grid input")
