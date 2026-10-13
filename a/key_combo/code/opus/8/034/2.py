from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RECT_SIZE_CHOICES = (6, 7, 8)
MARGIN_PAIRS = ((1, 1), (1, 2), (2, 1), (2, 2))
MAX_GENERATION_ATTEMPTS = 400


def _find_single_cell(grid: np.ndarray, color: int) -> tuple[int, int]:
    cells = np.argwhere(grid == color)
    assert cells.shape == (1, 2), f"expected exactly one color-{color} marker"
    return tuple(map(int, cells[0]))


def _find_checker_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid == RED) | (grid == BLUE)
    coords = np.argwhere(mask)
    assert coords.size > 0, "missing checkerboard region"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((sub == RED) | (sub == BLUE)), "checkerboard bbox must be fully filled"
    assert int(mask.sum()) == sub.size, "red/blue cells must form one solid rectangle"

    rows = np.arange(r0, r1 + 1)[:, None]
    cols = np.arange(c0, c1 + 1)[None, :]
    expected = np.where((rows + cols) % 2 == 0, RED, BLUE)
    assert np.array_equal(sub, expected), "checkerboard colors must match absolute parity"
    return r0, c0, r1, c1


def _extract_structure(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], bool, bool, tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREEN, YELLOW})

    h, w = g.shape
    green_pos = _find_single_cell(g, GREEN)
    yellow_pos = _find_single_cell(g, YELLOW)
    gr, gc = green_pos
    yr, yc = yellow_pos

    assert gc in (0, w - 1), "green marker must be on the left or right edge"
    assert yr in (0, h - 1), "yellow marker must be on the top or bottom edge"
    assert green_pos != yellow_pos, "markers must be distinct"
    assert gr != yr and gc != yc, "markers must not share a row or a column"

    bbox = _find_checker_bbox(g)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 >= 6 and c1 - c0 + 1 >= 6
    assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1, "checkerboard must be interior"
    assert grid[green_pos] == GREEN and grid[yellow_pos] == YELLOW

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    outside[green_pos] = BLACK
    outside[yellow_pos] = BLACK
    assert np.all(outside == BLACK), "only checkerboard and two markers are allowed"

    return bbox, gc == 0, yr == 0, green_pos, yellow_pos


def _render_from_dirs(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    push_left: bool,
    push_up: bool,
    *,
    include_red: bool = True,
    include_blue: bool = True,
    overlap_color: int = GREY,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.array(grid, copy=True)
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    red_mask = np.zeros_like(out, dtype=bool)
    blue_mask = np.zeros_like(out, dtype=bool)

    if include_red:
        for r in range(r0, r1 + 1):
            count = int(np.count_nonzero(grid[r, c0 : c1 + 1] == RED))
            if push_left:
                red_mask[r, c0 : c0 + count] = True
            else:
                red_mask[r, c1 - count + 1 : c1 + 1] = True

    if include_blue:
        for c in range(c0, c1 + 1):
            count = int(np.count_nonzero(grid[r0 : r1 + 1, c] == BLUE))
            if push_up:
                blue_mask[r0 : r0 + count, c] = True
            else:
                blue_mask[r1 - count + 1 : r1 + 1, c] = True

    only_red = red_mask & ~blue_mask
    only_blue = blue_mask & ~red_mask
    overlap = red_mask & blue_mask
    out[only_red] = RED
    out[only_blue] = BLUE
    out[overlap] = overlap_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the checkerboard by parity, compact the two classes toward orthogonal marked edges, and color their overlap gray.

    Input structure:
    - A black background contains one interior red/blue rectangular checkerboard whose colors follow absolute grid parity.
    - One green marker sits on the left or right border and specifies the horizontal compaction direction.
    - One yellow marker sits on the top or bottom border and specifies the vertical compaction direction.

    Rule:
    - Treat red checkerboard cells as one parity class and compact them within each checkerboard row toward the green-indicated side.
    - Treat blue checkerboard cells as the other parity class and compact them within each checkerboard column toward the yellow-indicated side.
    - Preserve both edge markers, keep empty cells black, color red-only cells red, blue-only cells blue, and double-occupied cells gray.

    Key insight:
    - The checkerboard parity determines which cells belong to the horizontal-moving class versus the vertical-moving class, and the hidden rule is that their crossings create a third color rather than choosing one class over the other.
    """
    bbox, push_left, push_up, _, _ = _extract_structure(np.asarray(grid, dtype=int))
    return _render_from_dirs(np.asarray(grid, dtype=int), bbox, push_left, push_up)


def _build_checkerboard(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    rows = np.arange(r0, r0 + h)[:, None]
    cols = np.arange(c0, c0 + w)[None, :]
    return np.where((rows + cols) % 2 == 0, RED, BLUE)


def _generate_attempt() -> np.ndarray:
    rect_h = random.choice(RECT_SIZE_CHOICES)
    rect_w = random.choice(RECT_SIZE_CHOICES)
    top_margin, bottom_margin = random.choice(MARGIN_PAIRS)
    left_margin, right_margin = random.choice(MARGIN_PAIRS)

    h = rect_h + top_margin + bottom_margin
    w = rect_w + left_margin + right_margin
    r0 = top_margin
    c0 = left_margin

    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = _build_checkerboard(r0, c0, rect_h, rect_w)

    green_row = random.randint(1, h - 2)
    yellow_col = random.randint(1, w - 2)
    grid[green_row, random.choice((0, w - 1))] = GREEN
    grid[random.choice((0, h - 1)), yellow_col] = YELLOW
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bbox, push_left, push_up, _, _ = _extract_structure(grid)
        out = main(grid)
    except AssertionError:
        return False

    colors = set(map(int, np.unique(out)))
    if not {RED, BLUE, GREY}.issubset(colors):
        return False

    actual = out
    flipped_h = _render_from_dirs(grid, bbox, not push_left, push_up)
    flipped_v = _render_from_dirs(grid, bbox, push_left, not push_up)
    red_wins = _render_from_dirs(grid, bbox, push_left, push_up, overlap_color=RED)
    blue_wins = _render_from_dirs(grid, bbox, push_left, push_up, overlap_color=BLUE)
    red_only = _render_from_dirs(grid, bbox, push_left, push_up, include_blue=False)
    blue_only = _render_from_dirs(grid, bbox, push_left, push_up, include_red=False)

    return (
        not np.array_equal(actual, grid)
        and not np.array_equal(actual, flipped_h)
        and not np.array_equal(actual, flipped_v)
        and not np.array_equal(actual, red_wins)
        and not np.array_equal(actual, blue_wins)
        and not np.array_equal(actual, red_only)
        and not np.array_equal(actual, blue_only)
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-compaction input")
