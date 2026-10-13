from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

GEN_MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    seen = set()
    q = deque([next(iter(coords))])
    seen.add(q[0])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _find_divider(grid: np.ndarray) -> int:
    h, _w = grid.shape
    candidates = [r for r in range(h) if np.all(grid[r] == PINK)]
    assert len(candidates) == 1, "expected exactly one full-width pink divider row"
    sep = candidates[0]
    assert sep * 2 == h - 1, "divider must split the grid into equal halves"
    return sep


def _compute_output_from_masks(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    top_fg = top != BLACK
    bottom_fg = bottom != BLACK
    both = top_fg & bottom_fg
    top_only = top_fg & ~bottom_fg
    bottom_only = bottom_fg & ~top_fg

    out = np.full((h, w), GREY, dtype=int)
    out[both] = RED

    top_targets = np.zeros((h, w), dtype=bool)
    bottom_targets = np.zeros((h, w), dtype=bool)
    top_targets[1:, :] = top_only[:-1, :]
    bottom_targets[:-1, :] = bottom_only[1:, :]

    only_top = top_targets & ~bottom_targets
    only_bottom = bottom_targets & ~top_targets
    meet = top_targets & bottom_targets
    free = ~both

    out[free & only_top] = BLUE
    out[free & only_bottom] = GREEN
    out[free & meet] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two divider-separated masks cellwise and route each position to recolor, shift, or background fill.

    Input structure:
    - The grid has black background with one centered full-width pink divider row.
    - Above the divider is a blue 4-connected mask; below it is a green 4-connected mask.
    - The two halves have identical shape and are aligned cell by cell.

    Rule:
    - Remove the divider and compare each aligned top/bottom cell pair.
    - If both cells are foreground, write red at that location.
    - If only the top cell is foreground, shift it one row downward in the output; if only the bottom cell is foreground, shift it one row upward.
    - If both are background, write grey; if the two opposite-direction shifts land on the same non-red cell, write yellow there; red keeps priority over any incoming shift.

    Key insight:
    - The cellwise comparison is only a routing decision: overlap recolors in place, exclusive occupancy becomes a one-step move toward the divider, and empty-on-both becomes an explicit background mark.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and h >= 3, "expected an odd-height grid with a divider"

    sep = _find_divider(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape, "top and bottom panels must match"

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    assert top_colors <= {BLACK, BLUE}, "top panel must use only black and blue"
    assert bottom_colors <= {BLACK, GREEN}, "bottom panel must use only black and green"
    assert BLUE in top_colors and GREEN in bottom_colors, "both masks must be present"

    top_coords = set(map(tuple, np.argwhere(top == BLUE)))
    bottom_coords = set(map(tuple, np.argwhere(bottom == GREEN)))
    assert 8 <= len(top_coords) <= 15, "top mask area must stay in the intended range"
    assert 8 <= len(bottom_coords) <= 15, "bottom mask area must stay in the intended range"
    assert _is_connected(top_coords), "top mask must be 4-connected"
    assert _is_connected(bottom_coords), "bottom mask must be 4-connected"

    return _compute_output_from_masks(top, bottom)


def _add_segment(
    cells: set[tuple[int, int]],
    row: int,
    start_col: int,
    end_col: int,
    *,
    h: int,
    w: int,
) -> None:
    c0, c1 = sorted((start_col, end_col))
    assert 0 <= row < h
    assert 0 <= c0 <= c1 < w
    for c in range(c0, c1 + 1):
        cells.add((row, c))


def _build_attempt() -> np.ndarray:
    half_h = random.randint(5, 7)
    width = random.randint(7, 10)
    trunk_col = random.randint(2, width - 3)

    left1 = trunk_col - 1
    left2 = trunk_col - 2
    right1 = trunk_col + 1
    right2 = trunk_col + 2

    top_cells = {(r, trunk_col) for r in range(half_h)}
    bottom_cells = {(r, trunk_col) for r in range(half_h)}

    yellow_target_row = random.randint(1, half_h - 2)
    blue_row = random.randint(0, half_h - 2)
    green_row = random.randint(1, half_h - 1)
    red_from_top = bool(random.getrandbits(1))
    if red_from_top:
        red_target_row = random.randint(1, half_h - 1)
    else:
        red_target_row = random.randint(0, half_h - 2)

    clip_top_col = random.choice([left1, left2])
    clip_bottom_col = random.choice([right1, right2])

    _add_segment(top_cells, yellow_target_row - 1, trunk_col, left1, h=half_h, w=width)
    _add_segment(bottom_cells, yellow_target_row + 1, trunk_col, left1, h=half_h, w=width)

    _add_segment(top_cells, blue_row, trunk_col, left2, h=half_h, w=width)
    _add_segment(bottom_cells, green_row, trunk_col, right2, h=half_h, w=width)

    _add_segment(top_cells, half_h - 1, trunk_col, clip_top_col, h=half_h, w=width)
    _add_segment(bottom_cells, 0, trunk_col, clip_bottom_col, h=half_h, w=width)

    _add_segment(top_cells, red_target_row, trunk_col, right1, h=half_h, w=width)
    _add_segment(bottom_cells, red_target_row, trunk_col, right1, h=half_h, w=width)
    if red_from_top:
        _add_segment(top_cells, red_target_row - 1, trunk_col, right1, h=half_h, w=width)
    else:
        _add_segment(bottom_cells, red_target_row + 1, trunk_col, right1, h=half_h, w=width)

    top = np.zeros((half_h, width), dtype=int)
    bottom = np.zeros((half_h, width), dtype=int)
    for r, c in top_cells:
        top[r, c] = BLUE
    for r, c in bottom_cells:
        bottom[r, c] = GREEN

    grid = np.zeros((2 * half_h + 1, width), dtype=int)
    grid[:half_h] = top
    grid[half_h] = PINK
    grid[half_h + 1 :] = bottom
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    sep = _find_divider(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    top_mask = top == BLUE
    bottom_mask = bottom == GREEN

    top_coords = set(map(tuple, np.argwhere(top_mask)))
    bottom_coords = set(map(tuple, np.argwhere(bottom_mask)))
    if not (8 <= len(top_coords) <= 15 and 8 <= len(bottom_coords) <= 15):
        return False
    if not (_is_connected(top_coords) and _is_connected(bottom_coords)):
        return False

    both = top_mask & bottom_mask
    top_only = top_mask & ~bottom_mask
    bottom_only = bottom_mask & ~top_mask
    both_bg = ~top_mask & ~bottom_mask
    if min(np.count_nonzero(both), np.count_nonzero(top_only), np.count_nonzero(bottom_only), np.count_nonzero(both_bg)) < 2:
        return False

    has_red_priority = bool((top_only[:-1] & both[1:]).any() or (bottom_only[1:] & both[:-1]).any())
    has_clip = bool(top_only[-1].any() and bottom_only[0].any())
    out = _compute_output_from_masks(top, bottom)
    colors = set(map(int, np.unique(out)))
    if not {RED, YELLOW, BLUE, GREEN, GREY} <= colors:
        return False
    if not has_red_priority or not has_clip:
        return False

    return np.array_equal(main(grid), out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_attempt()
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
