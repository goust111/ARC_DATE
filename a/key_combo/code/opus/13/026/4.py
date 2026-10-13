from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600

Square = tuple[int, int, int]


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0:
        total -= int(prefix[r0 - 1, c1])
    if c0:
        total -= int(prefix[r1, c0 - 1])
    if r0 and c0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _extract_palette(grid: np.ndarray) -> list[int]:
    return [int(grid[r, 0]) for r in range(grid.shape[0]) if int(grid[r, 0]) != BLACK]


def _selected_square_groups(search: np.ndarray) -> dict[int, list[Square]]:
    h, w = map(int, search.shape)
    blocked = (search != BLACK).astype(np.int16)
    prefix = np.cumsum(np.cumsum(blocked, axis=0), axis=1)
    occupied = np.zeros((h, w), dtype=bool)
    groups: dict[int, list[Square]] = {}

    for side in range(min(h, w), 1, -1):
        for r in range(h - side + 1):
            for c in range(w - side + 1):
                if occupied[r : r + side, c : c + side].any():
                    continue
                if _rect_sum(prefix, r, c, r + side - 1, c + side - 1) != 0:
                    continue
                groups.setdefault(side, []).append((r, c, side))
                occupied[r : r + side, c : c + side] = True
    return groups


def _paint_groups(grid: np.ndarray, groups: dict[int, list[Square]], palette: Iterable[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    palette_list = list(palette)
    for rank, side in enumerate(sorted(groups, reverse=True)):
        if rank >= len(palette_list):
            break
        color = int(palette_list[rank])
        for r, c, size in groups[side]:
            out[r : r + size, c + 1 : c + 1 + size] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Greedily pick non-overlapping blank squares and recolor each size rank from the palette column.

    Input structure:
    - The leftmost column is a palette: 3-5 non-black cells appear from top to bottom, then black.
    - Columns 1..end form the search area: black background with grey obstacle pixels.
    - Relevant objects are all-black axis-aligned square regions of side at least 2 inside the search area.

    Rule:
    - Enumerate every all-black square candidate in the search area.
    - Process side lengths from largest to smallest; within one side length, scan top-to-bottom then left-to-right, selecting a square only if it does not overlap any already selected square.
    - Group selected squares by side length, rank the groups from largest side to smallest side, and recolor rank k with the k-th non-black palette color from the left column.
    - If there are more size groups than palette colors, the extra smallest groups stay black. Palette cells and grey obstacles remain unchanged.

    Key insight:
    - The palette does not map to spatial position; it maps to the area ranking created only after the largest-first greedy square selection is completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected a palette column and a search area"

    palette = _extract_palette(g)
    groups = _selected_square_groups(g[:, 1:])
    return _paint_groups(g, groups, palette)


def _count_grey(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == GREY))


def _add_segment(
    grid: np.ndarray,
    orientation: str,
    fixed: int,
    start: int,
    length: int,
) -> int:
    before = _count_grey(grid)
    if orientation == "h":
        grid[fixed, start : start + length] = GREY
    else:
        grid[start : start + length, fixed] = GREY
    return _count_grey(grid) - before


def _attempt_search_region(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 18)
    w = rng.randint(15, 20)
    search = np.zeros((h, w), dtype=int)

    wall_row = rng.randint(3, h - 4)
    search[wall_row, :] = GREY

    wall_col = rng.randint(3, w - 4)
    upper_branch = rng.random() < 0.5
    span0, span1 = (0, wall_row) if upper_branch else (wall_row + 1, h)
    span = span1 - span0
    if span < 3:
        return None

    if rng.random() < 0.45:
        seg_len = span
    else:
        lo = min(6, max(3, span // 2))
        hi = min(span, 6)
        seg_len = rng.randint(lo, hi)
    start = rng.randint(span0, span1 - seg_len)
    _add_segment(search, "v", wall_col, start, seg_len)

    if rng.random() < 0.8:
        remaining = 25 - _count_grey(search)
        if remaining >= 3:
            seg_len = rng.randint(3, min(6, remaining, max(h, w)))
            if rng.random() < 0.5:
                row = rng.choice([r for r in range(h) if r != wall_row])
                c0 = rng.randint(0, w - seg_len)
                _add_segment(search, "h", row, c0, seg_len)
            else:
                col = rng.choice([c for c in range(w) if c != wall_col])
                r0 = rng.randint(0, h - seg_len)
                _add_segment(search, "v", col, r0, seg_len)

    grey_count = _count_grey(search)
    if not (15 <= grey_count <= 25):
        return None

    groups = _selected_square_groups(search)
    sides = sorted(groups, reverse=True)
    counts = [len(groups[side]) for side in sides]
    if not (3 <= len(sides) <= 5):
        return None
    if not all(1 <= count <= 4 for count in counts):
        return None
    if max(counts) < 2:
        return None
    if sides[0] > 10:
        return None
    return search


def _choose_palette_length(rng: random.Random, group_count: int) -> int:
    if group_count >= 4 and rng.random() < 0.45:
        return rng.randint(3, min(5, group_count - 1))
    if group_count < 5 and rng.random() < 0.3:
        return rng.randint(group_count + 1, 5)
    return rng.randint(3, min(5, group_count))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    search = _attempt_search_region(rng)
    if search is None:
        return None

    groups = _selected_square_groups(search)
    palette_len = _choose_palette_length(rng, len(groups))
    colors = PALETTE_COLORS[:]
    rng.shuffle(colors)
    palette = colors[:palette_len]

    grid = np.zeros((search.shape[0], search.shape[1] + 1), dtype=int)
    grid[:, 1:] = search
    for r, color in enumerate(palette):
        grid[r, 0] = color

    expected = _paint_groups(grid, groups, palette)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
