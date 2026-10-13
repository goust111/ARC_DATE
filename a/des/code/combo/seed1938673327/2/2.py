import random
from collections import Counter

import numpy as np


RING_ORDER = (
    (0, 0),  # upper-left
    (0, 1),  # up
    (0, 2),  # upper-right
    (1, 2),  # right
    (2, 2),  # lower-right
    (2, 1),  # down
    (2, 0),  # lower-left
    (1, 0),  # left
)

BG = 0
COLOR_POOL = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 300
GRID_MIN = 11
GRID_MAX = 18
MIN_OBJECTS = 2
MAX_OBJECTS = 8


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _find_palette(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    candidates: list[list[int]] = []

    def add_horizontal(row: int, inner_row: int) -> None:
        for start, end in _segments(grid[row] != bg):
            if end - start + 1 != 8:
                continue
            if not np.all(grid[inner_row, start : end + 1] == bg):
                continue
            candidates.append([int(v) for v in grid[row, start : end + 1]])

    def add_vertical(col: int, inner_col: int) -> None:
        for start, end in _segments(grid[:, col] != bg):
            if end - start + 1 != 8:
                continue
            if not np.all(grid[start : end + 1, inner_col] == bg):
                continue
            candidates.append([int(v) for v in grid[start : end + 1, col]])

    if h >= 2:
        add_horizontal(0, 1)
        add_horizontal(h - 1, h - 2)
    if w >= 2:
        add_vertical(0, 1)
        add_vertical(w - 1, w - 2)

    assert len(candidates) == 1, "expected exactly one edge palette"
    return candidates[0]


def _find_rings(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    rings: list[tuple[int, int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != bg:
                continue
            ring = [int(patch[r, c]) for r, c in RING_ORDER]
            if ring.count(bg) != 1:
                continue
            fg_colors = {color for color in ring if color != bg}
            if len(fg_colors) != 1:
                continue
            rings.append((r0, c0, ring.index(bg)))
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each broken 3x3 ring center using the edge palette entry that matches
    the break direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    palette = _find_palette(g, bg)
    rings = _find_rings(g, bg)
    assert rings, "expected at least one broken ring"

    out = g.copy()
    for r0, c0, gap_idx in rings:
        out[r0 + 1, c0 + 1] = palette[gap_idx]
    return out


def _choose_palette() -> list[int]:
    while True:
        palette = [random.choice(COLOR_POOL) for _ in range(8)]
        if len(set(palette)) >= 3:
            return palette


def _place_palette(grid: np.ndarray, occupied: np.ndarray, side: str, palette: list[int]) -> None:
    h, w = grid.shape
    if side == "top":
        start = random.randint(0, w - 8)
        grid[0, start : start + 8] = palette
        occupied[0:2, max(0, start - 1) : min(w, start + 9)] = True
    elif side == "bottom":
        start = random.randint(0, w - 8)
        grid[h - 1, start : start + 8] = palette
        occupied[h - 2 : h, max(0, start - 1) : min(w, start + 9)] = True
    elif side == "left":
        start = random.randint(0, h - 8)
        grid[start : start + 8, 0] = palette
        occupied[max(0, start - 1) : min(h, start + 9), 0:2] = True
    else:
        start = random.randint(0, h - 8)
        grid[start : start + 8, w - 1] = palette
        occupied[max(0, start - 1) : min(h, start + 9), w - 2 : w] = True


def _candidate_positions(grid: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - 3):
        for c0 in range(1, w - 3):
            if occupied[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_ring(grid: np.ndarray, r0: int, c0: int, ring_color: int, gap_idx: int) -> None:
    patch = np.full((3, 3), ring_color, dtype=int)
    patch[1, 1] = BG
    gap_r, gap_c = RING_ORDER[gap_idx]
    patch[gap_r, gap_c] = BG
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch


def _expected_output(inp: np.ndarray, objects: list[tuple[int, int, int]], palette: list[int]) -> np.ndarray:
    out = inp.copy()
    for r0, c0, gap_idx in objects:
        out[r0 + 1, c0 + 1] = palette[gap_idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(SIDES)
    palette = _choose_palette()

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occupied, side, palette)

    positions = _candidate_positions(grid, occupied)
    max_objects = min(MAX_OBJECTS, len(positions))
    if max_objects < MIN_OBJECTS:
        return None
    num_objects = random.randint(MIN_OBJECTS, max_objects)

    gap_order = list(range(8))
    random.shuffle(gap_order)
    gap_indices = gap_order[:num_objects]
    if len({palette[idx] for idx in gap_indices}) < 2:
        return None

    objects: list[tuple[int, int, int]] = []
    for gap_idx in gap_indices:
        positions = _candidate_positions(grid, occupied)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        ring_choices = [color for color in COLOR_POOL if color != palette[gap_idx]]
        ring_color = random.choice(ring_choices)
        _paint_ring(grid, r0, c0, ring_color, gap_idx)
        occupied[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4] = True
        objects.append((r0, c0, gap_idx))

    expected = _expected_output(grid, objects, palette)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    found = _find_rings(grid, BG)
    if len(found) != len(objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
