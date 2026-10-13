from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
TERRAIN_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]


def _escaping_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    escaping = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if mask[r, c] and not escaping[r, c]:
            escaping[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for rr, cc in ((r, c - 1), (r, c + 1), (r - 1, c)):
            if 0 <= rr < h and 0 <= cc < w:
                add(rr, cc)
    return escaping


def main(grid: np.ndarray) -> np.ndarray:
    """Fill background cells that cannot flow sideways or downward to a drain.

    Input structure:
    - A black background surrounds one non-black terrain color connected to the bottom edge.
    - The terrain is a solid rising contour with basin depressions; some side walls have a single missing black spillway pixel.
    - The left edge, right edge, and bottom outside the grid are drains, while the top edge is not a drain.

    Rule:
    - Preserve all terrain cells.
    - From a black cell, water may move only to a same-row neighboring black cell or to the black cell directly below.
    - Recolor exactly the black cells with no such escape path to the drains blue.

    Key insight:
    - Reverse the flow graph from drain-touching background cells using left, right, and upward moves; every unreached black cell is trapped water.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    background = g == BLACK
    escaping = _escaping_background(background)
    out = g.copy()
    out[background & ~escaping] = BLUE
    return out


def _terrain_connected_to_bottom(grid: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    terrain = grid == color
    if not np.any(terrain[-1]):
        return False

    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for c in range(w):
        if terrain[h - 1, c]:
            seen[h - 1, c] = True
            q.append((h - 1, c))

    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and terrain[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return bool(np.all(seen[terrain]))


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(19, 30)
    n_basins = random.randint(2, 3)

    basin_widths = [random.randint(5, 8) for _ in range(n_basins)]
    gaps_between = [random.randint(2, 4) for _ in range(n_basins - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    used = left_margin + right_margin + sum(basin_widths) + sum(gaps_between)
    if used > w:
        return None

    start = left_margin + random.randint(0, w - used)
    basins: list[dict[str, int]] = []
    c = start
    for i, width in enumerate(basin_widths):
        left = c
        right = c + width - 1
        basins.append({"left": left, "right": right})
        c = right + 1
        if i < len(gaps_between):
            c += gaps_between[i]

    base_surface = random.randint(h - 5, h - 3)
    surface = [base_surface + random.choice([0, 0, 1]) for _ in range(w)]
    for c in range(w):
        surface[c] = min(surface[c], h - 2)

    for basin in basins:
        floor = random.randint(max(base_surface + 1, h - 4), h - 2)
        rim_hi = max(5, floor - 4)
        if rim_hi < 3:
            return None
        rim = random.randint(3, rim_hi)
        basin["rim"] = rim
        basin["floor"] = floor

        left = basin["left"]
        right = basin["right"]
        surface[left] = rim
        surface[right] = rim
        for cc in range(left + 1, right):
            surface[cc] = min(h - 1, floor + random.choice([-1, 0, 0, 1]))
            if surface[cc] <= rim + 2:
                surface[cc] = rim + 3

    color = random.choice(TERRAIN_COLORS)
    grid = np.full((h, w), BLACK, dtype=int)
    for cc, top in enumerate(surface):
        grid[top:, cc] = color

    gap_specs: list[tuple[int, str]] = []
    if random.random() < 0.5:
        gap_specs.append((0, "left"))
    else:
        gap_specs.append((n_basins - 1, "right"))
    if n_basins == 3 and random.random() < 0.45:
        other = (n_basins - 1, "right") if gap_specs[0][0] == 0 else (0, "left")
        gap_specs.append(other)

    for idx, side in gap_specs:
        basin = basins[idx]
        wall_c = basin["left"] if side == "left" else basin["right"]
        grid[basin["rim"], wall_c] = BLACK

    if not _terrain_connected_to_bottom(grid, color):
        return None

    out = main(grid)
    water = out == BLUE
    if int(np.count_nonzero(water)) < 8:
        return None
    if not np.any((grid == BLACK) & (out == BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spillway basin input")
