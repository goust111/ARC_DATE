from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors8


BG = Color.BLACK
SOURCE = Color.BLUE
EVEN_COLOR = Color.RED
ODD_COLOR = Color.YELLOW
WALL = Color.GREY

MAX_GEN_ATTEMPTS = 500


def _king_distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    sr, sc = map(int, start)
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _component_size(mask: np.ndarray, start: tuple[int, int]) -> int:
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def _passable_connected(grid: np.ndarray) -> bool:
    source_cells = np.argwhere(grid == SOURCE)
    if len(source_cells) != 1:
        return False
    start = tuple(map(int, source_cells[0]))
    passable = grid != WALL
    return _component_size(passable, start) == int(passable.sum())


def _parity_shadow_size(grid: np.ndarray) -> int:
    source = tuple(map(int, np.argwhere(grid == SOURCE)[0]))
    passable = grid != WALL
    dist = _king_distances(passable, source)
    rr, cc = np.indices(grid.shape)
    cheb = np.maximum(np.abs(rr - source[0]), np.abs(cc - source[1]))
    mask = passable.copy()
    mask[source] = False
    return int(np.count_nonzero(mask & ((dist % 2) != (cheb % 2))))


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _build_single_wall(height: int, width: int) -> np.ndarray | None:
    if height < 8 or width < 8:
        return None
    grid = np.full((height, width), BG, dtype=int)
    wall_col = random.randint(max(3, width // 3), width - 4)
    gap_len = random.randint(1, min(3, height - 2))
    open_top = random.random() < 0.5
    if open_top:
        rows = range(gap_len, height)
        avoid_rows = range(0, min(height, gap_len + 2))
    else:
        rows = range(0, height - gap_len)
        avoid_rows = range(max(0, height - gap_len - 2), height)
    for r in rows:
        grid[r, wall_col] = WALL

    source_rows = [r for r in range(1, height - 1) if r not in avoid_rows]
    if not source_rows or wall_col < 3:
        return None
    sr = random.choice(source_rows)
    sc = random.randint(1, wall_col - 2)
    grid[sr, sc] = SOURCE
    return grid


def _build_double_wall(height: int, width: int) -> np.ndarray | None:
    if height < 9 or width < 11:
        return None
    grid = np.full((height, width), BG, dtype=int)
    wall_col1 = random.randint(3, width - 7)
    wall_col2 = random.randint(wall_col1 + 3, width - 4)
    gap1 = random.randint(1, min(3, height - 2))
    gap2 = random.randint(1, min(3, height - 2))

    for r in range(0, height - gap1):
        grid[r, wall_col1] = WALL
    for r in range(gap2, height):
        grid[r, wall_col2] = WALL

    source_candidates = [
        r
        for r in range(1, height - 1)
        if r < max(1, height - gap1 - 2) and r > min(height - 2, gap2 + 1)
    ]
    if not source_candidates:
        source_candidates = list(range(height // 3, max(height // 3 + 1, (2 * height) // 3)))
    source_candidates = [r for r in source_candidates if 1 <= r < height - 1]
    if not source_candidates or wall_col1 < 3:
        return None
    sr = random.choice(source_candidates)
    sc = random.randint(1, wall_col1 - 2)
    grid[sr, sc] = SOURCE
    return grid


def _grid_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if len(np.argwhere(grid == SOURCE)) != 1:
        return False
    if int(np.count_nonzero(grid == WALL)) < 5:
        return False
    if not _passable_connected(grid):
        return False
    shadow = _parity_shadow_size(grid)
    if shadow < 3:
        return False

    out = main(grid)
    if np.any(out == BG):
        return False
    odd_count = int(np.count_nonzero(out == ODD_COLOR))
    even_count = int(np.count_nonzero(out == EVEN_COLOR))
    return odd_count >= 4 and even_count >= 4


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every reachable empty cell by the parity of its king-move distance from the blue source.

    Input structure:
    - A black background contains exactly one blue source pixel and one or more grey wall pixels.
    - Every non-wall cell is part of the same 8-connected open region as the source.

    Rule:
    - Measure shortest-path distance from the blue source using 8-direction king moves.
    - Grey wall cells are impassable; distances must use real detours around them rather than raw geometric offsets.
    - Keep the source blue, keep walls grey, color odd-distance cells yellow, and color even-distance non-source cells red.

    Key insight:
    - Without walls the distance is just a square wave based on Chebyshev radius, but wall shadows shift the parity pattern because the wavefront must bend around obstacles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    source_cells = np.argwhere(g == SOURCE)
    assert len(source_cells) == 1, "expected exactly one blue source"
    source = tuple(map(int, source_cells[0]))

    passable = g != WALL
    dist = _king_distances(passable, source)
    assert np.all(dist[passable] >= 0), "all non-wall cells must be reachable from the source"

    out = np.full(g.shape, BG, dtype=int)
    out[g == WALL] = WALL

    odd_mask = passable & ((dist % 2) == 1)
    even_mask = passable & ((dist % 2) == 0)
    out[odd_mask] = ODD_COLOR
    out[even_mask] = EVEN_COLOR
    out[source] = SOURCE
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(10, 22)
        width = random.randint(10, 22)
        builder = _build_single_wall if random.random() < 0.5 else _build_double_wall
        grid = builder(height, width)
        if grid is None:
            continue
        grid = _apply_random_symmetry(grid)
        if _grid_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid wavefront-parity input")
