from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREY = 5

SCAFFOLD_SIZE = 19
CELL_STRIDE = 4
CENTER_OFFSET = 1
BOX_RADIUS = 1
GRID_CELLS = 5
OUTER_COUNT = 16
MAX_ATTEMPTS = 300


def _outer_ring_order() -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for c in range(GRID_CELLS):
        order.append((0, c))
    for r in range(1, GRID_CELLS):
        order.append((r, GRID_CELLS - 1))
    for c in range(GRID_CELLS - 2, -1, -1):
        order.append((GRID_CELLS - 1, c))
    for r in range(GRID_CELLS - 2, 0, -1):
        order.append((r, 0))
    assert len(order) == OUTER_COUNT
    return order


OUTER_RING = _outer_ring_order()


BASE_SHAPES = (
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    pts = {tuple(map(int, p)) for p in coords}
    start = next(iter(pts))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen == pts


def _is_valid_seed_mask(mask: np.ndarray) -> bool:
    if mask.shape != (3, 3):
        return False
    if int(mask.sum()) not in (3, 4):
        return False
    if not _is_connected(mask):
        return False
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _cell_center(top: int, left: int, rr: int, cc: int) -> tuple[int, int]:
    return top + CENTER_OFFSET + CELL_STRIDE * rr, left + CENTER_OFFSET + CELL_STRIDE * cc


def _center_lists(top: int, left: int) -> tuple[list[int], list[int]]:
    rows = [top + CENTER_OFFSET + CELL_STRIDE * i for i in range(GRID_CELLS)]
    cols = [left + CENTER_OFFSET + CELL_STRIDE * i for i in range(GRID_CELLS)]
    return rows, cols


def _scaffold_bbox(grid: np.ndarray) -> tuple[int, int]:
    mask = grid != BACKGROUND
    assert np.any(mask), "expected a visible scaffold"
    r0, c0, r1, c1 = _bbox(mask)
    assert r1 - r0 + 1 == SCAFFOLD_SIZE, "expected a 19x19 scaffold"
    assert c1 - c0 + 1 == SCAFFOLD_SIZE, "expected a 19x19 scaffold"
    return r0, c0


def _extract_seed_info(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int, np.ndarray]:
    top, left = _scaffold_bbox(grid)
    rows, cols = _center_lists(top, left)
    sub = grid[top : top + SCAFFOLD_SIZE, left : left + SCAFFOLD_SIZE]

    base = np.full((SCAFFOLD_SIZE, SCAFFOLD_SIZE), GREY, dtype=int)
    for rr in range(GRID_CELLS):
        for cc in range(GRID_CELLS):
            base[CENTER_OFFSET + CELL_STRIDE * rr, CENTER_OFFSET + CELL_STRIDE * cc] = BACKGROUND

    special = (sub == BLUE) | (sub == RED)
    assert np.array_equal(sub[~special], base[~special]), "scaffold pixels must match the frame lattice"

    red_indices = [
        idx
        for idx, (rr, cc) in enumerate(OUTER_RING)
        if grid[_cell_center(top, left, rr, cc)] == RED
    ]
    assert len(red_indices) == 1, "expected one red outer-ring marker"
    red_index = red_indices[0]

    for rr in range(1, GRID_CELLS - 1):
        for cc in range(1, GRID_CELLS - 1):
            cr, cc0 = _cell_center(top, left, rr, cc)
            assert grid[cr, cc0] == BACKGROUND, "inner cell centers must stay black"

    blue_coords = np.argwhere(grid == BLUE)
    assert blue_coords.size > 0, "expected one blue seed tile"

    seed_candidates = []
    for idx, (rr, cc) in enumerate(OUTER_RING):
        cr, cc0 = _cell_center(top, left, rr, cc)
        window = grid[cr - BOX_RADIUS : cr + BOX_RADIUS + 1, cc0 - BOX_RADIUS : cc0 + BOX_RADIUS + 1]
        if np.any(window == BLUE):
            if np.all(np.abs(blue_coords[:, 0] - cr) <= BOX_RADIUS) and np.all(np.abs(blue_coords[:, 1] - cc0) <= BOX_RADIUS):
                seed_candidates.append((idx, window))
    assert len(seed_candidates) == 1, "expected exactly one seeded outer cell"

    seed_index, window = seed_candidates[0]
    assert seed_index == (red_index + 1) % OUTER_COUNT, "seed tile must be clockwise-adjacent to the red marker"

    seed_mask = window == BLUE
    assert _is_valid_seed_mask(seed_mask), "seed tile must be a non-symmetric 4-connected 3x3 shape"

    return top, left, rows, cols, red_index, seed_mask


def _stamp_tile(out: np.ndarray, center: tuple[int, int], mask: np.ndarray) -> None:
    r, c = center
    out[r - BOX_RADIUS : r + BOX_RADIUS + 1, c - BOX_RADIUS : c + BOX_RADIUS + 1][mask] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the seeded blue tile around the scaffold's outer ring, rotating it 90 degrees clockwise at each step.

    Input structure:
    - A black background contains one grey 5x5 scaffold of hollow 3x3 frames, optionally padded by black margins.
    - The scaffold's 25 cell centers lie on a 4-pixel lattice; exactly one outer-ring center is red.
    - Exactly one clockwise-adjacent outer-ring cell contains a blue 3x3 non-symmetric connected seed tile, while all inner centers remain black.

    Rule:
    - Recover the scaffold bbox and its 25 center coordinates.
    - Order the 16 outer-ring cells clockwise starting from the top-left cell.
    - Use the red center as the reference cell; the blue seed tile is the first placement cell at the next outer-ring position.
    - Traverse the remaining outer-ring cells clockwise, rotating the current blue tile 90 degrees clockwise before stamping it into each next cell.
    - Stop before the traversal returns to the red marker, preserving the red center and the grey scaffold.

    Key insight:
    - The red pixel fixes where the ring traversal starts, while the lone blue tile fixes the initial orientation; every other tile follows by counting clockwise steps and accumulating quarter-turn rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, left, _rows, _cols, red_index, tile = _extract_seed_info(g)
    out = g.copy()

    current = tile.copy()
    idx = (red_index + 1) % OUTER_COUNT
    while True:
        idx = (idx + 1) % OUTER_COUNT
        if idx == red_index:
            break
        current = np.rot90(current, -1)
        rr, cc = OUTER_RING[idx]
        _stamp_tile(out, _cell_center(top, left, rr, cc), current)

    return out


def _build_base_grid(height: int, width: int, top: int, left: int) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[top : top + SCAFFOLD_SIZE, left : left + SCAFFOLD_SIZE] = GREY
    for rr in range(GRID_CELLS):
        for cc in range(GRID_CELLS):
            r, c = _cell_center(top, left, rr, cc)
            grid[r, c] = BACKGROUND
    return grid


def _random_seed_mask() -> np.ndarray:
    base = random.choice(BASE_SHAPES)
    mask = np.rot90(base, random.randrange(4))
    assert _is_valid_seed_mask(mask)
    return mask


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(SCAFFOLD_SIZE, 30)
    width = random.randint(SCAFFOLD_SIZE, 30)
    top = random.randint(0, height - SCAFFOLD_SIZE)
    left = random.randint(0, width - SCAFFOLD_SIZE)

    grid = _build_base_grid(height, width, top, left)
    red_index = random.randrange(OUTER_COUNT)
    seed_index = (red_index + 1) % OUTER_COUNT

    red_rr, red_cc = OUTER_RING[red_index]
    red_center = _cell_center(top, left, red_rr, red_cc)
    grid[red_center] = RED

    seed_mask = _random_seed_mask()
    seed_rr, seed_cc = OUTER_RING[seed_index]
    _stamp_tile(grid, _cell_center(top, left, seed_rr, seed_cc), seed_mask)

    if int(np.sum(grid == RED)) != 1:
        return None
    if int(np.sum(grid == BLUE)) not in (3, 4):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not np.any(out == BLUE) or int(np.sum(out == BLUE)) <= int(np.sum(grid == BLUE)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
