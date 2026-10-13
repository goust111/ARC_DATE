from __future__ import annotations

import random

import numpy as np


AZURE = 8
BLUE = 1
RED = 2
GREEN = 3

MAX_ATTEMPTS = 500
SHAPE_SIDE_RANGE = (3, 7)
SHAPE_AREA_RANGE = (5, 15)
GRID_SIDE_RANGE = (12, 30)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size, "expected a non-empty object"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _all_rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_rotation_key(np.rot90(mask, k)) for k in range(4)}


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    target = occupied[r0 : r0 + h, c0 : c0 + w]
    return not bool((target & mask).any())


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the blue object to the red object's orientation and overlay the aligned shapes.

    Input structure:
    - Azure is the background.
    - There is exactly one Red foreground object and one Blue foreground object.
    - The two foreground objects have the same binary shape up to a 0/90/180/270 degree rotation, but they appear in different orientations and positions.

    Rule:
    - Crop the Red and Blue object masks to their tight bounding boxes.
    - Find the rotation that makes the Blue mask match the Red mask's orientation.
    - Keep the Red object fixed, clear the Blue object's original cells, and paste the rotated Blue object so its bounding-box center coincides with the Red object's bounding-box center.
    - In the overlay, Red-only cells stay Red, Blue-only cells are Blue, and cells occupied by both become Green.

    Key insight:
    The objects match by shape after rotation; the output is not a copy of either input object alone, but the color-coded overlap after rotation and center alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    red_global = g == RED
    blue_global = g == BLUE
    assert red_global.any(), "expected a red object"
    assert blue_global.any(), "expected a blue object"

    rr0, rc0, rr1, rc1 = _bbox(red_global)
    br0, bc0, _br1, _bc1 = _bbox(blue_global)
    red_mask = red_global[rr0 : rr1 + 1, rc0 : rc1 + 1].copy()
    blue_mask = blue_global[br0 : _br1 + 1, bc0 : _bc1 + 1].copy()

    rotated_blue = None
    for k in range(4):
        candidate = np.rot90(blue_mask, k).copy()
        if candidate.shape == red_mask.shape and np.array_equal(candidate, red_mask):
            rotated_blue = candidate
            break
    assert rotated_blue is not None, "blue object must match red after rotation"

    h, w = rotated_blue.shape
    red_center_r2 = rr0 + rr1
    red_center_c2 = rc0 + rc1
    dst_r2 = red_center_r2 - (h - 1)
    dst_c2 = red_center_c2 - (w - 1)
    assert dst_r2 % 2 == 0 and dst_c2 % 2 == 0, "aligned top-left must land on cells"
    dst_r = dst_r2 // 2
    dst_c = dst_c2 // 2
    assert 0 <= dst_r and 0 <= dst_c and dst_r + h <= H and dst_c + w <= W

    out = np.full_like(g, AZURE)
    out[red_global] = RED
    for dr, dc in np.argwhere(rotated_blue):
        r = int(dst_r + dr)
        c = int(dst_c + dc)
        out[r, c] = GREEN if int(out[r, c]) == RED else BLUE
    return out


def _grow_mask(rows: int, cols: int, area: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    start = (random.randrange(rows), random.randrange(cols))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True

    return _crop_mask(mask)


def _make_shape() -> np.ndarray | None:
    rows = random.randint(*SHAPE_SIDE_RANGE)
    cols = random.randint(*SHAPE_SIDE_RANGE)
    max_area = min(SHAPE_AREA_RANGE[1], rows * cols - 1)
    if max_area < SHAPE_AREA_RANGE[0]:
        return None
    area = random.randint(SHAPE_AREA_RANGE[0], max_area)
    mask = _grow_mask(rows, cols, area)
    h, w = mask.shape
    if h < 2 or w < 2:
        return None
    if int(mask.sum()) < SHAPE_AREA_RANGE[0]:
        return None
    if len(_all_rotation_keys(mask)) != 4:
        return None
    return mask


def _place_two_objects(red_mask: np.ndarray, blue_mask: np.ndarray) -> np.ndarray | None:
    min_rows = max(GRID_SIDE_RANGE[0], red_mask.shape[0], blue_mask.shape[0]) + 4
    min_cols = max(GRID_SIDE_RANGE[0], red_mask.shape[1], blue_mask.shape[1]) + 4
    if min_rows > GRID_SIDE_RANGE[1] or min_cols > GRID_SIDE_RANGE[1]:
        return None

    rows = random.randint(min_rows, GRID_SIDE_RANGE[1])
    cols = random.randint(min_cols, GRID_SIDE_RANGE[1])
    grid = np.full((rows, cols), AZURE, dtype=int)

    rh, rw = red_mask.shape
    bh, bw = blue_mask.shape
    rr = random.randint(1, rows - rh - 1)
    rc = random.randint(1, cols - rw - 1)
    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[rr : rr + rh, rc : rc + rw][red_mask] = True

    blocked = _dilate(occupied)
    candidates = [
        (r, c)
        for r in range(1, rows - bh)
        for c in range(1, cols - bw)
        if _can_place(blocked, blue_mask, r, c)
    ]
    if not candidates:
        return None

    br, bc = random.choice(candidates)
    _paint(grid, red_mask, rr, rc, RED)
    _paint(grid, blue_mask, br, bc, BLUE)
    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray | None:
    try:
        return main(inp)
    except AssertionError:
        return None


def _gen_attempt() -> np.ndarray | None:
    base = _make_shape()
    if base is None:
        return None

    red_turns = random.randrange(4)
    required_turns = random.choice([1, 2, 3])
    blue_turns = (red_turns - required_turns) % 4
    red_mask = np.rot90(base, red_turns).copy()
    blue_mask = np.rot90(base, blue_turns).copy()
    if np.array_equal(red_mask, blue_mask):
        return None

    grid = _place_two_objects(red_mask, blue_mask)
    if grid is None:
        return None

    actual = _expected_output(grid)
    if actual is None:
        return None
    if actual.ndim != 2 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-overlay input")
