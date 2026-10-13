from __future__ import annotations

import random

import numpy as np


BG = 0
RING = 5
RING_OFFSETS = (
    (-2, -1), (-2, 0), (-2, 1),
    (-1, -2), (-1, 2),
    (0, -2), (0, 2),
    (1, -2), (1, 2),
    (2, -1), (2, 0), (2, 1),
)
CORNER_OFFSETS = ((-2, -2), (-2, 2), (2, -2), (2, 2))
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

FOOTPRINT_H = 9
FOOTPRINT_W = 5
CLOCK_PADDING = 1
MIN_CLOCKS = 2
MAX_CLOCKS = 4
MAX_GENERATION_ATTEMPTS = 400
POINTER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SATELLITE_COLORS = POINTER_COLORS


def _detect_clock(grid: np.ndarray, center_r: int, center_c: int) -> tuple[int, int, int, int] | None:
    """Return (sat_top, sat_left, clockwise_turns, pointer_color) for a valid clock."""
    h, w = grid.shape
    if not (2 <= center_r < h - 2 and 2 <= center_c < w - 2):
        return None

    for dr, dc in RING_OFFSETS:
        if int(grid[center_r + dr, center_c + dc]) != RING:
            return None
    for dr, dc in CORNER_OFFSETS:
        if int(grid[center_r + dr, center_c + dc]) != BG:
            return None

    pointer_color = int(grid[center_r, center_c])
    if pointer_color in {BG, RING}:
        return None

    direction = None
    for idx, (pdr, pdc) in enumerate(DIRECTIONS):
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                value = int(grid[center_r + dr, center_c + dc])
                expected_pointer = (dr, dc) in {(0, 0), (pdr, pdc)}
                if expected_pointer and value != pointer_color:
                    ok = False
                if not expected_pointer and value != BG:
                    ok = False
        if ok:
            direction = idx
            break
    if direction is None:
        return None

    sat_top = center_r - 6
    sat_left = center_c - 1
    if sat_top < 0 or sat_left < 0 or sat_top + 3 > h or sat_left + 3 > w:
        return None
    if np.any(grid[center_r - 3, center_c - 1 : center_c + 2] != BG):
        return None

    satellite = grid[sat_top : sat_top + 3, sat_left : sat_left + 3]
    if satellite.shape != (3, 3) or np.any(satellite == BG):
        return None
    if len({int(v) for v in satellite.flat}) < 2:
        return None

    return sat_top, sat_left, direction, pointer_color


def _find_clocks(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    clocks: list[tuple[int, int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            info = _detect_clock(grid, r, c)
            if info is not None:
                clocks.append(info)
    assert clocks, "expected at least one clock"
    pointer_colors = [pointer_color for *_rest, pointer_color in clocks]
    assert len(pointer_colors) == len(set(pointer_colors)), "clock pointer colors must be unique"
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each satellite above a clock according to that clock's pointer direction.

    Input structure:
    - The background is black.
    - Each clock is a 5x5 grey ring with black corners and a two-cell colored pointer inside the 3x3 center area.
    - Every clock has a 3x3 non-black satellite directly above it, separated by one black row.

    Rule:
    - Detect every clock and read the pointer direction: up means 0 degrees, right means 90 degrees clockwise, down means 180 degrees, and left means 270 degrees clockwise.
    - Rotate the corresponding 3x3 satellite above that clock by the indicated clockwise amount.
    - Write each rotated satellite back to its original location while leaving the clocks and all other cells unchanged.

    Key insight:
    - The grey ring identifies the clock body, while the colored center-and-neighbor pointer encodes the clockwise rotation to apply to the satellite directly above it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    for sat_top, sat_left, quarter_turns, _pointer_color in _find_clocks(g):
        satellite = g[sat_top : sat_top + 3, sat_left : sat_left + 3]
        out[sat_top : sat_top + 3, sat_left : sat_left + 3] = np.rot90(satellite, -quarter_turns)
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _place_footprints(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    candidates = [(top, left) for top in range(h - FOOTPRINT_H + 1) for left in range(w - FOOTPRINT_W + 1)]
    random.shuffle(candidates)

    boxes: list[tuple[int, int, int, int]] = []
    for top, left in candidates:
        box = (top, left, top + FOOTPRINT_H - 1, left + FOOTPRINT_W - 1)
        if any(_boxes_overlap(box, other, CLOCK_PADDING) for other in boxes):
            continue
        boxes.append(box)
        if len(boxes) == count:
            return boxes
    return None


def _random_satellite(seen: set[tuple[int, ...]]) -> np.ndarray | None:
    for _ in range(200):
        palette = random.sample(SATELLITE_COLORS, random.randint(2, 4))
        block = np.array(random.choices(palette, k=9), dtype=int).reshape(3, 3)
        if len({int(v) for v in block.flat}) < 2:
            continue
        if any(np.array_equal(block, np.rot90(block, k)) for k in (1, 2, 3)):
            continue
        key = tuple(int(v) for v in block.flat)
        if key in seen:
            continue
        seen.add(key)
        return block
    return None


def _draw_clock(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    pointer_color: int,
    direction: int,
    satellite: np.ndarray,
) -> None:
    top, left, _bottom, _right = box
    center_r = top + 6
    center_c = left + 2

    grid[top : top + 3, left + 1 : left + 4] = satellite
    for dr, dc in RING_OFFSETS:
        grid[center_r + dr, center_c + dc] = RING
    pdr, pdc = DIRECTIONS[direction]
    grid[center_r, center_c] = pointer_color
    grid[center_r + pdr, center_c + pdc] = pointer_color


def _gen_attempt() -> np.ndarray | None:
    clock_count = random.randint(MIN_CLOCKS, MAX_CLOCKS)
    min_h = 11 + 2 * clock_count
    min_w = 7 + 2 * clock_count
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    boxes = _place_footprints(h, w, clock_count)
    if boxes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    directions = random.sample(range(4), clock_count)
    pointer_colors = random.sample(POINTER_COLORS, clock_count)

    seen_satellites: set[tuple[int, ...]] = set()
    for box, pointer_color, direction in zip(boxes, pointer_colors, directions):
        satellite = _random_satellite(seen_satellites)
        if satellite is None:
            return None
        _draw_clock(grid, box, pointer_color, direction, satellite)

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
