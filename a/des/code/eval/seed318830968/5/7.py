import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8

RAY_DIRS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]

OUTER_RING = [
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (2, -1),
    (2, -2),
    (1, -2),
    (0, -2),
    (-1, -2),
    (-2, -2),
    (-2, -1),
]

COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
PERIOD_CHOICES = (2, 3, 4, 5)

GRID_MIN = 15
GRID_MAX = 30
CLOCK_COUNT_MAX = 4
PLACEMENT_RETRIES = 100
GENERATION_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _infer_ring(values: list[int | None]) -> list[int]:
    assert len(values) == len(OUTER_RING)
    missing = [i for i, value in enumerate(values) if value is None]
    assert len(missing) <= 1

    for period in range(1, len(values) + 1):
        by_class: list[int | None] = [None] * period
        valid = True
        for idx, value in enumerate(values):
            if value is None:
                continue
            slot = idx % period
            if by_class[slot] is None:
                by_class[slot] = int(value)
            elif by_class[slot] != int(value):
                valid = False
                break
        if not valid:
            continue
        if missing:
            if by_class[missing[0] % period] is None:
                continue

        filled: list[int] = []
        for idx, value in enumerate(values):
            if value is None:
                replacement = by_class[idx % period]
                if replacement is None:
                    valid = False
                    break
                filled.append(int(replacement))
            else:
                filled.append(int(value))
        if valid:
            return filled

    raise AssertionError("could not infer a periodic outer ring")


def _detect_clock(g: np.ndarray, r: int, c: int) -> dict | None:
    if int(g[r, c]) != BLACK:
        return None

    ray_idx = None
    for idx, (dr, dc) in enumerate(RAY_DIRS):
        rr = r + dr
        cc = c + dc
        value = int(g[rr, cc])
        if value == AZURE:
            if ray_idx is not None:
                return None
            ray_idx = idx
        elif value != GREY:
            return None
    if ray_idx is None:
        return None

    dr, dc = RAY_DIRS[ray_idx]
    if not _in_bounds(g.shape, r + 3 * dr, c + 3 * dc):
        return None

    second = int(g[r + 2 * dr, c + 2 * dc])
    third = int(g[r + 3 * dr, c + 3 * dc])
    if second == AZURE:
        length = 3 if third == AZURE else 2
    else:
        if third == AZURE:
            return None
        length = 1

    outer_values: list[int | None] = []
    outer_azure_indices: list[int] = []
    for idx, (odr, odc) in enumerate(OUTER_RING):
        value = int(g[r + odr, c + odc])
        if value == AZURE:
            outer_values.append(None)
            outer_azure_indices.append(idx)
        else:
            if value in (BLACK, GREY):
                return None
            outer_values.append(value)

    expected_outer = [] if length == 1 else [2 * ray_idx]
    if outer_azure_indices != expected_outer:
        return None

    return {
        "center": (r, c),
        "ray_idx": ray_idx,
        "length": length,
        "ring": _infer_ring(outer_values),
    }


def _find_clocks(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    clocks: list[dict] = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            clock = _detect_clock(g, r, c)
            if clock is not None:
                clocks.append(clock)
    assert clocks, "expected at least one clock"
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each clock dial's outer-ring colors clockwise by the pointer length.

    Input structure:
    - Black background.
    - Each clock has a 5x5 hollow square dial: grey inner ring, colored outer ring, black center.
    - An azure pointer starts at the center and extends 1-3 cells in one of eight directions.

    Rule:
    - Read each pointer length.
    - Treat the outer 16 dial cells, ordered clockwise from the top-middle cell, as a cyclic array.
    - Rotate that array clockwise by the pointer length.
    - Restore the inner ring to grey, remove any pointer segment outside the dial, and keep the center black.

    Key insight:
    - Pointer direction is irrelevant; only its length controls the cyclic rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for clock in _find_clocks(g):
        r, c = clock["center"]
        ray_idx = int(clock["ray_idx"])
        length = int(clock["length"])
        shifted_ring = np.roll(np.asarray(clock["ring"], dtype=int), length)

        out[r, c] = BLACK
        for dr, dc in RAY_DIRS:
            out[r + dr, c + dc] = GREY
        for (dr, dc), color in zip(OUTER_RING, shifted_ring):
            out[r + dr, c + dc] = int(color)

        if length == 3:
            dr, dc = RAY_DIRS[ray_idx]
            out[r + 3 * dr, c + 3 * dc] = BLACK

    return out


def _clock_slots(size: int) -> int:
    return 1 + (size - 9) // 9


def _sample_clock_spec() -> dict:
    while True:
        length = random.randint(1, 3)
        period = random.choice(PERIOD_CHOICES)
        if length % period == 0:
            continue
        base = random.sample(COLOR_POOL, period)
        ring = [base[i % period] for i in range(len(OUTER_RING))]
        return {
            "ray_idx": random.randrange(len(RAY_DIRS)),
            "length": length,
            "ring": ring,
        }


def _place_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    centers: list[tuple[int, int]] = []

    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(4, h - 5)
            c = random.randint(4, w - 5)
            if occupied[r - 4 : r + 5, c - 4 : c + 5].any():
                continue
            occupied[r - 4 : r + 5, c - 4 : c + 5] = True
            centers.append((r, c))
            placed = True
            break
        if not placed:
            return None

    return centers


def _paint_clock(grid: np.ndarray, center: tuple[int, int], spec: dict) -> None:
    r, c = center
    ray_idx = int(spec["ray_idx"])
    length = int(spec["length"])

    grid[r, c] = BLACK
    for dr, dc in RAY_DIRS:
        grid[r + dr, c + dc] = GREY
    for (dr, dc), color in zip(OUTER_RING, spec["ring"]):
        grid[r + dr, c + dc] = int(color)

    dr, dc = RAY_DIRS[ray_idx]
    for step in range(1, length + 1):
        grid[r + step * dr, c + step * dc] = AZURE


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    max_clocks = min(CLOCK_COUNT_MAX, _clock_slots(h) * _clock_slots(w))
    count = random.randint(1, max_clocks)

    centers = _place_centers(h, w, count)
    if centers is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for center in centers:
        _paint_clock(grid, center, _sample_clock_spec())

    detected = _find_clocks(grid)
    if len(detected) != count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out == AZURE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock puzzle input")
