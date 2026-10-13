from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIZE_CHOICES = (5, 7)
SIDES = ("up", "right", "down", "left")
ROTATE_CW_STEPS = {
    "up": 1,
    "right": 2,
    "down": 3,
    "left": 0,
}
GEN_ATTEMPTS = 400
MIN_GRID_SIDE = 18
MAX_GRID_SIDE = 30
MAX_OBJECTS = 5
SLOT_MIN = 9  # fits a 7x7 object plus a 1-cell safety margin on each side


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _gap_position(size: int, side: str) -> tuple[int, int]:
    mid = size // 2
    if side == "up":
        return 0, mid
    if side == "right":
        return mid, size - 1
    if side == "down":
        return size - 1, mid
    if side == "left":
        return mid, 0
    raise AssertionError(f"unknown side: {side}")


def _ring_patch(size: int, color: int, gap_side: str) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gr, gc = _gap_position(size, gap_side)
    patch[gr, gc] = BG
    return patch


def _identify_gap_side(patch: np.ndarray, color: int) -> str:
    size = patch.shape[0]
    assert patch.shape == (size, size)
    assert size in SIZE_CHOICES

    sides = [side for side in SIDES if int(patch[_gap_position(size, side)]) == BG]
    assert len(sides) == 1, "expected exactly one side-center gap"
    gap_side = sides[0]
    expected = _ring_patch(size, color, gap_side)
    assert np.array_equal(patch, expected), "object must be a hollow square ring with one centered gap"
    return gap_side


def _transform_patch(patch: np.ndarray, color: int, gap_side: str) -> np.ndarray:
    turns = ROTATE_CW_STEPS[gap_side]
    rotated = np.rot90(patch, -turns).copy()
    rotated[1:-1, 1:-1] = color
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each gapped square ring according to its gap direction, then fill its cavity.

    Input structure:
    - Black background.
    - Each foreground object is a single-color 5x5 or 7x7 square ring.
    - The ring has a 1-pixel-wide centered gap on exactly one side.
    - The interior cavity is entirely background.

    Rule:
    - Gap on top    -> rotate object clockwise 90 degrees.
    - Gap on right  -> rotate object clockwise 180 degrees.
    - Gap on bottom -> rotate object clockwise 270 degrees.
    - Gap on left   -> keep orientation unchanged.
    - After rotation, fill the ring's interior cavity with the ring color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    components = _components4(g != BG)
    assert components, "expected at least one foreground object"

    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        color = colors.pop()

        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w, "each object must have a square bounding box"
        assert h in SIZE_CHOICES, "only 5x5 and 7x7 rings are valid"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        gap_side = _identify_gap_side(patch, color)
        out[r0 : r1 + 1, c0 : c1 + 1] = _transform_patch(patch, color, gap_side)

    return out


def _slot_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base = length // parts
    extra = length % parts
    start = 0
    bounds: list[tuple[int, int]] = []
    for i in range(parts):
        span = base + (1 if i < extra else 0)
        bounds.append((start, start + span - 1))
        start += span
    return bounds


def _sample_directions(rng: random.Random, count: int) -> list[str]:
    dirs: list[str]
    if count >= 4:
        dirs = list(SIDES)
        rng.shuffle(dirs)
        while len(dirs) < count:
            dirs.append(rng.choice(SIDES))
    else:
        dirs = [rng.choice(SIDES) for _ in range(count)]
    rng.shuffle(dirs)
    return dirs


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)
    width = rng.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)

    max_slot_rows = height // SLOT_MIN
    max_slot_cols = width // SLOT_MIN
    if max_slot_rows == 0 or max_slot_cols == 0:
        return None

    slot_rows = rng.randint(1, max_slot_rows)
    slot_cols = rng.randint(1, max_slot_cols)
    slot_count = slot_rows * slot_cols
    object_count = rng.randint(1, min(MAX_OBJECTS, slot_count))

    row_bounds = _slot_bounds(height, slot_rows)
    col_bounds = _slot_bounds(width, slot_cols)
    slots = [(rb, cb) for rb in row_bounds for cb in col_bounds]
    chosen_slots = rng.sample(slots, object_count)

    colors = rng.sample(COLORS, object_count)
    directions = _sample_directions(rng, object_count)

    grid = np.zeros((height, width), dtype=int)
    for idx, (((r0, r1), (c0, c1)), color, direction) in enumerate(zip(chosen_slots, colors, directions)):
        del idx  # loop index only helps make zip unpacking explicit
        max_size = min(r1 - r0 + 1, c1 - c0 + 1) - 2
        valid_sizes = [size for size in SIZE_CHOICES if size <= max_size]
        if not valid_sizes:
            return None
        size = rng.choice(valid_sizes)

        top_min = r0 + 1
        top_max = r1 - size
        left_min = c0 + 1
        left_max = c1 - size
        if top_min > top_max or left_min > left_max:
            return None

        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)
        patch = _ring_patch(size, color, direction)
        grid[top : top + size, left : left + size] = patch

    try:
        transformed = main(grid)
    except AssertionError:
        return None
    except Exception:
        return None

    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
