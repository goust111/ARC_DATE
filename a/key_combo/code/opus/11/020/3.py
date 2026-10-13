from __future__ import annotations

import math
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GRID_SIZES = (7, 9, 11)
SHAPE_SIZE_RANGE = (5, 9)
MASK_RATIO_RANGE = (0.30, 0.40)
MAX_GENERATION_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the red pattern four ways and keep only the masked pixels with angle-based repainting.

    Input structure:
    - The input is an odd square grid on a black background.
    - Red cells form one connected asymmetric pattern.
    - Grey cells form a scattered mask and are separate from the red input cells.

    Rule:
    - Extract the red pattern and the grey mask from the same canvas.
    - Rotate the red pattern around the grid center by 0, 90 clockwise, 180, and 270 clockwise.
    - Start from an all-black output and draw the four rotated pattern masks in that order.
    - For each rotation, keep only cells that also belong to the grey mask; recolor them as red, blue, green, and yellow respectively.
    - Later rotations overwrite earlier ones where multiple retained rotations land on the same cell.

    Key insight:
    - The grey layer does not move; it selects different subsets from the four centered rotations, and overlap conflicts are resolved only by rotation order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"

    pattern = g == RED
    mask = g == GREY
    out = np.full(g.shape, BLACK, dtype=int)

    rotations = [
        pattern,
        np.rot90(pattern, -1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 1),
    ]
    colors = (RED, BLUE, GREEN, YELLOW)
    for rotated, color in zip(rotations, colors):
        keep = rotated & mask
        out[keep] = int(color)
    return out


def _strict_quadrant(r: int, c: int, center: int) -> int | None:
    if r == center or c == center:
        return None
    if r < center and c < center:
        return 0
    if r < center and c > center:
        return 1
    if r > center and c < center:
        return 2
    return 3


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _local_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rotationally_asymmetric(cells: set[tuple[int, int]]) -> bool:
    mask = _local_mask(cells)
    return not any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _build_pattern(shape_size: int) -> set[tuple[int, int]] | None:
    radius = int(np.random.choice((2, 3)))
    allowed = {
        (r, c)
        for r in range(radius + 1)
        for c in range(radius + 1)
        if (r, c) != (0, 0)
    }
    cells: set[tuple[int, int]] = {(0, 1), (1, 0), (1, 1)}
    growth_steps = shape_size - len(cells)

    while len(cells) < shape_size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in allowed and nxt not in cells:
                    frontier.append(nxt)
        if not frontier:
            return None

        candidates = sorted(set(frontier))
        weights = []
        for r, c in candidates:
            weight = 2.0
            if r > 0 and c > 0:
                weight += 1.5
            if r == 0 or c == 0:
                weight += 0.7
            if (r == c) and r > 0:
                weight -= 0.4
            weights.append(weight)
        idx = int(np.random.choice(len(candidates), p=np.array(weights) / np.sum(weights)))
        cells.add(candidates[idx])

    strict_interior = sum(r > 0 and c > 0 for r, c in cells)
    axis_touch_count = sum(r == 0 or c == 0 for r, c in cells)
    if strict_interior < 2 or axis_touch_count < 2:
        return None
    if not _is_connected(cells):
        return None
    if not _rotationally_asymmetric(cells):
        return None
    if len(cells) != shape_size:
        return None
    if growth_steps >= 3 and radius == 2 and max(max(r, c) for r, c in cells) < 2:
        return None
    return cells


def _pattern_grid(size: int, local_cells: set[tuple[int, int]]) -> np.ndarray:
    center = size // 2
    grid = np.zeros((size, size), dtype=int)
    for dr, dc in local_cells:
        grid[center + dr, center + dc] = RED
    return grid


def _rotation_masks(pattern_grid: np.ndarray) -> list[np.ndarray]:
    pattern = pattern_grid == RED
    return [
        pattern,
        np.rot90(pattern, -1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 1),
    ]


def _choose_one(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    pick = coords[int(np.random.randint(len(coords)))]
    return int(pick[0]), int(pick[1])


def _build_mask(pattern_grid: np.ndarray) -> np.ndarray | None:
    size = int(pattern_grid.shape[0])
    center = size // 2
    total = size * size
    target = int(np.random.randint(
        math.ceil(MASK_RATIO_RANGE[0] * total),
        math.floor(MASK_RATIO_RANGE[1] * total) + 1,
    ))

    rotations = _rotation_masks(pattern_grid)
    base = rotations[0]
    non_base = ~base

    visible_blue = rotations[1] & ~rotations[2] & ~rotations[3] & non_base
    visible_green = rotations[2] & ~rotations[3] & non_base
    visible_yellow = rotations[3] & non_base
    conflict = (
        ((rotations[1] & rotations[2]) | (rotations[1] & rotations[3]) | (rotations[2] & rotations[3]))
        & non_base
    )

    if not visible_blue.any() or not visible_green.any() or not visible_yellow.any() or not conflict.any():
        return None

    chosen: set[tuple[int, int]] = {
        _choose_one(visible_blue),
        _choose_one(visible_green),
        _choose_one(visible_yellow),
        _choose_one(conflict),
    }

    available = np.zeros_like(base, dtype=bool)
    available[non_base] = True

    for quadrant in range(4):
        pool = np.zeros_like(base, dtype=bool)
        for r in range(size):
            for c in range(size):
                if available[r, c] and _strict_quadrant(r, c, center) == quadrant:
                    pool[r, c] = True
        if not pool.any():
            return None
        chosen.add(_choose_one(pool))

    useful = (rotations[1] | rotations[2] | rotations[3]) & non_base
    extra_useful_coords = np.argwhere(useful)
    if len(extra_useful_coords) < 6:
        return None
    extra_useful_take = min(len(extra_useful_coords), int(np.random.randint(3, 6)))
    order = np.random.permutation(len(extra_useful_coords))
    for idx in order[:extra_useful_take]:
        r, c = extra_useful_coords[idx]
        chosen.add((int(r), int(c)))

    available_coords = np.argwhere(non_base)
    if len(chosen) > target or len(available_coords) < target:
        return None

    remaining = [tuple(map(int, rc)) for rc in available_coords if tuple(map(int, rc)) not in chosen]
    np.random.shuffle(remaining)
    chosen_list = list(chosen) + remaining[: target - len(chosen)]

    mask = np.zeros_like(pattern_grid, dtype=bool)
    for r, c in chosen_list:
        mask[r, c] = True

    quadrant_hits = {
        _strict_quadrant(r, c, center)
        for r, c in np.argwhere(mask)
        if _strict_quadrant(int(r), int(c), center) is not None
    }
    if quadrant_hits != {0, 1, 2, 3}:
        return None

    return mask


def _generate_attempt() -> np.ndarray | None:
    size = int(np.random.choice(GRID_SIZES))
    shape_size = int(np.random.randint(*SHAPE_SIZE_RANGE))
    local_cells = _build_pattern(shape_size)
    if local_cells is None:
        return None

    pattern_grid = _pattern_grid(size, local_cells)
    mask = _build_mask(pattern_grid)
    if mask is None:
        return None

    inp = np.zeros((size, size), dtype=int)
    inp[mask] = GREY
    inp[pattern_grid == RED] = RED

    rotate_k = int(np.random.randint(4))
    if rotate_k:
        inp = np.rot90(inp, rotate_k)

    out = main(inp)
    colors = set(map(int, out.flatten())) - {BLACK}
    if not {BLUE, GREEN, YELLOW}.issubset(colors):
        return None

    pattern = inp == RED
    if np.count_nonzero(pattern) != shape_size:
        return None
    if np.count_nonzero(inp == GREY) < math.ceil(MASK_RATIO_RANGE[0] * size * size):
        return None

    rotations = [
        np.rot90(pattern, 0),
        np.rot90(pattern, -1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 1),
    ]
    overlap_cells = ((rotations[1].astype(int) + rotations[2].astype(int) + rotations[3].astype(int)) >= 2) & (inp == GREY)
    if not overlap_cells.any():
        return None
    if not (4 <= np.count_nonzero(out) <= 16):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid selective-rotation-mask input")
