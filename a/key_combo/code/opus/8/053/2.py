from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GRID_SIZE = 4
PALETTE = [1, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 600

MASK_BBOX_CHOICES = [
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
]


def _quadrant(r: int, c: int, size: int) -> tuple[int, int]:
    half = size // 2
    return int(r >= half), int(c >= half)


def _apply_quadrant_flip(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    half = out.shape[0] // 2
    if r >= half:
        out = np.flipud(out)
    if c >= half:
        out = np.fliplr(out)
    return out


def _nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid != BG)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_quadrant(bbox: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    rr = r - r0
    cc = c - c0
    return int(rr >= height / 2.0), int(cc >= width / 2.0)


def _all_flips_distinct(grid: np.ndarray) -> bool:
    flips = [
        grid,
        np.fliplr(grid),
        np.flipud(grid),
        np.flipud(np.fliplr(grid)),
    ]
    for i in range(len(flips)):
        for j in range(i + 1, len(flips)):
            if np.array_equal(flips[i], flips[j]):
                return False
    return True


def _grow_connected_mask(height: int, width: int, count: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    while int(mask.sum()) < count:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        mask[random.choice(tuple(frontier))] = True
    return mask


def _paint_mask(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(2, min(4, len(coords)))
    colors = random.sample(PALETTE, color_count)
    values = colors[:]
    while len(values) < len(coords):
        values.append(random.choice(colors))
    random.shuffle(values)
    painted = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, values):
        painted[r, c] = int(color)
    return painted


def _sample_base_pattern() -> np.ndarray:
    box_h, box_w = random.choice(MASK_BBOX_CHOICES)
    r0 = random.randint(0, GRID_SIZE - box_h)
    c0 = random.randint(0, GRID_SIZE - box_w)
    area = box_h * box_w
    filled = random.randint(max(4, area // 2), area - 1)
    local_mask = _grow_connected_mask(box_h, box_w, filled)
    local_grid = _paint_mask(local_mask)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[r0 : r0 + box_h, c0 : c0 + box_w] = local_grid
    return grid


def _choose_red_cells(base: np.ndarray, require_bbox_conflict: bool) -> list[tuple[int, int]]:
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    red_count = random.randint(2, 4)
    distinct_quadrants = random.randint(2, min(4, red_count))
    chosen_quadrants = random.sample(quadrants, distinct_quadrants)

    candidates_by_quadrant: dict[tuple[int, int], list[tuple[int, int]]] = {q: [] for q in quadrants}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            candidates_by_quadrant[_quadrant(r, c, GRID_SIZE)].append((r, c))

    red_cells = [random.choice(candidates_by_quadrant[q]) for q in chosen_quadrants]
    while len(red_cells) < red_count:
        q = random.choice(chosen_quadrants)
        pool = [cell for cell in candidates_by_quadrant[q] if cell not in red_cells]
        if not pool:
            break
        red_cells.append(random.choice(pool))

    if require_bbox_conflict:
        trial = base.copy()
        for r, c in red_cells:
            trial[r, c] = RED
        bbox = _nonzero_bbox(trial)
        if bbox is None or bbox == (0, 0, GRID_SIZE - 1, GRID_SIZE - 1):
            return []
        if not any(_quadrant(r, c, GRID_SIZE) != _bbox_quadrant(bbox, r, c) for r, c in red_cells):
            return []

    return red_cells


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    size = g.shape[0]
    out = np.zeros((size * size, size * size), dtype=int)
    for r, c in np.argwhere(g == RED):
        rr = int(r)
        cc = int(c)
        stamp = _apply_quadrant_flip(g, rr, cc)
        out[rr * size : (rr + 1) * size, cc * size : (cc + 1) * size] = stamp
    return out


def _semantic_check(grid: np.ndarray, require_bbox_conflict: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if np.count_nonzero(g == RED) < 2 or np.count_nonzero(g == RED) > 4:
        return False
    if not np.any(g == BG):
        return False

    non_red = sorted(int(color) for color in np.unique(g) if color not in (BG, RED))
    if len(non_red) < 2:
        return False

    red_quadrants = {_quadrant(int(r), int(c), GRID_SIZE) for r, c in np.argwhere(g == RED)}
    if len(red_quadrants) < 2:
        return False

    if not _all_flips_distinct(g):
        return False

    bbox = _nonzero_bbox(g)
    if bbox is None:
        return False
    if require_bbox_conflict:
        if bbox == (0, 0, GRID_SIZE - 1, GRID_SIZE - 1):
            return False
        if not any(
            _quadrant(int(r), int(c), GRID_SIZE) != _bbox_quadrant(bbox, int(r), int(c))
            for r, c in np.argwhere(g == RED)
        ):
            return False

    transformed_by_quadrant: dict[tuple[int, int], np.ndarray] = {}
    for q in red_quadrants:
        qr, qc = q
        probe_r = 0 if qr == 0 else GRID_SIZE - 1
        probe_c = 0 if qc == 0 else GRID_SIZE - 1
        transformed_by_quadrant[q] = _apply_quadrant_flip(g, probe_r, probe_c)
    variants = list(transformed_by_quadrant.values())
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if np.array_equal(variants[i], variants[j]):
                return False

    out = main(g)
    if out.shape != (GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE):
        return False

    expected = _expected_output(g)
    if not np.array_equal(out, expected):
        return False

    size = GRID_SIZE
    for r in range(size):
        for c in range(size):
            block = out[r * size : (r + 1) * size, c * size : (c + 1) * size]
            if int(g[r, c]) == RED:
                want = _apply_quadrant_flip(g, r, c)
                if not np.array_equal(block, want):
                    return False
            elif np.any(block != BG):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the whole input into red-selected output blocks, with the stamp flip encoded by the red pixel's quadrant.

    Input structure:
    - The input is an even-sized square grid on a black background.
    - Non-red colored pixels form an asymmetric source pattern, and red pixels act as control cells.
    - The same red pixel coordinate is read in two ways: as an output block address and as a quadrant code within the full input grid.

    Rule:
    - Split the input into four equal quadrants by the full-grid midpoint.
    - Build an output block grid where each input cell maps to one input-sized block.
    - For each red pixel, copy the complete input pattern into that pixel's output block.
    - Use no flip for top-left red pixels, a left-right flip for top-right, an up-down flip for bottom-left, and both flips for bottom-right; non-red cells contribute all-black blocks.

    Key insight:
    - A red control pixel simultaneously chooses where to stamp and how to orient the stamp, so both exact cell position and quadrant membership are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even-sized square grid"
    assert h * h <= 30 and w * w <= 30, "output must stay within ARC size limits"

    out = np.zeros((h * h, w * w), dtype=int)
    for r, c in np.argwhere(g == RED):
        rr = int(r)
        cc = int(c)
        stamp = _apply_quadrant_flip(g, rr, cc)
        out[rr * h : (rr + 1) * h, cc * w : (cc + 1) * w] = stamp
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        require_bbox_conflict = random.random() < 0.6
        base = _sample_base_pattern()
        red_cells = _choose_red_cells(base, require_bbox_conflict)
        if not red_cells:
            continue

        grid = base.copy()
        for r, c in red_cells:
            grid[r, c] = RED

        if _semantic_check(grid, require_bbox_conflict):
            return grid
    raise RuntimeError("failed to generate a valid input")
