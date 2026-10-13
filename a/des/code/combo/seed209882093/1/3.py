import random

import numpy as np


BG = 0
PALETTE_LEN = 5
COLOR_POOL = list(range(1, 10))
GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
RING_COUNT_RANGE = (2, 5)
STEP_RANGE = (2, 3)
MARGIN_RANGE = (0, 4)
TOP_MARGIN_RANGE = (2, 6)
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 > r0:
        coords.extend((r1, c) for c in range(c0, c1 + 1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r0 + 1, r1))
        coords.extend((r, c1) for r in range(r0 + 1, r1))
    return coords


def _find_palette(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    pr, pc = map(int, coords[0])
    assert pc + PALETTE_LEN <= grid.shape[1]
    palette = [int(grid[pr, pc + i]) for i in range(PALETTE_LEN)]
    assert all(color != BG for color in palette)
    assert len(set(palette)) == PALETTE_LEN
    if pc + PALETTE_LEN < grid.shape[1]:
        assert int(grid[pr, pc + PALETTE_LEN]) == BG
    return palette, [(pr, pc + i) for i in range(PALETTE_LEN)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested concentric rectangular rings by indexing the corner palette
    with each ring's missing-pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette, palette_cells = _find_palette(g)

    ring_mask = g != BG
    for r, c in palette_cells:
        ring_mask[r, c] = False

    remaining_colors = set(map(int, np.unique(g[ring_mask]))) if np.any(ring_mask) else set()
    assert len(remaining_colors) == 1

    out = g.copy()
    active = ring_mask.copy()
    while np.any(active):
        box = _bbox(active)
        assert box is not None
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2

        coords = _boundary_coords(r0, c0, r1, c1)
        rows = np.array([r for r, _c in coords], dtype=int)
        cols = np.array([c for _r, c in coords], dtype=int)
        present = active[rows, cols]
        missing = len(coords) - int(np.count_nonzero(present))
        assert 0 <= missing < PALETTE_LEN

        out[rows[present], cols[present]] = int(palette[missing])
        active[rows[present], cols[present]] = False

    return out


def _sample_gap_counts(n_rings: int) -> list[int]:
    if n_rings == 2:
        if random.random() < 0.25:
            k = random.randrange(PALETTE_LEN)
            return [k, k]
        a, b = random.sample(range(PALETTE_LEN), 2)
        return [a, b]
    if random.random() < 0.5:
        return random.sample(range(PALETTE_LEN), n_rings)
    while True:
        pool_size = random.randint(2, min(4, PALETTE_LEN))
        pool = random.sample(range(PALETTE_LEN), pool_size)
        counts = [random.choice(pool) for _ in range(n_rings)]
        if len(set(counts)) >= 2 and len(set(counts)) < n_rings:
            return counts


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _carve_gaps(grid: np.ndarray, bbox: tuple[int, int, int, int], count: int) -> None:
    if count == 0:
        return
    r0, c0, r1, c1 = bbox
    sides = random.sample(["top", "bottom", "left", "right"], count)
    for side in sides:
        if side == "top":
            c = random.randint(c0 + 1, c1 - 1)
            grid[r0, c] = BG
        elif side == "bottom":
            c = random.randint(c0 + 1, c1 - 1)
            grid[r1, c] = BG
        elif side == "left":
            r = random.randint(r0 + 1, r1 - 1)
            grid[r, c0] = BG
        else:
            r = random.randint(r0 + 1, r1 - 1)
            grid[r, c1] = BG


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    n_rings = random.randint(*RING_COUNT_RANGE)

    v_steps = [random.randint(*STEP_RANGE) for _ in range(n_rings - 1)]
    h_steps = [random.randint(*STEP_RANGE) for _ in range(n_rings - 1)]

    min_outer_h = 3 + 2 * sum(v_steps)
    min_outer_w = 3 + 2 * sum(h_steps)
    avail_h = h - top_margin - bottom_margin
    avail_w = w - left_margin - right_margin
    if avail_h < min_outer_h or avail_w < min_outer_w:
        return None

    outer_h = random.randint(min_outer_h, avail_h)
    outer_w = random.randint(min_outer_w, avail_w)

    palette_and_ring = random.sample(COLOR_POOL, PALETTE_LEN + 1)
    palette = palette_and_ring[:PALETTE_LEN]
    ring_color = palette_and_ring[-1]

    gap_counts = _sample_gap_counts(n_rings)

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :PALETTE_LEN] = np.array(palette, dtype=int)

    outer_r0 = top_margin
    outer_c0 = left_margin
    outer_r1 = outer_r0 + outer_h - 1
    outer_c1 = outer_c0 + outer_w - 1

    r_inset = 0
    c_inset = 0
    ring_boxes: list[tuple[int, int, int, int]] = []
    for idx in range(n_rings):
        box = (
            outer_r0 + r_inset,
            outer_c0 + c_inset,
            outer_r1 - r_inset,
            outer_c1 - c_inset,
        )
        ring_boxes.append(box)
        if idx < n_rings - 1:
            r_inset += v_steps[idx]
            c_inset += h_steps[idx]

    for box in ring_boxes:
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        _draw_ring(grid, box, ring_color)

    for box, count in zip(ring_boxes, gap_counts):
        _carve_gaps(grid, box, count)

    if not np.all(grid[0, PALETTE_LEN:] == BG):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
