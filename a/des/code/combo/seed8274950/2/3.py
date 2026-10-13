from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    legend_rows = [int(r) for r in np.flatnonzero(g[:, 0] != bg)]
    assert legend_rows, "left-edge legend is required"
    legend_colors = [int(g[r, 0]) for r in legend_rows]
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be distinct"

    work = g.copy()
    work[legend_rows, 0] = bg

    comps = _components(work != bg)
    assert len(comps) == len(legend_colors), "legend/object count mismatch"

    areas = [len(comp) for comp in comps]
    assert len(set(areas)) == len(areas), "object areas must be distinct"

    out = work.copy()
    order = np.argsort(areas)
    for rank, comp_idx in enumerate(order):
        color = legend_colors[rank]
        for r, c in comps[int(comp_idx)]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object according to the left-edge legend ordered by area rank.

    Input structure:
    - The background color fills most of the grid.
    - The leftmost column contains N legend pixels with distinct colors.
    - Elsewhere there are N separated monochrome 4-connected objects with distinct areas.

    Rule:
    - Sort the objects by area from smallest to largest.
    - Assign the k-th legend color from top to bottom to the k-th smallest object.
    - Recolor each object in place and erase the legend to background.
    """
    return _solve(grid)


# --------------- generator ---------------

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 25)
NUM_OBJECTS_RANGE = (2, 5)
AREA_CHOICES = tuple(range(2, 11))
SHAPE_MAX_DIM = 6
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 400


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(mask, int(RNG.integers(0, 4)))
    if float(RNG.random()) < 0.5:
        transformed = np.fliplr(transformed)
    if float(RNG.random()) < 0.5:
        transformed = np.flipud(transformed)
    return np.array(transformed, dtype=bool, copy=True)


def _bbox_candidates(area: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    max_dim = min(SHAPE_MAX_DIM, max(area, 2))
    slack = max(2, area // 2)
    for h in range(1, max_dim + 1):
        for w in range(1, max_dim + 1):
            box_area = h * w
            if area <= box_area <= area + slack:
                candidates.append((h, w))
    return candidates


def _random_shape(area: int) -> np.ndarray:
    candidates = _bbox_candidates(area)
    assert candidates

    for _ in range(SHAPE_RETRIES):
        h, w = candidates[int(RNG.integers(len(candidates)))]
        mask = np.zeros((h, w), dtype=bool)
        seed_r = int(RNG.integers(h))
        seed_c = int(RNG.integers(w))
        mask[seed_r, seed_c] = True

        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(RNG.integers(len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue

        mask = _trim(mask)
        if area >= 4 and min(mask.shape) == 1 and float(RNG.random()) < 0.8:
            continue
        if area >= 6 and int(mask.sum()) == int(mask.shape[0] * mask.shape[1]) and float(RNG.random()) < 0.6:
            continue
        return _random_transform(mask)

    # Guaranteed fallback: a compact rectangle with a missing corner if possible.
    h = max(1, min(SHAPE_MAX_DIM, int(np.sqrt(area))))
    while h > 1 and area % h != 0:
        h -= 1
    w = int(np.ceil(area / h))
    mask = np.zeros((h, w), dtype=bool)
    filled = 0
    for r in range(h):
        for c in range(w):
            if filled < area:
                mask[r, c] = True
                filled += 1
    return _random_transform(_trim(mask))


def _can_place(blocked: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    for dr, dc in np.argwhere(shape):
        if blocked[r0 + int(dr), c0 + int(dc)]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(shape):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                blocked[nr, nc] = True


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    bg = int(RNG.integers(10))
    num_objects = int(RNG.integers(NUM_OBJECTS_RANGE[0], NUM_OBJECTS_RANGE[1] + 1))

    if num_objects > len(AREA_CHOICES) or num_objects > h:
        return None

    palette = [color for color in range(10) if color != bg]
    legend_start = int(RNG.integers(0, h - num_objects + 1))
    legend_rows = np.arange(legend_start, legend_start + num_objects)

    legend_colors = [int(color) for color in RNG.choice(palette, size=num_objects, replace=False)]
    object_colors = [int(color) for color in RNG.choice(palette, size=num_objects, replace=True)]
    areas = [int(area) for area in RNG.choice(AREA_CHOICES, size=num_objects, replace=False)]
    RNG.shuffle(areas)

    grid = np.full((h, w), bg, dtype=int)
    grid[legend_rows, 0] = legend_colors

    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :2] = True
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, -1] = True

    for area, color in zip(areas, object_colors):
        shape = _random_shape(area)
        sh, sw = map(int, shape.shape)
        if sh >= h - 2 or sw >= w - 3:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(RNG.integers(1, h - sh))
            c0 = int(RNG.integers(2, w - sw))
            if not _can_place(blocked, shape, r0, c0):
                continue
            for dr, dc in np.argwhere(shape):
                grid[r0 + int(dr), c0 + int(dc)] = color
            _mark_blocked(blocked, shape, r0, c0)
            placed = True
            break
        if not placed:
            return None

    if _background_color(grid) != bg:
        return None

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[legend_rows, 0] = True
    if not np.any((grid != out) & ~legend_mask):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
