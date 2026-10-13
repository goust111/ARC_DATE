import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_MIN = 15
GRID_MAX = 30
GROUP_COUNT_RANGE = (2, 5)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
SIZE_CAP_BY_COUNT = {
    2: 14,
    3: 12,
    4: 10,
    5: 9,
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _edge_distance(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    row_dist = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    col_dist = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    return np.minimum(row_dist, col_dist)


def _paint_group(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    dist = _edge_distance(shape)
    out = np.full(shape, BG, dtype=int)
    for layer, color in enumerate(colors):
        out[dist == layer] = int(color)
    return out


def _extract_layers(subgrid: np.ndarray) -> tuple[list[int], np.ndarray]:
    dist = _edge_distance(subgrid.shape)
    colors: list[int] = []
    for layer in range(int(dist.max()) + 1):
        vals = np.unique(subgrid[dist == layer])
        if vals.size == 1 and int(vals[0]) == BG:
            break
        assert vals.size == 1 and int(vals[0]) != BG, "each frame layer must be monochrome"
        colors.append(int(vals[0]))

    assert 2 <= len(colors) <= 4, "each group must have 2 to 4 layers"
    assert len(colors) == len(set(colors)), "colors inside one group must be distinct"
    assert bool(np.any(dist >= len(colors))), "the innermost frame must enclose black background"

    expected = _paint_group(subgrid.shape, colors)
    assert np.array_equal(subgrid, expected), "foreground component must be a nested rectangular frame group"
    return colors, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color order of each nested rectangular frame group in place.

    Input structure:
    - The background is black.
    - The grid contains several separated 4-connected groups of concentric one-cell-thick rectangular frames.
    - Each group has 2 to 4 layers with distinct colors, and the innermost frame encloses black background.

    Rule:
    - For each group, read the frame colors from the outermost layer to the innermost layer.
    - Reverse that color sequence.
    - Repaint the same frame layers with the reversed sequence, leaving all geometry, positions, and black holes unchanged.

    Key insight:
    - Each connected object is a rectangular border band whose layer index is the cell's distance to the bounding-box edge, so solving the task means identifying the full layer stack and then flipping that visible sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors, dist = _extract_layers(sub)
        recolored = np.full(sub.shape, BG, dtype=int)
        for layer, color in enumerate(reversed(colors)):
            recolored[dist == layer] = int(color)
        out[r0 : r1 + 1, c0 : c1 + 1] = recolored

    return out


# --------------- generator ---------------

def _group_count_bounds(h: int, w: int) -> tuple[int, int]:
    min_side = min(h, w)
    if min_side < 18:
        return 2, 3
    if min_side < 22:
        return 2, 4
    return GROUP_COUNT_RANGE


def _sample_group_spec(rng: random.Random, group_count: int, h: int, w: int) -> tuple[int, int, list[int]]:
    layers = rng.randint(2, 4)
    size_cap = min(SIZE_CAP_BY_COUNT[group_count], h, w)
    min_size = 2 * layers + 1
    assert min_size <= size_cap
    outer_h = rng.randint(min_size, size_cap)
    outer_w = rng.randint(min_size, size_cap)
    colors = rng.sample(PALETTE, layers)
    return outer_h, outer_w, colors


def _reserve(guard: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(guard.shape[0], r + h + 1)
    cc1 = min(guard.shape[1], c + w + 1)
    guard[rr0:rr1, cc0:cc1] = True


def _can_place(guard: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(guard.shape[0], r + h + 1)
    cc1 = min(guard.shape[1], c + w + 1)
    return not bool(np.any(guard[rr0:rr1, cc0:cc1]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    lo, hi = _group_count_bounds(h, w)
    group_count = rng.randint(lo, hi)

    specs = [_sample_group_spec(rng, group_count, h, w) for _ in range(group_count)]
    specs.sort(key=lambda spec: spec[0] * spec[1], reverse=True)

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    guard = np.zeros((h, w), dtype=bool)

    for outer_h, outer_w, colors in specs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = rng.randint(0, h - outer_h)
            c = rng.randint(0, w - outer_w)
            if not _can_place(guard, r, c, outer_h, outer_w):
                continue
            inp_group = _paint_group((outer_h, outer_w), colors)
            out_group = _paint_group((outer_h, outer_w), list(reversed(colors)))
            grid[r : r + outer_h, c : c + outer_w] = inp_group
            expected[r : r + outer_h, c : c + outer_w] = out_group
            _reserve(guard, r, c, outer_h, outer_w)
            placed = True
            break
        if not placed:
            return None

    if len(_components(grid != BG)) != group_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
