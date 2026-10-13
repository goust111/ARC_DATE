from __future__ import annotations

import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
LEGEND_LEN = 4

# Generator bounds.
GRID_H_RANGE = (13, 24)
GRID_W_RANGE = (13, 24)
LOCAL_H_RANGE = (8, 13)
LOCAL_W_RANGE = (8, 14)
SHAPE_COLOR_RANGE = (3, 5)
GROWTH_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = _neighbors4(0, 0) if connectivity == 4 else _neighbors8(0, 0)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _legend_and_shape(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = _components(mask, connectivity=8)
    assert len(comps) == 2, "expected one large foreground shape and one legend"

    legend_candidates = []
    for i, coords in enumerate(comps):
        if len(coords) != LEGEND_LEN:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1 and c1 - c0 + 1 == LEGEND_LEN:
            if len({(int(r), int(c)) for r, c in coords}) == LEGEND_LEN:
                legend_candidates.append(i)

    assert len(legend_candidates) == 1, "expected a unique 1x4 legend strip"
    legend_i = legend_candidates[0]
    shape_i = 1 - legend_i

    shape = np.zeros_like(mask, dtype=bool)
    shape[comps[shape_i][:, 0], comps[shape_i][:, 1]] = True
    legend = np.zeros_like(mask, dtype=bool)
    legend[comps[legend_i][:, 0], comps[legend_i][:, 1]] = True
    assert int(shape.sum()) > LEGEND_LEN
    return legend, shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the large foreground region by its number of non-background 4-neighbors.

    The input contains:
    - black background,
    - one large multicolor 8-connected foreground shape,
    - one separate 1x4 legend strip whose four colors map counts 0, 1, 2, 3-4.

    The output keeps the legend and the background unchanged, and recolors every
    shape cell according to its local 4-neighbor count within the shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_mask, shape_mask = _legend_and_shape(g != BG)
    legend_coords = np.argwhere(legend_mask)
    r0, c0, r1, c1 = _bbox(legend_coords)
    assert r0 == r1
    legend_colors = [int(x) for x in g[r0, c0 : c1 + 1]]
    assert len(legend_colors) == LEGEND_LEN

    out = g.copy()
    h, w = g.shape
    for r, c in np.argwhere(shape_mask):
        degree = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and shape_mask[nr, nc]:
                degree += 1
        out[r, c] = legend_colors[min(degree, 3)]
    return out


def _anchor_shape(local_h: int, local_w: int) -> np.ndarray:
    mask = np.zeros((local_h, local_w), dtype=bool)
    mask[1, 0] = True
    mask[2, 1] = True
    mask[2:5, 2:5] = True
    return mask


def _random_growth_mask(local_h: int, local_w: int) -> np.ndarray | None:
    base = _anchor_shape(local_h, local_w)
    mask = base.copy()
    allowed = np.zeros_like(mask, dtype=bool)
    allowed[3:, 3:] = True
    allowed &= ~mask
    reachable = int(base.sum() + allowed.sum())
    target_min = max(int(base.sum()) + 8, int(np.ceil(local_h * local_w * 0.30)))
    target_max = min(reachable, int(np.floor(local_h * local_w * 0.58)))
    if target_min > target_max:
        return None
    target_area = random.randint(target_min, target_max)

    frontier: set[tuple[int, int]] = set()

    def refresh(cell: tuple[int, int]) -> None:
        rr, cc = cell
        for nr, nc in _neighbors8(rr, cc):
            if 0 <= nr < local_h and 0 <= nc < local_w and allowed[nr, nc]:
                frontier.add((nr, nc))

    for r, c in np.argwhere(mask):
        refresh((int(r), int(c)))

    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        options = list(frontier)
        weights = []
        for rr, cc in options:
            n8 = sum(
                0 <= nr < local_h and 0 <= nc < local_w and mask[nr, nc]
                for nr, nc in _neighbors8(rr, cc)
            )
            weights.append({1: 5, 2: 4, 3: 3, 4: 2}.get(int(n8), 1))
        rr, cc = random.choices(options, weights=weights, k=1)[0]
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        allowed[rr, cc] = False
        refresh((rr, cc))

    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _neighbor_histogram(mask: np.ndarray) -> dict[int, int]:
    hist = {i: 0 for i in range(5)}
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        degree = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        hist[degree] += 1
    return hist


def _valid_shape(mask: np.ndarray) -> bool:
    if len(_components(mask, connectivity=8)) != 1:
        return False
    if mask.shape[0] < 5 or mask.shape[1] < 5:
        return False
    if not np.any(mask[0]) or not np.any(mask[-1]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
        return False

    hist = _neighbor_histogram(mask)
    return hist[0] > 0 and hist[1] > 0 and hist[2] > 0 and (hist[3] > 0 or hist[4] > 0)


def _make_shape_mask() -> np.ndarray:
    for _ in range(GROWTH_ATTEMPTS):
        local_h = random.randint(*LOCAL_H_RANGE)
        local_w = random.randint(*LOCAL_W_RANGE)
        if local_h < 5 or local_w < 5:
            continue
        mask = _random_growth_mask(local_h, local_w)
        if mask is None:
            continue

        mask = _trim(mask)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if _valid_shape(mask):
            return mask
    raise AssertionError("failed to build a valid connected shape")


def _seed_cells(cells: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    shuffled = cells[:]
    random.shuffle(shuffled)
    seeds: list[tuple[int, int]] = []
    for cell in shuffled:
        if all(abs(cell[0] - sr) + abs(cell[1] - sc) >= 3 for sr, sc in seeds):
            seeds.append(cell)
            if len(seeds) == k:
                return seeds
    return random.sample(cells, k)


def _paint_shape(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    k = random.randint(*SHAPE_COLOR_RANGE)
    palette = random.sample(colors, k)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    seeds = _seed_cells(cells, k)

    patch = np.full(mask.shape, BG, dtype=int)
    for r, c in cells:
        best_i = min(
            range(k),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        patch[r, c] = int(palette[best_i])

    assert len({int(v) for v in patch[mask]}) == k
    return patch


def _legend_slots(shape_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape_mask.shape
    slots: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w - LEGEND_LEN + 1):
            if np.any(shape_mask[r, c : c + LEGEND_LEN]):
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + LEGEND_LEN)
            if np.any(shape_mask[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            slots.append((r, c))
    return slots


def _build_sample() -> np.ndarray | None:
    shape_mask = _make_shape_mask()
    shape_h, shape_w = shape_mask.shape

    grid_h_lo = max(GRID_H_RANGE[0], shape_h + 3)
    grid_h_hi = max(grid_h_lo, GRID_H_RANGE[1])
    grid_w_lo = max(GRID_W_RANGE[0], shape_w + 3, LEGEND_LEN + 2)
    grid_w_hi = max(grid_w_lo, GRID_W_RANGE[1])
    if grid_h_lo > grid_h_hi or grid_w_lo > grid_w_hi:
        return None

    grid_h = random.randint(grid_h_lo, grid_h_hi)
    grid_w = random.randint(grid_w_lo, grid_w_hi)
    top = random.randint(0, grid_h - shape_h)
    left = random.randint(0, grid_w - shape_w)

    occupancy = np.zeros((grid_h, grid_w), dtype=bool)
    occupancy[top : top + shape_h, left : left + shape_w] = shape_mask
    slots = _legend_slots(occupancy)
    if not slots:
        return None

    legend_colors = random.sample(list(ALL_COLORS), LEGEND_LEN)
    shape_pool = [color for color in ALL_COLORS if color not in legend_colors]
    shape_patch = _paint_shape(shape_mask, shape_pool)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    target = grid[top : top + shape_h, left : left + shape_w]
    target[shape_mask] = shape_patch[shape_mask]

    lr, lc = random.choice(slots)
    grid[lr, lc : lc + LEGEND_LEN] = np.array(legend_colors, dtype=int)

    legend_mask, detected_shape = _legend_and_shape(grid != BG)
    assert int(legend_mask.sum()) == LEGEND_LEN
    assert np.array_equal(detected_shape, occupancy)
    assert len({int(v) for v in grid[occupancy]}) >= 3
    assert _valid_shape(shape_mask)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(PLACEMENT_ATTEMPTS):
        grid = _build_sample()
        if grid is None:
            continue
        _ = main(grid)
        return grid
    raise AssertionError("failed to generate a valid sample")
