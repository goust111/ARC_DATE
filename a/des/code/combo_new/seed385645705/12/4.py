from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BLUE = 1
BLACK = 0
SEED_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _layer_distances(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=int)
    q = deque()

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if any(not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            dist[r, c] = 1
            q.append((r, c))

    assert q, "each region must have a boundary layer"

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = nd
                q.append((nr, nc))

    return dist


def _infer_wall_and_bg(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    wall = Counter(int(v) for v in border).most_common(1)[0][0]
    non_wall = [int(v) for v in g.ravel() if int(v) != wall]
    assert non_wall, "grid must contain at least one non-wall cell"
    bg = Counter(non_wall).most_common(1)[0][0]
    return wall, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region from its boundary inward until the seed's erosion layer.

    Input structure:
    - Blue divider lines split the grid into several closed interior regions.
    - Each region is mostly black and contains exactly one colored seed pixel.
    - Region shapes may be irregular, but each region stays 4-connected.

    Rule:
    - Treat every non-blue 4-connected component as one region.
    - Inside each region, compute boundary layers by repeated 4-neighbor erosion:
      layer 1 touches the blue boundary, layer 2 touches layer 1 after one peel,
      and so on.
    - Let K be the layer index of the region's seed pixel. Recolor every region
      cell in layers 1..K to the seed color, keep deeper cells black, and leave
      blue divider pixels unchanged.

    Key insight:
    - The seed does not spread outward by itself; instead, its boundary-distance
      layer decides how many inward contour layers of the enclosing region get
      painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall, bg = _infer_wall_and_bg(g)
    out = g.copy()

    for comp in _components(g != wall):
        mask = np.zeros_like(g, dtype=bool)
        rows, cols = zip(*comp)
        mask[rows, cols] = True

        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) != bg]
        assert len(seed_cells) == 1, "each region must contain exactly one seed"

        sr, sc = seed_cells[0]
        seed_color = int(g[sr, sc])
        dist = _layer_distances(mask)
        max_layer = int(dist[sr, sc])
        out[mask & (dist <= max_layer)] = seed_color

    return out


def _has_hole(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    h, w = padded.shape
    seen = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~padded) & (~seen)))


def _pick_partition_seeds(ch: int, cw: int, count: int, rng: random.Random):
    cells = [(r, c) for r in range(ch) for c in range(cw)]
    for min_sep in (2, 1):
        pool = cells[:]
        rng.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        for cell in pool:
            if all(abs(cell[0] - r) + abs(cell[1] - c) >= min_sep for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _grow_partition(ch: int, cw: int, count: int, rng: random.Random):
    seeds = _pick_partition_seeds(ch, cw, count, rng)
    if seeds is None:
        return None

    labels = np.full((ch, cw), -1, dtype=int)
    areas = [0] * count
    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label
        areas[label] = 1

    while np.any(labels < 0):
        candidates = []
        for r in range(ch):
            for c in range(cw):
                if labels[r, c] != -1:
                    continue
                nbr_labels = {
                    int(labels[nr, nc])
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < ch and 0 <= nc < cw and labels[nr, nc] != -1
                }
                if nbr_labels:
                    candidates.append((r, c, tuple(nbr_labels)))

        if not candidates:
            return None

        r, c, nbr_labels = candidates[rng.randrange(len(candidates))]
        if rng.random() < 0.65:
            min_area = min(areas[label] for label in nbr_labels)
            options = [label for label in nbr_labels if areas[label] == min_area]
        else:
            options = list(nbr_labels)

        label = options[rng.randrange(len(options))]
        labels[r, c] = label
        areas[label] += 1

    if any(_has_hole(labels == label) for label in range(count)):
        return None

    return labels


def _build_grid_from_labels(labels: np.ndarray, scale: int) -> np.ndarray:
    ch, cw = labels.shape
    h = ch * (scale + 1) + 1
    w = cw * (scale + 1) + 1
    grid = np.full((h, w), BLUE, dtype=int)

    for r in range(ch):
        for c in range(cw):
            r0 = 1 + r * (scale + 1)
            c0 = 1 + c * (scale + 1)
            grid[r0 : r0 + scale, c0 : c0 + scale] = BLACK

    for r in range(ch):
        for c in range(cw - 1):
            if labels[r, c] == labels[r, c + 1]:
                r0 = 1 + r * (scale + 1)
                c_wall = 1 + c * (scale + 1) + scale
                grid[r0 : r0 + scale, c_wall] = BLACK

    for r in range(ch - 1):
        for c in range(cw):
            if labels[r, c] == labels[r + 1, c]:
                r_wall = 1 + r * (scale + 1) + scale
                c0 = 1 + c * (scale + 1)
                grid[r_wall, c0 : c0 + scale] = BLACK

    return grid


def _pick_seed_cell(mask: np.ndarray, dist: np.ndarray, target_depth: int, rng: random.Random):
    depths = sorted({int(v) for v in dist[mask]})
    chosen_depth = min(depths, key=lambda d: (abs(d - target_depth), d > target_depth))
    coords = np.argwhere(mask & (dist == chosen_depth))
    sr, sc = coords[rng.randrange(len(coords))]
    return int(sr), int(sc)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scale = 3 if rng.random() < 0.55 else 2
    max_cells = 7 if scale == 3 else 8
    ch = rng.randint(4, max_cells)
    cw = rng.randint(4, max_cells)
    region_count = rng.randint(3, min(6, ch * cw))

    labels = _grow_partition(ch, cw, region_count, rng)
    if labels is None:
        return None

    grid = _build_grid_from_labels(labels, scale)
    regions = _components(grid != BLUE)
    if len(regions) != region_count or len(regions) > len(SEED_COLORS):
        return None

    infos = []
    for comp in regions:
        if len(comp) < 4:
            return None
        mask = np.zeros_like(grid, dtype=bool)
        rows, cols = zip(*comp)
        mask[rows, cols] = True
        dist = _layer_distances(mask)
        max_depth = int(dist[mask].max())
        infos.append((mask, dist, max_depth))

    partial_candidates = [i for i, (_mask, _dist, max_depth) in enumerate(infos) if max_depth >= 2]
    if not partial_candidates:
        return None

    colors = rng.sample(SEED_COLORS, len(infos))
    showcase = partial_candidates[rng.randrange(len(partial_candidates))]
    grid = grid.copy()

    for i, (mask, dist, max_depth) in enumerate(infos):
        if max_depth == 1:
            target_depth = 1
        elif i == showcase:
            target_depth = rng.randint(1, max_depth - 1)
        else:
            roll = rng.random()
            if roll < 0.34:
                target_depth = 1
            elif roll < 0.67:
                target_depth = max_depth
            else:
                target_depth = rng.randint(1, max_depth)

        sr, sc = _pick_seed_cell(mask, dist, target_depth, rng)
        grid[sr, sc] = colors[i]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
