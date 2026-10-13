from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
MAX_GEN_ATTEMPTS = 400
GRID_MIN = 11
GRID_MAX = 24
SEED_GAP = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_source(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[list[int], list[tuple[int, int]]]] = []

    for coords in _find_components(g != BLACK):
        if len(coords) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 3 or side_h % 2 == 0:
            continue
        if not (r0 == 0 or c0 == 0):
            continue
        if len(coords) != side_h * side_w:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BLACK):
            continue

        center = side_h // 2
        layers: list[int] = []
        ok = True
        for dist in range(center + 1):
            shell_vals = []
            for rr in range(side_h):
                for cc in range(side_w):
                    if max(abs(rr - center), abs(cc - center)) == dist:
                        shell_vals.append(int(patch[rr, cc]))
            if len(set(shell_vals)) != 1:
                ok = False
                break
            layers.append(shell_vals[0])
        if not ok or len(set(layers)) != len(layers):
            continue

        candidates.append((layers, coords))

    assert len(candidates) == 1, "expected exactly one propagation source"
    layers, coords = candidates[0]
    source_mask = np.zeros_like(g, dtype=bool)
    for r, c in coords:
        source_mask[r, c] = True
    return layers, source_mask


def _seed_distances(black_mask: np.ndarray, seed: tuple[int, int], max_dist: int) -> np.ndarray:
    h, w = black_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = seed
    dist[sr, sc] = 0
    q = deque([seed])

    while q:
        r, c = q.popleft()
        if dist[r, c] >= max_dist:
            continue
        for nr, nc in _neighbors4(r, c, h, w):
            if dist[nr, nc] != -1:
                continue
            if not black_mask[nr, nc]:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers, source_mask = _extract_source(g)
    center_color = layers[0]
    seed_coords = [tuple(map(int, coord)) for coord in np.argwhere((g == center_color) & ~source_mask)]
    assert seed_coords, "expected at least one seed pixel matching the source center color"

    out = g.copy()
    max_dist = len(layers) - 1
    if max_dist <= 0:
        return out

    black_mask = g == BLACK
    dist_maps = np.stack([_seed_distances(black_mask, seed, max_dist) for seed in seed_coords], axis=0)
    inf = max_dist + 10
    masked = np.where(dist_maps >= 0, dist_maps, inf)
    best = masked.min(axis=0)
    tie_count = (masked == best[None, :, :]).sum(axis=0)
    unique = (best <= max_dist) & (tie_count == 1)

    for dist in range(1, max_dist + 1):
        mask = black_mask & unique & (best == dist)
        out[mask] = layers[dist]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a concentric source sprite on the top or left edge, then expand every
    matching seed pixel outward layer by layer using the sprite's center-to-edge
    color order. Equal-distance collisions stay black.
    """
    return _solve(grid)


def _build_source_patch(layers: list[int]) -> np.ndarray:
    side = 2 * len(layers) - 1
    center = side // 2
    patch = np.zeros((side, side), dtype=int)
    for r in range(side):
        for c in range(side):
            patch[r, c] = layers[max(abs(r - center), abs(c - center))]
    return patch


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _seed_ok(pos: tuple[int, int], seeds: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    r, c = pos
    if forbidden[r, c]:
        return False
    for sr, sc in seeds:
        if max(abs(sr - r), abs(sc - c)) <= SEED_GAP:
            return False
    return True


def _place_symmetric_pair(
    rng: random.Random,
    forbidden: np.ndarray,
    layers: int,
    seeds: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = forbidden.shape
    max_mid_dist = max(1, layers - 1)
    for _ in range(120):
        horizontal = rng.random() < 0.5
        mid_dist = rng.randint(1, max_mid_dist)
        if horizontal:
            row = rng.randint(1, h - 2)
            center_col = rng.randint(1 + mid_dist, w - 2 - mid_dist)
            pair = [(row, center_col - mid_dist), (row, center_col + mid_dist)]
        else:
            col = rng.randint(1, w - 2)
            center_row = rng.randint(1 + mid_dist, h - 2 - mid_dist)
            pair = [(center_row - mid_dist, col), (center_row + mid_dist, col)]
        if all(_seed_ok(pos, seeds, forbidden) for pos in pair):
            return pair
    return None


def _random_seed(
    rng: random.Random,
    forbidden: np.ndarray,
    seeds: list[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = forbidden.shape
    for _ in range(200):
        pos = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        if _seed_ok(pos, seeds, forbidden):
            return pos
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(2, 4)
    source_colors = rng.sample(list(range(1, 10)), layer_count)
    source_patch = _build_source_patch(source_colors)
    side = source_patch.shape[0]

    min_h = max(GRID_MIN, side + 4)
    min_w = max(GRID_MIN, side + 4)
    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    orient = rng.choice(["top", "left"])
    if orient == "top":
        r0 = 0
        c0 = rng.randint(0, w - side)
    else:
        r0 = rng.randint(0, h - side)
        c0 = 0

    grid[r0 : r0 + side, c0 : c0 + side] = source_patch
    source_bbox = (r0, c0, r0 + side - 1, c0 + side - 1)

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[0, :] = True
    forbidden[h - 1, :] = True
    forbidden[:, 0] = True
    forbidden[:, w - 1] = True
    _reserve_bbox(forbidden, source_bbox, layer_count + 1)

    scenario = rng.choices(
        ["scatter", "pair", "pair_plus"],
        weights=[3, 2, 3],
        k=1,
    )[0]
    seeds: list[tuple[int, int]] = []

    if scenario in {"pair", "pair_plus"}:
        pair = _place_symmetric_pair(rng, forbidden, layer_count, seeds)
        if pair is None:
            return None
        seeds.extend(pair)

    max_seeds = min(6, ((h - 2) * (w - 2)) // 18 + 2)
    min_seeds = 1 if scenario == "scatter" else 2
    target_seeds = rng.randint(min_seeds, max_seeds)

    while len(seeds) < target_seeds:
        pos = _random_seed(rng, forbidden, seeds)
        if pos is None:
            break
        seeds.append(pos)

    if len(seeds) < min_seeds:
        return None

    for r, c in seeds:
        grid[r, c] = source_colors[0]

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None

    if scenario in {"pair", "pair_plus"}:
        changed = int(np.count_nonzero(out != grid))
        if changed < len(seeds) + 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
