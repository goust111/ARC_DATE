from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(10))
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _path_color_and_seeds(grid: np.ndarray) -> tuple[int, list[int]]:
    bg = _get_bg(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected foreground colors"
    path_color = max(colors, key=lambda c: int(np.count_nonzero(grid == c)))
    seed_colors = [c for c in colors if c != path_color]
    assert all(int(np.count_nonzero(grid == c)) == 1 for c in seed_colors), "seeds must be single pixels"
    return int(path_color), sorted(seed_colors)


def _find_hub(mask: np.ndarray) -> tuple[int, int, int]:
    h, w = mask.shape
    max_k = min(h, w)
    if max_k % 2 == 0:
        max_k -= 1
    for k in range(max_k, 2, -2):
        hits = []
        for r0 in range(h - k + 1):
            for c0 in range(w - k + 1):
                if bool(mask[r0 : r0 + k, c0 : c0 + k].all()):
                    hits.append((r0, c0))
        if hits:
            assert len(hits) == 1, "hub square must be unique"
            r0, c0 = hits[0]
            return int(k), int(r0), int(c0)
    raise AssertionError("failed to find odd solid hub square")


def _hub_boundary(k: int, r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    r1 = r0 + k - 1
    c1 = c0 + k - 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.add((r, c))
    return cells


def _seed_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.shape[0] == 1
    return int(pts[0, 0]), int(pts[0, 1])


def _geodesic_distance(
    grid: np.ndarray,
    path_color: int,
    seed_color: int,
    target_cells: set[tuple[int, int]],
) -> int:
    start = _seed_coord(grid, seed_color)
    allowed = grid == int(path_color)
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    h, w = grid.shape
    while q:
        r, c, d = q.popleft()
        if (r, c) in target_cells:
            return int(d)
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) == int(seed_color) or bool(allowed[nr, nc]):
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    raise AssertionError("seed cannot reach hub boundary through the network")


def _radial_distance(seed: tuple[int, int], boundary: set[tuple[int, int]]) -> float:
    sr, sc = seed
    return min(((sr - r) ** 2 + (sc - c) ** 2) ** 0.5 for r, c in boundary)


def _ring_index(r: int, c: int, k: int, r0: int, c0: int) -> int:
    r1 = r0 + k - 1
    c1 = c0 + k - 1
    return int(min(r - r0, c - c0, r1 - r, c1 - c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the seeds by geodesic distance through the network, then paint hub rings from outside to center in that order.

    Input structure:
    - The background contains a 1-cell-wide orthogonal path network whose center is an odd k×k solid hub square.
    - Each branch ends at a single-pixel seed of a distinct color, and the number of seeds equals the number of concentric hub layers.
    - Outside the hub, the network may bend, so visual nearness is not the same as path distance.

    Rule:
    - Infer the network color and the unique odd solid hub square.
    - For each seed, compute the shortest 4-connected distance from that seed to any hub-edge cell while traveling only through the network.
    - Recolor the hub so the nearest seed paints the outer ring, the next seed paints the next inner ring, and the farthest seed paints the center; leave the paths and seeds unchanged.

    Key insight:
    - The ranking key is geodesic distance along the visible network, not straight-line proximity to the hub.
    """
    g = np.asarray(grid, dtype=int)
    path_color, seed_colors = _path_color_and_seeds(g)
    k, r0, c0 = _find_hub(g == int(path_color))
    layer_count = (k + 1) // 2
    assert len(seed_colors) == layer_count, "seed count must match ring count"

    boundary = _hub_boundary(k, r0, c0)
    ranked = []
    for color in seed_colors:
        dist = _geodesic_distance(g, path_color, color, boundary)
        ranked.append((dist, color))
    dvals = [d for d, _c in ranked]
    assert len(set(dvals)) == len(dvals), "seed distances must be unique"
    ranked.sort()

    out = g.copy()
    for r in range(r0, r0 + k):
        for c in range(c0, c0 + k):
            out[r, c] = ranked[_ring_index(r, c, k, r0, c0)][1]
    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1, "segments must be orthogonal"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _pick_other(rng: random.Random, values: list[int], current: int) -> int:
    choices = [v for v in values if v != current]
    assert choices
    return int(rng.choice(choices))


def _build_side_branch(
    side: str,
    rng: random.Random,
    k: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    radial: int,
    swings: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r1 = r0 + k - 1
    c1 = c0 + k - 1
    inner_cols = list(range(c0 + 1, c1))
    inner_rows = list(range(r0 + 1, r1))
    assert inner_cols and inner_rows

    if side == "top":
        start = (r0 - 1, int(rng.choice(inner_cols)))
        seed = (r0 - radial, int(rng.choice(inner_cols)))
        leaf = (seed[0] + 1, seed[1])
        free_rows = list(range(seed[0] + 2, start[0]))
        checkpoints = sorted(rng.sample(free_rows, swings), reverse=True) if swings else []
        points = [start]
        cur = start[1]
        extremes = [inner_cols[0], inner_cols[-1]]
        for row in checkpoints:
            points.append((row, cur))
            cur = _pick_other(rng, extremes + [seed[1]], cur)
            points.append((row, cur))
        points.append((leaf[0], cur))
        if cur != seed[1]:
            points.append((leaf[0], seed[1]))
        return _polyline(points), seed

    if side == "bottom":
        start = (r1 + 1, int(rng.choice(inner_cols)))
        seed = (r1 + radial, int(rng.choice(inner_cols)))
        leaf = (seed[0] - 1, seed[1])
        free_rows = list(range(start[0] + 1, leaf[0]))
        checkpoints = sorted(rng.sample(free_rows, swings)) if swings else []
        points = [start]
        cur = start[1]
        extremes = [inner_cols[0], inner_cols[-1]]
        for row in checkpoints:
            points.append((row, cur))
            cur = _pick_other(rng, extremes + [seed[1]], cur)
            points.append((row, cur))
        points.append((leaf[0], cur))
        if cur != seed[1]:
            points.append((leaf[0], seed[1]))
        return _polyline(points), seed

    if side == "left":
        start = (int(rng.choice(inner_rows)), c0 - 1)
        seed = (int(rng.choice(inner_rows)), c0 - radial)
        leaf = (seed[0], seed[1] + 1)
        free_cols = list(range(seed[1] + 2, start[1]))
        checkpoints = sorted(rng.sample(free_cols, swings), reverse=True) if swings else []
        points = [start]
        cur = start[0]
        extremes = [inner_rows[0], inner_rows[-1]]
        for col in checkpoints:
            points.append((cur, col))
            cur = _pick_other(rng, extremes + [seed[0]], cur)
            points.append((cur, col))
        points.append((cur, leaf[1]))
        if cur != seed[0]:
            points.append((seed[0], leaf[1]))
        return _polyline(points), seed

    assert side == "right"
    start = (int(rng.choice(inner_rows)), c1 + 1)
    seed = (int(rng.choice(inner_rows)), c1 + radial)
    leaf = (seed[0], seed[1] - 1)
    free_cols = list(range(start[1] + 1, leaf[1]))
    checkpoints = sorted(rng.sample(free_cols, swings)) if swings else []
    points = [start]
    cur = start[0]
    extremes = [inner_rows[0], inner_rows[-1]]
    for col in checkpoints:
        points.append((cur, col))
        cur = _pick_other(rng, extremes + [seed[0]], cur)
        points.append((cur, col))
    points.append((cur, leaf[1]))
    if cur != seed[0]:
        points.append((seed[0], leaf[1]))
    return _polyline(points), seed


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _validate_structure(grid: np.ndarray) -> bool:
    try:
        path_color, seed_colors = _path_color_and_seeds(grid)
        path_mask = grid == int(path_color)
        k, r0, c0 = _find_hub(path_mask)
        layers = (k + 1) // 2
        if len(seed_colors) != layers or len(seed_colors) < 3:
            return False
        if not _connected(path_mask):
            return False

        hub_mask = np.zeros_like(path_mask, dtype=bool)
        hub_mask[r0 : r0 + k, c0 : c0 + k] = True
        outside = path_mask & ~hub_mask
        h, w = grid.shape
        for rr in range(h - 1):
            for cc in range(w - 1):
                block = outside[rr : rr + 2, cc : cc + 2]
                if bool(block.all()):
                    return False

        boundary = _hub_boundary(k, r0, c0)
        geodesic = []
        radial = []
        detour_seen = False
        for color in seed_colors:
            seed = _seed_coord(grid, color)
            neigh = [(nr, nc) for nr, nc in _neighbors4(*seed) if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == path_color]
            if len(neigh) != 1:
                return False
            dist = _geodesic_distance(grid, path_color, color, boundary)
            straight = _radial_distance(seed, boundary)
            geodesic.append((dist, color))
            radial.append((straight, color))
            if dist > straight + 3.0:
                detour_seen = True

        if not detour_seen:
            return False
        if len({d for d, _c in geodesic}) != len(geodesic):
            return False
        geo_order = [c for _d, c in sorted(geodesic)]
        rad_order = [c for _d, c in sorted(radial)]
        if geo_order == rad_order:
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if not np.array_equal(out[~hub_mask], grid[~hub_mask]):
            return False

        expected = [c for _d, c in sorted(geodesic)]
        seen_colors = [int(out[r0 + i, c0 + i]) for i in range(layers)]
        if seen_colors != expected:
            return False
        return True
    except Exception:
        return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    k = int(rng.choice([5, 7]))
    layers = (k + 1) // 2
    min_margin = 6 if k == 7 else 5
    h = int(rng.randint(max(19, k + 2 * min_margin + 1), 30))
    w = int(rng.randint(max(19, k + 2 * min_margin + 1), 30))

    r0_lo = max(min_margin, h // 2 - 2 - k // 2)
    r0_hi = min(h - k - min_margin, h // 2 + 2 - k // 2)
    c0_lo = max(min_margin, w // 2 - 2 - k // 2)
    c0_hi = min(w - k - min_margin, w // 2 + 2 - k // 2)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = int(rng.randint(r0_lo, r0_hi))
    c0 = int(rng.randint(c0_lo, c0_hi))
    r1 = r0 + k - 1
    c1 = c0 + k - 1

    color_pool = rng.sample(PALETTE, layers + 2)
    bg = int(color_pool[0])
    path_color = int(color_pool[1])
    seed_colors = [int(c) for c in color_pool[2:]]
    rng.shuffle(seed_colors)

    grid = np.full((h, w), bg, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = path_color

    all_sides = ["top", "bottom", "left", "right"]
    if layers == 4:
        sides = all_sides
    else:
        sides = rng.sample(all_sides, layers)

    trick_side = rng.choice(sides)
    direct_side = rng.choice([s for s in sides if s != trick_side])

    margins = {
        "top": r0,
        "bottom": h - 1 - r1,
        "left": c0,
        "right": w - 1 - c1,
    }

    for side, color in zip(sides, seed_colors):
        max_rad = margins[side]
        if max_rad < 5:
            return None
        if side == trick_side:
            lo = 5
            hi = min(max_rad, 7)
            if lo > hi:
                return None
            radial = int(rng.randint(lo, hi))
            max_swings = min(3, radial - 3)
            if max_swings < 2:
                return None
            swings = int(rng.randint(2, max_swings))
        elif side == direct_side:
            lo = max(6, max_rad - 2)
            hi = max_rad
            if lo > hi:
                return None
            radial = int(rng.randint(lo, hi))
            swings = int(rng.randint(0, 1))
        else:
            lo = 4
            hi = max_rad
            if lo > hi:
                return None
            radial = int(rng.randint(lo, hi))
            swings = int(rng.randint(0, min(2, radial - 3)))

        branch, seed = _build_side_branch(side, rng, k, r0, c0, h, w, radial, swings)
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in branch):
            return None
        if not (0 <= seed[0] < h and 0 <= seed[1] < w):
            return None
        if int(grid[seed[0], seed[1]]) != bg:
            return None
        if any(int(grid[rr, cc]) != bg for rr, cc in branch):
            return None
        if seed in branch:
            return None

        for rr, cc in branch:
            grid[rr, cc] = path_color
        grid[seed[0], seed[1]] = int(color)

    if not _validate_structure(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
