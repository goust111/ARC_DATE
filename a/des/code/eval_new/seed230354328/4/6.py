from __future__ import annotations

import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1 or any(
                not mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)
            ):
                dist[r, c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    return dist


def _single_pixel_seeds(grid: np.ndarray, bg: int) -> np.ndarray:
    seed_mask = np.zeros_like(grid, dtype=bool)
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if len(comp) == 1:
                r, c = comp[0]
                seed_mask[r, c] = True
    return seed_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed loop's interior by expanding seed colors outward with depth-based priority.

    Input structure:
    - The most common color is the background.
    - Several 1-pixel-thick closed loop outlines of different colors surround interior regions.
    - Inside each loop are 1-3 isolated single-pixel seeds whose colors differ from the loop color and from one another.

    Rule:
    - Treat background cells and isolated seed pixels as passable, then find every 4-connected passable region that does not touch the grid boundary; each such region is one loop interior.
    - Within one interior, compute each seed's 4-neighbor distance to the interior boundary and sort seeds from farthest to nearest.
    - Run a multi-source 4-neighbor expansion from the seeds in that priority order; each background cell takes the color of the first wave that reaches it, while loop outlines and already-claimed cells stay unchanged.

    Key insight:
    - The loop interior behaves like a bounded flood-fill arena, and the hidden ordering cue is each seed's depth from the enclosing boundary: deeper seeds expand first and win tie cells.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)
    seed_mask = _single_pixel_seeds(g, bg)

    passable = (g == bg) | seed_mask
    out = g.copy()

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        region = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            region[r, c] = True

        dist = _boundary_distance(region)
        seeds = [
            (dist[r, c], r, c, int(g[r, c]))
            for r, c in comp
            if seed_mask[r, c]
        ]
        if not seeds:
            continue

        seeds.sort(key=lambda item: (-item[0], item[1], item[2]))
        owner = np.full((h, w), -1, dtype=int)
        q = deque()

        for _depth, r, c, color in seeds:
            owner[r, c] = color
            q.append((r, c, color))

        while q:
            r, c, color = q.popleft()
            for nr, nc in _neighbors4(r, c, h, w):
                if not region[nr, nc] or owner[nr, nc] != -1 or g[nr, nc] != bg:
                    continue
                owner[nr, nc] = color
                q.append((nr, nc, color))

        for r, c in comp:
            if g[r, c] == bg and owner[r, c] != -1:
                out[r, c] = owner[r, c]

    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1 or any(
                not mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)
            ):
                boundary[r, c] = True
    return boundary


def _random_filled_shape(rng: random.Random, box_h: int, box_w: int) -> np.ndarray:
    mask = np.zeros((box_h, box_w), dtype=bool)

    core_h = rng.randint(5, box_h - 2)
    core_w = rng.randint(5, box_w - 2)
    r0 = rng.randint(1, box_h - core_h - 1)
    c0 = rng.randint(1, box_w - core_w - 1)
    mask[r0 : r0 + core_h, c0 : c0 + core_w] = True

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    bump_count = rng.choice([0, 1, 1, 2, 2, 3])

    for side in sides[:bump_count]:
        if side == "top" and r0 > 0:
            depth = rng.randint(1, min(2, r0))
            width = rng.randint(max(2, core_w // 3), core_w)
            start = rng.randint(c0, c0 + core_w - width)
            mask[r0 - depth : r0, start : start + width] = True
        elif side == "bottom" and r0 + core_h < box_h:
            depth = rng.randint(1, min(2, box_h - (r0 + core_h)))
            width = rng.randint(max(2, core_w // 3), core_w)
            start = rng.randint(c0, c0 + core_w - width)
            mask[r0 + core_h : r0 + core_h + depth, start : start + width] = True
        elif side == "left" and c0 > 0:
            depth = rng.randint(1, min(2, c0))
            height = rng.randint(max(2, core_h // 3), core_h)
            start = rng.randint(r0, r0 + core_h - height)
            mask[start : start + height, c0 - depth : c0] = True
        elif side == "right" and c0 + core_w < box_w:
            depth = rng.randint(1, min(2, box_w - (c0 + core_w)))
            height = rng.randint(max(2, core_h // 3), core_h)
            start = rng.randint(r0, r0 + core_h - height)
            mask[start : start + height, c0 + core_w : c0 + core_w + depth] = True

    return _trim(mask)


def _pick_depth_levels(depth_levels: list[int], seed_count: int, rng: random.Random) -> list[int]:
    if seed_count == 1:
        top_half = depth_levels[len(depth_levels) // 2 :]
        return [rng.choice(top_half)]
    if seed_count == 2:
        return [depth_levels[-1], rng.choice(depth_levels[:-1])]
    mid = depth_levels[len(depth_levels) // 2]
    return [depth_levels[-1], mid, depth_levels[0]]


def _select_seed_cells(
    dist: np.ndarray, interior: np.ndarray, levels: list[int], rng: random.Random
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    for level in levels:
        candidates = [(int(r), int(c)) for r, c in np.argwhere((dist == level) & interior)]
        rng.shuffle(candidates)
        picked = None
        for r, c in candidates:
            if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
                picked = (r, c)
                break
        if picked is None and candidates:
            picked = candidates[0]
        if picked is None:
            return []
        chosen.append(picked)
    return chosen


def _build_object(
    rng: random.Random, ring_color: int, seed_colors: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    seed_count = len(seed_colors)
    for _ in range(80):
        box_h = rng.randint(7, 13)
        box_w = rng.randint(7, 13)
        filled = _random_filled_shape(rng, box_h, box_w)
        boundary = _shape_boundary(filled)
        interior = filled & ~boundary

        if len(_components(interior)) != 1:
            continue
        if int(interior.sum()) < 5 + 2 * seed_count:
            continue

        dist = _boundary_distance(interior)
        depth_levels = sorted({int(dist[r, c]) for r, c in np.argwhere(interior)})
        if len(depth_levels) < seed_count:
            continue

        levels = _pick_depth_levels(depth_levels, seed_count, rng)
        if len(set(levels)) != seed_count:
            continue
        seed_cells = _select_seed_cells(dist, interior, levels, rng)
        if len(seed_cells) != seed_count:
            continue

        local = np.full(filled.shape, -1, dtype=int)
        local[boundary] = ring_color
        for (r, c), color in zip(seed_cells, seed_colors):
            local[r, c] = color
        return local, filled
    return None


def _valid_seed_plan(n_objects: int, rng: random.Random) -> list[int]:
    for _ in range(50):
        counts = [rng.randint(1, 3) for _ in range(n_objects)]
        if n_objects + sum(counts) <= 9:
            return counts
    if n_objects == 3:
        return [2, 2, 2]
    if n_objects == 2:
        return [3, 3]
    return [3]


def _place_objects(
    rng: random.Random, grid: np.ndarray, objects: list[tuple[np.ndarray, np.ndarray]]
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    pad = 2

    order = list(range(len(objects)))
    rng.shuffle(order)
    placements: list[tuple[int, int, np.ndarray]] = []

    for idx in order:
        obj, _filled = objects[idx]
        oh, ow = obj.shape
        if oh + 2 * pad > h or ow + 2 * pad > w:
            return False

        placed = False
        for _ in range(120):
            r0 = rng.randint(pad, h - oh - pad)
            c0 = rng.randint(pad, w - ow - pad)
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + oh + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + ow + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            placements.append((r0, c0, obj))
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False

    for r0, c0, obj in placements:
        mask = obj >= 0
        sub = grid[r0 : r0 + obj.shape[0], c0 : c0 + obj.shape[1]]
        sub[mask] = obj[mask]

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    bg = rng.randint(0, 9)
    n_objects = rng.randint(1, 3)
    seed_plan = _valid_seed_plan(n_objects, rng)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    ring_colors = palette[:n_objects]
    cursor = n_objects
    objects: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_objects):
        seed_count = seed_plan[i]
        seed_colors = palette[cursor : cursor + seed_count]
        cursor += seed_count
        built = _build_object(rng, ring_colors[i], seed_colors)
        if built is None:
            return None
        obj, filled = built
        if int((filled & ~_shape_boundary(filled)).sum()) <= seed_count:
            return None
        objects.append((obj, filled))

    grid = np.full((h, w), bg, dtype=int)
    if not _place_objects(rng, grid, objects):
        return None

    vals, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(vals == bg)[0][0]])
    other_max = max((int(cnt) for val, cnt in zip(vals, counts) if int(val) != bg), default=0)
    if bg_count <= other_max:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
