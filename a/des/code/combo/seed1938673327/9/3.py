from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BG_COLOR = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
CLUSTER_COUNT_CHOICES = (2, 3, 4, 5)
SEED_SIZE_RANGE = (1, 5)
SEPARATION_MARGIN = 1
EDGE_CLUSTER_MAX = 2


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append({"color": color, "coords": coords, "area": len(coords)})

    return comps


def _grow_components(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g, bg)

    owner = np.full((h, w), -1, dtype=int)
    colors: list[int] = []
    limits: list[int] = []
    frontiers: list[set[tuple[int, int]]] = []

    for cid, comp in enumerate(comps):
        colors.append(int(comp["color"]))
        limits.append(int(comp["area"]))
        frontier: set[tuple[int, int]] = set()
        for r, c in comp["coords"]:
            owner[r, c] = cid
            frontier.add((int(r), int(c)))
        frontiers.append(frontier)

    for step in range(1, max(limits, default=0) + 1):
        proposals: dict[tuple[int, int], list[int]] = {}
        for cid, frontier in enumerate(frontiers):
            if step > limits[cid]:
                continue
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] == -1 and int(g[nr, nc]) == bg:
                        proposals.setdefault((int(nr), int(nc)), []).append(cid)

        new_frontiers = [set() for _ in comps]
        for (r, c), proposers in proposals.items():
            if len(proposers) != 1:
                continue
            cid = int(proposers[0])
            owner[r, c] = cid
            new_frontiers[cid].add((int(r), int(c)))
        frontiers = new_frontiers

    out = np.full_like(g, bg)
    for cid, color in enumerate(colors):
        out[owner == cid] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each 4-connected seed cluster outward for a number of steps equal to its initial area.

    Each growth step expands the current frontier into 4-neighbor background cells only.
    All frontier cells for a cluster grow simultaneously, and collisions are left unresolved.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    return _grow_components(g, bg)


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_list = list(cells)
    min_r = min(r for r, _ in cell_list)
    min_c = min(c for _, c in cell_list)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cell_list)


def _transform_shape(cells: list[tuple[int, int]], mode: int) -> list[tuple[int, int]]:
    transformed = []
    for r, c in cells:
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, -c
        elif mode == 2:
            nr, nc = -r, c
        elif mode == 3:
            nr, nc = -r, -c
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, -r
        elif mode == 6:
            nr, nc = -c, r
        else:
            nr, nc = -c, -r
        transformed.append((int(nr), int(nc)))
    return _normalize(transformed)


def _random_connected_shape(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (int(r + dr), int(c + dc))
                if nxt not in cells:
                    frontier.append(nxt)
        frontier = sorted(set(frontier))
        if rng.random() < 0.5:
            frontier.sort(key=lambda rc: (abs(rc[0]) + abs(rc[1]), abs(rc[0]), abs(rc[1])))
            choice = rng.choice(frontier[: min(3, len(frontier))])
        else:
            choice = rng.choice(frontier)
        cells.add(choice)
    shape = _normalize(cells)
    return _transform_shape(shape, rng.randrange(8))


def _shape_bbox(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in shape]


def _min_pair_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _placement_ok(
    cells: list[tuple[int, int]],
    size: int,
    placed: list[dict],
    h: int,
    w: int,
) -> bool:
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    for comp in placed:
        required = int(size + comp["size"] + SEPARATION_MARGIN)
        if _min_pair_distance(cells, comp["cells"]) <= required:
            return False
    return True


def _sample_position(
    rng: random.Random,
    shape: list[tuple[int, int]],
    h: int,
    w: int,
    want_edge: bool,
) -> tuple[int, int]:
    sh, sw = _shape_bbox(shape)
    max_r0 = h - sh
    max_c0 = w - sw
    assert max_r0 >= 0 and max_c0 >= 0

    if want_edge:
        side = rng.choice(("top", "bottom", "left", "right", "corner"))
        if side == "top":
            return 0, rng.randint(0, max_c0)
        if side == "bottom":
            return max_r0, rng.randint(0, max_c0)
        if side == "left":
            return rng.randint(0, max_r0), 0
        if side == "right":
            return rng.randint(0, max_r0), max_c0
        vertical = rng.choice((0, max_r0))
        horizontal = rng.choice((0, max_c0))
        return vertical, horizontal

    inner_r0 = 1 if max_r0 >= 2 else 0
    inner_r1 = max_r0 - 1 if max_r0 >= 2 else max_r0
    inner_c0 = 1 if max_c0 >= 2 else 0
    inner_c1 = max_c0 - 1 if max_c0 >= 2 else max_c0
    r0 = rng.randint(inner_r0, inner_r1)
    c0 = rng.randint(inner_c0, inner_c1)
    return r0, c0


def _place_component(
    rng: random.Random,
    shape: list[tuple[int, int]],
    size: int,
    placed: list[dict],
    h: int,
    w: int,
    want_edge: bool,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        r0, c0 = _sample_position(rng, shape, h, w, want_edge)
        cells = _translate(shape, r0, c0)
        if _placement_ok(cells, size, placed, h, w):
            return cells
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    max_clusters = min(CLUSTER_COUNT_CHOICES[-1], max(2, (h * w) // 70 + 2))
    cluster_choices = [n for n in CLUSTER_COUNT_CHOICES if n <= max_clusters]
    n_clusters = rng.choice(cluster_choices)
    sizes = [rng.randint(*SEED_SIZE_RANGE) for _ in range(n_clusters)]
    colors = rng.sample(PALETTE, n_clusters)

    edge_target = rng.randint(0, min(EDGE_CLUSTER_MAX, n_clusters))
    edge_ids = set(rng.sample(range(n_clusters), edge_target))

    placed: list[dict] = []
    for idx in sorted(range(n_clusters), key=lambda i: sizes[i], reverse=True):
        shape = _random_connected_shape(rng, sizes[idx])
        cells = _place_component(
            rng=rng,
            shape=shape,
            size=sizes[idx],
            placed=placed,
            h=h,
            w=w,
            want_edge=idx in edge_ids,
        )
        if cells is None:
            return None
        placed.append(
            {
                "size": sizes[idx],
                "color": colors[idx],
                "cells": cells,
                "touches_edge": _touches_edge(cells, h, w),
            }
        )

    if edge_target > 0 and not any(comp["touches_edge"] for comp in placed):
        return None
    if _get_bg(grid) != BG_COLOR:
        return None

    for comp in placed:
        for r, c in comp["cells"]:
            grid[r, c] = int(comp["color"])

    if _get_bg(grid) != BG_COLOR:
        return None

    comps = _find_components(grid, BG_COLOR)
    if len(comps) != n_clusters:
        return None
    if sorted(int(comp["area"]) for comp in comps) != sorted(sizes):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if _get_bg(out) != BG_COLOR:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
