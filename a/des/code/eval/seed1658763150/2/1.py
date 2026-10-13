import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PATTERN_LIBRARY = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=bool),
)

GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _edge_reachable(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _legend_patterns(grid: np.ndarray) -> dict[int, np.ndarray]:
    legend = np.asarray(grid[-3:], dtype=int)
    w = legend.shape[1]
    mapping: dict[int, np.ndarray] = {}
    c = 0
    while c <= w - 3:
        key = int(legend[0, c])
        if key not in (BLACK, GREY, MAROON):
            block = legend[:, c : c + 3]
            mask = block == MAROON
            # The keyed top-left cell is the anchored "growth" cell.
            mask[0, 0] = True
            mapping[key] = mask
            c += 3
        else:
            c += 1
    assert mapping, "legend must contain at least one keyed pattern"
    return mapping


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    upper = np.asarray(grid[:-3], dtype=int)
    passable = upper != GREY
    outside = _edge_reachable(passable)
    inside = passable & ~outside
    regions = []
    for comp in _components(inside):
        seeds = [(r, c, int(upper[r, c])) for r, c in comp if int(upper[r, c]) != BLACK]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        sr, sc, color = seeds[0]
        regions.append(
            {
                "cells": comp,
                "seed": (int(sr), int(sc)),
                "color": int(color),
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 6 and g.shape[1] >= 9

    patterns = _legend_patterns(g)
    regions = _enclosed_regions(g)

    out = g.copy()
    for region in regions:
        sr, sc = region["seed"]
        color = int(region["color"])
        pattern = patterns[color]
        for r, c in region["cells"]:
            out[r, c] = color if pattern[(r - sr) % 3, (c - sc) % 3] else BLACK
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_polyomino(rng: random.Random, area: int, force_irregular: bool) -> set[tuple[int, int]]:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(rng.choice(frontier))
        cells = _normalize(cells)
        rows = max(r for r, _ in cells) + 1
        cols = max(c for _, c in cells) + 1
        if rows > 4 or cols > 5:
            continue
        if force_irregular and rows * cols == area:
            continue
        return cells
    raise RuntimeError("polyomino generation failed")


def _random_shape(rng: random.Random) -> set[tuple[int, int]]:
    if rng.random() < 0.35:
        candidates = [(h, w) for h in range(1, 4) for w in range(2, 5) if 2 <= h * w <= 6]
        h, w = rng.choice(candidates)
        cells = {(r, c) for r in range(h) for c in range(w)}
        if h * w >= 4 and rng.random() < 0.5:
            removable = [
                (r, c)
                for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
                if (r, c) in cells and (r, c) != (0, 0)
            ]
            removed = rng.choice(removable)
            if h * w - 1 >= 2:
                cells.remove(removed)
        return _normalize(cells)
    area = rng.randint(2, 6)
    force_irregular = rng.random() < 0.7 and area >= 3
    return _random_polyomino(rng, area, force_irregular)


def _region_from_shape(shape: set[tuple[int, int]], rng: random.Random) -> dict:
    interior: set[tuple[int, int]] = set()
    anchors = []
    for br, bc in sorted(shape):
        top = 1 + 3 * br
        left = 1 + 3 * bc
        anchors.append((top, left))
        for dr in range(3):
            for dc in range(3):
                interior.add((top + dr, left + dc))

    boundary: set[tuple[int, int]] = set()
    for r, c in interior:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) not in interior:
                    boundary.add((nr, nc))

    h = max(r for r, _ in boundary | interior) + 1
    w = max(c for _, c in boundary | interior) + 1
    seed = rng.choice(anchors)
    return {
        "interior": sorted(interior),
        "boundary": sorted(boundary),
        "seed": seed,
        "height": h,
        "width": w,
    }


def _expanded_cells(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _render_legend(grid: np.ndarray, entries: list[tuple[int, np.ndarray]], rng: random.Random) -> None:
    h, w = grid.shape
    gap = rng.choice((0, 1))
    legend_width = 3 * len(entries) + gap * (len(entries) - 1)
    start_c = rng.randint(0, w - legend_width)
    row0 = h - 3
    c = start_c
    for color, pattern in entries:
        block = np.full((3, 3), BLACK, dtype=int)
        block[0, 0] = int(color)
        for rr in range(3):
            for cc in range(3):
                if (rr, cc) == (0, 0):
                    continue
                if pattern[rr, cc]:
                    block[rr, cc] = MAROON
        grid[row0 : row0 + 3, c : c + 3] = block
        c += 3 + gap


def _build_expected(
    grid: np.ndarray,
    placements: list[dict],
    pattern_map: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for spec in placements:
        color = int(spec["color"])
        seed_r, seed_c = spec["seed"]
        pattern = pattern_map[color]
        for r, c in spec["interior_global"]:
            out[r, c] = color if pattern[(r - seed_r) % 3, (c - seed_c) % 3] else BLACK
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_count = rng.randint(3, 5)
    legend_colors = rng.sample(SEED_COLORS, legend_count)
    chosen_patterns = [p.copy() for p in rng.sample(PATTERN_LIBRARY, legend_count)]
    pattern_map = {color: pat for color, pat in zip(legend_colors, chosen_patterns)}

    region_count = rng.randint(2, min(4, legend_count))
    region_colors = rng.sample(legend_colors, region_count)
    regions = []
    for color in region_colors:
        region = _region_from_shape(_random_shape(rng), rng)
        region["color"] = int(color)
        regions.append(region)

    gap = 1
    legend_width = 3 * legend_count + gap * (legend_count - 1)
    max_region_h = max(region["height"] for region in regions)
    max_region_w = max(region["width"] for region in regions)

    min_h = max(16, max_region_h + 6)
    max_h = max(min_h, min(30, min_h + 8))
    min_w = max(18, legend_width + 2, max_region_w + 4)
    max_w = max(min_w, min(36, min_w + 10))

    if min_h > 30 or min_w > 36:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    usable_bottom = h - 4
    if usable_bottom <= 2:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    for region in sorted(regions, key=lambda item: item["height"] * item["width"], reverse=True):
        rh = int(region["height"])
        rw = int(region["width"])
        max_start_r = usable_bottom - rh
        max_start_c = w - rw - 1
        if max_start_r < 1 or max_start_c < 1:
            return None

        placed = False
        for _ in range(200):
            start_r = rng.randint(1, max_start_r)
            start_c = rng.randint(1, max_start_c)

            footprint = [(start_r + r, start_c + c) for r, c in region["interior"] + region["boundary"]]
            if any(occupied[r, c] for r, c in footprint):
                continue

            for r, c in region["boundary"]:
                grid[start_r + r, start_c + c] = GREY
            seed_r = start_r + region["seed"][0]
            seed_c = start_c + region["seed"][1]
            grid[seed_r, seed_c] = int(region["color"])

            margin = _expanded_cells(footprint, h, w)
            for r, c in margin:
                occupied[r, c] = True

            placements.append(
                {
                    "color": int(region["color"]),
                    "seed": (int(seed_r), int(seed_c)),
                    "interior_global": [(start_r + r, start_c + c) for r, c in region["interior"]],
                }
            )
            placed = True
            break

        if not placed:
            return None

    _render_legend(grid, list(pattern_map.items()), rng)
    expected = _build_expected(grid, placements, pattern_map)
    if np.array_equal(expected, grid):
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
    raise RuntimeError("failed to generate a valid input")
