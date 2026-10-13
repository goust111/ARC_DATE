from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
GEN_ATTEMPTS = 300


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _layer_distances(grid: np.ndarray, region: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    region_set = set(region)
    q = deque()
    dist: dict[tuple[int, int], int] = {}
    for cell in region:
        r, c = cell
        if any((nr, nc) not in region_set and (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GRAY) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            dist[cell] = 1
            q.append(cell)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nxt = (nr, nc)
            if nxt in region_set and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint concentric inner-wall layers by ordering each enclosed region's seeds from nearest wall to deepest.

    Input structure:
    - The background is black.
    - Gray closed boundaries enclose one or more interior regions.
    - Inside each enclosed region, one to three differently colored single-pixel seeds sit among black cells.

    Rule:
    - Find each enclosed non-gray region that does not touch the grid edge.
    - Number its interior cells by layer distance from the surrounding gray wall: cells adjacent to the wall are layer 1, the next peel inward is layer 2, and so on.
    - Sort that region's seed colors by the layer of their seed pixels, from nearest wall to deepest.
    - Repaint layer 1 with the nearest seed's color, layer 2 with the next seed's color, and so on; deeper layers beyond the seed count stay black.

    Key insight:
    - The seeds do not mark their own final locations; their relative depth inside the enclosure defines an outer-to-inner color order for the whole region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    region_mask = g != GRAY
    for comp in _components(region_mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        dists = _layer_distances(g, comp)
        seeds = [(dists[(r, c)], int(g[r, c])) for r, c in comp if int(g[r, c]) not in (BG, GRAY)]
        assert seeds
        seeds.sort()
        ordered_colors = [color for _depth, color in seeds]
        for r, c in comp:
            layer = dists[(r, c)]
            out[r, c] = ordered_colors[layer - 1] if layer <= len(ordered_colors) else BG
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG, dtype=int)
        region_count = random.randint(2, 4)
        ok = True
        palette = [1, 2, 3, 4, 6, 7, 8, 9]

        for _region in range(region_count):
            placed = False
            for _ in range(100):
                rh = random.randint(6, 10)
                rw = random.randint(6, 10)
                r0 = random.randint(1, h - rh - 1)
                c0 = random.randint(1, w - rw - 1)
                area = grid[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1]
                if np.any(area != BG):
                    continue

                grid[r0, c0 : c0 + rw] = GRAY
                grid[r0 + rh - 1, c0 : c0 + rw] = GRAY
                grid[r0 : r0 + rh, c0] = GRAY
                grid[r0 : r0 + rh, c0 + rw - 1] = GRAY

                max_depth = min(rh - 2, rw - 2) // 2
                seed_count = random.randint(1, min(3, max_depth))
                depths = random.sample(range(1, max_depth + 1), seed_count)
                colors = random.sample(palette, seed_count)
                for depth, color in zip(sorted(depths), colors):
                    candidates = [
                        (r, c)
                        for r in range(r0 + 1, r0 + rh - 1)
                        for c in range(c0 + 1, c0 + rw - 1)
                        if min(r - r0, c - c0, r0 + rh - 1 - r, c0 + rw - 1 - c) == depth
                    ]
                    sr, sc = random.choice(candidates)
                    grid[sr, sc] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
