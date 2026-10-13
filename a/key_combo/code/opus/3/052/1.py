from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _read_band(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    for r in range(grid.shape[0] - 1, -1, -1):
        cols = np.where(grid[r] != 0)[0]
        if len(cols) >= 3:
            c0, c1 = int(cols.min()), int(cols.max())
            cells = {(r, c) for c in range(c0, c1 + 1) if grid[r, c] != 0}
            return [int(grid[r, c]) for c in range(c0, c1 + 1) if grid[r, c] != 0], cells
    return [4, 5, 6, 7], set()


def _red_component_map(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], list[list[tuple[int, int]]]]:
    comps = _components(grid == 2)
    owner: dict[tuple[int, int], int] = {}
    for i, comp in enumerate(comps):
        for p in comp:
            owner[p] = i
    return owner, comps


def _adjacent_red(pos: list[tuple[int, int]], red_owner: dict[tuple[int, int], int]) -> int | None:
    hits: list[int] = []
    for r, c in pos:
        for dr, dc in DIRS:
            owner = red_owner.get((r + dr, c + dc))
            if owner is not None:
                hits.append(owner)
    return min(hits) if hits else None


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each marked sprite toward its encoded direction and recolor it by per-obstacle contact order."""
    grid = np.array(grid, copy=True)
    band, band_cells = _read_band(grid)
    red_owner, _ = _red_component_map(grid)

    sprite_mask = (grid != 0) & (grid != 2) & (grid != 3)
    for r, c in band_cells:
        sprite_mask[r, c] = False
    sprites = _components(sprite_mask)
    sprites.sort(key=lambda comp: (_bbox(comp)[0], _bbox(comp)[1]))

    out = np.array(grid, copy=True)
    out[out == 3] = 0
    for comp in sprites:
        for r, c in comp:
            out[r, c] = 0

    contact_counts: dict[int, int] = {}
    green = set(map(tuple, np.argwhere(grid == 3)))
    for comp in sprites:
        comp_set = set(comp)
        marker = None
        direction = None
        for r, c in comp:
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in green:
                    marker = p
                    direction = (dr, dc)
                    break
            if marker is not None:
                break
        if direction is None:
            continue

        pos = list(comp_set)
        obstacle = _adjacent_red(pos, red_owner)
        guard = 0
        while obstacle is None and guard < grid.size:
            dr, dc = direction
            nxt = [(r + dr, c + dc) for r, c in pos]
            if any(r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1] for r, c in nxt):
                break
            pos = nxt
            obstacle = _adjacent_red(pos, red_owner)
            guard += 1
        if obstacle is None:
            obstacle = -1
        contact_counts[obstacle] = contact_counts.get(obstacle, 0) + 1
        color = band[(contact_counts[obstacle] - 1) % len(band)]
        for r, c in pos:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == 0:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    band = random.sample([4, 5, 6, 7, 8, 9], 5)
    grid[17, : len(band)] = band

    top_r = random.choice([2, 3, 4])
    vert_c = random.choice([13, 14, 15])
    obstacles = [
        [(top_r, c) for c in range(5, 12)],
        [(r, vert_c) for r in range(6, 13)],
    ]
    for comp in obstacles:
        for r, c in comp:
            grid[r, c] = 2

    jitter = random.choice([0, 1])
    specs = [
        ([(8, 2 + jitter), (8, 3 + jitter), (9, 2 + jitter), (10, 2 + jitter)], (0, 1)),
        ([(12, 9), (12, 10), (13, 10), (13, 11)], (0, 1)),
        ([(6, 7), (7, 7), (7, 8), (8, 8)], (-1, 0)),
        ([(4, 16), (5, 16), (5, 15)], (0, -1)),
        ([(14, 3), (14, 4), (15, 4)], (0, 1)),
    ]
    random.shuffle(specs)
    for cells, direction in specs[: random.randint(3, 4)]:
        for r, c in cells:
            grid[r, c] = 1
        dr, dc = direction
        sr = sum(r for r, _ in cells) // len(cells)
        sc = sum(c for _, c in cells) // len(cells)
        candidates = [(r + dr, c + dc) for r, c in cells if grid[r + dr, c + dc] == 0]
        mr, mc = candidates[0] if candidates else (sr + dr, sc + dc)
        grid[mr, mc] = 3
    return grid
