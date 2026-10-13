from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
RAY_COLORS = (1, 2, 3, 7)
FILL_TWO_COLORS = 4
FILL_THREE_PLUS = 5
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRY_LIMIT = 500


def _color_priority(color: int) -> tuple[int, int]:
    if color in RAY_COLORS:
        return (0, RAY_COLORS.index(color))
    return (1, int(color))


def _overlay_color(current: int, new: int) -> int:
    if current == BACKGROUND:
        return int(new)
    return int(current if _color_priority(current) <= _color_priority(new) else new)


def _seed_list(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BACKGROUND)]


def _paint_rays(shape: tuple[int, int], seeds: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    rays = np.zeros((h, w), dtype=int)
    for r0, c0, color in seeds:
        for dr, dc in DIAGONALS:
            r, c = r0, c0
            while 0 <= r < h and 0 <= c < w:
                rays[r, c] = _overlay_color(int(rays[r, c]), color)
                r += dr
                c += dc
    return rays


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _region_boundary_colors(rays: np.ndarray, region: list[tuple[int, int]]) -> set[int]:
    h, w = rays.shape
    colors: set[int] = set()
    for r, c in region:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and rays[nr, nc] != BACKGROUND:
                colors.add(int(rays[nr, nc]))
    return colors


def _fill_color(colors: set[int]) -> int:
    if len(colors) == 1:
        return next(iter(colors))
    if len(colors) == 2:
        return FILL_TWO_COLORS
    return FILL_THREE_PLUS


def _solve_with_analysis(grid: np.ndarray) -> tuple[np.ndarray, list[set[int]]]:
    seeds = _seed_list(grid)
    rays = _paint_rays(grid.shape, seeds)
    out = rays.copy()
    boundary_sets: list[set[int]] = []
    for region in _components(rays == BACKGROUND):
        colors = _region_boundary_colors(rays, region)
        boundary_sets.append(colors)
        if not colors:
            continue
        out_color = _fill_color(colors)
        for r, c in region:
            out[r, c] = out_color
    return out, boundary_sets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from isolated colored pixels and fill each ray-bounded background region by the number of ray colors on its boundary.

    Input structure:
    - The grid is mostly black background.
    - A few isolated interior pixels use distinct colors chosen from blue, red, green, and orange.
    - There are no other foreground objects; these isolated pixels are the ray sources.

    Rule:
    - From every non-background pixel, extend rays in the four diagonal directions all the way to the grid boundary using that source color.
    - When different colored rays overlap, the visible ray cell keeps the fixed priority order blue, then red, then green, then orange.
    - Treat the remaining black cells as 4-connected regions separated by the ray network, collect the distinct ray colors adjacent to each region, and fill the whole region with the unique boundary color if there is one, yellow if there are two, or gray if there are three or more.
    - Keep all ray cells themselves unchanged in the output.

    Key insight:
    - The input contains only isolated emitters; the meaningful regions do not exist beforehand and emerge only after all diagonal rays are drawn and used as boundaries.
    """
    g = np.asarray(grid, dtype=int)
    out, _ = _solve_with_analysis(g)
    return out


def _well_separated(candidate: tuple[int, int], placed: list[tuple[int, int, int]]) -> bool:
    r, c = candidate
    return all(abs(r - rr) + abs(c - cc) >= 3 for rr, cc, _ in placed)


def _candidate_positions(h: int, w: int, parity: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if (r + c) % 2 == parity
    ]


def _random_seeds(h: int, w: int) -> list[tuple[int, int, int]] | None:
    n_seeds = random.choice((3, 3, 4))
    colors = random.sample(list(RAY_COLORS), n_seeds)
    positions = _candidate_positions(h, w, random.randint(0, 1))
    random.shuffle(positions)
    seeds: list[tuple[int, int, int]] = []
    for color in colors:
        options = [pos for pos in positions if _well_separated(pos, seeds)]
        if not options:
            return None
        r, c = random.choice(options)
        seeds.append((r, c, color))
        positions.remove((r, c))
    return seeds


def _valid_seed_layout(h: int, w: int, seeds: list[tuple[int, int, int]]) -> bool:
    grid = np.zeros((h, w), dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color
    _, boundary_sets = _solve_with_analysis(grid)
    counts = [len(colors) for colors in boundary_sets if colors]
    if len(boundary_sets) < 5:
        return False
    if 1 not in counts or 2 not in counts or max(counts, default=0) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        h = random.randint(14, 22)
        w = random.randint(14, 22)
        seeds = _random_seeds(h, w)
        if seeds is None:
            continue
        if not _valid_seed_layout(h, w, seeds):
            continue
        grid = np.zeros((h, w), dtype=int)
        for r, c, color in seeds:
            grid[r, c] = color
        return grid
    raise RuntimeError("generate_input failed to produce a valid ray-partition puzzle")
