from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MIX = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, GREEN)): MAROON,
    frozenset((BLUE, GREEN)): AZURE,
}

ANTENNA_COLORS = (RED, BLUE, YELLOW, GREEN)


def _grey_buildings(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    buildings: list[tuple[int, int]] = []
    used = np.zeros((h, w), dtype=bool)
    for r in range(h - 2):
        for c in range(w - 2):
            if used[r : r + 3, c : c + 3].any():
                continue
            if np.all(grid[r : r + 3, c : c + 3] == GREY):
                buildings.append((r, c))
                used[r : r + 3, c : c + 3] = True
    return buildings


def _antenna_cell(top: int, left: int, side: str) -> tuple[int, int]:
    if side == "top":
        return top - 1, left + 1
    if side == "bottom":
        return top + 3, left + 1
    if side == "left":
        return top + 1, left - 1
    return top + 1, left + 3


def _extract_antennas(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    antennas: list[dict] = []
    for top, left in _grey_buildings(grid):
        for side, direction in DIRS.items():
            r, c = _antenna_cell(top, left, side)
            if not (0 <= r < h and 0 <= c < w):
                continue
            color = int(grid[r, c])
            if color not in (BG, GREY):
                antennas.append({"cell": (r, c), "dir": direction, "color": color})
    return antennas


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color antenna rays to the grid edge and mark crossings with mixed colors.

    Buildings and original antenna pixels remain unchanged. A crossing is any background
    pixel traversed by rays of two different colors; the crossing's 2x2 marker uses the
    fixed visible color-mixing table.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    coverage: dict[tuple[int, int], set[int]] = defaultdict(set)

    for ant in _extract_antennas(g):
        color = int(ant["color"])
        for cell in _ray_cells(ant["cell"], ant["dir"], g.shape):
            coverage[cell].add(color)

    for (r, c), colors in coverage.items():
        if int(g[r, c]) == BG and len(colors) == 1:
            out[r, c] = next(iter(colors))

    h, w = g.shape
    for (r, c), colors in coverage.items():
        if len(colors) < 2:
            continue
        key = frozenset(colors)
        if key not in MIX:
            continue
        marker = MIX[key]
        for rr in range(r, min(h, r + 2)):
            for cc in range(c, min(w, c + 2)):
                if int(g[rr, cc]) == BG:
                    out[rr, cc] = marker

    return out


def _fits(occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    r0, r1 = top - 2, top + 5
    c0, c1 = left - 2, left + 5
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return False
    return not occupied[r0:r1, c0:c1].any()


def _place_buildings(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    buildings: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _attempt in range(300):
            top = random.randint(2, h - 5)
            left = random.randint(2, w - 5)
            if not _fits(occupied, top, left):
                continue
            occupied[top - 2 : top + 5, left - 2 : left + 5] = True
            buildings.append((top, left))
            placed = True
            break
        if not placed:
            return None
    return buildings


def _assign_antennas(buildings: list[tuple[int, int]], pair_count: int) -> list[tuple[int, str, int]] | None:
    per_building: list[set[str]] = [set() for _ in buildings]
    assignments: list[tuple[int, str, int]] = []
    colors = list(ANTENNA_COLORS[:pair_count])
    random.shuffle(colors)

    for color in colors:
        candidates = [i for i, dirs in enumerate(per_building) if len(dirs) < 3]
        if len(candidates) < 2:
            return None
        b1, b2 = random.sample(candidates, 2)
        for b in (b1, b2):
            available = [side for side in DIRS if side not in per_building[b]]
            if not available:
                return None
            side = random.choice(available)
            per_building[b].add(side)
            assignments.append((b, side, color))

    if any(len(dirs) == 0 for dirs in per_building):
        return None
    return assignments


def _render_input(h: int, w: int, buildings: list[tuple[int, int]], assignments: list[tuple[int, str, int]]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    for top, left in buildings:
        grid[top : top + 3, left : left + 3] = GREY

    for b, side, color in assignments:
        r, c = _antenna_cell(*buildings[b], side)
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return None
        grid[r, c] = color
    return grid


def _valid_scene(grid: np.ndarray) -> bool:
    antennas = _extract_antennas(grid)
    if not (4 <= len(antennas) <= 8):
        return False
    counts = {color: 0 for color in ANTENNA_COLORS}
    for ant in antennas:
        if ant["color"] not in counts:
            return False
        counts[ant["color"]] += 1
    used_counts = [v for v in counts.values() if v]
    if any(v != 2 for v in used_counts):
        return False

    h, w = grid.shape
    original_non_bg = grid != BG
    coverage: dict[tuple[int, int], set[int]] = defaultdict(set)
    antenna_cells = {ant["cell"] for ant in antennas}
    for ant in antennas:
        for cell in _ray_cells(ant["cell"], ant["dir"], grid.shape):
            if cell != ant["cell"] and original_non_bg[cell]:
                return False
            coverage[cell].add(int(ant["color"]))

    crossings = []
    for cell, colors in coverage.items():
        if cell in antenna_cells:
            continue
        if len(colors) == 2:
            if frozenset(colors) not in MIX:
                return False
            crossings.append(cell)
        elif len(colors) > 2:
            return False
    if not crossings:
        return False
    if len(crossings) > 8:
        return False

    for r, c in crossings:
        if r >= h - 1 or c >= w - 1:
            return False
        block = [(rr, cc) for rr in (r, r + 1) for cc in (c, c + 1)]
        if any(original_non_bg[rr, cc] and (rr, cc) not in antenna_cells for rr, cc in block):
            return False
    _ = main(grid)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    pair_count = random.randint(2, 4)
    building_count = random.randint(max(3, (2 * pair_count + 2) // 3), min(6, 2 * pair_count))
    buildings = _place_buildings(h, w, building_count)
    if buildings is None:
        return None
    assignments = _assign_antennas(buildings, pair_count)
    if assignments is None:
        return None
    grid = _render_input(h, w, buildings, assignments)
    if grid is None or not _valid_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid antenna-ray input")
