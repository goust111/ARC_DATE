from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _legend(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    r = 0
    while r < grid.shape[0]:
        color = int(grid[r, 0])
        if color in (BG, GREY):
            break
        mapping[color] = r + 1
        r += 1
    assert mapping, "expected a top-left vertical color-rank legend"
    return mapping


def _seed_cells(grid: np.ndarray, legend: dict[int, int]) -> list[tuple[int, int, int]]:
    seeds = []
    for color in legend:
        pts = np.argwhere(grid == color)
        pts = [tuple(map(int, p)) for p in pts if int(p[1]) != 0]
        for r, c in pts:
            seeds.append((r, c, int(color)))
    return seeds


def _perimeter(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield r0, c, "up"
        yield r1, c, "down"
    for r in range(r0 + 1, r1):
        yield r, c0, "left"
        yield r, c1, "right"


def _find_room(grid: np.ndarray, sr: int, sc: int) -> tuple[int, int, int, int, set[str]]:
    h, w = grid.shape
    best = None
    for r0 in range(max(0, sr - 9), sr - 1):
        for r1 in range(sr + 2, min(h, sr + 10)):
            for c0 in range(max(1, sc - 9), sc - 1):
                for c1 in range(sc + 2, min(w, sc + 10)):
                    per = list(_perimeter(r0, c0, r1, c1))
                    values = [int(grid[r, c]) for r, c, _side in per]
                    if any(v not in (BG, GREY) for v in values):
                        continue
                    grey_count = sum(v == GREY for v in values)
                    if grey_count < max(6, int(0.65 * len(values))):
                        continue
                    sides_with_grey = {side for (r, c, side), v in zip(per, values, strict=False) if v == GREY}
                    if sides_with_grey != {"up", "down", "left", "right"}:
                        continue
                    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if np.any(inner == GREY):
                        continue
                    gap_sides = {side for (r, c, side), v in zip(per, values, strict=False) if v == BG}
                    if not gap_sides:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    cand = (area, r0, c0, r1, c1, gap_sides)
                    if best is None or cand < best:
                        best = cand
    assert best is not None, "room frame not found"
    _area, r0, c0, r1, c1, gap_sides = best
    return r0, c0, r1, c1, gap_sides


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays from each room seed. Legend rank gives length; frame gaps give directions.
    """
    g = np.asarray(grid, dtype=int)
    legend = _legend(g)
    out = g.copy()
    h, w = out.shape
    for sr, sc, color in _seed_cells(g, legend):
        length = legend[color]
        _r0, _c0, _r1, _c1, gap_sides = _find_room(g, sr, sc)
        for side in gap_sides:
            dr, dc = DIRS[side]
            r, c = sr, sc
            for _ in range(length):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if int(out[nr, nc]) == GREY:
                    break
                out[nr, nc] = color
                r, c = nr, nc
    return out


def _fits(occupied: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + rh > h or c0 + rw > w:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + rh + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + rw + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_room(grid: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    rh = random.randint(5, 8)
    rw = random.randint(5, 9)
    for _ in range(100):
        r0 = random.randint(2, h - rh - 1)
        c0 = random.randint(3, w - rw - 1)
        if not _fits(occupied, r0, c0, rh, rw):
            continue
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY

        sides = random.sample(list(DIRS), random.randint(1, 3))
        for side in sides:
            if side in ("up", "down"):
                c = random.randint(c0 + 1, c1 - 1)
                grid[r0 if side == "up" else r1, c] = BG
            else:
                r = random.randint(r0 + 1, r1 - 1)
                grid[r, c0 if side == "left" else c1] = BG

        sr = random.randint(r0 + 2, r1 - 2)
        sc = random.randint(c0 + 2, c1 - 2)
        grid[sr, sc] = color
        occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
    for r, color in enumerate(colors):
        grid[r, 0] = color
        occupied[r, 0] = True

    room_colors = random.sample(colors, random.randint(2, min(4, len(colors))))
    for color in room_colors:
        if not _draw_room(grid, occupied, color):
            return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ray-growth input")
