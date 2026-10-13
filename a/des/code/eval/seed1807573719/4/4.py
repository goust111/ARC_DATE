from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
RED = 2
BLUE = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    return sum(1 for rr, cc in _neighbors4(r, c, *mask.shape) if mask[rr, cc])


def _valve_near(grid: np.ndarray, r: int, c: int) -> tuple[int, tuple[int, int]] | None:
    for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
        for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) in (RED, BLUE):
                return int(grid[rr, cc]), (rr - r, cc - c)
    return None


def _edge_lights(grid: np.ndarray, pipe: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = grid.shape
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, GREY, RED, BLUE):
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            nbrs = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if pipe[rr, cc]]
            if len(nbrs) == 1:
                out.append(((r, c), nbrs[0], color))
    return out


def _trace(pipe: np.ndarray, grid: np.ndarray, start: tuple[int, int], prev: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = pipe.shape
    path = [start]
    cur = start
    seen = {cur}
    for _ in range(h * w):
        deg = _degree(pipe, *cur)
        choices = [(rr, cc) for rr, cc in _neighbors4(*cur, h, w) if pipe[rr, cc] and (rr, cc) != prev]
        if deg == 1 and cur != start:
            return path
        if deg >= 3:
            valve = _valve_near(grid, *cur)
            in_vec = (cur[0] - prev[0], cur[1] - prev[1])
            if valve is None:
                return path
            valve_color, valve_offset = valve
            if valve_color == BLUE:
                target = (cur[0] + in_vec[0], cur[1] + in_vec[1])
                choices = [target] if target in choices else []
            else:
                turns = [p for p in choices if (p[0] - cur[0], p[1] - cur[1]) != in_vec]
                if len(turns) > 1:
                    vr, vc = valve_offset
                    preferred = [p for p in turns if (p[0] - cur[0]) * vr + (p[1] - cur[1]) * vc > 0]
                    turns = preferred or turns
                choices = turns[:1]
        if len(choices) != 1:
            return path
        nxt = choices[0]
        if nxt in seen:
            return path
        prev, cur = cur, nxt
        path.append(cur)
        seen.add(cur)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pipe = g == GREY
    out = g.copy()
    for light_cell, entry, color in _edge_lights(g, pipe):
        path = _trace(pipe, g, entry, light_cell)
        if not path:
            continue
        end = path[-1]
        if _degree(pipe, *end) != 1:
            continue
        for cell in path[-4:]:
            out[cell] = color
    return out


def _draw_path(grid: np.ndarray, pts: list[tuple[int, int]]) -> None:
    for a, b in zip(pts, pts[1:]):
        r0, c0 = a
        r1, c1 = b
        if r0 == r1:
            for c in range(min(c0, c1), max(c0, c1) + 1):
                grid[r0, c] = GREY
        else:
            for r in range(min(r0, r1), max(r0, r1) + 1):
                grid[r, c0] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 26)
    w = random.randint(17, 28)
    grid = np.zeros((h, w), dtype=int)
    row = random.randint(5, h - 6)
    c0 = random.randint(2, 5)
    c1 = random.randint(w // 2, w - 5)
    up = random.randint(2, row - 2)
    down = random.randint(2, h - row - 3)
    _draw_path(grid, [(0, c0), (row, c0), (row, c1), (row + down, c1)])
    # A red valve at the T/corner area forces the signal to turn toward the marker side.
    valve_pos = (row - 1, c0 + 1)
    if grid[valve_pos] != BG:
        return None
    grid[valve_pos] = RED
    color = random.choice([3, 4, 6, 7, 8, 9])
    grid[0, c0] = color
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valve-pipe input")
