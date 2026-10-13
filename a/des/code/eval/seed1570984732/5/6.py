import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
PIECE_COLORS = [RED, BLUE]
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_signal(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) not in PIECE_COLORS:
        return False
    h, w = grid.shape
    track_neighbors = 0
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in (GREY, GREEN, ORANGE):
            track_neighbors += 1
    return track_neighbors >= 2


def _network_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.isin(grid, [GREY, GREEN, ORANGE])
    for r, c in np.argwhere(np.isin(grid, PIECE_COLORS)):
        if _is_signal(grid, int(r), int(c)):
            mask[int(r), int(c)] = True
    return mask


def _network_neighbors(mask: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = pos
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _piece_components(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    pieces = []
    signal_cells = set(map(tuple, np.argwhere(mask & np.isin(grid, PIECE_COLORS))))
    for color in PIECE_COLORS:
        piece_mask = (grid == color).copy()
        for r, c in signal_cells:
            piece_mask[r, c] = False
        for comp in _components(piece_mask):
            if len(comp) != 4:
                continue
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
                cells = set(comp)
                pieces.append({"color": color, "cells": cells, "top": min(rs), "left": min(cs)})
    return pieces


def _initial_dir(mask: np.ndarray, start: tuple[int, int]) -> tuple[int, int] | None:
    neigh = _network_neighbors(mask, start)
    for nr, nc in neigh:
        return nr - start[0], nc - start[1]
    return None


def _find_piece_for_start(pieces: list[dict], start: tuple[int, int], used: set[int]) -> int | None:
    sr, sc = start
    best = None
    best_dist = 10**9
    for i, piece in enumerate(pieces):
        if i in used:
            continue
        dist = min(abs(sr - r) + abs(sc - c) for r, c in piece["cells"])
        if dist < best_dist:
            best = i
            best_dist = dist
    return best if best_dist <= 4 else None


def _turn(direction: tuple[int, int], right: bool) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx + (1 if right else -1)) % 4]


def _trace(grid: np.ndarray, mask: np.ndarray, start: tuple[int, int], color: int):
    direction = _initial_dir(mask, start)
    if direction is None:
        return None
    prev = start
    pos = start
    h, w = grid.shape
    for steps in range(h * w):
        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
            choices = [p for p in _network_neighbors(mask, pos) if p != prev]
            if not choices:
                return None
            nr, nc = choices[0]
            direction = (nr - pos[0], nc - pos[1])
        prev, pos = pos, (nr, nc)
        val = int(grid[pos])
        if val == GREEN:
            return {"end": pos, "dir": direction, "steps": steps + 1}
        if _is_signal(grid, pos[0], pos[1]):
            direction = _turn(direction, val == color)
    return None


def _dest_top_left(end: tuple[int, int], direction: tuple[int, int], index: int) -> tuple[int, int]:
    r, c = end
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return r - 1, c + 1 + 2 * index
    if (dr, dc) == (0, -1):
        return r - 1, c - 2 - 2 * index
    if (dr, dc) == (1, 0):
        return r + 1 + 2 * index, c - 1
    return r - 2 - 2 * index, c - 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mask = _network_mask(g)
    pieces = _piece_components(g, mask)
    starts = [tuple(map(int, p)) for p in np.argwhere(g == ORANGE)]
    used: set[int] = set()
    arrivals = []
    for start in sorted(starts):
        idx = _find_piece_for_start(pieces, start, used)
        if idx is None:
            continue
        used.add(idx)
        piece = pieces[idx]
        traced = _trace(g, mask, start, piece["color"])
        if traced is None:
            continue
        for r, c in piece["cells"]:
            out[r, c] = BG
        arrivals.append((traced["steps"], traced["end"], traced["dir"], piece["color"]))

    counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for _, end, direction, color in sorted(arrivals):
        key = (end, direction)
        idx = counts.get(key, 0)
        counts[key] = idx + 1
        top, left = _dest_top_left(end, direction, idx)
        if 0 <= top < out.shape[0] - 1 and 0 <= left < out.shape[1] - 1:
            out[top : top + 2, left : left + 2] = color
    return out


def _draw_route(grid: np.ndarray, row: int, col: int, end_row: int, piece_color: int) -> None:
    start = (row, 3)
    grid[start] = ORANGE
    for c in range(4, col):
        grid[row, c] = GREY
    turn_south = end_row > row
    signal = piece_color if turn_south else (BLUE if piece_color == RED else RED)
    grid[row, col] = signal
    step = 1 if turn_south else -1
    for r in range(row + step, end_row, step):
        grid[r, col] = GREY
    grid[end_row, col] = GREEN
    if col + 1 < grid.shape[1]:
        grid[row, col + 1] = GREY
    dead_r = row - step
    if 0 <= dead_r < grid.shape[0]:
        grid[dead_r, col] = GREY
    grid[row - 1 : row + 1, 1:3] = piece_color


def _gen_attempt() -> np.ndarray | None:
    h = 30
    w = 30
    grid = np.full((h, w), BG, dtype=int)
    route_count = random.randint(2, 3)
    bands = [(6, (10, 12)), (15, (10, 20)), (24, (18, 20))]
    random.shuffle(bands)
    used = bands[:route_count]
    cols = random.sample(range(9, 25), route_count)
    for i, (row, end_options) in enumerate(used):
        color = random.choice(PIECE_COLORS)
        if isinstance(end_options, tuple) and len(end_options) == 2:
            end_row = random.choice(list(range(min(end_options), max(end_options) + 1)))
        else:
            end_row = int(end_options)
        if row == 15:
            end_row = random.choice([10, 20])
        _draw_route(grid, row + random.choice([-1, 0, 1]), cols[i], end_row, color)
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-track input")
