from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

BLACK = 0
WALL = 2
BLUE = 1


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                x, y = q.pop()
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
                        comp.append((nx, ny))
            out.append(comp)
    return out


def _nearest_cell_in_set(comp: set[tuple[int, int]], rr: float, cc: float) -> tuple[int, int]:
    best = None
    best_d = 10**9
    for r, c in sorted(comp):
        d = abs(rr - r) + abs(cc - c)
        if d < best_d:
            best_d = d
            best = (r, c)
    if best is None:
        raise ValueError("_nearest_cell_in_set: empty")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    wall = out == WALL
    black = out == BLACK
    # border flood fill -> exterior black
    ext = np.zeros_like(black, dtype=bool)
    q = deque()
    for c in range(w):
        if black[0, c]:
            ext[0, c] = True
            q.append((0, c))
        if black[h - 1, c]:
            ext[h - 1, c] = True
            q.append((h - 1, c))
    for r in range(h):
        if black[r, 0]:
            ext[r, 0] = True
            q.append((r, 0))
        if black[r, w - 1]:
            ext[r, w - 1] = True
            q.append((r, w - 1))
    while q:
        r, c = q.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not ext[nr, nc] and not wall[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))

    enclosed = black & ~ext
    regions = _components(enclosed)
    n = len(regions)
    if n == 0:
        return out

    for comp in regions:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        cr = float(sum(rs) / len(rs))
        cc = float(sum(cs) / len(cs))
        rr, cc0 = round(cr), round(cc)
        rr_i = int(rr)
        cc_i = int(cc0)
        comp_set = set(comp)
        if (rr_i, cc_i) not in comp_set:
            rr_i, cc_i = _nearest_cell_in_set(comp_set, cr, cc0)
        if out[rr_i, cc_i] == BLACK:
            out[rr_i, cc_i] = BLUE

        r, c = rr_i, cc_i
        arm = 1
        turns = 0
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_i = 0
        steps_to_draw = 2 * n
        for _arm in range(1, steps_to_draw + 1):
            length = ( _arm + 1) // 2
            dr, dc = dirs[dir_i]
            for _ in range(length):
                r += dr
                c += dc
                if (r, c) in comp_set and out[r, c] == BLACK:
                    out[r, c] = BLUE
            dir_i = (dir_i + 1) % 4

    return out


def _draw_border(out: np.ndarray, top: int, left: int, h: int, w: int, color: int = WALL) -> None:
    out[top, left : left + w] = color
    out[top + h - 1, left : left + w] = color
    out[top : top + h, left] = color
    out[top : top + h, left + w - 1] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    size = 21
    out = np.zeros((size, size), dtype=int)
    wall = np.zeros_like(out, dtype=bool)
    n_regions = int(rng.integers(2, 6))
    attempts = 0
    created = 0
    while created < n_regions and attempts < 300:
        attempts += 1
        hh = int(rng.integers(6, 9))
        ww = int(rng.integers(6, 9))
        top = int(rng.integers(1, size - hh - 1))
        left = int(rng.integers(1, size - ww - 1))

        # border cells for the frame rectangle
        cand = np.zeros_like(wall, dtype=bool)
        cand[top, left : left + ww] = True
        cand[top + hh - 1, left : left + ww] = True
        cand[top : top + hh, left] = True
        cand[top : top + hh, left + ww - 1] = True

        # require isolation by at least one cell around wall
        expanded = np.zeros_like(wall, dtype=bool)
        pad = np.zeros((cand.shape[0] + 2, cand.shape[1] + 2), dtype=bool)
        pad[1:-1, 1:-1] = cand
        expanded = (
            pad[:-2, 1:-1]
            | pad[2:, 1:-1]
            | pad[1:-1, :-2]
            | pad[1:-1, 2:]
            | pad[:-2, :-2]
            | pad[:-2, 2:]
            | pad[2:, :-2]
            | pad[2:, 2:]
        )
        if np.any(expanded & wall):
            continue
        out[cand] = WALL
        wall |= cand
        created += 1

    # ensure outer wall not touched by any object
    out[0, :] = 0
    out[:, 0] = 0
    out[size - 1, :] = 0
    out[:, size - 1] = 0

    return out

