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


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the red sprite's mirror symmetries and use them to flip odd rows or columns of a 3x3 tile array."""
    tile = np.array(grid[:3, :3], copy=True)
    sprites = _components(grid == 2)
    sprite = max(sprites, key=len) if sprites else []
    if not sprite:
        return np.tile(tile, (3, 3))
    r0, c0, r1, c1 = _bbox(sprite)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in sprite:
        shape[r - r0, c - c0] = True
    sym_lr = np.array_equal(shape, np.fliplr(shape))
    sym_ud = np.array_equal(shape, np.flipud(shape))
    out = np.zeros((9, 9), dtype=grid.dtype)
    for tr in range(3):
        for tc in range(3):
            piece = tile.copy()
            if sym_lr and tc % 2 == 1:
                piece = np.fliplr(piece)
            if sym_ud and tr % 2 == 1:
                piece = np.flipud(piece)
            out[tr * 3 : tr * 3 + 3, tc * 3 : tc * 3 + 3] = piece
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    grid[:3, :3] = np.array(random.sample(list(range(1, 10)), 9)).reshape(3, 3)
    shapes = [
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
    ]
    shape = random.choice(shapes)
    r0, c0 = random.choice([(10, 10), (9, 10), (10, 9), (8, 11)])
    color = random.choice([2, 3, 4, 6])
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color
    return grid
