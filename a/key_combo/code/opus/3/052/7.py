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
    """Paste the source-window blue pattern into each target window with marker alignment plus magenta-coded one-cell offset."""
    out = np.array(grid, copy=True)
    windows = []
    for comp in _components(grid != 0):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3 and np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == 5):
            windows.append((r0, c0, r1, c1))
    source = None
    targets = []
    for box in windows:
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == 1):
            source = box
        elif np.any(sub == 6):
            targets.append(box)
    if source is None:
        return out

    sr0, sc0, sr1, sc1 = source
    yellow = np.argwhere(grid[sr0 : sr1 + 1, sc0 : sc1 + 1] == 4)
    if len(yellow) == 0:
        return out
    sm = (sr0 + int(yellow[0, 0]), sc0 + int(yellow[0, 1]))
    offsets = [(int(r) - sm[0], int(c) - sm[1]) for r, c in np.argwhere(grid == 1)]

    for r0, c0, r1, c1 in targets:
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        ys = np.argwhere(sub == 4)
        ms = np.argwhere(sub == 6)
        if len(ys) == 0 or len(ms) == 0:
            continue
        marker = (r0 + int(ys[0, 0]), c0 + int(ys[0, 1]))
        mag = (r0 + int(ms[0, 0]), c0 + int(ms[0, 1]))
        extra = (mag[0] - marker[0], mag[1] - marker[1])
        for rr, cc in (marker, mag):
            out[rr, cc] = 5
        for dr, dc in offsets:
            nr, nc = marker[0] + dr + extra[0], marker[1] + dc + extra[1]
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                out[nr, nc] = 1
    return out


def _window(grid: np.ndarray, r: int, c: int, h: int = 5, w: int = 5) -> None:
    grid[r : r + h, c : c + w] = 5


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    _window(grid, 1, 1)
    marker = (3, 3)
    grid[marker] = 4
    sprite = random.choice([
        [(1, 1), (2, 2), (3, 1), (4, 3)],
        [(1, 2), (2, 1), (3, 2), (4, 2)],
        [(1, 1), (1, 2), (2, 2), (4, 3)],
    ])
    for p in sprite:
        grid[p] = 1
    targets = [(1, 12, (0, 1)), (12, 1, (-1, 0)), (12, 12, (1, 0)), (8, 12, (0, -1))]
    random.shuffle(targets)
    for r, c, extra in targets[: random.randint(2, 3)]:
        _window(grid, r, c)
        m = (r + 2, c + 2)
        grid[m] = 4
        grid[m[0] + extra[0], m[1] + extra[1]] = 6
    return grid
