from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


def _bg(grid: np.ndarray) -> int:
    v, c = np.unique(grid, return_counts=True)
    return int(v[int(np.argmax(c))])


def _extract_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    target: Tuple[int, int],
    walkable: np.ndarray,
) -> list[Tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    tr, tc = target
    parent: Dict[Tuple[int, int], Tuple[int, int] | None] = {(sr, sc): None}
    q: Deque[Tuple[int, int]] = deque([(sr, sc)])
    found = False

    while q:
        r, c = q.popleft()
        if (r, c) == (tr, tc):
            found = True
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in parent and walkable[nr, nc]:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    if not found:
        return []

    path = [(tr, tc)]
    cur = (tr, tc)
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = _bg(out)
    if h < 4 or w < 4:
        return out

    template = out[:3, :3]
    orange = (template == 7)

    # Step 2: 3x3 template tiling across main region
    for r in range(4, h):
        for c in range(4, w):
            tr = (r - 4) % 3
            tc = (c - 4) % 3
            if orange[tr, tc]:
                out[r, c] = 7

    # Step 3: obstacle + 4-neighbor exclusion area
    blocked = out == 7
    blocked2 = blocked.copy()
    for r in range(h):
        for c in range(w):
            if not blocked[r, c]:
                continue
            blocked2[r, c] = True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked2[nr, nc] = True
    out[blocked2 & (out == 8)] = bg

    # Step 4: unique safe path between endpoints in remaining track
    endpoints = list(map(tuple, np.argwhere(out == 9)))
    if len(endpoints) < 2:
        return out
    s, t = endpoints[0], endpoints[1]
    walkable = (out == 8) | (out == 9)
    path = _extract_path(out, s, t, walkable)
    if not path:
        return out

    path_set = set(path)
    recolor = np.zeros_like(out, dtype=bool)
    for r, c in path_set:
        if out[r, c] == 8:
            recolor[r, c] = True
    out[path_set] = np.where(out[path_set] == 8, 3, out[path_set])
    out[out == 8] = bg
    return out


_RNG = np.random.default_rng()


def generate_input() -> np.ndarray:
    n = 15
    g = np.zeros((n, n), dtype=int)
    colors = [1, 2, 3, 4, 8]

    # Template in 3x3 top-left region (2~3 oranges)
    orange_pos = {(1, 1), (2, 2), (0, 2)}
    if int(_RNG.integers(0, 2)) == 0:
        orange_pos = {(0, 0), (2, 1), (1, 2)}
    for r, c in orange_pos:
        g[r, c] = 7

    # Main region is dense 8-track to ensure abundant pre-tiling paths
    g[4:, 4:] = 8
    # two endpoints
    g[4 + 1, 4 + 1] = 9
    g[n - 2, n - 2] = 9
    # add noise-free dense mosaic colors not necessary in main region
    for r in range(n):
        for c in range(n):
            if g[r, c] != 0:
                continue
            if r < 4 and c < 4:
                continue
            g[r, c] = int(_RNG.choice(colors))

    return g
