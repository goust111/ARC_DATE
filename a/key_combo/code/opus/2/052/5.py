from __future__ import annotations

from collections import Counter, deque

import numpy as np

BLACK = 0
GRAY = 5
YELLOW = 4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
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
            comps.append(comp)
    return comps


def _best_axis(grid: np.ndarray, color: int) -> int:
    h, w = grid.shape
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return w // 2
    best_axis = 0
    best_score = -1
    for axis in range(w):
        score = 0
        for r, c in coords:
            mc = 2 * axis - int(c)
            if 0 <= mc < w and grid[r, mc] == color:
                score += 1
        if score > best_score:
            best_score = score
            best_axis = axis
    return int(best_axis)


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    colors = sorted({int(x) for x in out.flatten() if int(x) != BLACK})

    # per-color symmetry completion (erase unmatched asymmetric noise)
    for color in colors:
        axis = _best_axis(out, color)
        coords = np.argwhere(out == color)
        keep = np.zeros(len(coords), dtype=bool)
        for i, (r, c) in enumerate(coords):
            mc = 2 * axis - int(c)
            if 0 <= mc < w and out[r, mc] == color:
                keep[i] = True
        for i, (r, c) in enumerate(coords):
            if not keep[i]:
                out[int(r), int(c)] = BLACK

    fg_mask = out != BLACK
    comps = _components(fg_mask)
    comp_id = np.full(out.shape, -1, dtype=int)
    for i, comp in enumerate(comps):
        for r, c in comp:
            comp_id[r, c] = i

    # black components
    black = out == BLACK
    black_comps = _components(black)

    ext = np.zeros_like(black, dtype=bool)
    q = deque()
    # seed boundary black pixels
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
        x, y = q.pop()
        for nx, ny in _neighbors4(x, y):
            if 0 <= nx < h and 0 <= ny < w and black[nx, ny] and not ext[nx, ny]:
                ext[nx, ny] = True
                q.append((nx, ny))

    hole_cells: dict[int, list[tuple[int, int]]] = {}
    for comp in black_comps:
        if not comp:
            continue
        touches_border = False
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if touches_border:
            continue
        owners = set()
        for r, c in comp:
            for nx, ny in _neighbors8(r, c):
                if 0 <= nx < h and 0 <= ny < w and comp_id[nx, ny] >= 0:
                    owners.add(int(comp_id[nx, ny]))
        if len(owners) == 1:
            owner = next(iter(owners))
            hole_cells.setdefault(owner, []).extend(comp)

    outer_mark = np.zeros_like(out, dtype=bool)
    inner_mark = np.zeros_like(out, dtype=bool)
    hole_set_cache = {k: set(v) for k, v in hole_cells.items()}

    for i, comp in enumerate(comps):
        if i not in hole_set_cache:
            continue
        hset = hole_set_cache[i]
        for r, c in comp:
            for nx, ny in _neighbors8(r, c):
                if 0 <= nx < h and 0 <= ny < w and out[nx, ny] == BLACK:
                    if (nx, ny) in hset:
                        inner_mark[nx, ny] = True
                    elif ext[nx, ny]:
                        outer_mark[nx, ny] = True

    for r in range(h):
        for c in range(w):
            if inner_mark[r, c]:
                out[r, c] = YELLOW
            elif outer_mark[r, c]:
                out[r, c] = GRAY

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 15
    w = 17
    out = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3]

    for color in colors:
        axis = int(rng.integers(4, w - 4))
        n_points = int(rng.integers(6, 10))
        for _ in range(n_points):
            r = int(rng.integers(2, h - 2))
            left = int(rng.integers(1, axis))
            right = 2 * axis - left
            if 0 <= right < w:
                out[r, left] = color
                out[r, right] = color

        # add one symmetric cavity-like object for topological variety
        top = int(rng.integers(3, h - 5))
        for rr in range(top, top + 3):
            out[rr, max(2, axis - 2)] = color
            out[rr, min(w - 3, axis + 2)] = color
        out[top + 1, axis] = color

    # intentionally asymmetric noise
    for color in colors:
        noise = int(rng.integers(2, 6))
        for _ in range(noise):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            out[r, c] = color

    return out

