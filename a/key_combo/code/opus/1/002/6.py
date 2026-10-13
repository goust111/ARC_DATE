import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
CYAN = 8


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _recolor_mirror(mirror, contact_left):
    h, w = mirror.shape
    out = np.zeros_like(mirror)
    for r in range(h):
        for c in range(w):
            d = (w - 1 - c) if contact_left else c
            val = int(mirror[r, c])
            if d % 2 == 0:
                out[r, c] = CYAN if val == RED else BG
            else:
                out[r, c] = BLUE if val == RED else GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the red shape across the touched vertical boundary and recolor the
    mirrored half by alternating distance from the reflection seam.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    contact_left = np.any(g[:, 0] == RED)
    contact_right = np.any(g[:, w - 1] == RED)
    assert contact_left ^ contact_right, "shape must touch exactly one vertical boundary"
    mirror = np.fliplr(g)
    recolored = _recolor_mirror(mirror, contact_left)
    if contact_left:
        return np.concatenate([recolored, g.copy()], axis=1)
    return np.concatenate([g.copy(), recolored], axis=1)


def _grow_shape(h, w, left):
    cells = {(random.randint(1, h - 2), 0 if left else w - 1)}
    target = random.randint(6, 15)
    limit_cols = range(0, min(w, 6)) if left else range(max(0, w - 6), w)
    limit = {(r, c) for r in range(0, h) for c in limit_cols}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in limit and 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    if left and nc == w - 1:
                        continue
                    if not left and nc == 0:
                        continue
                    nbrs.append((nr, nc))
        if nbrs:
            cells.add(random.choice(nbrs))
        if len(cells) < target and len(cells) > h * 6:
            break
    cols = [c for _, c in cells]
    if max(cols) - min(cols) < 2:
        return None
    return cells


def _gen_attempt():
    h = random.randint(8, 14)
    w = random.randint(8, 15)
    left = random.random() < 0.5
    cells = _grow_shape(h, w, left)
    if cells is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    for r, c in cells:
        grid[r, c] = RED
    if len(_components(grid == RED)) != 1:
        return None
    if np.any(grid[:, 0] == RED) and np.any(grid[:, w - 1] == RED):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
