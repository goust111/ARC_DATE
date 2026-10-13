import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mask(cells):
    r1, c1, r2, c2 = _bbox(cells)
    m = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
    for r, c in cells:
        m[r - r1, c - c1] = True
    return m, (r1, c1, r2, c2)


def _outside_black(arr):
    h, w = arr.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == 0 and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    split = max(range(h), key=lambda r: int(np.sum(grid[r] == 5)))
    guides = []
    magenta = grid == 6
    for r in range(split + 1, h - 1):
        for c in range(1, w - 1):
            if all(magenta[r + dr, c + dc] for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                guides.append((r, c))

    lower = np.zeros_like(grid, dtype=bool)
    lower[split + 1:, :] = True
    sprite_mask = lower & (grid != 0) & (grid != 5) & (grid != 6)
    sprites = []
    for comp in _components(sprite_mask):
        color = int(grid[comp[0]])
        m, box = _mask(comp)
        br, bc = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if guides:
            guide = min(guides, key=lambda g: abs(g[0] - br) + abs(g[1] - bc))
        else:
            guide = (split + 2, int(bc))
        sprites.append((color, m, guide[1]))

    out = np.zeros_like(grid)
    for color, m, gc in sorted(sprites, key=lambda x: x[0]):
        mh, mw = m.shape
        r0 = split - mh
        c0 = int(gc - (mw - 1) // 2)
        for rr in range(mh):
            for cc in range(mw):
                r, c = r0 + rr, c0 + cc
                if m[rr, cc] and 0 <= r < split and 0 <= c < w:
                    out[r, c] = max(out[r, c], color)

    foreground = out != 0
    outside = _outside_black(out)
    out[(out == 0) & (~outside)] = 4
    outline = np.zeros_like(out, dtype=bool)
    for r in range(h):
        for c in range(w):
            if out[r, c] != 0 or not outside[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and foreground[nr, nc]:
                        outline[r, c] = True
    out[outline] = 3
    return out


def generate_input() -> np.ndarray:
    h = random.randint(19, 23)
    w = random.randint(12, 16)
    split = random.randint(11, min(14, h - 7))
    grid = np.zeros((h, w), dtype=int)
    grid[split, :] = 5
    a = np.array([[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 1, 1, 1]], dtype=bool)
    b = np.array([[1, 1],
                  [0, 1],
                  [1, 1]], dtype=bool)
    c_left = random.randint(2, w - 7)
    sprites = [(2, a, c_left + 1), (7, b, c_left + 3)]
    if random.random() < 0.6:
        c = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool)
        sprites.append((1, c, random.choice([max(2, c_left), min(w - 3, c_left + 5)])))
    lower_cols = [1, max(4, w // 2), max(7, w - 4)]
    for idx, (color, m, gc) in enumerate(sprites):
        r0 = h - m.shape[0] - 1
        c0 = lower_cols[idx]
        grid[r0:r0 + m.shape[0], c0:c0 + m.shape[1]][m] = color
        gr, gcol = split + 2, gc
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            grid[gr + dr, gcol + dc] = 6
    return grid
