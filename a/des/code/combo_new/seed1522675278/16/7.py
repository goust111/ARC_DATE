import random
from collections import defaultdict, deque

import numpy as np


BG = 0


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = defaultdict(list)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps[color].append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    for color, comps in _components_by_color(grid).items():
        anchor = next(c for c in comps if len(c) == 1)
        obj = max((c for c in comps if len(c) > 1), key=len)
        ar, ac = anchor[0]
        rows = [r for r, _ in obj]
        cols = [c for _, c in obj]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        coords = []
        for r, c in sorted(obj):
            bbox[r - r0, c - c0] = color
            coords.append((r - r0, c - c0))
        if ar >= r0 and ar <= r1:
            if ac < c0:
                s = c0 - ac - 1
            else:
                s = ac - c1 - 1
        else:
            if ar < r0:
                s = r0 - ar - 1
            else:
                s = ar - r1 - 1
        coords = coords[: max(1, min(s, len(coords)))]
        top = ar - bbox.shape[0] // 2
        left = ac - bbox.shape[1] // 2
        for rr, cc in coords:
            nr, nc = top + rr, left + cc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color
    return out


def _random_mask():
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    cells = {(random.randrange(h), random.randrange(w))}
    target = random.randint(4, min(8, h * w))
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    arr = np.zeros((h, w), dtype=int)
    for r, c in cells:
        arr[r, c] = 1
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    return arr[rows][:, cols]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    for color in colors:
        mask = _random_mask()
        mh, mw = mask.shape
        area = int(mask.sum())
        for _ in range(200):
            top = random.randint(2, h - mh - 3)
            left = random.randint(2, w - mw - 3)
            if occupied[top - 1 : top + mh + 1, left - 1 : left + mw + 1].any():
                continue
            horizontal = random.random() < 0.5
            if horizontal:
                anchor_r = top + random.randrange(mh)
                dist = random.randint(1, area)
                if random.random() < 0.5:
                    anchor_c = left - dist - 1
                else:
                    anchor_c = left + mw + dist
            else:
                anchor_c = left + random.randrange(mw)
                dist = random.randint(1, area)
                if random.random() < 0.5:
                    anchor_r = top - dist - 1
                else:
                    anchor_r = top + mh + dist
            if not (0 <= anchor_r < h and 0 <= anchor_c < w):
                continue
            if occupied[anchor_r, anchor_c]:
                continue
            grid[top : top + mh, left : left + mw][mask == 1] = color
            grid[anchor_r, anchor_c] = color
            occupied[top : top + mh, left : left + mw] |= mask == 1
            occupied[anchor_r, anchor_c] = True
            break
    return grid
