import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
BASE_KEY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def _rot_key(color, k):
    return np.where(np.rot90(BASE_KEY, k) == 1, color, BG)


def _key_orientation(mask):
    for k in range(4):
        if mask.shape == np.rot90(BASE_KEY, k).shape and np.array_equal(mask > 0, np.rot90(BASE_KEY, k) > 0):
            return k
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    frame_comps = _components(grid == GREY)
    key_by_color = {}
    containers = []
    for comp in frame_comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 >= 4 and c1 - c0 >= 4:
            containers.append((r0, r1, c0, c1))
    for color in [v for v in np.unique(grid) if v not in (BG, GREY)]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if any(r0 < min(rows) < r1 and c0 < min(cols) < c1 and max(rows) < r1 and max(cols) < c1 for r0, r1, c0, c1 in containers):
                continue
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            key_by_color[color] = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            for r, c in comp:
                out[r, c] = BG
    dir_to_k = {"U": 0, "L": 1, "D": 2, "R": 3}
    for r0, r1, c0, c1 in containers:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        vals = [int(v) for v in np.unique(inner) if v not in (BG, GREY)]
        if not vals:
            continue
        color = max(vals, key=lambda v: np.count_nonzero(inner == v))
        pts = np.argwhere(inner == color)
        center = np.array([(inner.shape[0] - 1) / 2.0, (inner.shape[1] - 1) / 2.0])
        if len(set(pts[:, 0])) == 1:
            direction = "L" if pts[:, 1].mean() < center[1] else "R"
        else:
            direction = "U" if pts[:, 0].mean() < center[0] else "D"
        key = key_by_color[color]
        k0 = _key_orientation(key)
        k1 = dir_to_k[direction]
        rot = np.rot90(key, k1 - k0)
        ih, iw = inner.shape
        kh, kw = rot.shape
        top = r0 + 1 + (ih - kh) // 2
        left = c0 + 1 + (iw - kw) // 2
        patch = out[top : top + kh, left : left + kw]
        out[top : top + kh, left : left + kw] = np.where(rot != BG, rot, patch)
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    positions = [(2, 2), (2, 18), (17, 2)]
    dirs = ["U", "R", "D", "L"]
    for color, pos in zip(colors, positions):
        k = random.randrange(4)
        key = _rot_key(color, k)
        r, c = pos
        grid[r : r + key.shape[0], c : c + key.shape[1]] = key
    for idx, color in enumerate(colors):
        r0 = 8 + idx * 5
        c0 = 9
        h = 6
        w = 8
        grid[r0, c0 : c0 + w] = GREY
        grid[r0 + h - 1, c0 : c0 + w] = GREY
        grid[r0 : r0 + h, c0] = GREY
        grid[r0 : r0 + h, c0 + w - 1] = GREY
        inner = (r0 + 1, r0 + h - 2, c0 + 1, c0 + w - 2)
        direction = random.choice(dirs)
        if direction in ("L", "R"):
            row = (inner[0] + inner[1]) // 2
            if direction == "L":
                cols = [inner[2], inner[2] + 1]
            else:
                cols = [inner[3] - 1, inner[3]]
            for c in cols:
                grid[row, c] = color
        else:
            col = (inner[2] + inner[3]) // 2
            if direction == "U":
                rows = [inner[0], inner[0] + 1]
            else:
                rows = [inner[1] - 1, inner[1]]
            for r in rows:
                grid[r, col] = color
    return grid
