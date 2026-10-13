import random
from collections import defaultdict, deque

import numpy as np


BG = 0

SHAPES = {
    "U": np.array([[1, 0, 1], [1, 1, 1]]),
    "D": np.array([[1, 1, 1], [1, 0, 1]]),
    "L": np.array([[1, 1], [0, 1], [1, 1]]),
    "R": np.array([[1, 1], [1, 0], [1, 1]]),
}
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def _classify(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=int)
    for r, c in comp:
        mask[r - r0, c - c0] = 1
    for d, pat in SHAPES.items():
        if mask.shape == pat.shape and np.array_equal(mask, pat):
            return d, (r0, max(rows), c0, max(cols))
    return "R", (r0, max(rows), c0, max(cols))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    by_color = defaultdict(list)
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            by_color[color].append(comp)
    for color, comps in by_color.items():
        if len(comps) != 2:
            continue
        comps = sorted(comps, key=lambda comp: min(r for r, _ in comp) + min(c for _, c in comp))
        d1, b1 = _classify(comps[0])
        d2, b2 = _classify(comps[1])
        s1 = ((b1[0] + b1[1]) // 2, (b1[2] + b1[3]) // 2)
        s2 = ((b2[0] + b2[1]) // 2, (b2[2] + b2[3]) // 2)
        v1 = DIRS[d1]
        v2 = DIRS[d2]
        p1 = (s1[0] + v1[0], s1[1] + v1[1])
        p2 = (s2[0] + v2[0], s2[1] + v2[1])
        if v1[0] == 0:
            elbow = (p2[0], p1[1])
        else:
            elbow = (p1[0], p2[1])
        path = []
        r, c = p1
        while (r, c) != elbow:
            path.append((r, c))
            r += 0 if elbow[0] == r else (1 if elbow[0] > r else -1)
            c += 0 if elbow[1] == c else (1 if elbow[1] > c else -1)
        while (r, c) != p2:
            path.append((r, c))
            r += 0 if p2[0] == r else (1 if p2[0] > r else -1)
            c += 0 if p2[1] == c else (1 if p2[1] > c else -1)
        for rr, cc in path:
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BG:
                out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    dirs = [("R", "U"), ("R", "D"), ("L", "U"), ("L", "D")]
    for color in colors:
        d1, d2 = random.choice(dirs)
        pat1, pat2 = SHAPES[d1], SHAPES[d2]
        r1 = random.randint(1, h // 2 - pat1.shape[0] - 1)
        c1 = random.randint(1, w // 2 - pat1.shape[1] - 1)
        r2 = random.randint(h // 2, h - pat2.shape[0] - 2)
        c2 = random.randint(w // 2, w - pat2.shape[1] - 2)
        for rr in range(pat1.shape[0]):
            for cc in range(pat1.shape[1]):
                if pat1[rr, cc]:
                    grid[r1 + rr, c1 + cc] = color
        for rr in range(pat2.shape[0]):
            for cc in range(pat2.shape[1]):
                if pat2[rr, cc]:
                    grid[r2 + rr, c2 + cc] = color
    return grid
