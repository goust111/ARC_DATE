import random
from collections import deque
import numpy as np


def _bresenham(r0, c0, r1, c1):
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            return points
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def _external_zero(arr):
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    pairs = []
    for color in sorted(set(int(v) for v in grid.ravel()) - {0}):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if len(pts) == 2:
            (a, b) = pts
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            top = min(a, b)
            pairs.append((dist, top[0], top[1], color, a, b))
    pairs.sort()
    order = {color: i for i, (*_, color, _a, _b) in enumerate(pairs)}
    out = np.zeros_like(grid)
    for _dist, _tr, _tc, color, a, b in pairs:
        for r, c in _bresenham(a[0], a[1], b[0], b[1]):
            out[r, c] = color
    exterior = _external_zero(out)
    for comp in _components((out == 0) & (~exterior)):
        counts = {}
        boundary = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] != 0:
                    boundary.add((nr, nc))
        for r, c in boundary:
            counts[int(out[r, c])] = counts.get(int(out[r, c]), 0) + 1
        color = max(counts, key=lambda c: (counts[c], order.get(c, -1)))
        for r, c in comp:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((14, 14), dtype=int)
    top = random.randint(2, 4)
    bottom = random.randint(9, 11)
    left = random.randint(2, 4)
    right = random.randint(9, 11)
    colors = random.sample([1, 3, 4, 5, 8], 5)
    anchors = {
        colors[0]: [(top, left - 1), (top, right + 1)],
        colors[1]: [(bottom, left - 1), (bottom, right + 1)],
        colors[2]: [(top - 1, left), (bottom + 1, left)],
        colors[3]: [(top - 1, right), (bottom + 1, right)],
    }
    if random.random() < 0.6:
        anchors[colors[4]] = [(top - 1, left + 1), (bottom + 1, right - 1)]
    for color, pts in anchors.items():
        for r, c in pts:
            grid[r, c] = color
    used = set(anchors)
    for color in random.sample([2, 6, 7, 9], random.randint(1, 3)):
        for _ in range(50):
            r, c = random.randint(1, 12), random.randint(1, 12)
            if grid[r, c] == 0:
                grid[r, c] = color
                break
    return grid
