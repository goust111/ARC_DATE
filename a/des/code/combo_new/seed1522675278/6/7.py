import random
from collections import deque

import numpy as np


BG = 0


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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    segs = {}
    marks = []
    colors = [c for c in np.unique(grid) if c != BG]
    for color in colors:
        comps = _components(grid == color)
        for comp in comps:
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(comp) >= 3 and (len(rows) == 1 or len(cols) == 1):
                segs[color] = comp
            elif len(comp) == 1:
                marks.append((comp[0], color))
    endpoints = {}
    for color, comp in segs.items():
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if len(set(rows)) == 1:
            pts = sorted(comp, key=lambda p: p[1])
        else:
            pts = sorted(comp, key=lambda p: p[0])
        endpoints[color] = [pts[0], pts[-1]]
    links = {}
    starts = []
    for color, ends in endpoints.items():
        target_colors = []
        for e in ends:
            found = color
            for (r, c), mc in marks:
                if abs(r - e[0]) + abs(c - e[1]) == 1:
                    found = mc
            target_colors.append(found)
        if color in target_colors:
            starts.append(color)
        nxt = [x for x in target_colors if x != color]
        links[color] = nxt[0] if nxt else None
    if not segs:
        return np.zeros((3, 3), dtype=int)
    order = [starts[0] if starts else next(iter(segs))]
    while links.get(order[-1]) in segs and links[order[-1]] not in order:
        order.append(links[order[-1]])
    lengths = [len(segs[c]) for c in order]
    out = np.zeros((3, sum(lengths) + 2), dtype=int)
    x = 1
    for color in order:
        ln = len(segs[color])
        out[1, x : x + ln] = color
        x += ln
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 22)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    n = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    prev = None
    x = 1
    for i, color in enumerate(colors):
        ln = random.randint(3, 6)
        r = random.randint(2, h - 3)
        c = x
        if c + ln + 2 >= w:
            c = 1
            x = 1
        grid[r, c : c + ln] = color
        left_mark = color if i == 0 else prev
        right_mark = color if i == n - 1 else colors[i + 1]
        grid[r - 1, c] = left_mark
        grid[r + 1, c + ln - 1] = right_mark
        prev = color
        x += ln + 3
    return grid
