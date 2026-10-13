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
    rings = []
    for color in [c for c in np.unique(grid) if c != BG]:
        comps = _components(grid == color)
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if len(comp) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            pts = np.argwhere(inner != BG)
            if pts.size == 0:
                mark = color
            else:
                mark = int(inner[pts[0][0], pts[0][1]])
            rings.append({"color": color, "mark": mark, "size": (r1 - r0 + 1, c1 - c0 + 1)})
    by_color = {x["color"]: x for x in rings}
    pointed = {x["mark"] for x in rings if x["mark"] in by_color and x["mark"] != x["color"]}
    chain = []
    cur = next(x for x in rings if x["color"] not in pointed)
    while True:
        chain.append(cur)
        nxt = [x for x in rings if cur["mark"] == x["color"] and x["color"] != cur["color"]]
        if not nxt:
            break
        cur = nxt[0]
    h = max(x["size"][0] for x in chain)
    w = max(x["size"][1] for x in chain)
    out = np.zeros((h, w), dtype=int)
    for i, ring in enumerate(chain):
        rh, rw = ring["size"]
        top = (h - rh) // 2
        left = (w - rw) // 2
        out[top : top + rh, left] = ring["color"]
        out[top : top + rh, left + rw - 1] = ring["color"]
        out[top, left : left + rw] = ring["color"]
        out[top + rh - 1, left : left + rw] = ring["color"]
        if i == len(chain) - 1:
            out[top + rh // 2, left + rw // 2] = ring["mark"]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    count = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], count)
    marks = colors[1:] + [colors[-1]]
    sizes = sorted([(random.randrange(5, 11, 2), random.randrange(5, 11, 2)) for _ in range(count)], key=lambda x: x[0] * x[1], reverse=True)
    placed = []
    for (color, mark), (rh, rw) in zip(zip(colors, marks), sizes):
        for _ in range(100):
            top = random.randint(0, h - rh)
            left = random.randint(0, w - rw)
            if all(top + rh < t or t + hh < top or left + rw < l or l + ww < left for t, l, hh, ww in placed):
                placed.append((top, left, rh, rw))
                break
        top, left, rh, rw = placed[-1]
        grid[top : top + rh, left] = color
        grid[top : top + rh, left + rw - 1] = color
        grid[top, left : left + rw] = color
        grid[top + rh - 1, left : left + rw] = color
        grid[top + rh // 2, left + rw // 2] = mark
    return grid
