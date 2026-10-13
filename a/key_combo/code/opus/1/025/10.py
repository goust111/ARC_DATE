import random
import numpy as np


def _bars(grid):
    h, w = grid.shape
    bars = {}
    for c in range(w):
        if grid[h - 1, c] not in (1, 2):
            continue
        color = int(grid[h - 1, c])
        r = h - 1
        while r >= 0 and grid[r, c] == color:
            r -= 1
        bars[c] = {"col": c, "top": r + 1, "height": h - 1 - r, "color": color}
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    bars = _bars(grid)
    rotate = []
    used = set()
    for c, bar in sorted(bars.items()):
        if bar["color"] != 1:
            continue
        cur = bar
        while True:
            if cur["col"] in used:
                break
            rotate.append(cur)
            used.add(cur["col"])
            nxt = cur["col"] + cur["height"]
            if nxt in bars and bars[nxt]["color"] == 2:
                cur = bars[nxt]
            else:
                break
    segments = []
    for bar in rotate:
        c = bar["col"]
        out[h - bar["height"]:h, c] = 0
    for idx, bar in enumerate(rotate):
        r = h - 1
        c0 = bar["col"]
        c1 = min(w - 1, c0 + bar["height"] - 1)
        out[r, c0:c1 + 1] = bar["color"]
        segments.append({"id": idx, "row": r, "c0": c0, "c1": c1, "color": bar["color"]})
    grey_rows = np.argwhere(grid == 5)
    if len(grey_rows):
        gr = int(np.bincount(grey_rows[:, 0]).argmax())
        pairs = {}
        for col in range(w):
            hits = [s for s in segments if s["c0"] <= col <= s["c1"]]
            above = [s for s in hits if s["row"] < gr]
            below = [s for s in hits if s["row"] > gr]
            if len(above) == 1 and len(below) == 1:
                a, b = above[0]["id"], below[0]["id"]
                pairs.setdefault(a, b)
                pairs.setdefault(b, a)
        swapped = set()
        for a, b in list(pairs.items()):
            if a in swapped or b in swapped:
                continue
            ca, cb = segments[a]["color"], segments[b]["color"]
            segments[a]["color"], segments[b]["color"] = cb, ca
            swapped.update((a, b))
        for s in segments:
            out[s["row"], s["c0"]:s["c1"] + 1] = s["color"]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 18), random.randint(12, 16)
    g = np.zeros((h, w), dtype=int)
    gr = random.randint(5, h - 5)
    g[gr, :] = 5
    candidates = list(range(1, w - 3))
    cols = sorted(random.sample(candidates, random.randint(5, min(7, len(candidates)))))
    heights = [random.randint(2, 5) for _ in cols]
    start = cols[0]
    g[h - heights[0]:h, start] = 1
    next_col = start + heights[0]
    if next_col < w:
        g[h - random.randint(2, 5):h, next_col] = 2
    for c, ht in zip(cols[1:], heights[1:]):
        if c != next_col:
            g[h - ht:h, c] = 2
    if random.random() < 0.5 and len(cols) > 3:
        c = cols[2]
        g[h - heights[2]:h, c] = 1
        nc = c + heights[2]
        if nc < w:
            g[h - random.randint(2, 5):h, nc] = 2
    return g
