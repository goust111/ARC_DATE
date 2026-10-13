import random
from collections import deque
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid, colors=None):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            if colors is not None and int(grid[r, c]) not in colors:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(grid[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append({"color": color, "cells": cells})
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    out = g.copy()
    segments = []
    for comp in _components(g):
        rows = [p[0] for p in comp["cells"]]
        cols = [p[1] for p in comp["cells"]]
        if len(comp["cells"]) > 1 and len(set(cols)) == 1 and max(rows) - min(rows) + 1 == len(comp["cells"]):
            segments.append({
                "color": comp["color"],
                "col": cols[0],
                "start": min(rows),
                "end": max(rows),
                "length": len(comp["cells"]),
                "cells": comp["cells"],
            })
    if not segments:
        return out
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = 0
    segments.sort(key=lambda s: s["col"])
    ref = segments[len(segments) // 2]
    start, length = ref["start"], ref["length"]
    end = start + length - 1
    h, _ = g.shape
    for seg in segments:
        for r in range(start, min(end + 1, h)):
            out[r, seg["col"]] = seg["color"]
    for left, right in zip(segments, segments[1:]):
        fill = left["color"] if left["length"] >= right["length"] else right["color"]
        for r in range(start, min(end + 1, h)):
            for c in range(left["col"] + 1, right["col"]):
                out[r, c] = fill
    return out


def generate_input() -> np.ndarray:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    g = np.zeros((h, w), dtype=int)
    possible = list(range(1, w - 1, 2))
    count = 7 if len(possible) >= 7 and random.choice([True, False]) else 5
    cols = sorted(random.sample(possible, count))
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], count)
    for col, color in zip(cols, colors):
        length = random.randint(3, min(10, h - 2))
        start = random.randint(0, h - length)
        g[start:start + length, col] = color
    return g
