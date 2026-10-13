import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
FRAME_ORDER = (1, 2, 3, 4)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))



def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dr, dc in DIRS4:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _edge_color(arr: np.ndarray, default: int) -> int:
    vals = [int(v) for v in np.unique(arr) if v != BG]
    return vals[0] if vals else default


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    top_color = _edge_color(g[0, 1:w - 1], FRAME_ORDER[0])
    right_color = _edge_color(g[1:h - 1, w - 1], FRAME_ORDER[1])
    bottom_color = _edge_color(g[h - 1, 1:w - 1], FRAME_ORDER[2])
    left_color = _edge_color(g[1:h - 1, 0], FRAME_ORDER[3])
    palette = [top_color, right_color, bottom_color, left_color]

    inner_mask = np.zeros_like(g, dtype=bool)
    inner_mask[1 : h - 1, 1 : w - 1] = g[1 : h - 1, 1 : w - 1] == GRAY
    bars = _components(inner_mask)
    bars = sorted(bars, key=len, reverse=True)

    out = g.copy()
    for comp in bars:
        for r, c in comp:
            out[r, c] = BG

    for rank, comp in enumerate(bars[:4]):
        color = palette[rank % 4]
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[rank % 4]
        for r, c in comp:
            rr, cc = r + dr, c + dc
            if 0 < rr < h - 1 and 0 < cc < w - 1:
                out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    h, w = 13, 13
    for _ in range(2000):
        g = np.zeros((h, w), dtype=int)
        top_color, right_color, bottom_color, left_color = FRAME_ORDER

        g[0, 1 : w - 1] = top_color
        g[h - 1, 1 : w - 1] = bottom_color
        g[1 : h - 1, 0] = left_color
        g[1 : h - 1, w - 1] = right_color

        heights = random.sample(range(2, 7), 4)
        cols = random.sample(range(2, w - 2), 4)
        ok = True
        for col, height in zip(cols, heights):
            top = random.randint(2, h - 3 - height + 1)
            if top <= 1 or top + height - 1 >= h - 1:
                ok = False
                break
            g[top : top + height, col] = GRAY

        if ok:
            return g
    # pragma: no cover
    return g
