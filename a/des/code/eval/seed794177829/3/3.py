import heapq
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _pipe_width_score(mask, r, c):
    h, w = mask.shape
    row_run = 1
    cc = c - 1
    while cc >= 0 and mask[r, cc]:
        row_run += 1
        cc -= 1
    cc = c + 1
    while cc < w and mask[r, cc]:
        row_run += 1
        cc += 1
    col_run = 1
    rr = r - 1
    while rr >= 0 and mask[rr, c]:
        col_run += 1
        rr -= 1
    rr = r + 1
    while rr < h and mask[rr, c]:
        col_run += 1
        rr += 1
    return min(3, max(1, min(row_run, col_run)))


def _sources(grid, bg):
    h, w = grid.shape
    src = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col in (bg, GREY):
                continue
            adj = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                    adj.append((nr, nc))
            if adj:
                src.append((r, c, col, adj))
    return src


def main(grid: np.ndarray) -> np.ndarray:
    bg = _background(grid)
    pipe = grid == GREY
    out = grid.copy()
    best = {}
    heap = []
    seq = 0
    for sr, sc, color, adj in _sources(grid, bg):
        for r, c in adj:
            width = _pipe_width_score(pipe, r, c)
            key = (width, 0, color)
            if key > best.get((r, c), (-1, -10**9, -1)):
                best[(r, c)] = key
                heapq.heappush(heap, (-width, 0, seq, r, c, color, width))
                seq += 1
    while heap:
        negw, dist, _seq, r, c, color, max_width = heapq.heappop(heap)
        if best.get((r, c)) != (max_width, -dist, color):
            continue
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or not pipe[nr, nc]:
                continue
            nw = max(max_width, _pipe_width_score(pipe, nr, nc))
            nd = dist + 1
            key = (nw, -nd, color)
            if key > best.get((nr, nc), (-1, -10**9, -1)):
                best[(nr, nc)] = key
                heapq.heappush(heap, (-nw, nd, seq, nr, nc, color, nw))
                seq += 1
    for (r, c), (_w, _d, color) in best.items():
        out[r, c] = color
    return out


def _clear(grid, r0, c0, r1, c1, pad=1):
    h, w = grid.shape
    if not (1 <= r0 <= r1 < h - 1 and 1 <= c0 <= c1 < w - 1):
        return False
    rr0, rr1 = max(0, r0 - pad), min(h - 1, r1 + pad)
    cc0, cc1 = max(0, c0 - pad), min(w - 1, c1 + pad)
    return np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BG)


def _draw_pipe(grid, color):
    h, w = grid.shape
    for _ in range(200):
        orient = random.choice(["h", "v"])
        thick = random.randint(1, 3)
        if orient == "h":
            max_len = min(16, w - 8)
            if max_len < 7:
                continue
            length = random.randint(7, max_len)
            r0 = random.randint(3, h - thick - 4)
            c0 = random.randint(3, w - length - 4)
            r1, c1 = r0 + thick - 1, c0 + length - 1
            src = (r0 + thick // 2, c0 - 1)
        else:
            max_len = min(16, h - 8)
            if max_len < 7:
                continue
            length = random.randint(7, max_len)
            r0 = random.randint(3, h - length - 4)
            c0 = random.randint(3, w - thick - 4)
            r1, c1 = r0 + length - 1, c0 + thick - 1
            src = (r0 - 1, c0 + thick // 2)
        if _clear(grid, r0, c0, r1, c1) and int(grid[src]) == BG:
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
            grid[src] = color
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(20, 32)
        w = random.randint(20, 32)
        grid = np.zeros((h, w), dtype=int)
        for color in random.sample(COLORS, random.randint(2, 4)):
            if not _draw_pipe(grid, color):
                break
        else:
            if np.any(grid == GREY):
                return grid
    raise RuntimeError("failed to generate seed794177829/3/3 input")
