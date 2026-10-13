import random
from collections import deque

import numpy as np


DIRS = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _gray_blocks(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    blocks = []
    cell_to = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 5 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 5 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            idx = len(blocks)
            blocks.append(cells)
            for cell in cells:
                cell_to[cell] = idx
    return blocks, cell_to


def _decode_arrow(coords):
    pts = sorted(coords)
    by_row = {}
    by_col = {}
    for r, c in pts:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    for r, cols in by_row.items():
        cols = sorted(cols)
        if len(cols) >= 4 and cols[-1] - cols[0] == 3:
            cross_col = max(by_col, key=lambda c: len(by_col[c]))
            mean = (cols[0] + cols[-1]) / 2
            if cross_col < mean:
                return (r, cols[0]), DIRS["L"]
            return (r, cols[-1]), DIRS["R"]
    for c, rows in by_col.items():
        rows = sorted(rows)
        if len(rows) >= 4 and rows[-1] - rows[0] == 3:
            cross_row = max(by_row, key=lambda r: len(by_row[r]))
            mean = (rows[0] + rows[-1]) / 2
            if cross_row < mean:
                return (rows[0], c), DIRS["U"]
            return (rows[-1], c), DIRS["D"]
    return pts[0], DIRS["R"]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    h, w = grid.shape
    blocks, cell_to_block = _gray_blocks(grid)
    arrows = []
    for color in sorted(int(x) for x in np.unique(grid) if x not in (0, 2, 5)):
        coords = list(map(tuple, np.argwhere(grid == color)))
        if len(coords) < 6:
            continue
        tip, d = _decode_arrow(coords)
        top_left = min(coords)
        arrows.append({"color": color, "pos": tip, "dir": d, "origin": top_left, "active": True})
    occupied = {}
    hits = {}
    for step in range(1, h * w + 1):
        proposals = {}
        for i, ray in enumerate(arrows):
            if not ray["active"]:
                continue
            r, c = ray["pos"]
            dr, dc = ray["dir"]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                ray["active"] = False
                continue
            if (nr, nc) in cell_to_block:
                bi = cell_to_block[(nr, nc)]
                hits.setdefault(bi, []).append((step, i))
                ray["active"] = False
                continue
            if (nr, nc) in occupied:
                if grid[r, c] == 0:
                    out[r, c] = 2
                ray["active"] = False
                continue
            if grid[nr, nc] not in (0,):
                ray["active"] = False
                continue
            proposals.setdefault((nr, nc), []).append(i)
        if not proposals and not any(ray["active"] for ray in arrows):
            break
        for cell, idxs in proposals.items():
            if len(idxs) > 1:
                out[cell] = 2
                for i in idxs:
                    arrows[i]["active"] = False
            else:
                i = idxs[0]
                out[cell] = arrows[i]["color"]
                occupied[cell] = i
                arrows[i]["pos"] = cell
    for bi, events in hits.items():
        first_step = min(s for s, _ in events)
        tied = [i for s, i in events if s == first_step]
        chosen = min(tied, key=lambda i: arrows[i]["origin"])
        for r, c in blocks[bi]:
            out[r, c] = arrows[chosen]["color"]
    return out


def _arrow(grid, color, orient, r, c):
    if orient == "R":
        pts = [(r, c + i) for i in range(4)] + [(r - 1, c + 2), (r + 1, c + 2)]
    elif orient == "L":
        pts = [(r, c + i) for i in range(4)] + [(r - 1, c + 1), (r + 1, c + 1)]
    elif orient == "D":
        pts = [(r + i, c) for i in range(4)] + [(r + 2, c - 1), (r + 2, c + 1)]
    else:
        pts = [(r + i, c) for i in range(4)] + [(r + 1, c - 1), (r + 1, c + 1)]
    for pr, pc in pts:
        grid[pr, pc] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((17, 17), dtype=int)
    colors = random.sample([1, 3, 4, 6, 7], 3)
    _arrow(grid, colors[0], "R", 5, 1)
    _arrow(grid, colors[1], "D", 1, 8)
    _arrow(grid, colors[2], "L", 13, 11)
    for r, c in [(4, 10), (10, 7), (12, 4)]:
        grid[r : r + 2, c : c + 2] = 5
    return grid

