import random
from collections import deque

import numpy as np


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _holes(grid: np.ndarray, color: int, cells):
    pts = np.array(cells)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    h, w = sub.shape
    seen = np.zeros_like(sub, dtype=bool)
    holes = []
    for r in range(h):
        for c in range(w):
            if sub[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region = []
            touch = False
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not sub[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touch:
                holes.append([(r0 + rr, c0 + cc) for rr, cc in region])
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each solid raw block and fill the matching same-color hollow mold cavity.

    Input structure:
    - The grid has a dominant background color.
    - For each foreground color there are exactly two disconnected 4-connected objects.
    - One object is a hollow closed outline with an enclosed background cavity; the other is a separate solid material block.

    Rule:
    - For each color pair, identify the mold as the object that contains an enclosed background hole.
    - Delete the separate solid block from its original position.
    - Fill the mold's enclosed cavity with the same color so the hollow outline becomes solid.

    Key insight:
    - The two same-color objects have asymmetric roles: the hole-bearing object is the destination mold, and the other object is only a disposable material source.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    for color in [int(v) for v in np.unique(g) if v != bg]:
        comps = _components(g == color)
        hole_counts = [len(_holes(g, color, comp)) for comp in comps]
        mold_index = int(np.argmax(hole_counts))
        mold = comps[mold_index]
        for i, raw in enumerate(comps):
            if i == mold_index:
                continue
            for r, c in raw:
                out[r, c] = bg
        for hole in _holes(g, color, mold):
            for r, c in hole:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(16, 26), random.randint(16, 26)
        bg = random.choice([0, 8, 1])
        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        colors = random.sample([c for c in range(10) if c != bg], random.randint(2, 4))
        ok = True
        for color in colors:
            outer_h = random.randint(5, 8)
            outer_w = random.randint(5, 8)
            raw_h = random.randint(3, 5)
            raw_w = random.randint(3, 6)
            placed = False
            for _ in range(100):
                top = random.randint(1, h - outer_h - 1)
                left = random.randint(1, w - outer_w - 1)
                rtop = random.randint(1, h - raw_h - 1)
                rleft = random.randint(1, w - raw_w - 1)
                if abs(top - rtop) < max(outer_h, raw_h) and abs(left - rleft) < max(outer_w, raw_w):
                    continue
                if occ[max(0, top - 1) : min(h, top + outer_h + 1), max(0, left - 1) : min(w, left + outer_w + 1)].any():
                    continue
                if occ[max(0, rtop - 1) : min(h, rtop + raw_h + 1), max(0, rleft - 1) : min(w, rleft + raw_w + 1)].any():
                    continue
                grid[top : top + outer_h, left : left + outer_w] = color
                inner_pad = 1
                grid[top + inner_pad : top + outer_h - inner_pad, left + inner_pad : left + outer_w - inner_pad] = bg
                occ[max(0, top - 1) : min(h, top + outer_h + 1), max(0, left - 1) : min(w, left + outer_w + 1)] = True
                grid[rtop : rtop + raw_h, rleft : rleft + raw_w] = color
                occ[max(0, rtop - 1) : min(h, rtop + raw_h + 1), max(0, rleft - 1) : min(w, rleft + raw_w + 1)] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
