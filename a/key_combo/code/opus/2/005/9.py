import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r1, c1 = min(rs), min(cs)
    arr = np.zeros((max(rs) - r1 + 1, max(cs) - c1 + 1), dtype=bool)
    for r, c in cells:
        arr[r - r1, c - c1] = True
    return arr


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    rows = [r for r in range(h) if np.sum(grid[r] == 5) >= 6]
    cols = [c for c in range(w) if np.sum(grid[:, c] == 5) >= 4]
    if len(rows) < 2 or len(cols) < 2:
        return out
    sr1, sr2 = min(rows), max(rows)
    sc1, sc2 = min(cols), max(cols)
    in_skel = np.zeros_like(grid, dtype=bool)
    in_skel[sr1:sr2 + 1, sc1:sc2 + 1] = True

    sprites = {}
    legend_mask = (~in_skel) & (grid != 0) & (grid != 5)
    for comp in _components(legend_mask):
        color = int(grid[comp[0]])
        sprites[color] = _mask(comp)

    occupied = set()
    for ri in range(len(rows) - 1):
        for ci in range(len(cols) - 1):
            top, bot = rows[ri], rows[ri + 1]
            left, right = cols[ci], cols[ci + 1]
            if any((rr, cc) in occupied for rr in range(top, bot + 1) for cc in range(left, right + 1)):
                continue
            inside = [(r, c) for r in range(top + 1, bot) for c in range(left + 1, right)
                      if grid[r, c] not in (0, 5)]
            if not inside:
                continue
            color = int(grid[inside[0]])
            same = [(r, c) for r, c in inside if grid[r, c] == color]
            etop, ebot, eleft, eright = top, bot, left, right
            if len(same) >= 2:
                (r1, c1), (r2, c2) = same[:2]
                if r1 == r2:
                    d = abs(c1 - c2)
                    eleft = max(0, left - d)
                    eright = min(w - 1, right + d)
                elif c1 == c2:
                    d = abs(r1 - r2)
                    etop = max(0, top - d)
                    ebot = min(h - 1, bot + d)
            out[etop, eleft:eright + 1] = color
            out[ebot, eleft:eright + 1] = color
            out[etop:ebot + 1, eleft] = color
            out[etop:ebot + 1, eright] = color
            sprite = sprites.get(color)
            if sprite is not None:
                ch = ebot - etop - 1
                cw = eright - eleft - 1
                r0 = etop + 1 + (ch - sprite.shape[0]) // 2
                c0 = eleft + 1 + (cw - sprite.shape[1]) // 2
                for rr in range(sprite.shape[0]):
                    for cc in range(sprite.shape[1]):
                        if sprite[rr, cc] and 0 <= r0 + rr < h and 0 <= c0 + cc < w:
                            out[r0 + rr, c0 + cc] = color
            for rr in range(etop, ebot + 1):
                for cc in range(eleft, eright + 1):
                    occupied.add((rr, cc))
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    rows = [1, 5, 9, 13]
    cols = [5, 9, 13, 17]
    for r in rows:
        grid[r, cols[0]:cols[-1] + 1] = 5
    for c in cols:
        grid[rows[0]:rows[-1] + 1, c] = 5
    legend = {
        1: np.array([[1, 1], [1, 0]], dtype=bool),
        2: np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        3: np.array([[0, 1], [1, 1]], dtype=bool),
    }
    anchors = {1: (15, 0), 2: (15, 4), 3: (15, 9)}
    for color, m in legend.items():
        r0, c0 = anchors[color]
        grid[r0:r0 + m.shape[0], c0:c0 + m.shape[1]][m] = color
    cells = [(0, 0), (0, 1), (1, 2), (2, 0)]
    colors = [1, 2, 3, random.choice([1, 2, 3])]
    for (ri, ci), color in zip(cells, colors):
        top, bot = rows[ri], rows[ri + 1]
        left, right = cols[ci], cols[ci + 1]
        if random.random() < 0.5:
            r = random.randint(top + 1, bot - 1)
            c = random.randint(left + 1, right - 1)
            grid[r, c] = color
        elif random.random() < 0.5:
            r = random.randint(top + 1, bot - 1)
            grid[r, left + 1] = color
            grid[r, right - 1] = color
        else:
            c = random.randint(left + 1, right - 1)
            grid[top + 1, c] = color
            grid[bot - 1, c] = color
    return grid
