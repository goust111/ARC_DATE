import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _draw(out, pattern, top, left):
    h, w = pattern.shape
    for r in range(h):
        for c in range(w):
            if pattern[r, c] != 0:
                out[top + r, left + c] = pattern[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    comps = _components(grid)
    template_cells = max(comps, key=lambda cells: (len(set(int(grid[p]) for p in cells)), len(cells)))
    tr0, tc0, tr1, tc1 = _bbox(template_cells)
    template = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]

    fragments = sorted([cells for cells in comps if cells is not template_cells], key=lambda cells: _bbox(cells)[:2])
    dirs = {0: (0, template.shape[1]), 1: (template.shape[0], 0), 2: (0, -template.shape[1]), 3: (-template.shape[0], 0)}
    h, w = grid.shape
    for cells in fragments:
        r0, c0, r1, c1 = _bbox(cells)
        frag = grid[r0 : r1 + 1, c0 : c1 + 1]
        match = None
        for k in range(4):
            rot = np.rot90(template, -k)
            if frag.shape[0] <= rot.shape[0] and frag.shape[1] <= rot.shape[1]:
                ok = True
                for r in range(frag.shape[0]):
                    for c in range(frag.shape[1]):
                        if frag[r, c] != 0 and frag[r, c] != rot[r, c]:
                            ok = False
                if ok:
                    match = (k, rot)
                    break
        if match is None:
            continue
        k, rot = match
        dr, dc = dirs[k]
        top, left = r0, c0
        while 0 <= top and 0 <= left and top + rot.shape[0] <= h and left + rot.shape[1] <= w:
            _draw(out, rot, top, left)
            top += dr
            left += dc
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    template = np.array([[1, 2, 0, 3], [1, 1, 2, 3], [0, 2, 2, 3]], dtype=int)
    grid[0:3, 0:4] = template
    placements = [(4, 4, 0), (4, 11, 1), (12, 11, 2), (12, 4, 3)]
    random.shuffle(placements)
    for top, left, k in placements[: random.randint(2, 4)]:
        rot = np.rot90(template, -k)
        pts = [(r, c) for r in range(rot.shape[0]) for c in range(rot.shape[1]) if rot[r, c] != 0]
        keep = random.sample(pts, random.randint(max(3, len(pts) * 4 // 10), max(4, len(pts) * 7 // 10)))
        for r, c in keep:
            grid[top + r, left + c] = rot[r, c]
    return grid
