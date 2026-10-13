import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = [(r, c)]
        seen[r, c] = True
        comp = []
        while q:
            cr, cc = q.pop()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _settle(grid):
    out = np.zeros_like(grid)
    h, w = grid.shape
    for c in range(w):
        vals = [int(grid[r, c]) for r in range(h) if grid[r, c] != 0]
        for i, v in enumerate(vals):
            out[h - len(vals) + i, c] = v
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Settle all non-black pixels by column, align settled gray anchors, overlay sprites, and crop the composite."""
    settled = _settle(grid)
    sprites = []
    for comp in _components(settled != 0):
        anchors = [p for p in comp if settled[p] == 5]
        if len(anchors) != 1:
            continue
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        anchor = anchors[0]
        pix = []
        for r, c in comp:
            pix.append((r - anchor[0], c - anchor[1], int(settled[r, c])))
        sprites.append((anchor[1], pix))
    if not sprites:
        return settled
    sprites.sort(key=lambda x: x[0])
    coords = [(dr, dc) for _, pix in sprites for dr, dc, _ in pix]
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    occupied = np.zeros_like(out, dtype=bool)
    for _, pix in sprites:
        for dr, dc, val in pix:
            rr, cc = dr - min_r, dc - min_c
            if not occupied[rr, cc]:
                out[rr, cc] = val
                occupied[rr, cc] = True
    nz = np.argwhere(out != 0)
    if nz.size == 0:
        return out
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)
    cols_for_shapes = random.sample([1, 2, 3, 4, 6, 7], 3)
    shapes = [
        [(0, 0, 5), (1, 0, cols_for_shapes[0]), (1, 1, cols_for_shapes[0]), (2, 0, cols_for_shapes[0])],
        [(0, 1, cols_for_shapes[1]), (1, 0, cols_for_shapes[1]), (1, 1, 5), (2, 1, cols_for_shapes[1])],
        [(0, 0, cols_for_shapes[2]), (0, 1, cols_for_shapes[2]), (1, 1, 5), (2, 1, cols_for_shapes[2])],
    ]
    if random.random() < 0.5:
        shapes[0].append((2, 1, cols_for_shapes[0]))
    if random.random() < 0.5:
        shapes[2].append((1, 0, cols_for_shapes[2]))
    cols = [1, 5, 10]
    random.shuffle(cols)
    for shape, c0 in zip(shapes, cols):
        r0 = random.randint(1, 4)
        for dr, dc, val in shape:
            g[r0 + dr, c0 + dc] = val
    return g
