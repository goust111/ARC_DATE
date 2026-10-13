import random
import numpy as np


BG = 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Flip each object according to which grid border its bounding box touches.

    Input structure: A blue background contains separated monochrome asymmetric
    objects. Some object bounding boxes touch one outer edge or a corner, while
    others touch no edge.
    Rule: Objects touching left or right are mirrored horizontally in their own
    bbox. Objects touching top or bottom are mirrored vertically. Corner objects
    receive both flips, equivalent to 180-degree rotation. Interior objects stay
    unchanged.
    Key insight: The transformation is local to each object's bbox, and the
    touched border axes determine which mirror operations apply.
    """
    out = np.full_like(grid, BG)
    h, w = grid.shape
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        for comp in _components(grid == col):
            r0, c0, r1, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True
            if c0 == 0 or c1 == w - 1:
                mask = np.fliplr(mask)
            if r0 == 0 or r1 == h - 1:
                mask = np.flipud(mask)
            for rr, cc in zip(*np.where(mask)):
                out[r0 + rr, c0 + cc] = col
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([18, 20, 22, 24])
    grid = np.full((h, w), BG, dtype=int)
    colors = random.sample([0, 2, 3, 4, 5, 6, 7, 8, 9], 5)
    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]
    anchors = [(0, 2), (h - 3, 8), (6, 0), (10, w - 2), (7, 13)]
    for col, (r0, c0) in zip(colors, anchors):
        for dr, dc in shape:
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                grid[rr, cc] = col
    return grid
