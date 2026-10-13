import random
import numpy as np


BG = 0
GREY = 5


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


def _norm(cells):
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _holes_for_object(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    obj = set(cells)
    local_bg = []
    for comp in _components(grid[r0:r1 + 1, c0:c1 + 1] == BG):
        shifted = [(r + r0, c + c0) for r, c in comp]
        if not any(r in (r0, r1) or c in (c0, c1) for r, c in shifted):
            local_bg.append(shifted)
    return local_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each holed object by matching its hole shape to an edge legend.

    Input structure: Central monochrome objects each enclose one black hole.
    Along one edge are small grey shape legends, each adjacent to a singleton
    color marker.
    Rule: Normalize each object's internal hole shape and find the identical
    grey legend shape without rotation or reflection. Recolor the object's shell
    pixels with the adjacent legend marker color, while preserving the legend,
    holes, and background.
    Key insight: The outside object silhouette is irrelevant; the hidden hole
    shape is the key that indexes the edge legend.
    """
    out = grid.copy()
    h, w = grid.shape
    legend = {}
    legend_cells = set()
    for comp in _components(grid == GREY):
        if len(comp) <= 6 and any(c <= 2 or c >= w - 3 for _, c in comp):
            key = _norm(comp)
            legend_cells.update(comp)
            mark = None
            for r, c in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] not in (BG, GREY):
                        mark = int(grid[nr, nc])
            if mark is not None:
                legend[key] = mark
    for col in [int(x) for x in np.unique(grid) if x not in (BG, GREY)]:
        for comp in _components(grid == col):
            if len(comp) < 8 or any(c <= 2 or c >= w - 3 for _, c in comp):
                continue
            holes = _holes_for_object(grid, comp)
            if len(holes) != 1:
                continue
            key = _norm(holes[0])
            if key in legend:
                for r, c in comp:
                    out[r, c] = legend[key]
    return out


def generate_input() -> np.ndarray:
    h = w = 26
    grid = np.zeros((h, w), dtype=int)
    hole_shapes = [[(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 0), (0, 1), (1, 1), (2, 1)]]
    marker_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    for i, (shape, mcol) in enumerate(zip(hole_shapes, marker_cols)):
        r0 = 2 + i * 7
        for dr, dc in shape:
            grid[r0 + dr, dc] = GREY
        grid[r0, 3] = mcol
    obj_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    positions = [(3, 8), (4, 16), (14, 10)]
    for shape, col, (r0, c0) in zip(hole_shapes, obj_cols, positions):
        grid[r0:r0 + 6, c0:c0 + 6] = col
        for dr, dc in shape:
            grid[r0 + 2 + dr, c0 + 2 + dc] = BG
    return grid
