import numpy as np


BLACK = 0
MAROON = 9
GREY = 5


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_path(grid):
    start = None
    for r in range(grid.shape[0]):
        if grid[r, 0] != BLACK:
            start = (r, 0)
            break
    path = [start]
    prev = None
    cur = start
    while cur[1] != grid.shape[1] - 1:
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != BLACK and (nr, nc) != prev:
                nbrs.append((nr, nc))
        prev, cur = cur, nbrs[0]
        path.append(cur)
    colors = []
    last = None
    for r, c in path:
        color = int(grid[r, c])
        if color != last:
            colors.append(color)
            last = color
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the lower grey blocks by the color-segment order of the upper polyline path.

    Input structure:
    - A maroon separator row splits the grid into an upper path panel and a lower block panel.
    - The upper panel contains one left-to-right orthogonal path whose straight segments each have a distinct color.
    - The lower panel contains equal-sized grey blocks, each with one colored center pixel matching one path segment color.

    Rule:
    - Trace the path from the left edge to the right edge and record the segment colors in traversal order.
    - Match each lower block by its center color.
    - Erase the old lower blocks and restamp them from left to right above the bottom edge, with one black column between consecutive blocks.

    Key insight:
    - The path's geometry hides the true ordering: the lower arrangement must follow traversal time along the bends, not simple left-to-right reading.
    """
    out = grid.copy()
    sep = int(np.where(np.all(grid == MAROON, axis=1))[0][0])
    order = _trace_path(grid[:sep])
    blocks = {}
    for comp in _components(grid[sep + 1 :] == GREY):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs) + sep + 1, max(rs) + sep + 1
        c0, c1 = min(cs), max(cs)
        if (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
            block = out[r0 : r1 + 1, c0 : c1 + 1].copy()
            blocks[int(block[1, 1])] = block
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    col = 1
    row = grid.shape[0] - 3
    for color in order:
        block = blocks[color]
        out[row : row + 3, col : col + 3] = block
        col += 4
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    sep = 11
    grid[sep, :] = MAROON
    colors = list(rng.choice([1, 2, 3, 4, 6, 7, 8], size=int(rng.choice([3, 5])), replace=False))
    row = int(rng.integers(1, 8))
    col = 0
    remaining_h = len(colors[::2])
    for idx, color in enumerate(colors):
        if idx % 2 == 0:
            remaining_h -= 1
            if remaining_h == 0:
                new_col = 23
            else:
                min_end = col + 3
                max_end = 23 - 3 * remaining_h
                new_col = int(rng.integers(min_end, max_end + 1))
            grid[row, col : new_col + 1] = color
            col = new_col
        else:
            new_row = int(rng.integers(1, 8))
            while new_row == row:
                new_row = int(rng.integers(1, 8))
            lo, hi = sorted((row, new_row))
            grid[lo : hi + 1, col] = color
            row = new_row
    cols = rng.choice(np.arange(1, 20, 4), size=len(colors), replace=False)
    for color, col in zip(colors, cols):
        row = int(rng.integers(sep + 2, 20))
        grid[row : row + 3, col : col + 3] = GREY
        grid[row + 1, col + 1] = color
    return grid
