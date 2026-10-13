import numpy as np


BLACK = 0
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill template cells only where the left bar length and top bar length point to the same palette key.

    Input structure:
    - Black background with a largest grey component that forms a partitioned template grid.
    - Grey bar segments above the template encode one key per template column by horizontal length.
    - Grey bar segments left of the template encode one key per template row by vertical length, and a right-edge palette contains colored blocks whose heights are keys one through four.

    Rule:
    - Read the right-edge palette as a mapping from block height to palette color.
    - Read top bars left-to-right and left bars top-to-bottom to obtain a column key and row key for every template cell.
    - For each cell, fill its black interior with the palette color of that key when row key equals column key; otherwise leave the cell black.

    Key insight:
    - The grey bar lengths are symbolic keys rather than drawing parts of the template, and a cell activates only when its row key and column key agree.
    """
    out = grid.copy()
    palette = {}
    for color in np.unique(grid):
        if color in (BLACK, GREY):
            continue
        for comp in _components(grid == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if max(cs) == grid.shape[1] - 1:
                palette[max(rs) - min(rs) + 1] = int(color)
    grey_comps = _components(grid == GREY)
    template = max(grey_comps, key=len)
    trs = [p[0] for p in template]
    tcs = [p[1] for p in template]
    r0, r1 = min(trs), max(trs)
    c0, c1 = min(tcs), max(tcs)
    row_lines = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GREY)]
    col_lines = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == GREY)]
    row_intervals = list(zip(row_lines[:-1], row_lines[1:]))
    col_intervals = list(zip(col_lines[:-1], col_lines[1:]))

    top_bars = []
    left_bars = []
    for comp in grey_comps:
        if comp == template:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if max(rs) < r0:
            top_bars.append((min(cs), max(cs) - min(cs) + 1))
        elif max(cs) < c0:
            left_bars.append((min(rs), max(rs) - min(rs) + 1))
    top_bars.sort()
    left_bars.sort()
    col_keys = [length for _, length in top_bars]
    row_keys = [length for _, length in left_bars]

    for i, (ra, rb) in enumerate(row_intervals):
        for j, (ca, cb) in enumerate(col_intervals):
            if i < len(row_keys) and j < len(col_keys) and row_keys[i] == col_keys[j]:
                out[ra + 1 : rb, ca + 1 : cb] = palette[row_keys[i]]
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    rows = int(rng.integers(2, 5))
    cols = int(rng.integers(2, 5))
    cell_h = int(rng.integers(2, 4))
    cell_w = int(rng.integers(2, 4))
    tmpl_h = rows * cell_h + rows + 1
    tmpl_w = cols * cell_w + cols + 1
    grid = np.zeros((30, 30), dtype=int)
    r0, c0 = 6, 7
    for r in range(rows + 1):
        rr = r0 + r * (cell_h + 1)
        grid[rr, c0 : c0 + tmpl_w] = GREY
    for c in range(cols + 1):
        cc = c0 + c * (cell_w + 1)
        grid[r0 : r0 + tmpl_h, cc] = GREY

    row_keys = list(rng.integers(1, 5, size=rows))
    col_keys = list(rng.integers(1, 5, size=cols))
    for i, key in enumerate(row_keys):
        rr = r0 + 1 + i * (cell_h + 1)
        start = r0 - 1 + i * 5
        grid[start : start + key, c0 - 3] = GREY
    for j, key in enumerate(col_keys):
        start = c0 - 1 + j * 5
        grid[r0 - 3, start : start + key] = GREY

    palette_colors = rng.choice([1, 2, 3, 4, 6, 7, 8, 9], size=4, replace=False)
    pr = 1
    pc = grid.shape[1] - 2
    for key, color in enumerate(palette_colors, start=1):
        grid[pr : pr + key, pc : pc + 2] = int(color)
        pr += key + 1
    return grid
