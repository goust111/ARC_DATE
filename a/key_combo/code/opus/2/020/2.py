import numpy as np
from collections import defaultdict


def _tile_layout(grid):
    h, w = grid.shape
    row_sep = [r for r in range(h) if np.all(grid[r] == 5)]
    col_sep = [c for c in range(w) if np.all(grid[:, c] == 5)]
    if len(row_sep) < 4 or len(col_sep) < 4:
        return None
    tile_h = row_sep[0]
    tile_w = col_sep[0]
    rows = [r - tile_h for r in row_sep[:4]]
    cols = [c - tile_w for c in col_sep[:4]]
    return tile_h, tile_w, rows, cols


def _slide_in_tile(g, r0, c0, start, drow, dcol):
    y, x = start
    while True:
        ny, nx = y + drow, x + dcol
        if not (r0 <= ny < r0 + 3 and c0 <= nx < c0 + 3):
            break
        if g[ny, nx] != 0:
            break
        y, x = ny, nx
    return y, x


def main(grid):
    """
    1) Detect 4x4 tile layout and decode direction from top-left legend pair.
    2) Slide each endpoint in normal tiles along the decoded direction.
    3) Connect same-color endpoints that share same subcell and same tile row/col.
    4) Keep all separators and legend unchanged.
    """
    g = np.array(grid, dtype=int).copy()
    layout = _tile_layout(g)
    if layout is None:
        return g
    _, _, rows, cols = layout

    # decode direction from top-left tile
    r0, c0 = rows[0], cols[0]
    drow = 0
    dcol = 0
    if g[r0 + 1, c0 + 0] == 9:  # impossible in generator, but keep fallback
        drow = -1
    if g[r0 + 1, c0 + 2] == 9:
        dcol = 0
        drow = 1
    if g[r0 + 0, c0 + 1] == 9:
        drow = 0
        dcol = -1
    if g[r0 + 2, c0 + 1] == 9:
        drow = 0
        dcol = 1
    if drow == 0 and dcol == 0:
        drow = 0
        dcol = 1

    moved = {}
    for ti, r in enumerate(rows):
        for tj, c in enumerate(cols):
            if ti == 0 and tj == 0:
                continue
            block = g[r : r + 3, c : c + 3]
            pts = np.argwhere((block != 0) & (block != 5))
            if len(pts) == 0:
                continue
            ly, lx = map(int, pts[0])
            color = int(block[ly, lx])
            src = (r + ly, c + lx)
            dst = _slide_in_tile(g, r, c, src, drow, dcol)
            if src != dst:
                g[src] = 0
            g[dst] = color
            moved.setdefault((color, ly, lx), []).append((ti, tj, dst))

    for (color, ly, lx), lst in moved.items():
        by_row = defaultdict(list)
        by_col = defaultdict(list)
        for ti, tj, (rr, cc) in lst:
            by_row[ti].append((tj, cc))
            by_col[tj].append((ti, rr))

        for ti, arr in by_row.items():
            if len(arr) < 2:
                continue
            arr = sorted(arr)
            for (a, _), (b, __) in zip(arr[:-1], arr[1:]):
                for tj in range(min(a, b) + 1, max(a, b)):
                    g[rows[ti] + ly, cols[tj] + lx] = color
        for tj, arr in by_col.items():
            if len(arr) < 2:
                continue
            arr = sorted(arr)
            for (a, _), (b, __) in zip(arr[:-1], arr[1:]):
                for ti in range(min(a, b) + 1, max(a, b)):
                    g[rows[ti] + ly, cols[tj] + lx] = color

    return g


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 13 + 23)
    g = np.zeros((17, 17), dtype=int)

    # separators (4x4 tiles, inner 3x3)
    for i in range(1, 4):
        idx = i * 4 - 1
        g[idx, :] = 5
        g[:, idx] = 5

    # direction encoding
    dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    g[1, 1] = 9
    g[1 + dr, 1 + dc] = 9

    colors = [1, 2, 3]
    tile_ids = [i for i in range(16) if i != 0]
    assignments = {}

    # enforce at least one same-row same-col pair per color
    for color in colors:
        row = int(rng.integers(0, 4))
        c1, c2 = rng.choice(np.arange(4), size=2, replace=False)
        ly, lx = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        assignments[row * 4 + c1] = (color, ly, lx)
        assignments[row * 4 + c2] = (color, ly, lx)

    # additional optional points
    for tid in tile_ids:
        if tid in assignments:
            continue
        if rng.random() < 0.45:
            color = int(rng.choice(colors))
            ly, lx = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            assignments[tid] = (color, ly, lx)

    for tid, (color, ly, lx) in assignments.items():
        tr, tc = divmod(tid, 4)
        r0 = tr * 4
        c0 = tc * 4
        g[r0 + ly, c0 + lx] = color

    return g
