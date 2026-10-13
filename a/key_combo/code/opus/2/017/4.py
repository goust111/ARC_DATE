import numpy as np

_RNG = np.random.default_rng(3104)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _connect_lines(cell: np.ndarray):
    h, w = cell.shape
    out = cell.copy()
    # row connections
    for r in range(h):
        cols = np.where(out[r] == 2)[0]
        if len(cols) >= 2:
            out[r, cols.min():cols.max() + 1] = 2
    # column connections
    for c in range(w):
        rows = np.where(out[:, c] == 2)[0]
        if len(rows) >= 2:
            out[rows.min():rows.max() + 1, c] = 2
    return out


def _is_hollow_rect(mask: np.ndarray):
    if np.all(mask == 0):
        return False
    rs, cs = np.where(mask)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    if r1 <= r0 or c1 <= c0:
        return False

    h, w = mask.shape
    if not (0 <= r0 < h and 0 <= r1 < h and 0 <= c0 < w and 0 <= c1 < w):
        return False

    for c in range(c0, c1 + 1):
        if mask[r0, c] != 2 or mask[r1, c] != 2:
            return False
    for r in range(r0, r1 + 1):
        if mask[r, c0] != 2 or mask[r, c1] != 2:
            return False
    if mask[r0:r1 + 1, c0:c1 + 1].sum() == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
        return True
    return False


def main(grid: np.ndarray):
    """Task 4 transform.

    1) Detect gray separator lattice and process each 3x3 cell independently.
    2) For each cell, connect red marks in rows/cols and test for closed hollow rectangle.
    3) Valid cells: erase boundary marks and fill interior with green.
    4) Invalid cells are cleared to black; crop around all valid cells and return.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    sep_rows = [i for i in range(h) if np.all(g[i] == 5)]
    sep_cols = [j for j in range(w) if np.all(g[:, j] == 5)]
    if 0 not in sep_rows:
        sep_rows = [0] + sep_rows
    if h - 1 not in sep_rows:
        sep_rows = sep_rows + [h - 1]
    if 0 not in sep_cols:
        sep_cols = [0] + sep_cols
    if w - 1 not in sep_cols:
        sep_cols = sep_cols + [w - 1]

    sep_rows = sorted(set(sep_rows))
    sep_cols = sorted(set(sep_cols))

    valid = set()
    for ri in range(len(sep_rows) - 1):
        r0, r1 = sep_rows[ri] + 1, sep_rows[ri + 1] - 1
        if r0 > r1:
            continue
        for cj in range(len(sep_cols) - 1):
            c0, c1 = sep_cols[cj] + 1, sep_cols[cj + 1] - 1
            if c0 > c1:
                continue

            cell = out[r0:r1 + 1, c0:c1 + 1]
            red = (cell == 2)
            after = _connect_lines(red.copy())
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    cell_val = after[rr - r0, cc - c0]
                    out[rr, cc] = 2 if cell_val else 0

            mask_rect = (out[r0:r1 + 1, c0:c1 + 1] == 2)
            ok = _is_hollow_rect(mask_rect)
            if ok:
                rs, cs = np.where(mask_rect)
                rr0, rr1 = int(rs.min()), int(rs.max())
                cc0, cc1 = int(cs.min()), int(cs.max())
                out[r0 + rr0 + 1:r0 + rr1, c0 + cc0 + 1:c0 + cc1] = 3
                out[r0:r1 + 1, c0:c1 + 1] = np.where(mask_rect, 0, out[r0:r1 + 1, c0:c1 + 1])
                valid.add((ri, cj))
            else:
                out[r0:r1 + 1, c0:c1 + 1] = 0

    if not valid:
        return np.array([[0]], dtype=int)

    vr = [v[0] for v in valid]
    vc = [v[1] for v in valid]
    r_lo, r_hi = min(vr), max(vr)
    c_lo, c_hi = min(vc), max(vc)

    top = sep_rows[r_lo]
    bottom = sep_rows[r_hi + 1]
    left = sep_cols[c_lo]
    right = sep_cols[c_hi + 1]
    return out[top:bottom + 1, left:right + 1]


def _draw_border(g, r0, c0, r1, c1):
    g[r0, c0:c1 + 1] = 5
    g[r1, c0:c1 + 1] = 5
    g[r0:r1 + 1, c0] = 5
    g[r0:r1 + 1, c1] = 5


def generate_input():
    """Task 4 input generator.

    1) Build a 19x19 lattice by full gray separators into 3x3 cells.
    2) Randomly pick 3-5 valid cells with proper hollow-rect diagnostics.
    3) Put random red markers in cells; valid cells form complete connected rectangles.
    """

    h = w = 19
    g = np.zeros((h, w), dtype=int)

    _draw_border(g, 0, 0, h - 1, 0)
    _draw_border(g, 0, w - 1, h - 1, w - 1)
    _draw_border(g, 0, 0, 0, w - 1)
    _draw_border(g, h - 1, 0, h - 1, w - 1)

    for i in range(0, 18, 6):
        for j in range(0, 18, 6):
            if i == 0 or j == 0 or i == 18 or j == 18:
                continue
            _draw_border(g, i, j, i + 5, j)
            _draw_border(g, i, j, i, j + 5)
            _draw_border(g, i, j + 5, i + 5, j + 5)
            _draw_border(g, i + 5, j, i + 5, j + 5)

    sep_rows = [0, 6, 12, 18]
    sep_cols = [0, 6, 12, 18]
    cell_rows = [(1, 5), (7, 11), (13, 17)]
    cell_cols = [(1, 5), (7, 11), (13, 17)]

    # choose 3-5 connected valid cells in a rectangle block
    valid_num = int(_RNG.integers(3, 6))
    shape_h = int(_RNG.integers(1, 3 + 1))
    shape_w = int(_RNG.integers(1, valid_num + 1))
    shape_h = min(shape_h, 3)
    shape_w = min(shape_w, 3)
    # clamp by area
    while shape_h * shape_w > valid_num:
        if shape_h > 1:
            shape_h -= 1
        else:
            shape_w -= 1

    r0 = int(_RNG.integers(0, 4 - shape_h))
    c0 = int(_RNG.integers(0, 4 - shape_w))
    valid_cells = [(r0 + dr, c0 + dc) for dr in range(shape_h) for dc in range(shape_w)]
    valid_cells = valid_cells[:valid_num]

    for ri in range(3):
        for ci in range(3):
            rstart, rstop = cell_rows[ri]
            cstart, cstop = cell_cols[ci]
            if (ri, ci) in valid_cells:
                a = int(_RNG.integers(rstart + 1, rstop - 1))
                b = int(_RNG.integers(a + 1, rstop))
                c = int(_RNG.integers(cstart + 1, cstop - 1))
                d = int(_RNG.integers(c + 1, cstop))
                # corners of inner rectangle
                pts = [(a, c), (a, d), (b, c), (b, d)]
                for rr, cc in pts:
                    g[rr, cc] = 2
            else:
                kind = int(_RNG.integers(2))
                if kind == 0:
                    row = int(_RNG.integers(rstart, rstop + 1))
                    cA = int(_RNG.integers(cstart, cstop - 1))
                    cB = int(_RNG.integers(cA + 1, cstop + 1))
                    g[row, cA] = 2
                    g[row, cB] = 2
                else:
                    col = int(_RNG.integers(cstart, cstop + 1))
                    rA = int(_RNG.integers(rstart, rstop - 1))
                    rB = int(_RNG.integers(rA + 1, rstop + 1))
                    g[rA, col] = 2
                    g[rB, col] = 2

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
