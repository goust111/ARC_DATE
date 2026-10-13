import numpy as np

_RNG = np.random.default_rng(3110)


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
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _fill_cell_with_template(cell: np.ndarray, template: np.ndarray, color: int, base: int):
    h, w = cell.shape
    th, tw = template.shape
    out = cell.copy()
    for rr in range(th):
        for cc in range(tw):
            if rr >= h or cc >= w:
                continue
            out[rr, cc] = color if template[rr, cc] else base
    return out


def main(grid: np.ndarray):
    """Task 10 transform.

    1) Parse legend swatches and their ordered 4x4 templates.
    2) Connect same-color markers along fixed local offsets per cell lattice.
    3) Determine activated cells: exactly one non-base color inside a cell.
    4) Replace activated cells by aligned template in that color; unactivated cells stay.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape

    # legend parse
    swatches = []
    for c in range(w):
        v = int(g[0, c])
        if v in (0, 5):
            continue
        swatches.append((c, v))
    swatches = sorted(swatches, key=lambda x: x[0])

    comp_masks = []
    vis = np.zeros_like(g, dtype=bool)
    for r in range(h):
        for c in range(w):
            if r > 4 or g[r, c] != 5 or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and g[nx, ny] == 5 and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r0 <= 1 and r0 < 4 and (r1 - r0 + 1) <= 4 and (c1 - c0 + 1) <= 4:
                tm = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                for rr, cc in comp:
                    tm[rr - r0, cc - c0] = True
                comp_masks.append((c0, tm))

    comp_masks = sorted(comp_masks, key=lambda x: x[0])
    colors = [v for _, v in swatches]
    templates = {}
    for col, (_, tm) in zip(colors, comp_masks):
        templates[col] = tm

    out = g.copy()

    # separator rows and cols are full 5 in main area
    sep_rows = [r for r in range(h) if np.all(g[r] == 5)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == 5)]

    if len(sep_rows) < 2 or len(sep_cols) < 2:
        return out

    # main area starts at 5 in prompt structure
    row_start = 5
    cell_rows = [row_start, 9, 14, 18]
    cell_cols = [3, 7, 12, 16]

    base = 9
    # collect marker pixels by cell-local positions
    for ri in range(3):
        r0 = cell_rows[ri]
        r1 = cell_rows[ri + 1]
        for cj in range(3):
            c0 = cell_cols[cj]
            c1 = cell_cols[cj + 1]
            # local offsets and markers may exist, process fixed set below
            pass

    # 3x3 cells
    for ri in range(3):
        r0 = cell_rows[ri] + 1
        r1 = cell_rows[ri + 1]
        for cj in range(3):
            c0 = cell_cols[cj] + 1
            c1 = cell_cols[cj + 1]
            cell = out[r0:r1, c0:c1]

            # apply connectivity and fill for each color-offset pair
            for color in colors:
                for dr, dc in [(1, 1), (2, 2)]:
                    # find marker positions matching local position
                    pos = []
                    for rr in range(r0, r1):
                        for cc in range(c0, c1):
                            if out[rr, cc] == color and (rr - r0, cc - c0) == (dr, dc):
                                pos.append((rr, cc))

                    if not pos:
                        continue
                    if len(pos) == 1:
                        # no segment to extend
                        pass
                    elif pos:
                        # connect positions if aligned
                        rows = sorted(set(p[0] for p in pos))
                        cols = sorted(set(p[1] for p in pos))
                        if len(rows) == 1:
                            rr = rows[0]
                            cmin, cmax = min(cols), max(cols)
                            for cc in range(cmin, cmax + 1):
                                out[rr, cc] = color
                        elif len(cols) == 1:
                            cc = cols[0]
                            rmin, rmax = min(rows), max(rows)
                            for rr in range(rmin, rmax + 1):
                                out[rr, cc] = color

    base = 9
    # determine activated cells and expand
    for ri in range(3):
        r0 = cell_rows[ri] + 1
        r1 = cell_rows[ri + 1]
        for cj in range(3):
            c0 = cell_cols[cj] + 1
            c1 = cell_cols[cj + 1]
            cell = out[r0:r1, c0:c1]
            vals = np.unique(cell[cell != base])
            vals = vals[(vals != 0)]
            vals = vals[(vals != 5)]
            if vals.size != 1:
                continue
            col = int(vals[0])
            if col not in templates:
                continue
            tm = templates[col]
            filled = _fill_cell_with_template(cell, tm, col, base)
            out[r0:r1, c0:c1] = filled

    return out


def generate_input():
    """Task 10 input generator.

    1) Build legend row with three color swatches and 4x4 templates.
    2) Build 3x3 lattice of base color 9 cells with separators.
    3) Place marker pixels so connects can create same-color activations in intermediate cells.
    """

    h = 19
    w = 21
    g = np.zeros((h, w), dtype=int)

    # legend
    template_colors = [2, 3, 1]
    patterns = [
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    ]

    for i, col in enumerate(template_colors):
        g[0, 2 + 7 * i] = col
        top = 1
        left = 1 + 7 * i
        for r in range(4):
            for c in range(4):
                if r < patterns[i].shape[0] and c < patterns[i].shape[1] and patterns[i][r, c]:
                    g[top + r, left + c] = 5

    # main separators and base color
    for r in [9, 14]:
        g[r, :] = 5
    for c in [7, 12]:
        g[:, c] = 5

    for r in range(5, 19):
        for c in range(3, 16):
            if g[r, c] != 5:
                g[r, c] = 9

    # local offsets tied to color
    offsets = {
        2: (1, 1),
        3: (1, 2),
        1: (2, 1),
    }

    # generate line segments per color
    cell_top = [5, 10, 15]
    cell_left = [3, 8, 13]

    for color in template_colors:
        or_, oc = offsets[color]
        used_cells = set()
        # place 2 marker pairs (can be absent) producing activation path
        for _ in range(2):
            if _RNG.random() < 0.5:
                # connect horizontally across two cells in same row
                row_cells = int(_RNG.integers(0, 3))
                c1 = int(_RNG.integers(0, 3))
                c2 = int(_RNG.integers(0, 3))
                if c1 == c2:
                    c2 = (c2 + 1) % 3
                s = sorted([c1, c2])
                cc1 = (cell_left[s[0]] + oc)
                cc2 = (cell_left[s[1]] + oc)
                rr = cell_top[row_cells] + or_
                key1 = (row_cells, s[0], or_, oc)
                key2 = (row_cells, s[1], or_, oc)
                if key1 in used_cells or key2 in used_cells:
                    continue
                g[rr, cc1] = color
                g[rr, cc2] = color
                used_cells.add((row_cells, s[0], or_, oc))
                used_cells.add((row_cells, s[1], or_, oc))
            else:
                col_cells = int(_RNG.integers(0, 3))
                r1 = int(_RNG.integers(0, 3))
                r2 = int(_RNG.integers(0, 3))
                if r1 == r2:
                    r2 = (r2 + 1) % 3
                s = sorted([r1, r2])
                rr1 = cell_top[s[0]] + or_
                rr2 = cell_top[s[1]] + or_
                cc = cell_left[col_cells] + oc
                g[rr1, cc] = color
                g[rr2, cc] = color

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
