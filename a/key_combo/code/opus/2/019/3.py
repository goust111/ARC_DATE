import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = 1
            comp = []
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = 1
                        q.append((nx, ny))
            out.append(comp)
    return out


def _crop_pattern(grid: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> tuple[np.ndarray, tuple[int, int]]:
    """crop a cell interior to tight bbox of foreground pixels"""
    sub = grid[r1:r2, c1:c2]
    ys, xs = np.where(sub != 0)
    if len(ys) == 0:
        return np.zeros((0, 0), dtype=int), (0, 0)
    tr0, tr1 = ys.min(), ys.max() + 1
    tc0, tc1 = xs.min(), xs.max() + 1
    patt = sub[tr0:tr1, tc0:tc1].copy()
    return patt, (r1 + tr0, c1 + tc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Grid with orange(7) separator rows/cols dividing cells and colored row markers in left column.
    Output:
        Same-size grid where each non-left cell is stamped by the duplicated template.
    Procedure:
        Parse separator grid, find exact template appearing twice, clear cells,
        then center and recolor template with each row's marker color.
    Complexity:
        O(HW + K*B^2), where K is cell count and B max cell-bbox size.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape

    # separator indices
    rows = [i for i in range(H) if np.all(inp[i, :] == 7)]
    cols = [j for j in range(W) if np.all(inp[:, j] == 7)]
    if not rows or not cols:
        return inp.copy()

    rows = sorted(rows)
    cols = sorted(cols)

    # left column marker colors
    marker_colors = {}
    for ri in range(len(rows) - 1):
        r0, r1 = rows[ri] + 1, rows[ri + 1]
        c0 = cols[0] + 1
        c1 = cols[1]
        cell = inp[r0:r1, c0:c1]
        ys, xs = np.where(cell != 0)
        if len(ys) == 0:
            marker_colors[ri] = 1
            continue
        marker_colors[ri] = int(cell[ys[0], xs[0]])

    # sprite patterns in non-left cells
    patterns = {}
    for ri in range(len(rows) - 1):
        for ci in range(1, len(cols) - 1):
            r0, r1 = rows[ri] + 1, rows[ri + 1]
            c0, c1 = cols[ci] + 1, cols[ci + 1]
            patt, _ = _crop_pattern(inp, r0, r1, c0, c1)
            if patt.size == 0:
                continue
            key = patt.tobytes(), patt.shape
            patterns.setdefault(key, []).append((ri, ci, patt))

    template_entry = None
    for key, v in patterns.items():
        if len(v) == 2:
            template_entry = (key, v[0][2])
            break
    if template_entry is None:
        return inp.copy()

    template = template_entry[1]
    out = inp.copy()

    # clear all non-left internal cells
    for ri in range(len(rows) - 1):
        for ci in range(1, len(cols) - 1):
            out[rows[ri] + 1:rows[ri + 1], cols[ci] + 1:cols[ci + 1]] = 0

    # stamp templates
    for ri in range(len(rows) - 1):
        r0, r1 = rows[ri] + 1, rows[ri + 1]
        c0, c1 = cols[1] + 1, cols[2]
        h_int, w_int = r1 - r0, c1 - c0
        th, tw = template.shape

        off_r = (h_int - th) // 2
        off_c = (w_int - tw) // 2
        rr = r0 + max(0, off_r)
        cc = c0 + max(0, off_c)

        for t in range(len(rows) - 1):
            rr2 = r0
            rr2 = max(rr2, rows[t] + 1)
        for t in range(1, len(cols) - 1):
            rr = rows[ri] + 1
            cc = cols[t] + 1
            ih, iw = r1 - (rows[ri] + 1), cols[t + 1] - (cols[t] + 1)
            off_r = max(0, (ih - th) // 2)
            off_c = max(0, (iw - tw) // 2)
            tr = rows[ri] + 1 + off_r
            tc = cols[t] + 1 + off_c
            color = marker_colors.get(ri, 1)
            for i in range(th):
                for j in range(tw):
                    if template[i, j] == 0:
                        continue
                    if tr + i < rows[ri + 1] and tc + j < cols[t + 1]:
                        out[tr + i, tc + j] = color

    return out


def generate_input() -> np.ndarray:
    """
    Random input generator for task 3.
    Returns:
        A fixed-grid game-like picture with separator lines and one template repeated twice.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(3303)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    rows = 4
    cols = 4
    h = int(3 + rs.randint(0, 3))
    w = int(3 + rs.randint(0, 3))
    H = rows * h + (rows + 1)
    W = cols * w + (cols + 1)

    g = np.zeros((H, W), dtype=int)

    # boundaries and separators
    for i in range(rows + 1):
        g[i * (h + 1), :] = 7
    for j in range(cols + 1):
        g[:, j * (w + 1)] = 7

    # markers in left column
    marker_colors = [1, 2, 3, 4]
    for r in range(rows):
        rr = r * (h + 1) + 1
        rc0 = 0 * (w + 1) + 1
        rr0 = rr + rs.randint(0, h)
        cc0 = rc0 + rs.randint(0, w)
        g[rr0, cc0] = marker_colors[r % len(marker_colors)]

    # build non-left cell templates
    def make_template():
        ph = int(rs.randint(2, h))
        pw = int(rs.randint(2, w))
        rr = rs.randint(0, h - ph + 1)
        cc = rs.randint(0, w - pw + 1)
        mask = np.zeros((h, w), dtype=int)
        # connected random walk inside template box
        cells = {(rr, cc)}
        frontier = [(rr, cc)]
        target = int(rs.randint(2, 6))
        for _ in range(target - 1):
            x, y = frontier[rs.randint(0, len(frontier))]
            cand = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            cand = [(a, b) for a, b in cand if 0 <= a < h and 0 <= b < w and (a, b) not in cells]
            if not cand:
                break
            a, b = cand[rs.randint(0, len(cand))]
            cells.add((int(a), int(b)))
            frontier.append((int(a), int(b)))
        color = int(rs.randint(1, 5))
        for a, b in cells:
            mask[a, b] = color
        return mask

    templates = []
    positions = []
    for r in range(rows):
        for c in range(1, cols):
            positions.append((r, c))

    template = make_template()
    repeat_two = rs.choice(len(positions), size=2, replace=False)
    for idx, (ri, ci) in enumerate(positions):
        if idx in repeat_two:
            patt = template
        else:
            patt = make_template()
            tries = 0
            while patt.tobytes() == template.tobytes() and tries < 20:
                patt = make_template()
                tries += 1
        t = patt
        ph, pw = t.shape
        r0 = ri * (h + 1) + 1
        c0 = ci * (w + 1) + 1
        ys, xs = np.where(t != 0)
        if len(ys) == 0:
            continue
        min_r, min_c = ys.min(), xs.min()
        place_r = r0 + rs.randint(0, max(1, h - (ys.max() - min_r + 1) + 1))
        place_c = c0 + rs.randint(0, max(1, w - (xs.max() - min_c + 1) + 1))
        g2 = t[min_r : ys.max() + 1, min_c : xs.max() + 1]
        sr, sc = g2.shape
        g[place_r:place_r + sr, place_c:place_c + sc] = g2

    return g
