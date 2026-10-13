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


def _is_vertical_bar(comp: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if max(cols) != min(cols):
        return False
    col = cols[0]
    r0, r1 = min(rows), max(rows)
    if len(comp) != (r1 - r0 + 1):
        return False
    return all((r, col) in set(comp) for r in range(r0, r1 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Vertical bars with one or more colored markers.
    Output:
        Bars recolored by cyclic slot shift, then marker rays extend left/right.
    Idea:
        Find 1-column vertical bars, rotate their colors by slot order,
        then draw horizontal rays per marker with same-color-bar blocking.
    Complexity:
        O(HW + B*W) where B is number of rays.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    out = inp.copy()

    comps = _components(inp != 0)
    bars = []
    for comp in comps:
        if _is_vertical_bar(comp):
            col = inp[comp[0][0], comp[0][1]]
            c = comp[0][1]
            rs = [r for r, _ in comp]
            bars.append((c, min(rs), max(rs), col, set(comp)))

    if len(bars) < 2:
        return inp.copy()

    bars.sort(key=lambda x: x[0])

    # cyclic color shift
    new_colors = [bars[(i - 1) % len(bars)][3] for i in range(len(bars))]

    # clear old bar pixels
    for _, r0, r1, _, cells in bars:
        c = next(iter(cells))[1]
        for r in range(r0, r1 + 1):
            out[r, c] = 0

    bar_pixels = {}
    for i, (c, r0, r1, _, cells) in enumerate(bars):
        nc = new_colors[i]
        for r in range(r0, r1 + 1):
            out[r, c] = nc
            bar_pixels[(r, c)] = nc

    # markers
    bar_set = set((r, c) for _, _, _, _, s in bars for r, c in s)
    markers = [(r, c) for r in range(H) for c in range(W) if inp[r, c] != 0 and (r, c) not in bar_set]
    markers.sort(key=lambda x: (x[0], x[1]))

    for mr, mc in markers:
        color = inp[mr, mc]

        # left ray
        c = mc - 1
        while c >= 0:
            if (mr, c) in bar_pixels:
                if bar_pixels[(mr, c)] == color:
                    break
                out[mr, c] = color
            else:
                out[mr, c] = color
            c -= 1

        # right ray
        c = mc + 1
        while c < W:
            if (mr, c) in bar_pixels:
                if bar_pixels[(mr, c)] == color:
                    break
                out[mr, c] = color
            else:
                out[mr, c] = color
            c += 1

        out[mr, mc] = inp[mr, mc]

    return out


def generate_input() -> np.ndarray:
    """
    Random generator for task 6.
    Returns:
        12x12 grid where cyclic shift introduces same-color blockers.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(6606)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    H = W = 12
    base_colors = [1, 2, 3, 4, 6, 8]

    for _ in range(120):
        g = np.zeros((H, W), dtype=int)
        cols = list(range(1, W - 1))
        rs.shuffle(cols)

        m = int(4 + rs.randint(0, 3))
        cols = sorted(cols[:m])
        picked_colors = list(rs.choice(base_colors, size=m, replace=False))

        bars = []
        for i, c in enumerate(cols):
            h = int(rs.randint(3, 9))
            top = H - h
            col = picked_colors[i % len(picked_colors)]
            for r in range(top, H):
                g[r, c] = col
            bars.append((c, top, H - 1, col))

        shifted = [bars[(i - 1) % m][3] for i in range(m)]
        if not shifted:
            continue

        marker_cells = set()
        # place 2-3 markers; first one guaranteed to match a shifted bar color on some bar row
        k = int(2 + rs.randint(0, 2))

        # choose one guaranteed blocking marker
        j = rs.randint(0, m)
        c_match = shifted[j]
        col_match = shifted[j]
        c0, r0, r1, _ = bars[j]
        r = int(rs.randint(r0, r1 + 1))
        c = int(rs.randint(0, W))
        if c == c0:
            c = 0 if c0 > 1 else W - 1
        g[r, c] = col_match
        marker_cells.add((r, c))

        # additional markers with any shifted colors
        all_rows = [(rs.randint(0, H), rs.randint(0, W)) for __ in range(k - 1)]
        for i in range(k - 1):
            color = shifted[int(rs.randint(0, len(shifted)))]
            rr, cc = all_rows[i]
            tries = 0
            while tries < 40 and (rr, cc) in marker_cells:
                rr, cc = rs.randint(0, H), rs.randint(0, W)
                tries += 1
            if tries >= 40:
                break
            # avoid placing on bars by shifting
            if cc == c0:
                cc = (cc + 2) % W if W > 2 else cc
            if cc < 1 or cc > W - 2:
                cc = 1
            g[rr, cc] = color
            marker_cells.add((rr, cc))

        return g

    return np.zeros((H, W), dtype=int)
