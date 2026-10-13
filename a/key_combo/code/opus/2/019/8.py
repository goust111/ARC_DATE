import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    comps = []
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
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        One blue horizontal line, one red vertical line, and several short horizontal segments.
    Output:
        Aligned-line rectangle weaved with blue/red checkerboard and short segments overlaid.
    Idea:
        Select median row segment by row coordinate, align all segment columns to it,
        then fill shared row-span x col-span rectangle by parity weaving.
    Complexity:
        O(HW + S*L), S short segments, L total span.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    out = inp.copy()

    # detect blue long row and red long column
    blue_row = 0
    best_len = -1
    for r in range(H):
        cnt = np.count_nonzero(inp[r] == 1)
        if cnt > best_len:
            best_len = cnt
            blue_row = r
    red_col = 0
    best_len = -1
    for c in range(W):
        cnt = np.count_nonzero(inp[:, c] == 2)
        if cnt > best_len:
            best_len = cnt
            red_col = c

    # detect short segments in other colors
    segs = []
    colors_used = set()
    for color in np.unique(inp):
        if color in (0, 1, 2):
            continue
        comps = _components(inp == color)
        for comp in comps:
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            if len(set(rs)) == 1 and 3 <= len(comp) <= 6:
                rs0 = rs[0]
                c0, c1 = min(cs), max(cs)
                segs.append((rs0, c0, c1, color))
                colors_used.add(color)

    if not segs:
        return inp.copy()

    segs.sort(key=lambda x: x[0])
    med_i = (len(segs) - 1) // 2
    ref_r, ref_l, ref_r2, ref_col = segs[med_i]
    ref_len = ref_r2 - ref_l + 1
    # align every short segment
    aligned = []
    for r, l0, r0, col in segs:
        nl = ref_l
        nr = ref_l + ref_len - 1
        aligned.append((r, nl, nr, col))

    r_top = min(r for r, _, _, _ in aligned)
    r_bottom = max(r for r, _, _, _ in aligned)
    c_left = ref_l
    c_right = ref_l + ref_len - 1

    # weave fill
    for r in range(r_top, r_bottom + 1):
        for c in range(c_left, c_right + 1):
            out[r, c] = 1 if ((r + c) % 2 == 0) else 2

    # overlay aligned segments
    for r, l, rr, col in aligned:
        for c in range(l, rr + 1):
            if 0 <= c < W:
                out[r, c] = col

    return out


def generate_input() -> np.ndarray:
    """
    Random input generator for task 8.
    Returns:
        14x14 grid with crossing long blue/red bars and short horizontal segments.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(8808)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    H = W = 14
    g = np.zeros((H, W), dtype=int)

    blue_row = int(2 + rs.randint(0, H - 4))
    red_col = int(2 + rs.randint(0, W - 4))
    # draw near-full lines
    l0 = rs.randint(0, 2)
    l1 = W - rs.randint(0, 2)
    for c in range(l0, l1):
        g[blue_row, c] = 1
    for r in range(0, H):
        g[r, red_col] = 2

    # 3-5 short segments, odd count for simple median selection
    num = int(3 + rs.randint(0, 3))
    if num % 2 == 0:
        num += 1
    available_rows = list(range(1, H - 1))
    rs.shuffle(available_rows)
    rows = sorted(available_rows[:num])

    colors = [3, 4, 5, 6, 8, 9]
    rs.shuffle(colors)
    seg_color_iter = iter(colors)

    for i in range(num):
        r = rows[i]
        length = int(3 + rs.randint(0, 4))
        # include red intersection in reference-like manner
        left_low = max(0, red_col - length + 1)
        right_high = min(W - length, red_col)
        if left_low > right_high:
            left_low, right_high = 0, W - length
        l = int(rs.randint(left_low, right_high + 1))
        col = next(seg_color_iter)
        for c in range(l, l + length):
            g[r, c] = col

    return g
