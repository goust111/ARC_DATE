import numpy as np

_RNG = np.random.default_rng(20260707)

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def main(grid):
    """
1) 识别首尾两条蓝线与中间红色分隔行，计算蓝线重叠列区间。
2) 按较长线的一侧在两线间填充黄色到重叠列范围。
3) 指示像素位于黄色区域边缘，作为黄色延伸方向起点。
4) 黄色区域向指示方向延申到蓝线/红线/边界前停，再沿同方向画等长绿色线。
"""

    out = np.array(grid, dtype=int, copy=True)
    H, W = out.shape

    top_r = 0
    bot_r = H - 1
    top_cols = np.where(out[top_r] == 1)[0]
    bot_cols = np.where(out[bot_r] == 1)[0]
    if len(top_cols) == 0 or len(bot_cols) == 0:
        return out

    t0, t1 = int(top_cols.min()), int(top_cols.max())
    b0, b1 = int(bot_cols.min()), int(bot_cols.max())
    o0, o1 = max(t0, b0), min(t1, b1)
    if o0 > o1:
        return out

    sep_rows = np.where(np.all(out == 2, axis=1))[0]
    if len(sep_rows) == 0:
        return out
    sep = int(sep_rows[0])

    top_long = (t1 - t0 + 1) >= (b1 - b0 + 1)
    fill_cols = range(o0, o1 + 1)

    if top_long:
        fill_rs = range(1, sep)
    else:
        fill_rs = range(sep + 1, H - 1)

    for r in fill_rs:
        for c in fill_cols:
            if out[r, c] == 0:
                out[r, c] = 4

    inds = np.argwhere(out == 3)
    if len(inds) != 1:
        return out
    sr, sc = map(int, inds[0])

    fill_c0, fill_c1 = o0, o1
    fill_r0 = fill_rs.start
    fill_r1 = fill_rs.stop - 1

    if sc == fill_c0:
        dr, dc = 0, -1
    elif sc == fill_c1:
        dr, dc = 0, 1
    elif sr == fill_r0:
        dr, dc = -1, 0
    elif sr == fill_r1:
        dr, dc = 1, 0
    else:
        return out

    length = 0
    nr, nc = sr + dr, sc + dc
    while 0 <= nr < H and 0 <= nc < W:
        if dr in (1, -1):
            blocked = False
            for cc in fill_cols:
                if out[nr, cc] in (1, 2):
                    blocked = True
                    break
            if blocked:
                break
            for cc in fill_cols:
                out[nr, cc] = 4
        else:
            blocked = False
            for rr in fill_rs:
                if out[rr, nc] in (1, 2):
                    blocked = True
                    break
            if blocked:
                break
            for rr in fill_rs:
                out[rr, nc] = 4
        length += 1
        nr += dr
        nc += dc

    nr, nc = sr, sc
    for _ in range(length):
        nr += dr
        nc += dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if out[nr, nc] in (1, 2):
            break
        out[nr, nc] = 3

    return out


def generate_input(seed=None):
    rng = _rng(seed)
    H, W = 12, 15

    while True:
        grid = np.zeros((H, W), dtype=int)
        sep = int(rng.integers(4, 8))

        top_len = int(rng.integers(4, 11))
        bot_len = int(rng.integers(4, 11))
        top_start = int(rng.integers(0, W - top_len + 1))
        bot_start = int(rng.integers(0, W - bot_len + 1))

        top_cols = np.arange(top_start, top_start + top_len)
        bot_cols = np.arange(bot_start, bot_start + bot_len)
        overlap = np.intersect1d(top_cols, bot_cols)
        if len(overlap) < 2:
            continue

        for c in top_cols:
            grid[0, c] = 1
        for c in bot_cols:
            grid[H - 1, c] = 1
        grid[sep, :] = 2

        o0, o1 = int(overlap.min()), int(overlap.max())
        top_long = top_len >= bot_len

        if top_long:
            # 放左/右边缘
            if rng.random() < 0.5:
                indicator_c = o0
            else:
                indicator_c = o1
            indicator_r = int(rng.integers(1, sep))
            if indicator_r < 1 or indicator_r >= sep:
                continue
        else:
            if rng.random() < 0.5:
                indicator_c = o0
            else:
                indicator_c = o1
            indicator_r = int(rng.integers(sep + 1, H - 1))
            if indicator_r <= sep or indicator_r >= H - 1:
                continue

        if not (0 <= indicator_r < H and 0 <= indicator_c < W):
            continue
        if grid[indicator_r, indicator_c] != 0:
            continue
        grid[indicator_r, indicator_c] = 3

        return grid


if __name__ == "__main__":
    pass
