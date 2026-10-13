import numpy as np

_RNG = np.random.default_rng(20260103)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            touches_border = False
            qi = 0
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                comp.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                comps.append(comp)
    return comps


def main(grid):
    """Input:
    约 18x18 灰色网格表格与单色控制条。

    Returns:
    按每个表格单元长度生成升降序列并 wrap 成二维输出。

    Algorithm:
    找到不接边界的灰色封闭区域作为单元；读取每个单元中的控制条颜色与长度；生成 1..n..1 斜坡序列并插入 1/2 个灰分隔符后拼接，最后按 C*5 换行。

    Complexity:
    O(HW + L) ，L 为输出长度。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    gray = g == 5

    comps = _components(gray)
    # 由边界框行主序排序
    info = []
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        cells = g[r0:r1+1, c0:c1+1]
        colors = sorted({int(v) for v in cells.flat if v not in (0, 5)})
        if not colors:
            continue
        color = colors[0]
        n = int(np.sum(cells == color))
        n = max(1, min(4, n))
        info.append((r0, c0, color, n))

    info.sort(key=lambda x: (x[0], x[1]))
    if not info:
        return g.copy()

    seq = []
    cols = max(1, len(set(c0 for _, c0, _, _ in info)))

    for idx, (_, _, color, n) in enumerate(info):
        if idx > 0:
            seq.extend([5, 5])
        up = list(range(1, n + 1))
        dn = list(range(n - 1, 0, -1))
        for run in up + dn:
            seq.extend([color] * run)
            seq.append(5)
        if seq and seq[-1] == 5:
            seq.pop()

    if not seq:
        return np.zeros((1, cols * 5), dtype=int)

    width = cols * 5
    height = (len(seq) + width - 1) // width
    out = np.full((height, width), 5, dtype=int)
    p = 0
    for i in range(height):
        for j in range(width):
            if p >= len(seq):
                break
            out[i, j] = seq[p]
            p += 1

    return out


def generate_input():
    """随机且稳定地生成输入：2x3 或 3x2 由灰色单元构成的表格。"""
    h = w = 18
    g = np.zeros((h, w), dtype=int)
    layout = int(_RNG.integers(0, 2))
    rows = 2 if layout == 0 else 3
    cols = 3 if layout == 0 else 2
    total = rows * cols
    if total < 3:
        rows, cols = 2, 2

    cell_h = int(_RNG.integers(3, 6))
    cell_w = int(_RNG.integers(4, 7))
    gap = 1

    start_r = 1
    start_c = 1
    colors = [1, 2, 3, 4, 6, 7]
    _RNG.shuffle(colors)

    used = []
    for idx in range(total):
        rr = idx // cols
        cc = idx % cols
        r0 = start_r + rr * (cell_h + 1)
        c0 = start_c + cc * (cell_w + 1)
        if r0 + cell_h >= h - 1 or c0 + cell_w >= w - 1:
            continue
        # 画一个小灰块作为表格单元
        g[r0:r0+cell_h, c0:c0+cell_w] = 5
        color = colors[idx % len(colors)]
        bar_len = int(_RNG.integers(1, 5))
        bar_row = r0 + 1
        c1 = c0 + 1
        c2 = c1 + bar_len
        if c2 > c0 + cell_w - 1:
            c2 = c0 + cell_w - 1
        g[bar_row, c1:c2+1] = color
        used.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))

    return g
