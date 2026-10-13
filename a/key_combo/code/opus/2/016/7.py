import numpy as np

_RNG = np.random.default_rng(20260107)


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
            qi = 0
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(q)
    return comps


def main(grid):
    """Input:
    约 15x20，包含若干实心矩形与空白背景。

    Returns:
    矩形上移自身高度后绘制转折射线的网格。

    Algorithm:
    识别实心矩形，整体上移其高度；按新下边顺序在每个右下角画右延长+向下转折线，后一个标记行以上为终点。

    Complexity:
    组件扫描 O(HW)，线段绘制 O(HW)。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    rect_infos = []
    for comp in _components(g != 0):
        color = int(g[comp[0][0], comp[0][1]])
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if len(set(g[r, c] for r, c in comp)) != 1:
            continue
        if (r1 - r0 + 1) == 0 or (c1 - c0 + 1) == 0:
            continue
        rect_infos.append((r0, c0, r1, c1, color))

    rect_infos.sort(key=lambda x: x[0])
    out = g.copy()
    moved = []

    # 清空原始并绘制
    for r0, c0, r1, c1, col in rect_infos:
        out[r0 : r1 + 1, c0 : c1 + 1] = 0

    for r0, c0, r1, c1, col in rect_infos:
        height = r1 - r0 + 1
        nr0 = r0 - height
        nr1 = r1 - height
        nc0, nc1 = c0, c1
        if nr0 < 0:
            nr0 = 0
            nr1 = height - 1
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = col
        br, bc = nr1, c1
        moved.append((br, bc, nr0, nr1, col))

    rect_mask = out != 0
    moved.sort(key=lambda x: x[0])

    for idx, (br, bc, nr0, nr1, col) in enumerate(moved):
        # 水平线
        for cc in range(bc, w):
            if not rect_mask[nr1, cc]:
                out[nr1, cc] = col
        # 竖线
        if idx + 1 < len(moved):
            target_row = moved[idx + 1][0] - 1
        else:
            target_row = h - 1
        for rr in range(nr1 + 1, target_row + 1):
            if 0 <= rr < h and not rect_mask[rr, w - 1]:
                out[rr, w - 1] = col

    return out


def _put_rect(g, tr, lc, hgt, wid, col):
    g[tr : tr + hgt, lc : lc + wid] = col


def generate_input():
    """随机且稳定地生成输入：3~4 个不同高宽、不同颜色的实心矩形。"""
    h, w = 15, 20
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4, 5]
    cnt = int(_RNG.integers(3, 5))

    rows = sorted(_RNG.choice(np.arange(4, 9), size=cnt, replace=False))
    widths = [int(_RNG.integers(2, 5)) for _ in range(cnt)]
    heights = [int(_RNG.integers(2, 6)) for _ in range(cnt)]
    for i in range(cnt):
        hr = heights[i]
        top = rows[i] + 6
        col = int(_RNG.integers(0, 10))
        wid = widths[i]
        _put_rect(g, top, col, hr, wid, colors[i % len(colors)])

    return g
