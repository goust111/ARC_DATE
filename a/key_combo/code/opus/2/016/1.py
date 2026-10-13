import numpy as np

_RNG = np.random.default_rng(20260101)


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
            comps.append((comp, touches_border))
    return comps


def main(grid):
    """Input:
    2D 整数网格。

    Returns:
    按射线交叉分区规则处理后的二维数组。

    Algorithm:
    先从每个前景像素向四方向画射线；若同一格同时存在水平与垂直射线，按源像素行列优先级决策颜色；再找不接边界的零区域并用其上方水平射线颜色填充。

    Complexity:
    O(HW) 逐格处理与一次 BFS，空间 O(HW)。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    points = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if g[r, c] != 0]
    if not points:
        return g.copy()

    color_h = np.zeros((h, w), dtype=int)
    color_v = np.zeros((h, w), dtype=int)
    src_h = np.full((h, w, 2), -1, dtype=int)
    src_v = np.full((h, w, 2), -1, dtype=int)

    for r0, c0, col in points:
        for cc in range(w):
            color_h[r0, cc] = col
            src_h[r0, cc] = (r0, c0)
        for rr in range(h):
            color_v[rr, c0] = col
            src_v[rr, c0] = (r0, c0)

    line = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            has_h = src_h[r, c, 0] != -1
            has_v = src_v[r, c, 0] != -1
            if not has_h and not has_v:
                continue
            if has_h and not has_v:
                line[r, c] = color_h[r, c]
            elif has_v and not has_h:
                line[r, c] = color_v[r, c]
            else:
                srh, sch = src_h[r, c]
                svr, svc = src_v[r, c]
                if srh < svr or (srh == svr and sch < svc):
                    line[r, c] = color_h[r, c]
                else:
                    line[r, c] = color_v[r, c]

    for r, c, col in points:
        line[r, c] = col

    filled = line.copy()
    for comp, touches in _components(filled == 0):
        if touches:
            continue
        top = min(r for r, _ in comp)
        if top == 0:
            continue
        top_colors = []
        for _, cc in comp:
            if filled[top - 1, cc] != 0:
                top_colors.append(filled[top - 1, cc])
        if not top_colors:
            continue
        # 选最多出现者，避免生成边界奇异点。
        fill_color = max(set(top_colors), key=top_colors.count)
        for rr, cc in comp:
            filled[rr, cc] = fill_color

    return filled


def generate_input():
    """随机且稳定地生成输入：
    1) 12x12 黑底
    2) 3~4 个互不同行列且距 2 以上的前景点
    """
    h = w = 12
    g = np.zeros((h, w), dtype=int)

    k = int(_RNG.integers(3, 5))
    colors = [1, 2, 3, 4]
    _RNG.shuffle(colors)
    cols = colors[:k]

    points = []
    used_rows = []
    used_cols = []
    attempts = 0
    while len(points) < k and attempts < 5000:
        attempts += 1
        rr = int(_RNG.integers(2, h - 2))
        cc = int(_RNG.integers(2, w - 2))
        if any(abs(rr - rv) < 2 for rv in used_rows):
            continue
        if any(abs(cc - cv) < 2 for cv in used_cols):
            continue
        used_rows.append(rr)
        used_cols.append(cc)
        points.append((rr, cc))

    while len(points) < k:
        rr = 2 + len(points)
        cc = 2 + len(points)
        points.append((rr, cc))

    for (r, c), col in zip(points, cols):
        g[r, c] = col

    return g
