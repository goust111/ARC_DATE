import numpy as np

_RNG = np.random.default_rng(20260108)


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


def _monotone_union(start, end, blocked):
    sr, sc = start
    er, ec = end
    if sr == er and sc == ec:
        return {start}
    dr = 0
    dc = 0
    if er > sr:
        dr = 1
    elif er < sr:
        dr = -1
    if ec > sc:
        dc = 1
    elif ec < sc:
        dc = -1
    if dr == 0 and dc == 0:
        return set()

    H, W = blocked.shape
    rows = list(range(sr, er + dr, dr or 1)) if dr else [sr] * (abs(ec - sc) + 1)
    cols = list(range(sc, ec + dc, dc or 1)) if dc else [sc] * (abs(er - sr) + 1)

    # 映射为非负索引
    R = len(rows)
    C = len(cols)

    f1 = np.zeros((R, C), dtype=bool)
    f2 = np.zeros((R, C), dtype=bool)

    for i in range(R):
        for j in range(C):
            r = rows[i]
            c = cols[j]
            if blocked[r, c]:
                continue
            if i == 0 and j == 0:
                f1[i, j] = True
            else:
                can = False
                if i > 0 and f1[i - 1, j]:
                    can = True
                if j > 0 and f1[i, j - 1]:
                    can = True
                f1[i, j] = can

    for i in range(R - 1, -1, -1):
        for j in range(C - 1, -1, -1):
            r = rows[i]
            c = cols[j]
            if blocked[r, c]:
                continue
            if i == R - 1 and j == C - 1:
                f2[i, j] = True
            else:
                can = False
                if i + 1 < R and f2[i + 1, j]:
                    can = True
                if j + 1 < C and f2[i, j + 1]:
                    can = True
                f2[i, j] = can

    union = set()
    for i in range(R):
        for j in range(C):
            if f1[i, j] and f2[i, j]:
                union.add((rows[i], cols[j]))

    return union


def main(grid):
    """Input:
    约 14x14，含两个门框、灰色障碍和空白背景。

    Returns:
    先算两门洞间单调最短路径并集，再在转弯格按角补位印蓝色块。

    Algorithm:
    检测两个门洞外部入口；在障碍上做单调最短路径并集；并集中的转弯格按方向在对角补刻画二点。

    Complexity:
    O(HW) 的门洞识别与并集 DP。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    rings = []
    for r in range(h - 2):
        for c in range(w - 2):
            if g[r + 1, c + 1] != 9:
                continue
            border = [
                (r, c), (r, c + 1), (r, c + 2),
                (r + 1, c), (r + 1, c + 2),
                (r + 2, c), (r + 2, c + 1), (r + 2, c + 2),
            ]
            if any(g[x, y] != 6 for x, y in border):
                continue
            neighbors = [(r, c + 1), (r + 1, c), (r + 1, c + 2), (r + 2, c + 1)]
            holes = [p for p in neighbors if g[p] == 0]
            if len(holes) != 1:
                continue
            hole = holes[0]
            hr, hc = hole
            if hole == (r, c + 1):
                out = (r - 1, hc)
            elif hole == (r + 2, c + 1):
                out = (r + 3, hc)
            elif hole == (r + 1, c):
                out = (hr, c - 1)
            else:
                out = (hr, c + 3)
            rings.append((hole, out))

    if len(rings) < 2:
        return g.copy()

    start = rings[0][1]
    end = rings[1][1]

    blocked = g != 0
    union = _monotone_union(start, end, blocked)
    out = g.copy()
    for r, c in union:
        out[r, c] = 4

    for r, c in union:
        neigh = set()
        if (r, c - 1) in union:
            neigh.add("L")
        if (r, c + 1) in union:
            neigh.add("R")
        if (r - 1, c) in union:
            neigh.add("U")
        if (r + 1, c) in union:
            neigh.add("D")
        has_h = ("L" in neigh) or ("R" in neigh)
        has_v = ("U" in neigh) or ("D" in neigh)
        if not has_h or not has_v:
            continue
        if ("L" in neigh and "R" in neigh) or ("U" in neigh and "D" in neigh):
            continue

        stamp = []
        if "R" in neigh and "D" in neigh:
            stamp = [(r - 1, c + 1), (r + 1, c - 1)]
        elif "R" in neigh and "U" in neigh:
            stamp = [(r + 1, c + 1), (r - 1, c - 1)]
        elif "L" in neigh and "D" in neigh:
            stamp = [(r - 1, c - 1), (r + 1, c + 1)]
        elif "L" in neigh and "U" in neigh:
            stamp = [(r + 1, c - 1), (r - 1, c + 1)]

        for rr, cc in stamp:
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                out[rr, cc] = 8

    return out


def generate_input():
    """随机且稳定地生成输入：门框与障碍形成至少一条含转弯的单调最短走廊。"""
    h = w = 14
    g = np.zeros((h, w), dtype=int)

    def _ring(top, left, miss):
        g[top + 1, left + 1] = 9
        g[top, left + 1] = 6 if miss != "up" else 0
        g[top + 1, left] = 6 if miss != "left" else 0
        g[top + 1, left + 2] = 6 if miss != "right" else 0
        g[top + 2, left + 1] = 6 if miss != "down" else 0
        g[top, left] = 6
        g[top, left + 2] = 6
        g[top + 2, left] = 6
        g[top + 2, left + 2] = 6

        if miss == "left":
            return (top + 1, left), (top + 1, left - 1)
        if miss == "right":
            return (top + 1, left + 2), (top + 1, left + 3)
        if miss == "up":
            return (top, left + 1), (top - 1, left + 1)
        return (top + 2, left + 1), (top + 3, left + 1)

    sr = int(_RNG.integers(2, 4))
    er = int(_RNG.integers(8, 10))
    sc = int(_RNG.integers(1, 4))
    ec = int(_RNG.integers(8, 11))

    # 门洞外部入口：起点在左上象限，终点在右下，保持可达单调路径
    _, d1 = _ring(2, sc, "left")
    _, d2 = _ring(er, ec, "right")
    sdr, sdc = d1
    edr, edc = d2

    # 随机加障碍，确保起点与终点不被挡住
    mode = int(_RNG.integers(0, 3))
    if mode == 0:
        for c in range(4, min(edc, w - 1)):
            if c == sdc:
                continue
            g[sdr, c] = 5
        for r in range(sdr + 1, min(edr + 1, h)):
            g[r, sdc + 3] = 5
    elif mode == 1:
        for r in range(sr + 1, min(edr, h - 1)):
            g[r, sdc + 1] = 5
        for c in range(sdc, min(edc, w - 1)):
            g[sdr + 2, c] = 5
    else:
        for r in range(sdr + 2, min(edr, h - 1)):
            g[r, sdc + 2] = 5
        for c in range(sdc, min(edc, w - 1)):
            g[sdr + 1, c] = 5

    # 清出一条基准通路（保证最短路径并集非空）
    for c in range(sdc, edc + 1):
        if 0 <= sdr < h and 0 <= c < w:
            g[sdr, c] = 0
    for r in range(sdr + 1, edr + 1):
        if 0 <= r < h and 0 <= edc < w:
            g[r, edc] = 0
    for r in range(sdr, min(sdr + 2, h)):
        if 0 <= r < h and 0 <= edc < w:
            g[r, edc] = 0

    return g
