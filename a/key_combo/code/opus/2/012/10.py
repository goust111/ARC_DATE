import numpy as np

_rng = np.random.default_rng(202610)


def _components(grid, color):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if vis[r, c] or grid[r, c] != color:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == color:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_border(grid, color):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if vis[r, c] or grid[r, c] != color:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = [(r, c)]
            for cr, cc in q:
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == color:
                        vis[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if len(comp) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
                continue
            return color, r0, r1, c0, c1, len(comp)
    return None


def generate_input():
    """
    生成 12x12 网格：两个不同色的单像素矩形框 A(红) 与 B(蓝)，
    黄(4)标记分布在框内外。
    框内标记数分别独立计数并渲染到 3x3 输出（叠加有绿）。
    """
    H = W = 12
    while True:
        g = np.zeros((H, W), dtype=int)
        # 随机两个非重叠框
        h1 = int(_rng.integers(4, 7)); w1 = int(_rng.integers(4, 7))
        h2 = int(_rng.integers(4, 7)); w2 = int(_rng.integers(4, 7))

        r1 = int(_rng.integers(0, H - h1))
        c1 = int(_rng.integers(0, W - w1))
        r2 = int(_rng.integers(0, H - h2))
        c2 = int(_rng.integers(0, W - w2))

        if not (r1 + h1 <= r2 or r2 + h2 <= r1 or c1 + w1 <= c2 or c2 + w2 <= c1):
            # 允许少量接触但不重叠
            if r1 == r2 and c1 == c2:
                continue

        for c in range(c1, c1 + w1):
            g[r1, c] = 2
            g[r1 + h1 - 1, c] = 2
        for r in range(r1, r1 + h1):
            g[r, c1] = 2
            g[r, c1 + w1 - 1] = 2

        for c in range(c2, c2 + w2):
            g[r2, c] = 1
            g[r2 + h2 - 1, c] = 1
        for r in range(r2, r2 + h2):
            g[r, c2] = 1
            g[r, c2 + w2 - 1] = 1

        # 统计框内可放 marker 的最大值约束
        inter = np.zeros((H, W), dtype=bool)
        for rr in range(r1 + 1, r1 + h1 - 1):
            for cc in range(c1 + 1, c1 + w1 - 1):
                inter[rr, cc] = True

        inside = [(rr, cc) for rr in range(r2 + 1, r2 + h2 - 1) for cc in range(c2 + 1, c2 + w2 - 1)]

        # 选择计数值 1~9，且不相等
        a = int(_rng.integers(1, 10))
        b = int(_rng.integers(1, 10))
        if b == a:
            b = (b % 9) + 1
        max_a = len(inter)
        max_b = len(inside)
        if a > max_a or b > max_b or a <= 0 or b <= 0:
            continue

        # 放置黄点
        positions_a = [(r, c) for r in range(r1 + 1, r1 + h1 - 1) for c in range(c1 + 1, c1 + w1 - 1)]
        positions_b = [(r, c) for r in range(r2 + 1, r2 + h2 - 1) for c in range(c2 + 1, c2 + w2 - 1)]
        _rng.shuffle(positions_a)
        _rng.shuffle(positions_b)

        for rr, cc in positions_a[:a]:
            g[rr, cc] = 4
        for rr, cc in positions_b[:b]:
            g[rr, cc] = 4

        # 干扰项
        outside = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 0]
        _rng.shuffle(outside)
        for rr, cc in outside[: int(_rng.integers(2, 6))]:
            g[rr, cc] = 4

        return g


def main(grid):
    """
    输入: 12x12 里有红框 A、蓝框 B 和黄标记(4)。

    输出: 统计两框内黄标记数量后分别映射到 3x3 区域并叠加。

    算法: 1) 识别 A 与 B 的严格内部并计数 N_A、N_B；
    2) 在 3x3 按行优先绘制前 N_A 个为 A 层、前 N_B 个为 B 层；
    3) A∩B -> 3，A-only -> 2，B-only -> 1，其余 0。

    复杂度: O(H*W + 9)。
    """
    g = np.array(grid, dtype=int)

    a = _is_rect_border(g, 2)
    b = _is_rect_border(g, 1)
    if a is None or b is None:
        return np.zeros((3, 3), dtype=int)

    _, ar0, ar1, ac0, ac1, _ = a
    _, br0, br1, bc0, bc1, _ = b

    nA = 0
    nB = 0
    for r in range(ar0 + 1, ar1):
        for c in range(ac0 + 1, ac1):
            if g[r, c] == 4:
                nA += 1
    for r in range(br0 + 1, br1):
        for c in range(bc0 + 1, bc1):
            if g[r, c] == 4:
                nB += 1

    nA = int(np.clip(nA, 0, 9))
    nB = int(np.clip(nB, 0, 9))

    pos = [(r, c) for r in range(3) for c in range(3)]
    out = np.zeros((3, 3), dtype=int)

    for idx, (r, c) in enumerate(pos):
        a_layer = idx < nA
        b_layer = idx < nB
        if a_layer and b_layer:
            out[r, c] = 3
        elif a_layer:
            out[r, c] = 2
        elif b_layer:
            out[r, c] = 1

    return out
