import numpy as np

_RNG = np.random.default_rng(20260110)


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
            touches = False
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches = True
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append((q, touches))
    return comps


def main(grid):
    """Input:
    约 15x15，左侧/上侧 A 区含灰色样本单元，右下区域含角落编码。

    Returns:
    从 A 读取颜色序列，并按角落编码层级映射为同心环图案。

    Algorithm:
    提取 A 中不接边缘的灰色单元颜色序列；解析 B 区角落编码层号并反向映射颜色；在 B 区每个层环填色。

    Complexity:
    O(HW) 的区域搜索与输出构建。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    # 1) 分割区域：优先使用纵向灰分割线
    sep_col = None
    sep_row = None
    for c in range(w):
        cnt = np.sum(g[:, c] == 5)
        if cnt >= h - 2:
            sep_col = c
            break
    if sep_col is None:
        for r in range(h):
            cnt = np.sum(g[r, :] == 5)
            if cnt >= w - 2:
                sep_row = r
                break

    if sep_col is not None:
        A = g[:, :sep_col]
        B = g[:, sep_col + 1 :]
    elif sep_row is not None:
        A = g[:sep_row, :]
        B = g[sep_row + 1 :, :]
    else:
        A = g[:, : w // 2]
        B = g[:, w // 2 :]

    # 2) 读取 A 区序列
    seq_colors = []
    for comp, touch in _components(A == 5):
        if touch:
            continue
        color_set = sorted({int(A[r, c]) for r, c in comp if A[r, c] not in (0, 5)})
        if not color_set:
            continue
        color = color_set[0]
        seq_colors.append((min(r for r, _ in comp), min(c for _, c in comp), color))

    if not seq_colors:
        return g.copy()

    seq_colors.sort(key=lambda x: (x[0], x[1]))
    colors = [c for _, _, c in seq_colors]
    N = len(colors)

    # 3) 解析 B 区角落编码
    ors = np.argwhere(B == 7)
    if ors.size == 0:
        return B.copy()
    br, bc = B.shape
    layers = set()
    for r, c in ors:
        d = max(abs((br - 1) - r), abs((bc - 1) - c))
        layer = d // 2 + 1
        if layer > 0:
            layers.add(layer)

    layers = sorted(layers)
    if len(layers) != N:
        # 容错：若编码层数与序列不一致，以最小 N 对齐
        N = min(N, len(layers) if layers else 1)

    layers = sorted(layers)
    color_map = {}
    for i, layer in enumerate(layers):
        ci = colors[-(i + 1)] if i < len(colors) else colors[0]
        color_map[layer] = ci

    # 4) 角落编码展开为同心图案
    out = np.zeros((br, bc), dtype=int)
    for r in range(br):
        for c in range(bc):
            d = max(abs((br - 1) - r), abs((bc - 1) - c))
            layer = d // 2 + 1
            if layer in color_map:
                out[r, c] = color_map[layer]

    return out


def generate_input():
    """随机且稳定地生成输入：A 区序列 + B 区角落编码。"""
    h = 15
    w = 16
    g = np.zeros((h, w), dtype=int)

    sep = 5
    g[:, sep] = 5

    N = int(_RNG.integers(3, 6))
    colors = [1, 2, 3, 4, 6]
    _RNG.shuffle(colors)
    use = colors[:N]

    for idx in range(N):
        rs = 1 + idx * 4
        if rs + 2 >= h:
            break
        cc = 1
        g[rs:rs+3, cc:cc+3] = 5
        g[rs + 1, cc + 1] = use[idx % len(use)]

    Br = h
    Bc = w - sep - 1
    br, bc = Br, Bc
    for d in range(1, N + 1):
        off = 2 * d - 1
        points = [
            (br - 1 - off, bc - 1),
            (br - 1, bc - 1 - off),
        ]
        for r, c in points:
            if 0 <= r < br and 0 <= c < bc:
                g[r - 0, sep + 1 + c] = 7

    # 简化布设：放 N 层在 B 区右下角对应偏移
    for d in range(1, N + 1):
        off = 2 * d - 1
        r0 = h - 1 - off
        c0 = w - sep - 1 - 1 - off
        if 0 <= r0 < h and 0 <= c0 < Bc:
            g[r0, sep + 1 + c0] = 7

    return g
