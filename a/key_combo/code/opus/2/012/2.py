import numpy as np

_rng = np.random.default_rng(202602)


def _l_shape_orientations(base_r, base_c, ori):
    # 在 2x2 方块中缺角朝向，放置三个颜色 8 像素
    cells = []
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    miss = ori
    for idx, (dr, dc) in enumerate(coords):
        if idx == miss:
            continue
        cells.append((base_r + dr, base_c + dc))
    return cells, [(base_r + coords[miss][0], base_c + coords[miss][1])]


def _cc_with_8(grid):
    h, w = grid.shape
    vis = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 8 or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == 8:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def generate_input():
    """
    生成 12x12 网格：若干青色 2x2 缺角 L 簇 + 几个独立标记像素。
    约束为每个簇缺角有且仅有唯一最近标记色，供主函数按距离补全。
    """
    H = W = 12
    while True:
        k_clusters = int(_rng.integers(3, 6))
        k_markers = int(_rng.integers(3, 5))
        colors = [2, 3, 4, 1]

        grid = np.zeros((H, W), dtype=int)
        boxes = []
        clusters = []
        # 2x2 框边界至少留 2 像素间隔
        attempts = 0
        ok = True
        for _ in range(k_clusters):
            placed = False
            while attempts < 200 and not placed:
                attempts += 1
                r = int(_rng.integers(0, H - 1))
                c = int(_rng.integers(0, W - 1))
                if r + 1 >= H or c + 1 >= W:
                    continue
                br = (r - 2, c - 2, r + 3, c + 3)
                overlap = False
                for rr0, cc0, rr1, cc1 in boxes:
                    if not (br[2] <= rr0 or br[0] >= rr1 or br[3] <= cc0 or br[1] >= cc1):
                        overlap = True
                        break
                if overlap:
                    continue
                ori = int(_rng.integers(0, 4))
                cells, miss = _l_shape_orientations(r, c, ori)
                if any(grid[cr, cc] != 0 for cr, cc in cells):
                    continue
                for cr, cc in cells:
                    grid[cr, cc] = 8
                boxes.append((r - 2, c - 2, r + 3, c + 3))
                clusters.append(miss[0])
                placed = True
            if not placed:
                ok = False
                break
        if not ok:
            continue

        # 放置标记像素
        marker_cells = []
        marker_colors = colors[:k_markers]
        tries = 0
        while len(marker_cells) < k_markers and tries < 200:
            tries += 1
            r = int(_rng.integers(0, H))
            c = int(_rng.integers(0, W))
            if grid[r, c] != 0:
                continue
            # 不与青色簇四邻接
            bad = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 8:
                    bad = True
                    break
            if bad:
                continue
            marker_cells.append((r, c, marker_colors[len(marker_cells)]))

        if len(marker_cells) != k_markers:
            continue

        for r, c, col in marker_cells:
            grid[r, c] = col

        comps = _cc_with_8(grid)
        if len(comps) != k_clusters:
            continue

        missing = []
        for comp in comps:
            if len(comp) != 3:
                ok = False
                break
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 != 1 or c1 - c0 != 1:
                ok = False
                break
            miss = None
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if grid[rr, cc] != 8:
                        miss = (rr, cc)
            if miss is None:
                ok = False
                break
            missing.append(miss)

        if not ok:
            continue

        markers = [(r, c, col) for r, c, col in marker_cells]
        unique_ok = True
        for miss in missing:
            d_best = None
            count_best = None
            for r, c, col in markers:
                d = (miss[0] - r) ** 2 + (miss[1] - c) ** 2
                if d_best is None or d < d_best:
                    d_best = d
                    count_best = 1
                elif d == d_best:
                    count_best += 1
            if count_best != 1:
                unique_ok = False
                break
        if not unique_ok:
            continue

        return grid


def main(grid):
    """
    输入: 黑底网格，仅含青色(8)的三像素 L 簇、若干标记像素(1/2/3/4)与背景。

    输出: 每个 L 缺角补全为其最近标记色，随后擦除所有标记像素，其余保持不变。

    算法: 1) 识别每个 4 邻域 8 组件；
    2) 对每个组件求 2x2 内唯一缺角；
    3) 按欧氏平方距离选择唯一最近标记并赋色；
    4) 把所有非青色标记恢复为 0。

    复杂度: O(H*W*K)，K 为标记像素数。
    """
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    comps = []
    vis = np.zeros_like(g, dtype=bool)

    for r in range(h):
        for c in range(w):
            if g[r, c] != 8 or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            for cr, cc in q:
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and g[nr, nc] == 8:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) != 3:
                continue
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            miss = None
            for rr in (r0, r1):
                for cc in (c0, c1):
                    if g[rr, cc] == 0:
                        miss = (rr, cc)
            if miss is None:
                continue
            comps.append((comp, miss))

    markers = [(r, c, g[r, c]) for r in range(h) for c in range(w) if g[r, c] in (1, 2, 3, 4)]

    for _, miss in comps:
        best_d = None
        best_color = 0
        for mr, mc, col in markers:
            d = (miss[0] - mr) ** 2 + (miss[1] - mc) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_color = col
        g[miss] = best_color

    for r, c, _ in markers:
        g[r, c] = 0

    return g
