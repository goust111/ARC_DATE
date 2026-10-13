import numpy as np

_RNG = np.random.default_rng(20260102)


def _components(grid: np.ndarray):
    mask = grid > 0
    h, w = grid.shape
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
            qi = 0
            while qi < len(q):
                x, y = q[qi]
                qi += 1
                comp.append((x, y))
                for dr, dc in dirs:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _norm(coords):
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr.tolist()))


def _transforms(coords):
    arr = np.array(coords, dtype=int)
    x = arr[:, 0]
    y = arr[:, 1]
    variants = [
        (x, y),
        (x, -y),
        (-x, y),
        (-x, -y),
        (y, x),
        (y, -x),
        (-y, x),
        (-y, -x),
    ]
    out = set()
    for a, b in variants:
        out.add(_norm(np.stack([a, b], axis=1)))
    return out


def _connected_without_diag(points):
    return _norm(points)


def main(grid):
    """Input:
    16x14 网格，含分隔线与上下区域。

    Returns:
    先镜像后按形状匹配重涂色的网格。

    Algorithm:
    找出分隔线分区，提取图例形状并建立旋转/反射等价指纹；查询区橙色精灵体在查询区内水平镜像后匹配形状并改为图例颜色。

    Complexity:
    O(HW) 组件提取与形状匹配，形状规模固定下近似线性。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    divider_rows = [r for r in range(h) if np.all(g[r] == 5)]
    if divider_rows:
        div = divider_rows[0]
    else:
        div = h // 2

    top = g[:div]
    bottom = g[div + 1 :]
    q_top = 0
    q_row0 = div + 1

    # 图例映射：所有形状模板颜色
    legend_map = {}
    seen = set()
    for comp in _components(top):
        vals = sorted({int(top[r, c]) for r, c in comp if top[r, c] not in (0, 5)})
        if not vals:
            continue
        color = vals[0]
        pts = [(r, c) for r, c in comp]
        for t in _transforms(pts):
            if t not in legend_map:
                legend_map[t] = color

    # 输出拷贝
    out = g.copy()

    # 查询区每个橙色组件：先镜像，再匹配
    q_cells = (bottom == 7)
    if np.any(q_cells):
        q_components = []
        vis = np.zeros_like(q_cells, dtype=bool)
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for r in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                if not q_cells[r, c] or vis[r, c]:
                    continue
                q = [(r, c)]
                vis[r, c] = True
                comp = []
                qi = 0
                while qi < len(q):
                    cr, cc = q[qi]
                    qi += 1
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < bottom.shape[0] and 0 <= nc < bottom.shape[1] and q_cells[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                q_components.append(comp)

        # 清除原查询对象
        for comp in q_components:
            for r, c in comp:
                global_r = q_row0 + r
                global_c = c
                out[global_r, global_c] = 0

        # 重绘镜像后对象
        q_width = bottom.shape[1]
        for comp in q_components:
            mirrored = [(r, q_width - 1 - c) for r, c in comp]
            # 将镜像形状转为全局坐标
            global_m = [(r + q_row0, c) for r, c in mirrored]

            # 匹配（对镜像后的形状在旋转/反射等价域下）
            base = [(r, c - min(x for x, _ in mirrored)) for r, c in mirrored]
            cand = _transforms(base)
            color = 7
            for t in cand:
                if t in legend_map:
                    color = legend_map[t]
                    break

            for gr, gc in global_m:
                out[gr, gc] = color

    return out


def _poly_points(size):
    pts = [(0, 0)]
    while len(pts) < size:
        r, c = pts[-1]
        dirs = [
            (r + 1, c),
            (r - 1, c),
            (r, c + 1),
            (r, c - 1),
        ]
        nr, nc = dirs[int(_RNG.integers(4))]
        if (nr, nc) not in pts and min(nr, nc) >= 0:
            pts.append((nr, nc))
    return pts


def _normalize_local(pts):
    minr = min(r for r, _ in pts)
    minc = min(c for _, c in pts)
    return [(r - minr, c - minc) for r, c in pts]


def _try_place(canvas, cells, top_left, margin=1):
    h, w = canvas.shape
    rr, cc = top_left
    if rr < 0 or cc < 0:
        return False
    for r, c in cells:
        gr, gc = rr + r, cc + c
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        if canvas[gr, gc] != 0:
            return False
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and (dr != 0 or dc != 0):
                    if canvas[nr, nc] != 0:
                        return False
    return True


def generate_input():
    """随机且稳定地生成输入：16x14、水平分隔线、3~4 个图例与查询精灵。"""
    h, w = 16, 14
    g = np.zeros((h, w), dtype=int)
    div = 7
    g[div, :] = 5

    legend_n = int(_RNG.integers(3, 5))
    colors = [1, 2, 3, 4]
    _RNG.shuffle(colors)
    legend_colors = colors[:legend_n]

    templates = []
    for i in range(legend_n):
        size = int(_RNG.integers(3, 7))
        pts = _normalize_local(_poly_points(size))
        templates.append(pts)

    used = set()

    def place_component(pts, color, r0, c0):
        for r, c in pts:
            g[r0 + r, c0 + c] = color
            used.add((r0 + r, c0 + c))

    # 放图例到上半部分
    for i, (color, pts) in enumerate(zip(legend_colors, templates)):
        for _ in range(500):
            r0 = 1 + int(_RNG.integers(0, 5))
            c0 = 1 + int(_RNG.integers(0, 4)) + i * 3
            c0 = min(c0, 13 - (max(p[1] for p in pts) + 1))
            if _try_place(g[:div, :], pts, (r0, c0), margin=1):
                place_component(pts, color, r0, c0)
                break

    # 放查询区橙色精灵，形状为模板的变换版本
    start_row = div + 1
    query_area = g[start_row:]
    qh, qw = query_area.shape
    transform_options = [
        lambda p: [(x, y) for x, y in p],
        lambda p: [(x, -y) for x, y in p],
        lambda p: (-x, y) and [],
    ]

    for idx in range(legend_n):
        color, base = legend_colors[idx], templates[idx]
        pts = np.array(base, dtype=int)
        xs, ys = pts[:, 0], pts[:, 1]
        variants = [
            np.stack((xs, ys), axis=1),
            np.stack((ys, xs), axis=1),
            np.stack((xs, -ys), axis=1),
            np.stack((ys, -xs), axis=1),
            np.stack((-xs, ys), axis=1),
            np.stack((-ys, xs), axis=1),
            np.stack((-xs, -ys), axis=1),
            np.stack((-ys, -xs), axis=1),
        ]
        raw = variants[int(_RNG.integers(0, len(variants)))]
        shape = _normalize_local([tuple(map(int, p)) for p in raw])

        placed = False
        for _ in range(500):
            r0 = int(_RNG.integers(0, max(1, qh - max(p[0] for p in shape) - 1)))
            c0 = int(_RNG.integers(0, max(1, qw - max(p[1] for p in shape) - 1)))
            if all(g[start_row + r0 + rr, c0 + cc] == 0 for rr, cc in shape):
                # 简单间距检查
                bad = False
                for rr, cc in shape:
                    gr, gc = start_row + r0 + rr, c0 + cc
                    if any(0 <= gr + dr < h and 0 <= gc + dc < w and g[gr + dr, gc + dc] != 0 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        bad = True
                        break
                if not bad:
                    for rr, cc in shape:
                        g[start_row + r0 + rr, c0 + cc] = 7
                    placed = True
                    break
        if not placed:
            # 退化放置
            g[10, 1 + idx] = 7

    return g
