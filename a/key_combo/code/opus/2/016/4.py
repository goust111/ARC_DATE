import numpy as np

_RNG = np.random.default_rng(20260104)


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
            qi = 0
            while qi < len(q):
                cr, cc = q[qi]
                qi += 1
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_axis(points):
    coords = set(points)
    by_row = {}
    by_col = {}
    for r, c in points:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    best_axis = None
    axis_kind = None
    axis_len = 0
    for r, cs in by_row.items():
        cs_sorted = sorted(cs)
        if len(cs_sorted) >= 3 and cs_sorted == list(range(cs_sorted[0], cs_sorted[-1] + 1)):
            if len(cs_sorted) > axis_len:
                axis_len = len(cs_sorted)
                best_axis = ("row", r, cs_sorted[0], cs_sorted[-1])
                axis_kind = "H"
    for c, rs in by_col.items():
        rs_sorted = sorted(rs)
        if len(rs_sorted) >= 3 and rs_sorted == list(range(rs_sorted[0], rs_sorted[-1] + 1)):
            if len(rs_sorted) > axis_len:
                axis_len = len(rs_sorted)
                best_axis = ("col", c, rs_sorted[0], rs_sorted[-1])
                axis_kind = "V"

    if best_axis is None:
        return None

    if axis_kind == "H":
        _, rr, c0, c1 = best_axis
        axis_cells = [(rr, c) for c in range(c0, c1 + 1)]
        return "H", rr, axis_cells
    _, cc, r0, r1 = best_axis
    axis_cells = [(r, cc) for r in range(r0, r1 + 1)]
    return "V", cc, axis_cells


def main(grid):
    """Input:
    约 14x14，若干含黄轴线和控制像素的对象。

    Returns:
    先按轴对称补全再擦除对象局部边框的二维网格。

    Algorithm:
    每个对象取黄轴与黄控制标记确定源侧并镜像补全；补全后按对象新边界框去掉最外一圈像素。

    Complexity:
    O(HW) 组件检测与逐对象反射。
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    objs = _components(g != 0)
    for comp in objs:
        colorset = {g[r, c] for r, c in comp}
        if 4 not in colorset:
            continue
        axis_info = _find_axis([(r, c) for r, c in comp if g[r, c] == 4])
        if axis_info is None:
            continue
        kind, axis, axis_cells = axis_info
        control = [p for p in comp if g[p[0], p[1]] == 4 and p not in axis_cells]
        if not control:
            continue
        cr, cc = control[0]

        source_points = []
        for r, c in comp:
            val = g[r, c]
            if val == 4:
                continue
            if kind == "H":
                if cr < axis and r < axis:
                    source_points.append((r, c, val))
                elif cr > axis and r > axis:
                    source_points.append((r, c, val))
            else:
                if cc < axis and c < axis:
                    source_points.append((r, c, val))
                elif cc > axis and c > axis:
                    source_points.append((r, c, val))

        for r, c, col in source_points:
            if kind == "H":
                nr = axis * 2 - r
                if 0 <= nr < h:
                    out[nr, c] = col
            else:
                nc = axis * 2 - c
                if 0 <= nc < w:
                    out[r, nc] = col

    # 二次计算局部边界框并擦除边框
    for comp in _components(out != 0):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        out[r0, c0:c1 + 1] = 0
        out[r1, c0:c1 + 1] = 0
        out[r0:r1 + 1, c0] = 0
        out[r0:r1 + 1, c1] = 0

    return out


def generate_input():
    """随机且稳定地生成输入：2~3 个含黄轴线对象，含控制标记与源侧形状。"""
    h = w = 14
    g = np.zeros((h, w), dtype=int)
    obj_count = int(_RNG.integers(2, 4))
    colors = [1, 2]
    occ = np.zeros((h, w), dtype=bool)

    for _ in range(obj_count):
        color = colors[int(_RNG.integers(0, len(colors)))]
        kind = "H" if _RNG.random() < 0.5 else "V"

        for _ in range(200):
            if kind == "H":
                axis_r = int(_RNG.integers(4, 10))
                c0 = int(_RNG.integers(1, 10))
                axis_len = int(_RNG.integers(4, 7))
                if c0 + axis_len >= 13:
                    continue
                axis_cells = [(axis_r, c0 + i) for i in range(axis_len)]
                if any(occ[r, c] for r, c in axis_cells):
                    continue
                if _RNG.random() < 0.5:
                    # 源侧在上
                    base_r = axis_r - 4
                    if base_r < 0:
                        continue
                    shape = [(base_r + dr, c0 + i) for i in range(axis_len) for dr in range(1, 2)]
                    control = (axis_r - 1, c0)
                else:
                    # 源侧在下
                    base_r = axis_r + 2
                    if base_r + 2 >= h:
                        continue
                    shape = [(axis_r + 1 + dr, c0 + i) for i in range(axis_len) for dr in range(1, 2)]
                    control = (axis_r + 1, c0)
            else:
                axis_c = int(_RNG.integers(4, 10))
                r0 = int(_RNG.integers(1, 10))
                axis_len = int(_RNG.integers(4, 7))
                if r0 + axis_len >= 13:
                    continue
                axis_cells = [(r0 + i, axis_c) for i in range(axis_len)]
                if any(occ[r, c] for r, c in axis_cells):
                    continue
                if _RNG.random() < 0.5:
                    control = (r0, axis_c - 1)
                    if axis_c - 3 < 0:
                        continue
                    shape = [(r0 + i, axis_c - 2) for i in range(axis_len)]
                else:
                    control = (r0, axis_c + 1)
                    if axis_c + 3 >= w:
                        continue
                    shape = [(r0 + i, axis_c + 2) for i in range(axis_len)]

            pts = list(dict.fromkeys(axis_cells + [control] + shape))
            if len(pts) < 5:
                continue
            bad = False
            for r, c in pts:
                for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and occ[rr, cc]:
                        if (rr, cc) not in pts:
                            bad = True
                            break
                if bad:
                    break
            if bad:
                continue

            for r, c in axis_cells:
                g[r, c] = 4
                occ[r, c] = True
            cr, cc = control
            g[cr, cc] = 4
            for r, c in shape:
                g[r, c] = color
                occ[r, c] = True
            for rr, cc in shape:
                occ[rr, cc] = True
            for rr, cc in pts:
                occ[rr, cc] = True
            break

    return g
