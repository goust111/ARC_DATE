# Combo variant of 35ab12c3 (index 29)
# combo_spec: D7/0+D18/0
# Name: 控制像素指定形状类型+标记指定起始顶点
#
# Design analysis for the fused variant
# a) Concrete behavior and trace
#    - 每个真正的 shape 颜色现在都有两层边缘编码：
#      1. 一个“同色边缘像素”通过纯背景走廊指向内部某个顶点，这个顶点是起始顶点。
#      2. 该同色边缘像素旁边紧贴一个“异色边缘控制像素”，其颜色固定编码形状类型：
#         `1 -> line`, `2 -> triangle`, `3 -> rectangle`。
#    - solver 先把同色边缘像素从顶点集合里剥离，再读取它旁边的异色控制像素确定是开折线
#      还是闭合多边形，最后沿同色走廊确定从哪个顶点开始旋转极角序列。
#    - 焦点对象固定做成“四个矩形角点 + line 控制色 + 两个 seeded 顶点”的情形：
#      若忽略 D7，会把四点闭合成矩形；若忽略 D18，会从更小的诱饵 seeded 顶点开画，
#      从而缺边位置和阴影颜色都改变。融合后两层控制必须同时解读才得到正确输出。
# b) Complexity calibration
#    - 形状类型只保留 3 档：line / triangle / rectangle。这样控制色映射在少量样例里可重复观察，
#      不会引入过多类别。
#    - 所有样本都至少含一个焦点 line-shape，确保 D7 的“显式指定开折线”不是摆设；同时焦点对象
#      还带 active/decoy 两个 seeded 顶点，确保 D18 的起点控制也不是摆设。
#    - 第二个 shape 继续使用同样的双层边缘标记，但不强制阴影；这样画面不会过乱，同时让控制机制
#      在同一网格内至少出现两次。
# c) Implicit assumptions audit of original main()
#    - 原 `pts_rc = np.argwhere(g == col)` 假设同色像素全是顶点。这里不成立：每个真实 shape 色组
#      额外含一个同色边缘控制像素，因此必须先剥离它。
#    - 原逻辑把 `len(cycle) > 2` 一律当闭合多边形。这里不成立：line 控制色要求同样的 3/4 个顶点
#      只连相邻边而不闭合。
#    - 原逻辑用“有异色邻居的最小顶点”选 start。这里不成立：真正的 start 由同色边缘像素通过
#      纯背景走廊显式指定。
#    - 新 solver 还额外假设：每个真实 shape 色组恰有一个同色边缘控制像素；它旁边恰有一个合法
#      类型控制色；同色走廊上没有别的非背景像素。generator 会逐项保证这些前提。
# d) Placement legality
#    - 所有 shape patch 都放在离边界至少 1 格的位置，再单独布置边缘控制对。
#    - 生成器为每个同色边缘像素保留一条水平或垂直纯背景走廊，并把走廊记为 reserved，后续对象
#      不能占用。
#    - patch 的碰撞检测使用最终 `patch_out`，因此主形状与阴影都参与避碰；边缘控制像素和走廊
#      也单独检查，避免重叠或越界。
# e) Visual observability
#    - 人类能直接看到：每个 shape 颜色在边界多出一个同色像素，旁边还有一个不同颜色的控制像素。
#      同色像素告诉你“从哪里开始”，异色控制像素告诉你“这是折线还是闭合图形”。
#    - 焦点对象使用“矩形四角却只画三条边”的例子，和原题的“点数隐式决定闭合”形成明显对比。
# f) Visual cleanliness
#    - 双层控制全部压在边界，主体和阴影留在内部，避免把标记混进几何区域。
#    - 每个网格只放 2 个 shape，且只有 1 个焦点 shape 带 active/decoy 双阴影锚点，因此输出仍然清楚。
# g) Color inferability
#    - 颜色值只做可见的固定类别映射：`1/2/3` 三个控制色在所有样本中语义恒定。
#    - shape 色和 shadow 色保持“局部语义”，不参与任何隐藏排序。
# h) Variant metric used after implementation
#    - 指标 1: `main(inp) != _solve_variant(inp, use_shape_control=False, use_start_control=True)`，
#      说明形状类型控制真实生效。
#    - 指标 2: `main(inp) != _solve_variant(inp, use_shape_control=True, use_start_control=False)`，
#      说明起始顶点控制真实生效。
#    - generator 会强制这两个指标都成立，拒绝退化样本。
# i) New assumptions introduced by the modified main()
#    - `TYPE_COLOR_TO_SHAPE` 中的控制色只出现在边界，且不与真实 shape 色重合。
#    - 真正 shape 色在非边界至少有 2 个像素，因此 solver 可以把“只有边界控制像素/单个阴影像素”
#      的颜色安全地排除在 shape 之外。
#    - 对于 `triangle` / `rectangle` 控制色，顶点数分别固定为 3 / 4；generator 只生成这些合法组合。

from __future__ import annotations

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG = 0
TYPE_COLOR_TO_SHAPE = {1: "line", 2: "triangle", 3: "rectangle"}
SHAPE_TO_TYPE_COLOR = {v: k for k, v in TYPE_COLOR_TO_SHAPE.items()}
TYPE_COLORS = sorted(TYPE_COLOR_TO_SHAPE)
SHAPE_COLORS = [4, 6, 7, 8]
PLAIN_SHAPE_TYPE_CYCLE = ("triangle", "rectangle")
_generate_call_index = 0

# ---- CFG: canvas ----
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: placement / retries ----
PLACEMENT_PADDING = 1
PATCH_RETRIES = 80
PLACEMENT_RETRIES = 140
GENERATE_RETRIES = 500

# ---- CFG: geometry ----
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 8)
RECT_W_RANGE = (2, 8)


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _border_dirs(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    return dirs


def _start_from_control_marker(
    g: np.ndarray,
    col: int,
    control: tuple[int, int],
    vertices: list[tuple[int, int]],
    bg: int,
) -> tuple[int, int] | None:
    h, w = g.shape
    vertex_set = set(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in _border_dirs(control[0], control[1], h, w):
        rr = control[0] + dr
        cc = control[1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v == bg:
                rr += dr
                cc += dc
                continue
            if v == col and (rr, cc) in vertex_set:
                candidates.append((rr, cc))
            break
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return min(candidates)
    return None


def _shape_type_from_marker(g: np.ndarray, marker: tuple[int, int]) -> str | None:
    h, w = g.shape
    r0, c0 = marker
    found: list[str] = []
    for dr, dc in DIR4:
        rr = r0 + dr
        cc = c0 + dc
        if 0 <= rr < h and 0 <= cc < w and _is_border_cell(rr, cc, h, w):
            v = int(g[rr, cc])
            if v in TYPE_COLOR_TO_SHAPE:
                found.append(TYPE_COLOR_TO_SHAPE[v])
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return sorted(found)[0]
    return None


def _fallback_start_idx(g: np.ndarray, cycle: list[tuple[int, int]], col: int, bg: int) -> int:
    h, w = g.shape
    seed_idxs = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break
    if seed_idxs:
        return min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    return min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))


def _shape_edges_from_cycle(cycle: list[tuple[int, int]], shape_type: str | None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    n = len(cycle)
    if n < 2:
        return []
    if shape_type == "line":
        return [(cycle[i], cycle[i + 1]) for i in range(n - 1)]
    return [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]


def _resolved_shape_type(shape_type: str | None, n_vertices: int) -> str:
    if shape_type == "line" and n_vertices >= 2:
        return "line"
    if shape_type == "triangle" and n_vertices == 3:
        return "triangle"
    if shape_type == "rectangle" and n_vertices == 4:
        return "rectangle"
    if n_vertices == 2:
        return "line"
    return "rectangle"


def _solve_variant(grid: np.ndarray, *, use_shape_control: bool, use_start_control: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_all = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        interior_pts = [p for p in pts_all if not _is_border_cell(p[0], p[1], h, w)]
        if len(interior_pts) < 2:
            continue

        border_pts = [p for p in pts_all if _is_border_cell(p[0], p[1], h, w)]
        control = border_pts[0] if len(border_pts) == 1 else None
        pts_rc = [p for p in pts_all if p != control]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        start_rc = None
        if use_start_control and control is not None:
            start_rc = _start_from_control_marker(g, col, control, cycle, bg)
        if start_rc is not None and start_rc in cycle:
            start = cycle.index(start_rc)
        else:
            start = _fallback_start_idx(g, cycle, col, bg)
        cycle = cycle[start:] + cycle[:start]

        shape_type = None
        if use_shape_control and control is not None:
            shape_type = _shape_type_from_marker(g, control)
        shape_type = _resolved_shape_type(shape_type, len(cycle))

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        edges = _shape_edges_from_cycle(cycle, shape_type)
        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read two border-level control layers, then connect each color's interior vertices.

    Input structure:
    - Real shape colors appear as 3-4 interior vertex pixels plus exactly one same-color border marker.
    - Each same-color border marker has a neighboring border control pixel whose color encodes
      `line` / `triangle` / `rectangle`.
    - Optional non-shape colors still appear as single shadow-anchor pixels near one chosen vertex.

    Rule:
    - Remove the same-color border marker from the vertex set.
    - Use its adjacent control color to decide whether the angle-ordered cycle is open (`line`)
      or closed (`triangle` / `rectangle`).
    - Follow the border marker's empty row/column corridor inward to pick the start vertex, then
      use that vertex both to rotate the cycle and to choose which neighboring shadow color is copied.

    Key insight:
    - The border encodes two independent decisions: the adjacent foreign-color pixel chooses the
      shape family, while the same-color marker chooses where the cycle starts.
    """
    return _solve_variant(grid, use_shape_control=True, use_start_control=True)


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_triangle_vertices() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle_vertices() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_line_vertices() -> list[tuple[int, int]]:
    if np.random.rand() < 0.5:
        return _build_triangle_vertices()
    return _build_rectangle_vertices()


def _shape_cells_from_cycle(cycle: list[tuple[int, int]], shape_type: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges_from_cycle(cycle, shape_type):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _helper_offsets_for_vertex(
    vertices: list[tuple[int, int]],
    shape_type: str,
    start_vertex: tuple[int, int],
    vertex_idx: int,
    used_positions: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cycle = _angle_cycle(vertices)
    start = cycle.index(start_vertex)
    cycle = cycle[start:] + cycle[:start]
    cells = _shape_cells_from_cycle(cycle, shape_type)
    directions = [_step_dir(a[0], a[1], b[0], b[1]) for a, b in _shape_edges_from_cycle(cycle, shape_type)]
    anchor = vertices[vertex_idx]
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in cells or pos in used_positions:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    return candidates


def _render_shape_local(
    inp: np.ndarray,
    shape_color: int,
    vertices: list[tuple[int, int]],
    start_vertex: tuple[int, int],
    shape_type: str,
) -> np.ndarray:
    out = inp.copy()
    cycle = _angle_cycle(vertices)
    start = cycle.index(start_vertex)
    cycle = cycle[start:] + cycle[:start]

    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[0]
    h, w = inp.shape
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(inp[rr, cc])
            if v != BG and v != shape_color:
                carried.append((drn, dcn, v))

    edges = _shape_edges_from_cycle(cycle, shape_type)
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(inp, r0, c0, r1, c1, BG, allow={shape_color}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))

    for (r0, c0), (r1, c1) in drawable_edges:
        _draw_line_bg_only(out, r0, c0, r1, c1, shape_color, BG)

    for (r0, c0), (r1, c1) in drawable_edges:
        drs, dcs = _step_dir(r0, c0, r1, c1)
        for drn, dcn, ncol in carried:
            if drs * dcn - dcs * drn == 0:
                continue
            rr0 = r0 + drn
            cc0 = c0 + dcn
            rr1 = r1 + drn
            cc1 = c1 + dcn
            if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                continue
            if not _aligned_8(rr0, cc0, rr1, cc1):
                continue
            if not _clear_in_original(inp, rr0, cc0, rr1, cc1, BG, allow=set()):
                continue
            _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, BG)

    return out


def _patch_nonbg_coords(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in np.argwhere(patch != BG)]


def _can_place_patch(
    out: np.ndarray,
    reserved: np.ndarray,
    patch_coords: list[tuple[int, int]],
    r0: int,
    c0: int,
) -> bool:
    h, w = out.shape
    for pr, pc in patch_coords:
        rr = r0 + pr
        cc = c0 + pc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if out[rr, cc] != BG or reserved[rr, cc]:
            return False
        r_lo = max(0, rr - PLACEMENT_PADDING)
        r_hi = min(h, rr + PLACEMENT_PADDING + 1)
        c_lo = max(0, cc - PLACEMENT_PADDING)
        c_hi = min(w, cc + PLACEMENT_PADDING + 1)
        if np.any(out[r_lo:r_hi, c_lo:c_hi] != BG):
            return False
    return True


def _control_options_for_vertex(
    inp: np.ndarray,
    reserved: np.ndarray,
    translated_nonbg: set[tuple[int, int]],
    vertex: tuple[int, int],
) -> list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = inp.shape
    vr, vc = vertex
    options: list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []

    def path_clear(path: list[tuple[int, int]]) -> bool:
        for rr, cc in path:
            if (rr, cc) in translated_nonbg:
                return False
            if inp[rr, cc] != BG or reserved[rr, cc]:
                return False
        return True

    def type_positions_for(marker: tuple[int, int]) -> list[tuple[int, int]]:
        mr, mc = marker
        if mr == 0 or mr == h - 1:
            candidates = [(mr, mc - 1), (mr, mc + 1)]
        else:
            candidates = [(mr - 1, mc), (mr + 1, mc)]
        result: list[tuple[int, int]] = []
        for rr, cc in candidates:
            if 0 <= rr < h and 0 <= cc < w and _is_border_cell(rr, cc, h, w):
                if (rr, cc) in translated_nonbg or reserved[rr, cc]:
                    continue
                if inp[rr, cc] != BG:
                    continue
                result.append((rr, cc))
        return result

    if 0 < vc < w - 1:
        marker = (0, vc)
        path = [(rr, vc) for rr in range(1, vr)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            for tp in type_positions_for(marker):
                options.append((marker, [tp], path))
        marker = (h - 1, vc)
        path = [(rr, vc) for rr in range(vr + 1, h - 1)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            for tp in type_positions_for(marker):
                options.append((marker, [tp], path))

    if 0 < vr < h - 1:
        marker = (vr, 0)
        path = [(vr, cc) for cc in range(1, vc)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            for tp in type_positions_for(marker):
                options.append((marker, [tp], path))
        marker = (vr, w - 1)
        path = [(vr, cc) for cc in range(vc + 1, w - 1)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            for tp in type_positions_for(marker):
                options.append((marker, [tp], path))

    return options


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    spec: dict,
    shape_color: int,
) -> dict | None:
    H, W = inp.shape
    patch_in = spec["patch_in"]
    patch_out = spec["patch_out"]
    vertices_local = spec["vertices"]
    start_vertex_local = spec["start_vertex"]
    shape_type = spec["shape_type"]
    type_color = SHAPE_TO_TYPE_COLOR[shape_type]
    h, w = patch_out.shape
    if H - h <= 1 or W - w <= 1:
        return None

    patch_coords = _patch_nonbg_coords(patch_out)
    patch_in_coords = _patch_nonbg_coords(patch_in)
    for _ in range(PLACEMENT_RETRIES):
        r0 = int(np.random.randint(1, H - h))
        c0 = int(np.random.randint(1, W - w))
        if not _can_place_patch(out, reserved, patch_coords, r0, c0):
            continue
        translated_nonbg = {(r0 + pr, c0 + pc) for pr, pc in patch_in_coords}
        if any(reserved[rr, cc] for rr, cc in translated_nonbg):
            continue
        start_global = (r0 + start_vertex_local[0], c0 + start_vertex_local[1])
        options = _control_options_for_vertex(inp, reserved, translated_nonbg, start_global)
        if not options:
            continue
        pick = options[int(np.random.randint(len(options)))]
        marker, type_positions, path = pick
        type_pos = type_positions[0]
        paste(out, patch_out, r0, c0, transparent=BG)
        paste(inp, patch_in, r0, c0, transparent=BG)
        inp[marker] = int(shape_color)
        out[marker] = int(shape_color)
        inp[type_pos] = int(type_color)
        out[type_pos] = int(type_color)
        for rr, cc in path:
            reserved[rr, cc] = True
        return {
            "marker": marker,
            "type_pos": type_pos,
            "shape_type": shape_type,
            "start_vertex": start_global,
            "focus": bool(spec.get("focus", False)),
            "decoy_vertex": None if spec.get("decoy_vertex") is None else (r0 + spec["decoy_vertex"][0], c0 + spec["decoy_vertex"][1]),
        }
    return None


def _make_focus_shape_patch(shape_color: int, active_color: int, decoy_color: int) -> dict | None:
    for _ in range(PATCH_RETRIES):
        vertices = _build_rectangle_vertices()
        cycle0 = _angle_cycle(vertices)
        pair_candidates = [
            (active_idx, decoy_idx)
            for active_idx, active in enumerate(vertices)
            for decoy_idx, decoy in enumerate(vertices)
            if active_idx != decoy_idx and decoy < active
        ]
        np.random.shuffle(pair_candidates)
        for active_idx, decoy_idx in pair_candidates:
            start_vertex = vertices[active_idx]
            active_offsets = _helper_offsets_for_vertex(vertices, "line", start_vertex, active_idx, set())
            np.random.shuffle(active_offsets)
            for active_offset in active_offsets:
                active_pos = (vertices[active_idx][0] + active_offset[0], vertices[active_idx][1] + active_offset[1])
                decoy_offsets = _helper_offsets_for_vertex(vertices, "line", start_vertex, decoy_idx, {active_pos})
                np.random.shuffle(decoy_offsets)
                for decoy_offset in decoy_offsets:
                    decoy_pos = (vertices[decoy_idx][0] + decoy_offset[0], vertices[decoy_idx][1] + decoy_offset[1])
                    cycle = cycle0[:]
                    start = cycle.index(start_vertex)
                    cycle = cycle[start:] + cycle[:start]
                    shape_cells = _shape_cells_from_cycle(cycle, "line")
                    shadow_cells = {(r + active_offset[0], c + active_offset[1]) for r, c in shape_cells}
                    coords = list(shape_cells) + list(vertices) + list(shadow_cells) + [active_pos, decoy_pos]
                    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
                    h, w = r1 - r0 + 1, c1 - c0 + 1
                    inp = np.full((h, w), BG, dtype=int)
                    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
                    for r, c in shifted_vertices:
                        inp[r, c] = shape_color
                    active_pos_s = (active_pos[0] - r0, active_pos[1] - c0)
                    decoy_pos_s = (decoy_pos[0] - r0, decoy_pos[1] - c0)
                    if inp[active_pos_s] != BG or inp[decoy_pos_s] != BG:
                        continue
                    inp[active_pos_s] = active_color
                    inp[decoy_pos_s] = decoy_color
                    start_vertex_s = shifted_vertices[active_idx]
                    out = _render_shape_local(inp, int(shape_color), shifted_vertices, start_vertex_s, "line")
                    return {
                        "patch_in": inp,
                        "patch_out": out,
                        "vertices": shifted_vertices,
                        "start_vertex": start_vertex_s,
                        "shape_type": "line",
                        "focus": True,
                        "decoy_vertex": shifted_vertices[decoy_idx],
                    }
    return None


def _make_plain_shape_patch(shape_color: int, forced_shape_type: str) -> dict | None:
    for _ in range(PATCH_RETRIES):
        shape_type = forced_shape_type
        if shape_type == "line":
            vertices = _build_line_vertices()
        elif shape_type == "triangle":
            vertices = _build_triangle_vertices()
        else:
            vertices = _build_rectangle_vertices()
        if len(vertices) < 2:
            continue
        start_vertex = vertices[int(np.random.randint(len(vertices)))]
        cycle = _angle_cycle(vertices)
        start = cycle.index(start_vertex)
        cycle = cycle[start:] + cycle[:start]
        shape_cells = _shape_cells_from_cycle(cycle, shape_type)
        coords = list(shape_cells) + list(vertices)
        r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)
        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for r, c in shifted_vertices:
            inp[r, c] = shape_color
        start_vertex_s = (start_vertex[0] - r0, start_vertex[1] - c0)
        out = _render_shape_local(inp, int(shape_color), shifted_vertices, start_vertex_s, shape_type)
        return {
            "patch_in": inp,
            "patch_out": out,
            "vertices": shifted_vertices,
            "start_vertex": start_vertex_s,
            "shape_type": shape_type,
            "focus": False,
            "decoy_vertex": None,
        }
    return None


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt(other_shape_type: str) -> tuple[np.ndarray, dict] | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    palette = list(np.random.permutation(SHAPE_COLORS))
    focus_shape_color = int(palette[0])
    other_shape_color = int(palette[1])
    active_color = int(palette[2])
    decoy_color = int(palette[3])

    focus_spec = _make_focus_shape_patch(focus_shape_color, active_color, decoy_color)
    other_spec = _make_plain_shape_patch(other_shape_color, other_shape_type)
    if focus_spec is None or other_spec is None:
        return None

    placements = []
    for spec, color in [(focus_spec, focus_shape_color), (other_spec, other_shape_color)]:
        placed = _try_place_shape(inp, out, reserved, spec, int(color))
        if placed is None:
            return None
        placements.append(placed)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    shape_ablation = _solve_variant(inp, use_shape_control=False, use_start_control=True)
    start_ablation = _solve_variant(inp, use_shape_control=True, use_start_control=False)
    if np.array_equal(shape_ablation, out):
        return None
    if np.array_equal(start_ablation, out):
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != 6:
        return None

    focus_meta = next(p for p in placements if p["focus"])
    if focus_meta["decoy_vertex"] is None or focus_meta["decoy_vertex"] >= focus_meta["start_vertex"]:
        return None

    return inp, {
        "shape_control_effect": True,
        "start_control_effect": True,
        "focus_marker": focus_meta["marker"],
        "focus_type_pos": focus_meta["type_pos"],
        "focus_start_vertex": focus_meta["start_vertex"],
        "focus_decoy_vertex": focus_meta["decoy_vertex"],
    }


def generate_input() -> np.ndarray:
    global _generate_call_index
    other_shape_type = PLAIN_SHAPE_TYPE_CYCLE[_generate_call_index % len(PLAIN_SHAPE_TYPE_CYCLE)]
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(other_shape_type)
        if result is not None:
            _generate_call_index += 1
            return result[0]
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
