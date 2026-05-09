# Variant of 35ab12c3
# Dimension: D18 — 起始顶点选择策略 (meta)
# Change: 有阴影锚点的顶点优先作为起始点（seed_idxs），若有多个则取坐标最小的；无阴影时取坐标最小的顶点。起始顶点决定了极... → 随机起始顶点（由输入编码）
#
# Design analysis for this variant
# a) Concrete behavior and trace
#    - 每个真正的 shape 颜色现在都额外带一个“同色边缘控制像素”。这个像素位于网格边界，
#      但它不是普通顶点；solver 必须先把它从同色点集中剥离。
#    - 该控制像素沿着一条纯背景的水平/垂直走廊指向某个内部顶点；这个顶点就是起始顶点。
#    - 焦点 shape 还会在两个不同顶点旁各放一个异色邻居像素：其中一个是“真的阴影锚点”，
#      另一个是“旧规则会误选的诱饵锚点”。旧逻辑会选坐标更小的 seeded vertex，新逻辑则跟随边缘控制像素。
#    - 小例子：若同色三角形有两个 seeded vertex，左上顶点旁是颜色 6，右下顶点旁是颜色 7，
#      且边缘控制像素沿列指向右下顶点，则新规则复制颜色 7 的平行阴影；旧规则会因为左上 seeded
#      顶点坐标更小而复制颜色 6。两者在同一输入上输出不同，这保证了 D18 不是退化改动。
# b) Complexity calibration
#    - 控制变量只引入一种额外结构：同色边缘单像素 + 到目标顶点的空走廊。
#    - 为了让变化可见，生成器每个样本都强制至少一个“二选一 seeded vertex”的焦点 shape；
#      其余 shape 保持原题的纯顶点连线结构，避免整个任务被过多新机制淹没。
# c) Implicit assumptions audit of original main()
#    - 原 `seed_idxs` 代码假设“所有不同色邻居都只是用来决定 start，且 start 一定选最小 seeded vertex”。
#      在本变体里这个假设失效，因此不能再从邻居位置反推 start。
#    - 原 `pts_rc = np.argwhere(g == col)` 默认所有同色像素都是顶点；现在不成立，因为边缘控制像素同色但非顶点。
#    - 原 `carried` 仍然可复用，但前提变成“控制像素解析出的 start 顶点周围的异色邻居才是 active carried set”。
# d) Placement legality
#    - 所有 shape patch 都放在离边界至少 1 格的位置；控制像素再单独放到边缘。
#    - 为了让 solver 能从控制像素追到唯一顶点，生成器会保留一条纯背景走廊，并把这条走廊登记为 reserved，
#      禁止后续对象占用。
#    - 对象间仍保持 >=1 像素缓冲，输出阴影 region 也参与碰撞检查。
# e) Visual observability
#    - 人类能直接看到：某个颜色在边缘多出一个孤立同色像素，而输出中真正生效的阴影来自它指向的那个顶点，
#      不一定来自左上/最小 seeded vertex。2-3 个样例就足以支持归纳。
#    - 未使用颜色 id 做任何不可见排序；所有判定只依赖边缘控制像素、走廊、像素邻接和坐标可见关系。
# f) Visual cleanliness
#    - 控制像素固定是单像素，且与主体之间由空走廊连接，不与阴影线叠在一起。
#    - 焦点 shape 只放一个真实阴影和一个诱饵锚点，避免单个区域出现过多异色覆盖。
# g) Color inferability
#    - 颜色本身不做映射。异色邻居仍然表示“若该顶点被选中，则复制该颜色的平行线”。
#      新规则只改变“哪个顶点被选中”，因此颜色语义仍有足够结构线索。
# h) Variant metric used after implementation
#    - 关键指标：焦点 shape 上 `control_start != smallest_seeded_vertex`，并且把同一输入分别交给新旧 solver，
#      输出应不同。生成器会强制这一条件。
# i) New assumptions introduced by the modified main()
#    - 每个真正 shape 色组都有且仅有一个同色边缘控制像素。
#    - 控制像素到目标顶点的水平/垂直走廊中没有其他非背景像素。
#    - `generate_input()` 通过 reserved corridors 和非边界 placement 保证这些前提成立。

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_all = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        border_pts = [p for p in pts_all if _is_border_cell(p[0], p[1], h, w)]
        control = border_pts[0] if len(border_pts) == 1 and len(pts_all) - 1 >= 2 else None
        pts_rc = [p for p in pts_all if p != control] if control is not None else pts_all
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        start_rc = None
        if control is not None:
            start_rc = _start_from_control_marker(g, col, control, cycle, bg)

        if start_rc is not None and start_rc in cycle:
            start = cycle.index(start_rc)
        else:
            start = _fallback_start_idx(g, cycle, col, bg)

        cycle = cycle[start:] + cycle[:start]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

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


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

PATCH_RETRIES = 80
PLACEMENT_RETRIES = 120
GENERATE_RETRIES = 500


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _helper_offsets_for_vertex(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    vertex_idx: int,
    used_positions: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    anchor = vertices[vertex_idx]
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells or pos in used_positions:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    return candidates


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _render_shape_local(
    inp: np.ndarray,
    shape_color: int,
    vertices: list[tuple[int, int]],
    start_vertex: tuple[int, int],
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

    if len(cycle) == 2:
        edges = [(cycle[0], cycle[1])]
    else:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

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


def _make_plain_shape_patch(shape_color: int) -> dict | None:
    for _ in range(PATCH_RETRIES):
        vertices = _make_vertices()
        cells = _shape_cells(vertices)
        coords = list(cells) + vertices
        r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        inp = np.full((h, w), BG, dtype=int)
        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        for r, c in shifted_vertices:
            inp[r, c] = shape_color
        out = _render_shape_local(inp, int(shape_color), shifted_vertices, min(shifted_vertices))
        return {"patch_in": inp, "patch_out": out, "vertices": shifted_vertices, "focus": False}
    return None


def _make_focus_shape_patch(shape_color: int, active_color: int, decoy_color: int) -> dict | None:
    for _ in range(PATCH_RETRIES):
        vertices = _make_vertices()
        cells = _shape_cells(vertices)
        helpers_by_idx = {i: _helper_offsets_for_vertex(vertices, cells, i, set()) for i in range(len(vertices))}
        pair_candidates = [
            (active_idx, decoy_idx)
            for active_idx in range(len(vertices))
            for decoy_idx in range(len(vertices))
            if active_idx != decoy_idx
            and vertices[decoy_idx] < vertices[active_idx]
            and helpers_by_idx[active_idx]
            and helpers_by_idx[decoy_idx]
        ]
        if not pair_candidates:
            continue
        np.random.shuffle(pair_candidates)
        built = None
        for active_idx, decoy_idx in pair_candidates:
            active_offsets = helpers_by_idx[active_idx].copy()
            np.random.shuffle(active_offsets)
            for active_offset in active_offsets:
                active_pos = (vertices[active_idx][0] + active_offset[0], vertices[active_idx][1] + active_offset[1])
                active_shadow = _shift_cells(cells, active_offset)
                decoy_offsets = helpers_by_idx[decoy_idx].copy()
                np.random.shuffle(decoy_offsets)
                for decoy_offset in decoy_offsets:
                    decoy_pos = (vertices[decoy_idx][0] + decoy_offset[0], vertices[decoy_idx][1] + decoy_offset[1])
                    if decoy_pos == active_pos or decoy_pos in active_shadow:
                        continue
                    coords = list(cells) + vertices + list(active_shadow) + [active_pos, decoy_pos]
                    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
                    h, w = r1 - r0 + 1, c1 - c0 + 1
                    inp = np.full((h, w), BG, dtype=int)
                    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
                    for r, c in shifted_vertices:
                        inp[r, c] = shape_color
                    active_anchor = shifted_vertices[active_idx]
                    decoy_anchor = shifted_vertices[decoy_idx]
                    active_pos_s = (active_pos[0] - r0, active_pos[1] - c0)
                    decoy_pos_s = (decoy_pos[0] - r0, decoy_pos[1] - c0)
                    inp[active_pos_s] = active_color
                    inp[decoy_pos_s] = decoy_color
                    out = _render_shape_local(inp, int(shape_color), shifted_vertices, active_anchor)
                    built = {
                        "patch_in": inp,
                        "patch_out": out,
                        "vertices": shifted_vertices,
                        "focus": True,
                        "control_vertex": active_anchor,
                        "active_anchor": active_anchor,
                        "decoy_anchor": decoy_anchor,
                        "seeded_vertices": [active_anchor, decoy_anchor],
                    }
                    break
                if built is not None:
                    break
            if built is not None:
                break
        if built is not None:
            return built
    return None


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], int, int, int | None] | None:
    total = shape_count + 2 + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    active_color = int(palette[shape_count])
    decoy_color = int(palette[shape_count + 1])
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, active_color, decoy_color, distractor_color


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
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = inp.shape
    vr, vc = vertex
    options: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    def path_clear(path: list[tuple[int, int]]) -> bool:
        for rr, cc in path:
            if (rr, cc) in translated_nonbg:
                return False
            if inp[rr, cc] != BG or reserved[rr, cc]:
                return False
        return True

    if 0 < vc < w - 1:
        marker = (0, vc)
        path = [(rr, vc) for rr in range(1, vr)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            options.append((marker, path))
        marker = (h - 1, vc)
        path = [(rr, vc) for rr in range(vr + 1, h - 1)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            options.append((marker, path))

    if 0 < vr < h - 1:
        marker = (vr, 0)
        path = [(vr, cc) for cc in range(1, vc)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            options.append((marker, path))
        marker = (vr, w - 1)
        path = [(vr, cc) for cc in range(vc + 1, w - 1)]
        if inp[marker] == BG and not reserved[marker] and path_clear(path):
            options.append((marker, path))

    return options


def _choose_control_spec(
    inp: np.ndarray,
    reserved: np.ndarray,
    translated_nonbg: set[tuple[int, int]],
    vertices_global: list[tuple[int, int]],
    forced_vertex: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    candidates: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    if forced_vertex is not None:
        for marker, path in _control_options_for_vertex(inp, reserved, translated_nonbg, forced_vertex):
            candidates.append((forced_vertex, marker, path))
    else:
        order = list(range(len(vertices_global)))
        np.random.shuffle(order)
        for idx in order:
            vertex = vertices_global[idx]
            for marker, path in _control_options_for_vertex(inp, reserved, translated_nonbg, vertex):
                candidates.append((vertex, marker, path))
    if not candidates:
        return None
    pick = candidates[int(np.random.randint(len(candidates)))]
    return pick


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    vertices_local: list[tuple[int, int]],
    shape_color: int,
    forced_vertex_local: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    H, W = inp.shape
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
        vertices_global = [(r0 + vr, c0 + vc) for vr, vc in vertices_local]
        forced_global = None if forced_vertex_local is None else (r0 + forced_vertex_local[0], c0 + forced_vertex_local[1])
        control = _choose_control_spec(inp, reserved, translated_nonbg, vertices_global, forced_global)
        if control is None:
            continue
        chosen_vertex, marker, path = control
        paste(out, patch_out, r0, c0, transparent=BG)
        paste(inp, patch_in, r0, c0, transparent=BG)
        inp[marker] = int(shape_color)
        out[marker] = int(shape_color)
        for rr, cc in path:
            reserved[rr, cc] = True
        return chosen_vertex, marker, (r0, c0)
    return None


def _try_place_distractor(inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    H, W = inp.shape
    h, w = patch.shape
    if H - h <= 1 or W - w <= 1:
        return False
    patch_coords = _patch_nonbg_coords(patch)
    for _ in range(PLACEMENT_RETRIES):
        r0 = int(np.random.randint(1, H - h))
        c0 = int(np.random.randint(1, W - w))
        if not _can_place_patch(out, reserved, patch_coords, r0, c0):
            continue
        translated = [(r0 + pr, c0 + pc) for pr, pc in patch_coords]
        if any(reserved[rr, cc] for rr, cc in translated):
            continue
        paste(out, patch, r0, c0, transparent=BG)
        paste(inp, patch, r0, c0, transparent=BG)
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, active_color, decoy_color, distractor_color = palette
    focus_idx = int(np.random.randint(shape_count))

    shape_specs: list[dict] = []
    for idx, shape_color in enumerate(shape_colors):
        if idx == focus_idx:
            patch = _make_focus_shape_patch(int(shape_color), int(active_color), int(decoy_color))
        else:
            patch = _make_plain_shape_patch(int(shape_color))
        if patch is None:
            return None
        patch["shape_color"] = int(shape_color)
        shape_specs.append(patch)

    order = [focus_idx] + [i for i in range(shape_count) if i != focus_idx]
    placements: list[dict] = []
    for idx in order:
        spec = shape_specs[idx]
        placed = _try_place_shape(
            inp,
            out,
            reserved,
            spec["patch_in"],
            spec["patch_out"],
            spec["vertices"],
            int(spec["shape_color"]),
            spec.get("control_vertex"),
        )
        if placed is None:
            return None
        chosen_vertex, marker, loc = placed
        placements.append(
            {
                "idx": idx,
                "shape_color": int(spec["shape_color"]),
                "chosen_vertex": chosen_vertex,
                "marker": marker,
                "loc": loc,
                "focus": bool(spec["focus"]),
                "seeded_vertices": [
                    (loc[0] + r, loc[1] + c) for r, c in spec.get("seeded_vertices", [])
                ],
            }
        )

    if has_distractor and distractor_color is not None:
        distractor_in, _ = _make_distractor_patch(int(distractor_color))
        if not _try_place_distractor(inp, out, reserved, distractor_in):
            return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    total_colors = shape_count + 2 + int(has_distractor)
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None

    focus_meta = next(p for p in placements if p["focus"])
    seeded_sorted = sorted(focus_meta["seeded_vertices"])
    if len(seeded_sorted) < 2:
        return None
    if focus_meta["chosen_vertex"] == seeded_sorted[0]:
        return None

    return inp, {
        "shape_count": shape_count,
        "has_distractor": has_distractor,
        "focus_marker": focus_meta["marker"],
        "focus_chosen_vertex": focus_meta["chosen_vertex"],
        "focus_smallest_seeded": seeded_sorted[0],
    }


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
