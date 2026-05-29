# Combo variant of 35ab12c3 (index 14)
# combo_spec: D14/1+D18/2
# Name: 锚点在任意顶点附近+每个顶点都是阴影锚定点
#
# Design analysis for the combo
# -----------------------------
# a) Concrete behavior:
#    原题先对某个 shape 色恢复一个顶点环，然后只在单一起始顶点附近找异色邻居；这些邻居的偏移会被
#    复制到整条轮廓上。D14/1 放宽为“任意顶点都能提供阴影锚点”，D18/2 再把“全局统一携带”改成
#    “每条边只继承自己起点顶点的邻居”。融合后，solver 的执行顺序应当是：
#    1. 恢复稳定的顶点环；
#    2. 对每个顶点单独扫描 8 邻域里的异色像素；
#    3. 第 i 条边 (v_i -> v_{i+1}) 只使用 v_i 附近的锚点；
#    4. 不同顶点可以携带不同颜色、不同偏移，所以同一 shape 的不同边会长出不同方向的局部平行阴影。
#    小例子：三角形三个顶点 A/B/C，A 旁边有红色锚点 offset=(-1,0)，B 旁边有青色锚点 offset=(0,1)。
#    原题会整圈只复制一种阴影；D14 单变体会把红、青两种阴影都整圈复制；本融合版则只让从 A 出发的边带红影，
#    从 B 出发的边带青影，形成更局部、更多向的辐射图案。
# b) Complexity calibration:
#    如果只给一个 shape 一个锚点，输入会退化回原题；如果多个顶点共用同色同偏移，也很接近 D18 单变体。
#    因此生成器强制至少一个 shape 同时满足：
#    - 至少 2 个不同顶点带锚点；
#    - 至少 2 个不同 shadow 颜色；
#    - 至少 2 个不同 offset 方向。
#    同时总锚点数上限保持在 3，避免颜色预算和画面复杂度失控。
# c) Hidden assumptions audit:
#    - 原代码 `seed_idxs/start` 假设阴影锚定依赖单一起点；融合后这个前提失效，起点只保留为稳定旋转用途。
#    - 原代码 `carried` 只从 `cycle[0]` 读邻居；融合后必须改成 `carried_by_vertex[idx]`。
#    - 原代码 phase 2 对所有边复用同一个偏移集合；融合后这会把某顶点的锚点错误扩散到别的边。
#    - 原代码默认锚点像素不会同时贴着多个顶点；融合后 solver 仍依赖这点，因此生成器只接受
#      “8 邻域中恰好贴着一个顶点”的锚点。
#    - 原代码 `_clear_in_original(... allow=set())` 假设阴影路径内部没有别的输入像素；融合后如果一个锚点落到
#      另一条阴影边的内部，这条边会被错误阻塞。因此生成器显式避免锚点和阴影线互相穿插。
# d) Placement legality:
#    patch 级别先选好顶点、局部阴影锚点和每条局部阴影段，再求 bbox。候选锚点必须：
#    - 在锚定顶点的 8 邻域里；
#    - 不落在主 shape 顶点或主边上；
#    - 只邻接本 shape 的一个顶点；
#    - 不与该顶点对应的主边共线；
#    - 其平移后的阴影段不与主 shape、其他锚点、其他阴影段重叠。
#    全局放置仍用 `random_free_location` 基于输出 patch 做隔离，因此不同对象不会在最终输出里相撞。
# e) Visual observability:
#    人类应该能从 2-4 个样例直接看到：同一 shape 上有多个不同颜色的小锚点，且这些颜色并不是整圈统一平移，
#    而是分别只沿部分边扩散。这里没有使用颜色 id 做排序或数值判定；颜色只承担可见分组标签。
# f) Visual cleanliness:
#    为避免画面过乱，生成器控制为“每顶点最多一个锚点、每 shape 最多 3 个锚点、对象间至少 1 格缓冲”。
#    还禁止局部阴影段之间重叠，避免后画的阴影被 `bg_only` 规则吞掉。
# g) Color inference:
#    每个 shadow 颜色在输入中只出现 1 个锚点像素，在输出中只扩展成它对应的局部阴影段。
#    颜色语义稳定：某个颜色就是某一条局部阴影的颜色，不存在隐式颜色映射表。
#
# Post-change solver assumptions and generator guarantees
# ------------------------------------------------------
# 1. Solver assumes a valid shadow anchor pixel is 8-adjacent to exactly one vertex of its host shape.
#    Generator enforces this when sampling anchor candidates.
# 2. Solver assumes shadow segments for different anchors do not require overlap tie-breaking.
#    Generator enforces pairwise-disjoint shifted shadow cells.
# 3. Solver assumes every shadow color stays a non-shape in the input.
#    Generator uses each shadow color exactly once in the input, as a single anchor pixel.
# 4. Solver assumes the intended shadowed shape is the only >=2-pixel group of its color.
#    Generator keeps all shape colors unique and isolated with padding.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    return dr != 0 and abs(dr) == abs(dc)


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


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


def _rotate_cycle_to_min(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points_rc:
        return []
    start = min(range(len(points_rc)), key=lambda i: (points_rc[i][0], points_rc[i][1]))
    return points_rc[start:] + points_rc[:start]


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _all_collinear(points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return True
    r0, c0 = points[0]
    r1, c1 = points[1]
    dr = r1 - r0
    dc = c1 - c0
    for r, c in points[2:]:
        if (r - r0) * dc != (c - c0) * dr:
            return False
    return True


def _collect_unique_vertex_shadows(
    g: np.ndarray, cycle: list[tuple[int, int]], self_color: int, bg: int
) -> list[list[tuple[int, int, int]]]:
    h, w = g.shape
    hits: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for idx, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != self_color:
                    hits.setdefault((rr, cc), []).append((idx, drn, dcn, v))

    carried_by_vertex: list[list[tuple[int, int, int]]] = [[] for _ in cycle]
    for specs in hits.values():
        if len(specs) != 1:
            continue
        idx, drn, dcn, v = specs[0]
        carried_by_vertex[idx].append((drn, dcn, v))
    return carried_by_vertex


def _legacy_main_single_anchor(grid: np.ndarray) -> np.ndarray:
    """Original single-start-anchor solver, kept only for degeneration analysis."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

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
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
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

        edges = _shape_edges(cycle)
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


# Overview:
# Restore each shape from its same-color vertices, then let every edge inherit shadow anchors
# only from its own starting vertex, so one shape can emit several local shadows in different
# colors and directions.
# Input structure:
# Background 0; each shape color appears as 2+ isolated vertices; some vertices have one
# 8-neighbor pixel of another color acting as a local shadow anchor; distractor colors remain
# as non-alignable singleton/small groups.
# Rule:
# Draw all aligned shape edges first. Then for edge (v_i -> v_{i+1}), look only at unique
# non-shape neighbors around v_i; each such neighbor defines a parallel offset and color for
# a local shadow copy of that edge alone.
# Key insight:
# The fusion is not “collect all anchors globally” and not “choose one start vertex”; it is
# “scan all vertices, but apply anchors locally to the outgoing edge of the vertex that owns them”.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _rotate_cycle_to_min(_angle_cycle(pts_rc))
        if len(cycle) < 2:
            continue

        carried_by_vertex = _collect_unique_vertex_shadows(g, cycle, col, bg)
        edges = _shape_edges(cycle)

        drawable_edges: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for idx, ((r0, c0), (r1, c1)) in enumerate(edges):
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append((idx, (r0, c0), (r1, c1)))

        for _, (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for idx, (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried_by_vertex[idx]:
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


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_ANCHORS_CAP = 3
MIN_TOTAL_ANCHORS = 2
DISTRACTOR_PROB = 0.4
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

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
SHADOWABLE_WEIGHTS = np.array([0.24, 0.24, 0.18, 0.18, 0.16], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50
SHADOW_ASSIGN_RETRIES = 120
SHAPE_PLACEMENT_RETRIES = 80
GENERATE_RETRIES = 700


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
_SHADOWABLE_BUILDERS = [_build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices(require_shadowable: bool = False) -> list[tuple[int, int]]:
    if require_shadowable:
        weights = SHADOWABLE_WEIGHTS / SHADOWABLE_WEIGHTS.sum()
        idx = int(np.random.choice(len(_SHADOWABLE_BUILDERS), p=weights))
        return _SHADOWABLE_BUILDERS[idx]()
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _normalized_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _rotate_cycle_to_min(_angle_cycle(vertices))


def _anchor_candidates(
    cycle: list[tuple[int, int]],
    start_idx: int,
    anchor_positions: set[tuple[int, int]],
    used_shadow_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    shape_cells = _shape_cells(cycle)
    vertex_set = set(cycle)
    edges = _shape_edges(cycle)
    start, end = edges[start_idx]
    drs, dcs = _step(start, end)
    candidates: list[dict[str, object]] = []

    for dr, dc in DIR8:
        if drs * dc - dcs * dr == 0:
            continue
        anchor_pos = (start[0] + dr, start[1] + dc)
        if anchor_pos in shape_cells or anchor_pos in vertex_set or anchor_pos in anchor_positions or anchor_pos in used_shadow_cells:
            continue
        if sum(1 for vr, vc in cycle if max(abs(anchor_pos[0] - vr), abs(anchor_pos[1] - vc)) == 1) != 1:
            continue

        shifted_cells = {(r + dr, c + dc) for r, c in _gen_segment_cells(start, end)}
        if shifted_cells & shape_cells:
            continue
        if shifted_cells & vertex_set:
            continue
        if shifted_cells & anchor_positions:
            continue
        if shifted_cells & used_shadow_cells:
            continue

        candidates.append(
            {
                "vertex_idx": start_idx,
                "offset": (dr, dc),
                "anchor_pos": anchor_pos,
                "shifted_cells": shifted_cells,
            }
        )
    return candidates


def _choose_anchor_specs(
    cycle: list[tuple[int, int]], anchor_colors: list[int]
) -> list[tuple[int, tuple[int, int], int]] | None:
    if not anchor_colors:
        return []
    if len(cycle) < len(anchor_colors):
        return None
    if len(anchor_colors) >= 2 and (len(cycle) < 3 or _all_collinear(cycle)):
        return None

    for _ in range(SHADOW_ASSIGN_RETRIES):
        used_vertices: set[int] = set()
        used_offsets: set[tuple[int, int]] = set()
        used_shadow_cells: set[tuple[int, int]] = set()
        anchor_positions: set[tuple[int, int]] = set()
        specs: list[tuple[int, tuple[int, int], int]] = []

        for shadow_color in anchor_colors:
            vertex_order = np.random.permutation(len(cycle))
            picked = None
            for raw_idx in vertex_order:
                idx = int(raw_idx)
                if idx in used_vertices:
                    continue
                candidates = _anchor_candidates(cycle, idx, anchor_positions, used_shadow_cells)
                if not candidates:
                    continue
                fresh = [cand for cand in candidates if tuple(cand["offset"]) not in used_offsets]
                pool = fresh or candidates
                picked = pool[int(np.random.randint(len(pool)))]
                break
            if picked is None:
                break

            idx = int(picked["vertex_idx"])
            offset = tuple(picked["offset"])
            anchor_pos = tuple(picked["anchor_pos"])
            shifted_cells = set(picked["shifted_cells"])
            specs.append((idx, offset, int(shadow_color)))
            used_vertices.add(idx)
            used_offsets.add(offset)
            anchor_positions.add(anchor_pos)
            used_shadow_cells.update(shifted_cells)

        if len(specs) != len(anchor_colors):
            continue
        if len(anchor_colors) >= 2 and len({offset for _, offset, _ in specs}) < 2:
            continue
        return specs

    return None


def _patch_bbox(
    cycle: list[tuple[int, int]],
    shadow_specs: list[tuple[int, tuple[int, int], int]],
) -> tuple[int, int, int, int]:
    coords = list(cycle) + list(_shape_cells(cycle))
    edges = _shape_edges(cycle)
    for idx, (dr, dc), _shadow_color in shadow_specs:
        start, end = edges[idx]
        coords.append((start[0] + dr, start[1] + dc))
        coords.extend((r + dr, c + dc) for r, c in _gen_segment_cells(start, end))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, anchor_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    cycle = _normalized_cycle(_make_vertices(require_shadowable=bool(anchor_colors)))
    shadow_specs = _choose_anchor_specs(cycle, anchor_colors)
    if shadow_specs is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cycle, shadow_specs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in cycle:
        inp[r - r0, c - c0] = int(shape_color)

    for idx, (dr, dc), shadow_color in shadow_specs:
        ar, ac = cycle[idx]
        rr = ar + dr - r0
        cc = ac + dc - c0
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(shadow_color)
        if sum(1 for vr, vc in cycle if max(abs((ar + dr) - vr), abs((ac + dc) - vc)) == 1) != 1:
            return None

    out = main(inp)
    for shadow_color in anchor_colors:
        if np.count_nonzero(inp == shadow_color) != 1:
            return None
        if np.count_nonzero(out == shadow_color) <= 1:
            return None

    if len(anchor_colors) >= 2:
        stats = _analyze_fusion_usage(inp)
        if not any(
            stat["anchor_vertices"] >= 2 and stat["shadow_colors"] >= 2 and stat["distinct_offsets"] >= 2
            for stat in stats
        ):
            return None
        if np.array_equal(out, _legacy_main_single_anchor(inp)):
            return None

    return inp, out


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
        patch = np.full((1, 1), int(color), dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = int(color)
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), int(color), dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_anchors = min(MAX_ANCHORS_CAP, len(COLORS) - shape_count - int(has_distractor))
    if max_anchors < MIN_TOTAL_ANCHORS:
        return None
    if max_anchors == MIN_TOTAL_ANCHORS:
        anchor_count = MIN_TOTAL_ANCHORS
    else:
        # Keep 2-anchor and 3-anchor cases both visible in small evaluation batches.
        anchor_count = MIN_TOTAL_ANCHORS if np.random.rand() < 0.5 else max_anchors
    total = shape_count + anchor_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    anchor_colors = [int(v) for v in palette[shape_count:shape_count + anchor_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, anchor_colors, distractor_color


def _assign_anchors(shape_count: int, anchor_colors: list[int]) -> dict[int, list[int]]:
    if len(anchor_colors) < 2:
        return {}
    primary = int(np.random.randint(shape_count))
    mapping: dict[int, list[int]] = {}

    if len(anchor_colors) == 2:
        mapping[primary] = [int(v) for v in anchor_colors]
        return mapping

    if shape_count == 1 or np.random.rand() < 0.5:
        mapping[primary] = [int(v) for v in anchor_colors]
        return mapping

    mapping[primary] = [int(anchor_colors[0]), int(anchor_colors[1])]
    others = [idx for idx in range(shape_count) if idx != primary]
    if others:
        secondary = int(np.random.choice(others))
        mapping[secondary] = [int(anchor_colors[2])]
    else:
        mapping[primary].append(int(anchor_colors[2]))
    return mapping


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, anchor_colors, distractor_color = palette
    anchor_map = _assign_anchors(shape_count, anchor_colors)
    total_colors = shape_count + len(anchor_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        wanted_anchors = anchor_map.get(idx, [])
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), wanted_anchors)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    stats = _analyze_fusion_usage(inp)
    if not any(
        stat["anchor_vertices"] >= 2 and stat["shadow_colors"] >= 2 and stat["distinct_offsets"] >= 2
        for stat in stats
    ):
        return None
    if np.array_equal(main(inp), _legacy_main_single_anchor(inp)):
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")


def _analyze_fusion_usage(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    stats: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts) < 2:
            continue
        cycle = _rotate_cycle_to_min(_angle_cycle(pts))
        edges = _shape_edges(cycle)
        carried_by_vertex = _collect_unique_vertex_shadows(g, cycle, col, bg)

        drawable = []
        for idx, ((r0, c0), (r1, c1)) in enumerate(edges):
            if _aligned_8(r0, c0, r1, c1) and _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                drawable.append(idx)

        anchor_vertices = [idx for idx in drawable if carried_by_vertex[idx]]
        offsets = {(dr, dc) for idx in anchor_vertices for dr, dc, _ in carried_by_vertex[idx]}
        shadow_colors = {ncol for idx in anchor_vertices for _dr, _dc, ncol in carried_by_vertex[idx]}
        stats.append(
            {
                "color": col,
                "vertex_count": len(cycle),
                "drawable_edges": len(drawable),
                "anchor_vertices": len(anchor_vertices),
                "distinct_offsets": len(offsets),
                "shadow_colors": len(shadow_colors),
            }
        )
    return stats
