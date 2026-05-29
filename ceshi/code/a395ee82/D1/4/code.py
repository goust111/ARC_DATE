# Variant of a395ee82
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: color_match — 单例像素中唯一与模板同色(T)的像素被识别为原点，其余O色像素为普通晶格点... → 连通性匹配

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# design_analysis:
#   a_specific_behavior:
#     input_objects:
#       - 一个高填充率的单色模板精灵，颜色为 T。
#       - 另一处是单色 O 的“节点+细线”图：节点仍然落在 gunit=2 的粗晶格上，相邻节点之间用 1 像素宽的正交线段连接。
#       - 原点节点不再靠颜色区分；所有节点与连线同色，原点只由图结构决定。
#     output_behavior:
#       - 仍然按节点相对原点的粗晶格偏移去平移模板。
#       - 原点对应那份模板改涂为 O，其余副本保持 T；若发生重叠，重叠区也涂 O。
#     concrete_trace:
#       - 原题中，solver 先找到唯一一个与模板同色的单点，把它当作 (0,0)。
#       - 本变体中，solver 改为在 O 色图内找 4 邻域度数最大的那个像素作为原点；随后在整张图上做 BFS，
#         把到原点距离为偶数的像素视为“真实节点”，奇数距离像素视为连接线中点。
#       - 差异在于：原逻辑直接用颜色读出原点；新逻辑必须先恢复图的连通结构，再由结构推出原点和节点集。
#   b_complexity_calibration:
#     - 若允许任意连通图，很多节点会是度 2，局部形态与连接线中点难区分，视觉上也不稳定。
#     - 为了让 D1 的变化清晰可见，生成器只产生“唯一分叉中心 + 若干短臂”的细线树：
#       原点度数固定为 3 或 4，其他节点度数最多 2。
#     - 这样解题者能肉眼看到“中心分叉最多的点”就是特殊点，同时仍保留 5-6 个节点、3-4 格跨度等原题尺度。
#   c_implicit_assumption_audit:
#     - 原 `template = max(comps, key=area)` 依赖模板是最大前景组件；现在图连成一整块后，图组件可能更大，所以失效。
#       新 solver 改为在单色组件里选“bbox 填充率最高”的组件作为模板，因为模板稠密、图稀疏。
#     - 原 `origin = singleton whose color == templ_col` 完全依赖唯一同色单点；现在不存在这种单点，必须删除。
#     - 原 `other_cols` / `Counter` 用来推 O 色；现在直接取另一个单色组件的颜色作为 O。
#     - 原 `deltas` 直接来自所有单点；现在先从图组件里恢复节点子集，再计算偏移。
#     - 原 `gcd` 仍然可复用，但前提从“所有单点坐标差”变成“所有节点坐标差”；生成器保证节点仍落在统一 2 像素晶格。
#   d_placement_legality:
#     - 模板与图组件的 bbox 仍要求至少隔 1 个背景像素，避免 solver 将它们并成一个前景区域。
#     - 所有模板平移副本必须完整落在 22x22 内；沿用 output-first 计算合法窗口。
#     - 图只用正交 1 像素连线，且所有连接都发生在节点中点，不会出现交叉或越界。
#   e_visual_observability:
#     - 输入里已经看不到“唯一 T 单点”，因此肉眼能明显感觉这不是原题。
#     - 同时图的唯一分叉中心在 2-4 个样例里足够稳定，能支撑人类归纳“最高连通度节点是原点”。
#   f_visual_cleanliness:
#     - 图只占一个局部区域，模板只占另一个局部区域；避免多层叠压导致难分辨。
#     - 连接线始终 1 像素粗，节点仍是离散晶格点，结构整洁。
#   g_color_inference:
#     - 仍只用 3 种颜色：bg、T、O。
#     - O 同时承担“图颜色”和“输出中特殊副本/重叠颜色”，规则仍靠结构提示，不靠颜色 id。
#   h_variant_metric:
#     - 关键指标 1：模板之外的 T 色单像素数量应恒为 0，证明不再存在原题的同色原点提示。
#     - 关键指标 2：图原点的像素度数应恒为 3 或 4，且为唯一最大值，证明 solver 确实在使用连通性而非退化回旧规则。
#   i_new_solver_assumptions:
#     - 假设前景单色大组件恰有 2 个：稠密模板 + 稀疏细线图。生成器显式保证。
#     - 假设图是 4 连通、线宽 1、节点间距统一、原点为唯一最高度数像素。生成器显式保证。
#     - 假设从原点做 BFS 时，偶数距离像素恰好是节点。生成器只画长度为 1 的中点连线，因此该假设成立。


def _fill_ratio(comp: dict) -> float:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return float(int(comp["area"])) / float(int(bbox_h((r0, c0, r1, c1))) * int(bbox_w((r0, c0, r1, c1))))


def _graph_degrees(coords: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r, c in coords:
        cnt = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (int(r + dr), int(c + dc)) in coords:
                cnt += 1
        deg[(int(r), int(c))] = int(cnt)
    return deg


def _extract_origin_and_nodes(graph_comp: dict) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    graph_coords = {(int(r), int(c)) for r, c in graph_comp["coords"]}
    assert graph_coords

    deg = _graph_degrees(graph_coords)
    max_deg = max(deg.values())
    origin_candidates = [pt for pt, d in deg.items() if int(d) == int(max_deg)]
    assert len(origin_candidates) == 1
    assert int(max_deg) >= 3
    origin = origin_candidates[0]

    dist = {origin: 0}
    q: deque[tuple[int, int]] = deque([origin])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (int(r + dr), int(c + dc))
            if nxt not in graph_coords or nxt in dist:
                continue
            dist[nxt] = int(dist[(r, c)] + 1)
            q.append(nxt)
    assert len(dist) == len(graph_coords)

    nodes = [pt for pt, d in dist.items() if int(d) % 2 == 0]
    assert origin in nodes
    assert len(nodes) >= 2
    return origin, nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a connected node graph.

    Input structure:
    - bg = background. One dense monochrome template sprite uses color T.
    - Elsewhere, a second monochrome component in color O forms a thin orthogonal graph:
      lattice nodes are joined by 1-pixel connector segments, all in the same color.
    - The graph's origin is not color-marked; it is the unique node with the highest degree.

    Rule:
    - Recover the graph origin from connectivity, then keep only graph pixels at even BFS distance
      from that origin; those are the true lattice nodes.
    - Convert each node offset from the origin into coarse lattice units.
    - Stamp the template at each offset, repainting the origin copy to O.
    - Any pixel covered by two or more copies is also recolored to O.

    Key insight: The special node is no longer signaled by color. The solver must use the
    connector pattern to find the branch center, then ignore the odd-distance connector pixels.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    big = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(big) == 2

    template = max(big, key=_fill_ratio)
    graph_comp = big[0] if big[1] is template else big[1]

    templ_coords = template["coords"]
    templ_col = int(template["color"])
    overlap_col = int(graph_comp["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    origin, nodes = _extract_origin_and_nodes(graph_comp)
    orr, orc = map(int, origin)

    deltas = [(int(r - orr), int(c - orc), int((r, c) == origin)) for r, c in nodes]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(is_origin)) for dr, dc, is_origin in deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator (connectivity variant) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
NODE_COUNT_RANGE = (5, 6)
GUNIT = 2

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
GRAPH_PLACE_RETRIES = 60
GENERATE_RETRIES = 400


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _gen_graph_nodes(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    need = int(n - 1)
    for up in range(3):
        for down in range(3):
            for left in range(3):
                for right in range(3):
                    lengths = [up, down, left, right]
                    arm_count = sum(int(x > 0) for x in lengths)
                    if arm_count not in (3, 4):
                        continue
                    if sum(lengths) != need:
                        continue
                    row_extent = int(up + down + 1)
                    col_extent = int(left + right + 1)
                    if row_extent not in (3, 4) or col_extent not in (3, 4):
                        continue

                    pts = {(0, 0)}
                    for step in range(1, up + 1):
                        pts.add((-step, 0))
                    for step in range(1, down + 1):
                        pts.add((step, 0))
                    for step in range(1, left + 1):
                        pts.add((0, -step))
                    for step in range(1, right + 1):
                        pts.add((0, step))
                    if len(pts) == int(n):
                        candidates.append(sorted(pts))
    if not candidates:
        return None
    nodes = random.choice(candidates)
    return nodes, (0, 0)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _graph_pixels(deltas: list[tuple[int, int]], orr: int, orc: int) -> set[tuple[int, int]]:
    coarse = {(int(dr), int(dc)) for dr, dc in deltas}
    pts: set[tuple[int, int]] = set()
    for dr, dc in coarse:
        pts.add((int(orr + dr * GUNIT), int(orc + dc * GUNIT)))
    for dr, dc in coarse:
        for adr, adc in ((1, 0), (0, 1)):
            nbr = (int(dr + adr), int(dc + adc))
            if nbr not in coarse:
                continue
            r1 = int(orr + dr * GUNIT)
            c1 = int(orc + dc * GUNIT)
            r2 = int(orr + nbr[0] * GUNIT)
            c2 = int(orc + nbr[1] * GUNIT)
            pts.add((int((r1 + r2) // 2), int((c1 + c2) // 2)))
    return pts


def _place_graph(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(GRAPH_PLACE_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * GUNIT)
        sr1 = int(orr + max_dr * GUNIT)
        sc0 = int(orc + min_dc * GUNIT)
        sc1 = int(orc + max_dc * GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        graph_pts = _graph_pixels(deltas, orr, orc)
        if any(pt in occupied for pt in graph_pts):
            continue
        if any(not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE) for rr, cc in graph_pts):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for rr, cc in graph_pts:
            grid[int(rr), int(cc)] = int(O)

        try:
            main(grid)
        except Exception:
            continue
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    node_count = random.randint(*NODE_COUNT_RANGE)
    graph_spec = _gen_graph_nodes(node_count)
    if graph_spec is None:
        return None
    lattice_nodes, origin = graph_spec
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_nodes]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)

    return _place_graph(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        r0,
        c0,
        h,
        w,
        bg,
        T,
        O,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
