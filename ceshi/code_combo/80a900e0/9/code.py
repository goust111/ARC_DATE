# Combo variant of 80a900e0 (index 9)
# combo_spec: D5/0+D6/1
# Name: 方形加L标记

# Design analysis for the fused variant
# -------------------------------------
# spec_variant:
#   grid:
#     keep the original 20-24 x 14-30 checkerboard canvas
#   background:
#     unchanged; bg[r,c] = (r+c+1)%2 with colors 0/1
#   source:
#     old: 8-pixel hollow diamond in a 5x5 bbox
#     new: 16-pixel hollow square ring in a 5x5 bbox, axis-aligned
#   outer_marker:
#     old: 2-3 pixel diagonal segment, one per chosen diamond edge
#     new: 3-pixel L marker occupying three cells of a 2x2 block
#     placement: one marker in each chosen outward corner quadrant of the square
#     quadrants: TL / TR / BL / BR
#   ray_rule:
#     old: infer outward direction from the segment slope plus which side faces the diamond
#     new: infer outward diagonal directly from the L bend orientation; the two degree-1 tips
#          emit rays, and the ray direction is the sum of the two unit arm vectors from bend to tips
#
# analysis:
#   a_specific_behavior_trace: |
#     原题在 diamond 外放一段对角短线。solver 先判断短线是 rc_diff 还是 rc_sum，再比较哪条法向
#     朝内能与 diamond 重合，最后从两端点沿反方向发射对角射线。
#     融合后，中心对象变成 5x5 空心方框，外物体也不再是共线短线，而是放在方框四个外角象限中的 L 标记。
#     例如方框右上角外的 TR 标记使用相对坐标 {(-2,5),(-1,5),(-1,6)}。其中 bend 是 (-1,5)，
#     从 bend 到两个 tips 的单位向量分别是 (-1,0) 和 (0,+1)，其和为 (-1,+1)，
#     所以两条射线都向右上对角发射。与原题相比，
#     “方框负责提供归属上下文，L 负责提供方向编码”，不再依赖线段斜率或朝内平移测试。
#   b_complexity_calibration: |
#     方向仍然只有四档 TL/TR/BL/BR，对应四个对角方向，没有引入额外自由度。
#     但视觉基元同时变了两处：中心从菱形变成方框，外标记从斜短线变成折角，肉眼能明显区分。
#     每个方框仍随机带 2-4 个标记，单个样例内部就有重复支撑，不需要依赖隐藏属性。
#   c_implicit_assumption_audit: |
#     - 原 _is_diamond8 假设源对象是 8 像素菱形；融合后必须改成 16 像素方框识别。
#     - 原 _outer_line_info 假设外部组件共线；L 标记下完全失效，必须用 4 邻域度数找 bend/tips。
#     - 原 main 只收集与源对象 8 邻接的非背景像素再分组件；对本变体不够，因为 L 的某些格会离方框角
#       有 2 格距离。新 solver 直接遍历整张图上的非背景组件，再按邻接得分归属到唯一方框。
#     - 原生成器的 even top-left / phase 假设来自菱形只落在一种棋盘相位。方框占据两种相位，
#       这个假设不再成立，所以放置时不再限制 top-left 奇偶性。
#     - 多项改动叠加后新增的主要风险是“外部 L 可能被其他对象的射线误配给错误方框”。
#       生成器通过方框保护区阻止其他对象的像素进入任一方框的 8 邻域，从而保证归属唯一。
#   d_placement_legality: |
#     生成器继续 output-first。某个 placement 合法当且仅当：
#     1. 5x5 方框完全在界内；2. 每个选中的 L 模板三格都在界内；3. marker、square、rays 互不重叠；
#     4. 每个 marker 至少产生 1 个射线像素，避免输入输出相同；5. 所有方框之间的 all_cells 不相交，
#        且任何别的对象都不能进入某个方框的保护区；6. 最终必须满足 main(inp) == out。
#   e_visual_observability: |
#     训练样例里，解题者能直接看到“空心方框 + 四类折角 L + 输出中的对角延长线”。
#     方框形状和原题菱形明显不同，L 的开口象限和射线方向一一对应，也能从 2-3 组例子里归纳。
#     颜色仅仅被复制到射线，不参与方向或排序推断，因此不存在不可见规则。
#   f_cleanliness: |
#     L 标记固定在方框四个外角象限，远离方框内部，也不会贴到别的方框保护区。
#     这样输入里每个局部结构都很清楚：一个 5x5 方框，周围若干小 L，输出再从对应 tips 拉出对角线。
#   g_color_inference: |
#     方框颜色彼此不同；marker 使用其他非背景色，但不同 marker 可以复用颜色。
#     射线始终继承 marker 颜色，这和原题一致，且人类能直接从局部对应关系看出来。
#   new_solver_assumptions_checked:
#     - marker 必须是 3 像素、4 邻域下唯一 bend + 两个 tips 的 L；生成器只放这种模板。
#     - 每个 marker 必须对某个方框有唯一最高邻接得分；保护区保证这一点。
#     - 射线方向必须是 tips 相对 bend 的两个单位向量之和；模板构造保证恰好得到四个非轴向对角方向。
#     - solver 只会在背景格上延长射线；output-first 校验保证不会因遮挡导致错误输出。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_SQUARE16_OFFSETS = {
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
}
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_square_ring16(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 16:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _SQUARE16_OFFSETS


def _l_marker_info(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(pts) != 3:
        return None

    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 1 or (c1 - c0) != 1:
        return None

    norm = {(r - r0, c - c0) for r, c in pts}
    if len(norm) != 3 or any(p not in {(0, 0), (0, 1), (1, 0), (1, 1)} for p in norm):
        return None

    pset = set(pts)
    deg = {
        p: sum((p[0] + dr, p[1] + dc) in pset for dr, dc in _DIR4)
        for p in pset
    }
    bends = [p for p, d in deg.items() if d == 2]
    tips = [p for p, d in deg.items() if d == 1]
    if len(bends) != 1 or len(tips) != 2:
        return None

    bend = bends[0]
    arm_vecs = [(tr - bend[0], tc - bend[1]) for tr, tc in tips]
    if sorted(abs(dr) + abs(dc) for dr, dc in arm_vecs) != [1, 1]:
        return None
    if arm_vecs[0][0] * arm_vecs[1][0] + arm_vecs[0][1] * arm_vecs[1][1] != 0:
        return None

    outward = (arm_vecs[0][0] + arm_vecs[1][0], arm_vecs[0][1] + arm_vecs[1][1])
    if outward[0] == 0 or outward[1] == 0:
        return None

    return {
        "bend": bend,
        "tips": sorted(tips),
        "outward": outward,
    }


def _adjacency_score(pts: list[tuple[int, int]], square_set: set[tuple[int, int]]) -> int:
    score = 0
    for r, c in pts:
        for rr, cc in neighbors8(r, c):
            if (rr, cc) in square_set:
                score += 1
    return score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from L-shaped corner markers around hollow square rings on a checkerboard.

    Input structure:
    - Checkerboard background whose two most common colors form the alternating pattern.
    - Hollow 5x5 square rings (16 pixels) in non-background colors.
    - Three-pixel L markers in other non-background colors, placed in outward corner quadrants near a square.

    Rule:
    - For each L marker, find the unique bend cell of 4-neighbor degree 2 and the two degree-1 tips.
    - The diagonal ray direction is the sum of the two unit vectors from the bend to those tips.
      Starting from each tip, paint background cells in that direction with the marker color until
      a non-background cell or the grid boundary is reached.

    Key insight:
    - The square tells you which local markers belong together, but the firing direction comes from
      the L orientation itself rather than from a line slope, color id, or hidden ordering.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)

    squares: list[tuple[int, list[tuple[int, int]]]] = []
    markers: list[tuple[int, list[tuple[int, int]], dict[str, object]]] = []
    for col, pts in comps:
        if _is_square_ring16(pts):
            squares.append((col, pts))
            continue
        info = _l_marker_info(pts)
        if info is not None:
            markers.append((col, pts, info))

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    sorted_squares = sorted(squares, key=bbox_key)
    square_sets = [set(pts) for _, pts in sorted_squares]
    assignments: list[list[tuple[int, dict[str, object]]]] = [[] for _ in sorted_squares]

    for mcol, mpts, minfo in markers:
        scores = [_adjacency_score(mpts, sset) for sset in square_sets]
        best = max(scores, default=0)
        if best <= 0 or scores.count(best) != 1:
            continue
        assignments[scores.index(best)].append((mcol, minfo))

    for square_markers in assignments:
        for mcol, minfo in square_markers:
            dr, dc = minfo["outward"]
            for pr, pc in minfo["tips"]:
                rr, cc = pr + dr, pc + dc
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = mcol
                    else:
                        break
                    rr += dr
                    cc += dc

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
SQUARE_COUNT_RANGE = (1, 3)
MARKERS_PER_SQUARE = (2, 4)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500

_SQUARE_COORDS = tuple(sorted(_SQUARE16_OFFSETS))
_MARKER_INFO = {
    "TL": {"marker": ((-2, -1), (-1, -2), (-1, -1))},
    "TR": {"marker": ((-2, 5), (-1, 5), (-1, 6))},
    "BL": {"marker": ((5, -2), (5, -1), (6, -1))},
    "BR": {"marker": ((5, 5), (5, 6), (6, 5))},
}
_MARKER_NAMES = tuple(_MARKER_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _protection_zone(square: list[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    zone = set(square)
    for r, c in square:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                zone.add((rr, cc))
    return frozenset(zone)


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, markers: tuple[str, ...]
) -> dict[str, object] | None:
    square = [(r0 + dr, c0 + dc) for dr, dc in _SQUARE_COORDS]
    all_cells = set(square)
    marker_parts = []

    for name in markers:
        marker = [(r0 + dr, c0 + dc) for dr, dc in _MARKER_INFO[name]["marker"]]
        if not all(0 <= r < h and 0 <= c < w for r, c in marker):
            return None

        info = _l_marker_info(marker)
        if info is None:
            return None

        tips = list(info["tips"])
        outward = info["outward"]
        rays = _ray_cells((h, w), tips[0], outward) + _ray_cells((h, w), tips[1], outward)
        if not rays:
            return None

        part_cells = marker + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        marker_parts.append({"marker": marker, "rays": rays, "weight": len(part_cells)})

    return {
        "square": square,
        "markers": marker_parts,
        "all_cells": frozenset(all_cells),
        "protected": _protection_zone(square, (h, w)),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4)]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for marker_spec in {specs[i] for i in order}:
        positions = toplefts[:]
        random.shuffle(positions)
        candidate_cache[marker_spec] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, marker_spec)) is not None
        ]

    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int, occupied: frozenset[tuple[int, int]], protected: frozenset[tuple[int, int]]) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            guard = placement["protected"]
            if occupied.isdisjoint(cells) and protected.isdisjoint(cells) and occupied.isdisjoint(guard):
                placed[spec_i] = placement
                if backtrack(i + 1, occupied | cells, protected | guard):
                    return True
                placed[spec_i] = None
        return False

    if not backtrack(0, frozenset(), frozenset()):
        return None
    return [p for p in placed if p is not None]


def _assign_marker_colors(placements: list[dict[str, object]], square_colors: list[int]) -> list[list[int]]:
    square_set = set(square_colors)
    palette = [c for c in COLOR_RANGE if c not in square_set]
    counts = Counter({c: 0 for c in palette})
    for c in square_colors:
        counts[c] += len(_SQUARE_COORDS)

    weighted_markers = sorted(
        (
            (mp["weight"], random.random(), pi, mi)
            for pi, placement in enumerate(placements)
            for mi, mp in enumerate(placement["markers"])
        ),
        reverse=True,
    )
    marker_colors: list[list[int]] = [[0] * len(placement["markers"]) for placement in placements]
    for _, _, pi, mi in weighted_markers:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        marker_colors[pi][mi] = chosen
        counts[chosen] += placements[pi]["markers"][mi]["weight"]
    return marker_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    square_count = random.randint(*SQUARE_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_MARKER_NAMES, random.randint(*MARKERS_PER_SQUARE))))
        for _ in range(square_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    square_colors = random.sample(list(COLOR_RANGE), len(placements))
    marker_colors = _assign_marker_colors(placements, square_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["square"], square_colors[pi])
        for mi, marker_part in enumerate(placement["markers"]):
            color = marker_colors[pi][mi]
            _paint_cells(out, marker_part["marker"], color)
            _paint_cells(out, marker_part["rays"], color)
            ray_coords.extend(marker_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.array(rs)
        cs = np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
