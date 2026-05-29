# Variant of 80a900e0
# Dimension: D6 — 外边框线段的位置与形状 (obstacle)
# Change: 3像素对角线段平行于菱形边缘且向外偏移1步，每条线段对应菱形的一条边... → 外边框为L形标记

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


# Design analysis
# ---------------
# spec_variant:
#   unchanged_core:
#     grid: 14-30 cols x 20-24 rows
#     background: checkerboard of colors 0 and 1; bg[r,c]=(r+c+1)%2
#     diamond:
#       count: 1-3
#       shape: same 8-pixel hollow diamond outline in a 5x5 bbox
#       color: each diamond still uses a distinct non-bg color
#     output_rule_core:
#       rays: each obstacle still emits exactly two same-colored diagonal rays until blocked or boundary
#   D6_change:
#     old_obstacle:
#       shape: 2-3 pixel diagonal collinear segment
#       direction_source: line slope + relation to diamond
#     new_obstacle:
#       shape: canonical 3-pixel L marker occupying 3 cells of a 2x2 block
#       direction_source: the bend orientation alone
#       endpoint_rule: the two degree-1 arm tips emit the rays
#       direction_rule: outward diagonal = sum of the two unit arm vectors from bend to the tips
#   concrete_templates_relative_to_diamond_bbox:
#     TL: [(-1, 1), (0, 0), (0, 1)]   # open quadrant NW => rays (-1,-1)
#     TR: [(-1, 3), (0, 3), (0, 4)]   # open quadrant NE => rays (-1,+1)
#     BL: [(4, 0), (4, 1), (5, 1)]    # open quadrant SW => rays (+1,-1)
#     BR: [(4, 3), (4, 4), (5, 3)]    # open quadrant SE => rays (+1,+1)
# analysis:
#   a_specific_behavior_trace: |
#     原题在 diamond 的某条边外放一段对角线，solver 先判断该段是 rc_diff 还是 rc_sum，再用“朝 diamond 平移是否重合”
#     来决定 inward normal，最后反向得到 outward 并从两个端点发射射线。
#     现在这条链路不成立，因为外物体不再共线。新规则下，solver 对每个小外物体先做 L 识别：
#     3 像素必须正好形成一个 2x2 缺角块；唯一的 4 邻域度数 2 的格子是 bend，另外两格是 tips。
#     例如 TR 标记 {(-1,3),(0,3),(0,4)} 的 bend 在 (0,3)，两条臂向量分别是 (-1,0) 和 (0,+1)，
#     和为 (-1,+1)，因此两条射线都朝 NE 发射，完全不需要读取线段斜率。
#   b_complexity_calibration: |
#     方向只有四档（NW/NE/SW/SE），和原任务一致，没有引入额外自由度；但视觉上从“斜线”变成“折角”，
#     人眼可明显看出这是新的方向编码方式。每个样例仍有 2-4 个边标记，重复足够支撑归纳。
#   c_implicit_assumption_audit:
#     - 原 _outer_line_info 假设组件共线；变体下失效，必须移除。
#     - 原 main 依赖“法向朝 diamond 平移会增加重合”；变体下 L 标记没有统一法向，失效。
#     - 原 main 从按邻接收集到的像素重新分组件；对于 L 标记，某些格可能与 diamond 距离 2，不能只看邻接种子。
#       新实现直接在全图非背景组件中识别 diamond 与 marker，再按“与哪个 diamond 8 邻接最多”进行归属。
#     - 原 generator 只需检查 segment+rays；现在必须检查 marker+rays，并保证两个 ray 起点来自 marker 的两个 arm tips。
#   d_placement_legality: |
#     generator 继续 output-first。某个 diamond placement 只有在以下条件全满足时才合法：
#     1. diamond 本体在界内；2. 每个选中 edge 的 3 像素 L 标记全部在界内；3. 标记与 diamond/其他标记/射线都不重叠；
#     4. 两条 rays 在发射前就由 marker 的两个 tips 唯一确定；5. 所有对象放完后，新 solver 复算输出完全一致。
#   e_visual_observability: |
#     解题者能直接看到：输入中的短障碍不再是斜短线，而是四种朝向的 L；输出仍然是从两个臂端沿某个对角方向延长。
#     因为四种 L 的“开口象限”与四个 diagonal direction 一一对应，2-3 个训练样例足以归纳。
#     虽然题面允许 2-3 像素 L，但 generator 故意固定为完整 3 像素形式，避免边界裁剪导致的方向歧义。
#   f_cleanliness: |
#     L 标记仍然各自贴在 diamond 四边附近，且 generator 用全体占位冲突检测避免标记、diamond、rays 互相穿插。
#   g_color_inference: |
#     颜色逻辑不变：ray 使用对应标记颜色。颜色不参与排序或方向推断，避免不可见假设。
#   new_solver_assumptions_checked:
#     - marker 必须是 3 像素、4 邻域下形如 L 的单组件；generator 显式只生成这种形状。
#     - 每个 marker 只能归属一个 diamond；generator 的 placement search 通过 all_cells disjoint 保证局部不拥挤。
#     - ray 前方初始单元必须是背景或边界外；若被占，output-first 验证会失败并重试。


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _l_marker_info(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    """Return bend/tips/outward for a 3-cell L marker, or None if shape is not a valid marker."""
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


def _adjacency_score(pts: list[tuple[int, int]], dset: set[tuple[int, int]]) -> int:
    score = 0
    for r, c in pts:
        for rr, cc in neighbors8(r, c):
            if (rr, cc) in dset:
                score += 1
    return score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays outward from L-shaped outer markers adjacent to diamond shapes on a checkerboard.

    Input structure:
    - Checkerboard bg (two most common colors).
    - Hollow diamond outlines (8-pixel) in a non-bg color.
    - Short L-shaped outer markers in other non-bg colors, one marker per occupied diamond edge.

    Rule:
    - For each L marker, infer the outward diagonal from the bend orientation:
      the two degree-1 arm tips emit the rays, and the diagonal direction is the sum of
      the two unit vectors from the bend to those tips.
    - From each tip, cast a diagonal ray outward, painting over bg pixels with the marker's
      color until hitting a non-bg pixel or the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)

    diamonds: list[tuple[int, list[tuple[int, int]]]] = []
    markers: list[tuple[int, list[tuple[int, int]], dict[str, object]]] = []
    for col, pts in comps:
        if _is_diamond8(pts):
            diamonds.append((col, pts))
            continue
        info = _l_marker_info(pts)
        if info is not None:
            markers.append((col, pts, info))

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    sorted_diamonds = sorted(diamonds, key=bbox_key)
    assignments: list[list[tuple[int, list[tuple[int, int]], dict[str, object]]]] = [[] for _ in sorted_diamonds]

    for ocol, pts, info in markers:
        scores = [_adjacency_score(pts, set(dpts)) for _, dpts in sorted_diamonds]
        best = max(scores, default=0)
        if best <= 0:
            continue
        if scores.count(best) != 1:
            continue
        assignments[scores.index(best)].append((ocol, pts, info))

    for di, (_, dpts) in enumerate(sorted_diamonds):
        _ = dpts
        for ocol, _, info in assignments[di]:
            outward = info["outward"]
            for pr, pc in info["tips"]:
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = ocol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


# --------------- generator ---------------

# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300

# ── Geometry constants ──────────────────────────────────────
_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TR": {"marker": ((-1, 3), (0, 3), (0, 4))},
    "TL": {"marker": ((-1, 1), (0, 0), (0, 1))},
    "BL": {"marker": ((4, 0), (4, 1), (5, 1))},
    "BR": {"marker": ((4, 3), (4, 4), (5, 3))},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


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


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []

    for edge in edges:
        marker = [(r0 + dr, c0 + dc) for dr, dc in _EDGE_INFO[edge]["marker"]]
        if not all(0 <= r < h and 0 <= c < w for r, c in marker):
            return None

        info = _l_marker_info(marker)
        if info is None:
            return None

        tips = list(info["tips"])
        outward = info["outward"]
        rays = _ray_cells((h, w), tips[0], outward) + _ray_cells((h, w), tips[1], outward)
        part_cells = marker + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        edge_parts.append({"marker": marker, "rays": rays, "weight": len(part_cells)})

    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells):
                occupied.update(cells)
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_marker_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    marker_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        marker_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return marker_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    marker_colors = _assign_marker_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = marker_colors[pi][ei]
            _paint_cells(out, edge_part["marker"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs, cs = np.array(rs), np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
