# Combo variant of 80a900e0 (index 29)
# combo_spec: D12/1+D16/2
# Name: 极端比例扩展网格
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 输入和输出同尺寸，射线只在当前网格里走到先撞到的边界为止
#     fused: 输入改成极端宽/高矩形；输出先只向“真正截断射线的短轴边”扩展，再让射线继续画到新边界
#   concrete_geometry:
#     visible_input:
#       - 仍是两色棋盘格背景
#       - 仍有 1-2 个 8 像素菱形轮廓，以及贴在菱形边外侧的短对角线段
#       - 输入里仍然看不到射线本体
#     output:
#       - 保留原输入内容，并在短轴方向补出新的棋盘格带
#       - 在超宽网格里，新增区域通常只出现在上/下；在超高网格里，新增区域通常只出现在左/右
#       - 射线颜色仍等于对应短线段颜色
#   small_trace:
#     - 例：10x30 的超宽输入中，一个朝东北的外段端点距上边 2 格、距右边 14 格
#     - 原题会只画 2 格，因为先撞到上边
#     - 本变体把上边再补 12 格，使“向上可走步数”追平“向右可走步数”，于是该射线会继续走满 14 格
#     - 右边本来就不是瓶颈，所以不会同步向右扩；差异正体现在“只补受限边”
#   complexity_calibration:
#     - 仍保留多菱形、每个 2-4 条外段、多个颜色，避免退化成单对象单方向
#     - 极端宽高比只限制整体画布，不减少局部结构复杂度
#     - generator 强制样例出现明显的短轴扩展，同时禁止长轴扩展，保证融合特征稳定可见
#   hidden_assumption_audit:
#     - 原 main 里的 `out = g.copy()` 假设输出尺寸不变；这里必须先求 pad 再建新画布
#     - 原 while 循环默认“越界就是结束”；这里仍成立，但边界先由每个端点的受限轴决定
#     - 原 D12 单变体按可见 bbox 的对侧空白镜像扩展；本融合里这条假设不再成立，因为是否扩边要看每条射线哪个轴先耗尽
#     - 新画布仍需保持棋盘格相位连续，所以 paste 前要按 `(top_pad, left_pad)` 平移背景相位
#   placement_legality:
#     - 生成器先搜索只含可见骨架的合法放置，并验证“若在原输入尺寸内补回射线”也不会撞到其他对象
#     - 同时检查每个端点的短轴确实是限制轴，避免 solver 需要处理长轴扩展样本
#   visual_observability:
#     - 人类能直接看到：输入是超宽或超高矩形，而输出只朝短轴方向长出棋盘格边带
#     - 外段的朝向解释了为什么射线继续走；极端长宽比解释了为什么只扩某一边而不是四边都扩
#   cleanliness:
#     - 新增区域只有连续棋盘格和几条 45 度彩色细射线，没有额外标记层
#     - 生成器拒绝扩展太小或 pad 对称性不明显的样例，避免视觉上像普通 D12 扩边
#   color_inference:
#     - 不引入任何新颜色规则；线段和其射线始终同色
#     - 颜色只负责绑定“哪条外段发出哪两条射线”，推断负担仍在几何关系而不是颜色表
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设输入里仍只有棋盘格、菱形轮廓和邻接外段；generator 保持这一点
#     - solver 假设短轴受限信息可以从端点到四边的距离直接读出；generator 保证样例里至少一批端点满足严格不等式
#     - solver 仍依赖背景两色是出现次数最多的颜色；generator 继续控制非背景占比

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8, paste


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
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


_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)


_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


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


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _checkerboard(
    h: int,
    w: int,
    even_color: int = 1,
    odd_color: int = 0,
    shift: tuple[int, int] = (0, 0),
) -> np.ndarray:
    rows, cols = np.indices((h, w))
    parity = (rows - int(shift[0]) + cols - int(shift[1])) % 2
    return np.where(parity == 0, int(even_color), int(odd_color)).astype(int)


def _checkerboard_colors(g: np.ndarray, bg_colors: set[int]) -> tuple[int, int]:
    parity_counts = {0: Counter(), 1: Counter()}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v in bg_colors:
                parity_counts[(r + c) % 2][v] += 1
    even_color = int(parity_counts[0].most_common(1)[0][0])
    odd_color = int(parity_counts[1].most_common(1)[0][0])
    return even_color, odd_color


def _extract_structure(g: np.ndarray) -> tuple[set[int], tuple[int, int], list[dict[str, object]]]:
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    rays: list[dict[str, object]] = []

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info
            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue

            rays.append(
                {
                    "color": int(g[oc[0][0], oc[0][1]]),
                    "endpoints": (endpoints[0], endpoints[-1]),
                    "outward": (-best_n[0], -best_n[1]),
                }
            )

    return bg_colors, (h, w), rays


def _axis_rooms(shape: tuple[int, int], point: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = point
    dr, dc = direction
    row_room = int(r if dr < 0 else h - 1 - r)
    col_room = int(c if dc < 0 else w - 1 - c)
    return row_room, col_room


def _endpoint_pad_need(
    shape: tuple[int, int], point: tuple[int, int], direction: tuple[int, int]
) -> tuple[str, int, str | None]:
    row_room, col_room = _axis_rooms(shape, point, direction)
    dr, dc = direction
    if row_room < col_room:
        side = "top" if dr < 0 else "bottom"
        return side, int(col_room - row_room), "row"
    if col_room < row_room:
        side = "left" if dc < 0 else "right"
        return side, int(row_room - col_room), "col"
    return "none", 0, None


def _expansion_pads(shape: tuple[int, int], rays: list[dict[str, object]]) -> tuple[int, int, int, int]:
    pads = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for spec in rays:
        outward = tuple(int(v) for v in spec["outward"])
        for pt in spec["endpoints"]:
            side, need, _limited_axis = _endpoint_pad_need(shape, pt, outward)
            if side != "none":
                pads[side] = max(pads[side], int(need))
    return int(pads["top"]), int(pads["bottom"]), int(pads["left"]), int(pads["right"])


def _render_with_pads(
    g: np.ndarray,
    bg_colors: set[int],
    rays: list[dict[str, object]],
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    top_pad, bottom_pad, left_pad, right_pad = map(int, pads)
    h, w = g.shape
    even_color, odd_color = _checkerboard_colors(g, bg_colors)
    out = _checkerboard(
        h + top_pad + bottom_pad,
        w + left_pad + right_pad,
        even_color,
        odd_color,
        shift=(top_pad, left_pad),
    )
    paste(out, g, top_pad, left_pad)

    for spec in rays:
        color = int(spec["color"])
        dr, dc = spec["outward"]
        for pr, pc in spec["endpoints"]:
            rr = int(pr + top_pad + dr)
            cc = int(pc + left_pad + dc)
            while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = color
                else:
                    break
                rr += dr
                cc += dc
    return out


def _grid_family(shape: tuple[int, int]) -> str:
    h, w = shape
    return "wide" if w > h else "tall"


def _variant_metrics(inp: np.ndarray, out: np.ndarray | None = None) -> dict[str, object]:
    bg_colors, shape, rays = _extract_structure(inp)
    pads = _expansion_pads(shape, rays)
    h, w = shape
    family = _grid_family(shape)
    endpoint_needs = []
    for spec in rays:
        outward = tuple(int(v) for v in spec["outward"])
        for pt in spec["endpoints"]:
            endpoint_needs.append(_endpoint_pad_need(shape, pt, outward))

    if out is None:
        out = _render_with_pads(inp, bg_colors, rays, pads)

    top_pad, bottom_pad, left_pad, right_pad = pads
    outer_mask = np.ones(out.shape, dtype=bool)
    outer_mask[top_pad : top_pad + h, left_pad : left_pad + w] = False
    added_non_bg = int(np.count_nonzero(~np.isin(out, list(bg_colors)) & outer_mask))
    short_axis_total = int(top_pad + bottom_pad) if family == "wide" else int(left_pad + right_pad)
    long_axis_total = int(left_pad + right_pad) if family == "wide" else int(top_pad + bottom_pad)
    return {
        "shape": shape,
        "family": family,
        "pads": pads,
        "short_axis_total": short_axis_total,
        "long_axis_total": long_axis_total,
        "added_non_bg_outside_input": added_non_bg,
        "ray_count": len(rays),
        "endpoint_needs": endpoint_needs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the canvas only along the boundary directions that actually truncate the hidden rays.

    Input structure:
    - A two-color checkerboard background fills an extreme wide or extreme tall rectangle.
    - Non-background pixels are hollow 8-cell diamonds plus short diagonal outer segments attached one step beyond diamond edges.
    - The long diagonal rays implied by those outer segments are missing from the input.

    Rule:
    - For each outer segment, infer its outward diagonal direction from the neighboring diamond.
    - Each endpoint emits one ray in that outward direction, but before drawing, compare how many diagonal steps remain to the row-side boundary versus the column-side boundary.
    - If one axis runs out sooner, extend only that limiting side just enough to match the farther axis, then draw the ray to the new grid edge.

    Key insight:
    - In these extreme-aspect grids, the hidden ray is usually cut off by the short axis only.
    - So the output does not mirror-expand all ray directions; it selectively grows the single side that was the bottleneck for each endpoint.
    """
    g = np.asarray(grid, dtype=int)
    bg_colors, shape, rays = _extract_structure(g)
    pads = _expansion_pads(shape, rays)
    return _render_with_pads(g, bg_colors, rays, pads)


GRID_SHAPES = {
    "wide": {"h": (10, 12), "w": (26, 30)},
    "tall": {"h": (26, 30), "w": (10, 12)},
}
DIAMOND_COUNT_RANGE = (1, 2)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 600
MIN_SHORT_AXIS_PAD = 4
MIN_ADDED_NON_BG_OUTSIDE = 4


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


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


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        endpoint_needs = [_endpoint_pad_need((h, w), pt, info["outward"]) for pt in (segment[0], segment[-1])]
        edge_parts.append(
            {
                "name": edge,
                "segment": segment,
                "weight": len(part_cells),
                "endpoint_needs": endpoint_needs,
            }
        )
    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _placement_matches_family(placement: dict[str, object], family: str, shape: tuple[int, int]) -> bool:
    h, w = shape
    short_axis_seen = False
    for edge_part in placement["edges"]:
        for side, need, limited_axis in edge_part["endpoint_needs"]:
            if limited_axis is None:
                continue
            if family == "wide":
                if limited_axis != "row":
                    return False
                if side not in {"top", "bottom"}:
                    return False
            else:
                if limited_axis != "col":
                    return False
                if side not in {"left", "right"}:
                    return False
            if need > 0:
                short_axis_seen = True

    if not short_axis_seen:
        return False

    diamond = placement["diamond"]
    r0 = min(r for r, _ in diamond)
    c0 = min(c for _, c in diamond)
    r1 = max(r for r, _ in diamond)
    c1 = max(c for _, c in diamond)
    if family == "wide":
        left_margin = c0
        right_margin = w - 1 - c1
        return min(left_margin, right_margin) >= 5
    top_margin = r0
    bottom_margin = h - 1 - r1
    return min(top_margin, bottom_margin) >= 5


def _search_placements(
    h: int,
    w: int,
    specs: list[tuple[str, ...]],
    family: str,
) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidates = []
        for r0, c0 in positions:
            placement = _candidate_placement(h, w, r0, c0, edges)
            if placement is None:
                continue
            if not _placement_matches_family(placement, family, (h, w)):
                continue
            candidates.append(placement)
        if not candidates:
            return None
        candidate_cache[edges] = candidates

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


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((edge_part["weight"], random.random(), pi, ei) for pi, placement in enumerate(placements) for ei, edge_part in enumerate(placement["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _choose_shape() -> tuple[str, int, int]:
    family = random.choice(tuple(GRID_SHAPES))
    dims = GRID_SHAPES[family]
    h = random.randint(*dims["h"])
    w = random.randint(*dims["w"])
    return family, h, w


def _gen_attempt() -> np.ndarray | None:
    family, h, w = _choose_shape()
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs, family)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    inp = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(inp, edge_part["segment"], segment_colors[pi][ei])

    if not _bg_counts_safe(inp):
        return None

    out = main(inp)
    metrics = _variant_metrics(inp, out)
    top_pad, bottom_pad, left_pad, right_pad = metrics["pads"]
    if family != metrics["family"]:
        return None
    if metrics["short_axis_total"] < MIN_SHORT_AXIS_PAD:
        return None
    if metrics["long_axis_total"] != 0:
        return None
    if metrics["added_non_bg_outside_input"] < MIN_ADDED_NON_BG_OUTSIDE:
        return None
    if family == "wide" and (top_pad + bottom_pad == 0):
        return None
    if family == "tall" and (left_pad + right_pad == 0):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
