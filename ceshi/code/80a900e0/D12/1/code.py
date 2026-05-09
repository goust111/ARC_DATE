# Variant of 80a900e0
# Dimension: D12 — 输出网格形状 (grid_shape)
# Change: 输出与输入同尺寸（same）... → 扩展网格以容纳所有射线
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 在原网格尺寸内，从每条外侧短对角线段的两个端点向外发射射线，直到撞到当前网格边界
#     variant: 先根据输入里“可见骨架”（钻石轮廓 + 外侧短段）推断应向哪些边扩展，再在扩展后的棋盘格上发射同样的射线
#   concrete_geometry:
#     visible_input:
#       - 输入仍然只有棋盘格背景、8 像素菱形轮廓、以及贴着菱形四边外侧的短对角线段
#       - 输入里仍然不显示射线
#     output:
#       - 输出保留原输入的全部内容
#       - 如果某条射线的方向包含上/下/左/右分量，则对应边界可能被向外补出新的棋盘格区域
#       - 射线在补出的新区域里继续延伸，直到新的网格边界
#   small_trace:
#     - 原逻辑：一个 TR 段向东北发射，若段离右边界很近，则射线很快在原网格右边停下
#     - 新逻辑：先看整张输入中所有非背景像素的 bbox；若存在向右发射的段，则把 bbox 左侧已有的空白镜像到右侧，
#       若存在向上发射的段，则把 bbox 下侧已有的空白镜像到上侧。然后把原输入平移贴到更大的棋盘格中，再让该 TR 段
#       向东北继续发射到新边界。差异不在局部作画规则，而在“边界先被重建”。
#   complexity_calibration:
#     - 保留 1-3 个钻石、每个 2-4 条外段、颜色和原题同级别，避免把题目简化成单对象单方向
#     - padding 大小不是常数，而是由可见骨架 bbox 在对边留下的空白量决定，因此每个样例可变，且多个段会重复支撑同一扩展方向
#   hidden_assumption_audit:
#     - 原 main 里 `out = g.copy()` 隐含“输出尺寸不变”；变体下必须先创建更大的画布，否则所有新射线仍会被截断
#     - 原 while 循环把越界当终止条件，这仍然成立，但“边界坐标”必须改成扩展后的边界
#     - 原代码默认输入背景就是完整棋盘格；变体新增 padding 后，若直接用固定相位重建背景，原输入与新外圈棋盘格可能错相。
#       因此需要按 `(top_pad + left_pad)` 的奇偶来平移棋盘格相位，保证原输入区域无缝接上新背景
#     - 原代码按原坐标读取段端点；变体下端点仍然从输入里识别，但绘制时必须整体加上 `(top_pad, left_pad)` 平移
#   placement_legality:
#     - 生成器继续沿用原题的可见对象放置搜索，并用“在原尺寸内若把射线画出来也不能碰撞”这一强约束来保证清晰度
#     - 由于新增射线只会出现在原网格外，原网格内已验证无碰撞，则扩展部分天然不会撞到其他对象
#   visual_observability:
#     - 人类能直接看到：输出不再与输入同尺寸；而且扩展发生在哪些边，取决于哪些外段朝那些方向发射
#     - 扩展量也可从输入里看出来：可见骨架离某一侧很近，则输出会在其对边补出相同厚度的棋盘格空白
#   cleanliness:
#     - 新增像素全部是连续棋盘格和沿 45 度方向继续前进的细射线，没有额外标记层
#     - 生成器拒绝没有实际新增射线像素的样例，避免“只是多了空白边框”的脏输出
#   color_inference:
#     - 颜色规则不变：射线颜色仍等于对应外段颜色
#     - 不引入新的颜色映射，避免把 D12 变化和颜色变化混在一起
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设输入里仍是标准两色棋盘格，且非背景对象只由菱形与外段构成；generator 保持这一点
#     - solver 用所有可识别外段的发射方向决定扩展边；generator 至少放置一个会带来非零扩展且真的产生新增射线像素的方向
#     - solver 依赖背景仍是最常见的两种颜色；generator 继续限制非背景像素占比，保证这一点稳定

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


def _checkerboard(h: int, w: int, even_color: int = 1, odd_color: int = 0, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
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


def _visible_bbox(g: np.ndarray, bg_colors: set[int]) -> tuple[int, int, int, int] | None:
    mask = ~np.isin(g, list(bg_colors))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def _expansion_pads(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None, rays: list[dict[str, object]]) -> tuple[int, int, int, int]:
    if bbox is None or not rays:
        return 0, 0, 0, 0
    h, w = shape
    r0, c0, r1, c1 = bbox
    top_margin = int(r0)
    bottom_margin = int(h - 1 - r1)
    left_margin = int(c0)
    right_margin = int(w - 1 - c1)

    need_top = any(int(spec["outward"][0]) < 0 for spec in rays)
    need_bottom = any(int(spec["outward"][0]) > 0 for spec in rays)
    need_left = any(int(spec["outward"][1]) < 0 for spec in rays)
    need_right = any(int(spec["outward"][1]) > 0 for spec in rays)

    top_pad = bottom_margin if need_top else 0
    bottom_pad = top_margin if need_bottom else 0
    left_pad = right_margin if need_left else 0
    right_pad = left_margin if need_right else 0
    return top_pad, bottom_pad, left_pad, right_pad


def _render_with_pads(g: np.ndarray, bg_colors: set[int], rays: list[dict[str, object]], pads: tuple[int, int, int, int]) -> np.ndarray:
    top_pad, bottom_pad, left_pad, right_pad = map(int, pads)
    h, w = g.shape
    even_color, odd_color = _checkerboard_colors(g, bg_colors)
    out = _checkerboard(h + top_pad + bottom_pad, w + left_pad + right_pad, even_color, odd_color, shift=(top_pad, left_pad))
    paste(out, g, top_pad, left_pad)

    for spec in rays:
        ocol = int(spec["color"])
        dr, dc = spec["outward"]
        for pr, pc in spec["endpoints"]:
            rr = int(pr + top_pad + dr)
            cc = int(pc + left_pad + dc)
            while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = ocol
                else:
                    break
                rr += dr
                cc += dc
    return out


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, object]:
    bg_colors, shape, rays = _extract_structure(inp)
    bbox = _visible_bbox(inp, bg_colors)
    pads = _expansion_pads(shape, bbox, rays)
    top_pad, _bottom_pad, left_pad, _right_pad = pads
    h, w = inp.shape
    outer_mask = np.ones(out.shape, dtype=bool)
    outer_mask[top_pad : top_pad + h, left_pad : left_pad + w] = False
    added_non_bg = int(np.count_nonzero(~np.isin(out, list(bg_colors)) & outer_mask))
    return {
        "pads": pads,
        "expanded_area": int(out.size - inp.size),
        "added_non_bg_outside_input": added_non_bg,
        "ray_count": len(rays),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the canvas in the ray directions, then draw the same diagonal rays to the new boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg_colors, shape, rays = _extract_structure(g)
    bbox = _visible_bbox(g, bg_colors)
    pads = _expansion_pads(shape, bbox, rays)
    return _render_with_pads(g, bg_colors, rays, pads)


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
MIN_EXPANDED_AREA = 8

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
        edge_parts.append({"segment": segment, "weight": len(part_cells)})
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


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


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
    if (top_pad + bottom_pad + left_pad + right_pad) == 0:
        return None
    if metrics["expanded_area"] < MIN_EXPANDED_AREA:
        return None
    if metrics["added_non_bg_outside_input"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
