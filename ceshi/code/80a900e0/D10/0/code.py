# Variant of 80a900e0
# Dimension: D10 — 背景类型 (meta)
# Change: 棋盘格背景（颜色0和1交替，左上角为1），所有前景像素位于(r+c)%2==0的格子上... → 纯色背景
"""
design_analysis:
  variant_goal:
    preserve:
      - 5x5 bbox 内的 8 像素空心菱形
      - 每条被选中的菱形边外侧有一条短对角线段
      - 每条短线段只从两个端点向外发射 2 条对角射线
    change:
      - 背景从 0/1 棋盘格改为纯色 0
      - 输入里删掉射线时，不再恢复棋盘格，只恢复纯背景色 0
      - generator 不再要求对象锚点满足固定相位；菱形可放在任意 (r0,c0)
        上，因此前景会落在偶相位或奇相位两类位置上
  concrete_behavior_and_trace:
    input:
      - 纯色 0 背景
      - 若某个菱形放在 (r0+c0) 为奇数的位置，则该菱形、其外框线段和未来
        射线路径全都落在原题不允许的奇相位格上
      - 输入只包含菱形和短外框线段，不包含射线
    output:
      - 与原题一样，从每条外框线段两个端点沿远离菱形的对角方向涂到边界
      - 但被填充的路径位于纯色背景上，不再和棋盘格交错
    trace_example:
      - 原题: 背景色集合由前两大颜色 {0,1} 给出；删射线时要按 (r+c+1)%2 恢复
        棋盘格；放置时 r0+c0 必须是偶数
      - 变体: 背景只取最常见颜色 0；删射线时直接恢复 0；放置时 r0+c0 可奇可偶
      - 差异: solver 的“可涂色单元”判定从 `in {0,1}` 变为 `== bg`，generator 的
        “恢复背景”从棋盘函数变为常数 0，同时取消固定相位搜索
  complexity_calibration:
    - 只改 D10，不改菱形尺寸、边段数量范围、射线方向规则
    - 保留 1-3 个菱形、每个 2-4 条外框边段，避免任务退化成单对象单颜色
    - 前景颜色仍限制在 2-9，避免把 D10 与调色板扩张混在一起
  hidden_assumptions_review:
    - original `_top_k_colors(..., 2)`: 依赖背景恰好有两色；纯色背景下会误把某个前景色
      当背景，因此必须改成“最常见单色背景”
    - original `non_bg = unique - bg_colors`: 依赖前两大色都可穿透；变体中只能把真实背景
      视为可覆盖区域
    - original ray painting `if out[rr,cc] in bg_colors`: 在纯色背景下必须收窄为 `== bg`
    - original `_checkerboard` 和 `inp[ray_coords] = (r+c+1)%2`: 直接依赖旧背景，必须删除
    - original placement filter `(r+c)%2==0`: 这是旧相位约束；若保留，样本会表面上接近原题，
      无法体现 D10 的“去棋盘格/去相位”变化
  placement_legality:
    - 候选放置仍然 output-first 计算完整射线，要求同一对象内部、不同对象之间都不重叠
    - 允许边段因贴近边界而裁成 2 或 3 像素，但少于 2 像素则拒绝
    - 生成器额外要求至少一个菱形锚点为奇相位，确保样本不退化回原题固定相位
    - 还要求背景色 0 依旧是全图最常见颜色，保证 solver 的“最常见色=背景”假设成立
  visual_observability:
    - 人眼能直接看到背景从棋盘格变成纯黑底
    - 至少一个对象落在原题不允许的奇相位上，因此即使把网格想象成叠加棋盘，
      也能看到前景不再跟随原题相位
    - 规则仍然统一为“外框端点向外发射”，不存在多个隐蔽分支
  visual_cleanliness:
    - 仍禁止对象、边段、射线互相重叠，避免纯色背景下出现颜色覆盖混乱
    - 菱形和边段仍保持一格外侧距离，结构清晰
  color_inference:
    - 颜色不承载额外映射语义；颜色仅区分不同菱形和边段
    - 规则完全由几何关系决定，因此 2-4 个样例足以归纳
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设最常见颜色就是纯背景色；generator 用大画布 + 计数检查保证
    - solver 假设与菱形相邻的非背景像素会组成对角短线段；generator 只生成这类结构
    - solver 假设不同对象的射线不会先互相撞上再改变语义；generator 在 output-first
      搜索里预先排除了所有碰撞
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _background_color(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays outward from short outer-border segments adjacent to diamond outlines.

    D10 variant change:
    - Background is a single most-common color instead of a 0/1 checkerboard.
    - Rays may be restored onto any phase because generator no longer constrains placement parity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    non_bg = set(int(v) for v in np.unique(g)) - {bg}
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset and int(g[rr, cc]) != bg:
                    adj.append((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
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
            outward = (-best_n[0], -best_n[1])

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) != bg:
                        break
                    out[rr, cc] = ocol
                    rr += outward[0]
                    cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
BG_COLOR = 0
MAX_GEN_RETRIES = 300

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


def _solid_background(h: int, w: int) -> np.ndarray:
    return np.full((h, w), BG_COLOR, dtype=int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


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


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
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
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells(
            (h, w), segment[-1], info["outward"]
        )
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "anchor": (r0, c0),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    all_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4)]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = all_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            placement
            for r0, c0 in positions
            if (placement := _candidate_placement(h, w, r0, c0, edges)) is not None
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
    return [placement for placement in placed if placement is not None]


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for color in diamond_colors:
        counts[color] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((edge_part["weight"], random.random(), pi, ei)
         for pi, placement in enumerate(placements) for ei, edge_part in enumerate(placement["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda color: (counts[color], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_count_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[color] for color in COLOR_RANGE), default=0)
    return counts[BG_COLOR] > max_non_bg


def _has_non_original_phase(placements: list[dict[str, object]]) -> bool:
    return any((placement["anchor"][0] + placement["anchor"][1]) % 2 == 1 for placement in placements)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None or not _has_non_original_phase(placements):
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _solid_background(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_count_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        inp[np.array(rs), np.array(cs)] = BG_COLOR

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
