# Variant of 80a900e0
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 连续涂色：射线路径上每个背景像素都涂上外边框颜色（每个都涂，无跳过）... → 间歇跳格涂色（每隔一个像素涂色）
#
# Design analysis
# ===============
# source_task:
#   none
# variant_spec:
#   preserved_structure:
#     - 背景仍是 0/1 棋盘格，左上角固定为 1。
#     - 菱形内框仍是 5x5 包围盒内的 8 像素空心菱形。
#     - 外边框仍是贴着菱形四条边之一的短对角线段，方向判定、内外法向量判定、两端点发射机制都保持不变。
#   changed_behavior:
#     - 原题：从每个外边框端点沿 outward 对角方向逐格前进，遇到的每个背景格都上色，直到边界或障碍。
#     - 变体：路径仍逐格前进，也仍在遇到首个非背景格时停止；但只给路径上第 1、3、5... 个格子上色，第 2、4、6... 个格子保留棋盘格背景。
#     - 该“跳格”从每个端点各自重新开始计数，因此同一外边框的两条射线都遵循 paint, skip, paint, skip...。
#   concrete_trace:
#     - 例：某端点 outward=(1,1)，连续路径为 [(5,5),(6,6),(7,7),(8,8)]。
#     - 原题输出会着色 4 个点。
#     - 变体输出只着色 [(5,5),(7,7)]，[(6,6),(8,8)] 保持背景；若 (7,7) 前就遇到非背景格，则在该障碍处停止，不会跨过去继续“隔格”涂色。
#   complexity_calibration:
#     - 若所有射线都非常短，虚线效果会退化得不明显，因此生成器额外要求样例里存在足够多的“被跳过格”，保证人眼能从少量样例看出节奏变化。
#     - 颜色规则不变，不引入新的颜色映射；变化只体现在几何轨迹的涂色间距上，避免多维同时变化带来歧义。
#   implicit_assumption_audit:
#     - `bg_colors = top-2 colors` 仍成立，因为生成器继续保证 0/1 比任一单独非背景色更常见。
#     - `_is_diamond8`、`_outer_line_info`、法向量推断依赖的几何结构均未改变。
#     - 原 solver 在 while 循环里“见背景就涂，否则停止”。变体中如果只在偶数步检查背景，会错误跨越奇数步上的障碍，所以新 solver 必须在每一步都先检查是否为背景，再决定这一格是涂还是跳过。
#     - 原 generator 用最终着色后的 ray cells 做占位即可；变体中如果只保留“实际着色格”做占位，solver 会在被跳过但仍属于路径的格子上撞到别的对象并提前终止。因此生成器仍需为完整连续路径保留空间。
#   placement_legality:
#     - 放置时仍禁止完整路径与任何菱形、外边框、其他射线路径重叠，确保 solver 的路径推进不会被意外阻断。
#     - 输出仍采用 output-first：先构造含虚线射线的输出，再把实际着色的射线格恢复为棋盘格得到输入，并验证 `main(input)==output`。
#   observability_and_cleanliness:
#     - 视觉上，虚线射线与连续射线的差异体现在同色对角线上出现规律性空隙，且外边框/菱形仍是实体，因而角色分离清晰。
#     - 生成器要求至少产生若干 skipped cells，避免样例整体与原题连续版本几乎一致。

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
    Variant rule: trace the same diagonal rays as the original task, but paint only
    every other background cell along each ray, starting with the first step away
    from the outer segment endpoint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
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
                step_idx = 0
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) not in bg_colors:
                        break
                    if step_idx % 2 == 0:
                        out[rr, cc] = ocol
                    step_idx += 1
                    rr += outward[0]
                    cc += outward[1]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300
MIN_TOTAL_SKIPPED_CELLS = 4

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


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


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


def _painted_path_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path[::2]


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    total_skipped = 0
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
        ray_paths = [
            _ray_cells((h, w), segment[0], info["outward"]),
            _ray_cells((h, w), segment[-1], info["outward"]),
        ]
        full_rays = [cell for path in ray_paths for cell in path]
        painted_rays = [cell for path in ray_paths for cell in _painted_path_cells(path)]
        part_cells = segment + full_rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        skipped_here = sum(len(path) // 2 for path in ray_paths)
        total_skipped += skipped_here
        edge_parts.append(
            {
                "segment": segment,
                "ray_paths": ray_paths,
                "painted_rays": painted_rays,
                "weight": len(segment) + len(painted_rays),
                "skipped_count": skipped_here,
            }
        )
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "skipped_count": total_skipped,
    }


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
        ((ep["weight"], random.random(), pi, ei)
         for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
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


def _variant_visibility_ok(placements: list[dict[str, object]]) -> bool:
    total_skipped = sum(int(pl["skipped_count"]) for pl in placements)
    return total_skipped >= MIN_TOTAL_SKIPPED_CELLS


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None or not _variant_visibility_ok(placements):
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["painted_rays"], color)
            ray_coords.extend(edge_part["painted_rays"])

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
