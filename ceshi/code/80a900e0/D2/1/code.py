# Variant of 80a900e0
# Dimension: D2 — 射线发射点 (path)
# Change: 从外边框线段的两个端点发射，每条线段恰好发射2条射线... → 仅从距菱形最远的端点发射

# Design analysis for this D2 variant
# a) Concrete behavior:
#    原题里，每条外侧短对角线段都从两个端点各发 1 条外射线。D2 变体要把“发射点”
#    改成只保留 1 个端点，而且该端点必须由可见几何决定，而不是颜色值或隐藏顺序。
#    直接沿用原始 3 像素对称线段会有问题：两端点到 5x5 菱形中心的距离完全相同，
#    “最远端点”没有唯一答案，solver 可以随便选，但人类无法从样例推出规则。
#    因此生成器做一个必要的适配：每条外线段仍保持原来的方向、仍紧贴对应菱形边外侧，
#    但长度改为 2 像素，并且取原 3 像素线段的左/右（或上/下）半段之一。这样两端点
#    到菱形中心的距离一近一远，新的 D2 规则才有唯一且可观察的答案。
#    小例子：TR 边原始外线段排序后为 [(-1,3),(0,4),(1,5)]。
#    若变体输入使用前两格 [(-1,3),(0,4)]，原题输出会从两端都向 (-1,+1) 发射两条平行射线；
#    变体输出只从更远的 (-1,3) 端点发射。若输入使用后两格 [(0,4),(1,5)]，则只从 (1,5)
#    发射。差异完全体现在“哪一个端点会延伸”。
# b) Complexity calibration:
#    只允许 2 种可见锚定状态（截取 3 像素线段的前两格或后两格），避免自由度过多。
#    每个网格仍保留 1-3 个菱形、每个菱形 2-4 条外线段，保证同一题内会重复出现
#    “远端在这一头/那一头”的两类证据，不会退化成单个偶然案例。
# c) Implicit assumptions audit:
#    - _outer_line_info 只要求对角共线且长度 >= 2；2 像素线段仍满足。
#    - inward/outward 法线判定依赖“把外线段整体向内平移 1 格后能覆盖菱形边”。
#      对这类 2 像素子线段仍成立，因为它们只是原 3 像素线段的连续子段。
#    - 原 main() 最关键的旧假设是“两个端点都要发射”。这在变体里失效，必须替换成
#      “比较端点到菱形 bbox 中心的距离，只保留更远者”。若距离并列，solver 用确定性
#      次级排序兜底；生成器不会产出并列样例。
#    - 旧 generator 默认 output-first，但没要求 suppressed ray（被删掉的旧射线）必须为空。
#      变体里如果 suppressed ray 正好撞到别的物体，会导致新旧规则输出偶然相同，因此搜索
#      时必须同时为“保留的射线”和“被抑制的旧射线”预留空间。
# d) Placement legality:
#    生成器仍采用 output-first。先决定菱形位置、边集合、每条边的 2 像素锚定方式，再同时
#    计算新规则射线和旧规则下的另一条 suppressed ray。搜索时禁止任何对象占用这些旧/新射线
#    轨迹，保证网格不重叠、不越界，也保证 D2 差异一定真实存在。
# e) Visual observability:
#    人类在样例里能直接看到：同样是 2 像素对角短线，输出只会从远离菱形中心的一端延长。
#    因为锚定既有“靠上/靠左”也有“靠下/靠右”两种，规则不是固定发射上端或下端，而是
#    真正依赖“离中心更远”的可见几何关系。
# f) Visual cleanliness:
#    保留原题棋盘背景、空心菱形和对角射线的整体视觉语言，只把外线段变短并偏置。
#    搜索时对输入物体加 8 邻域缓冲，避免不同菱形或线段贴得太近，防止 solver 把外物体误认
#    成当前菱形的邻接部分。
# g) Color inferability:
#    颜色规则不变：菱形颜色与外线段颜色独立，背景仍是 0/1 棋盘。颜色不参与排序或端点选择，
#    因此解题者只需观察几何，不需要推断额外的颜色映射。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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


def _diamond_center2(pts: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(pts)
    return r0 + r1, c0 + c1


def _dist2_from_center2(pt: tuple[int, int], center2: tuple[int, int]) -> int:
    dr2 = 2 * pt[0] - center2[0]
    dc2 = 2 * pt[1] - center2[1]
    return dr2 * dr2 + dc2 * dc2


def _choose_emitter(
    endpoints: list[tuple[int, int]], center2: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    ranked = sorted(
        endpoints,
        key=lambda p: (_dist2_from_center2(p, center2), p[0], p[1]),
        reverse=True,
    )
    emitter = ranked[0]
    other = ranked[1] if len(ranked) > 1 else None
    return emitter, other


def main(grid: np.ndarray) -> np.ndarray:
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
        center2 = _diamond_center2(dpts)

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
                sc = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            emitter, _ = _choose_emitter(endpoints, center2)
            rr, cc = emitter[0] + outward[0], emitter[1] + outward[1]
            while 0 <= rr < h and 0 <= cc < w:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = ocol
                else:
                    break
                rr += outward[0]
                cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

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


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _segment_from_anchor(full_segment: list[tuple[int, int]], anchor: int) -> list[tuple[int, int]]:
    if len(full_segment) != 3:
        raise ValueError("expected unclipped 3-cell outer segment")
    if anchor not in (0, 1):
        raise ValueError("anchor must be 0 or 1")
    return full_segment[anchor:anchor + 2]


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edge_specs: tuple[tuple[str, int], ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    dset = set(diamond)
    center2 = (2 * (r0 + 2), 2 * (c0 + 2))
    actual_cells = set(diamond)
    legacy_cells = set(diamond)
    input_cells = set(diamond)
    edge_parts = []

    for edge, anchor in edge_specs:
        info = _EDGE_INFO[edge]
        full_segment = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
        if not all(0 <= r < h and 0 <= c < w for r, c in full_segment):
            return None
        full_segment = _sort_segment(full_segment, info["slope"])
        segment = _segment_from_anchor(full_segment, anchor)
        emitter, other = _choose_emitter(segment, center2)
        if other is None:
            return None

        ray_map = {
            segment[0]: _ray_cells((h, w), segment[0], info["outward"]),
            segment[-1]: _ray_cells((h, w), segment[-1], info["outward"]),
        }
        if not ray_map[segment[0]] or not ray_map[segment[-1]]:
            return None

        rays = ray_map[emitter]
        suppressed = ray_map[other]
        actual_part = segment + rays
        legacy_part = segment + ray_map[segment[0]] + ray_map[segment[-1]]
        if len(set(legacy_part)) != len(legacy_part):
            return None
        if any(cell in dset for cell in legacy_part):
            return None
        if any(cell in actual_cells for cell in legacy_part):
            return None

        actual_cells.update(actual_part)
        legacy_cells.update(legacy_part)
        input_cells.update(segment)
        edge_parts.append(
            {
                "segment": segment,
                "rays": rays,
                "suppressed": suppressed,
                "weight": len(actual_part),
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(actual_cells),
        "legacy_cells": frozenset(legacy_cells),
        "input_cells": frozenset(input_cells),
        "input_halo": frozenset(_expand8(input_cells, (h, w))),
    }


def _search_placements(h: int, w: int, specs: list[tuple[tuple[str, int], ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[tuple[str, int], ...], list[dict[str, object]]] = {}
    for edge_specs in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edge_specs] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edge_specs)) is not None
        ]
        if not candidate_cache[edge_specs]:
            return None

    occupied_legacy: set[tuple[int, int]] = set()
    occupied_input_halo: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            if not occupied_legacy.isdisjoint(placement["legacy_cells"]):
                continue
            if not occupied_input_halo.isdisjoint(placement["input_cells"]):
                continue

            prev_legacy = set(occupied_legacy)
            prev_halo = set(occupied_input_halo)
            occupied_legacy.update(placement["legacy_cells"])
            occupied_input_halo.update(placement["input_halo"])
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_legacy.clear()
            occupied_legacy.update(prev_legacy)
            occupied_input_halo.clear()
            occupied_input_halo.update(prev_halo)
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


def _sample_specs(diamond_count: int) -> list[tuple[tuple[str, int], ...]]:
    specs: list[tuple[tuple[str, int], ...]] = []
    anchors: list[int] = []
    for _ in range(diamond_count):
        edges = sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND)))
        edge_specs = []
        for edge in edges:
            anchor = random.randint(0, 1)
            edge_specs.append((edge, anchor))
            anchors.append(anchor)
        specs.append(tuple(edge_specs))

    if len(anchors) >= 2 and len(set(anchors)) == 1:
        pi = random.randrange(len(specs))
        ei = random.randrange(len(specs[pi]))
        edge, anchor = specs[pi][ei]
        mutable = [list(s) for s in specs]
        mutable[pi][ei] = (edge, 1 - anchor)
        specs = [tuple(s) for s in mutable]
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _sample_specs(diamond_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
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
