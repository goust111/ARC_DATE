# Combo variant of 80a900e0 (index 3)
# combo_spec: D2/1+D6/0
# Name: 极简点射
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    原题每条菱形边外侧都有 3 像素对角线段，并从两端各发 1 条射线。这里把两次改动真正融合：
#    先采用 D2/1 的“只保留更远端点发射”，再采用 D6/0 的“把可见外物体压缩成单像素”。
#    因此输入里每条被选边只剩下 1 个单像素标记，而且这个点不是原 D6 的中心角点，而是原 3 像素
#    外线段的某一个端点。例：TR 边原始外线段是 [(-1,3),(0,4),(1,5)]。若锚定左半段，
#    真正发射点是 (-1,3)；若锚定右半段，真正发射点是 (1,5)。融合后输入直接只显示这个点，
#    输出也只从该点沿 outward 方向补 1 条射线。
# b) Complexity calibration:
#    每条边只有 2 个合法点位，四条边共 8 个离散点位，复杂度比“任意单点”低得多，但比 D6/0 的
#    4 个角点更丰富。点位既编码“属于哪条边”，也编码“该边选了哪一端”，人类能从少量样例观察到
#    同一 outward 方向会对应两个不同点位，不会退化成固定的四角查表题。
# c) Implicit assumptions audit:
#    - 原 `main()` 依赖 `_outer_line_info()` 从至少 2 个像素恢复斜率；单点不再满足，必须改成
#      直接查 diamond bbox 周围 8 个候选点位。
#    - 原 `main()` 依赖“两个端点都发射”；融合后只允许 1 条射线，且发射点就是可见单点本身。
#    - D6/0 单变体把单点当作 3 像素线段中点并重建双端点；这里该前提失效，因为可见点本身就是
#      D2/1 选中的远端点，不再存在“再往两侧补两个虚拟端点”的理由。
#    - 新 solver 假设 marker 只会落在 diamond bbox 周围 8 个精确位置；generator 会只在这些位置放点。
# d) Placement legality:
#    生成器继续 output-first。每条边先选 edge，再选 anchor(0/1) 决定保留原 3 像素外线段的哪一端。
#    搜索时同时为三类轨迹留空：可见 marker、本规则真实射线、以及旧双端点规则下会出现但如今被抑制
#    的 companion ray。这样既避免对象重叠，也保证融合差异不会因碰撞而被掩盖。
# e) Visual observability:
#    输入里每个菱形外侧只剩稀疏单点，输出里每点只长出 1 条同色对角线。点位不是纯四角，而是围绕四角
#    的 8 个离散位置，因此解题者必须利用相对位置推断 outward 方向，同时观察“同一边有两种可见端点”。
# f) Visual cleanliness:
#    保留原题的棋盘背景和 5x5 空心菱形，只减少外物体密度。搜索时对输入可见物体加 8 邻域缓冲，避免
#    不同 diamond/marker 太贴近，防止视觉上把别的 marker 误看成当前 diamond 的外标记。
# g) Color inferability:
#    颜色规则保持简单：diamond 用独立颜色，marker 与其射线同色，颜色值不参与方向或端点选择。
# h) Fusion-specific non-degeneracy target:
#    若被抑制的 companion ray 长度为 0，则“单射线”和“旧双射线”在该边上可能看起来一样。生成器强制
#    每个被选 marker 的 companion ray 也至少有 1 格潜在可见长度，并在放置时为它预留空间。
# i) New solver assumptions and generator guarantees:
#    - solver 逐个 diamond 只检查 bbox 周围 8 个候选 marker 位；generator 只在这些位点放 marker。
#    - solver 默认 marker 外侧射线不会立刻撞到别的非背景对象；generator 在 output-first 搜索时显式保证。
#    - solver 不解析任意噪声单点；generator 用输入 halo 让不同对象之间保留缓冲，从而保持局部结构清晰。

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


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
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


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_RAY_CELLS = 1
MIN_COMPANION_RAY_CELLS = 1
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TL": {
        "full_segment": ((-1, 1), (0, 0), (1, -1)),
        "marker_offsets": ((-1, 1), (1, -1)),
        "outward": (-1, -1),
    },
    "TR": {
        "full_segment": ((-1, 3), (0, 4), (1, 5)),
        "marker_offsets": ((-1, 3), (1, 5)),
        "outward": (-1, 1),
    },
    "BL": {
        "full_segment": ((3, -1), (4, 0), (5, 1)),
        "marker_offsets": ((3, -1), (5, 1)),
        "outward": (1, -1),
    },
    "BR": {
        "full_segment": ((3, 5), (4, 4), (5, 3)),
        "marker_offsets": ((3, 5), (5, 3)),
        "outward": (1, 1),
    },
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _marker_candidates(r0: int, c0: int) -> dict[tuple[int, int], dict[str, object]]:
    candidates: dict[tuple[int, int], dict[str, object]] = {}
    for edge, info in _EDGE_INFO.items():
        for anchor, (dr, dc) in enumerate(info["marker_offsets"]):
            candidates[(r0 + dr, c0 + dc)] = {"edge": edge, "anchor": anchor, "outward": info["outward"]}
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one diagonal ray from each sparse point marker beside a hollow diamond on a checkerboard.

    Input structure:
    - the background is the usual two-color checkerboard;
    - diamonds are 8-pixel hollow outlines in a non-background color;
    - each occupied diamond edge is represented by a single non-background point, chosen from one of
      two endpoint positions outside that edge, for a total of eight legal marker sites around the bbox.

    Rule:
    - For each diamond, inspect only those eight legal marker sites around its 5x5 bbox.
    - A present marker identifies both the underlying edge and which endpoint of that edge survived the
      D2 reduction.
    - From that marker pixel itself, cast exactly one diagonal ray in the corresponding outward direction,
      painting background cells with the marker color until another non-background pixel or the boundary.

    Key insight:
    The visible point is already the chosen emitting endpoint, not the midpoint of a hidden 3-pixel segment,
    so the solver must recover direction from the point's relative position and emit only one ray per point.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        r0, c0, _, _ = _bbox(dpts)
        for (mr, mc), info in _marker_candidates(r0, c0).items():
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            mcol = int(g[mr, mc])
            if mcol in bg_colors:
                continue
            dr, dc = info["outward"]
            rr, cc = mr + dr, mc + dc
            while 0 <= rr < h and 0 <= cc < w:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = mcol
                else:
                    break
                rr += dr
                cc += dc

    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edge_specs: tuple[tuple[str, int], ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    actual_cells = set(diamond)
    legacy_cells = set(diamond)
    input_cells = set(diamond)
    edge_parts = []

    for edge, anchor in edge_specs:
        info = _EDGE_INFO[edge]
        full_segment = [(r0 + dr, c0 + dc) for dr, dc in info["full_segment"]]
        if not all(0 <= r < h and 0 <= c < w for r, c in full_segment):
            return None

        marker = full_segment[0] if anchor == 0 else full_segment[-1]
        companion = full_segment[-1] if anchor == 0 else full_segment[0]
        actual_ray = _ray_cells((h, w), marker, info["outward"])
        companion_ray = _ray_cells((h, w), companion, info["outward"])
        if len(actual_ray) < MIN_RAY_CELLS or len(companion_ray) < MIN_COMPANION_RAY_CELLS:
            return None

        actual_part = [marker] + actual_ray
        legacy_part = full_segment + actual_ray + companion_ray
        if len(set(legacy_part)) != len(legacy_part):
            return None
        if any(cell in actual_cells for cell in legacy_part):
            return None
        if any(cell in legacy_cells for cell in legacy_part):
            return None

        actual_cells.update(actual_part)
        legacy_cells.update(legacy_part)
        input_cells.add(marker)
        edge_parts.append(
            {
                "edge": edge,
                "anchor": anchor,
                "marker": marker,
                "rays": actual_ray,
                "companion_ray": companion_ray,
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
            p for r0, c0 in positions if (p := _candidate_placement(h, w, r0, c0, edge_specs)) is not None
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
    marker_colors = _assign_marker_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = marker_colors[pi][ei]
            _paint_cells(out, [edge_part["marker"]], color)
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
