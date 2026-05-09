# Combo variant of 80a900e0 (index 26)
# combo_spec: D10/0+D18/2
# Name: 纯底补全射线
"""
design_analysis:
  a) concrete_behavior_and_trace:
    - 保留原题的核心几何：5x5 bbox 内的 8 像素空心菱形；被选中的 2-4 条菱形边外各有一条
      2-3 像素短对角线段；每条线段仍从两个端点沿远离菱形的对角方向发射两条射线。
    - D10 改动把背景从 0/1 棋盘格改成纯色 0，因此删除射线后不再恢复棋盘格，只恢复 0，
      且对象锚点不再受 `(r0+c0)%2==0` 约束。
    - D18 改动把输入从“完全无射线”改成“保留每条射线的一段连续前缀”，输出补全剩余后缀。
    - 具体追踪一条边：
      original: output 先画满线段+整条射线，input 把整条射线删空；solver 看到独立短线段后
      直接从端点往外画到边界。
      combo: output 仍先画满；input 只把每条射线远端后缀擦回纯背景 0，于是输入里变成
      “短线段 + 一截同色射线 stub”；solver 不能再把邻接组件当作单一直线段，而必须先按
      菱形的四个固定边模板识别出短线段，再沿该方向穿过已有同色 stub，继续把剩余 0 补到边界。
    - 与原题的关键差异有两个：背景不再提供棋盘相位线索；输入里已经出现部分射线，补全长度
      需要从“直到边界”为终点来推断。
  b) complexity_calibration:
    - 不改变菱形尺寸、边模板、颜色角色、对象数量范围，避免把多个维度混成全新任务。
    - 截断自由度控制为“每条射线仅保留一个从端点出发的连续前缀”，不允许空洞或多段残留，
      这样肉眼仍能直接看出延伸方向。
    - 生成器要求至少 2 条 genuinely partial rays，且至少 1 个菱形锚点落在原题禁用的奇相位，
      防止样本退化成“纯底但仍像原题排布”或“根本没有部分射线”的弱版本。
  c) hidden_assumptions_review:
    - 原 solver 用 `_top_k_colors(..., 2)` 推断棋盘格背景；纯底后这个前提失效，必须改成
      “最常见单色 = 背景”。
    - 原 solver 把菱形周围非背景像素分成 8 连通分量，并假设每个分量本身是对角短线段。
      部分射线出现后，线段与 stub 会合成折线组件，这个假设失效。
    - 原绘制循环假设端点外第一格是背景；现在第一格可能已经是同色 stub，所以扩展逻辑必须
      允许“穿过同色、填补背景、遇到其他非背景则停止”。
    - 原 generator 的 `_checkerboard`、`(r+c+1)%2` 恢复规则、偶相位 placement filter 都依赖
      棋盘格；融合后都必须删除。
    - 多维叠加后的新冲突点是：若不强制纯底样本出现奇相位对象，则虽然逻辑正确，但视觉上仍
      可能像原题删了棋盘；若不强制部分射线，则又会退化回 D10 单变体。因此二者都要进生成约束。
  d) placement_legality:
    - 仍采用 output-first：先构造完整输出，再把射线后缀擦回输入。
    - 候选 placement 对每条边先算完整两条射线；要求同一对象内部、不同对象之间、不同射线
      之间都不重叠，且所有单元都在边界内。
    - 允许边段因贴边裁成 2 或 3 像素；少于 2 像素则拒绝。这样既保留原题边界行为，又保证
      solver 至少能看到稳定的短线段模板。
  e) visual_observability:
    - 人类能直接看到纯色背景，而不是两色棋盘。
    - 人类也能直接看到某些边段外已经挂着短 stub，输出把 stub 延长到边界；这是 2-4 个样例
      足以归纳的统一规则，不依赖颜色数值、隐藏排序或不可视元信息。
    - 至少一个对象落在奇相位，会让前景明显不再服从原题“只占棋盘一半格子”的视觉节奏。
  f) visual_cleanliness:
    - 保持对象和完整射线互不相交，避免纯底上颜色覆盖造成混乱。
    - 只增加“长度补全”这一类变化，不引入额外标记或多层遮挡，保证菱形、线段、stub 关系清楚。
  g) color_inference:
    - 颜色不编码额外映射；线段和其两条射线始终同色，菱形颜色仅用于区分对象。
    - 因此解题关键完全来自几何与邻接结构，而不是颜色表。
  h) new_solver_assumptions_and_generator_guarantees:
    - solver 假设最常见颜色就是纯背景色；generator 通过大面积背景和计数检查保证。
    - solver 假设真实边段仍完整占据菱形四个固定外侧模板之一，且模板像素单色非背景；
      generator 保证这一点。
    - solver 假设可见射线 stub 只会是从端点向外的连续前缀；generator 只删除远端后缀。
    - solver 假设完整输出中射线不会先撞到无关前景再改变语义；generator 在 output-first
      搜索时提前排除碰撞。
"""

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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


# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
BG_COLOR = 0
MAX_GEN_RETRIES = 600

# ── Geometry constants ──────────────────────────────────────
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


def _diamond_anchor(dpts: list[tuple[int, int]]) -> tuple[int, int]:
    return min(r for r, _ in dpts), min(c for _, c in dpts)


def _visible_segment(anchor: tuple[int, int], shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = anchor
    info = _EDGE_INFO[edge]
    cells = [
        (r0 + dr, c0 + dc)
        for dr, dc in info["segment"]
        if 0 <= r0 + dr < h and 0 <= c0 + dc < w
    ]
    if len(cells) < MIN_SEGMENT_LEN:
        return []
    return _sort_segment(cells, info["slope"])


def _iter_diamonds(g: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    non_bg = set(int(v) for v in np.unique(g)) - {bg}
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]
    return sorted(diamonds, key=lambda item: _diamond_anchor(item[1]))


def _detect_edge_segment(
    g: np.ndarray, bg: int, anchor: tuple[int, int], edge: str
) -> tuple[int, list[tuple[int, int]], tuple[int, int]] | None:
    segment = _visible_segment(anchor, g.shape, edge)
    if not segment:
        return None
    vals = [int(g[r, c]) for r, c in segment]
    if any(v == bg for v in vals):
        return None
    if len(set(vals)) != 1:
        return None
    return vals[0], segment, _EDGE_INFO[edge]["outward"]


def _extend_ray(
    out: np.ndarray,
    bg: int,
    color: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        v = int(out[r, c])
        if v == color:
            r += dr
            c += dc
            continue
        if v == bg:
            out[r, c] = color
            r += dr
            c += dc
            continue
        break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete clipped diagonal rays around hollow diamonds on a solid background.

    Input structure:
    - A single dominant background color, generated as pure 0 rather than a checkerboard.
    - One to three 8-pixel hollow diamonds in 5x5 boxes.
    - On some diamond edges, a short diagonal outer segment is fully visible.
    - Each segment may already have a same-color ray prefix attached to one or both endpoints.

    Rule:
    - For every diamond, inspect the four fixed outer-edge templates to find monochrome non-background segments.
    - From each detected segment endpoint, move in the corresponding outward diagonal direction.
    - Skip across any visible same-color prefix, then fill remaining background cells with that segment color until the boundary or another foreground object.

    Key insight:
    - Partial rays merge with their segment into bent 8-connected shapes, so connected-component shape alone no longer identifies the segment.
    - The reliable cue is the diamond's fixed geometry: locate the segment by template relative to the diamond, then continue through the visible stub on the solid background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for _, dpts in _iter_diamonds(g, bg):
        anchor = _diamond_anchor(dpts)
        for edge in _EDGE_NAMES:
            found = _detect_edge_segment(g, bg, anchor, edge)
            if found is None:
                continue
            color, segment, outward = found
            for endpoint in (segment[0], segment[-1]):
                _extend_ray(out, bg, color, endpoint, outward)

    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = _visible_segment((r0, c0), (h, w), edge)
        if not segment:
            return None
        ray_a = _ray_cells((h, w), segment[0], info["outward"])
        ray_b = _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + ray_a + ray_b
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": (ray_a, ray_b),
                "weight": len(part_cells),
            }
        )
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


def _choose_keep_len(full_len: int) -> int:
    if full_len == 0:
        return 0
    if full_len == 1:
        return random.randint(0, 1)
    roll = random.random()
    if roll < 0.7:
        return random.randint(1, full_len - 1)
    if roll < 0.85:
        return 0
    return full_len


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
    ray_groups: list[list[tuple[int, int]]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            for ray in edge_part["rays"]:
                _paint_cells(out, ray, color)
                ray_groups.append(ray)

    if not _bg_count_safe(out):
        return None

    long_rays = [ray for ray in ray_groups if len(ray) >= 2]
    if len(long_rays) < 2 or max((len(ray) for ray in ray_groups), default=0) < 4:
        return None

    inp = out.copy()
    partial_ray_count = 0
    shown_ray_cells = 0
    hidden_ray_cells = 0
    for ray in ray_groups:
        keep = _choose_keep_len(len(ray))
        if 0 < keep < len(ray):
            partial_ray_count += 1
        shown_ray_cells += keep
        hidden_ray_cells += len(ray) - keep
        for r, c in ray[keep:]:
            inp[r, c] = BG_COLOR

    if partial_ray_count < 2 or shown_ray_cells == 0 or hidden_ray_cells == 0:
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
