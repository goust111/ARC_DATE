# Combo variant of 80a900e0 (index 8)
# combo_spec: D4/0+D10/0
# Name: 纯底虚线射线
"""
design_analysis:
  fused_rule_summary:
    - 保留 5x5 包围盒内的 8 像素空心菱形，以及贴着菱形边外侧的短对角线段。
    - 背景从 0/1 棋盘格改为纯色 0，因此前景不再受固定相位限制。
    - 从每条外线段的两个端点向外发射对角射线，但只给路径上的第 1、3、5... 个背景格上色；
      第 2、4、6... 个格子保持纯背景色，形成黑底虚线。
  concrete_behavior_trace:
    - 原题在 checkerboard 上把完整路径的每个背景格都上色，删射线时恢复到 `(r+c+1)%2`。
    - 本融合变体在 solid background 上仍逐格前进并逐格检查障碍，但采用 paint/skip/paint/skip
      的节奏；删射线时直接恢复背景 0。
    - 例：某端点 outward=(1,1)，完整路径为 [(5,5),(6,6),(7,7),(8,8),(9,9)]。
      原题输出会着色全部 5 格；融合后只着色 [(5,5),(7,7),(9,9)]，其余两格仍为 0。
      若 (7,7) 被别的前景占据，solver 必须在该处停止，不能跳过障碍继续给 (9,9) 上色。
  complexity_calibration:
    - 只改变背景类型和射线路径节奏，不改变菱形尺寸、边段方向库、每菱形 2-4 条边段的数量级。
    - 生成器要求至少一个菱形锚点落在原题不允许的奇相位，避免样本看起来像“只是把 1 去掉”。
    - 生成器要求样例里存在足够多的 skipped ray cells，确保 2-4 个样例中能直接看出虚线规律。
  implicit_assumption_audit:
    - 原 solver 的 `top-2 colors = background` 在纯底下失效，必须改成“最常见单色 = 背景”。
    - 原 `if out[rr,cc] in bg_colors` 也必须收窄为 `== bg`，否则会误把前景色当可覆盖背景。
    - D4 的跳格规则不能只在偶数步看障碍；必须每一步都先检查当前格是不是背景，再决定涂或跳。
    - 生成器若只为“实际着色格”占位，会让被跳过但仍属于路径的格子被别的对象占掉，导致
      solver 提前停止；因此搜索阶段仍要为完整连续路径保留空间。
    - `main()` 通过“菱形周围相邻非背景像素”识别外线段，这隐含要求别的对象不能贴着菱形。
      融合版 generator 显式为每个菱形保留相邻空壳，防止误吸入其他对象。
  placement_legality:
    - output-first：先在纯背景上构造完整输出，再把实际着色的射线格恢复成 0 得到输入。
    - 候选放置时禁止菱形、外线段、完整连续射线路径彼此重叠。
    - 同时禁止任意其他对象落入某个菱形的 8 邻域空壳（外线段自身除外），保证 solver 的局部匹配成立。
    - 外线段贴边时允许被裁成 2 或 3 格；少于 2 格则拒绝。
  visual_observability:
    - 纯黑底让虚线空隙非常显眼，不再被棋盘相位掩盖。
    - 至少一个对象放在奇相位，说明前景不再遵守原题的固定对角相位。
    - 颜色不承担额外映射语义，解题者只需看几何：菱形、外线段、端点发射、隔格上色。
  visual_cleanliness_and_inference:
    - 同色虚线只沿单条对角线路径出现，和实体菱形/实体边段层次清楚。
    - 不允许射线或对象相互碰撞，也不允许贴菱形误邻接，因此画面不会出现难以解释的颜色团块。
    - 颜色种类仍限制在 2-9，变化由位置和连通结构给出，人类可从少量样例稳定归纳。
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设最常见颜色就是唯一背景色；generator 用纯底并检查计数保证这一点。
    - solver 假设每个 diamond component 的相邻非背景像素会分解成若干对角短线段；
      generator 用几何模板和邻域留空保证。
    - solver 假设完整射线路径上不会被别的对象意外截断；generator 用完整路径占位保证。
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
    Extend dashed diagonal rays outward from the short outer segments around hollow diamonds on a solid background.

    Input structure:
    - A single-color background, inferred as the most common color in the grid.
    - One to three 8-pixel hollow diamond outlines in non-background colors.
    - Short diagonal outer segments, also non-background, sitting just outside selected diamond edges.

    Rule:
    - For each outer segment, infer which perpendicular diagonal points inward to the diamond, then reverse it to get the outward ray direction.
    - From each of the segment's two endpoints, walk cell by cell along that outward diagonal until the boundary or the first non-background obstacle.
    - Only paint the 1st, 3rd, 5th, ... reachable background cells on that walk; leave the alternating cells as untouched background.

    Key insight:
    - The geometry that finds segments and ray directions is the same as the original family, but the background is now a single passable color and the ray effect is a dash pattern, not a solid fill.
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
                step_idx = 0
                while 0 <= rr < h and 0 <= cc < w:
                    # Obstacle detection is based on the original input geometry, not
                    # on rays painted earlier in this pass.
                    if int(g[rr, cc]) != bg:
                        break
                    if step_idx % 2 == 0:
                        out[rr, cc] = ocol
                    step_idx += 1
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


def _painted_path_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path[::2]


def _diamond_shell(
    shape: tuple[int, int],
    diamond: list[tuple[int, int]],
    allowed_adjacent: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    shell: set[tuple[int, int]] = set()
    dset = set(diamond)
    for r, c in diamond:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset and (rr, cc) not in allowed_adjacent:
                shell.add((rr, cc))
    return shell


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    segment_cells: set[tuple[int, int]] = set()
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
        segment_cells.update(segment)
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
        "anchor": (r0, c0),
        "skipped_count": total_skipped,
        "forbidden_shell": frozenset(_diamond_shell((h, w), diamond, segment_cells)),
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
    forbidden_counts: Counter[tuple[int, int]] = Counter()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = set(placement["all_cells"])
            shell = set(placement["forbidden_shell"])
            if not occupied.isdisjoint(cells):
                continue
            if any(forbidden_counts[cell] > 0 for cell in cells):
                continue
            if not occupied.isdisjoint(shell):
                continue
            occupied.update(cells)
            for cell in shell:
                forbidden_counts[cell] += 1
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied.difference_update(cells)
            for cell in shell:
                forbidden_counts[cell] -= 1
                if forbidden_counts[cell] == 0:
                    del forbidden_counts[cell]
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
        (
            (edge_part["weight"], random.random(), pi, ei)
            for pi, placement in enumerate(placements)
            for ei, edge_part in enumerate(placement["edges"])
        ),
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


def _variant_visibility_ok(placements: list[dict[str, object]]) -> bool:
    total_skipped = sum(int(placement["skipped_count"]) for placement in placements)
    return total_skipped >= MIN_TOTAL_SKIPPED_CELLS


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
    if placements is None or not _variant_visibility_ok(placements) or not _has_non_original_phase(placements):
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
            _paint_cells(out, edge_part["painted_rays"], color)
            ray_coords.extend(edge_part["painted_rays"])

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
