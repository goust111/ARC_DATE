# Combo variant of 80a900e0 (index 15)
# combo_spec: D13/0+D17/2
# Name: 有序覆盖射线

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8

"""
Design analysis for the combo variant
=====================================

spec_delta:
  keep:
    - 背景仍是 0/1 棋盘格，所有输入对象仍落在同一对角相位
    - 菱形仍是 5x5 bbox 内的 8 像素空心轮廓；外侧线段仍是紧贴某条边的短对角段
    - 射线仍只从每条外侧线段的两个端点发出，并沿垂直于线段的对角方向前进
  change:
    - 不再把所有射线看成彼此独立；先按菱形位置排序，再逐个发射
    - 顺序固定为 bbox 的 (top, left)；更靠下或同排更靠右的菱形后处理
    - 在射线层上，后处理的射线不会被先前射线阻断，而是直接覆盖它们；真正的障碍只剩原始输入里的非背景对象

a_concretization:
  input:
    - 至少两个菱形；强制安排一对菱形共用同一组对角轨道
    - 上方菱形用 BR 或 BL 外线段，下方菱形分别用 TL 或 TR 外线段
  traced_example:
    - 原题/先到先占基线: 上方菱形先把对角轨道涂满，下方菱形的对应射线一碰到这条已着色轨道就停止
    - 融合后: 仍然是上方菱形先发射，但下方菱形后发射时会沿同一轨道继续走，并把重叠段改成自己的颜色
    - 差异: 最终一条长对角线上会出现“上段是后者颜色、下段保留前者颜色”的有序覆盖痕迹

b_complexity_calibration:
  - 若不强制至少一组轨道重合，样例会退化回原题，因此 diamond_count 下限提升到 2
  - 只强制一组覆盖对，其余边仍随机，既能看出规律，也不会把整张图做成大面积网状交叉
  - 覆盖对使用不同的线段颜色，否则即使发生覆盖，肉眼也看不出优先级

c_implicit_assumptions_review:
  - bg_colors = top-2-most-common: 仍成立，因为输入只含少量非背景对象，射线不在输入里
  - diamonds = same-color 8 连通 diamond8: 仍成立；生成器保证每个菱形颜色互异
  - adjacent non-bg around a diamond are its own outer segments: 仍需成立；生成器继续用 halo 隔开输入对象
  - 原代码里 while int(out[rr,cc]) in bg_colors 隐含“已有射线是障碍”；融合后这条假设失效
  - 新逻辑必须把“原始输入对象”和“运行时已画出的射线”区分开：停止条件看 g，覆盖写入看 out

d_placement_legality:
  - 所有输入对象在输入层都保持 8 邻域分离，避免把别的菱形线段误识别成当前菱形的邻接外段
  - 强制覆盖只发生在输出射线层；输入里不允许对象重叠或越界
  - 生成器用显式几何构造共轨菱形对，再用仿真确认确实产生覆盖差异

e_visual_observability:
  - 排序完全依赖可见位置，从上到下、再从左到右；不使用颜色编号等人类不可见因素
  - 覆盖对故意做成“上方先画、下方后改色”，少量样例就能看出顺序优先级
  - 至少两条端点射线会从“被先到先占挡住”变成“继续前进并改色”，规则可归纳

f_visual_cleanliness:
  - 输入对象之间保留 halo，避免菱形与别人的线段粘连
  - 只强制单通道覆盖，减少同一区域多层交叉
  - 额外菱形仍随机，但必须与已有输入对象分离

g_color_inference:
  - 颜色不编码顺序，顺序只由位置给出
  - 覆盖对的两条线段必须异色，保证后到覆盖能被直接看见
  - 其余线段允许复用颜色，保持与原题相近的颜色自由度

h_generator_contract:
  - generate_input() 返回只含菱形与外段的输入网格
  - main(input) 必须等于覆盖仿真输出
  - 同时要求覆盖输出与“先到先占”基线输出不同，且至少 2 条射线真的发生延伸并覆盖旧色

i_new_main_assumptions:
  - main() 显式依赖 diamond 的稳定位置排序；生成器保证所有 diamond bbox 可稳定比较
  - main() 假设重叠只发生在射线层，不发生在输入层；生成器用 halo 与验证仿真保证这点
"""


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


def _diamond_sort_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
    _, pts = item
    return _bbox(pts)


def _segment_descriptors(
    g: np.ndarray,
    bg_colors: set[int],
    dpts: list[tuple[int, int]],
) -> list[dict[str, object]]:
    h, w = g.shape
    dset = set(dpts)
    adj: list[tuple[int, int]] = []
    for r, c in dpts:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                if int(g[rr, cc]) not in bg_colors:
                    adj.append((rr, cc))
    if not adj:
        return []

    adj_mask = np.zeros((h, w), dtype=bool)
    for r, c in adj:
        adj_mask[r, c] = True

    descs: list[dict[str, object]] = []
    for oc in _cc8(adj_mask):
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

        descs.append(
            {
                "color": int(g[oc[0][0], oc[0][1]]),
                "cells": tuple(sorted(oc)),
                "endpoints": (endpoints[0], endpoints[-1]),
                "outward": (-best_n[0], -best_n[1]),
            }
        )

    descs.sort(key=lambda d: _bbox(list(d["cells"])))
    return descs


def _simulate_rays(
    grid: np.ndarray,
    *,
    overwrite_rays: bool,
) -> tuple[np.ndarray, list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]

    out = g.copy()
    lengths: list[int] = []
    overwrite_cells = 0
    rays_with_overwrite = 0

    for _, dpts in sorted(diamonds, key=_diamond_sort_key):
        for seg in _segment_descriptors(g, bg_colors, dpts):
            color = int(seg["color"])
            dr, dc = seg["outward"]
            for pr, pc in seg["endpoints"]:
                rr, cc = pr + dr, pc + dc
                ray_len = 0
                touched_prior_ray = False
                while 0 <= rr < h and 0 <= cc < w:
                    if int(g[rr, cc]) not in bg_colors:
                        break
                    prev = int(out[rr, cc])
                    if not overwrite_rays and prev not in bg_colors:
                        break
                    if overwrite_rays and prev not in bg_colors and prev != color:
                        overwrite_cells += 1
                        touched_prior_ray = True
                    out[rr, cc] = color
                    ray_len += 1
                    rr += dr
                    cc += dc
                lengths.append(ray_len)
                rays_with_overwrite += int(touched_prior_ray)

    return out, lengths, overwrite_cells, rays_with_overwrite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit ordered diagonal rays from the outer segments around checkerboard diamonds, with later rays repainting earlier ones.

    Input structure:
    - The background is a 0/1 checkerboard, and all non-background pixels lie on the checkerboard phase.
    - Each object is a hollow 8-pixel diamond outline plus 2-4 adjacent short diagonal outer segments in non-background colors.
    - No rays are drawn in the input; only the diamonds and their nearby outer segments are visible.

    Rule:
    - Detect all diamonds, sort them by position from top to bottom and then left to right, and process them in that order.
    - For every outer segment, infer its outward diagonal direction and launch one ray from each endpoint.
    - A ray stops only when it reaches an original non-background input cell or the grid boundary.
    - If a later ray passes through cells already painted by earlier rays, it overwrites those cells with its own color.

    Key insight:
    Rays are still anchored only at the two endpoints of each short outer segment, but the crucial change is the interaction layer: position order decides paint priority, and the original input objects are the only true obstacles.
    """
    return _simulate_rays(grid, overwrite_rays=True)[0]


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (2, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 3
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500
OVERLAP_DELTAS = (8, 10, 12)

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)
_PAIR_TYPES = (
    ("BR", "TL", "same_rc_diff"),
    ("BL", "TR", "same_rc_sum"),
)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _expand8(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _candidate_input_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    edges: tuple[str, ...],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    if any(not (0 <= r < h and 0 <= c < w) for r, c in diamond):
        return None

    all_input = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = []
        for dr, dc in info["segment"]:
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                segment.append((rr, cc))
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        if any(cell in all_input for cell in segment):
            return None
        all_input.update(segment)
        edge_parts.append(
            {
                "name": edge,
                "segment": tuple(segment),
                "slope": info["slope"],
                "outward": info["outward"],
            }
        )

    return {
        "top_left": (r0, c0),
        "diamond": tuple(diamond),
        "edges": edge_parts,
        "input_cells": frozenset(all_input),
        "halo": frozenset(_expand8((h, w), all_input)),
    }


def _placements_compatible(placements: list[dict[str, object]]) -> bool:
    for i, left in enumerate(placements):
        for right in placements[i + 1 :]:
            if not left["halo"].isdisjoint(right["input_cells"]):
                return False
            if not right["halo"].isdisjoint(left["input_cells"]):
                return False
    return True


def _random_edges_with_required(required: str) -> tuple[str, ...]:
    others = [e for e in _EDGE_NAMES if e != required]
    count = random.randint(*EDGES_PER_DIAMOND)
    chosen = [required] + random.sample(others, count - 1)
    return tuple(sorted(chosen))


def _build_overlap_pair(h: int, w: int) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    pair_types = list(_PAIR_TYPES)
    random.shuffle(pair_types)

    for upper_edge, lower_edge, mode in pair_types:
        deltas = [d for d in OVERLAP_DELTAS if d < h - 6 and d < w - 6]
        random.shuffle(deltas)
        for delta in deltas:
            positions: list[tuple[int, int]] = []
            if mode == "same_rc_diff":
                for r0 in range(1, h - 4 - delta):
                    for c0 in range(1, w - 4 - delta):
                        if (r0 + c0) % 2 == 0:
                            positions.append((r0, c0))
            else:
                for r0 in range(1, h - 4 - delta):
                    for c0 in range(delta, w - 4):
                        if (r0 + c0) % 2 == 0:
                            positions.append((r0, c0))
            random.shuffle(positions)

            for r0, c0 in positions:
                if mode == "same_rc_diff":
                    top_left_b = (r0 + delta, c0 + delta)
                else:
                    top_left_b = (r0 + delta, c0 - delta)

                for _ in range(24):
                    edges_a = _random_edges_with_required(upper_edge)
                    edges_b = _random_edges_with_required(lower_edge)
                    pa = _candidate_input_placement(h, w, r0, c0, edges_a)
                    pb = _candidate_input_placement(h, w, top_left_b[0], top_left_b[1], edges_b)
                    if pa is None or pb is None:
                        continue
                    if _placements_compatible([pa, pb]):
                        return [pa, pb], {"pair_edges": ((0, upper_edge), (1, lower_edge))}
    return None


def _search_extra_placement(
    h: int,
    w: int,
    placed: list[dict[str, object]],
) -> dict[str, object] | None:
    positions = [(r, c) for r in range(1, h - 4) for c in range(1, w - 4) if (r + c) % 2 == 0]
    random.shuffle(positions)
    for r0, c0 in positions:
        for _ in range(12):
            edges = tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
            pl = _candidate_input_placement(h, w, r0, c0, edges)
            if pl is not None and _placements_compatible(placed + [pl]):
                return pl
    return None


def _assign_colors(
    placements: list[dict[str, object]],
    pair_info: dict[str, object],
) -> tuple[list[int], dict[tuple[int, str], int]] | None:
    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    palette = [c for c in COLOR_RANGE if c not in diamond_colors]
    if len(palette) < 2:
        return None

    early_color, late_color = random.sample(palette, 2)
    edge_colors: dict[tuple[int, str], int] = {
        pair_info["pair_edges"][0]: early_color,
        pair_info["pair_edges"][1]: late_color,
    }

    for pi, placement in enumerate(placements):
        allowed = [c for c in COLOR_RANGE if c != diamond_colors[pi]]
        for edge in placement["edges"]:
            key = (pi, edge["name"])
            if key not in edge_colors:
                edge_colors[key] = random.choice(allowed)

    return diamond_colors, edge_colors


def _render_input(
    h: int,
    w: int,
    placements: list[dict[str, object]],
    diamond_colors: list[int],
    edge_colors: dict[tuple[int, str], int],
) -> np.ndarray:
    grid = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        _paint_cells(grid, placement["diamond"], diamond_colors[pi])
        for edge in placement["edges"]:
            _paint_cells(grid, edge["segment"], edge_colors[(pi, edge["name"])])
    return grid


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    built = _build_overlap_pair(h, w)
    if built is None:
        return None
    placements, pair_info = built

    target_count = random.randint(*DIAMOND_COUNT_RANGE)
    while len(placements) < target_count:
        extra = _search_extra_placement(h, w, placements)
        if extra is None:
            return None
        placements.append(extra)

    color_assignment = _assign_colors(placements, pair_info)
    if color_assignment is None:
        return None
    diamond_colors, edge_colors = color_assignment

    inp = _render_input(h, w, placements, diamond_colors, edge_colors)
    if set(_top_k_colors(inp, 2)) != {0, 1}:
        return None
    if not _bg_counts_safe(inp):
        return None

    combo_out, combo_lengths, overwrite_cells, overwrite_rays = _simulate_rays(inp, overwrite_rays=True)
    block_out, block_lengths, _, _ = _simulate_rays(inp, overwrite_rays=False)

    extended_endpoints = sum(int(a > b) for a, b in zip(combo_lengths, block_lengths))
    diff_pixels = int(np.count_nonzero(combo_out != block_out))
    if overwrite_cells < 4 or overwrite_rays < 2 or extended_endpoints < 2 or diff_pixels < 4:
        return None

    if not np.array_equal(main(inp), combo_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
