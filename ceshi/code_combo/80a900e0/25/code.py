# Combo variant of 80a900e0 (index 25)
# combo_spec: D9/1+D13/0+D17/2
# Name: 密集有序覆盖

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8

"""
Design analysis for the combo variant
=====================================

spec_delta:
  keep:
    - 背景仍是 0/1 棋盘格，左上角固定为 1，所有非背景像素仍落在 (r+c)%2==0 的相位上
    - 菱形仍是 5x5 bbox 内的 8 像素空心轮廓；外侧短线段仍贴着四条斜边之一
    - 射线仍只从每条外侧线段的两个端点发出，方向仍是“垂直于线段并远离菱形”的对角方向
  change:
    - 菱形数量从 1-3 提高到 4-6；generator 需要构造更大的画布和更高密度的潜在重叠
    - 所有菱形按 bbox(top,left) 排序依次处理，顺序是可见几何顺序，不允许依赖颜色编号
    - 射线碰到“原始输入里的非背景物体”才停止；碰到更早画出的射线不会停止，而是继续前进并覆盖其颜色

a_concretization:
  traced_example:
    - 设有一条 NW->SE 的链：A 在最上左，只带 BR；B 在中间，同时带 TL 和 BR；C 在更下右，只带 TL
    - 原题/阻挡逻辑里：A 的 BR 会先占住通道，B 的 TL 在第一批已画像素处就停止，C 也只会被更早像素截断
    - 本融合里：A 先向 SE 画完整条通道；随后 B 的 TL 从反方向沿同一组对角线往 NW 走，不会被 A 挡住，而是把共享区段改成 B 的颜色；再之后 C 会把它与 B 共用的那部分再改成 C 的颜色
    - 差异不再是“后者变短”，而是“共享走廊的最终颜色由后处理的菱形决定”，因此最终输出强依赖顺序

b_complexity_calibration:
  - 如果只随机撒 4-6 个菱形，很多样例会退化成互不干扰；这里固定生成两条对角链（r-c 常数链和 r+c 常数链），让重叠反复出现
  - 每条链长度控制在 2-4，所有菱形都属于某条链；这样既有高密度覆盖，又不会引入过多互相独立的自由度
  - 每个菱形仍保留 2-4 条外侧线段，但链上负责共享走廊的必选边会被强制保留，其余边再随机补充

c_implicit_assumptions_review:
  - bg_colors = top-2-most-common: 仍只在输入上使用；generator 保证输入里 0/1 仍是最常见的两色
  - diamonds = same-color 8-connected diamond8 components: 仍成立；generator 保证 4-6 个菱形颜色互不相同
  - “diamond 周围的相邻非背景像素只来自它自己的外侧线段”: 原题的隐式前提在高密度下更脆弱；generator 用 halo 分离所有输入对象
  - 原 solver 的 while int(out[rr,cc]) in bg_colors 隐含“旧射线会阻挡新射线”；融合后这条假设失效，停止判断必须改成看 g，不看 out
  - 顺序相关逻辑必须只依赖 bbox(top,left,bottom,right)；若排序依赖颜色，则人类无法从样例中观察到

d_placement_legality:
  - 所有菱形 top-left 都放在 (r+c)%2==0 的相位上，保证形状与棋盘相位一致
  - 只允许完整 3 像素外侧线段；generator 不使用边界裁剪后的 2 像素线段，避免高密度时识别变脆
  - 输入层对象之间必须 8 邻域分离，防止某个 diamond 把别人的线段当成自己的邻接外线段
  - 输出层允许射线重叠；这正是 D17/2 的核心行为

e_visual_observability:
  - 顺序来自“从上到下、从左到右”的几何位置，输入中直接可见
  - 链式布局会让同一条斜向走廊被多个菱形重复着色；观察 2-3 个样例就能看出“越后面的菱形越压过前面的颜色”
  - generator 强制相邻链节点在共享走廊上的线段颜色不同，否则覆盖发生了也肉眼不可见

f_visual_cleanliness:
  - 只使用两种走廊族：同 r-c 链与同 r+c 链；不把所有菱形混成满图随机交叉
  - 额外边数受限在 2-4，总画布放大到能容纳 4-6 个对象并保留 halo
  - 验证阶段要求顺序正反转输出不同，同时与“旧阻挡逻辑”输出不同，筛掉看起来像原题的样例

g_color_inference:
  - 颜色不编码顺序，顺序只由位置给出
  - 菱形颜色两两不同，便于对象分离；共享走廊用重复但交替的线段颜色，让覆盖关系清晰可见

h_generator_contract:
  - generate_input() 返回输入网格；main(input) 必须等于融合规则的顺序覆盖输出
  - 同时要求样例中存在跨菱形覆盖、顺序翻转会改输出、且融合输出不同于旧阻挡输出

i_new_main_assumptions:
  - main() 现在假设输入层对象已被分离到足以可靠找出“某菱形自己的邻接外线段”
  - main() 允许 out 上出现非背景射线再被覆盖；generator 必须保证这种覆盖只发生在输出层，不破坏输入识别
"""


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
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
    obstacle_mode: str,
    reverse_diamond_order: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]
    forward = sorted(diamonds, key=_diamond_sort_key)
    ordered = list(reversed(forward)) if reverse_diamond_order else forward
    diamond_ids = {tuple(sorted(pts)): idx for idx, (_, pts) in enumerate(forward)}

    out = g.copy()
    owner_by_cell: dict[tuple[int, int], int] = {}
    overwritten_cells: set[tuple[int, int]] = set()
    overwritten_events = 0
    ray_pixels = 0

    for _, dpts in ordered:
        diamond_id = diamond_ids[tuple(sorted(dpts))]
        for seg in _segment_descriptors(g, bg_colors, dpts):
            color = int(seg["color"])
            dr, dc = seg["outward"]
            for pr, pc in seg["endpoints"]:
                rr, cc = pr + dr, pc + dc
                while 0 <= rr < h and 0 <= cc < w:
                    obstacle_grid = g if obstacle_mode == "input" else out
                    if int(obstacle_grid[rr, cc]) not in bg_colors:
                        break
                    prev_owner = owner_by_cell.get((rr, cc))
                    if prev_owner is not None and prev_owner != diamond_id:
                        overwritten_cells.add((rr, cc))
                        overwritten_events += 1
                    out[rr, cc] = color
                    owner_by_cell[(rr, cc)] = diamond_id
                    ray_pixels += 1
                    rr += dr
                    cc += dc

    return out, {
        "diamond_count": len(forward),
        "cross_overwrite_cells": len(overwritten_cells),
        "cross_overwrite_events": overwritten_events,
        "ray_pixels_written": ray_pixels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill outward diagonal rays from the short outer segments around multiple checkerboard diamonds.

    Input structure:
    - A 0/1 checkerboard background, with all non-background pixels staying on the checkerboard phase.
    - Four to six hollow 8-pixel diamonds, each in its own non-background color.
    - Around some diamond edges, 3-pixel diagonal outer segments in non-background colors adjacent to the diamond.

    Rule:
    - Sort diamonds by position from top to bottom, then left to right.
    - For each outer segment of each diamond, identify the outward diagonal direction and launch rays from the two segment endpoints.
    - A ray stops only when it reaches a non-background pixel from the original input or the grid boundary.
    - If a later diamond's ray passes through pixels already painted by earlier rays, it keeps going and overwrites those earlier ray colors.

    Key insight:
    - The decisive change is that processing order matters because rays ignore previous ray paint as obstacles.
      Earlier rays can be recolored by later diamonds, so the final shared corridors are owned by the last diamond in positional order.
    """
    return _simulate_rays(grid, obstacle_mode="input")[0]


# --------------- generator for the combo variant ---------------

GRID_H_RANGE = (26, 32)
GRID_W_RANGE = (26, 34)
DIAMOND_COUNT_RANGE = (4, 6)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 3
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 700
CHAIN_DELTA_CHOICES = (8, 10)
MIN_OVERWRITE_CELLS = 8
MIN_ORDER_DIFF_PIXELS = 8
MIN_BLOCKED_DIFF_PIXELS = 8

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
_CHAIN_MODES = ("diff", "sum")


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], color: int) -> None:
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


def _count_split(total: int) -> list[tuple[int, int]]:
    if total == 4:
        return [(2, 2)]
    if total == 5:
        return [(3, 2), (2, 3)]
    return [(3, 3), (4, 2), (2, 4)]


def _required_edges_for_chain(mode: str, idx: int, size: int) -> set[str]:
    req: set[str] = set()
    if mode == "diff":
        if idx > 0:
            req.add("TL")
        if idx < size - 1:
            req.add("BR")
    else:
        if idx > 0:
            req.add("TR")
        if idx < size - 1:
            req.add("BL")
    return req


def _random_edges(required: set[str]) -> tuple[str, ...]:
    count = random.randint(max(len(required), EDGES_PER_DIAMOND[0]), EDGES_PER_DIAMOND[1])
    extras = [e for e in _EDGE_NAMES if e not in required]
    chosen = set(required)
    if count > len(chosen):
        chosen.update(random.sample(extras, count - len(chosen)))
    return tuple(sorted(chosen))


def _chain_starts(h: int, w: int, mode: str, size: int, delta: int) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    max_r0 = h - 6 - (size - 1) * delta
    if max_r0 < 1:
        return starts
    if mode == "diff":
        max_c0 = w - 6 - (size - 1) * delta
        if max_c0 < 1:
            return starts
        for r0 in range(1, max_r0 + 1):
            for c0 in range(1, max_c0 + 1):
                if (r0 + c0) % 2 == 0:
                    starts.append((r0, c0))
    else:
        min_c0 = 1 + (size - 1) * delta
        max_c0 = w - 6
        if min_c0 > max_c0:
            return starts
        for r0 in range(1, max_r0 + 1):
            for c0 in range(min_c0, max_c0 + 1):
                if (r0 + c0) % 2 == 0:
                    starts.append((r0, c0))
    return starts


def _chain_positions(start: tuple[int, int], mode: str, size: int, delta: int) -> list[tuple[int, int]]:
    r0, c0 = start
    if mode == "diff":
        return [(r0 + i * delta, c0 + i * delta) for i in range(size)]
    return [(r0 + i * delta, c0 - i * delta) for i in range(size)]


def _build_chain(
    h: int,
    w: int,
    mode: str,
    size: int,
    existing: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[set[str]]] | None:
    deltas = list(CHAIN_DELTA_CHOICES)
    random.shuffle(deltas)
    for delta in deltas:
        starts = _chain_starts(h, w, mode, size, delta)
        random.shuffle(starts)
        for start in starts:
            positions = _chain_positions(start, mode, size, delta)
            for _ in range(48):
                chain: list[dict[str, object]] = []
                reqs: list[set[str]] = []
                for idx, (r0, c0) in enumerate(positions):
                    required = _required_edges_for_chain(mode, idx, size)
                    edges = _random_edges(required)
                    pl = _candidate_input_placement(h, w, r0, c0, edges)
                    if pl is None:
                        chain = []
                        break
                    chain.append(pl)
                    reqs.append(required)
                if chain and _placements_compatible(existing + chain):
                    return chain, reqs
    return None


def _build_layout(h: int, w: int, target_count: int) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    split_options = _count_split(target_count)
    random.shuffle(split_options)
    for sizes in split_options:
        mode_orders = [list(_CHAIN_MODES), list(reversed(_CHAIN_MODES))]
        random.shuffle(mode_orders)
        for modes in mode_orders:
            placements: list[dict[str, object]] = []
            chain_infos: list[dict[str, object]] = []
            ok = True
            for size, mode in zip(sizes, modes):
                built = _build_chain(h, w, mode, size, placements)
                if built is None:
                    ok = False
                    break
                chain, reqs = built
                start_index = len(placements)
                placements.extend(chain)
                chain_infos.append(
                    {
                        "mode": mode,
                        "indices": list(range(start_index, start_index + size)),
                        "required": reqs,
                    }
                )
            if ok and len(placements) == target_count:
                return placements, chain_infos
    return None


def _choose_edge_color(
    diamond_color: int,
    forbidden: set[int],
    preferred: list[int],
) -> int:
    for c in preferred:
        if c != diamond_color and c not in forbidden:
            return c
    for c in COLOR_RANGE:
        if c != diamond_color and c not in forbidden:
            return c
    raise RuntimeError("no valid edge color")


def _assign_colors(
    placements: list[dict[str, object]],
    chain_infos: list[dict[str, object]],
) -> tuple[list[int], dict[tuple[int, str], int]]:
    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    edge_colors: dict[tuple[int, str], int] = {}

    for chain in chain_infos:
        palette = random.sample(list(COLOR_RANGE), 3)
        previous_color = None
        for local_idx, pi in enumerate(chain["indices"]):
            preferred = [palette[local_idx % 2], palette[2], palette[(local_idx + 1) % 2]]
            chain_color = _choose_edge_color(diamond_colors[pi], set(), preferred)
            if previous_color is not None and chain_color == previous_color:
                chain_color = _choose_edge_color(diamond_colors[pi], {previous_color}, list(COLOR_RANGE))
            previous_color = chain_color
            for edge_name in chain["required"][local_idx]:
                edge_colors[(pi, edge_name)] = chain_color

    accent_palette = random.sample(list(COLOR_RANGE), 3)
    for pi, placement in enumerate(placements):
        used_here = {edge_colors[(pi, edge["name"])] for edge in placement["edges"] if (pi, edge["name"]) in edge_colors}
        for edge in placement["edges"]:
            key = (pi, edge["name"])
            if key in edge_colors:
                continue
            preferred = accent_palette + list(used_here)
            edge_colors[key] = _choose_edge_color(diamond_colors[pi], set(), preferred)
            used_here.add(edge_colors[key])

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
    return counts[0] > max_non_bg and counts[1] > max_non_bg and set(_top_k_colors(grid, 2)) == {0, 1}


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    target_count = random.randint(*DIAMOND_COUNT_RANGE)

    built = _build_layout(h, w, target_count)
    if built is None:
        return None
    placements, chain_infos = built

    diamond_colors, edge_colors = _assign_colors(placements, chain_infos)
    inp = _render_input(h, w, placements, diamond_colors, edge_colors)
    if not _bg_counts_safe(inp):
        return None

    combo_out, combo_stats = _simulate_rays(inp, obstacle_mode="input")
    reverse_out, _ = _simulate_rays(inp, obstacle_mode="input", reverse_diamond_order=True)
    blocked_out, _ = _simulate_rays(inp, obstacle_mode="painted")

    overwrite_cells = combo_stats["cross_overwrite_cells"]
    order_diff = int(np.count_nonzero(combo_out != reverse_out))
    blocked_diff = int(np.count_nonzero(combo_out != blocked_out))

    if overwrite_cells < MIN_OVERWRITE_CELLS:
        return None
    if order_diff < MIN_ORDER_DIFF_PIXELS:
        return None
    if blocked_diff < MIN_BLOCKED_DIFF_PIXELS:
        return None
    if combo_stats["diamond_count"] < DIAMOND_COUNT_RANGE[0]:
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
