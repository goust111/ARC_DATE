# Variant of 80a900e0
# Dimension: D13 — 多步处理顺序 (multi_step)
# Change: 所有射线同时/独立发射（无特定顺序依赖），射线遇非背景停止... → 按菱形位置顺序依次发射

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8

"""
Design analysis for the D13 variant
==================================

spec_delta:
  keep:
    - 背景仍是 0/1 棋盘格，所有非背景像素仍落在 (r+c)%2==0 的相位上
    - 菱形仍是 5x5 bbox 内的 8 像素空心轮廓
    - 外侧短对角线段仍紧贴菱形四边之一，射线仍只从线段两个端点出发
  change:
    - 输出不再等价于“每条射线只看原始输入中的障碍”
    - 先按菱形 bbox 的 (top, left) 排序；前面的菱形先发射，写入到 out 的射线像素会阻挡后面的菱形
    - 生成器必须构造至少一组会真正发生截断的菱形对，否则会退化回原题

a_concretization:
  input:
    - 仍然只显示菱形和外侧线段，不显示射线
    - 至少两个菱形；其中一对被安排在同一条可共享的对角轨道上
  output:
    - 上方菱形先射出的下行射线会占据靠近下方菱形外线段外侧的起始格
    - 下方菱形随后再发射时，某些端点的第一步就撞到已有射线，于是长度变成 0
  traced_example:
    - 原逻辑（独立发射）: 上方 diamond A 的 BR 射线和下方 diamond B 的 TL 射线都沿同一组 r-c 常数对角线延伸
    - 变体逻辑（顺序发射）: A 先画到 B 的 TL 线段外一格，B 再发射时第一步就撞到 A 的像素，因此 B 这两个 TL 端点都被截断
    - 差异: 原逻辑会得到一条更长的连续同色对角线；新逻辑只保留前序菱形贡献的那一段

b_complexity_calibration:
  - 仅改顺序会太弱，因为原 generator 故意避免所有潜在交互；因此这里把 diamond_count 下限改成 2
  - 只强制一组交互，其他边仍可随机，避免样式过死板
  - 交互对的两条相关 outer segment 共享颜色，避免“独立发射”下的重叠颜色冲突，也让人眼更容易看出两段本来属于同一条射线路径

c_implicit_assumptions_review:
  - bg_colors = top-2-most-common: 仍成立，因为输入只含少量非背景像素；生成器保留这一点
  - diamonds = 8-connected same-color components shaped like diamond8: 仍成立；生成器保证菱形颜色各不相同
  - adjacent non-bg around each diamond come only from its own outer segments: 原题依赖“对象间有空隙”，本变体仍需保证；生成器用 halo 分离所有输入对象
  - while int(out[rr,cc]) in bg_colors: 原题中这行因 generator 避冲突而几乎等价于看 g；变体里它是核心，必须保留且显式复用到仿真函数
  - diamond order key must be positional, not color-based: 这里固定使用 bbox(top,left,bottom,right)，禁止任何颜色参与排序

d_placement_legality:
  - 输入对象（diamond + segments）之间必须 8 邻域分离，防止某个 diamond 把别人的线段误识别为自己的邻接 outer component
  - 交互只发生在输出射线层，不发生在输入层
  - 生成器先造输入，再分别跑“顺序阻挡”和“独立阻挡”仿真；若无截断、或独立仿真出现异色覆盖冲突，则整例丢弃

e_visual_observability:
  - 变体依赖可见位置顺序：上面的 diamond 比下面的更早；同一条对角轨道上，前者的结果会吃掉后者的射线起步空间
  - 不使用颜色编号参与任何控制，只使用 diamond 的几何位置排序
  - 每个样例至少出现 2 个被截断的端点，确保 2-3 组样例就能看出“先处理的菱形会阻挡后处理的菱形”

f_visual_cleanliness:
  - 输入对象之间保留 1 格以上 halo，避免 diamond/segment 粘连
  - 交互集中在一对菱形的一条对角通道上，不把全图做成大面积交叉网

g_color_inference:
  - 颜色不编码顺序；共享颜色只用于让交互轨道可读
  - diamond 颜色保持互异，segment 颜色可重复，和原题一致

h_generator_contract:
  - generate_input() 返回输入网格
  - main(input) 必须等于顺序仿真输出
  - 并且顺序输出必须与独立输出不同，且至少 2 个 ray endpoints 被缩短

i_new_main_assumptions:
  - main() 现在显式依赖 diamond 的稳定排序；generate_input() 保证每个 diamond bbox 唯一可比
  - main() 依赖外侧线段仍能从输入中独立识别；generate_input() 禁止输入层的 8 邻接串扰
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


def _segment_descriptor(
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
    block_on_painted: bool,
) -> tuple[np.ndarray, list[int], bool]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]

    out = g.copy()
    lengths: list[int] = []
    conflict = False

    for _, dpts in sorted(diamonds, key=_diamond_sort_key):
        for seg in _segment_descriptor(g, bg_colors, dpts):
            color = int(seg["color"])
            dr, dc = seg["outward"]
            for pr, pc in seg["endpoints"]:
                rr, cc = pr + dr, pc + dc
                ray_len = 0
                while 0 <= rr < h and 0 <= cc < w:
                    obstacle_grid = out if block_on_painted else g
                    if int(obstacle_grid[rr, cc]) not in bg_colors:
                        break
                    if not block_on_painted:
                        painted = int(out[rr, cc])
                        if painted not in bg_colors and painted != color:
                            conflict = True
                    out[rr, cc] = color
                    ray_len += 1
                    rr += dr
                    cc += dc
                lengths.append(ray_len)

    return out, lengths, conflict


def main(grid: np.ndarray) -> np.ndarray:
    """
    D13 variant: process diamonds in positional order and let earlier rays block later ones.

    Detection is unchanged from the original task: find checkerboard bg colors, locate 8-pixel
    diamond outlines, then find adjacent diagonal outer segments. The change is that each ray
    tests occupancy against the progressively updated `out` grid, so already-painted rays are
    genuine obstacles for later diamonds.
    """
    return _simulate_rays(grid, block_on_painted=True)[0]


# --------------- generator for the D13 variant ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (2, 3)  # at least two diamonds are needed for visible order effects
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 3  # keep full 3-pixel segments for legibility
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
BLOCKING_DELTAS = (8, 10, 12)

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


def _build_blocking_pair(h: int, w: int) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    pair_types = list(_PAIR_TYPES)
    random.shuffle(pair_types)

    for upper_edge, lower_edge, mode in pair_types:
        deltas = [d for d in BLOCKING_DELTAS if d < h - 6 and d < w - 6]
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
    if not palette:
        return None

    shared_color = random.choice(palette)
    edge_colors: dict[tuple[int, str], int] = {}
    paired = set(pair_info["pair_edges"])

    for pi, edge_name in paired:
        if shared_color == diamond_colors[pi]:
            return None
        edge_colors[(pi, edge_name)] = shared_color

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

    built = _build_blocking_pair(h, w)
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

    seq_out, seq_lengths, _ = _simulate_rays(inp, block_on_painted=True)
    indep_out, indep_lengths, conflict = _simulate_rays(inp, block_on_painted=False)
    if conflict:
        return None

    blocked_endpoints = sum(int(a < b) for a, b in zip(seq_lengths, indep_lengths))
    zero_blocked = sum(int(a == 0 and b > 0) for a, b in zip(seq_lengths, indep_lengths))
    diff_pixels = int(np.count_nonzero(seq_out != indep_out))
    if blocked_endpoints < 2 or zero_blocked < 2 or diff_pixels < 4:
        return None

    if not np.array_equal(main(inp), seq_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
