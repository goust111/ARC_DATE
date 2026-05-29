# Variant of e87109e9
# Dimension: D2 — 转弯规则 (path)
# Change: 遇到障碍物时根据图例查找该颜色——图例中颜色条在左侧表示逆时针转弯，在右侧表示顺时针转弯... → 障碍物形状编码转弯方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 删除原题顶部图例，输入只保留主网格。
#     - 光束不再按颜色查表转向，而是直接读取被撞障碍物的几何属性。
#     - 规则压缩成两个可见档位：
#       * 宽度 > 高度 的矩形障碍物 -> 顺时针转弯
#       * 高度 > 宽度 的矩形障碍物 -> 逆时针转弯
#   concrete_objects:
#     - 仍保留一个实心青色(8)方块作为光束源。
#     - 仍保留 3-4 个实心矩形障碍物，均为单色、互不重叠、与光源至少隔 1 格。
#     - 为了让形状编码一眼可见，障碍物禁止正方形；生成器只放明显横向或纵向的长条。
#   behavior_trace:
#     old_logic:
#       - 光束下一步要撞到某色障碍物时，先去顶部图例找同色竖条。
#       - 竖条在图例格左侧则左转，在右侧则右转。
#     new_logic:
#       - 光束下一步要撞到某色障碍物时，直接定位该障碍物的连通块边界框。
#       - 若 bbox 更宽则右转，若 bbox 更高则左转。
#     difference:
#       - 决策来源从“颜色 -> 图例位置”变成“当前障碍物自身的外形”。
#       - 输出不再裁掉图例，因为输入里根本没有图例区域。
#   complexity_calibration:
#     - 只用两类形状（横条 / 竖条）编码方向，避免自由度过多。
#     - 仍保留原题的多障碍、多次碰撞路径，让 D2 的变化落在原来的关键步骤上，而不是把题目简化成直线涂色。
#     - 每个样例都强制至少存在一次二次碰撞；否则“转向”虽然发生，但形状控制在视觉上不够突出。
#   implicit_assumption_review:
#     - 原 main() 假设颜色 5 只出现在顶部图例，因此会先裁切图例；变体中该前提失效，必须删掉整段裁切逻辑。
#     - 原 main() 只需要知道障碍物颜色，不需要知道障碍物几何；变体中必须在模拟前预先解析每个障碍物的 bbox。
#     - 原 generator 可以让同一转向方向的障碍物颜色完全无结构；变体中颜色不再承载语义，因此必须靠形状本身可读。
#   placement_legality:
#     - 继续使用 edge-touching 长条矩形：上/下边放横条，左/右边放竖条。
#     - 这样几何类别与 path-first 约束天然兼容，且 bbox 方向稳定，不会出现宽高相等的模糊障碍物。
#     - 生成器用 required/excluded 区间控制哪些障碍物会成为第 2 次碰撞，避免第 3 次碰撞和循环。
#   visual_observability:
#     - 任一 3/4 边障碍物布局都同时包含横条和竖条，因此两个转向分支都会出现在输入中。
#     - 二次碰撞链会让“横条先右转、竖条再左转”或其对偶情况直接体现在输出折线路径里。
#   neatness:
#     - 所有障碍物仍是纯色整块，不叠加额外标记。
#     - 去掉顶部图例后，画面更干净，注意力集中在障碍物方向与光束折线之间的关系上。
#   main_assumptions:
#     - 背景色是全图最多的颜色。
#     - 全图恰有一个实心青色正方形光源。
#     - 每个障碍物颜色只对应一个 4 连通实心非正方形矩形。
#     - 任意一步的光束脚印至多覆盖一种障碍物颜色。
#   generator_guarantees:
#     - 障碍物颜色互异。
#     - 横条一定满足 width > height，竖条一定满足 height > width。
#     - 每题总碰撞次数固定为 4 或 5，且至少有 1 次二次碰撞。
#     - 用新 main() 对生成输入做 oracle 校验，确保 solver 与 generator 一致。


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0,c0,size) for a solid square teal component."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size)))
    assert len(candidates) == 1, "grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _turn_from_dims(hh: int, ww: int) -> str:
    assert hh != ww, "obstacle rectangles must not be square in this variant"
    return "R" if ww > hh else "L"


def _infer_obstacle_turns(g: np.ndarray, bg: int, teal: int = 8) -> dict[int, str]:
    """Infer each obstacle color's turn from its component shape."""
    turns: dict[int, str] = {}
    obstacle_colors = sorted(set(map(int, np.unique(g))) - {bg, teal})
    assert obstacle_colors, "at least one obstacle must be present"

    for color in obstacle_colors:
        comps = list(components_mask(g == color, connectivity=4))
        assert len(comps) == 1, "each obstacle color must form exactly one component"
        cells = comps[0]
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        assert int(cells.shape[0]) == hh * ww, "obstacles must be solid rectangles"
        turns[color] = _turn_from_dims(hh, ww)
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, turning by obstacle shape instead of a legend.

    Input structure:
    - Single main grid only: one background color, one solid teal (8) square, and 3-4 colored rectangular obstacles.
    - Obstacles are solid non-square rectangles; some are wider than tall and some are taller than wide.

    Rule:
    - Shoot a square-width beam from the teal square in each cardinal direction.
    - The moving beam paints teal over background cells only.
    - When the next beam footprint meets an obstacle, inspect that obstacle's rectangle:
      wider-than-tall means turn clockwise, taller-than-wide means turn counterclockwise.
    - Continue after the turn and stop when the next step would leave the grid.

    Key insight: The obstacle geometry itself replaces the removed legend, so turn direction is read from
    aspect ratio rather than from a separate color lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.size, "input grid must be non-empty"

    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(g, teal=teal)
    obstacle_turn = _infer_obstacle_turns(g, bg=bg, teal=teal)

    out = g.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in obstacle_turn, "beam hit an unrecognized obstacle"
            d = _rot(d, obstacle_turn[oc])

    return out


# --------------- generator ---------------

# Color palettes
OBSTACLE_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 300


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a (start, end) span covering all required intervals, avoiding all excluded."""
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(0, req1 - length + 1)
        start1 = min(req0, axis_len - length)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate beams on a main grid, returning output and per-beam hits."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(out[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)
            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint(r, c)
                continue
            assert obs in turns
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2
        hits.append(beam_hits)

    return out, hits


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
    """Compute obstacle thicknesses for the two ends of an axis (N/S or W/E)."""
    lo, hi_cap = OBS_THICKNESS_RANGE
    if a_present and b_present:
        room = axis - sz - OBS_BOTH_GAP
        if room < 2 * lo:
            return None
        a = random.randint(lo, min(hi_cap, room - lo))
        b = _rand_thickness(lo, min(hi_cap, room - a))
        return (a, b) if b is not None else None
    if a_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (t, None) if t is not None else None
    if b_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (None, t) if t is not None else None
    return None, None


def _obstacle_rect(
    side: int, thickness: int, span: tuple[int, int], h: int, w: int
) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) for an edge-touching obstacle rectangle."""
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _shape_turn_for_side(side: int) -> str:
    """Side bars on top/bottom are horizontal; left/right bars are vertical."""
    return "R" if side in (_N, _S) else "L"


def _select_chain(present: set[int]) -> dict[int, int] | None:
    """Select exactly one side whose beam will hit a second obstacle."""
    candidates = [
        (side, _rot(side, _shape_turn_for_side(side)))
        for side in present
        if _rot(side, _shape_turn_for_side(side)) in present
    ]
    if not candidates:
        return None
    side, nxt = random.choice(candidates)
    return {side: nxt}


def _compute_span_constraints(
    present: set[int],
    side_turn: dict[int, str],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals so only the selected chain gets a second hit."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(s: int) -> tuple[int, int]:
        return lane[s], lane[s] + sz - 1

    for side in present:
        nxt = _rot(side, side_turn[side])
        if nxt not in present:
            continue
        if chains.get(side) == nxt:
            required[nxt].append(lane_interval(side))
            after = _rot(nxt, side_turn[nxt])
            if after in present:
                excluded[after].append(lane_interval(nxt))
        else:
            excluded[nxt].append(lane_interval(side))

    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))

    ns = _thickness_pair(h, sz, _N in present, _S in present)
    ew = _thickness_pair(w, sz, _W in present, _E in present)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    if _N in present:
        min_r = max(min_r, thickness[_N] + 1)
    if _S in present:
        max_r = min(max_r, h - thickness[_S] - sz - 1)
    if _W in present:
        min_c = max(min_c, thickness[_W] + 1)
    if _E in present:
        max_c = min(max_c, w - thickness[_E] - sz - 1)
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    lane: dict[int, int] = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    chains = _select_chain(present)
    if chains is None:
        return None

    side_turn = {side: _shape_turn_for_side(side) for side in present}
    required, excluded = _compute_span_constraints(
        present, side_turn, chains, sr, sc, sz, lane
    )

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len,
            required[side],
            excluded[side],
            SPAN_MIN_MULT * sz,
            min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    obstacle_colors = random.sample(OBSTACLE_COLOR_POOL, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors))
    bg_choices = [c for c in BG_COLOR_POOL if c not in obstacle_colors]
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)

    turns: dict[int, str] = {}
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        color = side_color[side]
        base[r0 : r1 + 1, c0 : c1 + 1] = color
        turns[color] = _turn_from_dims(r1 - r0 + 1, c1 - c0 + 1)
        if turns[color] != side_turn[side]:
            return None

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    try:
        inferred = _infer_obstacle_turns(base, bg=bg, teal=TEAL)
        expected, hits = _simulate_paths(base, inferred, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != obstacle_count + 1 or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if inferred != turns:
        return None

    try:
        actual = main(base)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return base


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 D2 variant input")
