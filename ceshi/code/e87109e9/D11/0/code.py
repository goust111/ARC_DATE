# Variant of e87109e9
# Dimension: D11 — 障碍物对射线的阻挡行为 (path)
# Change: 射线碰到障碍物边缘时立即转弯，障碍物完全阻挡射线不让穿过... → 半透明障碍物（射线穿透后转弯）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the D11 variant
# ----------------------------------
# spec:
#   invariant_structure:
#     - 输入仍由 6x24 的顶部图例和下方主网格组成。
#     - 图例仍用 4 个竖条编码 4 种障碍颜色的左转/右转。
#     - 主网格仍包含一个唯一的青色实心方块和 3-4 个单色矩形障碍。
#   concrete_variant:
#     - 障碍物在输入里看起来没有变化，仍是普通实心矩形。
#     - 变化只发生在输出路径上：青色方块射线第一次碰到障碍时不转弯，而是继续直行穿过该同色矩形。
#     - 射线在障碍内部经过的 footprint 也会涂成青色；直到它穿过障碍并到达另一侧紧邻的出口格，才按照图例转向。
#     - 为了让这个“先出来再转弯”的事件在离散网格里真正发生，变体把障碍从原题常见的贴边长条改为悬空矩形：障碍与外边界之间保留至少 1 格出口空间。
#   trace_example:
#     - 例如 sz=2，向北射线的下一步 top-left 从 r=8 到 r=7 时第一次碰到位于 rows 4..6 的障碍。
#     - 旧规则：在 r=8 的位置就立刻根据图例改方向，障碍内部保持原色。
#     - 新规则：射线依次占据 r=7,6,5,4 这些与障碍相交的位置，并把这些位置刷成青色；随后再前进一步到出口 r=3，并在这个出口格转向。
#     - 因此输出里会多出一段穿过障碍的青色通道，而且后续横向/纵向段的起点会平移一个“障碍厚度”的距离。
#   complexity_calibration:
#     - 若障碍厚度常为 1，则新旧规则几乎不可区分，所以这里把厚度限制在 3-4，保证远侧转弯可见。
#     - 仍只保留左右两个转向档位，不引入额外颜色语义，避免样例中规则分支过多。
#   hidden_assumptions_review:
#     - 原 main() 的 obstacle_color_at_next() 读取的是可变输出 out；旧规则下障碍从不被涂色，所以无害。
#       新规则会覆盖障碍色，因此碰撞检测必须改为读取静态 base，否则进入障碍后就“看不见”障碍了。
#     - 原 paint_block() 只在 bg 上着色；旧规则不会进入障碍，所以无害。
#       新规则要求障碍内部也着色，因此这里必须允许覆盖障碍颜色。
#     - 原 while 循环假设“看到障碍 -> 只改方向，不移动”；新规则必须显式模拟“在同色矩形内部持续前进直到出口”。
#   placement_legality:
#     - generator 仍保证障碍互不重叠、与青色方块至少隔 1 格，并且每个 sz x sz footprint 同时只会命中 0 或 1 种障碍颜色。
#     - 为避免“刚转弯又仍在同一障碍里”的悖论，每个障碍在行进方向的远侧都预留一个出口格。
#     - 本变体生成器固定为“一束光对应一次穿透+一次转弯”，因此任何被转向指向的相邻障碍，都会显式避开该出口车道。
#     - generator 还会用新 solver oracle 以及旧规则 oracle 同时校验，并拒绝两者输出完全或近乎等价的样本。
#   observability_and_cleanliness:
#     - 人眼能直接看到“障碍里面出现青色通道”以及“转弯点从近侧移到远侧”这两个视觉证据。
#     - 不使用颜色 id 排序，不引入新的颜色映射；颜色仍只承担“查图例决定左/右转”的原任务角色。

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Legend is a top region with a background color (typically 5) and 4 vertical bars.
    Each bar's horizontal position inside its cell encodes turn direction:
      - bar on left  -> 'L' (turn counterclockwise)
      - bar on right -> 'R' (turn clockwise)
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        xs = np.where(interior == col)[1]
        x = int(xs[0])
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0,c0,size) for a solid square teal component."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four teal square-beams; colored blocks behave like semi-transparent turn gates.

    Input structure:
    - Top band: a grey legend whose four cells each contain one colored vertical bar; left-vs-right bar placement encodes how that color turns a beam.
    - Below: one main board with a background color, one solid teal source square, and several solid colored obstacle rectangles.

    Rule:
    - Remove the legend band in the output.
    - Launch one sz-by-sz teal beam from the source square in each cardinal direction.
    - Every footprint visited by a beam is painted teal, including obstacle cells.
    - When the next footprint first meets an obstacle of color C, the beam keeps moving straight through that same obstacle; once it emerges into the first non-obstacle footprint beyond C, it turns according to C's legend entry.
    - After turning, the beam continues from that exit footprint and stops only when the next step would leave the board.

    Key insight: The legend still tells you left or right for each obstacle color, but D11 changes when that turn happens: not at first contact, but only after the beam has crossed the obstacle and stepped out on the opposite side.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)

    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = teal

    def obstacle_color_at(r: int, c: int) -> int | None:
        blk = base[r : r + sz, c : c + sz]
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

            oc = obstacle_color_at(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            exited = False
            while True:
                r, c = nr, nc
                paint_block(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                oc2 = obstacle_color_at(nnr, nnc)
                if oc2 == oc:
                    nr, nc = nnr, nnc
                    continue
                assert oc2 is None, "different obstacles must not touch along a traversed lane"
                r, c = nnr, nnc
                paint_block(r, c)
                exited = True
                break
            if not exited:
                break
            d = _rot(d, legend_turn[oc])

    return out


# --------------- generator (o5x) ---------------

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (3, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 600


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span that covers all required intervals and avoids all excluded ones."""
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


def _simulate_transparent_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int, int]]]]:
    """Simulate the D11 rule: beams traverse a hit obstacle, exit it, then turn."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = TEAL

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(base[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        beam_hits: list[tuple[int, int, int, int, int, int]] = []
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
            entry_r, entry_c = nr, nc
            exited = False
            while True:
                r, c = nr, nc
                paint(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                obs2 = obstacle_at(nnr, nnc)
                if obs2 == obs:
                    nr, nc = nnr, nnc
                    continue
                assert obs2 is None
                r, c = nnr, nnc
                paint(r, c)
                exited = True
                break
            if not exited:
                break
            beam_hits.append((d, obs, entry_r, entry_c, r, c))
            d = _rot(d, turns[obs])
        hits.append(beam_hits)

    return out, hits


def _simulate_edge_turn_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Legacy rule oracle used only to reject degenerate samples."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(base[r : r + sz, c : c + sz])))
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
        hits.append(beam_hits)

    return out, hits


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their turn assignments."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


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


def _floating_obstacle_rect(
    side: int, primary: int, thickness: int, span: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) for an interior obstacle rectangle."""
    s0, s1 = span
    if side == _N:
        return primary, primary + thickness - 1, s0, s1
    if side == _S:
        return primary, primary + thickness - 1, s0, s1
    if side == _W:
        return s0, s1, primary, primary + thickness - 1
    return s0, s1, primary, primary + thickness - 1


def _select_chains(present: set[int], second_hits: int) -> dict[int, int] | None:
    """Pick non-overlapping (source -> adjacent target) pairs for multi-bounce beams."""
    candidates = [
        (side, nxt)
        for side in present
        for nxt in ((side - 1) % 4, (side + 1) % 4)
        if nxt in present
    ]
    random.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for side, nxt in candidates:
        if len(selected) == second_hits:
            break
        if side in used or nxt in used:
            continue
        selected.append((side, nxt))
        used.update((side, nxt))
    return dict(selected) if len(selected) == second_hits else None


def _assign_turns(present: set[int], chains: dict[int, int]) -> dict[int, str]:
    """Assign L/R turn direction for each obstacle side."""
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
        else:
            outward = [t for t in ("L", "R") if _rot(side, t) not in present]
            side_turn[side] = random.choice(outward if outward else ["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int],
    side_turn: dict[int, str],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for each obstacle span."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

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


def _choose_centered_span(
    anchor0: int,
    anchor1: int,
    allowed0: int,
    allowed1: int,
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a moderate-length span covering [anchor0, anchor1] inside [allowed0, allowed1]."""
    if allowed0 > anchor0 or allowed1 < anchor1:
        return None
    max_len = min(max_len, allowed1 - allowed0 + 1, min_len + 7)
    if max_len < min_len:
        return None
    start_lo = max(allowed0, anchor1 - max_len + 1)
    start_hi = min(anchor0, allowed1 - min_len + 1)
    if start_lo > start_hi:
        return None
    start = random.randint(start_lo, start_hi)
    end_lo = max(anchor1, start + min_len - 1)
    end_hi = min(allowed1, start + max_len - 1)
    if end_lo > end_hi:
        return None
    end = random.randint(end_lo, end_hi)
    return start, end


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4
    present = {_N, _E, _S, _W}
    target_hits = 4

    ns = _thickness_pair(h, sz, _N in present, _S in present)
    ew = _thickness_pair(w, sz, _W in present, _E in present)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    if _N in present:
        min_r = max(min_r, thickness[_N] + 2)
    if _S in present:
        max_r = min(max_r, h - thickness[_S] - sz - 2)
    if _W in present:
        min_c = max(min_c, thickness[_W] + 2)
    if _E in present:
        max_c = min(max_c, w - thickness[_E] - sz - 2)
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    north_row_lo, north_row_hi = 1, sr - thickness[_N] - 2
    south_row_lo, south_row_hi = sr + sz + 1, h - thickness[_S] - 2
    west_col_lo, west_col_hi = 1, sc - thickness[_W] - 2
    east_col_lo, east_col_hi = sc + sz + 1, w - thickness[_E] - 2
    if (
        north_row_lo > north_row_hi
        or south_row_lo > south_row_hi
        or west_col_lo > west_col_hi
        or east_col_lo > east_col_hi
    ):
        return None
    primary = {
        _N: random.randint(north_row_lo, north_row_hi),
        _S: random.randint(south_row_lo, south_row_hi),
        _W: random.randint(west_col_lo, west_col_hi),
        _E: random.randint(east_col_lo, east_col_hi),
    }

    lane = {
        _N: primary[_N] - 1,
        _S: primary[_S] + thickness[_S],
        _W: primary[_W] - 1,
        _E: primary[_E] + thickness[_E],
    }

    chains: dict[int, int] = {}
    side_turn = {side: random.choice(["L", "R"]) for side in present}
    required, excluded = _compute_span_constraints(present, side_turn, chains, sr, sc, sz, lane)

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len,
            required[side],
            excluded[side],
            max(5, SPAN_MIN_MULT * sz),
            min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _floating_obstacle_rect(side, primary[side], thickness[side], spans[side])
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, hits = _simulate_transparent_paths(base, turns, bg)
        legacy, _ = _simulate_edge_turn_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

    painted_obstacle_cells = int(np.count_nonzero((expected == TEAL) & (base != bg) & (base != TEAL)))
    delay_sum = sum(abs(er - rr) + abs(ec - cc) for beam in hits for _, _, rr, cc, er, ec in beam)
    diff_vs_legacy = int(np.count_nonzero(expected != legacy))
    if painted_obstacle_cells < sz * obstacle_count:
        return None
    if delay_sum < target_hits:
        return None
    if diff_vs_legacy < 3 * sz:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 D11 input")
