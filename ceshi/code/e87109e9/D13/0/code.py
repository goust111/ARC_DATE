# Variant of e87109e9
# Dimension: D13 — 图例与障碍物的对应关系 (symbol_instructions)
# Change: 图例中每个颜色条目与主区域中同色障碍物一一对应——图例颜色即障碍物颜色，4个条目对应最多4种障碍物颜色... → 图例颜色与障碍物颜色不同（间接映射）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保持原题整体版式: 顶部 6x24 图例, 下方主区域, 一个青色 2x2/3x3 方块向四向发射射线.
#    - 变体只改图例单元的内部编码. 每个 4x4 单元现在有两种非背景色:
#      1) 一根满高竖条, 放在最左列或最右列, 其左右位置仍表示 L/R 转向.
#      2) 一个固定放在中央的 2x2 色块, 其颜色才对应主区域里的障碍物颜色.
#    - 因此求解流程从原来的 "障碍物颜色 -> 同色图例条目 -> 读左右" 变成
#      "障碍物颜色 -> 找到拥有该中央样本色的图例单元 -> 取该单元的条形符号色 ->
#       读该条位于左/右 -> 得到转向".
#    - 例子: 原题里障碍物颜色 2 直接去找图例里颜色 2 的竖条. 变体里可能是
#      "紫条 + 红色样本", 而主区域障碍物是红色; 撞到红障碍时, 需要先定位到
#      这个带红样本的单元, 再读取紫条在左还是右.
#
# b) 复杂度校准
#    - 不引入额外几何对象, 只把每个图例单元从 1 色改成 2 色, 复杂度接近原题.
#    - 4 个单元仍然足够少, 人可以在少量样例里观察到 "障碍颜色看中央样本, 方向看侧边条".
#    - 生成器强制使用 derangement: 每个单元的条形符号色都不同于本单元中央样本色,
#      防止退化回原题的直接同色匹配.
#
# c) 隐式假设审查
#    - 原 `_parse_legend()` 假设每个图例单元只有一种非黑非灰颜色; 变体不成立,
#      必须改成分别解析 "侧边满高条" 与 "中央 2x2 样本".
#    - 原 `main()` 的 `assert oc in legend_turn` 依赖 "障碍色就是图例键"; 变体中
#      障碍色只存在于中央样本, 所以要改成 `oc -> symbol -> turn` 两步.
#    - 原生成器把 `turns` 直接按障碍物颜色写入图例; 变体里需要同时维护
#      `symbol_turn` 和 `sample_turn`, 并保证用到的障碍样本色都能从图例反查.
#
# d) 放置合法性
#    - 主区域的障碍放置、间距、厚度、碰撞检查全部沿用原题逻辑, 只改颜色分配.
#    - 图例单元内部布局固定: 中央 2x2 样本不碰左右边界, 不会与左右任一条形重叠.
#    - 由于条色和样本色来自同一 4 色集合的错排, 图例与主区域之间总能形成可见配对.
#
# e) 视觉可观察性
#    - 在输入图里, 障碍颜色与图例里的侧边条颜色故意不一致, 这个差异肉眼可见.
#    - 中央 2x2 样本是唯一与主区域障碍物直接同色的部件, 可以稳定提示配对关系.
#    - 左/右信息仍由明显的满高竖条承担, 不与样本位置混淆.
#
# f) 可视化整洁度
#    - 每个图例单元仍限制在 4x4 内, 只含 "一根边条 + 一个中心样本", 无额外叠加.
#    - 主区域完全保留原题整洁布局, 不增加噪声对象.
#
# g) 颜色变化可推断性
#    - 只使用 4 个图例单元, 每个单元恰好给出一次颜色配对示例.
#    - 配对规则是局部、可见、固定的: 同单元内中央样本色对应主区障碍色, 条形仅负责方向.
#    - 不使用颜色 id 排序或数值运算, 只使用同色匹配和单元内相对位置.
#
# h) main() 的新假设与生成器保证
#    - 假设每个图例单元宽度仍为 4, 且中央 2x2 区域只有 1 种样本色.
#      生成器固定使用 4x4 单元与中央 2x2 样本.
#    - 假设左右边界列中恰有一列是满高符号条.
#      生成器固定把符号条画在最左或最右整列.
#    - 假设每种被使用的障碍色在图例中央样本里唯一出现一次.
#      生成器使用 4 色双射并给主区域障碍分配互异颜色.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[dict[int, int], dict[int, str]]:
    """
    Parse the paired legend.

    Returns:
      obstacle_to_symbol: obstacle sample color -> symbol bar color
      symbol_turn: symbol bar color -> 'L' or 'R'
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    obstacle_to_symbol: dict[int, int] = {}
    symbol_turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        assert interior.shape == (4, 4), "legend cell interior must stay 4x4"

        left_col = set(map(int, interior[:, 0]))
        right_col = set(map(int, interior[:, -1]))
        left_col.discard(bg)
        left_col.discard(0)
        right_col.discard(bg)
        right_col.discard(0)

        if len(left_col) == 1 and not right_col:
            symbol = int(next(iter(left_col)))
            turn = "L"
        elif len(right_col) == 1 and not left_col:
            symbol = int(next(iter(right_col)))
            turn = "R"
        else:
            raise AssertionError("each legend cell must have one edge-aligned symbol bar")

        sample = interior[1:3, 1:3]
        sample_colors = set(map(int, np.unique(sample)))
        sample_colors.discard(bg)
        sample_colors.discard(0)
        assert len(sample_colors) == 1, "legend sample block must encode exactly one obstacle color"
        obstacle = int(next(iter(sample_colors)))
        assert obstacle != symbol, "paired legend colors must differ in this variant"

        obstacle_to_symbol[obstacle] = symbol
        symbol_turn[symbol] = turn

    assert len(obstacle_to_symbol) == 4, "legend must encode four obstacle samples"
    assert len(symbol_turn) == 4, "legend must encode four symbol bars"
    return obstacle_to_symbol, symbol_turn


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
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams whose turns are decoded from paired legend entries.

    Input structure:
    - Top band: grey legend cells, each containing an edge-aligned colored bar and a centered 2x2 color sample.
    - The 2x2 sample color matches one obstacle color in the main area, while the bar color is only a paired symbol.
    - Below: main grid with background, one solid teal square, and colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - From the teal square, launch one sz-by-sz beam in each cardinal direction, painting teal only onto background cells.
    - When a beam footprint reaches obstacle color C, find the legend cell whose central sample is C; a left bar means turn counterclockwise and a right bar means turn clockwise.
    - Continue after each turn until the beam would leave the grid.

    Key insight: obstacle colors no longer identify turn instructions directly; each obstacle must first be matched to its paired legend sample, and only that entry's bar position reveals the turn direction.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    obstacle_to_symbol, symbol_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

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

            assert oc in obstacle_to_symbol, "obstacle color must be represented by a legend sample"
            symbol = obstacle_to_symbol[oc]
            turn = symbol_turn[symbol]
            d = _rot_ccw(d) if turn == "L" else _rot_cw(d)

    return out


# --------------- generator ---------------

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
PAIR_COLOR_POOL = [1, 2, 3, 4, 6]
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
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 300


def _rot(d: int, turn: str) -> int:
    """Rotate direction d by turn ('L' = CCW, 'R' = CW)."""
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _random_derangement(items: list[int]) -> list[int]:
    """Return a shuffled copy where no item stays in its original slot."""
    perm = items[:]
    for _ in range(100):
        random.shuffle(perm)
        if all(a != b for a, b in zip(items, perm, strict=False)):
            return perm
    raise AssertionError("failed to build color derangement")


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a (start, end) span covering all required intervals and avoiding all excluded intervals."""
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
    """Simulate 4 beams from the teal square, returning painted grid and per-beam hit lists."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    teal_locs = np.argwhere(base == TEAL)
    assert teal_locs.size
    sr, sc = teal_locs.min(axis=0)
    r1, c1 = teal_locs.max(axis=0)
    sz = int(r1 - sr + 1)
    assert sz == int(c1 - sc + 1)

    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

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


def _make_legend(
    symbol_colors: list[int],
    obstacle_for_symbol: dict[int, int],
    symbol_turns: dict[int, str],
) -> np.ndarray:
    """Build the 6x24 legend from paired symbol colors and obstacle samples."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, symbol in zip(LEGEND_CELL_STARTS, symbol_colors, strict=False):
        obstacle = obstacle_for_symbol[symbol]
        legend[1:5, start : start + 4] = 0
        bar_col = start if symbol_turns[symbol] == "L" else start + 3
        legend[1:5, bar_col] = symbol
        legend[2:4, start + 1 : start + 3] = obstacle
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    """Return random thickness in [lo, hi], or None if infeasible."""
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


def _select_chains(
    present: set[int], second_hits: int
) -> dict[int, int] | None:
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


def _assign_turns(
    present: set[int], chains: dict[int, int]
) -> dict[int, str]:
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
    present: set[int], side_turn: dict[int, str], chains: dict[int, int],
    sr: int, sc: int, sz: int, lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for each obstacle's span selection."""
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
    target_hits = random.choice([4, 5])
    second_hits = target_hits - obstacle_count
    if second_hits < 0 or second_hits > 2:
        return None

    ns = _thickness_pair(h, sz, _N in present, _S in present)
    ew = _thickness_pair(w, sz, _W in present, _E in present)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew), strict=False) if t is not None}

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

    lane = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    chains = _select_chains(present, second_hits)
    if chains is None:
        return None

    side_turn = _assign_turns(present, chains)
    required, excluded = _compute_span_constraints(
        present, side_turn, chains, sr, sc, sz, lane,
    )

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len, required[side], excluded[side],
            SPAN_MIN_MULT * sz, min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    symbol_colors = random.sample(PAIR_COLOR_POOL, 4)
    sample_colors = _random_derangement(symbol_colors)
    obstacle_for_symbol = dict(zip(symbol_colors, sample_colors, strict=False))
    bg_choices = [c for c in BG_COLOR_POOL if c not in set(symbol_colors) | set(sample_colors)]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    used_obstacle_colors = random.sample(sample_colors, obstacle_count)
    side_color = dict(zip(sorted(present), used_obstacle_colors, strict=False))

    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
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

    sample_turns = {color: random.choice(["L", "R"]) for color in sample_colors}
    for side in present:
        sample_turns[side_color[side]] = side_turn[side]
    symbol_turns = {
        symbol: sample_turns[obstacle_for_symbol[symbol]]
        for symbol in symbol_colors
    }
    used_samples = set(side_color.values())
    if not any(sample_turns[color] != symbol_turns[color] for color in used_samples):
        return None

    try:
        expected, hits = _simulate_paths(base, sample_turns, bg)
    except AssertionError:
        return None
    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

    legend = _make_legend(symbol_colors, obstacle_for_symbol, symbol_turns)
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
    raise RuntimeError("failed to generate e87109e9 D13 input")
