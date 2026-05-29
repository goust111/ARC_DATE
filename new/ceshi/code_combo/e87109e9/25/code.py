# Combo variant of e87109e9 (index 25)
# combo_spec: D13/0+D19/2
# Name: 方向决定转弯加间接颜色映射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保留原题大框架: 顶部 6x24 图例, 下方主区域, 一个青色 2x2/3x3 方块, 3-4 个贴边障碍.
#    - D13/0 先把图例改成“配对式”: 每个 4x4 图例单元里, 中央 2x2 样本色对应主区障碍色,
#      左/右整列竖条是另一种符号色, 两者必须不同.
#    - D19/2 再把“左/右”从原来的固定 L/R 转向, 改成两种“来向轴-转向表”之一:
#      左边条  => 纵向来射(N/S)时顺时针, 横向来射(E/W)时逆时针;
#      右边条  => 纵向来射时逆时针, 横向来射时顺时针.
#    - 具体例子: 某单元可能是“红色中央样本 + 蓝色左边条”. 主区里所有红色障碍都先映射到这
#      个单元; 如果青色射线是向上/向下撞到红障碍, 就按顺时针转; 如果是向左/向右撞到同色
#      障碍, 就按逆时针转. 这比原题多了一步“颜色配对”, 也多了一步“按来向轴选表”.
#
# b) 复杂度校准
#    - 图例仍只有 4 个单元, 每个单元仍只含“边条 + 中央样本”两层信息, 不额外堆叠小符号.
#    - 转向组合只保留 2 档, 避免把 4x4 单元塞进过多状态, 也让少量样例足以观察规律.
#    - 生成器强制至少有一种障碍色同时出现在一个纵向命中侧(N/S)和一个横向命中侧(E/W),
#      这样同色障碍会展示出不同来向下的不同行为, 不会退化成“每色固定一种转法”.
#
# c) 隐式假设审查
#    - 原 `_parse_legend()` 假设单元内部只有一种非黑色; 融合后必须分别解析边条符号色与中央
#      样本色, 并建立 obstacle_color -> symbol_color 的间接映射.
#    - 原 `main()` 在撞色块后只需 `legend_turn[color]`; 现在必须先做 `obstacle -> symbol`,
#      再结合当前运动方向属于纵向还是横向, 用 symbol 所属组合求出实际转向.
#    - 原生成器默认“每个障碍色唯一且转向固定”; 融合后允许不同矩形共用一种障碍色, 但该色在
#      纵向命中和横向命中上必须给出相反的实际转向, 所以颜色分配与转向分配必须联动.
#
# d) 放置合法性
#    - 障碍矩形的厚度、跨度、贴边方式、与青块的间距、互不重叠约束沿用原题.
#    - 生成器仍先决定每个边障碍在路径图上的实际转向, 再据此选 span, 确保每个贴边矩形都会
#      被至少一条射线命中, 且总命中数仍保持 4-5, 每束最多 2 次碰撞.
#    - 颜色复用只发生在“一个纵向侧 + 一个横向侧”的命中配对上, 不会让同一来向轴上的两块
#      障碍共享颜色, 从而避免同色约束互相冲突.
#
# e) 视觉可观察性
#    - 人类先能看到“主区障碍色只在图例中央 2x2 样本里出现, 不是边条颜色”, 因而能推出间接映射.
#    - 再能看到“同一种障碍色既出现在纵向命中位置又出现在横向命中位置, 但输出里的青色路径
#      拐向不同”, 从而逼迫解题者考虑来向.
#    - 这里不用颜色数值、排序、奇偶等不可视属性; 线索完全来自同色匹配和位置/方向.
#
# f) 可视化整洁度
#    - 图例单元固定为左/右整列边条 + 中央 2x2 样本, 结构紧凑且与原题风格接近.
#    - 主区不加入额外噪声物体; 只是少量矩形允许颜色复用, 但它们仍位于四边, 不会挤在中心.
#
# g) 变化的可推断性
#    - 颜色变化规则由“同单元中央样本 ↔ 主区障碍色”这条局部对应给出, 每个样例都会重复出现.
#    - 来向变化规则由同一障碍色跨轴复用来支撑: 至少一色同时接受纵向与横向命中, 并表现相反
#      的转向, 让“方向决定转弯”成为样例中稳定可见的控制变量.
#
# h) main() 的新假设与生成器保证
#    - 假设每个图例单元的中央 2x2 恰有一种样本色, 左右两端恰有一侧是一整列符号条.
#      生成器固定按该模板绘制图例.
#    - 假设被使用的障碍色都能在图例中央样本中唯一反查到, 且样本色与符号色不同.
#      生成器用 4 色错排建立唯一映射, 并仅从样本色里给主区障碍着色.
#    - 假设至少存在一种被命中的障碍色同时用于纵向侧与横向侧.
#      生成器在颜色分配阶段强制构造这一复用对, 并用命中侧验证确保该现象真实发生.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[dict[int, int], dict[int, str]]:
    """
    Parse the paired legend.

    Returns:
      obstacle_to_symbol: obstacle sample color -> symbol bar color
      symbol_combo: symbol bar color -> 'L' or 'R'

    Here 'L'/'R' mean the bar is drawn on the left/right edge of the cell.
    The bar side selects one of two direction-sensitive turn tables.
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    obstacle_to_symbol: dict[int, int] = {}
    symbol_combo: dict[int, str] = {}
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
            combo = "L"
        elif len(right_col) == 1 and not left_col:
            symbol = int(next(iter(right_col)))
            combo = "R"
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
        symbol_combo[symbol] = combo

    assert len(obstacle_to_symbol) == 4, "legend must encode four obstacle samples"
    assert len(symbol_combo) == 4, "legend must encode four symbol bars"
    return obstacle_to_symbol, symbol_combo


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


def _turn_from_combo(combo: str, d: int) -> str:
    """
    Convert a legend combo and incoming beam direction into the actual turn.

    combo == 'L' (bar on left):
      - vertical arrival (N/S) -> 'R' (clockwise)
      - horizontal arrival (E/W) -> 'L' (counterclockwise)
    combo == 'R' (bar on right):
      - vertical arrival (N/S) -> 'L'
      - horizontal arrival (E/W) -> 'R'
    """
    vertical = d in (_N, _S)
    if combo == "L":
        return "R" if vertical else "L"
    assert combo == "R", "legend combo must be left or right"
    return "L" if vertical else "R"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams whose turns come from paired colors and incoming direction.

    Input structure:
    - Top band: a grey legend split into 4 cells; each cell contains a centered 2x2 obstacle-color sample and a full-height bar of a different symbol color on the left or right edge.
    - The centered sample gives the obstacle color found in the main area, while the edge bar is only a paired identifier.
    - Below: the main grid with one solid teal square, a uniform background, and 3-4 colored edge-touching obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam from the square in each cardinal direction, painting teal only onto background cells.
    - When a beam footprint reaches obstacle color C, first find the legend cell whose central sample is C.
    - Then use that cell's bar side to choose between two direction-sensitive turn tables: left bar means vertical arrivals turn clockwise and horizontal arrivals turn counterclockwise; right bar means the reverse.
    - Continue until the beam would leave the cropped main grid.

    Key insight: the obstacle color no longer names the turn directly. It must be mapped through a paired legend cell, and the same obstacle color can bend differently depending on whether the beam arrives vertically or horizontally.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    obstacle_to_symbol, symbol_combo = _parse_legend(g, legend_h)

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
            turn = _turn_from_combo(symbol_combo[symbol], d)
            d = _rot_ccw(d) if turn == "L" else _rot_cw(d)

    return out


# --------------- generator ---------------

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
MAX_RETRIES = 400


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


def _hit_side_for_step(
    rects: dict[int, tuple[int, int, int, int]],
    nr: int,
    nc: int,
    sz: int,
) -> int:
    """Identify which obstacle rectangle overlaps the beam footprint at (nr, nc)."""
    matches = []
    br0, br1 = nr, nr + sz - 1
    bc0, bc1 = nc, nc + sz - 1
    for side, (r0, r1, c0, c1) in rects.items():
        if not (r1 < br0 or br1 < r0 or c1 < bc0 or bc1 < c0):
            matches.append(side)
    assert len(matches) == 1, "beam footprint must overlap exactly one obstacle rectangle"
    return matches[0]


def _simulate_paths(
    base: np.ndarray,
    combo_for_color: dict[int, str],
    bg: int,
    rects: dict[int, tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int]]]]:
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
    hits: list[list[tuple[int, int, int, int, int]]] = []

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
        beam_hits: list[tuple[int, int, int, int, int]] = []
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
            assert obs in combo_for_color
            side = -1 if rects is None else _hit_side_for_step(rects, nr, nc, sz)
            beam_hits.append((d, obs, nr, nc, side))
            d = _rot(d, _turn_from_combo(combo_for_color[obs], d))
            assert len(beam_hits) <= 2
        hits.append(beam_hits)

    return out, hits


def _make_legend(
    symbol_colors: list[int],
    obstacle_for_symbol: dict[int, int],
    symbol_combo: dict[int, str],
) -> np.ndarray:
    """Build the 6x24 legend from paired symbol colors and obstacle samples."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, symbol in zip(LEGEND_CELL_STARTS, symbol_colors, strict=False):
        obstacle = obstacle_for_symbol[symbol]
        legend[1:5, start : start + 4] = 0
        bar_col = start if symbol_combo[symbol] == "L" else start + 3
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
    """Assign the actual L/R turn taken at each obstacle side."""
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


def _combo_for_side_turn(side: int, turn: str) -> str:
    """Return the legend combo needed to realize *turn* when hitting *side*."""
    if side in (_N, _S):
        return "L" if turn == "R" else "R"
    return "L" if turn == "L" else "R"


def _pick_shared_pair(
    present: set[int],
    side_turn: dict[int, str],
) -> tuple[int, int] | None:
    """Choose one vertical side and one horizontal side that need opposite turns."""
    vertical = [side for side in present if side in (_N, _S)]
    horizontal = [side for side in present if side in (_E, _W)]
    candidates = [
        (v, h)
        for v in vertical
        for h in horizontal
        if side_turn[v] != side_turn[h]
    ]
    return random.choice(candidates) if candidates else None


def _assign_colors_and_combos(
    sample_colors: list[int],
    present: set[int],
    side_turn: dict[int, str],
) -> tuple[dict[int, int], dict[int, str], tuple[int, int, int]] | None:
    """
    Assign obstacle colors to sides and choose each sample color's combo.

    Returns:
      side_color: side -> obstacle/sample color
      sample_combo: obstacle/sample color -> 'L'/'R' combo
      shared_info: (shared_color, vertical_side, horizontal_side)
    """
    pair = _pick_shared_pair(present, side_turn)
    if pair is None:
        return None
    v_side, h_side = pair

    palette = sample_colors[:]
    random.shuffle(palette)
    shared_color = palette.pop()
    side_color: dict[int, int] = {v_side: shared_color, h_side: shared_color}

    remaining_sides = [side for side in sorted(present) if side not in (v_side, h_side)]
    if len(remaining_sides) > len(palette):
        return None
    for side in remaining_sides:
        side_color[side] = palette.pop()

    sample_combo: dict[int, str] = {}
    for side, color in side_color.items():
        combo = _combo_for_side_turn(side, side_turn[side])
        if color in sample_combo and sample_combo[color] != combo:
            return None
        sample_combo[color] = combo

    for color in sample_colors:
        sample_combo.setdefault(color, random.choice(["L", "R"]))

    return side_color, sample_combo, (shared_color, v_side, h_side)


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
            axis_len,
            required[side],
            excluded[side],
            SPAN_MIN_MULT * sz,
            min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    symbol_colors = random.sample(PAIR_COLOR_POOL, 4)
    sample_colors = _random_derangement(symbol_colors)
    obstacle_for_symbol = dict(zip(symbol_colors, sample_colors, strict=False))

    color_pack = _assign_colors_and_combos(sample_colors, present, side_turn)
    if color_pack is None:
        return None
    side_color, sample_combo, shared_info = color_pack
    shared_color, shared_v_side, shared_h_side = shared_info

    bg_choices = [c for c in BG_COLOR_POOL if c not in set(symbol_colors) | set(sample_colors)]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    rects: dict[int, tuple[int, int, int, int]] = {}
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        rects[side] = (r0, r1, c0, c1)
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

    symbol_combo = {
        symbol: sample_combo[obstacle_for_symbol[symbol]]
        for symbol in symbol_colors
    }

    try:
        expected, hits = _simulate_paths(base, sample_combo, bg, rects)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_sides = {side for beam in hits for _, _, _, _, side in beam}
    shared_color_hit_sides = {
        side for beam in hits for _, color, _, _, side in beam if color == shared_color
    }
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if hit_sides != present:
        return None
    if not ({shared_v_side, shared_h_side} <= shared_color_hit_sides):
        return None
    if len({side_color[side] for side in present}) >= obstacle_count:
        return None

    legend = _make_legend(symbol_colors, obstacle_for_symbol, symbol_combo)
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
    raise RuntimeError("failed to generate e87109e9 combo input")
