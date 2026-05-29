# Variant of 9385bd28
# Dimension: D2 — 源对象：标记的形状结构 (src)
# Change: L 形 3 像素角标记，位于外接矩形的对角两端（共 6 像素）；另有 2x2 实心方块变体... → 边框标记（矩形边的一段）

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg

# Design analysis
# ----------------
# variant_spec:
#   preserved:
#   - 网格尺寸、左下图例、颜色图例映射、bottom->top 处理顺序、以及“根据被标出的对角线去填另一条对角线的菱形区域”都保留。
#   changed:
#   - 每个有效 marker 不再是两个 L 角或 2x2 方块，而是隐藏矩形边框上的两段连续像素。
#   - 两段一定落在一对相对边上：要么顶边+底边，要么左边+右边。
#   - TLBR 编码：触碰 TL 与 BR 两个角；TRBL 编码：触碰 TR 与 BL 两个角。
#   - 具体做法：
#     - 顶/底型：顶边只画左端或右端一段；底边画相反一端一段。
#     - 左/右型：左边只画上端或下端一段；右边画相反一端一段。
#   - 每段长度限制在对应边长的一半以内，使“只给部分边界、需要脑补完整矩形”在视觉上明显可见。
#
# behavior_trace:
#   original_local_step:
#   - 原 solver 先取某颜色所有像素的 bbox，再检查四角是否存在 L 形锚点，由此判断是 TLBR 还是 TRBL 被标记。
#   variant_local_step:
#   - 现在 bbox 仍由该颜色像素的极值给出，但锚点检测改成“哪两个角被连续边段触碰”。
#   - 例子：顶边左半段 + 底边右半段 => 触碰 TL 和 BR => 被标记的是 TLBR，对应填充另一条对角 TR-BL 的菱形。
#   difference:
#   - 原题直接给出角结构；变体只给出边界残片，需要从两段边界推回完整矩形和被标记对角线。
#
# complexity_calibration:
#   - 只保留两种朝向（顶/底，左/右），不引入更多边组合，避免规则分支过多。
#   - 但每个样例中对象可混合这两种朝向，且仍保留 noop 图例项与非图例干扰物，使规则不会退化成单一模板匹配。
#   - 去掉 2x2 特例，避免旧逻辑残留把 D2 变化冲淡；同时段长限制保证肉眼能看出“边框残片”而不是整条边。
#
# implicit_assumption_audit:
#   - `_l_anchor_present` 依赖 corner-L 结构，在本变体中失效，必须替换。
#   - 原 generator 的 bbox 合法性围绕 L 角重叠/tie 设计，在本变体中不再适用；新的合法性改为：
#     - bbox 至少 4x4，确保边段可短于整边且仍可见；
#     - 两段各自连续、各自只贴一端，保证对角线推断唯一。
#   - `main()` 新增假设：同色 marker 像素全部位于某个 bbox 的两条相对边上，且每条边上恰是一段连续区间；
#     generator 会显式保证这一点。
#
# placement_legality:
#   - 所有 marker 像素仍禁止与图例和其它 marker 像素重叠；bbox 内部可以与其它对象的未来填充区域相交，这与原题处理顺序一致。
#   - 图例左侧扫描完整性继续通过 forbidden mask 保护。
#
# visual_observability:
#   - 规则只依赖可见几何关系：哪条边有段、该段贴哪一端、以及图例颜色配对。
#   - 不用颜色 id 排序，也不把颜色数值作为几何判断条件。
#   - 边段长度有变化，但输出行为只由“贴到哪两个角”决定，人类能从少量样例中归纳。
#
# cleanliness:
#   - 采用 4x4 及以上 bbox，边段不占满整边，减少拥挤与误判。
#   - 允许 bbox 交叠但禁止 marker 像素重叠，保证输入阶段每个 marker 仍清晰可辨。


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _is_contiguous(vals: list[int]) -> bool:
    if not vals:
        return False
    ordered = sorted(vals)
    return len(ordered) == len(set(ordered)) and ordered[-1] - ordered[0] + 1 == len(ordered)


def _infer_fill_diagonal(marker_pts: set[tuple[int, int]]) -> tuple[int, int, int, int, tuple[int, int], tuple[int, int]]:
    """
    Infer the hidden bbox and the opposite fill diagonal from two partial boundary segments.

    Valid shapes are exactly one of:
    - top + bottom contiguous segments
    - left + right contiguous segments
    Each segment touches exactly one end of its side, which uniquely determines the marked diagonal.
    """
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    assert rmin < rmax and cmin < cmax, "marker bbox must have positive area"

    horizontal_only = all(r in (rmin, rmax) for r, _ in marker_pts)
    vertical_only = all(c in (cmin, cmax) for _, c in marker_pts)
    assert horizontal_only != vertical_only, "marker must occupy exactly one pair of opposite edges"

    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    if horizontal_only:
        top_cs = sorted(c for r, c in marker_pts if r == rmin)
        bot_cs = sorted(c for r, c in marker_pts if r == rmax)
        assert top_cs and bot_cs and _is_contiguous(top_cs) and _is_contiguous(bot_cs), (
            "top/bottom segments must be contiguous"
        )
        assert (top_cs[0] == cmin) ^ (top_cs[-1] == cmax), "top segment must touch exactly one corner"
        assert (bot_cs[0] == cmin) ^ (bot_cs[-1] == cmax), "bottom segment must touch exactly one corner"

        score_tlbr = int(top_cs[0] == cmin) + int(bot_cs[-1] == cmax)
        score_trbl = int(top_cs[-1] == cmax) + int(bot_cs[0] == cmin)
    else:
        left_rs = sorted(r for r, c in marker_pts if c == cmin)
        right_rs = sorted(r for r, c in marker_pts if c == cmax)
        assert left_rs and right_rs and _is_contiguous(left_rs) and _is_contiguous(right_rs), (
            "left/right segments must be contiguous"
        )
        assert (left_rs[0] == rmin) ^ (left_rs[-1] == rmax), "left segment must touch exactly one corner"
        assert (right_rs[0] == rmin) ^ (right_rs[-1] == rmax), "right segment must touch exactly one corner"

        score_tlbr = int(left_rs[0] == rmin) + int(right_rs[-1] == rmax)
        score_trbl = int(right_rs[0] == rmin) + int(left_rs[-1] == rmax)

    assert score_tlbr != score_trbl, "marker diagonal must be uniquely determined"
    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br
    return rmin, rmax, cmin, cmax, a, b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill inferred rectangles from partial boundary markers using a color legend.

    Input structure:
    - A background-colored grid contains a vertical 2-column legend near the bottom-left.
    - Each legend row gives (marker_color, fill_color), read from bottom to top.
    - Away from the legend, a marker color appears as two short edge segments on opposite sides of some hidden rectangle.

    Rule:
    - For each legend entry, collect all non-legend pixels of that marker color.
    - Use the extreme rows/cols to recover the hidden bbox, then read which corners are touched by the two edge segments.
    - The touched corners identify the marked diagonal; fill the opposite diagonal's Manhattan diamond with the legend fill color.
    - Keep legend pixels and all current/future marker pixels unchanged while processing earlier entries.

    Key insight: The marker no longer gives corners directly; it gives two boundary fragments whose attached ends reveal which diagonal was marked, and the fill still belongs on the other diagonal.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        rmin, rmax, cmin, cmax, a, b = _infer_fill_diagonal(marker_pts)
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3
EDGE_ORIENT_WEIGHTS = [0.55, 0.45]  # H, V

# ── CFG: edge-pair bbox shape ──────────────────────────────────────
BBOX_H_MIN = 4
BBOX_W_MIN = 4
BBOX_H_MAX = 12
BBOX_W_MAX = 12
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
EDGE_RETRIES = 160
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 300


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    h_hi = min(BBOX_H_MAX, H)
    w_hi = min(BBOX_W_MAX, W)
    if h_hi < BBOX_H_MIN or w_hi < BBOX_W_MIN:
        return None
    return random.randint(BBOX_H_MIN, h_hi), random.randint(BBOX_W_MIN, w_hi)


def _pick_segment_len(span: int) -> int:
    return random.randint(2, max(2, span // 2))


def _edge_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    orientation: str,
    diag: str,
    len1: int,
    len2: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if orientation == "H":
        if diag == "TLBR":
            cells.extend((rmin, c) for c in range(cmin, cmin + len1))
            cells.extend((rmax, c) for c in range(cmax - len2 + 1, cmax + 1))
        else:
            cells.extend((rmin, c) for c in range(cmax - len1 + 1, cmax + 1))
            cells.extend((rmax, c) for c in range(cmin, cmin + len2))
    else:
        if diag == "TLBR":
            cells.extend((r, cmin) for r in range(rmin, rmin + len1))
            cells.extend((r, cmax) for r in range(rmax - len2 + 1, rmax + 1))
        else:
            cells.extend((r, cmax) for r in range(rmin, rmin + len1))
            cells.extend((r, cmin) for r in range(rmax - len2 + 1, rmax + 1))
    return cells


def _try_place_edge_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1

        orientation = random.choices(["H", "V"], weights=EDGE_ORIENT_WEIGHTS)[0]
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        span = bw if orientation == "H" else bh
        len1 = _pick_segment_len(span)
        len2 = _pick_segment_len(span)
        cells = _edge_pair_cells(rmin, cmin, rmax, cmax, orientation, diag, len1, len2)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    types = ["edge"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for i, (_, marker, _) in enumerate(legend_rows):
        if types[i] == "noop":
            continue
        if not _try_place_edge_pair(H, W, marker, EDGE_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_edge_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
