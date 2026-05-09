# Combo variant of 9385bd28 (index 16)
# combo_spec: D2/2+D5/4+D9/2
# Name: 线段标记+同对角线填充+填充后擦除标记

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 原题输入里的有效 marker 是一对对角 L 角；本融合版把它改成两条同色短线段，长度固定为 3 或 4。
#    - 两条线段一定同向，并分别贴在同一个隐藏 bbox 的两条相对边上：
#      * 水平型:
#          TLBR 标记对: 顶边左端一条横线 + 底边右端一条横线
#          TRBL 标记对: 顶边右端一条横线 + 底边左端一条横线
#      * 垂直型:
#          TLBR 标记对: 左边上端一条竖线 + 右边下端一条竖线
#          TRBL 标记对: 左边下端一条竖线 + 右边上端一条竖线
#    - 原题代码用“角标所在对角线的对面”决定填充；但原 `diamond` 实际会把整块 bbox 都刷满，导致对角线选择不可见。
#      融合版沿用 D9 的做法：改成填充“更接近某一条对角线”的对角带区域，这样“同对角线”才会真实可见。
#    - 具体例子：若输入有“顶边左端横线 + 底边右端横线”，solver 先解码出 bbox 和标记对角线 = TLBR；
#      原逻辑会去刷 TRBL 那一侧，而融合版改为刷 TLBR 那一侧；刷完之后，再把这两条横线像素擦回背景。
#      所以输出差异同时体现在三处：标记形状从角变成线、填充方向从对面对角线改成同对角线、标记最终消失。
# b) 复杂度校准
#    - 只保留 4 个离散结构类别: 水平/垂直 × TLBR/TRBL。线段长度只允许 3 或 4，避免规则自由度过多。
#    - 去掉 2x2 block 分支，避免把“线段是临时脚手架”与“实心块直接改色”混在一起，削弱融合主线。
#    - 保留少量 noop entry 和少量非图例 distractor，但 generator 会强制至少有一个真实线段对样例能同时显出 D5 与 D9。
# c) 隐式假设审查
#    - 原 `_l_anchor_present(...)` 与角点打分逻辑已失效，因为 marker 不再落在四角；必须改为连通分量 + 直线段分类。
#    - 原 `protect |= current marker` 依赖“当前标记保留可见”；融合版不能这样做，否则擦除步骤失效。
#    - 但为了量化 D5 是否真的发生，代码里仍保留一个“保留标记”的对照求解分支，仅用于 generator 指标，不作为正式答案。
#    - 原 solver 直接从输入 `g` 取 marker，再写到 `out`；这个前提仍要保留。否则若边求边擦，会破坏后续颜色的定位。
#    - D9 的可见性依赖“同/反对角线的填充掩码不同”，因此需要新的 `_diagonal_fill_mask(...)`，不能继续用原 `diamond`。
# d) 放置合法性
#    - generator 保证每个有效 marker_color 的非图例像素恰好是两个 4 邻接连通分量，且每个分量都是连续直线。
#    - 水平型要求 `bbox_width >= seg_len + 1`，垂直型要求 `bbox_height >= seg_len + 1`，否则无法从“贴哪一端”恢复对角线。
#    - 所有线段都避开 legend 保护走廊且彼此不重叠；distractor 使用非图例颜色，避免被 solver 误认成真实 entry。
# e) 视觉可观察性
#    - 解题者能直接看到输入是“两条边上的短横/短竖”，而不是角标，D2 的变化是肉眼可见的。
#    - 输出里同一个 bbox 只会涂出偏某条对角线的一片带状/楔形区域，不再是对称到难以区分的整块矩形，D9 可见。
#    - 标记在输出里消失，至少最上层那个真实 entry 会留下“彩色填充仍在、线段变成背景”的稳定证据，D5 可见。
# f) 可视化整洁度
#    - 线段只占边的一小段，不画整条边，bbox 仍然隐式存在，图面不会被大面积描边。
#    - 允许不同 bbox 的填充区域重叠，但通过 bottom->top 顺序与 future-marker 保护维持可读性。
# g) 变化的可推断性
#    - 颜色变化仍完全由左下图例 `(marker_color, fill_color)` 给出；不依赖颜色编号大小或排序。
#    - generator 额外量化两种退化风险：
#      1. `same_vs_opp_delta`: 若改回“对面对角线填充”，输出必须明显不同。
#      2. `erase_vs_keep_delta`: 若改回“保留线段标记”，输出必须明显不同。
#    - 只有当这两个指标都达标时样本才被接受，避免生成看起来像融合版、实则行为退化回旧规则的输入。


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


def _build_legend_mask(
    g: np.ndarray,
    legend_c: int,
    entries: list[tuple[int, int, int]],
) -> np.ndarray:
    h, w = g.shape
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


def _extract_components(pts: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Return 4-neighborhood connected components of `pts`."""
    remaining = set(pts)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        remaining.remove(start)
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)
        comps.append(comp)
    return comps


def _classify_segment(comp: list[tuple[int, int]]) -> dict[str, int | str]:
    """Parse one connected component as a contiguous horizontal or vertical segment."""
    rs = sorted({r for r, _ in comp})
    cs = sorted({c for _, c in comp})
    n = len(comp)
    assert 3 <= n <= 4, "marker segment length must be 3 or 4"
    if len(rs) == 1:
        c0, c1 = min(cs), max(cs)
        assert n == c1 - c0 + 1, "horizontal segment must be contiguous"
        return {"orientation": "H", "row": rs[0], "c0": c0, "c1": c1, "length": n}
    if len(cs) == 1:
        r0, r1 = min(rs), max(rs)
        assert n == r1 - r0 + 1, "vertical segment must be contiguous"
        return {"orientation": "V", "col": cs[0], "r0": r0, "r1": r1, "length": n}
    raise AssertionError("marker component must be a straight segment")


def _infer_marker_diag(
    marker_pts: set[tuple[int, int]],
) -> tuple[int, int, int, int, str]:
    """
    Decode the bbox and anchor diagonal from two short edge segments.

    Returns:
      (rmin, rmax, cmin, cmax, diag)
    where diag is either 'TLBR' or 'TRBL'.
    """
    comps = _extract_components(marker_pts)
    assert len(comps) == 2, "each marker color must form exactly two edge segments"
    seg1 = _classify_segment(comps[0])
    seg2 = _classify_segment(comps[1])
    assert seg1["orientation"] == seg2["orientation"], "paired segments must share orientation"

    if seg1["orientation"] == "H":
        top, bottom = sorted((seg1, seg2), key=lambda s: int(s["row"]))
        rmin = int(top["row"])
        rmax = int(bottom["row"])
        assert rmin < rmax, "horizontal markers must occupy different rows"
        cmin = min(int(top["c0"]), int(bottom["c0"]))
        cmax = max(int(top["c1"]), int(bottom["c1"]))

        top_left = int(top["c0"]) == cmin and int(top["c1"]) < cmax
        top_right = int(top["c1"]) == cmax and int(top["c0"]) > cmin
        bottom_left = int(bottom["c0"]) == cmin and int(bottom["c1"]) < cmax
        bottom_right = int(bottom["c1"]) == cmax and int(bottom["c0"]) > cmin
        assert (top_left or top_right) and not (top_left and top_right), (
            "top segment must touch exactly one side"
        )
        assert (bottom_left or bottom_right) and not (bottom_left and bottom_right), (
            "bottom segment must touch exactly one side"
        )

        if top_left and bottom_right:
            return rmin, rmax, cmin, cmax, "TLBR"
        if top_right and bottom_left:
            return rmin, rmax, cmin, cmax, "TRBL"
        raise AssertionError("horizontal segment endpoints do not encode a valid diagonal")

    left, right = sorted((seg1, seg2), key=lambda s: int(s["col"]))
    cmin = int(left["col"])
    cmax = int(right["col"])
    assert cmin < cmax, "vertical markers must occupy different columns"
    rmin = min(int(left["r0"]), int(right["r0"]))
    rmax = max(int(left["r1"]), int(right["r1"]))

    left_top = int(left["r0"]) == rmin and int(left["r1"]) < rmax
    left_bottom = int(left["r1"]) == rmax and int(left["r0"]) > rmin
    right_top = int(right["r0"]) == rmin and int(right["r1"]) < rmax
    right_bottom = int(right["r1"]) == rmax and int(right["r0"]) > rmin
    assert (left_top or left_bottom) and not (left_top and left_bottom), (
        "left segment must touch exactly one side"
    )
    assert (right_top or right_bottom) and not (right_top and right_bottom), (
        "right segment must touch exactly one side"
    )

    if left_top and right_bottom:
        return rmin, rmax, cmin, cmax, "TLBR"
    if left_bottom and right_top:
        return rmin, rmax, cmin, cmax, "TRBL"
    raise AssertionError("vertical segment endpoints do not encode a valid diagonal")


def _opposite_diag(diag: str) -> str:
    return "TRBL" if diag == "TLBR" else "TLBR"


def _diagonal_fill_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    diag: str,
) -> np.ndarray:
    """
    Fill the subset of the bbox that is at least as close to `diag` as to the other diagonal.

    Distances are compared in integer form so TLBR and TRBL become visibly different complementary
    regions up to tie cells.
    """
    box_h = r1 - r0
    box_w = c1 - c0
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        rr = r - r0
        for c in range(c0, c1 + 1):
            cc = c - c0
            dist_tlbr = abs(rr * box_w - cc * box_h)
            dist_trbl = abs(rr * box_w + cc * box_h - box_h * box_w)
            if diag == "TLBR":
                m[r, c] = dist_tlbr <= dist_trbl
            else:
                m[r, c] = dist_trbl <= dist_tlbr
    return m


def _bbox_mask(shape: tuple[int, int], r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[r0 : r1 + 1, c0 : c1 + 1] = True
    return m


def _solve(grid: np.ndarray, fill_same_diagonal: bool, erase_markers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask(g, legend_c, entries)
    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]
    erase_mask = np.zeros_like(g, dtype=bool)

    for i, (_, marker, fill) in enumerate(entries):
        marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(marker_mask)))
        if not marker_pts:
            continue

        rmin, rmax, cmin, cmax, anchor_diag = _infer_marker_diag(marker_pts)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        if not erase_markers:
            protect |= marker_mask

        fill_diag = anchor_diag if fill_same_diagonal else _opposite_diag(anchor_diag)
        region = _diagonal_fill_mask(h, w, rmin, rmax, cmin, cmax, fill_diag)
        out[region & (~protect)] = int(fill)
        if erase_markers:
            erase_mask |= marker_mask

    if erase_markers:
        out[erase_mask] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a bottom-left color legend to paint diagonal-oriented regions from paired edge segments, then remove the guides.

    Input structure:
    - The grid has one dominant background color and a vertical 2-column legend near the lower-left corner.
    - Each legend row stores `(marker_color, fill_color)` and is processed from bottom to top.
    - Away from the legend, an active marker color appears as exactly two disjoint 3-4 pixel straight segments on opposite edges of one hidden axis-aligned rectangle.

    Rule:
    - For each legend entry, gather all non-legend pixels of that marker color and decode the rectangle from the two segments.
    - The side each segment clings to determines whether the marked diagonal is TL-BR or TR-BL.
    - Fill the cells of that same diagonal orientation with the legend's fill color, while still protecting the legend and all marker pixels that belong to later legend entries.
    - After all fills are complete, erase every processed marker segment pixel back to the background color.

    Key insight: The short line segments are temporary edge clues, not permanent output; they reveal both the rectangle and which diagonal-oriented half to paint, then they disappear.
    """
    return _solve(grid, fill_same_diagonal=True, erase_markers=True)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3

# ── CFG: segment-pair bbox shape ───────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
ORIENTATION_WEIGHTS = [0.55, 0.45]  # [horizontal, vertical]
SHAPE_WEIGHTS = [0.8, 0.2]          # [roomy, thin]
TLBR_PROB = 0.5
SEG_LEN_RANGE = (3, 4)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: variant observability ─────────────────────────────────────
MIN_SAME_VS_OPP_DELTA = 4
MIN_ERASE_VS_KEEP_DELTA = 4

# ── CFG: retry budgets ─────────────────────────────────────────────
PAIR_RETRIES = 160
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300


def _segment_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    orientation: str,
    diag: str,
    seg_len: int,
) -> list[tuple[int, int]]:
    """Return the cells for a 2-segment marker pair."""
    cells: list[tuple[int, int]] = []
    if orientation == "H":
        if diag == "TLBR":
            top_c0 = cmin
            bottom_c0 = cmax - seg_len + 1
        else:
            top_c0 = cmax - seg_len + 1
            bottom_c0 = cmin
        for c in range(top_c0, top_c0 + seg_len):
            cells.append((rmin, c))
        for c in range(bottom_c0, bottom_c0 + seg_len):
            cells.append((rmax, c))
        return cells

    if diag == "TLBR":
        left_r0 = rmin
        right_r0 = rmax - seg_len + 1
    else:
        left_r0 = rmax - seg_len + 1
        right_r0 = rmin
    for r in range(left_r0, left_r0 + seg_len):
        cells.append((r, cmin))
    for r in range(right_r0, right_r0 + seg_len):
        cells.append((r, cmax))
    return cells


def _pick_pair_size(H: int, W: int, orientation: str, seg_len: int) -> tuple[int, int] | None:
    """Pick a bbox size that preserves endpoint information for the chosen segment orientation."""
    shape = random.choices(["roomy", "thin"], weights=SHAPE_WEIGHTS)[0]
    if orientation == "H":
        bw_lo = seg_len + 1
        bw_hi = min(BBOX_W_MAX, W)
        if bw_lo > bw_hi:
            return None
        if shape == "thin":
            bh = 2
        else:
            if H < 3:
                return None
            bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(bw_lo, bw_hi)
        return bh, bw

    bh_lo = seg_len + 1
    bh_hi = min(BBOX_H_MAX, H)
    if bh_lo > bh_hi:
        return None
    if shape == "thin":
        bw = 2
    else:
        if W < 3:
            return None
        bw = random.randint(3, min(BBOX_W_MAX, W))
    bh = random.randint(bh_lo, bh_hi)
    return bh, bw


def _try_place_segment_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    """Try to place a paired short-line marker within `retries` attempts."""
    for _ in range(retries):
        orientation = random.choices(["H", "V"], weights=ORIENTATION_WEIGHTS)[0]
        seg_len = random.randint(*SEG_LEN_RANGE)
        size = _pick_pair_size(H, W, orientation, seg_len)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _segment_pair_cells(
            rmin,
            cmin,
            rmin + bh - 1,
            cmin + bw - 1,
            orientation,
            diag,
            seg_len,
        )
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
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
        fill = _pick_non_bg_color(bg)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
    """Protect the legend scan corridor from extra non-bg pixels."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _pick_non_bg_color(bg: int) -> int:
    return random.choice([c for c in range(10) if c != bg])


def _force_visible_top_entry(
    grid: np.ndarray,
    legend_rows: list[tuple[int, int, int]],
    legend_c: int,
    types: list[str],
    bg: int,
) -> None:
    """
    Guarantee one stable visible fusion example:
    the top legend entry is a real segment pair processed last, and its fill is non-bg.
    """
    top = len(types) - 1
    types[top] = "pair"
    r, marker, fill = legend_rows[top]
    if fill == bg:
        fill = _pick_non_bg_color(bg)
        legend_rows[top] = (r, marker, fill)
        grid[r, legend_c + 1] = fill


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _build_legend_mask(g, legend_c, entries)

    same_erase = _solve(g, fill_same_diagonal=True, erase_markers=True)
    opp_erase = _solve(g, fill_same_diagonal=False, erase_markers=True)
    same_keep = _solve(g, fill_same_diagonal=True, erase_markers=False)

    pair_count = 0
    top_same_vs_opp_delta = 0
    top_erase_vs_keep_delta = 0

    top_idx = len(entries) - 1
    top_bbox = np.zeros_like(g, dtype=bool)

    for i, (_, marker, _) in enumerate(entries):
        marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(marker_mask)))
        if not marker_pts:
            continue
        rmin, rmax, cmin, cmax, _ = _infer_marker_diag(marker_pts)
        pair_count += 1
        if i == top_idx:
            top_bbox = _bbox_mask(g.shape, rmin, rmax, cmin, cmax)

    same_vs_opp_delta = int(np.count_nonzero(same_erase != opp_erase))
    erase_vs_keep_delta = int(np.count_nonzero(same_erase != same_keep))
    if np.any(top_bbox):
        top_same_vs_opp_delta = int(np.count_nonzero((same_erase != opp_erase) & top_bbox))
        top_erase_vs_keep_delta = int(np.count_nonzero((same_erase != same_keep) & top_bbox))

    return {
        "pair_count": pair_count,
        "same_vs_opp_delta": same_vs_opp_delta,
        "erase_vs_keep_delta": erase_vs_keep_delta,
        "top_same_vs_opp_delta": top_same_vs_opp_delta,
        "top_erase_vs_keep_delta": top_erase_vs_keep_delta,
    }


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

    types = ["pair"] * len(legend_rows)
    idx_pool = list(range(len(legend_rows)))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    _force_visible_top_entry(grid, legend_rows, legend_c, types, bg)

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
        if not _try_place_segment_pair(H, W, marker, PAIR_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_segment_pair(
                H,
                W,
                random.choice(avail),
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    metrics = _variant_metrics(grid)
    if metrics["pair_count"] == 0:
        return None
    if metrics["same_vs_opp_delta"] < MIN_SAME_VS_OPP_DELTA:
        return None
    if metrics["erase_vs_keep_delta"] < MIN_ERASE_VS_KEEP_DELTA:
        return None
    if metrics["top_same_vs_opp_delta"] == 0:
        return None
    if metrics["top_erase_vs_keep_delta"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
