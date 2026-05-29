# Variant of 9385bd28
# Dimension: D2 — 源对象：标记的形状结构 (src)
# Change: L 形 3 像素角标记，位于外接矩形的对角两端（共 6 像素）；另有 2x2 实心方块变体... → 线段标记（水平/垂直短线）

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg

# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 每个有效 marker_color 不再对应两个 L 角，而是对应两条同色短线段。
#    - 两条线段总是同向、互相分离、位于同一个外接矩形的相对两边：
#      * 水平型:
#          TLBR 标记对: 顶边左端一条水平线 + 底边右端一条水平线
#          TRBL 标记对: 顶边右端一条水平线 + 底边左端一条水平线
#      * 垂直型:
#          TLBR 标记对: 左边上端一条竖线 + 右边下端一条竖线
#          TRBL 标记对: 左边下端一条竖线 + 右边上端一条竖线
#    - 线段长度固定为 3 或 4。输出规则保持原题核心不变:
#      仍然用 legend 给出的 fill_color 填充该 bbox 内“另一条对角线”的 diamond 区域。
#    - 小例子:
#      原题若 bbox 的 TL/BR 角上各有一个 L，则 solver 识别“标记对角线 = TLBR”，然后填 TR-BL diamond。
#      变体中同一个 bbox 改成“顶边左端水平线 + 底边右端水平线”，solver 不再找角，而是:
#      1. 找到两段水平线所在的顶/底行
#      2. 从左贴边的那段得到 cmin，从右贴边的那段得到 cmax
#      3. 推断标记对角线仍为 TLBR
#      4. 继续填 TR-BL diamond
#      差异在于推理入口从角点检测变成边段检测，输出几何仍然一致。
# b) 复杂度校准
#    - 只保留 4 类可见布局: 水平/垂直 × TLBR/TRBL。每类都在多个样本中重复出现，避免规则过散。
#    - 线段长度只取 3 或 4，bbox 仍有大中小变化，因此和原题相比视觉差异明显但不会过度复杂。
#    - 放弃“边中点”这一自由度: 两条短线若都不接触端点，会导致 bbox 左右或上下边界无法唯一恢复。
# c) 对原 main() 的隐式假设审查
#    - 原代码 `_l_anchor_present(...)` 假设 marker 像素恰好落在 bbox 四角附近；线段变体下该前提失效。
#    - 原代码用 `score_tlbr` / `score_trbl` 比较两条对角线上的 L 角数量；线段变体没有角，必须改为判断
#      “哪一条边段贴左/上端，哪一条边段贴右/下端”。
#    - 原代码把 2x2 block 当作特殊分支；新变体不再生成该结构，solver 也不再依赖该分支。
#    - 原代码默认 marker 的整体 bbox 已经是目标矩形；新变体仍保留这一点，但要求非 legend 像素必须恰好分成
#      两个 4 邻接连通分量，且每个分量是一条连续直线。
# d) 放置合法性
#    - generator 保证两条线段不越界、不重叠、不碰 legend 保护区。
#    - 对水平型要求 `bbox_width >= seg_len + 1`，否则左右贴边会退化成整条顶/底边并失去方向信息。
#    - 对垂直型要求 `bbox_height >= seg_len + 1`，理由同上。
#    - solver 的新假设都由 generator 构造保证: 恰好两个分量、同向、长度 3-4、分别贴相对两边的相反端。
# e) 视觉可观察性
#    - 解题者直接能看到“原本像角，现在像边上的短横/短竖”，且水平型与垂直型会同时出现。
#    - 颜色变化仍完全由左下 legend 提示；没有使用颜色 id 排序或其他不可见信号。
# f) 可视化整洁度
#    - 输入中只画两条短线而不画整条边，bbox 轮廓不会被完全描粗，网格仍然清爽。
#    - 继续禁止 marker 像素互相重叠；允许 bbox 之间套叠，但后续 legend 颜色仍通过保护规则保持可辨识。
# g) 颜色变化可推断性
#    - 颜色映射仍是一列明确的 `(marker_color, fill_color)` 图例，结构线索与原题一致。
#    - fill_color 可以复用其他颜色，但决定因素始终可见且稳定: “看 legend，用同色线段找矩形，再填另一条对角线”。


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


def _infer_fill_diagonal(marker_pts: set[tuple[int, int]]) -> tuple[int, int, int, int, tuple[int, int], tuple[int, int]]:
    """
    Decode the bbox and the fill diagonal from two short edge segments.

    Returns:
      (rmin, rmax, cmin, cmax, a, b)
    where a/b are the two corners defining the diamond to fill.
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
        assert (top_left or top_right) and not (top_left and top_right), "top segment must touch exactly one side"
        assert (bottom_left or bottom_right) and not (bottom_left and bottom_right), "bottom segment must touch exactly one side"

        if top_left and bottom_right:
            a, b = (rmin, cmax), (rmax, cmin)
        elif top_right and bottom_left:
            a, b = (rmin, cmin), (rmax, cmax)
        else:
            raise AssertionError("horizontal segment endpoints do not encode a valid diagonal")
        return rmin, rmax, cmin, cmax, a, b

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
    assert (left_top or left_bottom) and not (left_top and left_bottom), "left segment must touch exactly one side"
    assert (right_top or right_bottom) and not (right_top and right_bottom), "right segment must touch exactly one side"

    if left_top and right_bottom:
        a, b = (rmin, cmax), (rmax, cmin)
    elif left_bottom and right_top:
        a, b = (rmin, cmin), (rmax, cmax)
    else:
        raise AssertionError("vertical segment endpoints do not encode a valid diagonal")
    return rmin, rmax, cmin, cmax, a, b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles indicated by paired edge segments using a color legend.

    Input structure:
    - The grid has one dominant background color plus a vertical 2-column legend near the lower-left corner.
    - Each legend row stores `(marker_color, fill_color)`, read from bottom to top.
    - Away from the legend, each active marker color appears as two disjoint 3-4 pixel straight segments on opposite edges of a hidden axis-aligned rectangle.

    Rule:
    - Process legend entries from bottom to top and collect all non-legend pixels of that marker color.
    - Decode the rectangle from the two segments: their rows/columns give the touched edges, and the end they cling to identifies which diagonal is marked.
    - Fill the diamond on the opposite diagonal of that rectangle with the legend's fill color.
    - Keep legend cells and all later marker pixels protected so earlier fills do not erase them.

    Key insight: The markers no longer sit in the corners; the rectangle must be reconstructed from two short edge traces, and the unmarked diagonal is what gets filled.
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
        if not marker_pts:
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

# ── CFG: segment-pair bbox shape ───────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
ORIENTATION_WEIGHTS = [0.55, 0.45]   # [horizontal, vertical]
SHAPE_WEIGHTS = [0.8, 0.2]           # [roomy, thin]
TLBR_PROB = 0.8
SEG_LEN_RANGE = (3, 4)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

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
        if bh > H:
            return None
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
    if bw > W:
        return None
    return bh, bw


def _try_place_segment_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
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
        cells = _segment_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, orientation, diag, seg_len)
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
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Protect the legend scan corridor from extra non-bg pixels."""
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

    types = ["pair"] * len(legend_rows)
    idx_pool = list(range(len(legend_rows)))
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
        if not _try_place_segment_pair(H, W, marker, PAIR_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_segment_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
