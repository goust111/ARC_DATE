# Variant of 9385bd28
# Dimension: D3 — 目标对象：填充区域的形状 (dst)
# Change: 标记对角线的对边对角线上所有最短路径像素（菱形/矩形填充区域）... → 仅填充矩形边框
#
# Design analysis:
#   variant_spec:
#     preserved_input:
#       - 底左侧仍有 2 列宽的竖直图例，每行给出 (marker_color, draw_color)。
#       - 非图例区域仍放置与图例颜色对应的 L 形对角角标，少量条目仍可为空操作或 2x2 实心块。
#       - 图例读取顺序、未来 marker 保护规则、2x2 block 直接改色规则都保持原样。
#     changed_output:
#       - L 形对角角标围成一个轴对齐 bbox 后，原题会把整个 bbox 内部都染成图例颜色。
#       - 变体只在该 bbox 的四条边上着色，bbox 内部保持原来的颜色，因此形成空心矩形。
#     behavior_trace:
#       - 例：若某个颜色的 L 对角角标包围 rows 2..6, cols 4..8。
#       - 原题输出会尝试改写这 5x5 区域的全部 25 个像素（除被保护的 marker/legend 外）。
#       - 本变体只改写第 2/6 行与第 4/8 列；内部 rows 3..5, cols 5..7 保持不变。
#       - 差异正好发生在“构造目标 mask”这一步，输入解析与配色映射流程不需要重写。
#     complexity_calibration:
#       - 仍保留多图例项、处理顺序、distractor、noop、2x2 block，避免任务被简化成单个矩形描边。
#       - 但为了让 D3 肉眼可见，generator 额外要求至少出现一个 bbox 高宽都 >= 3 且 draw_color != bg 的 L-pair；
#         否则 2 行高/2 列宽边框会退化成实心条，或与背景同色导致空心效果不可观察。
#     implicit_assumption_review:
#       - `_find_legend` 依赖“最左下非背景是图例起点、图例宽 2 列”，变体未改动这点。
#       - `_l_anchor_present` 依赖 marker 仍是 3 像素 L 角标；变体没有改变 marker 形状，所以判定逻辑可复用。
#       - 原 `diamond_mask` 依赖“目标区域 = 整个 bbox 内部”；该前提失效，必须替换成边框 mask。
#       - `protect |= current marker` 在原题用于保留角标可见；变体同样需要，否则描边会把角标抹成边框色。
#       - 2x2 block 在原题被视为“无 bbox 需要填充”，这一点继续保留，否则会凭空引入新的对象语义。
#     placement_legality:
#       - marker 像素仍不得越界、不得覆盖 legend、不得与已放置 marker 冲突。
#       - 生成器继续只禁止 marker 像素重叠，允许不同矩形边框在输出阶段互相覆盖，由图例顺序解决冲突。
#       - 为避免锚点歧义，L-pair bbox 仍排除 (2,2)/(2,3)/(3,2) 这些会让对角判分打平的尺寸。
#     visual_observability:
#       - 至少一个可见空心矩形保证解题者能从少量样例看到“实心填充变描边”的核心变化。
#       - 颜色映射仍完全由邻接图例给出，不依赖颜色 id 排序或任何不可见属性。
#     cleanliness:
#       - 不额外增加新对象种类；变化只体现在矩形内部是否保留背景/既有颜色，视觉对比直接且整洁。
#     new_solver_assumptions:
#       - 每个需要描边的 legend marker 仍对应单个合法 bbox。
#       - 若某个条目是 L-pair，则其 bbox 至少有一条明确的 marker 对角线；generator 用尺寸约束保证这一点。

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


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
    assert start_r is not None and start_c is not None and start_c < w - 1, \
        "legend must exist at bottom-left with at least 2 columns"

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


def _l_anchor_present(pts: set[tuple[int, int]], r: int, c: int, arm1: tuple[int, int], arm2: tuple[int, int]) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _rect_border_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Mask for the perimeter of the axis-aligned rectangle [r0..r1] x [c0..c1]."""
    m = np.zeros((h, w), dtype=bool)
    m[r0, c0:c1 + 1] = True
    m[r1, c0:c1 + 1] = True
    m[r0:r1 + 1, c0] = True
    m[r0:r1 + 1, c1] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw hollow rectangles between diagonal marker pairs using a color legend.

    Input structure:
    - The background grid still contains a bottom-left vertical legend, with one marker-color / draw-color pair per row.
    - Away from the legend, each active marker color appears either as two L-shaped corner markers on one bbox diagonal or as a solid 2x2 block.
    - The two L corners identify the corners of one axis-aligned rectangle; distractor colors may also appear but are absent from the legend.

    Rule:
    - Read legend entries from bottom to top and gather all non-legend pixels of that marker color.
    - If the marker is a solid 2x2 block, recolor that block to the legend's draw color.
    - Otherwise, verify which bbox diagonal carries the L anchors, then use the full bbox of those markers as the rectangle to draw.
    - Color only the rectangle perimeter, while preserving legend cells and marker pixels belonging to the current entry or any later legend entry.

    Key insight: The L markers still specify a rectangle exactly as before, but the target region is no longer the filled interior; only the four bbox edges are painted, so the rectangle stays hollow.
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

    # Process legend entries bottom -> top; earlier entries must not destroy later marker pixels.
    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Protect legend cells and all marker pixels for entries that appear later (higher) in the legend.
        future_markers = set(marker_order[i + 1:])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        # Solid 2x2 markers do not define a rectangle to outline; they are recolored in place.
        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = eff_fill
            continue
        else:
            protect |= (g == marker) & (~legend_mask)

        if rmin == rmax or cmin == cmax:
            continue

        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)

        tl_anchor = _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0))
        tr_anchor = _l_anchor_present(marker_pts, tr[0], tr[1], (0, -1), (1, 0))
        bl_anchor = _l_anchor_present(marker_pts, bl[0], bl[1], (0, 1), (-1, 0))
        br_anchor = _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0))

        score_tlbr = int(tl_anchor) + int(br_anchor)
        score_trbl = int(tr_anchor) + int(bl_anchor)
        assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
        assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"

        border = _rect_border_mask(h, w, rmin, rmax, cmin, cmax)
        out[border & (~protect)] = eff_fill

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry (marker with no pixels)
BLOCK_PROB = 0.25                # prob of one 2x2 block entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)
VISIBLE_BORDER_MIN = 1           # require at least one visibly hollow rectangle

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25           # prob of adding a non-legend L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for an L-pair
DISTRACTOR_RETRIES = 60          # placement attempts for distractor
GEN_RETRIES = 300                # outer generate_input retry budget

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids an L-anchor score tie.

    Fails for (2,2), (2,3), (3,2). Safe: h>=3 and w>=3, or h==2 and w>=4, or h>=4 and w==2.
    """
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif shape_choice == "h2_w_ge4":
        bh = 2
        w_hi = min(BBOX_W_MAX, W)
        if w_hi < 4:
            return None
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _try_place_l_pair(H: int, W: int, color: int, retries: int,
                      can_place, do_place) -> tuple[int, int] | None:
    """Try to place an L-pair of `color` on the grid within `retries` attempts.

    Returns the placed bbox size (height, width) on success, else None.
    """
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return bh, bw
    return None


def _build_legend(H, W, bg):
    """Create legend entries and the corresponding grid/mask arrays.

    Returns (grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors) or None on failure.
    """
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
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity.

    Forbids non-bg placement in rows below legend_bot and in columns <= legend_c
    above the legend (except legend marker cells themselves).
    """
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1:, :] = True
    forbidden[:legend_bot + 1, :legend_c + 1] = True
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

    # Assign a marker type per legend entry.
    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    visible_hollow_count = 0

    # Place markers for each legend entry.
    for i, (_, m, fill) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            size = _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place)
            if size is None:
                return None
            bh, bw = size
            if bh >= 3 and bw >= 3 and fill != bg:
                visible_hollow_count += 1

    # Optional distractor L-pair in a non-legend color.
    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail),
                              DISTRACTOR_RETRIES, can_place, do_place)

    if visible_hollow_count < VISIBLE_BORDER_MIN:
        return None

    # Ensure bg remains the most common color (so get_bg picks it).
    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    # Oracle check: solver must run without errors.
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
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
