# Variant of 9385bd28
# Dimension: D5 — 填充变换方式 (transform)
# Change: rectangle_fill — 用图例对应的填充色填充标记对之间的菱形区域；2x2 方块直接重新上色... → 填充后擦除标记
#
# Design analysis
# variant_spec:
#   preserved_structure:
#     - 网格尺寸、左下角竖直图例、L 形对角标记、可选 2x2 方块、可选 distractor 都保持原题风格。
#     - 图例仍按自下而上读取；每行仍是 (marker_color, fill_color)。
#   D5_core_change:
#     - L 形标记不再是输出的一部分，而是“临时施工线”。
#     - solver 先用标记颜色和 L 角朝向确定 bbox 与填充对角线，完成菱形填充后，再把该 marker 的非图例像素擦成背景色。
#     - 2x2 方块不是临时施工线，仍直接整体改成 fill_color，不做擦除。
#   behavior_trace:
#     - 例：某颜色 marker 在 bbox 的 TL 与 BR 两角放置两个 L，说明真正要填的是 TR↔BL 那条对角线形成的菱形。
#     - 原逻辑：`protect |= current marker`，所以菱形填色时避开这 6 个 marker 像素，输出里会保留异色 L 标记。
#     - 新逻辑：不再保护当前 L 标记；先把菱形区域刷成 fill_color，再把这 6 个 marker 像素改回 bg。
#     - 差异：同一个输入下，原题输出会看到“填充区 + 异色角标”，变体输出会看到“填充区 + 背景色缺口”，解题者需理解标记只负责定位。
#   complexity_calibration:
#     - 只引入一个二元变化：L 标记是保留还是擦除；不新增额外排序、颜色规则或对象类型。
#     - 保持 2-5 个 legend entries 与原 bbox 尺寸分布，避免样例过简单。
#     - 保留少量 noop/block/distractor，让解题者仍需区分“会触发擦除的 L-pair”和“不擦除的 2x2 block”。
#   implicit_assumption_audit:
#     - `_find_legend`、`_l_anchor_present`、`_diamond_mask` 不依赖“标记最终保留”，可直接复用。
#     - 原 `protect |= (g == marker)` 明确依赖“当前 marker 需要保留”；在变体里这会阻止 D5 发生，必须删除。
#     - 原 `marker_pts` 从输入 `g` 读取而不是从 `out` 读取，这一点必须保留；否则一旦先擦除再定位，会破坏后续判定。
#     - 原 block 分支把 2x2 直接重着色；该语义与变体说明一致，不能顺手改成擦除。
#   placement_legality:
#     - 仍用 `occupied` + `forbidden` 防止 marker/legend 越界与重叠，保证 bbox 可被 solver 唯一解析。
#     - topmost legend entry 强制为真实 L-pair，且其 fill != bg；这样最后一步一定会留下“填充仍在、标记消失”的可观察证据。
#     - 其余 entries 仍可是 noop/block/L，避免生成策略退化成单一模板。
#   visual_observability:
#     - 最上方 legend entry 最后处理，因此它的擦除结果不会被后续填充覆盖，确保至少一处 D5 差异稳定可见。
#     - 要求该 entry 的 fill != bg，保证输出里不仅是“标记消失”，还会保留可见的彩色填充区域。
#   cleanliness:
#     - 不额外增加重叠层数；保持原题的局部覆盖风格即可。
#     - 擦除只作用于当前 marker 像素，不清空整块 bbox，因此图面仍然整洁且可追踪。
#   color_inference:
#     - 颜色关系仍完全由图例显式给出，不依赖颜色 id 排序或数值运算。
#   new_solver_assumptions_checked_against_generator:
#     - 假设 1：至少存在一个真实 L-pair，generator 强制 top entry 为 L。
#     - 假设 2：至少一个擦除案例在最终输出中可见，generator 强制 top L entry 的 fill != bg。
#     - 假设 3：current marker 像素与 legend 不重叠，generator 的 `occupied` / `forbidden` 已保证。

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


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


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    dr1, dc1 = arm1
    dr2, dc2 = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a bottom-left legend to paint diamonds, then erase the temporary corner guides.

    Input structure:
    - The background is the most common color.
    - A vertical 2-column legend sits near the bottom-left; each row stores (marker color, fill color).
    - Away from the legend, each marker color appears either as two opposite L-corners of one bbox or as a solid 2x2 block.

    Rule:
    - Read legend rows from bottom to top and gather all non-legend pixels of that marker color.
    - For an L-pair, decide which bbox diagonal carries the L anchors; the opposite diagonal defines the diamond to fill.
    - Paint that diamond with the legend fill color while still protecting legend cells and markers that belong to later legend entries.
    - After filling, erase the current L-marker pixels back to background; a solid 2x2 block is the exception and is simply recolored to the fill color.

    Key insight: The L-corners are construction scaffolding rather than permanent output content, so they guide the fill and then disappear.
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
    marker_order = [marker for _, marker, _ in entries]

    # Earlier entries may be overwritten later, but they must not destroy markers that have not been processed yet.
    for i, (_, marker, fill) in enumerate(entries):
        marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(marker_mask)))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            out[marker_mask & (~protect)] = int(fill)
            continue

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
        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)
        out[marker_mask] = bg

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
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300

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
    """True if (h, w) avoids an L-anchor score tie."""
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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
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
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows) -> np.ndarray:
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _pick_non_bg_color(bg: int) -> int:
    return random.choice([c for c in range(10) if c != bg])


def _force_visible_top_l_entry(
    grid: np.ndarray,
    legend_rows: list[tuple[int, int, int]],
    legend_c: int,
    types: list[str],
    bg: int,
) -> None:
    """
    Guarantee at least one visible D5 example:
    the top legend entry is an L-pair processed last, and its fill is non-bg.
    """
    top = len(types) - 1
    types[top] = "L"
    r, m, fill = legend_rows[top]
    if fill == bg:
        fill = _pick_non_bg_color(bg)
        legend_rows[top] = (r, m, fill)
        grid[r, legend_c + 1] = fill


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
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    _force_visible_top_l_entry(grid, legend_rows, legend_c, types, bg)

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

    for i, (_, marker, _) in enumerate(legend_rows):
        kind = types[i]
        if kind == "noop":
            continue
        if kind == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
