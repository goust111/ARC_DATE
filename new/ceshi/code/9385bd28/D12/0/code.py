# Variant of 9385bd28
# Dimension: D12 — 2x2 方块标记的处理方式 (meta)
# Change: 0-1 个图例条目的标记为 2x2 实心方块（不进行矩形填充，仅重新上色为填充色）... → 禁止 2x2 方块标记
#
# Design analysis:
#   concretization:
#     - 输入仍然是一个单背景色网格，左下角保留纵向 2 列图例；每行是 (marker_color, fill_color)。
#     - 每个真正出现的 marker_color 现在只能对应一个 L 形角对：两个 3 像素 L 角坐落在同一个轴对齐 bbox 的对角上。
#     - 原题里若某个颜色在非图例区恰好形成 2x2 实心块，solver 会只把这 4 格改成 fill_color 并跳过矩形填充。
#       本变体中该情况被视为非法输入；同一个图例色若出现，就必须是一个可识别的 L 角对，否则不应由 generator 产出。
#     - 具体追踪:
#       original: marker_pts = {(r,c),(r,c+1),(r+1,c),(r+1,c+1)} 时，main() 走 block 分支，只重涂这 4 格。
#       variant: 同一 legend 条目必须改为 6 格 L 角对，例如 TL-BR 对角的两角；solver 识别对角后，对另一条对角张成的矩形着色。
#       差异点就是“是否存在 block 特判”以及 generator 是否会产出 block。
#   complexity_calibration:
#     - 只移除一种 marker 形态，不改图例读取、颜色映射、处理顺序、可选 noop、可选 distractor。
#     - 这样不会把任务改成另一题，但又能让解题者稳定观察到“所有有效标记都是同一类 L 角对”。
#     - 保留两种对角方向和多种 bbox 尺寸，避免样例因过度单一而退化成模板匹配。
#   implicit_assumptions_review:
#     - _find_legend 只依赖左下角图例结构，与 D12 无关，可原样保留。
#     - 原 main() 中 `if rmax-rmin==1 and cmax-cmin==1 and len(marker_pts)==4` 依赖“允许 2x2 block”这一旧前提；
#       变体里这个前提失效，因此该分支必须删除，不能再把 2x2 当作合法对象。
#     - 原本的 protect 逻辑同时服务于 block 和 L-pair；删除 block 后，当前 marker 一律受保护即可，保证角标仍可见。
#     - 角点打分逻辑在细长 bbox 上会出现“额外角也像 L”的情况，所以仅凭分数不足以完全定义形态；
#       这里补了一个精确校验：marker_pts 必须与某个 bbox+diag 生成出来的 L 角对单元集合完全一致。
#   placement_legality:
#     - generator 仍然禁止覆盖图例区和已占用单元，所有 L 角对都必须完整落在网格内。
#     - bbox 继续沿用原合法尺寸约束，避免分数平局的模糊情况。
#     - 每个非 noop 图例项放置后都由 `_infer_l_pair_diag` 反查，确保它真的是一个合法 L 角对而不是别的 6 格形状。
#   visual_observability:
#     - 在输入图上，人类将只看到图例 + 若干同构的 L 角对，不再出现孤立的 2x2 实心块分支。
#     - 这个变化不依赖颜色 id，也不依赖不可见排序；它直接体现在像素形状类别的消失上。
#   cleanliness:
#     - 不新增重叠或额外装饰，保持原来的空间疏离规则。
#     - distractor 仍然只用非图例颜色，避免和真实 marker 混淆。
#   color_inference:
#     - 颜色规则保持原题：图例右列给出填充色，左列给出 marker 色。
#     - 变体不引入新的颜色计算，因此不会制造额外的不可推断映射。
#   new_main_assumptions:
#     - 非 noop 的 legend marker 在非图例区要么完全不存在，要么恰好是一个 6 格 L 角对。
#     - generator 通过只放置 L 角对、禁用 block 类型、并在生成后做形态校验来保证这些前提成立。

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


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
    where one thin L can accidentally satisfy an extra corner test.
    """
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _infer_l_pair_diag(marker_pts: set[tuple[int, int]]) -> str | None:
    """Return 'TLBR' or 'TRBL' when marker_pts are exactly one legal L-pair, else None."""
    if len(marker_pts) != 6:
        return None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    if rmin == rmax or cmin == cmax:
        return None

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
    if score_tlbr == score_trbl or (score_tlbr == 0 and score_trbl == 0):
        return None

    diag = "TLBR" if score_tlbr > score_trbl else "TRBL"
    expected = set(_l_pair_cells(rmin, cmin, rmax, cmax, diag))
    return diag if marker_pts == expected else None


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
    Recolor the rectangles implied by legend-coded L-corner pairs.

    Input structure:
    - The grid has one dominant background color and a vertical 2-column legend at the lower-left.
    - Each legend row stores a marker color in the left column and its fill color in the next column.
    - Away from the legend, every realized marker color appears only as one pair of 3-cell L corners on opposite bbox corners.

    Rule:
    - Read the legend from bottom to top and collect the non-legend pixels of that marker color.
    - If a legend color has no scattered pixels, it is a noop entry; otherwise those pixels must form exactly one L-pair.
    - The diagonal carrying the two L anchors identifies the marker orientation, and the opposite diagonal determines the rectangle to fill.
    - Paint that rectangle with the legend's fill color while preserving the legend itself, the current marker pixels, and all later legend markers.

    Key insight:
    - The marker pixels do not show the rectangle directly; they show which diagonal is occupied by the L anchors.
      Once that anchor diagonal is known, the fill region is the bbox spanned by the other diagonal.
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

    # Earlier fills must not erase marker pixels belonging to later legend rows.
    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if not marker_pts:
            continue

        diag = _infer_l_pair_diag(marker_pts)
        assert diag is not None, "variant requires each realized marker to be one L-pair, not a 2x2 block"

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)
        if diag == "TLBR":
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25           # prob of adding a non-legend L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
LPAIR_RETRIES = 150              # placement attempts for an L-pair
DISTRACTOR_RETRIES = 60          # placement attempts for distractor
GEN_RETRIES = 300                # outer generate_input retry budget


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


def _build_legend(H, W, bg):
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
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _grid_matches_variant(grid: np.ndarray, legend_rows, legend_mask: np.ndarray) -> bool:
    """Variant-specific generator guard: every realized legend marker must be one L-pair."""
    for _, marker, _ in legend_rows:
        marker_pts = set(map(tuple, np.argwhere((grid == marker) & (~legend_mask))))
        if not marker_pts:
            continue
        if _infer_l_pair_diag(marker_pts) is None:
            return False
    return True


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

    # Every non-noop legend entry is forced to be an L-pair in this variant.
    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"

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
        if types[i] == "noop":
            continue
        if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    if not _grid_matches_variant(grid, legend_rows, legend_mask):
        return None

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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
