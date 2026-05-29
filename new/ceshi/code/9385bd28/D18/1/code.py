# Variant of 9385bd28
# Dimension: D18 — L 形角的朝向判定逻辑 (meta)
# Change: 通过检测哪两个角有 L 形锚点来确定标记对角线方向（TL+BR 得分 vs TR+BL 得分）... → 额外的方向标记像素

"""
design_analysis:
  concrete_variant:
    original_object:
      - 每个 legend 颜色通常对应一个由两个对角 L 角组成的 6 像素 marker 对。
      - 原 solver 通过看哪一条对角线真的放了 L 角，去反推出“另一条对角线”才是要填充的菱形。
    new_object:
      - 保留原来的两个 L 角，但再增加 1 个同色的单像素方向提示点，总计 7 像素。
      - 这个提示点必须落在该对象外接矩形的边上、不是角点、并且与 L 角主体 4 邻接断开，视觉上像独立小点。
      - 若提示点在上边或下边，则填充 TL-BR 这条对角线上的菱形。
      - 若提示点在左边或右边，则填充 TR-BL 这条对角线上的菱形。
    behavior_trace:
      - 例：bbox 为 5x7，L 角真实放在 TL 与 BR。旧规则会据此选择 TR-BL 方向。
      - 新变体里若额外小点放在上边，则应选择 TL-BR 方向；这与旧规则相反。
      - 但原题代码里两条方向分支最终都会把整块 bbox 都填满，方向差异在视觉上不可见。
      - 因此这里做一处必要联动：选中的方向不再对应“整矩形”，而对应该 bbox 中更接近该对角线的那一侧斜向区域。
      - 这样真正变化的仍是“决定哪条方向”，只是把这个元信息转成了人眼可见的输出差异。

  complexity_calibration:
    - 只引入一个新线索维度，不改颜色映射与处理顺序，避免题意漂移。
    - 提示点只分为两类可见位置：水平边(top/bottom) 与垂直边(left/right)，对应两种填充对角线。
    - 为了让人类能从少量样例看出规律，单个网格里通常保留至少 2 个 L-pair，并尽量同时出现两类提示边。
    - L 角本身仍然存在，但不再可靠；生成器会显式制造一部分“L 角方向与真实填充方向冲突”的样本，防止退化回原题。

  implicit_assumption_audit:
    - 原 main() 中 `score_tlbr` / `score_trbl` 假设：哪条对角线承载 L 角，就由它唯一决定输出方向。该假设已失效。
    - 原 `_diamond_mask()` 对轴对齐矩形的两个对角线都会退化成整块矩形；若保留它，D18 变化不可观察，因此必须联动调整填充几何。
    - 原 `_l_pair_valid_bbox()` 的 tie-avoidance 仍然有价值，因为 solver 仍需识别两个角上的 L 形模板；因此保留安全 bbox 家族。
    - 原 `marker_pts` 默认只含 6 个 L 像素；现在需要允许 7 像素，并从中剥离出 1 个边缘提示点。
    - 原生成器允许很瘦的 bbox；新规则若要让边缘单像素与 L 角断开，水平提示至少需要宽 >= 5，垂直提示至少需要高 >= 5。

  placement_legality:
    - 提示点只从 bbox 边界内部位置挑选，且不能落在 L 角像素上，也不能与任一 L 角像素 4 邻接。
    - 水平提示(top/bottom) 只在 `width >= 5` 的 bbox 上生成；垂直提示(left/right) 只在 `height >= 5` 的 bbox 上生成。
    - 继续禁止对象像素彼此重叠、禁止侵入 legend 保护区，并保留 oracle 校验 `main(input)`。

  visual_observability:
    - 方向信息完全来自可见几何位置，而非颜色 id、排序或隐藏计数。
    - 额外小点与主 L 角断开，避免被误看成更大的折线。
    - top/bottom 与 left/right 这两类位置在输入中肉眼可直接比较；输出则对应两种互补的斜向填充半区，2-4 个样例可归纳。

  cleanliness_and_color_inference:
    - 颜色规则仍由 legend 控制，不增加新的颜色变换自由度。
    - 可选 distractor 也采用同样的 “L-pair + 单点” 造型，避免非 legend 物体在视觉上成为明显异类。

  new_solver_assumptions_and_generator_guarantees:
    - solver 假设每个真正的 L-pair marker 恰有两个可识别 L 角和一个剩余边缘单点。
    - generator 保证 block/noop 不会吞掉全部 L-pair，保证方向点满足边界与断开条件，并拒绝任何新旧规则输出完全一致的退化样本。
"""

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
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _directional_fill_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    diag: str,
) -> np.ndarray:
    """
    Fill the slanted half of the bbox that is closer to the chosen diagonal.

    For each cell in the bbox, compare its scaled distance to the two diagonals:
      TLBR metric: |(r-r0)*(w-1) - (c-c0)*(h-1)|
      TRBL metric: |(r-r0)*(w-1) - (c1-c)*(h-1)|

    Cells go to the chosen direction when that direction's metric is no larger than the other one.
    The resulting masks are visibly different even though the bbox itself is unchanged.
    """
    m = np.zeros((h, w), dtype=bool)
    hh = r1 - r0
    ww = c1 - c0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d_tlbr = abs((r - r0) * ww - (c - c0) * hh)
            d_trbl = abs((r - r0) * ww - (c1 - c) * hh)
            if diag == "TLBR":
                take = d_tlbr <= d_trbl
            else:
                take = d_trbl <= d_tlbr
            if take:
                m[r, c] = True
    return m


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _corner_positions(rmin: int, rmax: int, cmin: int, cmax: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (rmin, cmin),
        "TR": (rmin, cmax),
        "BL": (rmax, cmin),
        "BR": (rmax, cmax),
    }


def _present_anchors(marker_pts: set[tuple[int, int]], rmin: int, rmax: int, cmin: int, cmax: int) -> dict[str, bool]:
    corners = _corner_positions(rmin, rmax, cmin, cmax)
    return {
        "TL": _l_anchor_present(marker_pts, corners["TL"][0], corners["TL"][1], (0, 1), (1, 0)),
        "TR": _l_anchor_present(marker_pts, corners["TR"][0], corners["TR"][1], (0, -1), (1, 0)),
        "BL": _l_anchor_present(marker_pts, corners["BL"][0], corners["BL"][1], (0, 1), (-1, 0)),
        "BR": _l_anchor_present(marker_pts, corners["BR"][0], corners["BR"][1], (0, -1), (-1, 0)),
    }


def _anchor_cells_from_presence(
    anchors: dict[str, bool],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> set[tuple[int, int]]:
    corners = _corner_positions(rmin, rmax, cmin, cmax)
    cells: set[tuple[int, int]] = set()
    for name, present in anchors.items():
        if not present:
            continue
        rr, cc = corners[name]
        for dr, dc in _L_OFFSETS[name]:
            cells.add((rr + dr, cc + dc))
    return cells


def _fill_diag_from_anchor_scores(anchors: dict[str, bool]) -> str:
    score_tlbr = int(anchors["TL"]) + int(anchors["BR"])
    score_trbl = int(anchors["TR"]) + int(anchors["BL"])
    assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
    assert score_tlbr != score_trbl, "diagonal scores must not tie"
    return "TRBL" if score_tlbr > score_trbl else "TLBR"


def _fill_diag_from_edge_pixel(
    marker_pts: set[tuple[int, int]],
    anchors: dict[str, bool],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> str:
    present = [name for name, ok in anchors.items() if ok]
    assert len(present) == 2, "direction-marker variant expects exactly two detected L-corners"

    anchor_cells = _anchor_cells_from_presence(anchors, rmin, rmax, cmin, cmax)
    edge_pixels = [pt for pt in marker_pts if pt not in anchor_cells]
    assert len(edge_pixels) == 1, "direction-marker variant expects exactly one extra edge pixel"

    rr, cc = edge_pixels[0]
    on_h_edge = rr == rmin or rr == rmax
    on_v_edge = cc == cmin or cc == cmax
    assert on_h_edge ^ on_v_edge, "direction marker must lie on exactly one bbox edge family"
    assert not (rr in (rmin, rmax) and cc in (cmin, cmax)), "direction marker may not be a corner"

    return "TLBR" if on_h_edge else "TRBL"


def _solve_grid(grid: np.ndarray, use_edge_marker: bool) -> np.ndarray:
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

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = eff_fill
            continue

        protect |= (g == marker) & (~legend_mask)
        if rmin == rmax or cmin == cmax:
            continue

        anchors = _present_anchors(marker_pts, rmin, rmax, cmin, cmax)
        fill_diag = (
            _fill_diag_from_edge_pixel(marker_pts, anchors, rmin, rmax, cmin, cmax)
            if use_edge_marker
            else _fill_diag_from_anchor_scores(anchors)
        )

        fill_mask = _directional_fill_mask(h, w, rmin, rmax, cmin, cmax, fill_diag)
        out[fill_mask & (~protect)] = eff_fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill slanted bbox regions between marker pairs using a color legend and an extra edge cue.

    Input structure:
    - The lower-left contains the same vertical 2-column legend as the base task: marker color, then fill color.
    - Most legend markers appear as two opposite L-shaped corners plus one separate same-color direction pixel on the bbox edge.
    - Some legend markers are still plain 2x2 blocks, which behave as simple recolor cases.

    Rule:
    - Read the legend bottom-to-top and gather all non-legend pixels of each marker color.
    - For a 2x2 block, recolor the block directly with its legend fill color.
    - For an L-pair marker, ignore the old anchor-score tie-break: the extra edge pixel tells which diagonal family to use.
    - A direction pixel on the top or bottom edge means fill the TL-BR-oriented slanted half of the bbox; a pixel on the left or right edge means fill the TR-BL-oriented one.
    - As before, legend cells and later marker pixels are protected from earlier fills.

    Key insight: The two L corners only define the rectangle; the lone extra marker pixel chooses which diagonal-oriented half of that rectangle should be colored.
    """
    return _solve_grid(grid, use_edge_marker=True)


# --------------- generator (C1+D18) ---------------

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
EDGE_HINT_MISMATCH_PROB = 0.65           # prob that old anchor logic would disagree with the new hint

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 180
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300


def _opposite_diag(diag: str) -> str:
    return "TRBL" if diag == "TLBR" else "TLBR"


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
    """
    Reuse the original safe bbox families so L-corner recognition remains unambiguous.
    """
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _bbox_supports_fill_diag(h: int, w: int, fill_diag: str) -> bool:
    if not _l_pair_valid_bbox(h, w):
        return False
    if fill_diag == "TLBR":
        return w >= 5
    return h >= 5


def _pick_bbox_size(H: int, W: int, fill_diag: str) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W) and leaves room for an isolated edge hint."""
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
    if bh > H or bw > W or not _bbox_supports_fill_diag(bh, bw, fill_diag):
        return None
    return bh, bw


def _direction_pixel_candidates(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    anchor_diag: str,
    fill_diag: str,
) -> list[tuple[int, int]]:
    anchor_cells = set(_l_pair_cells(rmin, cmin, rmax, cmax, anchor_diag))

    candidates: list[tuple[int, int]] = []
    if fill_diag == "TLBR":
        edge_points = (
            [(rmin, c) for c in range(cmin + 1, cmax)],
            [(rmax, c) for c in range(cmin + 1, cmax)],
        )
    else:
        edge_points = (
            [(r, cmin) for r in range(rmin + 1, rmax)],
            [(r, cmax) for r in range(rmin + 1, rmax)],
        )

    for edge in edge_points:
        for pt in edge:
            if pt in anchor_cells:
                continue
            if any(abs(pt[0] - rr) + abs(pt[1] - cc) == 1 for rr, cc in anchor_cells):
                continue
            candidates.append(pt)
    return candidates


def _try_place_directional_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    anchor_diag: str,
    fill_diag: str,
    can_place,
    do_place,
) -> bool:
    """Try to place an L-pair plus one isolated edge hint pixel."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W, fill_diag)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1

        hint_candidates = _direction_pixel_candidates(rmin, cmin, rmax, cmax, anchor_diag, fill_diag)
        if not hint_candidates:
            continue

        cells = _l_pair_cells(rmin, cmin, rmax, cmax, anchor_diag)
        cells.append(random.choice(hint_candidates))
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
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _assign_types(n_entries: int) -> list[str]:
    """
    Keep at least two L-pairs whenever possible so the new direction cue is repeatedly visible.
    """
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    special_budget = max(0, n_entries - 2)
    if special_budget > 0 and random.random() < NOOP_PROB and idx_pool:
        types[idx_pool.pop()] = "noop"
        special_budget -= 1
    if special_budget > 0 and random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    return types


def _assign_direction_specs(types: list[str]) -> dict[int, tuple[str, str]]:
    """
    Returns {entry_index: (anchor_diag, fill_diag)} for L entries.
    The generator forces at least one old-vs-new disagreement and often shows both fill directions.
    """
    l_indices = [i for i, t in enumerate(types) if t == "L"]
    specs: dict[int, tuple[str, str]] = {}
    if not l_indices:
        return specs

    fill_diags = {i: random.choice(["TLBR", "TRBL"]) for i in l_indices}
    if len(l_indices) >= 2 and len({fill_diags[i] for i in l_indices}) == 1:
        pivot = random.choice(l_indices)
        fill_diags[pivot] = _opposite_diag(fill_diags[pivot])

    mismatches = set()
    for i in l_indices:
        fill_diag = fill_diags[i]
        mismatch = random.random() < EDGE_HINT_MISMATCH_PROB
        if mismatch:
            anchor_diag = fill_diag
            mismatches.add(i)
        else:
            anchor_diag = _opposite_diag(fill_diag)
        specs[i] = (anchor_diag, fill_diag)

    if not mismatches:
        pivot = random.choice(l_indices)
        _anchor_diag, fill_diag = specs[pivot]
        specs[pivot] = (fill_diag, fill_diag)
    return specs


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

    types = _assign_types(len(legend_rows))
    specs = _assign_direction_specs(types)
    if not specs:
        return None

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

    for i, (_, m, _) in enumerate(legend_rows):
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
            continue

        anchor_diag, fill_diag = specs[i]
        if not _try_place_directional_pair(
            H,
            W,
            m,
            LPAIR_RETRIES,
            anchor_diag,
            fill_diag,
            can_place,
            do_place,
        ):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            fill_diag = random.choice(["TLBR", "TRBL"])
            if random.random() < 0.5:
                anchor_diag = fill_diag
            else:
                anchor_diag = _opposite_diag(fill_diag)
            _try_place_directional_pair(
                H,
                W,
                random.choice(avail),
                DISTRACTOR_RETRIES,
                anchor_diag,
                fill_diag,
                can_place,
                do_place,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        new_out = main(grid)
        old_out = _solve_grid(grid, use_edge_marker=False)
    except Exception:
        return None

    if np.array_equal(new_out, old_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
