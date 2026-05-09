# Variant of 9385bd28
# Dimension: D11 — 干扰项 (meta)
# Change: 0-1 个干扰 L 对（使用非图例颜色的 L 形角标记），解题者需忽略非图例色的标记... → 无干扰项

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _solver_dir = _parent / "solvers"
    if (_solver_dir / "arc_utils.py").exists():
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break

from arc_utils import get_bg


# Design analysis for the D11 variant
# design_analysis:
#   core_change:
#     - The original family could include 0-1 non-legend-color L-pairs that the solver
#       had to ignore. This variant removes that branch entirely: every visible marker
#       pixel outside the legend now uses one of the legend marker colors.
#   concretization_and_trace:
#     - Input still contains a bottom-left vertical legend of (marker_color, fill_color)
#       pairs, plus 0-1 noop legend entries and 0-1 solid 2x2 marker blocks.
#     - The visible corner markers remain L-shaped 3-pixel anchors placed on one bbox
#       diagonal, and the filled output remains the opposite diagonal's diamond region.
#     - Small trace:
#       original: legend says 2->7 and 3->4; the grid also contains an extra color-8
#       L-pair. `main()` finds pixels for color 2 and color 3 only, so color 8 is ignored.
#       variant: the same grid is generated without the color-8 pair. The fill behavior
#       for colors 2 and 3 is unchanged, but now every colored corner the user sees is
#       actually relevant to the legend.
#   complexity_calibration:
#     - Only the distractor mechanism is removed. The task still keeps 2-5 legend rows,
#       mixed TL-BR/TR-BL diagonals, optional noop entries, and optional 2x2 blocks, so
#       it stays recognizably the same task rather than collapsing into a single pattern.
#   implicit_assumptions_review:
#     - `_find_legend` assumes the first non-bg pixels near the lower-left belong to the
#       legend. The generator still reserves rows below the legend and columns up through
#       the legend column, so this remains valid.
#     - `main()` collects all non-legend pixels of a legend marker color into one object.
#       That was safe before only because distractors used non-legend colors; it remains
#       safe here because the variant forbids any extra non-legend marker colors entirely.
#     - The diagonal-score asserts depend on bbox shapes avoiding ambiguous cases such as
#       2x2, 2x3, and 3x2 L-pairs. The generator keeps the same bbox validity filter.
#     - The 2x2 recolor path assumes the current marker contributes exactly four pixels in
#       a 2x2 bbox. Placement still forbids overlap, so that assumption remains true.
#   placement_legality:
#     - `occupied` prevents object overlap and legend collisions.
#     - `forbidden` preserves the legend scan corridor.
#     - Removing distractors only reduces crowding; it does not require new spacing rules.
#   visual_observability_and_cleanliness:
#     - The D11 change is visible because the input now contains no "mystery" color that
#       lacks a legend row. Every foreground marker color appears in the legend.
#     - This makes examples cleaner while preserving the original legend-to-shape logic.
#   color_inference:
#     - Fill-color inference is still fully visible in the legend, and no color-id-based
#       ordering is introduced. Marker order remains the legend's bottom-to-top order.
#   generator_guarantee:
#     - After placement, all non-legend, non-background pixels must use legend marker
#       colors; otherwise the sample is rejected.


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
    Recolor legend-coded marker groups by filling the diamond implied by each L-pair.

    Input structure:
    - A single-background grid has a two-column legend stacked near the bottom-left; each row stores
      (marker_color, fill_color), and rows are processed from bottom to top.
    - Outside the legend, every foreground marker pixel belongs to one of those legend marker colors.
    - A marker is either a pair of opposite L-shaped corners or a solid 2x2 block of one legend color.

    Rule:
    - For each legend row, gather all non-legend pixels of that marker color.
    - If those pixels form a 2x2 block, recolor the block to the legend's fill color.
    - Otherwise read which bbox diagonal carries the two L anchors, then fill the diamond on the opposite
      diagonal with the legend's fill color while preserving legend cells and later marker pixels.
    - Legend order matters: earlier fills must not erase markers that will be used by later legend rows.

    Key insight:
    - The L-corners identify one diagonal of the bounding box, but the cells to recolor lie on the other
      diagonal's shortest-path diamond. In this variant there are no extra non-legend-color decoys, so
      every visible marker color directly corresponds to a legend entry.
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

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for future_marker in future_markers:
            protect |= (g == future_marker) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

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
        assert score_tlbr != score_trbl, "diagonal scores must not tie"
        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

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
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
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
    """Create legend entries and the corresponding grid and mask arrays."""
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


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _variant_has_no_distractors(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    marker_colors: list[int],
    bg: int,
) -> bool:
    """All non-legend foreground pixels must use one of the legend marker colors."""
    visible_colors = set(map(int, np.unique(grid[~legend_mask])))
    visible_colors.discard(bg)
    return visible_colors <= set(marker_colors)


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
        marker_type = types[i]
        if marker_type == "noop":
            continue
        if marker_type == "block":
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

    if not _variant_has_no_distractors(grid, legend_mask, marker_colors, bg):
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
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
