# Variant of 9385bd28
# Dimension: D9 — 标记对角线方向 (meta)
# Change: anchor_diagonal 可以是 TL-BR 或 TR-BL（TL-BR 约 80%），填充发生在对面的对角线上... → 标记和填充在同一对角线

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   preserved_structure:
#     - grid size, legend layout, marker palette, optional noop entry, optional 2x2 block, distractor policy
#     - marker detection still uses diagonal opposite L-corners inside one axis-aligned bbox
#     - processing order is still legend bottom->top, with legend cells and future markers protected
#   core_change:
#     - the old code conceptually chose one diagonal for anchors and the opposite diagonal for filling
#     - however, its Manhattan "diamond" helper returned the full bbox for either diagonal, so D9 was not visible
#     - to make D9 observable, the fill region is changed to a diagonal-oriented dominance mask:
#       inside the bbox, fill the cells that are at least as close to the chosen diagonal as to the other diagonal
#     - this is the smallest geometry change that makes "same diagonal" vs "opposite diagonal" visibly different
#   concretization_and_trace:
#     - example bbox: 4x7 with anchors at TL and BR
#     - old behavior in code: detect TL-BR anchors, then choose TR-BL as the fill diagonal; due to the old helper,
#       either choice still painted the entire 4x7 rectangle, so the diagonal decision was hidden
#     - new variant behavior: detect TL-BR anchors, keep TL-BR as the fill diagonal, and paint the TL-BR-oriented
#       wedge/band of the bbox; choosing TR-BL instead would paint the complementary wedge/band
#     - visible difference: the same input now leads to different painted cells depending on the anchor diagonal
#   complexity_calibration:
#     - keep only two diagonal classes: TL-BR and TR-BL
#     - keep TL-BR more common (~80%) but still allow TR-BL often enough for contrast across examples
#     - retain 2x2 block and noop entries so the task is not reduced to one object type
#     - require at least one L-pair entry whose same-vs-opposite outputs differ on visible cells
#   implicit_assumption_review:
#     - _find_legend assumes the leftmost non-bg run at the bottom-left is the legend; generator still reserves that area
#     - anchor scoring assumes exactly one diagonal wins; generator still forbids bbox sizes that create score ties
#     - protection logic assumes later markers remain intact; this remains valid because only the fill mask changes
#     - the old "choose opposite diagonal" branch was harmless because both masks were identical; in this variant
#       it would be wrong, so the solver must explicitly choose the anchor diagonal itself
#   placement_legality:
#     - markers cannot overlap each other or the reserved legend corridor
#     - L-pairs stay inside-grid and keep the old valid-bbox constraints, which still avoid ambiguous anchor scoring
#     - filled regions may overlap in output as before; ordering resolves conflicts
#   visual_observability:
#     - the generator rejects samples where same-diagonal and opposite-diagonal solving produce the same final output
#     - it also rejects samples with no informative L-pair, preventing silent degeneration into block/noop-only cases
#   visual_cleanliness:
#     - marker pixels remain protected so the L-shapes are still readable after filling
#     - distractors are still rare and never use legend colors, so they do not confuse the legend mapping
#   color_inference:
#     - colors still come directly from the 2-column legend; no ordering or transform depends on color ids
#   new_solver_assumptions_checked_against_generator:
#     - every informative sample must contain at least one non-block L-pair
#     - for at least one such pair, the visible cells painted by the same-diagonal mask must differ from the
#       opposite-diagonal mask after protection; generator verifies this quantitatively before accepting the sample


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
        fill = int(g[r, c + 1]) if c + 1 < g.shape[1] else bg
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

    Distances are compared in normalized integer form, avoiding floating point and making the
    two diagonal choices complementary up to the tie cells.
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


def _detect_anchor_diag(marker_pts: set[tuple[int, int]], rmin: int, rmax: int, cmin: int, cmax: int) -> str:
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
    return "TLBR" if score_tlbr > score_trbl else "TRBL"


def _solve(grid: np.ndarray, fill_same_diagonal: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask(g, legend_c, entries)
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
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)
        if rmin == rmax or cmin == cmax:
            continue

        anchor_diag = _detect_anchor_diag(marker_pts, rmin, rmax, cmin, cmax)
        fill_diag = anchor_diag if fill_same_diagonal else _opposite_diag(anchor_diag)
        region = _diagonal_fill_mask(h, w, rmin, rmax, cmin, cmax, fill_diag)
        out[region & (~protect)] = int(fill)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a bottom-left color legend to paint diagonal-oriented regions between matching corner markers.

    Input structure:
    - The background grid contains a vertical 2-column legend near the lower-left corner.
    - Each legend row stores a marker color and its fill color, and matching non-legend pixels of that marker color
      appear elsewhere either as a diagonal pair of L-shaped corners or as one solid 2x2 block.
    - An L-pair spans the opposite corners of one axis-aligned bounding box, so it reveals both the box and a diagonal.

    Rule:
    - Read legend rows from bottom to top and gather all non-legend pixels of the current marker color.
    - If those pixels form a 2x2 block, recolor that block directly to the legend fill color.
    - Otherwise detect whether the L-corners sit on the TL-BR diagonal or the TR-BL diagonal of their bounding box.
    - Fill the cells of that same bounding box that lie on the same diagonal side/orientation as the markers,
      while preserving legend cells, current marker pixels, and later marker pixels.

    Key insight: The corner markers do not point to the complementary diagonal anymore; the painted region follows
    the very diagonal that carries the two L-shaped anchors.
    """
    return _solve(grid, fill_same_diagonal=True)


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
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: variant observability ─────────────────────────────────────
MIN_VARIANT_DELTA = 4

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
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _build_legend_mask(g, legend_c, entries)
    marker_order = [marker for _, marker, _ in entries]

    l_pairs = 0
    informative_pairs = 0
    tlbr_pairs = 0
    trbl_pairs = 0

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            continue
        if rmin == rmax or cmin == cmax:
            continue

        diag = _detect_anchor_diag(marker_pts, rmin, rmax, cmin, cmax)
        l_pairs += 1
        if diag == "TLBR":
            tlbr_pairs += 1
        else:
            trbl_pairs += 1

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        same_mask = _diagonal_fill_mask(g.shape[0], g.shape[1], rmin, rmax, cmin, cmax, diag)
        other_mask = _diagonal_fill_mask(
            g.shape[0], g.shape[1], rmin, rmax, cmin, cmax, _opposite_diag(diag)
        )
        delta_cells = np.count_nonzero((same_mask ^ other_mask) & (~protect) & (g != int(fill)))
        if delta_cells > 0:
            informative_pairs += 1

    same_out = _solve(g, fill_same_diagonal=True)
    opp_out = _solve(g, fill_same_diagonal=False)
    output_delta = int(np.count_nonzero(same_out != opp_out))

    return {
        "l_pairs": l_pairs,
        "informative_pairs": informative_pairs,
        "tlbr_pairs": tlbr_pairs,
        "trbl_pairs": trbl_pairs,
        "output_delta": output_delta,
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

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and len(idx_pool) > 1:
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

    metrics = _variant_metrics(grid)
    if metrics["l_pairs"] == 0:
        return None
    if metrics["informative_pairs"] == 0:
        return None
    if metrics["output_delta"] < MIN_VARIANT_DELTA:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
