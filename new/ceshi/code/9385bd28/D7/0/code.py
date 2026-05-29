# Variant of 9385bd28
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: same — 输出与输入网格尺寸完全相同... → 紧裁输出（仅保留变化区域）

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


# Design analysis for the D7 variant
# ----------------------------------
# concrete_behavior:
#   - Input objects stay the same as the base task: a bottom-left vertical legend,
#     plus L-corner marker pairs or 2x2 marker blocks elsewhere on the canvas.
#   - The fill logic also stays the same: each legend entry produces either a
#     diamond fill inside a marker bbox or a recolored 2x2 block.
#   - The D7 change is purely about the returned frame. Instead of returning the
#     whole transformed canvas, return the smallest rectangle that contains every
#     visibly changed cell, i.e. the bbox of (full_output != input).
#   - Example trace:
#       original: fill a diamond at rows 1..6, cols 4..9, keep legend at rows 8..9.
#                 Output shape stays HxW.
#       variant:  perform the same fill first, then crop to rows 1..6, cols 4..9.
#                 Legend and untouched background are discarded.
# complexity_calibration:
#   - The recognition burden should remain the same: the solver still has to read
#     the legend and infer which diagonal is filled.
#   - Tight-crop must be visually obvious, so generated active objects are kept
#     inside an interior "activity box" above and to the right of the legend.
#     This prevents accidental full-canvas crops.
# implicit_assumptions_to_fix:
#   - The base solver safely returned `out` because D7 was fixed to "same shape".
#     That assumption no longer holds; returning `out` would leak the unchanged
#     border and the legend.
#   - The old generator allowed markers on edges and on legend-adjacent rows.
#     Under tight-crop this can make the crop equal the full canvas or include
#     legend cells inside the crop rectangle. The generator now reserves margins.
# placement_legality:
#   - Legend placement is unchanged.
#   - All legend-controlled markers are placed in a rectangular activity region:
#       rows 1..legend_top-1, cols legend_c+2..W-2.
#     This keeps fills away from the outer border and from the legend columns.
#   - Optional distractors, when present, are placed below the activity region so
#     they never appear inside the cropped output.
# observability_and_cleanliness:
#   - At least one visible change is required; samples with zero changed cells are
#     rejected.
#   - Fill colors exclude the background, so L-pair fills remain visible.
#   - The crop is validated to be strictly smaller than the input in both axes.
# new_solver_assumptions_and_generator_guarantees:
#   - `main()` assumes only that the task follows the original marker/legend rules;
#     if no visible change exists, it returns an empty 0x0 array.
#   - `generate_input()` guarantees non-empty visible change, interior placement,
#     and a crop rectangle disjoint from the legend footprint.


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


def _solve_full(grid: np.ndarray) -> np.ndarray:
    """Run the original full-canvas transform before D7 cropping."""
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
        out[diamond & (~protect)] = eff_fill

    return out


def _tight_crop_changed(inp: np.ndarray, out: np.ndarray, bg: int) -> np.ndarray:
    """Crop `out` to the bbox of visible changes relative to `inp`."""
    changed = out != inp
    pts = np.argwhere(changed)
    if pts.size == 0:
        return np.full((0, 0), bg, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill legend-coded marker regions, then return only the changed window.

    Input structure:
    - A single-background grid with a vertical 2-column legend at the bottom-left.
    - Each legend row stores (marker color, fill color), read from bottom to top.
    - Away from the legend, each marker color appears either as two L-corners on a bbox diagonal or as a solid 2x2 block.

    Rule:
    - Apply the same fill logic as the base task: for each legend entry, detect the marker bbox, infer which diagonal carries the L-anchors, and fill the opposite diagonal's diamond with the legend fill color.
    - Preserve the legend and future-marker pixels while processing; recolor solid 2x2 marker blocks directly.
    - After the full-canvas transform is complete, compare output against input and keep only the smallest rectangle containing all changed cells.

    Key insight: The geometry rule is unchanged; the new step is that output framing depends on where the visible painting happened, not on the original canvas size.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = _solve_full(g)
    return _tight_crop_changed(g, out, bg)


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
    """True if (h, w) avoids an L-anchor score tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(region_h: int, region_w: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits inside a target region."""
    if region_h < 2 or region_w < 2:
        return None
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        if region_h < 3 or region_w < 3:
            return None
        bh = random.randint(3, min(BBOX_H_MAX, region_h))
        bw = random.randint(3, min(BBOX_W_MAX, region_w))
    elif shape_choice == "h2_w_ge4":
        w_hi = min(BBOX_W_MAX, region_w)
        if region_h < 2 or w_hi < 4:
            return None
        bh = 2
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, region_h)
        if h_hi < 4 or region_w < 2:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _try_place_l_pair(
    color: int,
    retries: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    can_place,
    do_place,
) -> bool:
    """Try to place an L-pair of `color` inside the given inclusive region."""
    region_h = row_hi - row_lo + 1
    region_w = col_hi - col_lo + 1
    if region_h < 2 or region_w < 2:
        return False
    for _ in range(retries):
        size = _pick_bbox_size(region_h, region_w)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(row_lo, row_hi - bh + 1)
        cmin = random.randint(col_lo, col_hi - bw + 1)
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
        m = marker_colors[i]
        fill_pool = [c for c in range(10) if c != bg and c != m]
        fill = random.choice(fill_pool)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
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


def _changed_bbox(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(out != inp)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    legend_top = min(r for r, _, _ in legend_rows)
    active_row_lo = 1
    active_row_hi = legend_top - 1
    active_col_lo = legend_c + 2
    active_col_hi = W - 2
    if active_row_hi - active_row_lo + 1 < 2 or active_col_hi - active_col_lo + 1 < 2:
        return None

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

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if not (active_row_lo <= r <= active_row_hi and active_col_lo <= c <= active_col_hi):
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
            if active_row_hi - active_row_lo + 1 < 2 or active_col_hi - active_col_lo + 1 < 2:
                return None
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(active_row_lo, active_row_hi - 1)
                c0 = random.randint(active_col_lo, active_col_hi - 1)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(
                m,
                LPAIR_RETRIES,
                active_row_lo,
                active_row_hi,
                active_col_lo,
                active_col_hi,
                can_place,
                do_place,
            ):
                return None

    distractor_row_lo = legend_top
    distractor_row_hi = H - 2
    distractor_col_lo = legend_c + 2
    distractor_col_hi = W - 2
    if random.random() < DISTRACTOR_PROB and distractor_row_lo <= distractor_row_hi:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            def can_place_distractor(cells):
                for r, c in cells:
                    if not (0 <= r < H and 0 <= c < W):
                        return False
                    if not (distractor_row_lo <= r <= distractor_row_hi and distractor_col_lo <= c <= distractor_col_hi):
                        return False
                    if occupied[r, c] or forbidden[r, c]:
                        return False
                return True

            _try_place_l_pair(
                random.choice(avail),
                DISTRACTOR_RETRIES,
                distractor_row_lo,
                distractor_row_hi,
                distractor_col_lo,
                distractor_col_hi,
                can_place_distractor,
                do_place,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        full_out = _solve_full(grid)
    except Exception:
        return None

    bbox = _changed_bbox(grid, full_out)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox

    if r0 <= 0 or c0 <= legend_c + 1 or r1 >= legend_top or c1 >= W - 1:
        return None

    crop = full_out[r0 : r1 + 1, c0 : c1 + 1]
    if crop.shape[0] >= H or crop.shape[1] >= W:
        return None

    try:
        cropped_out = main(grid)
    except Exception:
        return None
    if cropped_out.shape != crop.shape or not np.array_equal(cropped_out, crop):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
