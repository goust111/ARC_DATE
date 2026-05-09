# Variant of 9385bd28
# Dimension: D6 — 处理顺序与重叠规则 (multi_step)
# Change: ordered_processing — 图例条目从下到上顺序处理，后处理的条目覆盖先处理的；保护后续条目的标记像素和图... → 重叠区域使用混合色

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg

# Design analysis for the D6 variant.
# - Concrete form:
#   The legend grows from 2 columns to 3 columns. Each row is now
#   (marker_color, fill_color, mix_color). The first two columns keep the
#   original role; the third column repeats one shared "blend" color that is
#   used whenever two painted diamonds want the same cell.
# - Behavior trace:
#   Old rule on a cell hit by fills A then B: write A first, then overwrite with B.
#   New rule on the same cell: write A first, but when B also claims the cell,
#   convert that cell to the shared mix color instead of B. Marker protection and
#   legend protection still follow the original bottom-to-top traversal.
# - Complexity calibration:
#   The generator forces at least two real L-pairs and at least one visible
#   overlap zone. This avoids samples where the new third legend column is unused.
#   A single shared mix color is intentionally simpler than pair-specific mixing,
#   so 2-4 examples remain enough for human induction.
# - Implicit assumption audit:
#   The original solver relied on unconditional assignment
#   `out[diamond & (~protect)] = fill`, which is only harmless when order decides
#   conflicts. Here that would destroy earlier evidence of overlap, so the solver
#   must track whether a cell has already been painted by a previous diamond fill.
#   The legend parser also assumed width 2; it now must require width 3 and verify
#   that all legend rows agree on one mix color.
# - Placement legality:
#   Marker pixels still may not overlap each other or the legend. Overlap is only
#   allowed between filled diamonds, not between the visible L-shapes themselves.
#   The generator therefore separates marker placement from fill-overlap checking.
# - Visual observability and cleanliness:
#   The mix color is chosen distinct from bg, all marker colors, and all fill
#   colors. That makes the overlap zone visually unique and keeps the output from
#   looking like an ordinary later overwrite.
# - Color inference:
#   Repeating the same third legend color on every legend row gives an explicit
#   structural hint: it is not tied to one marker, but to the interaction between
#   filled regions. The generator rejects samples with zero visible mix cells.


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Legend is a vertical stack of triples at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1, mix_color at col c+2).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill, mix).
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
    assert start_r is not None and start_c is not None and start_c < w - 2, (
        "legend must exist at bottom-left with at least 3 columns"
    )

    entries: list[tuple[int, int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        mix = int(g[r, c + 2]) if c + 2 < w else bg
        entries.append((r, marker, fill, mix))
    return c, entries


def _build_legend_mask(g: np.ndarray, legend_c: int, entries: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = g.shape
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _, _ in entries:
        for dc in range(3):
            if 0 <= r < h and 0 <= legend_c + dc < w:
                legend_mask[r, legend_c + dc] = True
    return legend_mask


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


def _diamond_from_anchor_diag(
    h: int,
    w: int,
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    diag: str,
) -> np.ndarray:
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    return _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)


def _solve(grid: np.ndarray, use_mix_overlap: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill, mix) entry"

    mix_colors = {mix for _, _, _, mix in entries}
    assert len(mix_colors) == 1, "all legend rows must share one mix color"
    mix_color = next(iter(mix_colors))

    legend_mask = _build_legend_mask(g, legend_c, entries)
    out = g.copy()

    marker_order = [m for _, m, _, _ in entries]
    fill_written = np.zeros_like(g, dtype=bool)

    # Process legend entries bottom -> top. Marker/legend protection keeps the
    # original temporal behavior; only fill-fill conflicts switch to the mix color.
    for i, (_, marker, fill, _) in enumerate(entries):
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
        assert score_tlbr != score_trbl, "diagonal scores must not tie"
        if score_tlbr > score_trbl:
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, tr, bl)
        else:
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, tl, br)

        paint = diamond & (~protect)
        if not np.any(paint):
            continue

        if use_mix_overlap:
            overlap = paint & fill_written
            fresh = paint & (~fill_written)
            out[fresh] = eff_fill
            out[overlap] = mix_color
        else:
            out[paint] = eff_fill
        fill_written |= paint

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint legend-defined diamonds and convert diamond overlaps into a shared blend color.

    Input structure:
    - Background grid with a vertical 3-column legend near the bottom-left; each row is (marker, fill, mix).
    - Non-legend pixels of a marker color form either two L-shaped opposite corners of a bounding box or one solid 2x2 block.
    - The third legend column repeats one common mix color that is not tied to any single marker.

    Rule:
    - Read the legend bottom-to-top and locate all non-legend pixels of that marker color.
    - For an L-pair, detect which bbox diagonal contains the anchors; the opposite diagonal defines the diamond to paint.
    - Preserve the original legend cells and the still-unprocessed marker pixels; 2x2 blocks are still recolored directly.
    - If a diamond paints a fresh cell, write its fill color; if another diamond also paints that cell later, replace the conflict cell with the shared mix color instead of letting order decide it.

    Key insight:
    - The extra legend column announces one global overlap color, so traversal order still matters for marker protection,
      but rectangle-rectangle conflicts are recognized by color blending rather than by overwrite priority.
    """
    return _solve(grid, use_mix_overlap=True)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)     # capped at 4 so marker/fill/mix can all stay visually distinct

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# ── CFG: overlap visibility ────────────────────────────────────────
FORCED_OVERLAP_MIN = 4
FORCED_DIAMOND_MIN = 6

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.20

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 180
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 400

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


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    target_diamond: np.ndarray | None = None,
    min_overlap: int = 0,
    min_diamond_cells: int = 0,
) -> dict | None:
    """Try to place an L-pair of `color`. Returns placement info on success."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if not can_place(cells):
            continue
        diamond = _diamond_from_anchor_diag(H, W, rmin, cmin, rmax, cmax, diag)
        if int(diamond.sum()) < min_diamond_cells:
            continue
        if target_diamond is not None:
            if int(np.count_nonzero(diamond & target_diamond)) < min_overlap:
                continue
        do_place(cells, color)
        return {
            "cells": cells,
            "diamond": diamond,
            "bbox": (rmin, cmin, rmax, cmax),
            "diag": diag,
        }
    return None


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None
    if legend_c + 2 >= W:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < (2 * n_entries + 1):
        return None
    marker_colors = random.sample(pool, n_entries)
    remaining = [c for c in pool if c not in marker_colors]
    mix_color = random.choice(remaining)
    fill_pool = [c for c in remaining if c != mix_color]
    if len(fill_pool) < n_entries:
        return None
    fill_colors = random.sample(fill_pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = fill_colors[i]
        legend_rows.append((r, m, fill, mix_color))
        for dc, value in enumerate((m, fill, mix_color)):
            grid[r, legend_c + dc] = value
            legend_mask[r, legend_c + dc] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, fill_colors, mix_color


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows) -> np.ndarray:
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _build_legend_mask(g, legend_c, entries)
    mix_color = entries[0][3]
    new_out = _solve(g, use_mix_overlap=True)
    old_out = _solve(g, use_mix_overlap=False)
    mix_pixels = int(np.count_nonzero((new_out == mix_color) & (~legend_mask)))
    return {
        "mix_pixels": mix_pixels,
        "diff_vs_ordered": bool(np.any(new_out != old_out)),
        "active_entries": sum(
            1 for _, marker, _, _ in entries if np.count_nonzero((g == marker) & (~legend_mask)) >= 2
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    (
        grid,
        legend_rows,
        legend_mask,
        legend_c,
        legend_bot,
        marker_colors,
        _fill_colors,
        _mix_color,
    ) = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    optional_pool = list(range(2, n_entries))
    random.shuffle(optional_pool)
    if random.random() < NOOP_PROB and optional_pool:
        types[optional_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and optional_pool:
        types[optional_pool.pop()] = "block"

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

    first_info = _try_place_l_pair(
        H,
        W,
        legend_rows[0][1],
        LPAIR_RETRIES,
        can_place,
        do_place,
        min_diamond_cells=FORCED_DIAMOND_MIN,
    )
    if first_info is None:
        return None

    second_info = _try_place_l_pair(
        H,
        W,
        legend_rows[1][1],
        LPAIR_RETRIES,
        can_place,
        do_place,
        target_diamond=first_info["diamond"],
        min_overlap=FORCED_OVERLAP_MIN,
        min_diamond_cells=FORCED_DIAMOND_MIN,
    )
    if second_info is None:
        return None

    for i in range(2, n_entries):
        _, marker, _, _ = legend_rows[i]
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
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place) is None:
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
    if metrics["mix_pixels"] < FORCED_OVERLAP_MIN:
        return None
    if not metrics["diff_vs_ordered"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
