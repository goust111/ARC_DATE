# Variant of 9385bd28
# Dimension: D10 — 图例条目数量与复杂度 (meta)
# Change: 2-5 个图例条目，0-1 个无操作条目（标记色无实际标记像素）... → 多个无操作条目作为干扰

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the D10 variant
# ----------------------------------
# concretization:
#   - Input still contains the same two ingredients as the source task:
#     a bottom-left vertical legend of (marker, fill) pairs, and scattered marker
#     objects whose geometry determines the fill region.
#   - The visible change is in the legend inventory. Each generated puzzle now has
#     4-5 legend rows, and 2-3 of those rows are decoys: their marker color appears
#     only inside the legend and nowhere else in the grid.
#   - The solver therefore cannot assume "most legend rows are actionable" or
#     "there is at most one empty row". It must inspect each legend row separately
#     and skip rows whose marker has no non-legend pixels.
#   - Small trace:
#     original behavior: with legend rows [A, B, C, D], perhaps only D is empty.
#     The solver scans bottom->top, finds marker pixels for A/B/C, and skips D once.
#     variant behavior: with rows [A, B, C, D, E], rows B, D, E may all be empty.
#     The solver must perform the same validation for every row and apply fills only
#     for A and C. The output rule is unchanged; the extra work is deciding which
#     legend entries are real.
# complexity calibration:
#   - To make "multiple noop rows" visually obvious while still leaving enough
#     evidence of the original task, the generator restricts the legend to 4-5 rows
#     and always leaves at least 2 actionable entries.
#   - A single optional 2x2 block is retained, but only among actionable entries,
#     so there is always at least one true L-pair fill to read from the examples.
# implicit-assumption review of main():
#   - `_find_legend` only assumes a contiguous 2-column legend stack at bottom-left.
#     That remains true.
#   - `marker_pts = argwhere((g == marker) & (~legend_mask))` assumes a legend entry
#     is valid only if the marker color appears outside the legend. That remains the
#     exact noop test we want, so this line becomes the key D10 discriminator.
#   - `if len(marker_pts) < 2: continue` used to harmlessly skip a rare empty row.
#     In this variant it must robustly skip 2-3 rows per puzzle, so generation must
#     guarantee that noop rows have no stray pixels of their marker color elsewhere.
#   - Future-marker protection still works because marker colors remain distinct even
#     when some of them are decoys.
# placement legality:
#   - Legend scan integrity is protected exactly as before: no placed object may
#     intrude into rows below the legend bottom or the protected left strip.
#   - Actionable markers never overlap one another or the legend, because placement
#     checks use a shared `occupied` mask.
#   - Noop rows are legal by construction because they place nothing outside legend
#     cells, and distractor colors are chosen outside the legend marker palette.
# visual observability:
#   - A human can see the new pattern from examples because several legend colors
#     visibly fail to appear as body markers, yet the solver still ignores those rows
#     and only fills regions for the colors that do have matching L-pairs/blocks.
#   - Noop positions are shuffled, so the rule is "verify each row", not "ignore the
#     top rows" or any other hidden ordering convention.
# cleanliness / color reasoning:
#   - The transform still relies only on visible geometric/color correspondences in
#     the legend; no ordering or color-id arithmetic is introduced.
#   - The number of active regions stays small enough that outputs remain readable
#     even with the denser legend.


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
    Use the bottom-left legend to recolor the regions defined by matching marker shapes.

    Input structure:
    - The lower-left corner contains a vertical 2-column legend; each row pairs a marker color with its fill color.
    - Away from the legend, some marker colors appear as opposite L-corners of a bounding box, and some appear as a solid 2x2 block.
    - In this variant, several legend rows are deliberate decoys: their marker color never appears outside the legend at all.

    Rule:
    - Read legend rows from bottom to top and inspect each marker color independently.
    - If that color has no non-legend pixels, treat the row as a no-op and leave the grid unchanged for that entry.
    - Otherwise, solve the geometry exactly as in the base task: recolor a 2x2 block directly, or detect which bbox diagonal holds the L-anchors and fill the opposite diagonal's diamond using the legend fill color.
    - Preserve legend cells and marker pixels that belong to the current entry or any later legend entry while processing earlier rows.

    Key insight:
    - The extra difficulty is not a new geometric transform; it is that the legend can contain multiple fake rows, so every row must first be validated against the body of the grid before applying the usual diagonal-fill rule.
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

    # Every legend row is checked separately because 2-3 rows may be decoys with no body pixels.
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
LEGEND_ENTRY_RANGE = (4, 5)     # raised lower bound so 2-3 noop rows can coexist with 2 active rows
NOOP_ENTRY_RANGE = (2, 3)       # core D10 change

# ── CFG: marker types ──────────────────────────────────────────────
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
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
    """Create the legend grid and metadata for the current attempt."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [color for color in range(10) if color != bg]
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
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    """Protect the legend strip so scanning from the lower-left stays unambiguous."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _choose_entry_types(n_entries: int) -> list[str] | None:
    """Assign each legend row to L, block, or noop while enforcing the D10 distribution."""
    noop_max = min(NOOP_ENTRY_RANGE[1], n_entries - 2)
    if noop_max < NOOP_ENTRY_RANGE[0]:
        return None
    noop_count = random.randint(NOOP_ENTRY_RANGE[0], noop_max)

    types = ["L"] * n_entries
    indices = list(range(n_entries))
    random.shuffle(indices)
    noop_indices = indices[:noop_count]
    for idx in noop_indices:
        types[idx] = "noop"

    active_indices = [idx for idx, kind in enumerate(types) if kind != "noop"]
    if active_indices and random.random() < BLOCK_PROB:
        types[random.choice(active_indices)] = "block"
    return types


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

    types = _choose_entry_types(len(legend_rows))
    if types is None:
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
        available = [color for color in range(10) if color != bg and color not in marker_colors]
        if available:
            _try_place_l_pair(H, W, random.choice(available), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    # D10 guarantee: the marker colors of noop legend rows must not appear outside the legend.
    legend_markers = [marker for _, marker, _ in legend_rows]
    for kind, marker in zip(types, legend_markers):
        marker_pixels = int(np.count_nonzero((grid == marker) & (~legend_mask)))
        if kind == "noop" and marker_pixels != 0:
            return None
        if kind != "noop" and marker_pixels == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
