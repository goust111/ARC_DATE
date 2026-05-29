# Variant of 9385bd28
# Dimension: D16 — 背景色的角色 (meta)
# Change: 单一背景色（0-9 任意），填充色可以等于背景色（产生'隐形'填充效果）... → 多背景色（分区背景）

import random
from collections import Counter

import numpy as np


# Design analysis for the D16 variant
# -----------------------------------
# variant_spec:
#   visual:
#     - The canvas is split into 2-3 full-width horizontal background bands.
#     - Each band has its own background color; adjacent bands always differ.
#     - The legend is still a vertical stack near the bottom-left, but it sits inside the bottom band.
#     - A legend marker color may appear once in multiple bands. Those repeated same-color objects are the
#       visible cue that the legend mapping must be reused separately per background partition.
#   behavior_trace:
#     - Original logic collected all non-legend pixels of one marker color globally, built one bbox,
#       then filled one diamond.
#     - In this variant, the same marker color can appear in the top band and again in the bottom band.
#       The correct behavior is: split the grid by background bands first, then for each legend entry,
#       solve each band-local object independently.
#     - Example: marker color 6 appears as one L-pair in rows 0-4 and another in rows 9-13. Old logic
#       would merge them into one tall bbox and infer nonsense anchors. New logic produces two separate fills.
#   complexity_calibration:
#     - Bands are only horizontal and full-width. This keeps the partition rule obvious and avoids needing
#       arbitrary region segmentation.
#     - Each marker color appears in at most one object per band, so "same legend mapping reused per band"
#       is learnable without ambiguous clustering inside a band.
#   implicit_assumptions_in_original:
#     - `get_bg(g)` assumed one global background color. That fails here because no single color describes
#       all empty space.
#     - `_find_legend(..., bg)` assumed "non-bg" is globally meaningful. Here legend detection must use the
#       local row background instead.
#     - `marker_pts = argwhere(g == marker)` assumed one global object per marker color. Here that must be
#       restricted to one background band at a time.
#     - The original fill wrote across the entire bbox-derived diamond. Here the fill is clipped to the
#       current band, because the legend mapping is applied separately per background partition.
#   placement_validity:
#     - Generator guarantees every non-legend object lies fully inside one band.
#     - No band contains more than one object of the same marker color.
#     - The left strip used to scan the legend is protected from other placements, preserving legend parsing.
#   observability:
#     - Samples visibly show 2-3 large colored background stripes.
#     - At least one legend marker is reused in multiple bands in every generated puzzle, so the variant
#       does not collapse back to the original one-object-per-color behavior.
#   cleanliness:
#     - Horizontal bands keep the scene readable; objects do not straddle band boundaries.
#     - Overlap is still possible between different legend entries only through later-fill protection, as in
#       the original task, but heavy same-area clutter is avoided.
#   color_inference:
#     - Marker colors are never chosen from the band-background palette, so markers remain distinct from
#       empty cells.
#     - Fill colors may equal one or more band colors, making region-specific "visible vs invisible" fill
#       effects possible without relying on hidden numeric ordering.
# solver_assumptions_and_generator_guarantees:
#   - Row-majority color equals the background color of that row.
#   - Background bands are contiguous runs of equal row-majority colors, with count 2 or 3.
#   - Legend occupies the lowest row that contains any non-row-background cell.


def _row_backgrounds(g: np.ndarray) -> list[int]:
    """Infer the local background color of each row by majority vote."""
    return [int(Counter(map(int, g[r, :])).most_common(1)[0][0]) for r in range(g.shape[0])]


def _find_bands(row_bgs: list[int]) -> list[tuple[int, int, int]]:
    """Compress row backgrounds into contiguous horizontal bands."""
    assert row_bgs, "grid must have at least one row"
    bands: list[tuple[int, int, int]] = []
    start = 0
    color = row_bgs[0]
    for r in range(1, len(row_bgs)):
        if row_bgs[r] != color:
            bands.append((start, r - 1, color))
            start = r
            color = row_bgs[r]
    bands.append((start, len(row_bgs) - 1, color))
    return bands


def _find_legend(g: np.ndarray, row_bgs: list[int]) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != row_bgs[r])
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
        nz = np.flatnonzero(g[r, :] != row_bgs[r])
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == row_bgs[r]:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else row_bgs[r]
        entries.append((r, marker, fill))
    return c, entries


def _l_anchor_present(
    pts: set[tuple[int, int]], r: int, c: int, arm1: tuple[int, int], arm2: tuple[int, int]
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
    h: int, w: int, r0: int, r1: int, c0: int, c1: int, a: tuple[int, int], b: tuple[int, int]
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
    Fill diamonds from diagonal marker pairs using a color legend, with band-local background partitions.

    Input structure:
    - The grid background is split into 2-3 horizontal color bands instead of one global bg color.
    - A vertical legend still sits near the bottom-left; each row stores (marker_color, fill_color).
    - The same marker color may appear once in several different background bands, and some appearances are
      solid 2x2 blocks instead of diagonal L-pairs.

    Rule:
    - Infer the background color of each row, then compress those rows into horizontal bands.
    - Read the legend bottom-to-top; for each legend entry, inspect each band separately for that marker color.
    - Within one band, interpret the marker pixels exactly as in the original task: a 2x2 block gets recolored,
      while an L-pair defines the opposite diagonal whose diamond is filled with the legend color.
    - Earlier fills must still preserve legend cells and all pixels of later legend markers.

    Key insight: Background color is no longer a single global "empty" value; it partitions the grid into bands,
    and the legend mapping must be applied independently inside each band where the marker color appears.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_bgs = _row_backgrounds(g)
    bands = _find_bands(row_bgs)
    assert 2 <= len(bands) <= 3, "variant expects 2-3 horizontal background bands"

    legend_c, entries = _find_legend(g, row_bgs)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        if legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    band_masks: list[np.ndarray] = []
    for r0, r1, _ in bands:
        band_mask = np.zeros_like(g, dtype=bool)
        band_mask[r0 : r1 + 1, :] = True
        band_masks.append(band_mask)

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    # Process legend entries bottom -> top; earlier entries must not destroy later marker pixels.
    for i, (_, marker, fill) in enumerate(entries):
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        for band_mask in band_masks:
            marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask) & band_mask)))
            if len(marker_pts) < 2:
                continue

            rs = [r for r, _ in marker_pts]
            cs = [c for _, c in marker_pts]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)

            if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
                block = np.zeros_like(g, dtype=bool)
                for r, c in marker_pts:
                    block[r, c] = True
                out[block & (~protect)] = int(fill)
                continue

            local_protect = protect | ((g == marker) & (~legend_mask))
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
            out[diamond & band_mask & (~local_protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (11, 19)         # grid height bounds
GRID_W_RANGE = (10, 19)         # grid width bounds

# ── CFG: background bands ──────────────────────────────────────────
BAND_COUNT_CHOICES = [2, 3]     # number of horizontal background regions
MIN_BAND_HEIGHT = 3             # minimum height for non-bottom bands
MIN_BOTTOM_BAND_HEIGHT = 5      # reserve room for the legend

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)       # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)     # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.18                # prob of one legend entry with no non-legend instances
BLOCK_PROB = 0.22               # prob that a reused marker uses 2x2 blocks

# ── CFG: band reuse ────────────────────────────────────────────────
MULTI_BAND_REUSE_PROB = 0.75    # non-forced entries often appear in multiple bands too

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                 # max bbox height
BBOX_W_MAX = 12                 # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                 # prob of TL-BR diagonal (vs TR-BL)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.20          # prob of adding a non-legend object

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100             # placement attempts for a 2x2 block
LPAIR_RETRIES = 150             # placement attempts for an L-pair
DISTRACTOR_RETRIES = 60         # placement attempts for distractor
GEN_RETRIES = 400               # outer generate_input retry budget

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
        h_hi = min(BBOX_H_MAX, H)
        w_hi = min(BBOX_W_MAX, W)
        if h_hi < 3 or w_hi < 3:
            return None
        bh = random.randint(3, h_hi)
        bw = random.randint(3, w_hi)
    elif shape_choice == "h2_w_ge4":
        w_hi = min(BBOX_W_MAX, W)
        if H < 2 or w_hi < 4:
            return None
        bh = 2
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4 or W < 2:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _sample_band_heights(H: int, n_bands: int, bottom_min: int) -> list[int] | None:
    """Sample contiguous horizontal band heights."""
    mins = [MIN_BAND_HEIGHT] * n_bands
    mins[-1] = bottom_min
    remaining = H - sum(mins)
    if remaining < 0:
        return None
    heights = mins[:]
    for _ in range(remaining):
        heights[random.randrange(n_bands)] += 1
    return heights


def _build_background(H: int, W: int, n_entries: int):
    """Create the striped background and bottom band metadata."""
    n_bands = random.choice(BAND_COUNT_CHOICES)
    band_heights = _sample_band_heights(H, n_bands, max(n_entries, MIN_BOTTOM_BAND_HEIGHT))
    if band_heights is None:
        return None

    band_colors = random.sample(list(range(10)), n_bands)
    grid = np.zeros((H, W), dtype=int)
    bands: list[tuple[int, int, int]] = []
    r0 = 0
    for height, color in zip(band_heights, band_colors):
        r1 = r0 + height - 1
        grid[r0 : r1 + 1, :] = color
        bands.append((r0, r1, color))
        r0 = r1 + 1
    return grid, bands, set(band_colors)


def _build_legend(grid: np.ndarray, bands: list[tuple[int, int, int]], band_colors: set[int], n_entries: int):
    """Place the legend inside the bottom background band."""
    H, W = grid.shape
    legend_c = random.randint(*LEGEND_COL_RANGE)
    bottom_r0, bottom_r1, bottom_bg = bands[-1]
    legend_bot = bottom_r1
    if legend_bot - n_entries + 1 < bottom_r0:
        return None

    marker_pool = [c for c in range(10) if c not in band_colors]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

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

    return grid, legend_rows, legend_mask, legend_c, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    """Protect the left strip so the legend remains the first non-background content from the bottom."""
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _try_place_l_pair_in_band(
    band_r0: int, band_r1: int, W: int, color: int, retries: int, can_place, do_place
) -> bool:
    """Try to place an L-pair fully inside one horizontal band."""
    H_band = band_r1 - band_r0 + 1
    for _ in range(retries):
        size = _pick_bbox_size(H_band, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(band_r0, band_r1 - bh + 1)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _try_place_block_in_band(
    band_r0: int, band_r1: int, W: int, color: int, retries: int, can_place, do_place
) -> bool:
    """Try to place a 2x2 block fully inside one horizontal band."""
    if band_r1 - band_r0 + 1 < 2 or W < 2:
        return False
    for _ in range(retries):
        r0 = random.randint(band_r0, band_r1 - 1)
        c0 = random.randint(0, W - 2)
        cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _choose_active_bands(n_bands: int, force_multi: bool) -> list[int]:
    """Choose which bands receive a non-legend instance of a marker color."""
    if force_multi:
        k = 2 if n_bands == 2 else random.choice([2, 3])
    else:
        if random.random() < MULTI_BAND_REUSE_PROB:
            k = 2 if n_bands == 2 else random.choice([2, 3])
        else:
            k = 1
    return sorted(random.sample(range(n_bands), k))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)

    bg_result = _build_background(H, W, n_entries)
    if bg_result is None:
        return None
    grid, bands, band_colors = bg_result

    legend_result = _build_legend(grid, bands, band_colors, n_entries)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, marker_colors = legend_result
    legend_bot = legend_rows[0][0]

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

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

    forced_multi_idx = random.randrange(len(legend_rows))
    noop_idx = None
    if random.random() < NOOP_PROB and len(legend_rows) > 1:
        candidates = [i for i in range(len(legend_rows)) if i != forced_multi_idx]
        noop_idx = random.choice(candidates)

    for i, (_, marker, _) in enumerate(legend_rows):
        if i == noop_idx:
            continue

        active_bands = _choose_active_bands(len(bands), force_multi=(i == forced_multi_idx))
        use_block = random.random() < BLOCK_PROB
        for band_idx in active_bands:
            band_r0, band_r1, _ = bands[band_idx]
            if use_block:
                placed = _try_place_block_in_band(
                    band_r0, band_r1, W, marker, BLOCK_RETRIES, can_place, do_place
                )
            else:
                placed = _try_place_l_pair_in_band(
                    band_r0, band_r1, W, marker, LPAIR_RETRIES, can_place, do_place
                )
            if not placed:
                return None

    if random.random() < DISTRACTOR_PROB:
        distractor_pool = [c for c in range(10) if c not in band_colors and c not in marker_colors]
        if distractor_pool:
            band_idx = random.randrange(len(bands))
            band_r0, band_r1, _ = bands[band_idx]
            distractor_color = random.choice(distractor_pool)
            if random.random() < 0.5:
                _try_place_block_in_band(
                    band_r0, band_r1, W, distractor_color, DISTRACTOR_RETRIES, can_place, do_place
                )
            else:
                _try_place_l_pair_in_band(
                    band_r0, band_r1, W, distractor_color, DISTRACTOR_RETRIES, can_place, do_place
                )

    row_bgs = _row_backgrounds(grid)
    parsed_bands = _find_bands(row_bgs)
    if len(parsed_bands) != len(bands):
        return None
    if [bg for _, _, bg in parsed_bands] != [bg for _, _, bg in bands]:
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
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
