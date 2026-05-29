# Combo variant of 9385bd28 (index 9)
# combo_spec: D4/0+D5/3+D16/1
# Name: 多背景色分区+标记填充互换+任意位置图例

import random
from collections import Counter

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
# - The canvas is split into 2-3 full-width horizontal background bands. Row-majority color is therefore
#   the local background, not a single global value.
# - The legend remains a vertical 2-column strip, but it can be attached to either vertical edge and can
#   sit inside any one band rather than only the bottom-left corner.
# - The displayed legend semantics are reversed relative to the base task: the left cell of a legend row is
#   the actual fill/swap color, while the right cell is the actual marker color to search for in the scene.
# - Example trace for one entry:
#   1. Base task: legend row visually "(marker, fill)" and the solver finds marker-colored L-corners globally.
#   2. This combo: the same visual row must be read as "(fill, marker)", the solver finds right-cell-colored
#      markers separately inside each background band, then applies the D5 swap rule.
#   3. Inside one band, marker pixels recolor to the fill color; pre-existing non-background payload pixels
#      inside the target diamond flip back to the marker color; empty band-background cells become fill color.
# - Difference from the original is therefore threefold in one chain: locate the edge legend first, invert
#   the legend roles, then process same-color objects independently per background partition.
#
# complexity_calibration:
# - Bands are kept horizontal and full-width so humans can reliably infer the partition rule from a few examples.
# - The legend is constrained to the left or right edge, which still creates the intended search problem without
#   introducing rotation-dependent reading orders.
# - Every generated L-pair receives visible payload pixels of the legend's actual fill color, so the color-swap
#   behavior is repeated enough to notice.
# - At least one legend entry is reused across multiple bands, ensuring the partitioned processing is visible and
#   preventing the task from collapsing into a single-object interpretation.
#
# implicit_assumptions_review:
# - Original `get_bg(...)` is invalid because there is no single background color. We instead infer row-wise
#   backgrounds by majority vote, then compress them into contiguous bands.
# - Original `_find_legend(...)` assumed the first bottom-left non-bg pixel starts the legend. Here that fails
#   because the legend may be on either edge and in any band. The new parser searches both edge pairs for the
#   unique contiguous run of non-row-background rows.
# - Original legend rows were interpreted as `(marker, fill)`. That assumption is deliberately broken: the new
#   solver treats them as `(fill, marker)`.
# - Original marker collection was global per color. That fails when the same actual marker color appears once in
#   several bands, so the solver now processes each band separately.
# - Original fill protected current marker pixels. D5/3 breaks that assumption because marker pixels must now be
#   writable and swap to the fill color.
#
# placement_legality:
# - The generator reserves both edge pairs for legend usage and only releases the actual legend cells, preventing
#   fake legends and keeping the border visually clean.
# - All scene objects live in interior columns `2..W-3`, so they cannot touch the legend strip.
# - Each object reserves its full bbox, so different entries cannot overlap or partially nest in ways that would
#   corrupt anchor detection or payload swapping.
# - Each marker color appears at most once per band, and every object lies fully within one band.
#
# observability_and_cleanliness:
# - The three fused signals are all directly visible: multiple background bands, an off-corner edge legend, and
#   payload pixels that flip opposite to the marker corners.
# - Fill colors are chosen outside both the band palette and the actual marker palette, so payload is visible and
#   never masquerades as another marker family.
# - Distractors, when present, use unrelated colors and stay outside reserved bboxes.
#
# solver_assumptions_and_generator_guarantees:
# - Row-majority equals the intended background for every row.
# - Background bands are contiguous row runs and there are exactly 2 or 3 of them.
# - The legend is the only contiguous edge strip whose two columns both differ from the local row background.
# - Every active L entry has at least one payload cell in its target diamond, and at least one entry reuses the
#   same actual marker color in multiple bands.


def _row_backgrounds(g: np.ndarray) -> list[int]:
    """Infer the background color of each row by majority vote."""
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


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive row runs where a 1D boolean mask is True."""
    runs: list[tuple[int, int]] = []
    h = int(mask.shape[0])
    r = 0
    while r < h:
        if not bool(mask[r]):
            r += 1
            continue
        r0 = r
        while r + 1 < h and bool(mask[r + 1]):
            r += 1
        runs.append((r0, r))
        r += 1
    return runs


def _find_legend(g: np.ndarray, row_bgs: list[int]) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs attached to the left or right edge.
    Each legend row is read as (fill_color at col c, marker_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, fill, marker).
    """
    h, w = g.shape
    assert w >= 4, "grid must be at least 4 columns wide"

    candidate_cols = [0]
    if w - 2 != 0:
        candidate_cols.append(w - 2)

    candidates: list[tuple[int, list[tuple[int, int, int]]]] = []
    for c in candidate_cols:
        row_mask = np.array(
            [(g[r, c] != row_bgs[r]) and (g[r, c + 1] != row_bgs[r]) for r in range(h)],
            dtype=bool,
        )
        for r0, r1 in _contiguous_runs(row_mask):
            if r1 - r0 + 1 < 2:
                continue
            entries = [(r, int(g[r, c]), int(g[r, c + 1])) for r in range(r1, r0 - 1, -1)]
            candidates.append((c, entries))

    assert len(candidates) == 1, "legend must be the unique contiguous edge stack"
    return candidates[0]


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
    Mask for cells inside [r0..r1]x[c0..c1] that lie on a shortest 4-neigh path between corners a and b.
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


def _fill_endpoints_from_marker_diag(
    rmin: int, cmin: int, rmax: int, cmax: int, marker_diag: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert the anchored marker diagonal into the opposite fill diamond endpoints."""
    if marker_diag == "TLBR":
        return (rmin, cmax), (rmax, cmin)
    return (rmin, cmin), (rmax, cmax)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the diagonal-marker repaint task by combining band-local backgrounds, reversed legend semantics,
    and swap-style filling.

    Input structure:
    - The grid is partitioned into 2-3 horizontal background bands with different row-majority colors.
    - A vertical two-column legend is attached to either the left or right edge and may sit inside any band.
    - Each legend row must be interpreted as `(fill_color, marker_color)`, not `(marker_color, fill_color)`.
    - Away from the legend, each actual marker color appears per band either as one diagonal L-pair or as one
      solid 2x2 block, and L-pairs may contain payload pixels of the actual fill color inside their target diamond.

    Rule:
    - Infer the row backgrounds, compress them into bands, then locate the unique edge legend.
    - Read legend rows from bottom to top; for each row, use the right-cell color to find markers and the left-cell
      color as the repaint target.
    - Process each band independently: a 2x2 block simply recolors to the fill color, while an L-pair identifies
      the opposite diagonal diamond.
    - In that diamond, marker pixels become the fill color, pre-existing non-band-background pixels become the
      marker color, and remaining background cells become the fill color.
    - Never overwrite legend cells or pixels belonging to marker colors from later legend rows.

    Key insight: the legend no longer directly names the searched marker color, and the same marker color can recur
    in several background bands, so the solver must first find the edge legend, invert each row's meaning, and only
    then apply the color swap separately inside each partition.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_bgs = _row_backgrounds(g)
    bands = _find_bands(row_bgs)
    assert 2 <= len(bands) <= 3, "variant expects 2-3 horizontal background bands"

    legend_c, entries = _find_legend(g, row_bgs)
    assert entries, "legend must contain at least one (fill, marker) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    band_masks: list[tuple[np.ndarray, int]] = []
    for r0, r1, band_bg in bands:
        band_mask = np.zeros_like(g, dtype=bool)
        band_mask[r0 : r1 + 1, :] = True
        band_masks.append((band_mask, band_bg))

    out = g.copy()
    marker_order = [marker for _, _, marker in entries]

    for i, (_, fill, marker) in enumerate(entries):
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        for band_mask, band_bg in band_masks:
            current_marker_mask = (g == marker) & (~legend_mask) & band_mask
            marker_pts = set(map(tuple, np.argwhere(current_marker_mask)))
            if len(marker_pts) < 2:
                continue

            rs = [r for r, _ in marker_pts]
            cs = [c for _, c in marker_pts]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)

            if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
                out[current_marker_mask & (~protect)] = int(fill)
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
            assert score_tlbr != score_trbl, "diagonal scores must not tie"

            marker_diag = "TLBR" if score_tlbr > score_trbl else "TRBL"
            a, b = _fill_endpoints_from_marker_diag(rmin, cmin, rmax, cmax, marker_diag)
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b) & band_mask

            swap_region = diamond & (~protect) & (~current_marker_mask)
            swap_existing = swap_region & (g != band_bg)
            swap_background = swap_region & (g == band_bg)

            out[current_marker_mask & (~protect)] = int(fill)
            out[swap_background] = int(fill)
            out[swap_existing] = int(marker)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (11, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: background bands ──────────────────────────────────────────
BAND_COUNT_CHOICES = [2, 3]
MIN_BAND_HEIGHT = 3
MIN_LEGEND_BAND_HEIGHT = 5

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_ENTRY_RANGE = (2, 5)
LEGEND_SIDE_WEIGHTS = [0.5, 0.5]  # left / right
LEGEND_POS_WEIGHTS = [0.35, 0.30, 0.35]  # top / bottom / middle within the chosen band

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.15
BLOCK_PROB = 0.20
MULTI_BAND_REUSE_PROB = 0.65

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: payload for visible swap ───────────────────────────────────
PAYLOAD_MIN_CELLS = 1
PAYLOAD_MAX_CELLS = 3
PAYLOAD_RETRIES = 30

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.20

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 180
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 500

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


def _sample_band_heights(H: int, n_bands: int, legend_band_idx: int, n_entries: int) -> list[int] | None:
    """Sample band heights while reserving enough rows for the chosen legend band."""
    mins = [MIN_BAND_HEIGHT] * n_bands
    mins[legend_band_idx] = max(MIN_LEGEND_BAND_HEIGHT, n_entries)
    remaining = H - sum(mins)
    if remaining < 0:
        return None
    heights = mins[:]
    for _ in range(remaining):
        heights[random.randrange(n_bands)] += 1
    return heights


def _build_background(H: int, W: int, n_entries: int):
    """Create the horizontal background bands and choose which band will host the legend."""
    n_bands = random.choice(BAND_COUNT_CHOICES)
    legend_band_idx = random.randrange(n_bands)
    heights = _sample_band_heights(H, n_bands, legend_band_idx, n_entries)
    if heights is None:
        return None

    band_colors = random.sample(list(range(10)), n_bands)
    grid = np.zeros((H, W), dtype=int)
    bands: list[tuple[int, int, int]] = []
    r0 = 0
    for height, color in zip(heights, band_colors):
        r1 = r0 + height - 1
        grid[r0 : r1 + 1, :] = color
        bands.append((r0, r1, color))
        r0 = r1 + 1
    return grid, bands, set(band_colors), legend_band_idx


def _pick_legend_top(r0: int, r1: int, n_entries: int) -> int:
    """Choose a top row for the legend inside one band, covering top / bottom / middle cases."""
    max_top = r1 - n_entries + 1
    mode = random.choices(["top", "bottom", "middle"], weights=LEGEND_POS_WEIGHTS)[0]
    if mode == "top":
        return r0
    if mode == "bottom":
        return max_top
    if max_top - r0 <= 1:
        return random.randint(r0, max_top)
    return random.randint(r0 + 1, max_top - 1)


def _build_legend(
    grid: np.ndarray,
    bands: list[tuple[int, int, int]],
    band_colors: set[int],
    legend_band_idx: int,
    n_entries: int,
):
    """Place an edge legend whose displayed rows encode (fill, marker)."""
    H, W = grid.shape
    side = random.choices(["left", "right"], weights=LEGEND_SIDE_WEIGHTS)[0]
    legend_c = 0 if side == "left" else W - 2

    marker_pool = [c for c in range(10) if c not in band_colors]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

    fill_pool = [c for c in range(10) if c not in band_colors and c not in marker_colors]
    if not fill_pool:
        return None

    band_r0, band_r1, _ = bands[legend_band_idx]
    legend_top = _pick_legend_top(band_r0, band_r1, n_entries)

    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    fill_choices: list[int] = []
    for i in range(n_entries):
        r = legend_top + n_entries - 1 - i
        fill = random.choice(fill_pool)
        marker = marker_colors[i]
        legend_rows.append((r, fill, marker))
        fill_choices.append(fill)
        grid[r, legend_c] = fill
        grid[r, legend_c + 1] = marker
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, marker_colors, set(fill_choices)


def _build_forbidden(H: int, W: int, legend_c: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Reserve both edge pairs so only the actual legend occupies them."""
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[:, :2] = True
    forbidden[:, W - 2 :] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
        forbidden[r, legend_c + 1] = False
    return forbidden


def _choose_active_bands(n_bands: int, force_multi: bool) -> list[int]:
    """Choose which bands contain an instance of one legend marker color."""
    if force_multi:
        k = 2 if n_bands == 2 else random.choice([2, 3])
    else:
        if random.random() < MULTI_BAND_REUSE_PROB:
            k = 2 if n_bands == 2 else random.choice([2, 3])
        else:
            k = 1
    return sorted(random.sample(range(n_bands), k))


def _try_place_l_pair_in_band(
    band_r0: int,
    band_r1: int,
    c_lo: int,
    c_hi: int,
    color: int,
    retries: int,
    can_place,
    do_place,
):
    """Try to place an L-pair fully inside one band; return placement metadata on success."""
    H_band = band_r1 - band_r0 + 1
    usable_W = c_hi - c_lo + 1
    for _ in range(retries):
        size = _pick_bbox_size(H_band, usable_W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(band_r0, band_r1 - bh + 1)
        cmin = random.randint(c_lo, c_hi - bw + 1)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if can_place(cells, rmin, cmin, rmax, cmax):
            do_place(cells, color, rmin, cmin, rmax, cmax)
            return {
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmax,
                "cmax": cmax,
                "diag": diag,
                "cells": cells,
            }
    return None


def _try_place_block_in_band(
    band_r0: int,
    band_r1: int,
    c_lo: int,
    c_hi: int,
    color: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    """Try to place a 2x2 block fully inside one band."""
    if band_r1 - band_r0 + 1 < 2 or c_hi - c_lo + 1 < 2:
        return False
    for _ in range(retries):
        r0 = random.randint(band_r0, band_r1 - 1)
        c0 = random.randint(c_lo, c_hi - 1)
        r1, c1 = r0 + 1, c0 + 1
        cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if can_place(cells, r0, c0, r1, c1):
            do_place(cells, color, r0, c0, r1, c1)
            return True
    return False


def _diamond_payload_candidates(H: int, W: int, placement: dict) -> list[tuple[int, int]]:
    """List payload candidate cells inside the swap diamond, excluding marker cells."""
    rmin = int(placement["rmin"])
    cmin = int(placement["cmin"])
    rmax = int(placement["rmax"])
    cmax = int(placement["cmax"])
    diag = str(placement["diag"])
    a, b = _fill_endpoints_from_marker_diag(rmin, cmin, rmax, cmax, diag)
    diamond = _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)
    marker_cells = set(placement["cells"])
    coords = np.argwhere(diamond)
    return [(int(r), int(c)) for r, c in coords if (int(r), int(c)) not in marker_cells]


def _choose_payload_cells(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Choose a small connected payload when possible so the swap is visually salient."""
    if not candidates:
        return []
    cand_set = set(candidates)
    max_n = min(PAYLOAD_MAX_CELLS, len(candidates))
    min_n = min(PAYLOAD_MIN_CELLS, max_n)
    if max_n >= 2:
        min_n = max(min_n, 2)
    for _ in range(PAYLOAD_RETRIES):
        target = random.randint(min_n, max_n)
        start = random.choice(candidates)
        chosen = [start]
        chosen_set = {start}
        frontier = [start]
        while frontier and len(chosen) < target:
            base = random.choice(frontier)
            br, bc = base
            nbrs = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (br + dr, bc + dc)
                if nxt in cand_set and nxt not in chosen_set:
                    nbrs.append(nxt)
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = random.choice(nbrs)
            chosen.append(nxt)
            chosen_set.add(nxt)
            frontier.append(nxt)
        if len(chosen) >= min_n:
            return chosen
    return [random.choice(candidates)]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)

    bg_result = _build_background(H, W, n_entries)
    if bg_result is None:
        return None
    grid, bands, band_colors, legend_band_idx = bg_result

    legend_result = _build_legend(grid, bands, band_colors, legend_band_idx, n_entries)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, marker_colors, fill_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_rows)
    occupied = legend_mask.copy()
    reserved = forbidden.copy()

    c_lo, c_hi = 2, W - 3
    if c_hi - c_lo + 1 < 2:
        return None

    def can_place(cells, r0, c0, r1, c1):
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color, r0, c0, r1, c1):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    n_entries_actual = len(legend_rows)
    types = ["L"] * n_entries_actual
    forced_multi_idx = random.randrange(n_entries_actual)
    noop_idx = None
    if random.random() < NOOP_PROB and n_entries_actual > 1:
        noop_candidates = [i for i in range(n_entries_actual) if i != forced_multi_idx]
        if noop_candidates:
            noop_idx = random.choice(noop_candidates)
            types[noop_idx] = "noop"

    block_candidates = [i for i in range(n_entries_actual) if i != forced_multi_idx and i != noop_idx]
    if random.random() < BLOCK_PROB and block_candidates:
        block_idx = random.choice(block_candidates)
        if sum(1 for t in types if t == "L") > 1:
            types[block_idx] = "block"

    active_l_pairs = 0
    payload_pixels = 0
    multi_band_instances = 0

    for i, (_, fill, marker) in enumerate(legend_rows):
        if types[i] == "noop":
            continue

        active_bands = _choose_active_bands(len(bands), force_multi=(i == forced_multi_idx))
        if len(active_bands) >= 2:
            multi_band_instances += 1

        if types[i] == "block":
            for band_idx in active_bands:
                band_r0, band_r1, _ = bands[band_idx]
                placed = _try_place_block_in_band(
                    band_r0,
                    band_r1,
                    c_lo,
                    c_hi,
                    marker,
                    BLOCK_RETRIES,
                    can_place,
                    do_place,
                )
                if not placed:
                    return None
            continue

        for band_idx in active_bands:
            band_r0, band_r1, _ = bands[band_idx]
            placement = _try_place_l_pair_in_band(
                band_r0,
                band_r1,
                c_lo,
                c_hi,
                marker,
                LPAIR_RETRIES,
                can_place,
                do_place,
            )
            if placement is None:
                return None

            candidates = [cell for cell in _diamond_payload_candidates(H, W, placement) if not occupied[cell]]
            payload = _choose_payload_cells(candidates)
            if not payload:
                return None
            for r, c in payload:
                grid[r, c] = fill
                occupied[r, c] = True
            active_l_pairs += 1
            payload_pixels += len(payload)

    if active_l_pairs == 0 or payload_pixels == 0 or multi_band_instances == 0:
        return None

    if random.random() < DISTRACTOR_PROB:
        distractor_pool = [
            c
            for c in range(10)
            if c not in band_colors and c not in marker_colors and c not in fill_colors
        ]
        if distractor_pool:
            band_idx = random.randrange(len(bands))
            band_r0, band_r1, _ = bands[band_idx]
            distractor_color = random.choice(distractor_pool)
            if random.random() < 0.5:
                _ = _try_place_block_in_band(
                    band_r0,
                    band_r1,
                    c_lo,
                    c_hi,
                    distractor_color,
                    DISTRACTOR_RETRIES,
                    can_place,
                    do_place,
                )
            else:
                _ = _try_place_l_pair_in_band(
                    band_r0,
                    band_r1,
                    c_lo,
                    c_hi,
                    distractor_color,
                    DISTRACTOR_RETRIES,
                    can_place,
                    do_place,
                )

    row_bgs = _row_backgrounds(grid)
    parsed_bands = _find_bands(row_bgs)
    if len(parsed_bands) != len(bands):
        return None
    if [bg for _, _, bg in parsed_bands] != [bg for _, _, bg in bands]:
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
