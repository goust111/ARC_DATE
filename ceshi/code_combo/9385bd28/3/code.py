# Combo variant of 9385bd28 (index 3)
# combo_spec: D4/2+D5/1+D17/0
# Name: 三列图例编码条纹方向+从上到下阅读

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant:
# a) Concrete behavior and trace
#    - The visible input keeps the same family structure as the base task: a bottom-left
#      legend plus scattered non-legend markers. The fused change is that each legend row
#      is now three cells wide: (marker_color, fill_color, direction_code_color).
#    - The legend is read from top to bottom. This changes which entry wins on overlap.
#    - The first two columns behave like before: marker color identifies the object and
#      fill color is the pigment used inside its target diamond. The third column is new:
#      its color is a fixed symbolic code for one of three stripe directions:
#          7 -> horizontal stripes
#          8 -> vertical stripes
#          9 -> diagonal stripes using (row_offset + col_offset) parity
#    - Example trace on a concrete pair of overlapping entries:
#        top legend row    = (red marker, yellow fill, code 8)
#        bottom legend row = (blue marker, green fill, code 7)
#      Original task would read bottom-up and do solid fills. This combo instead reads the
#      red row first, paints a vertically striped diamond, then reads the blue row later and
#      overwrites the overlap with horizontal stripes. The same marker geometry therefore
#      differs from the base task in two places at once: fill texture and overwrite order.
#    - The diagonal code is intentionally a third discrete visible option, not something
#      inferred from bbox geometry. A slash-like alternating pattern appears wherever code 9
#      is used, even if the bbox is tall or wide.
#
# b) Complexity calibration
#    - Only three stripe modes are allowed. That keeps the codebook small enough for a human
#      to infer from a few examples while still being richer than the single-axis D5 variant.
#    - The generator enforces at least three active L-pairs so all three direction codes can
#      appear in one sample family. It also forces overlap between the topmost and bottommost
#      active entries so the changed reading order is actually observable.
#    - To prevent the puzzle from collapsing back into the geometry-driven D5 hypothesis, the
#      generator deliberately places at least one horizontal-coded wide bbox and one vertical-
#      coded tall bbox. That visible contradiction tells the solver to read the legend's third
#      column instead of guessing direction from aspect ratio.
#
# c) Implicit-assumption audit of the original main()
#    - Original `_find_legend()` assumed 2 legend columns and bottom->top semantics. Both are
#      false now, so the parser must require 3 columns and reverse the scan into top->bottom
#      reading order.
#    - Original `out[diamond & (~protect)] = fill` assumed solid repaint. That no longer holds
#      because stripe gaps must become background explicitly, even if earlier fills touched the
#      same cells. The fused solver therefore clears the writable target region to background
#      first, then paints only the stripe subset.
#    - Original conflict protection treated "future markers" as legend rows above the current
#      row. With top-down reading, future markers are lower rows instead. The protect mask must
#      follow processing order rather than physical row direction.
#    - Original logic had no reason to inspect a third legend column. The combo introduces a
#      new assumption: every legend row's third cell is one of the reserved direction-code
#      colors, and generate_input() must guarantee that.
#
# d) Placement legality
#    - Non-legend marker pixels still cannot overlap each other or the legend, and all cells
#      remain in-bounds.
#    - The legend scan still relies on a clean lower-left stack, so the generator protects the
#      region below the legend and columns <= legend_c above it exactly as in the base task.
#    - Top and bottom required L-pairs are forced to have overlapping writable diamonds, but
#      their marker pixels remain disjoint. This preserves object detectability while making the
#      top-down order visible in the output.
#
# e) Visual observability
#    - A human can see three separate legend columns, repeated third-column code colors, and
#      three distinct stripe textures. Because the same code colors recur across samples, the
#      direction mapping is learnable from the examples rather than hidden in numeric ids.
#    - The generator rejects samples unless the combo output differs both from a bottom-up
#      striped solver and from an aspect-ratio-striped solver. That guards against accidental
#      degeneration into either the base ordering or the D5 single-variant hypothesis.
#
# f) Visual cleanliness
#    - Only one controlled overlap is required; the rest of the scene is placed normally.
#      This keeps the board readable instead of making every diamond compete in the same area.
#    - Markers themselves are still sparse L-corners or rare 2x2 blocks, so the added legend
#      complexity does not turn the input into noise.
#
# g) Color-change inferability
#    - Marker and fill colors stay in a small palette separate from the reserved direction-code
#      colors. That prevents the third legend column from being confused with ordinary fill data.
#    - The background is also kept out of the reserved code palette, so every direction code is
#      visibly present. Fill colors are chosen away from background to keep the alternating bands
#      legible.
#
# New solver assumptions introduced by this combo and matched by generate_input():
#    - The legend is a contiguous 3-column vertical stack near the lower-left, read top->bottom.
#    - Every legend row's third cell is one of the fixed direction-code colors 7/8/9.
#    - There are at least three active L-pairs, including one diagonal-coded pair.
#    - At least one horizontal-coded pair is wide and at least one vertical-coded pair is tall,
#      so legend-coded direction cannot be mistaken for aspect-coded direction.
#    - The topmost and bottommost active L-pairs overlap enough that reversing processing order
#      changes multiple output cells.


HORIZONTAL_CODE = 7
VERTICAL_CODE = 8
DIAGONAL_CODE = 9

ORIENTATION_FROM_CODE = {
    HORIZONTAL_CODE: "horizontal",
    VERTICAL_CODE: "vertical",
    DIAGONAL_CODE: "diagonal",
}
CODE_FROM_ORIENTATION = {v: k for k, v in ORIENTATION_FROM_CODE.items()}
ORIENTATIONS = tuple(CODE_FROM_ORIENTATION.keys())

DATA_COLORS = tuple(range(7))

MIN_REQUIRED_L_PAIRS = 3
MIN_TOP_BOTTOM_OVERLAP = 4
MIN_ORDER_DIFF_CELLS = 4
MIN_CODE_DIFF_CELLS = 6
MIN_HV_CONTRADICTIONS = 2


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Legend is a vertical stack of triples at bottom-left.
    Each legend row: (marker_color at col c, fill_color at c+1, direction_code at c+2).
    Returns (legend_col, entries) where entries are ordered top->bottom as (row, marker, fill, code).
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

    bottom_up: list[tuple[int, int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1])
        code = int(g[r, c + 2])
        assert code in ORIENTATION_FROM_CODE, "legend direction code must be one of the reserved colors"
        bottom_up.append((r, marker, fill, code))
    return c, list(reversed(bottom_up))


def _build_legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
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

    This checks the anchor structure directly, which avoids bbox-corner ambiguity.
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
    Mask for cells inside [r0..r1]x[c0..c1] that lie on a shortest 4-neigh path between a and b.
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


def _legend_orientation(
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    orient_code: int,
) -> str:
    del rmin, rmax, cmin, cmax
    return ORIENTATION_FROM_CODE[orient_code]


def _aspect_orientation(
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    orient_code: int,
) -> str:
    del orient_code
    height = rmax - rmin + 1
    width = cmax - cmin + 1
    return "horizontal" if height >= width else "vertical"


def _paint_striped_region(
    out: np.ndarray,
    diamond: np.ndarray,
    protect: np.ndarray,
    fill: int,
    bg: int,
    orientation: str,
    rmin: int,
    cmin: int,
) -> None:
    """Paint alternating fill/bg stripes on the writable part of `diamond`."""
    paint_mask = diamond & (~protect)
    if not np.any(paint_mask):
        return

    out[paint_mask] = bg
    row_offsets = np.arange(out.shape[0])[:, None] - rmin
    col_offsets = np.arange(out.shape[1])[None, :] - cmin
    if orientation == "horizontal":
        stripe_fill = paint_mask & ((row_offsets % 2) == 0)
    elif orientation == "vertical":
        stripe_fill = paint_mask & ((col_offsets % 2) == 0)
    else:
        stripe_fill = paint_mask & (((row_offsets + col_offsets) % 2) == 0)
    out[stripe_fill] = fill


def _solve_from_entries(
    g: np.ndarray,
    entries: list[tuple[int, int, int, int]],
    legend_mask: np.ndarray,
    orientation_fn,
) -> np.ndarray:
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    marker_order = [marker for _, marker, _, _ in entries]

    for i, (_, marker, fill, orient_code) in enumerate(entries):
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
        assert score_tlbr != score_trbl, "diagonal scores must not tie"
        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        orientation = orientation_fn(rmin, rmax, cmin, cmax, orient_code)
        _paint_striped_region(out, diamond, protect, eff_fill, bg, orientation, rmin, cmin)

    return out


def _entry_geometry_metrics(
    g: np.ndarray,
    entries: list[tuple[int, int, int, int]],
    legend_mask: np.ndarray,
) -> dict[str, int]:
    active_l_pairs = 0
    diagonal_l_pairs = 0
    hv_contradictions = 0

    for _, marker, _, orient_code in entries:
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            continue

        active_l_pairs += 1
        legend_orientation = ORIENTATION_FROM_CODE[orient_code]
        aspect_orientation = _aspect_orientation(rmin, rmax, cmin, cmax, orient_code)
        if legend_orientation == "diagonal":
            diagonal_l_pairs += 1
        elif legend_orientation != aspect_orientation:
            hv_contradictions += 1

    return {
        "active_l_pairs": active_l_pairs,
        "diagonal_l_pairs": diagonal_l_pairs,
        "hv_contradictions": hv_contradictions,
    }


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the combo-specific direction coding and top-down order are active."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _build_legend_mask(g.shape, legend_c, entries)

    combo_out = _solve_from_entries(g, entries, legend_mask, _legend_orientation)
    bottom_up_out = _solve_from_entries(g, list(reversed(entries)), legend_mask, _legend_orientation)
    aspect_out = _solve_from_entries(g, entries, legend_mask, _aspect_orientation)
    geom = _entry_geometry_metrics(g, entries, legend_mask)

    return {
        "order_diff_cells": int(np.count_nonzero(combo_out != bottom_up_out)),
        "code_diff_cells": int(np.count_nonzero(combo_out != aspect_out)),
        **geom,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each marker-defined diamond using a top-down 3-column legend with striped fills.

    Input structure:
    - A mostly uniform grid contains a vertical legend near the lower-left whose rows are triples:
      (marker_color, fill_color, direction_code_color).
    - Away from the legend, each marker color appears either as two opposite L-corners of a bbox or
      as a solid 2x2 block.
    - The direction-code colors are fixed symbolic tokens: one means horizontal stripes, one vertical
      stripes, and one diagonal stripes.

    Rule:
    - Read the legend from top row to bottom row.
    - For each row, gather the non-legend pixels of that marker color and infer which bbox diagonal
      carries the L-anchors; the opposite diagonal defines the diamond target as in the base task.
    - If the marker is an L-pair, repaint the writable diamond with alternating fill/background stripes
      whose direction comes from the legend's third column, not from the bbox shape.
    - If the marker is a solid 2x2 block, recolor only that block; legend cells and markers belonging
      to rows still to be processed lower in the legend stay protected.

    Key insight:
    - The first two legend columns still answer "which object?" and "which fill color?", but the third
      legend column now supplies a second instruction: how the target diamond is striped. Overlaps are
      then resolved by the top-to-bottom reading order, so lower legend rows can overwrite earlier
      striped patterns.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one triple"
    legend_mask = _build_legend_mask(g.shape, legend_c, entries)
    return _solve_from_entries(g, entries, legend_mask, _legend_orientation)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (3, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.15
BLOCK_PROB = 0.15

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.70, 0.15, 0.15]
TLBR_PROB = 0.8
BBOX_PICK_RETRIES = 50

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

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
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int, preferred_shape: str | None = None) -> tuple[int, int] | None:
    for _ in range(BBOX_PICK_RETRIES):
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
                continue
            bw = random.randint(4, w_hi)
        else:
            h_hi = min(BBOX_H_MAX, H)
            if h_hi < 4:
                continue
            bh = random.randint(4, h_hi)
            bw = 2

        if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
            continue
        if preferred_shape == "tall" and bh <= bw:
            continue
        if preferred_shape == "wide" and bw <= bh:
            continue
        return bh, bw
    return None


def _l_pair_target_mask(H: int, W: int, rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> np.ndarray:
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    mask = _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)
    for r, c in _l_pair_cells(rmin, cmin, rmax, cmax, diag):
        mask[r, c] = False
    return mask


def _shape_name(height: int, width: int) -> str:
    if height > width:
        return "tall"
    if width > height:
        return "wide"
    return "square"


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    preferred_shape: str | None = None,
    overlap_mask: np.ndarray | None = None,
    min_overlap: int = 0,
) -> dict | None:
    for _ in range(retries):
        size = _pick_bbox_size(H, W, preferred_shape)
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
        target_mask = _l_pair_target_mask(H, W, rmin, cmin, rmax, cmax, diag)
        if overlap_mask is not None and int(np.count_nonzero(target_mask & overlap_mask)) < min_overlap:
            continue
        do_place(cells, color)
        return {
            "cells": cells,
            "target_mask": target_mask,
            "bbox": (rmin, cmin, rmax, cmax),
            "diag": diag,
            "shape": _shape_name(bh, bw),
        }
    return None


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    legend_top = legend_bot - n_entries + 1
    if legend_top < 0:
        return None

    data_pool = [c for c in DATA_COLORS if c != bg]
    if len(data_pool) < n_entries:
        return None
    marker_colors = random.sample(data_pool, n_entries)

    required_mid = n_entries // 2
    required_indices = [0, required_mid, n_entries - 1]
    required_orientations = random.sample(list(ORIENTATIONS), 3)
    orientation_by_index: dict[int, str] = {}
    for idx, orientation in zip(required_indices, required_orientations):
        orientation_by_index[idx] = orientation
    for idx in range(n_entries):
        orientation_by_index.setdefault(idx, random.choice(list(ORIENTATIONS)))

    fills: list[int] = []
    for i, marker in enumerate(marker_colors):
        fill_pool = [c for c in data_pool if c != marker]
        if not fill_pool:
            return None
        fills.append(random.choice(fill_pool))
    if fills[0] == fills[-1]:
        alt = [c for c in data_pool if c != marker_colors[-1] and c != fills[0]]
        if not alt:
            return None
        fills[-1] = random.choice(alt)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_top + i
        marker = marker_colors[i]
        fill = fills[i]
        code = CODE_FROM_ORIENTATION[orientation_by_index[i]]
        legend_rows.append((r, marker, fill, code))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        grid[r, legend_c + 2] = code
        legend_mask[r, legend_c : legend_c + 3] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, required_indices


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _assign_marker_types(n_entries: int, locked_indices: set[int]) -> list[str]:
    types = ["L"] * n_entries
    idx_pool = [i for i in range(n_entries) if i not in locked_indices]
    random.shuffle(idx_pool)

    l_remaining = n_entries
    if idx_pool and random.random() < NOOP_PROB and l_remaining - 1 >= MIN_REQUIRED_L_PAIRS:
        types[idx_pool.pop()] = "noop"
        l_remaining -= 1
    if idx_pool and random.random() < BLOCK_PROB and l_remaining - 1 >= MIN_REQUIRED_L_PAIRS:
        types[idx_pool.pop()] = "block"
        l_remaining -= 1
    return types


def _preferred_shape_for_orientation(orientation: str) -> str:
    if orientation == "horizontal":
        return "wide"
    if orientation == "vertical":
        return "tall"
    return random.choice(["tall", "wide"])


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(DATA_COLORS)

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
        required_indices,
    ) = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    locked_indices = set(required_indices)
    types = _assign_marker_types(len(legend_rows), locked_indices)
    placement_meta: dict[int, dict] = {}

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

    top_idx = 0
    mid_idx = required_indices[1]
    bottom_idx = len(legend_rows) - 1

    for idx in [top_idx, bottom_idx]:
        _, marker, _, code = legend_rows[idx]
        orientation = ORIENTATION_FROM_CODE[code]
        preferred_shape = _preferred_shape_for_orientation(orientation)
        overlap_mask = placement_meta[top_idx]["target_mask"] if idx == bottom_idx else None
        meta = _try_place_l_pair(
            H,
            W,
            marker,
            LPAIR_RETRIES,
            can_place,
            do_place,
            preferred_shape=preferred_shape,
            overlap_mask=overlap_mask,
            min_overlap=MIN_TOP_BOTTOM_OVERLAP if idx == bottom_idx else 0,
        )
        if meta is None:
            return None
        placement_meta[idx] = meta

    if mid_idx not in placement_meta:
        _, marker, _, code = legend_rows[mid_idx]
        orientation = ORIENTATION_FROM_CODE[code]
        meta = _try_place_l_pair(
            H,
            W,
            marker,
            LPAIR_RETRIES,
            can_place,
            do_place,
            preferred_shape=_preferred_shape_for_orientation(orientation),
        )
        if meta is None:
            return None
        placement_meta[mid_idx] = meta

    for i, (_, marker, _, code) in enumerate(legend_rows):
        if i in placement_meta or types[i] == "noop":
            continue
        if types[i] == "block":
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
            orientation = ORIENTATION_FROM_CODE[code]
            meta = _try_place_l_pair(
                H,
                W,
                marker,
                LPAIR_RETRIES,
                can_place,
                do_place,
                preferred_shape=_preferred_shape_for_orientation(orientation),
            )
            if meta is None:
                return None
            placement_meta[i] = meta

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in DATA_COLORS if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(
                H,
                W,
                random.choice(avail),
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
                preferred_shape=random.choice(["tall", "wide"]),
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    metrics = behavior_metrics(grid)
    if metrics["active_l_pairs"] < MIN_REQUIRED_L_PAIRS:
        return None
    if metrics["diagonal_l_pairs"] < 1:
        return None
    if metrics["hv_contradictions"] < MIN_HV_CONTRADICTIONS:
        return None
    if metrics["order_diff_cells"] < MIN_ORDER_DIFF_CELLS:
        return None
    if metrics["code_diff_cells"] < MIN_CODE_DIFF_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
