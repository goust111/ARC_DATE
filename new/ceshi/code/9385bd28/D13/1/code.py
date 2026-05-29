# Variant of 9385bd28
# Dimension: D13 — 标记保护机制 (meta)
# Change: 填充时保护后续图例条目的标记像素和图例区域不被覆盖... → 保护所有已有非背景像素

"""
Design analysis for the D13 variant
==================================

variant_spec:
  core_change:
    original_behavior: |
      Legend entries were processed bottom-to-top. Each step protected the legend
      plus marker pixels that belonged to later legend rows, but already-filled
      cells from earlier steps were not protected. If two diamonds overlapped,
      the later legend row could repaint that overlap.
    new_behavior: |
      Every step may paint only cells that are still background in the current
      output. This protects all pre-existing non-background pixels: legend cells,
      marker pixels, distractors, and any fill laid down by earlier legend rows.
      The visible rule becomes "first processed wins".

  concrete_trace:
    setup: |
      Suppose the bottom legend row maps marker A -> color 6 and the next row maps
      marker B -> color 4. Their inferred diamond masks overlap on five background
      cells.
    original: |
      A fills those five cells with 6 first, then B later overwrites them with 4.
    variant: |
      A still fills them first, but when B runs those cells are no longer
      background, so B can only paint its non-overlapping remainder. The overlap
      stays 6.
    visible_difference: |
      The overlap inherits the earlier legend row's color instead of the later
      row's color.

  complexity_calibration:
    - Keep the original legend format, L-corner parsing, bbox rules, and optional
      distractor markers so the task still reads as the same family.
    - Make the variant observable by forcing at least one pair of legend entries
      to be genuine L-pairs whose fill masks overlap on multiple background cells.
    - Use visible non-background fill colors for generated legend rows and reject
      any sample where the legacy solver and the variant solver produce identical
      outputs.

  implicit_assumptions_review:
    - `_find_legend(...)` depends only on the input grid, so output-time painting
      changes do not affect legend parsing.
    - Marker detection must still read from the original input `g`, not the
      partially filled output `out`; otherwise earlier fills could corrupt later
      marker lookup.
    - The old `protect` mask encoded a task-specific assumption: only future
      markers and the legend were untouchable. That assumption is invalid here.
      The replacement invariant is simpler: `out == bg` is the only writable set.
    - The L-anchor tie assertions remain valid, so the generator must still avoid
      the ambiguous bbox sizes `(2,2)`, `(2,3)`, and `(3,2)`.
    - Solid 2x2 markers are now naturally inert under the new rule because every
      pixel in the block is already non-background; the solver keeps supporting
      them, but generator frequency is kept low because they do not showcase D13.

  placement_legality:
    - Marker pixels never overlap each other or the legend, and always stay in
      bounds.
    - Fill-mask overlap is allowed and intentionally introduced, but only through
      background cells so the precedence rule is cleanly visible.
    - The forbidden-placement mask from the original task is preserved to keep the
      bottom-left legend scan unambiguous.

  observability_and_cleanliness:
    - The first two legend entries are always L-pairs with distinct visible fill
      colors and an overlapping background region, making "first-to-arrive keeps
      the overlap" inferable from a few examples.
    - Extra entries may still be L-pairs, one noop, one block, or one distractor,
      but only as secondary structure around the core conflict.
    - Generator rejection checks enforce both a minimum conflict-cell count and a
      non-zero pixel difference between legacy and variant outputs, preventing
      silent regression to the original task.

  post_change_solver_assumptions:
    - Processing order is still the legend order, bottom-to-top.
    - A writable cell is defined purely by current output occupancy, not by color
      identity or by special casing the legend area.
    - `generate_input()` guarantees at least one order-sensitive overlap so the
      new assumption is exercised rather than remaining latent.
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


def _legend_mask(h: int, w: int, legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            mask[r, legend_c + 1] = True
    return mask


def _entry_shape_and_mask(
    h: int, w: int, marker_pts: set[tuple[int, int]]
) -> tuple[str | None, np.ndarray | None]:
    if len(marker_pts) < 2:
        return None, None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
        mask = np.zeros((h, w), dtype=bool)
        for r, c in marker_pts:
            mask[r, c] = True
        return "block", mask

    if rmin == rmax or cmin == cmax:
        return None, None

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

    return "diamond", _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)


def _solve(grid: np.ndarray, protect_all_non_bg: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _legend_mask(h, w, legend_c, entries)
    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        kind, paint_mask = _entry_shape_and_mask(h, w, marker_pts)
        if kind is None or paint_mask is None:
            continue

        if protect_all_non_bg:
            out[paint_mask & (out == bg)] = int(fill)
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if kind == "block":
            out[paint_mask & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)
        out[paint_mask & (~protect)] = int(fill)

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, protect_all_non_bg=False)


def _variant_diff_cell_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(main(grid) != _legacy_main(grid)))


def _priority_conflict_cell_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _legend_mask(h, w, legend_c, entries)

    masks: list[np.ndarray] = []
    fills: list[int] = []
    for _, marker, fill in entries:
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        kind, paint_mask = _entry_shape_and_mask(h, w, marker_pts)
        if kind == "diamond" and paint_mask is not None:
            masks.append(paint_mask)
            fills.append(int(fill))

    if len(masks) < 2:
        return 0

    conflict = np.zeros((h, w), dtype=bool)
    bg_mask = g == bg
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if fills[i] == fills[j]:
                continue
            conflict |= masks[i] & masks[j] & bg_mask
    return int(conflict.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bottom-left legend to color regions implied by diagonal marker pairs.

    Input structure:
    - The grid has one dominant background color and a vertical 2-column legend at the lower left.
    - Each legend row stores a marker color in the first column and that marker's fill color in the next column.
    - Outside the legend, each marker color appears either as two L-shaped opposite corners of a bbox or as a solid 2x2 block.

    Rule:
    - Read legend entries from bottom to top and find the non-legend pixels of that marker color.
    - For L-pairs, detect which bbox diagonal carries the L anchors, then take the opposite diagonal's shortest-path diamond as the candidate fill region.
    - Paint only cells that are still background in the current output; every non-background pixel already present is protected, including earlier fills.
    - Solid 2x2 markers remain unchanged under this rule because they occupy non-background cells already.

    Key insight: The legend still determines each marker's fill color, but overlap resolution is now "first come, first served" because later entries can only claim untouched background cells.
    """
    return _solve(grid, protect_all_non_bg=True)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.20
BLOCK_PROB = 0.15

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: D13 observability ─────────────────────────────────────────
CONFLICT_RETRIES = 220
MIN_CONFLICT_CELLS = 3
MIN_DIFF_CELLS = 3

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
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


def _l_pair_fill_mask(H: int, W: int, rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> np.ndarray:
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    return _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)


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
    accept=None,
):
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
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
        spec = {
            "cells": cells,
            "diag": diag,
            "rmin": rmin,
            "cmin": cmin,
            "rmax": rmax,
            "cmax": cmax,
            "fill_mask": _l_pair_fill_mask(H, W, rmin, cmin, rmax, cmax, diag),
        }
        if accept is not None and not accept(spec):
            continue
        do_place(cells, color)
        return spec
    return None


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
    fill_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fill_colors[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


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
    idx_pool = list(range(2, n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and idx_pool:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

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

    first_marker = legend_rows[0][1]
    first_spec = _try_place_l_pair(H, W, first_marker, LPAIR_RETRIES, can_place, do_place)
    if first_spec is None:
        return None

    second_marker = legend_rows[1][1]

    def accept_conflict(spec) -> bool:
        conflict = first_spec["fill_mask"] & spec["fill_mask"] & (~occupied)
        return int(conflict.sum()) >= MIN_CONFLICT_CELLS

    second_spec = _try_place_l_pair(
        H,
        W,
        second_marker,
        CONFLICT_RETRIES,
        can_place,
        do_place,
        accept=accept_conflict,
    )
    if second_spec is None:
        return None

    for i, (_, marker, _) in enumerate(legend_rows[2:], start=2):
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
            continue

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
        new_out = main(grid)
        old_out = _legacy_main(grid)
    except Exception:
        return None

    if _priority_conflict_cell_count(grid) < MIN_CONFLICT_CELLS:
        return None
    if int(np.count_nonzero(new_out != old_out)) < MIN_DIFF_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
