# Combo variant of 9385bd28 (index 4)
# combo_spec: D6/3+D10/1+D13/1
# Name: 大量条目+重叠混合色+全保护

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg

"""
Design analysis for the combo variant
=====================================

1. Concrete form and behavior trace
-----------------------------------
- The legend expands from 2 columns to 3 columns and from 2-5 rows to 6-8 rows.
  Each row is now (marker, fill, mix), and the third column repeats one shared
  overlap color for the whole task.
- Non-legend objects still use the original geometry family: most legend colors
  appear as one L-pair whose opposite diagonal defines a diamond fill mask.
- The important fusion is how D6 and D13 interact:
  original task:
    entry A paints first, entry B paints later, and later paint can overwrite
    earlier fill on overlap cells.
  D13 alone:
    entry B would be blocked once A has already painted those cells.
  combo here:
    we treat only original background cells as writable targets, collect every
    legend entry that wants each such cell, and then decide the final color:
    one distinct fill claim -> that fill color; multiple distinct fill claims ->
    shared mix color.
- Small example: if bottom legend row says red-marker -> yellow-fill and the
  next row says green-marker -> blue-fill, then a background cell touched only
  by the red diamond becomes yellow, a cell touched only by the green diamond
  becomes blue, and a background cell touched by both becomes the shared mix
  color from legend column 3. Marker cells and any other pre-existing non-bg
  cells remain untouched.

2. Complexity calibration
-------------------------
- The generator uses a materially larger canvas and 6-8 legend rows so the
- solver must track many mappings at once.
- A single shared mix color is intentionally simpler than pair-specific mixing.
  That keeps the color rule inferable even with many legend rows.
- The generator forces:
  - at least two real L-pairs whose diamonds overlap on background cells, so the
    mix color is visibly used;
  - at least one later legend entry whose diamond crosses earlier non-legend
    non-bg cells, so the "protect all existing non-bg pixels" rule is also
    exercised instead of being latent.
- Optional noop rows remain rare, and 2x2 blocks are disabled because under the
  full-protection rule they become visually inert noise rather than useful signal.

3. Implicit assumption audit
----------------------------
- `_find_legend(...)` in the original solver assumed a 2-column legend. The
  combo must require 3 columns and verify that every row repeats the same mix
  color; otherwise the overlap rule is under-specified.
- The original solver wrote directly into `out` while iterating legend rows.
  That relied on overwrite order to resolve conflicts. In the combo, direct
  sequential painting would conflict with "protect all non-bg" because a later
  mix update would need to repaint an already-filled cell. The fix is to infer
  all candidate paint masks from the input grid first and resolve overlap on the
  set of original background cells.
- Marker detection must still read from the input grid `g`, never from the
  partially produced output, because later fills must not corrupt marker lookup.
- The original block special case recolored a solid 2x2 marker. That premise no
  longer holds once every existing non-bg cell is protected. The solver still
  recognizes blocks for robustness, but they contribute no writable cells.
- Ambiguous L-pair bounding boxes `(2,2)`, `(2,3)`, and `(3,2)` are still
  forbidden because diagonal inference still depends on anchor-score separation.

4. Placement legality
---------------------
- Marker pixels never overlap each other or the legend and always stay in bounds.
- Overlap is allowed only between inferred fill masks, not between the visible
  marker cells themselves.
- The legend scan corridor is reserved so the first non-bg cells from the bottom
  still form a clean vertical legend strip.
- When the generator forces a later diamond to cross earlier non-bg cells, it
  does so only via the fill mask; the later marker's own L-corners remain
  disjoint from previously occupied cells.

5. Visual observability
-----------------------
- Humans can infer the color rule because the third legend column repeats the
  same color in every row, signaling a role different from per-entry fill.
- The dataset remains structurally regular: same marker geometry family, same
  legend layout, just more rows and one extra legend column.
- A few examples are enough because every example shows both:
  - ordinary one-color fills;
  - visible overlap regions rendered in the shared mix color.
- The full-protection rule is also visible because some later diamonds pass
  through earlier marker/distractor cells without erasing them.

6. Visual cleanliness
---------------------
- Larger grids and moderate bbox sizes prevent the 6-8 marker groups from
  collapsing into an unreadable cluster.
- The generator avoids a second source of clutter: it does not rely on many
  distractors, and it does not require deep chains of multi-way overlap.
- The reserved 3-column legend corridor keeps the dense legend visually separate
  from the marker field.

7. New solver assumptions introduced by the combo
-------------------------------------------------
- Every legend row must share one global mix color.
- Mixed output is defined on original background cells only.
- If multiple entries claim the same background cell with the same fill color,
  the cell stays that fill; the mix color is reserved for genuine multi-color
  overlap.
- `generate_input()` guarantees both a non-zero overlap-on-background count and
  a non-zero count of later-fill masks crossing earlier non-bg cells, so the
  fused rule is actually exercised.
"""

# --------------- shared geometry helpers ---------------

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Legend is a vertical stack of triples at bottom-left.
    Each legend row: (marker at col c, fill at col c+1, mix at col c+2).
    Returns (legend_col, entries) where entries are ordered bottom->top as
    (row, marker, fill, mix).
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
        fill = int(g[r, c + 1])
        mix = int(g[r, c + 2])
        entries.append((r, marker, fill, mix))
    return c, entries


def _build_legend_mask(h: int, w: int, legend_c: int, entries: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, _, _, _ in entries:
        for dc in range(3):
            if 0 <= r < h and 0 <= legend_c + dc < w:
                mask[r, legend_c + dc] = True
    return mask


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
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
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                mask[r, c] = True
    return mask


def _mask_from_cells(h: int, w: int, cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _entry_kind_and_mask(
    h: int, w: int, marker_pts: set[tuple[int, int]]
) -> tuple[str | None, np.ndarray | None]:
    if len(marker_pts) < 2:
        return None, None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
        return "block", _mask_from_cells(h, w, list(marker_pts))

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
    assert score_tlbr != score_trbl, "diagonal scores must not tie"

    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br
    return "diamond", _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)


def _parse_context(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill, mix) entry"

    mix_colors = {mix for _, _, _, mix in entries}
    assert len(mix_colors) == 1, "all legend rows must share one mix color"
    mix_color = next(iter(mix_colors))

    legend_mask = _build_legend_mask(h, w, legend_c, entries)
    specs: list[dict] = []
    for _, marker, fill, _ in entries:
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        kind, paint_mask = _entry_kind_and_mask(h, w, marker_pts)
        specs.append(
            {
                "marker": int(marker),
                "fill": int(fill),
                "kind": kind,
                "paint_mask": paint_mask,
                "marker_pts": marker_pts,
            }
        )

    return {
        "g": g,
        "h": h,
        "w": w,
        "bg": int(bg),
        "legend_c": legend_c,
        "entries": entries,
        "mix_color": int(mix_color),
        "legend_mask": legend_mask,
        "specs": specs,
    }


def _claim_summary(ctx: dict) -> dict:
    g = ctx["g"]
    bg = ctx["bg"]
    specs = ctx["specs"]

    bg_mask = g == bg
    claimed = np.zeros_like(g, dtype=bool)
    mixed = np.zeros_like(g, dtype=bool)
    first_fill = np.full_like(g, bg, dtype=int)

    for spec in specs:
        paint_mask = spec["paint_mask"]
        if paint_mask is None:
            continue
        paint = paint_mask & bg_mask
        if not np.any(paint):
            continue

        differing = paint & claimed & (first_fill != spec["fill"])
        mixed[differing] = True

        fresh = paint & (~claimed)
        first_fill[fresh] = spec["fill"]
        claimed |= paint

    return {"claimed": claimed, "mixed": mixed, "first_fill": first_fill}


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    ctx = _parse_context(grid)
    g = ctx["g"]
    mix_color = ctx["mix_color"]
    summary = _claim_summary(ctx)
    claimed = summary["claimed"]
    mixed = summary["mixed"]
    first_fill = summary["first_fill"]

    out = g.copy()
    solid = claimed & (~mixed)
    out[solid] = first_fill[solid]
    out[mixed] = mix_color
    return out


def _solve_ordered_baseline(grid: np.ndarray) -> np.ndarray:
    ctx = _parse_context(grid)
    g = ctx["g"]
    legend_mask = ctx["legend_mask"]
    specs = ctx["specs"]
    marker_order = [spec["marker"] for spec in specs]

    out = g.copy()
    for i, spec in enumerate(specs):
        paint_mask = spec["paint_mask"]
        if paint_mask is None:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for marker in future_markers:
            protect |= (g == marker) & (~legend_mask)

        if spec["kind"] == "block":
            out[paint_mask & (~protect)] = spec["fill"]
            continue

        protect |= (g == spec["marker"]) & (~legend_mask)
        out[paint_mask & (~protect)] = spec["fill"]

    return out


def _solve_first_wins_baseline(grid: np.ndarray) -> np.ndarray:
    ctx = _parse_context(grid)
    specs = ctx["specs"]
    bg = ctx["bg"]
    out = ctx["g"].copy()

    for spec in specs:
        paint_mask = spec["paint_mask"]
        if paint_mask is None:
            continue
        out[paint_mask & (out == bg)] = spec["fill"]

    return out


def _ordered_vulnerable_hit_count(ctx: dict) -> int:
    g = ctx["g"]
    legend_mask = ctx["legend_mask"]
    specs = ctx["specs"]
    marker_order = [spec["marker"] for spec in specs]
    hits = np.zeros_like(g, dtype=bool)

    for i, spec in enumerate(specs):
        paint_mask = spec["paint_mask"]
        if paint_mask is None:
            continue
        future_mask = np.zeros_like(g, dtype=bool)
        for marker in marker_order[i + 1 :]:
            future_mask |= (g == marker) & (~legend_mask)
        hits |= (
            paint_mask
            & (~legend_mask)
            & (g != ctx["bg"])
            & (~future_mask)
            & (g != spec["marker"])
        )
    return int(hits.sum())


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    ctx = _parse_context(grid)
    combo_out = _solve_combo(grid)
    ordered_out = _solve_ordered_baseline(grid)
    first_wins_out = _solve_first_wins_baseline(grid)
    summary = _claim_summary(ctx)

    mix_pixels = int(np.count_nonzero(summary["mixed"]))
    claimed_pixels = int(np.count_nonzero(summary["claimed"]))
    active_entries = sum(
        1
        for spec in ctx["specs"]
        if spec["paint_mask"] is not None and np.any(spec["paint_mask"] & (ctx["g"] == ctx["bg"]))
    )

    return {
        "mix_pixels": mix_pixels,
        "claimed_pixels": claimed_pixels,
        "active_entries": active_entries,
        "diff_vs_ordered": int(np.count_nonzero(combo_out != ordered_out)),
        "diff_vs_first_wins": int(np.count_nonzero(combo_out != first_wins_out)),
        "protected_hits": _ordered_vulnerable_hit_count(ctx),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a tall three-column legend to paint many diamonds while blending true overlaps.

    Input structure:
    - The lower-left corner holds a 3-column vertical legend with 6-8 rows; each row is
      (marker color, fill color, shared mix color), and the third column repeats the same mix value.
    - Away from the legend, each marker color appears as one diagonal pair of L-shaped corner marks;
      rare malformed or block-like cases are tolerated but do not add writable area.
    - The rest of the grid is background, plus possible non-legend distractor pixels that must remain visible.

    Rule:
    - Read the legend bottom-to-top, find each marker color's non-legend pixels, and infer the diamond mask from the
      opposite diagonal of the marker bbox exactly as in the original task family.
    - Only cells that were background in the input are eligible to change; every pre-existing non-background pixel,
      including legend cells, markers, distractors, and previously implied regions, is protected from direct overwrite.
    - If exactly one legend entry claims a background cell, paint it with that entry's fill color.
    - If two or more entries with different fill colors claim the same background cell, paint that cell with the
      shared mix color from legend column 3.

    Key insight:
    - The L-corners still tell you which diamond each marker controls, but the extra legend column now signals a global
      overlap color, so overlaps are resolved by blending on background cells rather than by later overwrite priority.
    """
    return _solve_combo(grid)


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (22, 28)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (6, 8)
NOOP_PROB = 0.08

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 11
BBOX_W_MAX = 11
BBOX_SHAPE_WEIGHTS = [0.82, 0.10, 0.08]
TLBR_PROB = 0.78

# ── CFG: observability targets ─────────────────────────────────────
MIN_DIAMOND_CELLS = 7
MIN_MIX_CELLS = 4
MIN_PROTECTED_HITS = 1
MIN_DIFF_ORDERED = 4
MIN_DIFF_FIRST_WINS = 4
MIN_ACTIVE_ENTRIES = 6

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.12

# ── CFG: retry budgets ─────────────────────────────────────────────
LPAIR_RETRIES = 260
FORCED_PROTECT_RETRIES = 320
DISTRACTOR_RETRIES = 90
GEN_RETRIES = 500


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


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif choice == "h2_w_ge4":
        max_w = min(BBOX_W_MAX, W)
        if max_w < 4:
            return None
        bh = 2
        bw = random.randint(4, max_w)
    else:
        max_h = min(BBOX_H_MAX, H)
        if max_h < 4:
            return None
        bh = random.randint(4, max_h)
        bw = 2

    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _diamond_from_anchor_diag(H: int, W: int, rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> np.ndarray:
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    return _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    accept=None,
) -> dict | None:
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if not can_place(cells):
            continue
        paint_mask = _diamond_from_anchor_diag(H, W, rmin, cmin, rmax, cmax, diag)
        spec = {
            "cells": cells,
            "cell_mask": _mask_from_cells(H, W, cells),
            "paint_mask": paint_mask,
            "diag": diag,
            "bbox": (rmin, cmin, rmax, cmax),
        }
        if accept is not None and not accept(spec):
            continue
        do_place(cells, color)
        return spec
    return None


def _choose_fill_colors(bg: int, marker_colors: list[int], mix_color: int) -> list[int]:
    palette = [c for c in range(10) if c != bg and c != mix_color]
    fills: list[int] = []
    used: list[int] = []

    for i, marker in enumerate(marker_colors):
        candidates = [c for c in palette if c != marker]
        if not candidates:
            candidates = [c for c in range(10) if c != bg and c != marker]
        assert candidates, "at least one fill color must be available"

        diverse = [c for c in candidates if c not in used]
        if i < 3 and diverse:
            choice = random.choice(diverse)
        elif diverse and random.random() < 0.55:
            choice = random.choice(diverse)
        else:
            choice = random.choice(candidates)
        fills.append(choice)
        used.append(choice)

    if len(fills) >= 2 and fills[1] == fills[0]:
        alt = [c for c in palette if c != marker_colors[1] and c != fills[0]]
        if alt:
            fills[1] = random.choice(alt)
    return fills


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0 or legend_c + 2 >= W:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries + 1:
        return None
    marker_colors = random.sample(pool, n_entries)
    remaining = [c for c in pool if c not in marker_colors]
    if not remaining:
        return None
    mix_color = random.choice(remaining)
    fill_colors = _choose_fill_colors(bg, marker_colors, mix_color)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i, (marker, fill) in enumerate(zip(marker_colors, fill_colors)):
        r = legend_bot - i
        legend_rows.append((r, marker, fill, mix_color))
        for dc, value in enumerate((marker, fill, mix_color)):
            grid[r, legend_c + dc] = value
            legend_mask[r, legend_c + dc] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, fill_colors, mix_color


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows) -> np.ndarray:
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 3] = True
    for r, _, _, _ in legend_rows:
        for dc in range(3):
            forbidden[r, legend_c + dc] = False
    return forbidden


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
        fill_colors,
        _mix_color,
    ) = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    types = ["L"] * len(legend_rows)
    if len(legend_rows) >= 8 and random.random() < NOOP_PROB:
        idx = random.randint(4, len(legend_rows) - 1)
        types[idx] = "noop"

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

    first_info = _try_place_l_pair(
        H,
        W,
        legend_rows[0][1],
        LPAIR_RETRIES,
        can_place,
        do_place,
        accept=lambda spec: int(spec["paint_mask"].sum()) >= MIN_DIAMOND_CELLS,
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
        accept=lambda spec: (
            int(spec["paint_mask"].sum()) >= MIN_DIAMOND_CELLS
            and fill_colors[1] != fill_colors[0]
            and int(np.count_nonzero(spec["paint_mask"] & first_info["paint_mask"] & (~occupied))) >= MIN_MIX_CELLS
        ),
    )
    if second_info is None:
        return None

    earlier_non_bg = first_info["cell_mask"] | second_info["cell_mask"]
    third_info = _try_place_l_pair(
        H,
        W,
        legend_rows[2][1],
        FORCED_PROTECT_RETRIES,
        can_place,
        do_place,
        accept=lambda spec: (
            int(spec["paint_mask"].sum()) >= MIN_DIAMOND_CELLS
            and int(np.count_nonzero(spec["paint_mask"] & earlier_non_bg)) >= MIN_PROTECTED_HITS
            and int(np.count_nonzero(spec["paint_mask"] & (~occupied))) >= MIN_DIAMOND_CELLS
        ),
    )
    if third_info is None:
        return None

    for i in range(3, len(legend_rows)):
        if types[i] == "noop":
            continue
        if _try_place_l_pair(H, W, legend_rows[i][1], LPAIR_RETRIES, can_place, do_place) is None:
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors and c != legend_rows[0][3]]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    metrics = _behavior_metrics(grid)
    if metrics["mix_pixels"] < MIN_MIX_CELLS:
        return None
    if metrics["protected_hits"] < MIN_PROTECTED_HITS:
        return None
    if metrics["diff_vs_ordered"] < MIN_DIFF_ORDERED:
        return None
    if metrics["diff_vs_first_wins"] < MIN_DIFF_FIRST_WINS:
        return None
    if metrics["active_entries"] < MIN_ACTIVE_ENTRIES:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
