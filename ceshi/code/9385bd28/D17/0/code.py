# Variant of 9385bd28
# Dimension: D17 — 图例的阅读方向与编码方式 (symbol_instructions)
# Change: 图例为垂直堆叠的 (marker_color, fill_color) 对，左列标记色、右列填充色，从下到上阅读确定处理顺序 → 图例阅读方向从上到下

# Design analysis for the D17 variant
# a) Concretization and behavior trace
#    - The pixels still show the same lower-left legend: a 2-column vertical stack where each row is
#      (marker_color, fill_color). The legend is still anchored near the bottom-left and markers in the
#      main field are still L-corner pairs or occasional 2x2 blocks.
#    - The only semantic change is the order in which legend rows are applied. Original task:
#      bottom row first, then upward. Variant: top row first, then downward.
#    - Concrete trace on two overlapping entries:
#      top row = (red -> yellow), bottom row = (blue -> green). Their diamonds overlap.
#      Original order would paint blue/green first and then let red/yellow overwrite overlap cells.
#      Variant order paints red/yellow first and later lets blue/green overwrite those same cells.
#      The overlap region therefore flips to the lower legend row's fill in the variant.
#    - The affected pipeline stage is not shape detection but conflict resolution between fills.
#      The parser still finds the same legend rows and the same marker bboxes; only the processing
#      sequence and "future marker" protection set change.
# b) Complexity calibration
#    - If samples contain only isolated markers or only one active legend entry, top-down and bottom-up
#      produce the same output. That would make D17 invisible.
#    - To keep the variant observable without changing the task family, the generator forces the topmost
#      and bottommost legend entries to be active L-pairs with overlapping fillable regions and distinct
#      fill colors. Interior rows may still vary with noop/block options.
#    - The overlap is limited to two controlled entries; this keeps the puzzle readable instead of
#      creating many-way ambiguity.
# c) Implicit assumption audit against the original main()
#    - _find_legend originally returned rows in bottom->top order and main() trusted that order directly.
#      In the variant, the same scan is still used to locate the legend, but the returned list must be
#      reversed into top->bottom reading order.
#    - The loop over entries relied on "future_markers" meaning rows higher in the legend. That premise
#      is no longer valid. In the variant, "future" means rows lower in the legend because those are
#      processed later.
#    - The marker extraction, anchor scoring, diamond computation, and 2x2 block recoloring do not rely
#      on the old reading direction and remain correct unchanged.
# d) Placement legality
#    - Legend rows remain contiguous and start from the same left column; objects are forbidden from using
#      columns <= legend_c above the legend so legend detection stays unambiguous.
#    - Marker pixels still cannot overlap each other or the legend, and all placements are bounds-checked.
#    - Overlap is only required for the eventual fill masks, not for marker pixels. This preserves solver
#      assumptions while creating order-sensitive outputs.
# e) Visual observability
#    - The critical signal is geometric overlap plus different fill colors, not hidden numeric ordering.
#      A human can see that the bottom legend row wins in shared territory despite the legend being drawn
#      with the top row visually first, which reveals the changed reading direction.
#    - Generated examples are rejected unless top-down and bottom-up execution differ by multiple cells.
# f) Visual cleanliness
#    - Only the extreme legend rows are forced into overlap; other entries are placed normally.
#      This avoids dense multi-layer fill conflicts across the whole board.
#    - Marker shapes remain separated and easy to identify because only their filled diamonds overlap.
# g) Color inferability
#    - Colors are still read directly from the legend pair on each row; no arithmetic or id-based color
#      logic is introduced.
#    - The two order-sensitive rows are forced to have different non-background fill colors so the
#      direction change is visible in the overlap zone.
# h) Source-task check
#    - source_task is None, so there is no extra solver to inspect before implementation.
# i) Post-change assumptions to guarantee in generate_input()
#    - The legend parser assumes one contiguous vertical stack at a fixed left column.
#    - The order-sensitive effect assumes at least two active L-pairs whose fillable cells overlap.
#    - The "future marker" protection logic assumes marker colors are distinct across legend rows.
#    - The generator enforces all three conditions and rejects samples where the variant effect is too weak.

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


MIN_TOP_BOTTOM_OVERLAP = 2
MIN_VARIANT_DIFF_CELLS = 3


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered top->bottom for reading.
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

    bottom_up: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        bottom_up.append((r, marker, fill))
    return c, list(reversed(bottom_up))


def _build_legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
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

    This checks the anchor structure itself, which avoids bbox-corner ambiguity.
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


def _solve_from_entries(
    g: np.ndarray,
    entries: list[tuple[int, int, int]],
    legend_mask: np.ndarray,
) -> np.ndarray:
    h, w = g.shape
    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # In the variant, later rows in processing order sit lower in the legend.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the region implied by each marker pair, using the legend's top-to-bottom reading order.

    Input structure:
    - A mostly uniform background contains a 2-column legend stacked vertically near the lower-left.
    - Each legend row stores a marker color in the left cell and its fill color in the right cell.
    - Away from the legend, each marker color appears either as two opposite L-corners or as one solid 2x2 block.

    Rule:
    - Read the legend from top row to bottom row, even though it is physically anchored at the bottom-left.
    - For each legend entry, gather that marker's non-legend pixels and infer which bbox diagonal holds the L-anchors.
    - Fill the opposite diagonal's shortest-path diamond with the row's fill color, while preserving legend cells and
      any marker pixels that belong to entries still to be processed lower in the legend.
    - If the marker is a solid 2x2 block, recolor just that block instead of building a diamond.

    Key insight:
    - The geometry of each marker pair is unchanged; the only twist is that overlap conflicts are resolved by a
      top-to-bottom legend reading, so lower legend rows overwrite shared fill cells later.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"
    legend_mask = _build_legend_mask(g.shape, legend_c, entries)
    return _solve_from_entries(g, entries, legend_mask)


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
    mask = _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)
    for r, c in _l_pair_cells(rmin, cmin, rmax, cmax, diag):
        mask[r, c] = False
    return mask


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
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
    overlap_mask: np.ndarray | None = None,
    min_overlap: int = 0,
) -> dict | None:
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
        fill_mask = _l_pair_fill_mask(H, W, rmin, cmin, rmax, cmax, diag)
        if overlap_mask is not None and int(np.count_nonzero(fill_mask & overlap_mask)) < min_overlap:
            continue
        do_place(cells, color)
        return {
            "cells": cells,
            "fill_mask": fill_mask,
            "bbox": (rmin, cmin, rmax, cmax),
            "diag": diag,
        }
    return None


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    legend_top = legend_bot - n_entries + 1
    if legend_top < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    fill_pool = [c for c in range(10) if c != bg]
    fills = [random.randint(0, 9) for _ in range(n_entries)]
    fills[0] = random.choice(fill_pool)
    fills[-1] = random.choice([c for c in fill_pool if c != fills[0]])

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_top + i
        marker = marker_colors[i]
        fill = fills[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
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
    if n_entries > 2:
        idx_pool = list(range(1, n_entries - 1))
        random.shuffle(idx_pool)
        if random.random() < NOOP_PROB and len(idx_pool) > 1:
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

    top_marker = legend_rows[0][1]
    top_meta = _try_place_l_pair(H, W, top_marker, LPAIR_RETRIES, can_place, do_place)
    if top_meta is None:
        return None

    bottom_marker = legend_rows[-1][1]
    bottom_meta = _try_place_l_pair(
        H,
        W,
        bottom_marker,
        LPAIR_RETRIES,
        can_place,
        do_place,
        overlap_mask=top_meta["fill_mask"],
        min_overlap=MIN_TOP_BOTTOM_OVERLAP,
    )
    if bottom_meta is None:
        return None

    for i in range(1, n_entries - 1):
        _, marker, _ = legend_rows[i]
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
            meta = _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place)
            if meta is None:
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(
                H,
                W,
                random.choice(avail),
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out_top_down = main(grid)
    except Exception:
        return None

    out_bottom_up = _solve_from_entries(grid, list(reversed(legend_rows)), legend_mask)
    if int(np.count_nonzero(out_top_down != out_bottom_up)) < MIN_VARIANT_DIFF_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
