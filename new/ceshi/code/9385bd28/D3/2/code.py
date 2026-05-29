# Variant of 9385bd28
# Dimension: D3 — 目标对象：填充区域的形状 (dst)
# Change: 标记对角线的对边对角线上所有最短路径像素（菱形/矩形填充区域）... → 填充扩展到标记外一圈

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg


# Design analysis for the D3 variant
# a) Concrete behavior:
#    The original task paints the rectangle implied by the matched marker pair.
#    In this variant, the same rectangle is still inferred from the same L-corner
#    markers and the same legend, but the painted region grows by one extra cell
#    on all four sides, clipped by the grid boundary.
#    Example: if the original bbox is rows 3..5, cols 7..9, then the variant
#    paints rows 2..6, cols 6..10. The L markers themselves still remain visible
#    because current-marker pixels are protected just as before.
#    Flow difference:
#      original: detect bbox -> fill bbox
#      variant : detect bbox -> expand bbox by 1 -> fill expanded bbox
#    The changed step is only the destination mask; legend parsing, marker
#    grouping, diagonal detection, processing order, and 2x2-block handling stay.
# b) Complexity calibration:
#    If placement touched the border or another marker too closely, the extra ring
#    could be clipped or visually hidden, making the D3 change hard to notice.
#    The generator therefore keeps active L-pairs at least one cell away from the
#    canvas edge and reserves their expanded bbox while placing later markers.
#    This preserves the original symbolic structure while making the new ring easy
#    to see from a few examples.
# c) Hidden-assumption audit:
#    - _find_legend still assumes the legend is the lowest-left non-bg structure.
#      Generator keeps the same legend geometry.
#    - Marker grouping by color still assumes each legend color belongs to at most
#      one object outside the legend; generator keeps marker colors distinct.
#    - The original fill code only touched the bbox interior. With expansion,
#      writes can now reach cells outside that bbox, so protection against legend
#      and future markers must remain in place before painting. Solver keeps that.
#    - Border clipping is now relevant. Solver clips the expanded bbox; generator
#      avoids borders for visibility but solver does not rely on that for safety.
# d) Placement legality:
#    L-pairs are placed with a one-cell margin from the border and reserve their
#    expanded bbox so that no other marker pixels are inserted into the halo that
#    should visibly change in the output. 2x2 blocks are still allowed, but only
#    on free cells outside reserved regions.
# e) Visual observability:
#    The extra ring is a geometric, not color-id-based, change. The generator
#    rejects samples where no halo cell actually changes in the final output, so
#    the variant cannot silently collapse back to the original task.
# f) Visual cleanliness:
#    Reserving the expanded bbox prevents clutter inside the destination region.
#    This keeps each transformed area legible and makes the one-pixel margin easy
#    to compare against the input markers.
# g) Color inference:
#    Colors still come only from the visible two-column legend; no hidden ordering
#    by color id is introduced. The D3 change is shape-only.


FILL_PAD = 1


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

    This remains the robust way to identify which bbox diagonal contains the markers.
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
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path
    between corners a and b: dist(a,p) + dist(p,b) == dist(a,b).
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


def _expanded_fill_mask(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    diag: str,
) -> np.ndarray:
    """
    Expand the original fill rectangle by one cell outward on every side.

    We still phrase this via the opposite diagonal corners so the D3 change is
    isolated to the destination shape rather than to marker interpretation.
    """
    rr0 = max(0, rmin - FILL_PAD)
    rr1 = min(h - 1, rmax + FILL_PAD)
    cc0 = max(0, cmin - FILL_PAD)
    cc1 = min(w - 1, cmax + FILL_PAD)
    if diag == "TLBR":
        a = (rr0, cc1)
        b = (rr1, cc0)
    else:
        a = (rr0, cc0)
        b = (rr1, cc1)
    return _diamond_mask(h, w, rr0, rr1, cc0, cc1, a, b)


def _bbox_region_cells(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    pad: int,
) -> list[tuple[int, int]]:
    rr0 = max(0, rmin - pad)
    rr1 = min(h - 1, rmax + pad)
    cc0 = max(0, cmin - pad)
    cc1 = min(w - 1, cmax + pad)
    return [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a bottom-left color legend to paint expanded regions around matched corner markers.

    Input structure:
    - The grid has one dominant background color plus a two-column legend stacked near the lower-left edge.
    - Each legend row stores a marker color in the first column and that marker's fill color in the next column.
    - Away from the legend, a marker color appears either as two opposite L-shaped corners of one bounding box or as a solid 2x2 block.

    Rule:
    - Read the legend from bottom to top and process one marker color at a time.
    - For an L-pair, detect which bbox diagonal contains the L anchors; the fill is based on the rectangle implied by the opposite diagonal.
    - Paint that rectangle plus a one-cell outward border on all sides, clipped to the grid, while preserving legend cells and marker pixels for later legend entries.
    - If the marker is a solid 2x2 block, just recolor the block itself instead of expanding any region.

    Key insight: The markers still identify the same hidden rectangle as before, but the output color spreads one extra ring beyond that rectangle, so the painted area extends past the marker-defined bounds.
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
    marker_order = [m for _, m, _ in entries]

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
        diag = "TLBR" if score_tlbr > score_trbl else "TRBL"

        expanded = _expanded_fill_mask(h, w, rmin, rmax, cmin, cmax, diag)
        out[expanded & (~protect)] = eff_fill

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 200
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 300

# ── CFG: variant visibility ────────────────────────────────────────
MIN_VISIBLE_HALO_CELLS = 4

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
    if H < 2 or W < 2:
        return None
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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place):
    """Try to place one visible L-pair and reserve its expanded bbox."""
    for _ in range(retries):
        size = _pick_bbox_size(H - 2 * FILL_PAD, W - 2 * FILL_PAD)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(FILL_PAD, H - bh - FILL_PAD)
        cmin = random.randint(FILL_PAD, W - bw - FILL_PAD)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        reserve = _bbox_region_cells(H, W, rmin, rmax, cmin, cmax, FILL_PAD)
        if can_place(cells, reserve):
            do_place(cells, color, reserve)
            return {"type": "L", "bbox": (rmin, rmax, cmin, cmax), "diag": diag}
    return None


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
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
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


def _visible_halo_delta(inp: np.ndarray, out: np.ndarray, metas: list[dict]) -> int:
    """Count changed cells that live strictly in the extra one-cell ring of L-pairs."""
    total = 0
    for meta in metas:
        if meta["type"] != "L":
            continue
        rmin, rmax, cmin, cmax = meta["bbox"]
        halo_mask = np.zeros_like(inp, dtype=bool)
        rr0 = max(0, rmin - FILL_PAD)
        rr1 = min(inp.shape[0] - 1, rmax + FILL_PAD)
        cc0 = max(0, cmin - FILL_PAD)
        cc1 = min(inp.shape[1] - 1, cmax + FILL_PAD)
        halo_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        halo_mask[rmin : rmax + 1, cmin : cmax + 1] = False
        total += int(np.count_nonzero(halo_mask & (out != inp)))
    return total


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
    reserved = forbidden | legend_mask

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        block_idx = idx_pool.pop()
        if n_entries - types.count("noop") >= 2:
            types[block_idx] = "block"

    if "L" not in types:
        return None

    metas: list[dict] = []

    def can_place(cells, reserve_cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        for r, c in reserve_cells:
            if reserved[r, c]:
                return False
        return True

    def do_place(cells, color, reserve_cells):
        for r, c in reserve_cells:
            reserved[r, c] = True
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for i, (_, m, _) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            metas.append({"type": "noop"})
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells, cells):
                    do_place(cells, m, cells)
                    metas.append({"type": "block", "bbox": (r0, r0 + 1, c0, c0 + 1)})
                    placed = True
                    break
            if not placed:
                return None
        else:
            meta = _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place)
            if meta is None:
                return None
            metas.append(meta)

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _ = _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if _visible_halo_delta(grid, out, metas) < MIN_VISIBLE_HALO_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
