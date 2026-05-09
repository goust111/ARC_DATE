# Combo variant of 9385bd28 (index 6)
# combo_spec: D3/2+D7/0+D15/2
# Name: 极端纵横比+扩展填充+紧裁输出

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "solvers" / "arc_utils.py").exists():
        solver_dir = parent / "solvers"
        if str(solver_dir) not in sys.path:
            sys.path.append(str(solver_dir))
        break
else:
    raise RuntimeError("could not locate solvers/arc_utils.py from combo variant path")

from arc_utils import get_bg


# Design analysis for combo D3/2 + D7/0 + D15/2
# a) Concrete behavior and trace:
#    Base task: read the bottom-left legend, find each marker color's L-pair bbox,
#    infer the marker diagonal, then fill the opposite-diagonal rectangle while
#    keeping legend cells and marker pixels protected.
#    Combo task: keep the same object parsing, but enlarge each L-pair fill by one
#    cell on every side, then crop the final output to the bbox of all changed
#    cells. Example on a 10x25 horizontal strip:
#      base full-canvas fill: rows 2..4, cols 10..16
#      combo full-canvas fill: rows 1..5, cols 9..17
#      base return: still 10x25
#      combo return: only the union window of all changed regions, e.g. 5x17.
#    The D3 change affects where color is painted; the D7 change then uses that
#    larger painted support to determine the returned frame; D15 makes both the
#    input and the active regions read as long strips rather than near-squares.
# b) Complexity calibration:
#    A naive merge would let the enlarged halos overlap, or let tight-crop fall
#    back to almost the whole canvas. The generator therefore places active
#    objects inside a narrow interior activity strip aligned with the long grid
#    axis. This keeps the expansion visible, keeps the crop smaller than the
#    input, and preserves multiple repeated examples of the same rule.
# c) Hidden-assumption audit:
#    - `_find_legend` still assumes the lowest-left non-bg structure is the legend.
#      Generator keeps columns to the left of the activity strip empty.
#    - The original solver assumed the fill never extends beyond the marker bbox.
#      That is false now, so fill-mask construction is parameterized by `pad`.
#    - The original solver returned the full canvas. That is false now, so `main`
#      must crop to changed cells after the full transform.
#    - Tight crop introduces a new assumption: changed cells exist and lie away
#      from the legend and outer border. The generator guarantees this by margin
#      placement and by rejecting samples with zero visible halo or zero crop.
# d) Placement legality:
#    Every placed L-pair reserves its expanded bbox, not only its marker pixels.
#    This prevents later markers from entering a halo that should remain visible,
#    and keeps all objects inside bounds even after outward expansion.
# e) Visual observability:
#    A solver looking at 2-4 examples should be able to see three things at once:
#    the source grid is an extreme strip, each fill reaches one cell beyond the
#    marker-implied rectangle, and the output discards the legend and unused
#    border. The generator rejects instances where the expanded and unexpanded
#    outputs would crop to the same result.
# f) Visual cleanliness:
#    Active objects share one narrow strip but do not overlap after expansion.
#    This yields a clean long crop instead of many scattered windows or tangled
#    overlaps near the legend.
# g) Change inferability:
#    Colors are still explained only by the visible legend. Fill colors exclude
#    the background and the marker palette, so the meaningful change is geometric
#    and framing-based rather than dependent on hidden color-id ordering.


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
    """Mask for cells lying on some shortest 4-neigh path between corners a and b."""
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _fill_mask(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    diag: str,
    pad: int,
) -> np.ndarray:
    rr0 = max(0, rmin - pad)
    rr1 = min(h - 1, rmax + pad)
    cc0 = max(0, cmin - pad)
    cc1 = min(w - 1, cmax + pad)
    if diag == "TLBR":
        a = (rr0, cc1)
        b = (rr1, cc0)
    else:
        a = (rr0, cc0)
        b = (rr1, cc1)
    return _diamond_mask(h, w, rr0, rr1, cc0, cc1, a, b)


def _solve_full(grid: np.ndarray, pad: int) -> np.ndarray:
    """Run the full-canvas transform with optional outward fill expansion."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
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
        assert score_tlbr != score_trbl, "diagonal scores must not tie"
        diag = "TLBR" if score_tlbr > score_trbl else "TRBL"

        mask = _fill_mask(h, w, rmin, rmax, cmin, cmax, diag, pad)
        out[mask & (~protect)] = eff_fill

    return out


def _tight_crop_changed(inp: np.ndarray, out: np.ndarray, bg: int) -> np.ndarray:
    """Crop `out` to the bbox of visible changes relative to `inp`."""
    pts = np.argwhere(out != inp)
    if pts.size == 0:
        return np.full((0, 0), bg, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the legend-coded marker pairs, expand each inferred fill by one ring, and return only the changed window.

    Input structure:
    - The input is a single-background grid with an extreme aspect ratio, plus a two-column legend stacked at the lower-left.
    - Each legend row encodes `(marker_color, fill_color)` and is processed from bottom to top.
    - Away from the legend, a marker color appears as either one L-corner pair spanning a long thin bbox or as a solid 2x2 block.

    Rule:
    - For each legend entry, locate all non-legend pixels of that marker color.
    - If they form a 2x2 block, recolor the block directly; otherwise infer which bbox diagonal contains the L anchors, and use the opposite diagonal to define the fill rectangle.
    - Paint the rectangle expanded by one extra cell on every side, while still protecting the legend, the current marker pixels, and all later marker pixels.
    - After all colors are processed, crop the result to the smallest rectangle containing every changed cell.

    Key insight: The markers still reveal the same hidden rectangle as in the base task, but the visible answer is the enlarged version of that rectangle, and the returned grid is only the tight crop around all such enlarged changes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = _solve_full(g, pad=FILL_PAD)
    return _tight_crop_changed(g, out, bg)


# --------------- generator ---------------

# Extreme strip canvas
SHORT_SIDE_RANGE = (10, 12)
LONG_SIDE_RANGE = (24, 28)
MIN_INPUT_ASPECT_RATIO = 2.0
MIN_OUTPUT_ASPECT_RATIO = 1.8

# Legend
LEGEND_COL_RANGE = (0, 1)
LEGEND_ENTRY_RANGE = (3, 4)

# Marker types / branch coverage
FOUR_ENTRY_PROB = 0.75
THREE_ENTRY_NOOP_PROB = 0.35

# Placement / retries
LPAIR_RETRIES = 250
GEN_RETRIES = 400

# Variant validation
MIN_VISIBLE_HALO_CELLS = 12

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _pick_grid_shape() -> tuple[int, int, str]:
    if random.random() < 0.5:
        return random.randint(*SHORT_SIDE_RANGE), random.randint(*LONG_SIDE_RANGE), "horizontal"
    return random.randint(*LONG_SIDE_RANGE), random.randint(*SHORT_SIDE_RANGE), "vertical"


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners."""
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


def _expanded_bbox_cells(h: int, w: int, rmin: int, rmax: int, cmin: int, cmax: int) -> list[tuple[int, int]]:
    rr0 = max(0, rmin - FILL_PAD)
    rr1 = min(h - 1, rmax + FILL_PAD)
    cc0 = max(0, cmin - FILL_PAD)
    cc1 = min(w - 1, cmax + FILL_PAD)
    return [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]


def _pick_bbox_size(
    region_h: int,
    region_w: int,
    orientation: str,
    active_target: int,
) -> tuple[int, int] | None:
    """Pick a long-thin bbox sized to fit multiple visible objects in one strip."""
    candidates: list[tuple[int, int, float]] = []
    if orientation == "horizontal":
        width_cap = min(region_w, 12 if active_target <= 2 else 8)
        for bh in range(2, min(region_h, 4) + 1):
            for bw in range(5, width_cap + 1):
                if bw < bh * 2 + 1 or not _l_pair_valid_bbox(bh, bw):
                    continue
                weight = 3.0 if bh == 2 else 2.0
                if active_target >= 3:
                    weight *= max(1.0, 9.0 - bw)
                else:
                    weight *= 1.0 + 0.2 * bw
                candidates.append((bh, bw, weight))
    else:
        height_cap = min(region_h, 12 if active_target <= 2 else 8)
        for bh in range(5, height_cap + 1):
            for bw in range(2, min(region_w, 4) + 1):
                if bh < bw * 2 + 1 or not _l_pair_valid_bbox(bh, bw):
                    continue
                weight = 3.0 if bw == 2 else 2.0
                if active_target >= 3:
                    weight *= max(1.0, 9.0 - bh)
                else:
                    weight *= 1.0 + 0.2 * bh
                candidates.append((bh, bw, weight))
    if not candidates:
        return None

    sizes = [(bh, bw) for bh, bw, _ in candidates]
    weights = [weight for _, _, weight in candidates]
    return random.choices(sizes, weights=weights, k=1)[0]


def _choose_entry_types(n_entries: int) -> list[str]:
    """
    Keep each sample solvable and make branch coverage observable:
    - 3-entry legends alternate between 3 active objects and 2 active + 1 noop.
    - 4-entry legends always keep 3 active objects plus 1 noop, which is easier
      to place than 4 active objects while still exposing the larger legend case.
    """
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if n_entries == 4:
        types[idx_pool[0]] = "noop"
        return types
    if random.random() < THREE_ENTRY_NOOP_PROB:
        types[idx_pool[0]] = "noop"
    return types


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = 4 if random.random() < FOUR_ENTRY_PROB else 3
    if legend_bot - n_entries + 1 < 0:
        return None

    marker_pool = [c for c in range(10) if c != bg]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

    fill_pool = [c for c in range(10) if c != bg and c not in marker_colors]
    if not fill_pool:
        return None

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = random.choice(fill_pool)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    """Protect legend scan integrity: nothing may appear below or to the left of the legend."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _pick_activity_strip(H: int, W: int, legend_c: int, legend_rows, orientation: str):
    legend_top = min(r for r, _, _ in legend_rows)
    row_lo = 2
    row_hi = min(H - 3, legend_top - 2)
    col_lo = legend_c + 3
    col_hi = W - 3
    if row_hi < row_lo or col_hi < col_lo:
        return None

    if orientation == "horizontal":
        band_h_hi = min(5, row_hi - row_lo + 1)
        band_h_lo = 2 if band_h_hi == 2 else 3
        band_h = random.randint(band_h_lo, band_h_hi)
        band_r0 = random.randint(row_lo, row_hi - band_h + 1)
        return {
            "row_lo": band_r0,
            "row_hi": band_r0 + band_h - 1,
            "col_lo": col_lo,
            "col_hi": col_hi,
            "legend_top": legend_top,
        }

    band_w_hi = min(5, col_hi - col_lo + 1)
    band_w_lo = 2 if band_w_hi == 2 else 3
    band_w = random.randint(band_w_lo, band_w_hi)
    band_c0 = random.randint(col_lo, col_hi - band_w + 1)
    return {
        "row_lo": row_lo,
        "row_hi": row_hi,
        "col_lo": band_c0,
        "col_hi": band_c0 + band_w - 1,
        "legend_top": legend_top,
    }


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    orientation: str,
    strip: dict[str, int],
    active_target: int,
    can_place,
    do_place,
):
    region_h = strip["row_hi"] - strip["row_lo"] + 1
    region_w = strip["col_hi"] - strip["col_lo"] + 1
    if region_h < 2 or region_w < 2:
        return None

    for _ in range(LPAIR_RETRIES):
        size = _pick_bbox_size(region_h, region_w, orientation, active_target)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(strip["row_lo"], strip["row_hi"] - bh + 1)
        cmin = random.randint(strip["col_lo"], strip["col_hi"] - bw + 1)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = random.choice(["TLBR", "TRBL"])
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        reserve = _expanded_bbox_cells(H, W, rmin, rmax, cmin, cmax)
        if can_place(cells, reserve):
            do_place(cells, color, reserve)
            return {"type": "L", "bbox": (rmin, rmax, cmin, cmax), "diag": diag}
    return None


def _changed_bbox(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(out != inp)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _halo_delta(inp: np.ndarray, base_out: np.ndarray, combo_out: np.ndarray) -> int:
    """Cells changed only because of the one-cell expansion."""
    return int(np.count_nonzero((combo_out != inp) & (combo_out != base_out)))


def _strip_aspect_ok(h: int, w: int, threshold: float) -> bool:
    short_dim = min(h, w)
    long_dim = max(h, w)
    return short_dim > 0 and long_dim / short_dim >= threshold


def _gen_attempt() -> np.ndarray | None:
    H, W, orientation = _pick_grid_shape()
    if not _strip_aspect_ok(H, W, MIN_INPUT_ASPECT_RATIO):
        return None
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    strip = _pick_activity_strip(H, W, legend_c, legend_rows, orientation)
    if strip is None:
        return None

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    reserved = forbidden | legend_mask

    n_entries = len(legend_rows)
    types = _choose_entry_types(n_entries)
    active_target = types.count("L")
    if active_target < 2:
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

    for i, (_, marker, _) in enumerate(legend_rows):
        if types[i] == "noop":
            metas.append({"type": "noop"})
            continue
        meta = _try_place_l_pair(H, W, marker, orientation, strip, active_target, can_place, do_place)
        if meta is None:
            return None
        metas.append(meta)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        base_full = _solve_full(grid, pad=0)
        combo_full = _solve_full(grid, pad=FILL_PAD)
        combo_crop = main(grid)
    except Exception:
        return None

    bbox = _changed_bbox(grid, combo_full)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox

    if not (r0 > 0 and c0 > legend_c + 1 and r1 < H - 1 and c1 < W - 1):
        return None
    if r1 >= strip["legend_top"]:
        return None

    if combo_crop.shape[0] >= H or combo_crop.shape[1] >= W:
        return None
    if not _strip_aspect_ok(combo_crop.shape[0], combo_crop.shape[1], MIN_OUTPUT_ASPECT_RATIO):
        return None

    halo = _halo_delta(grid, base_full, combo_full)
    if halo < MIN_VISIBLE_HALO_CELLS:
        return None

    base_crop = _tight_crop_changed(grid, base_full, get_bg(grid))
    if combo_crop.shape == base_crop.shape and np.array_equal(combo_crop, base_crop):
        return None

    active_l_count = sum(meta["type"] == "L" for meta in metas)
    if active_l_count < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
