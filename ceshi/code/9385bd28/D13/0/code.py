# Variant of 9385bd28
# Dimension: D13 — 标记保护机制 (meta)
# Change: 填充时保护后续图例条目的标记像素和图例区域不被覆盖... → 无保护（后续标记可被覆盖）

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the D13 variant
# a) Concrete behavior and trace:
#    - The input format stays the same: a bottom-left vertical legend, plus non-legend markers made of
#      diagonal L-corner pairs or occasional solid 2x2 blocks.
#    - The original task used the legend order only to decide fill priority while preserving three things:
#      the legend cells, the current marker pixels, and all later marker pixels.
#    - This variant removes that protection entirely. Example: let entry A be processed first and entry B later.
#      If A's diamond passes through one of B's marker-arm cells, then:
#        original: that B cell survives A because it is protected;
#        variant: A paints over it immediately, so B may appear partially erased in the final image.
#      B is still interpreted from the original input, because the solver reads marker geometry from `g`, not from
#      the progressively modified output canvas.
#    - A 3x3 TL-BR marker pair shows the self-overwrite effect clearly: the opposite-diagonal diamond contains two of
#      the L-arm pixels, so those marker pixels are recolored in the variant but stayed visible before.
#
# b) Complexity calibration:
#    - If the generator allowed fully separated rectangles, the new rule would often collapse to the old one except
#      for self-overwrite. To make D13 visually meaningful, the generator now forces at least one early-vs-late
#      interaction where an early fill crosses the legend cells of higher legend rows.
#    - This preserves the original object vocabulary (legend rows, L-pairs, optional block, optional noop,
#      optional distractor) while making the removed protection visually undeniable.
#
# c) Implicit assumption audit against the original main():
#    - `marker_pts = argwhere((g == marker) & ~legend_mask)`: this already assumes marker detection comes from the
#      original input, not from the evolving canvas. That assumption must remain, otherwise early overwrites would
#      destroy the evidence needed for later entries.
#    - `protect = legend_mask.copy()` and the later union with future/current marker masks relied directly on the old
#      invariant that some pixels were never repaintable. That invariant is invalid here and those writes must vanish.
#    - The diagonal inference from the marker bbox remains valid because the input markers themselves are still placed
#      without overlap and still use the same L-shape conventions.
#
# d) Placement legality:
#    - Input markers still may not overlap each other or the legend.
#    - Bounding boxes may overlap spatially, but only through the fill regions that exist in the output process.
#    - The legend scan must remain stable, so marker placement is still forbidden in rows below the legend and in the
#      columns to the legend's left/upstream exactly as in the original generator.
#
# e) Visual observability:
#    - The legend order is visible spatially (bottom-to-top), so the destructive overwrite order is human-detectable.
#    - The forced legend-crossing fill ensures at least one sample-level effect that cannot be explained by the
#      original protected rule: an earlier fill paints over later legend cells that used to stay intact.
#
# f) Visual cleanliness:
#    - The generator only forces one interaction pair and leaves the rest close to the original task, avoiding dense
#      multi-layer chaos.
#    - Marker inputs remain non-overlapping, so the input canvas is still easy to parse even though the output can
#      contain destructive repainting.
#
# g) Color reasoning:
#    - Colors are still taken only from the visible legend mapping. No ordering or transformation depends on color ids.
#      The only new logic is overwrite order, which is shown by legend row position rather than hidden metadata.
#
# h) New solver assumptions and generator guarantees:
#    - `main()` now assumes that all later entries remain inferable from the original input grid. The generator
#      guarantees that every legend marker color remains distinct and that each intended L-pair/block is unambiguous
#      in the input.
#    - The generator additionally guarantees at least one overwrite of higher legend rows by an earlier fill, so the
#      variant genuinely diverges from the protected reference behavior without relying on hidden color-id logic.


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


def _mask_from_points(shape: tuple[int, int], pts: set[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r, c in pts:
        m[r, c] = True
    return m


def _marker_points(g: np.ndarray, legend_mask: np.ndarray, marker: int) -> set[tuple[int, int]]:
    return set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))


def _is_solid_block(marker_pts: set[tuple[int, int]]) -> bool:
    if len(marker_pts) != 4:
        return False
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    return max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1


def _paint_mask_from_points(shape: tuple[int, int], marker_pts: set[tuple[int, int]]) -> np.ndarray | None:
    if len(marker_pts) < 2:
        return None

    h, w = shape
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    if _is_solid_block(marker_pts):
        return _mask_from_points(shape, marker_pts)

    if rmin == rmax or cmin == cmax:
        return None

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

    return _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)


def _legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            mask[r, legend_c] = True
        if 0 <= r < h and legend_c + 1 < w:
            mask[r, legend_c + 1] = True
    return mask


def _paint_mask_for_marker(g: np.ndarray, legend_mask: np.ndarray, marker: int) -> np.ndarray | None:
    return _paint_mask_from_points(g.shape, _marker_points(g, legend_mask, marker))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint legend-driven marker regions with destructive overwrite semantics.

    Input structure:
    - A single-background grid contains a 2-column legend stacked near the lower-left edge.
    - Each legend row gives a marker color and its fill color, read from bottom to top.
    - Away from the legend, each marker color appears either as two opposite L-corners of one bbox or as one solid 2x2 block.

    Rule:
    - For every legend entry in bottom-to-top order, locate that marker color in the original input.
    - If it is a 2x2 block, repaint those four cells with the legend's fill color.
    - If it is an L-pair, infer which bbox diagonal carries the anchors and fill the opposite diagonal diamond.
    - Do not preserve any pre-existing pixels while painting: legend cells, the current marker, and later markers are all repaintable.

    Key insight:
    - The geometry is still read from the untouched input, but the output canvas is updated destructively in legend order,
      so earlier fills can erase evidence that belongs to later legend entries.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _legend_mask(g.shape, legend_c, entries)
    out = g.copy()

    for _, marker, fill in entries:
        paint = _paint_mask_for_marker(g, legend_mask, int(marker))
        if paint is None:
            continue
        out[paint] = int(fill)

    return out


def _main_protected_reference(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the original protected behavior, used only for generation diagnostics."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _legend_mask(g.shape, legend_c, entries)

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker = int(marker)
        marker_pts = _marker_points(g, legend_mask, marker)
        paint = _paint_mask_from_points(g.shape, marker_pts)
        if paint is None:
            continue

        protect = legend_mask.copy()
        for future_marker in marker_order[i + 1 :]:
            protect |= (g == future_marker) & (~legend_mask)

        if not _is_solid_block(marker_pts):
            protect |= _mask_from_points(g.shape, marker_pts)

        out[paint & (~protect)] = int(fill)

    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _legend_mask(g.shape, legend_c, entries)

    entry_info: list[dict[str, np.ndarray | int]] = []
    for _, marker, _ in entries:
        marker_pts = _marker_points(g, legend_mask, int(marker))
        marker_mask = _mask_from_points(g.shape, marker_pts)
        paint = _paint_mask_from_points(g.shape, marker_pts)
        if paint is None:
            paint = np.zeros(g.shape, dtype=bool)
        entry_info.append(
            {
                "marker": int(marker),
                "marker_mask": marker_mask,
                "paint": paint,
            }
        )

    future_legend_hits = 0
    future_marker_hits = 0
    legend_hits = 0
    self_hits = 0
    for i, info_i in enumerate(entry_info):
        paint_i = info_i["paint"]
        marker_mask_i = info_i["marker_mask"]
        assert isinstance(paint_i, np.ndarray)
        assert isinstance(marker_mask_i, np.ndarray)
        self_hits += int(np.count_nonzero(paint_i & marker_mask_i))
        legend_hits += int(np.count_nonzero(paint_i & legend_mask))
        for j in range(i + 1, len(entry_info)):
            marker_mask_j = entry_info[j]["marker_mask"]
            assert isinstance(marker_mask_j, np.ndarray)
            future_marker_hits += int(np.count_nonzero(paint_i & marker_mask_j))
            future_row = entries[j][0]
            if 0 <= future_row < g.shape[0]:
                for cc in (legend_c, legend_c + 1):
                    if 0 <= cc < g.shape[1] and paint_i[future_row, cc]:
                        future_legend_hits += 1

    protected_out = _main_protected_reference(g)
    variant_out = main(g)
    output_delta = int(np.count_nonzero(protected_out != variant_out))
    return {
        "future_legend_hits": future_legend_hits,
        "future_marker_hits": future_marker_hits,
        "legend_hits": legend_hits,
        "self_hits": self_hits,
        "output_delta": output_delta,
    }


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry
BLOCK_PROB = 0.25                # prob of one 2x2 block entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)

# ── CFG: D13 interaction forcing ───────────────────────────────────
DRIVER_FREE_SLOTS_MIN = 4        # keep the driver visually substantial
FUTURE_LEGEND_HIT_MIN = 1        # min later-legend cells crossed by an earlier fill

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25           # prob of adding a non-legend L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for a generic L-pair
DISTRACTOR_RETRIES = 60          # placement attempts for distractor
GEN_RETRIES = 300                # outer generate_input retry budget

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


def _l_pair_fill_mask(H: int, W: int, bbox: tuple[int, int, int, int], diag: str) -> np.ndarray:
    rmin, cmin, rmax, cmax = bbox
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    return _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)


def _sample_l_pair_placement(H: int, W: int) -> dict[str, object] | None:
    size = _pick_bbox_size(H, W)
    if size is None:
        return None
    bh, bw = size
    rmin = random.randint(0, H - bh)
    cmin = random.randint(0, W - bw)
    diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
    bbox = (rmin, cmin, rmin + bh - 1, cmin + bw - 1)
    cells = _l_pair_cells(bbox[0], bbox[1], bbox[2], bbox[3], diag)
    return {"cells": cells, "bbox": bbox, "diag": diag}


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place, accept=None):
    """Try to place an L-pair of `color` within `retries` attempts. Returns placement metadata or None."""
    for _ in range(retries):
        placement = _sample_l_pair_placement(H, W)
        if placement is None:
            continue
        cells = placement["cells"]
        assert isinstance(cells, list)
        if not can_place(cells):
            continue
        if accept is not None and not accept(placement):
            continue
        do_place(cells, color)
        return placement
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


def _driver_accept(
    placement,
    H: int,
    W: int,
    future_legend_mask: np.ndarray,
    forbidden: np.ndarray,
    occupied: np.ndarray,
) -> bool:
    bbox = placement["bbox"]
    diag = placement["diag"]
    cells = placement["cells"]
    assert isinstance(bbox, tuple)
    assert isinstance(diag, str)
    assert isinstance(cells, list)

    paint = _l_pair_fill_mask(H, W, bbox, diag)
    cell_set = set(cells)
    free_slots = 0
    for r, c in np.argwhere(paint):
        rc = (int(r), int(c))
        if rc in cell_set:
            continue
        if occupied[r, c] or forbidden[r, c]:
            continue
        free_slots += 1
    return free_slots >= DRIVER_FREE_SLOTS_MIN and int(np.count_nonzero(paint & future_legend_mask)) >= FUTURE_LEGEND_HIT_MIN


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
    driver_idx = 0
    types = ["L"] * n_entries
    idx_pool = [i for i in range(n_entries) if i != driver_idx]
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and idx_pool:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

    future_legend_mask = np.zeros((H, W), dtype=bool)
    for r, _, _ in legend_rows[driver_idx + 1 :]:
        if 0 <= r < H and 0 <= legend_c < W:
            future_legend_mask[r, legend_c] = True
        if 0 <= r < H and legend_c + 1 < W:
            future_legend_mask[r, legend_c + 1] = True

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

    driver_marker = legend_rows[driver_idx][1]
    driver_placement = _try_place_l_pair(
        H,
        W,
        driver_marker,
        LPAIR_RETRIES,
        can_place,
        do_place,
        accept=lambda placement: _driver_accept(placement, H, W, future_legend_mask, forbidden, occupied),
    )
    if driver_placement is None:
        return None

    for i, (_, marker, _) in enumerate(legend_rows):
        if i == driver_idx:
            continue
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
    if metrics["future_legend_hits"] < FUTURE_LEGEND_HIT_MIN:
        return None
    if metrics["output_delta"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
