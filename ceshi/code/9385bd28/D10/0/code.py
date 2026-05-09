# Variant of 9385bd28
# Dimension: D10 — 图例条目数量与复杂度 (meta)
# Change: 2-5 个图例条目，0-1 个无操作条目（标记色无实际标记像素）... → 单条目图例

# Design analysis for the D10 variant
# ===================================
# a) Concrete behavior and trace
#    - The legend is reduced to exactly one bottom-left row: `(marker_color, fill_color)`.
#    - The non-legend area now contains several disconnected objects of that single marker color.
#      Most are L-pairs (two opposite L-corners of one bbox); optionally one object is a 2x2 block.
#    - Every object uses the same fill color from the legend row.
#    - Small trace:
#      original task:
#        legend rows = [(marker 2 -> fill 7), (marker 3 -> fill 4)]
#        solver loops by legend row, so color 2 and color 3 define separate fills.
#      D10 variant:
#        legend row = [(marker 2 -> fill 7)]
#        non-legend area may contain three disconnected color-2 objects.
#        A color-grouped solver would wrongly merge all color-2 pixels into one bbox.
#        Therefore the new solver must split the marker color into connected components first,
#        then solve each component as its own L-pair/block, all with fill color 7.
#
# b) Complexity calibration
#    - Keeping only one legend row removes color-mapping complexity, so the spatial task must remain.
#    - To avoid becoming trivial, each sample contains 2-3 L-pairs, plus an optional 2x2 block.
#    - All objects share one marker color and one fill color; this is the visible signature of D10.
#
# c) Implicit assumptions from the original solver that break here
#    - `_find_legend` still works: it only needs the bottom-left stacked pair format; a single row is a valid stack.
#    - `for ... in entries` previously assumed one marker color per object family. With one legend row,
#      `marker_pts = all pixels of that color` is no longer one object; it becomes a union of many objects.
#    - `future_markers` protection is obsolete because there are no later legend entries.
#    - A crucial hidden detail: one L-pair is itself disconnected. A raw 4-connected split yields two corner pieces,
#      not one full object. The new solver must pair corner pieces back together before reusing the old bbox logic.
#
# d) Placement legality
#    - Generator keeps the legend isolated near the bottom-left.
#    - Objects must stay in bounds, cannot touch the legend orthogonally, and cannot touch each other
#      orthogonally (so connected-component splitting is stable).
#    - Their bboxes also cannot overlap or touch, which keeps output diamonds visually separate.
#
# e) Visual observability
#    - The crucial visible change is repetition: one legend row controls multiple same-color marker objects.
#    - Fill color is chosen different from both background and marker, so the transformation is always visible.
#
# f) Visual cleanliness
#    - No distractor colors and no noop row are generated here. Those were acceptable in the original
#      multi-entry task, but in the single-entry variant they add clutter without teaching the new rule.
#    - Disjoint bboxes keep the output readable instead of merging all filled diamonds into one blob.
#
# g) Color-rule inferability
#    - There is only one mapping `(marker -> fill)`, repeated several times per sample.
#    - Because the same marker color is reused across multiple objects, a solver can infer that the legend
#      is not object-specific; it is color-specific and applies repeatedly.

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
    Mask for cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b.
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


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """4-connected components of a boolean mask, returned as sets of (r, c)."""
    remaining = set(map(tuple, np.argwhere(mask)))
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    comps.sort(key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))
    return comps


def _mask_from_pts(shape: tuple[int, int], pts: set[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r, c in pts:
        m[r, c] = True
    return m


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _is_block_pts(pts: set[tuple[int, int]]) -> bool:
    rmin, cmin, rmax, cmax = _bbox_of_pts(pts)
    return len(pts) == 4 and rmax - rmin == 1 and cmax - cmin == 1


def _l_pair_fill_diagonal(pts: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the fill diagonal endpoints for a valid 6-cell L-pair object, else None."""
    if len(pts) != 6:
        return None

    rmin, cmin, rmax, cmax = _bbox_of_pts(pts)
    if rmin == rmax or cmin == cmax:
        return None

    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    tl_anchor = _l_anchor_present(pts, tl[0], tl[1], (0, 1), (1, 0))
    tr_anchor = _l_anchor_present(pts, tr[0], tr[1], (0, -1), (1, 0))
    bl_anchor = _l_anchor_present(pts, bl[0], bl[1], (0, 1), (-1, 0))
    br_anchor = _l_anchor_present(pts, br[0], br[1], (0, -1), (-1, 0))

    score_tlbr = int(tl_anchor) + int(br_anchor)
    score_trbl = int(tr_anchor) + int(bl_anchor)
    if score_tlbr == 0 and score_trbl == 0:
        return None
    if score_tlbr == score_trbl:
        return None
    if score_tlbr > score_trbl:
        return tr, bl
    return tl, br


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reuse a single legend mapping across several marker objects.

    Input structure:
    - The bottom-left legend is exactly one 2-cell row: marker color in the left cell, fill color in the right cell.
    - Outside the legend, the same marker color appears in several disconnected objects.
    - Each object is either an L-pair at opposite corners of one bbox or a solid 2x2 block.

    Rule:
    - Read the lone legend row to get one `(marker, fill)` mapping.
    - Split all non-legend pixels of that marker color into small connected pieces.
    - Treat any 2x2 piece as a standalone block, and pair the remaining corner-pieces into valid L-pair objects.
    - For every reconstructed L-pair, infer which diagonal carries the anchors and fill the opposite diagonal's diamond.
    - Recolor each 2x2 block directly to the fill color while leaving other marker pixels intact.

    Key insight:
    - The legend no longer selects among multiple colors; instead, one color rule is reused repeatedly,
      so the solver must rebuild same-colored objects from corner fragments before applying the original geometry.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert len(entries) == 1, "variant expects a single legend entry"

    legend_r, marker, fill = entries[0]
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[legend_r, legend_c] = True
    legend_mask[legend_r, legend_c + 1] = True

    marker_mask = (g == marker) & (~legend_mask)
    pieces = _connected_components(marker_mask)
    assert pieces, "single-entry legend must drive at least one non-legend marker object"

    out = g.copy()
    all_marker_mask = marker_mask.copy()

    objects: list[set[tuple[int, int]]] = []
    corner_pieces: list[set[tuple[int, int]]] = []
    for pts in pieces:
        if _is_block_pts(pts):
            objects.append(pts)
        else:
            corner_pieces.append(pts)

    used_corner_idxs: set[int] = set()
    for i, pts in enumerate(corner_pieces):
        if i in used_corner_idxs:
            continue
        matches: list[int] = []
        for j in range(i + 1, len(corner_pieces)):
            if j in used_corner_idxs:
                continue
            if _l_pair_fill_diagonal(pts | corner_pieces[j]) is not None:
                matches.append(j)
        assert len(matches) == 1, "each L-corner piece must pair with exactly one opposite corner piece"
        j = matches[0]
        used_corner_idxs.add(i)
        used_corner_idxs.add(j)
        objects.append(pts | corner_pieces[j])
    assert len(used_corner_idxs) == len(corner_pieces), "all non-block marker pieces must be paired"

    objects.sort(key=_bbox_of_pts)

    for pts in objects:
        rmin, cmin, rmax, cmax = _bbox_of_pts(pts)
        comp_mask = _mask_from_pts(g.shape, pts)

        if _is_block_pts(pts):
            protect = legend_mask | (all_marker_mask & (~comp_mask))
            out[comp_mask & (~protect)] = int(fill)
            continue

        diag = _l_pair_fill_diagonal(pts)
        assert diag is not None, "non-block marker object must be a valid L-pair"
        a, b = diag
        protect = legend_mask | all_marker_mask
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (D10 single-entry variant) ---------------

# Solver assumptions guaranteed by the generator:
# 1. Legend has exactly one row.
# 2. There are 2-3 L-pair components, optionally plus one 2x2 block.
# 3. All non-legend marker objects share the legend marker color.
# 4. Components are 4-disconnected and their bboxes do not overlap or touch.
# 5. Fill color differs from bg and marker, so the transformation is visible.

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ROW_OFFSET_CHOICES = (0, 1)   # row = H-1 or H-2

# ── CFG: repeated objects under one legend entry ───────────────────
LPAIR_COUNT_RANGE = (2, 3)
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]   # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 180
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


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Treat a 1-cell gap as separated; touching edges/corners counts as too close."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        bbox = (rmin, cmin, rmin + bh - 1, cmin + bw - 1)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, bbox[2], bbox[3], diag)
        if can_place(cells, bbox):
            do_place(cells, color, bbox)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    """Create the single legend row and its masks."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_r = H - 1 - random.choice(LEGEND_ROW_OFFSET_CHOICES)

    pool = [c for c in range(10) if c != bg]
    marker = random.choice(pool)
    fill_pool = [c for c in range(10) if c != bg and c != marker]
    fill = random.choice(fill_pool)

    grid = np.full((H, W), bg, dtype=int)
    legend_mask = np.zeros((H, W), dtype=bool)
    grid[legend_r, legend_c] = marker
    grid[legend_r, legend_c + 1] = fill
    legend_mask[legend_r, legend_c] = True
    legend_mask[legend_r, legend_c + 1] = True
    return grid, legend_mask, legend_c, legend_r, marker, fill


def _build_forbidden(H: int, W: int, legend_c: int, legend_r: int) -> np.ndarray:
    """Reserve space so the one-row legend remains isolated and easy to parse."""
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[legend_r:, :] = True
    forbidden[:legend_r, :legend_c + 1] = True
    forbidden[legend_r, legend_c] = False
    forbidden[legend_r, legend_c + 1] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    grid, legend_mask, legend_c, legend_r, marker, _ = _build_legend(H, W, bg)
    forbidden = _build_forbidden(H, W, legend_c, legend_r)

    occupied = legend_mask.copy()
    used_boxes: list[tuple[int, int, int, int]] = []
    corner_pieces: list[set[tuple[int, int]]] = []

    def can_place(cells, bbox):
        cell_set = set(cells)
        if len(cell_set) != len(cells):
            return False
        if any(_boxes_touch_or_overlap(bbox, other) for other in used_boxes):
            return False
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cell_set:
                    continue
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                    return False
        candidate_components = _connected_components(_mask_from_pts((H, W), cell_set))
        if len(candidate_components) == 2:
            for cand in candidate_components:
                for existing in corner_pieces:
                    if _l_pair_fill_diagonal(cand | existing) is not None:
                        return False
        return True

    def do_place(cells, color, bbox):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        used_boxes.append(bbox)
        comps = _connected_components(_mask_from_pts((H, W), set(cells)))
        if len(comps) == 2:
            corner_pieces.extend(comps)

    n_pairs = random.randint(*LPAIR_COUNT_RANGE)
    for _ in range(n_pairs):
        if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place):
            return None

    if random.random() < BLOCK_PROB:
        placed = False
        for _ in range(BLOCK_RETRIES):
            r0 = random.randint(0, H - 2)
            c0 = random.randint(0, W - 2)
            bbox = (r0, c0, r0 + 1, c0 + 1)
            cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
            if can_place(cells, bbox):
                do_place(cells, marker, bbox)
                placed = True
                break
        if not placed:
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    marker_mask = (grid == marker) & (~legend_mask)
    pieces = _connected_components(marker_mask)
    objects = sum(1 for pts in pieces if _is_block_pts(pts)) + len(corner_pieces) // 2
    if objects < 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out != grid):
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
