# Variant of 9385bd28
# Dimension: D6 — 处理顺序与重叠规则 (multi_step)
# Change: ordered_processing — 图例条目从下到上顺序处理，后处理的条目覆盖先处理的；保护后续条目的标记像素和图... → 无重叠保证（所有矩形互不相交）

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis
# - Concrete behavior:
#   The visible objects stay the same as in the base task: a bottom-left vertical
#   legend, several off-legend L-corner pairs, and an occasional 2x2 solid block.
#   The output still recolors each 2x2 block directly and still fills the
#   opposite diagonal diamond inside each L-pair bounding box.
#   The D6 change is only about interaction: every active object's effect region
#   is now spatially isolated, so no fill ever needs to protect a later object.
# - Trace of the changed step:
#   In the base task, if marker A has a large bbox and marker B sits inside or
#   across it, processing A first would overwrite B unless B's pixels were put
#   into a "future protection" mask. In this variant, generation forbids that
#   geometry: A's bbox and B's bbox must be disjoint. The solver can therefore
#   fill A while protecting only A's own marker cells and the legend.
# - Complexity calibration:
#   The legend still has 2-5 entries, diagonal orientation still varies, and
#   fill colors are still read from the legend. To keep the D6 change visible,
#   the generator avoids collapsing into single-object cases by preserving at
#   least two real L-pairs whenever possible.
# - Implicit assumptions from the base solver:
#   The removed assumption is that later marker pixels may lie inside an earlier
#   fill region. The kept assumptions are: each L-pair has one unambiguous anchor
#   diagonal, 2x2 blocks are exact 4-pixel squares, and marker detection should
#   read from the original input grid rather than the evolving output.
# - Placement legality:
#   The generator now reserves each full bbox before writing marker pixels.
#   This guarantees no active rectangles overlap, keeps all regions in-bounds,
#   and keeps the legend scan corridor clear.
# - Visual observability and cleanliness:
#   Because diamonds are separated rather than nested, the difference from the
#   base task is visible across a few examples: each legend entry acts on its own
#   island. No color-id ordering is used anywhere.
# - New solver assumptions and generator guarantees:
#   The solver now assumes that legend objects do not overlap in their effect
#   regions. The generator guarantees this by reserving each L-pair bbox and each
#   block footprint before placement, and by keeping distractors disjoint too.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor isolated marker-defined regions according to a bottom-left legend.

    Input structure:
    - The grid has one dominant background color and a vertical two-column legend near the lower-left corner.
    - Each legend row gives a marker color in the left legend column and its fill color beside it.
    - Away from the legend, each active marker is either a diagonal pair of 3-pixel L-corners or a solid 2x2 block.

    Rule:
    - Read every legend entry, find all off-legend pixels of that marker color, and ignore entries with no such pixels.
    - A solid 2x2 marker block is recolored directly to its legend fill color.
    - For an L-pair, detect which bbox diagonal contains the L-anchors; fill the opposite diagonal diamond inside that bbox.
    - Preserve the legend cells and preserve the current marker pixels so the corner clues remain visible.

    Key insight:
    - The variant removes order interactions by construction: marker regions do not overlap, so each legend item can be solved independently.
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

    for _, marker, fill in entries:
        marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(marker_mask)))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            out[marker_mask] = int(fill)
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

        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        protect = legend_mask | marker_mask
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# -- CFG: canvas & grid -------------------------------------------------------
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# -- CFG: legend --------------------------------------------------------------
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# -- CFG: marker types --------------------------------------------------------
NOOP_PROB = 0.25
BLOCK_PROB = 0.20
MIN_REAL_LPAIRS = 2

# -- CFG: L-pair bbox shape ---------------------------------------------------
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# -- CFG: distractor ----------------------------------------------------------
DISTRACTOR_PROB = 0.25

# -- CFG: retry budgets -------------------------------------------------------
BLOCK_RETRIES = 120
LPAIR_RETRIES = 180
DISTRACTOR_RETRIES = 80
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


def _region_free(mask: np.ndarray, forbidden: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    """True iff the inclusive rectangle is in-bounds and clear in both masks."""
    H, W = mask.shape
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    return not mask[r0 : r1 + 1, c0 : c1 + 1].any() and not forbidden[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve_region(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Reserve an inclusive rectangle."""
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _write_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    """Paint a list of cells with one color."""
    for r, c in cells:
        grid[r, c] = color


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    reserved_regions: np.ndarray,
    forbidden: np.ndarray,
    grid: np.ndarray,
) -> bool:
    """Place one L-pair while reserving its full bbox."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        if not _region_free(reserved_regions, forbidden, rmin, cmin, rmax, cmax):
            continue
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        _reserve_region(reserved_regions, rmin, cmin, rmax, cmax)
        _write_cells(grid, cells, color)
        return True
    return False


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
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    """Protect the legend scan corridor and the rows below the legend."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 2] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
        forbidden[r, legend_c + 1] = False
    return forbidden


def _choose_types(n_entries: int) -> list[str]:
    """Choose entry types while keeping enough visible L-pairs."""
    types = ["L"] * n_entries
    if n_entries <= MIN_REAL_LPAIRS:
        return types

    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)

    if random.random() < NOOP_PROB and (types.count("L") - 1) >= MIN_REAL_LPAIRS:
        types[idx_pool.pop()] = "noop"

    if idx_pool and random.random() < BLOCK_PROB and (types.count("L") - 1) >= MIN_REAL_LPAIRS:
        types[idx_pool.pop()] = "block"

    return types


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    reserved_regions = legend_mask.copy()
    types = _choose_types(len(legend_rows))

    for i, (_, marker, _) in enumerate(legend_rows):
        entry_type = types[i]
        if entry_type == "noop":
            continue
        if entry_type == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                r1 = r0 + 1
                c1 = c0 + 1
                if not _region_free(reserved_regions, forbidden, r0, c0, r1, c1):
                    continue
                _reserve_region(reserved_regions, r0, c0, r1, c1)
                _write_cells(grid, [(r0, c0), (r0, c1), (r1, c0), (r1, c1)], marker)
                placed = True
                break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, reserved_regions, forbidden, grid):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, reserved_regions, forbidden, grid)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
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
