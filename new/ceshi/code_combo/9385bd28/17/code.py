# Combo variant of 9385bd28 (index 17)
# combo_spec: D3/2+D6/2+D12/0
# Name: 扩展填充+无重叠保证+禁止2x2

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "solvers").is_dir():
    ROOT = ROOT.parent
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg


# Design analysis for the combo D3/2 + D6/2 + D12/0
# a) Concrete behavior and trace:
#    - The visible input still has a dominant background, a 2-column legend near
#      the lower-left edge, and scattered off-legend marker shapes.
#    - The original task allowed two marker kinds: an exact 6-cell L-pair or an
#      exact 2x2 block. It also allowed overlapping fill regions, so earlier
#      legend items had to avoid erasing later markers.
#    - In this combo, every realized legend marker must be exactly one L-pair.
#      Once the anchor diagonal is inferred, the solver fills the opposite
#      diagonal diamond of the bbox after first expanding the bbox by one cell on
#      all four sides. Example: if a marker pair spans rows 4..7 and cols 9..11,
#      the original fill lived inside 4..7 x 9..11; the combo fill instead uses
#      the expanded frame 3..8 x 8..12 and paints the corresponding larger
#      diamond, while the marker pixels themselves stay visible.
#    - Because the generator guarantees the expanded effect regions of all active
#      L-pairs are disjoint, there is no meaningful overwrite order anymore.
#      Solver-side protection can therefore drop "future marker" handling and
#      keep only legend cells plus the current marker cells protected.
# b) Complexity calibration:
#    - The task is simpler than the base puzzle, so the generator must preserve
#      enough variation to stop it from collapsing into a trivial template:
#      2-5 legend entries, optional noop rows, both diagonals, and multiple bbox
#      aspect classes are all kept.
#    - To make the D3 change legible in 2-4 examples, active L-pairs are placed
#      at least one cell away from the border and the generator rejects samples
#      where too few cells change only because of the extra one-cell expansion.
#    - To keep the D6 change visible, the generator preserves at least two real
#      L-pairs whenever possible so the solver is clearly handling multiple
#      independent regions rather than a single isolated object.
# c) Hidden-assumption audit:
#    - _find_legend still assumes the lowest-left non-background structure is the
#      legend stack; the generator keeps that geometry and clears both legend
#      columns above the stack.
#    - The base solver assumed some marker colors could form 2x2 blocks. That
#      branch is invalid now, so main() must reject any realized legend marker
#      that is not one exact L-pair.
#    - The base D3 logic could rely on future-marker protection because expanded
#      writes may leave the original bbox. Under D6/2 the new assumption is that
#      those expanded effect regions are already disjoint, so simpler protection
#      is sound only if the generator reserves those regions up front.
#    - Thin bboxes can satisfy extra corner tests, so diagonal scoring alone is
#      not enough. The solver now infers a diagonal and then checks that the
#      marker pixels exactly equal the canonical 6 L-pair cells for that bbox.
# d) Placement legality:
#    - Each active L-pair reserves its full expanded bbox rectangle, not merely
#      its 6 marker cells. This is stronger than required for exact diamond
#      disjointness and guarantees non-overlap, in-bounds placement, and clean
#      margins around the fill.
#    - The generator forbids any off-legend object from entering the legend scan
#      corridor and places active objects with a one-cell border margin so the
#      expansion is never clipped away.
# e) Visual observability:
#    - A human can read three stable cues from a few examples: all valid markers
#      are L-pairs, the painted area extends one cell beyond the marker-implied
#      rectangle, and different legend items act on separate islands instead of
#      nesting or overwriting each other.
#    - No ordering depends on color ids or other non-visual metadata; all rules
#      are recoverable from geometry and the visible legend.
# f) Visual cleanliness:
#    - Reserving the expanded bbox prevents different fills, markers, and
#      distractors from crowding the same neighborhood. This avoids ambiguous
#      examples where the extra ring would be hidden by another object.
# g) Change inferability and color mapping:
#    - Colors are still fully specified by the visible legend: left cell is the
#      marker color, right cell is the fill color. The combo changes only shape
#      and interaction constraints, not the color mapping mechanism.
#    - The generator rejects under-informative samples by measuring how many
#      output changes lie in the "expanded-only" halo, so the D3 component is
#      not allowed to degenerate back to the original task.
# h) New main() assumptions and generator guarantees:
#    - Every realized legend marker has either zero off-legend pixels (noop) or
#      exactly one legal 6-cell L-pair.
#    - Active L-pairs have mutually disjoint expanded effect regions.
#    - The generator constructs and checks those properties directly, so the
#      stricter solver assumptions are satisfied by design.


FILL_PAD = 1


GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

NOOP_PROB = 0.25
MIN_REAL_LPAIRS = 2

BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]

DISTRACTOR_PROB = 0.25

LPAIR_RETRIES = 240
DISTRACTOR_RETRIES = 120
GEN_RETRIES = 600

MIN_VISIBLE_HALO_CELLS = 6


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


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


def _l_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    diag: str,
) -> list[tuple[int, int]]:
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


def _infer_l_pair_diag(marker_pts: set[tuple[int, int]]) -> str | None:
    """Return the L-pair diagonal when marker_pts are exactly one legal L-pair."""
    if len(marker_pts) != 6:
        return None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
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
    if score_tlbr == score_trbl or (score_tlbr == 0 and score_trbl == 0):
        return None

    diag = "TLBR" if score_tlbr > score_trbl else "TRBL"
    expected = set(_l_pair_cells(rmin, cmin, rmax, cmax, diag))
    return diag if marker_pts == expected else None


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


def _fill_mask_for_bbox(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    diag: str,
) -> np.ndarray:
    """Base-task fill mask inside the unexpanded bbox."""
    if diag == "TLBR":
        a = (rmin, cmax)
        b = (rmax, cmin)
    else:
        a = (rmin, cmin)
        b = (rmax, cmax)
    return _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)


def _expanded_fill_mask(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    diag: str,
) -> np.ndarray:
    """Combo fill mask after expanding the inferred bbox by one cell on every side."""
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


def _expanded_bbox_cells(
    h: int,
    w: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> list[tuple[int, int]]:
    rr0 = max(0, rmin - FILL_PAD)
    rr1 = min(h - 1, rmax + FILL_PAD)
    cc0 = max(0, cmin - FILL_PAD)
    cc1 = min(w - 1, cmax + FILL_PAD)
    return [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids ambiguous L-anchor score ties."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _candidate_bbox_sizes(h: int, w: int) -> list[tuple[int, int]]:
    """Enumerate valid L-pair bbox sizes that fit within the given area."""
    sizes: list[tuple[int, int]] = []
    h_hi = min(BBOX_H_MAX, h)
    w_hi = min(BBOX_W_MAX, w)

    for bh in range(3, h_hi + 1):
        for bw in range(3, w_hi + 1):
            sizes.append((bh, bw))

    if h >= 2:
        for bw in range(4, w_hi + 1):
            sizes.append((2, bw))

    if w >= 2:
        for bh in range(4, h_hi + 1):
            sizes.append((bh, 2))

    return [size for size in sizes if _l_pair_valid_bbox(*size)]


def _pick_bbox_size(h: int, w: int) -> tuple[int, int] | None:
    """Pick a valid bbox size, biased toward smaller rectangles for easier packing."""
    sizes = _candidate_bbox_sizes(h, w)
    if not sizes:
        return None

    weights = []
    for bh, bw in sizes:
        shape_weight = (
            BBOX_SHAPE_WEIGHTS[0]
            if bh >= 3 and bw >= 3
            else BBOX_SHAPE_WEIGHTS[1]
            if bh == 2
            else BBOX_SHAPE_WEIGHTS[2]
        )
        weights.append(shape_weight / float(bh * bw))
    return random.choices(sizes, weights=weights, k=1)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor disjoint expanded regions implied by legend-coded L-corner pairs.

    Input structure:
    - The grid has one dominant background color and a vertical 2-column legend near the lower-left corner.
    - Each legend row stores a marker color in the left legend column and its fill color in the cell immediately to the right.
    - Away from the legend, every realized legend marker appears only as one exact pair of 3-cell L corners on opposite bbox corners; 2x2 blocks are not part of this variant.

    Rule:
    - Read each legend entry, collect the off-legend pixels of that marker color, and skip rows with no such pixels.
    - If a marker is realized, its pixels must form exactly one legal L-pair; infer which bbox diagonal contains the anchors.
    - Fill the opposite-diagonal diamond after first expanding the bbox outward by one cell on all four sides.
    - Preserve the legend cells and the current marker cells; no later-marker protection is needed because valid generated instances keep expanded effect regions disjoint.

    Key insight:
    - The L corners still reveal a hidden bbox and an anchor diagonal, but the painted region is larger than the marker-implied rectangle by exactly one outer ring, and each legend item acts on its own isolated island.
    """
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

    for _, marker, fill in entries:
        marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(marker_mask)))
        if not marker_pts:
            continue

        diag = _infer_l_pair_diag(marker_pts)
        assert diag is not None, "variant requires each realized marker to be one exact L-pair"

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        expanded = _expanded_fill_mask(h, w, rmin, rmax, cmin, cmax, diag)
        protect = legend_mask | marker_mask
        out[expanded & (~protect)] = int(fill)

    return out


def _build_legend(h: int, w: int, bg: int):
    """Create the legend stack and its masks."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([h - 1, h - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((h, w), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((h, w), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    h: int,
    w: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
    """Protect the legend scan corridor and all rows below the legend."""
    forbidden = np.zeros((h, w), dtype=bool)
    if legend_bot + 1 < h:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 2] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
        forbidden[r, legend_c + 1] = False
    return forbidden


def _choose_types(n_entries: int) -> list[str]:
    """Choose legend entry types while preserving multiple real L-pairs."""
    types = ["L"] * n_entries
    if n_entries <= MIN_REAL_LPAIRS:
        return types

    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and (types.count("L") - 1) >= MIN_REAL_LPAIRS:
        types[idx_pool.pop()] = "noop"
    return types


def _choose_l_pair_diags(types: list[str]) -> dict[int, str]:
    """Assign diagonals with near-balanced coverage across realized L-pairs."""
    l_indices = [i for i, item_type in enumerate(types) if item_type == "L"]
    if not l_indices:
        return {}

    if len(l_indices) == 1:
        return {l_indices[0]: random.choice(["TLBR", "TRBL"])}

    start = random.choice(["TLBR", "TRBL"])
    other = "TRBL" if start == "TLBR" else "TLBR"
    assigned = [start if k % 2 == 0 else other for k in range(len(l_indices))]
    random.shuffle(assigned)
    return {idx: diag for idx, diag in zip(l_indices, assigned)}


def _try_place_l_pair(
    h: int,
    w: int,
    color: int,
    diag: str,
    retries: int,
    reserved: np.ndarray,
    grid: np.ndarray,
    occupied: np.ndarray,
):
    """Place one L-pair and reserve its expanded bbox."""
    inner_h = h - 2 * FILL_PAD
    inner_w = w - 2 * FILL_PAD
    if inner_h < 2 or inner_w < 2:
        return None

    for _ in range(retries):
        size = _pick_bbox_size(inner_h, inner_w)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(FILL_PAD, h - bh - FILL_PAD)
        cmin = random.randint(FILL_PAD, w - bw - FILL_PAD)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        reserve_cells = _expanded_bbox_cells(h, w, rmin, rmax, cmin, cmax)

        if any(occupied[r, c] for r, c in cells):
            continue
        if any(reserved[r, c] for r, c in reserve_cells):
            continue

        for r, c in reserve_cells:
            reserved[r, c] = True
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        return {"type": "L", "bbox": (rmin, rmax, cmin, cmax), "diag": diag}
    return None


def _grid_matches_variant(
    grid: np.ndarray,
    legend_rows: list[tuple[int, int, int]],
    legend_mask: np.ndarray,
) -> bool:
    """Every realized legend marker must be exactly one L-pair."""
    for _, marker, _ in legend_rows:
        marker_pts = set(map(tuple, np.argwhere((grid == marker) & (~legend_mask))))
        if marker_pts and _infer_l_pair_diag(marker_pts) is None:
            return False
    return True


def _visible_halo_delta(inp: np.ndarray, out: np.ndarray, metas: list[dict]) -> int:
    """Count output changes that live only in the combo's extra one-cell expansion."""
    h, w = inp.shape
    total = 0
    for meta in metas:
        if meta["type"] != "L":
            continue
        rmin, rmax, cmin, cmax = meta["bbox"]
        diag = meta["diag"]
        base = _fill_mask_for_bbox(h, w, rmin, rmax, cmin, cmax, diag)
        expanded = _expanded_fill_mask(h, w, rmin, rmax, cmin, cmax, diag)
        halo_only = expanded & (~base)
        total += int(np.count_nonzero(halo_only & (out != inp)))
    return total


def _expanded_regions_are_disjoint(h: int, w: int, metas: list[dict]) -> bool:
    """Check the exact expanded fill masks for pairwise overlap."""
    seen = np.zeros((h, w), dtype=bool)
    for meta in metas:
        if meta["type"] != "L":
            continue
        rmin, rmax, cmin, cmax = meta["bbox"]
        mask = _expanded_fill_mask(h, w, rmin, rmax, cmin, cmax, meta["diag"])
        if np.any(seen & mask):
            return False
        seen |= mask
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(h, w, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(h, w, legend_c, legend_bot, legend_rows)
    reserved = forbidden | legend_mask
    occupied = legend_mask.copy()
    types = _choose_types(len(legend_rows))
    diag_by_index = _choose_l_pair_diags(types)
    if types.count("L") < MIN_REAL_LPAIRS:
        return None

    metas: list[dict] = []
    for i, (_, marker, _) in enumerate(legend_rows):
        if types[i] == "noop":
            metas.append({"type": "noop"})
            continue
        meta = _try_place_l_pair(
            h,
            w,
            marker,
            diag_by_index[i],
            LPAIR_RETRIES,
            reserved,
            grid,
            occupied,
        )
        if meta is None:
            return None
        metas.append(meta)

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(
                h,
                w,
                random.choice(avail),
                random.choice(["TLBR", "TRBL"]),
                DISTRACTOR_RETRIES,
                reserved,
                grid,
                occupied,
            )

    if not _grid_matches_variant(grid, legend_rows, legend_mask):
        return None
    if not _expanded_regions_are_disjoint(h, w, metas):
        return None

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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts"
    )
