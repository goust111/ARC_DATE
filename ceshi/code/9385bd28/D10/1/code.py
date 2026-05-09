# Variant of 9385bd28
# Dimension: D10 — 图例条目数量与复杂度 (meta)
# Change: 2-5 个图例条目，0-1 个无操作条目（标记色无实际标记像素）... → 大量图例条目（6-8 个）
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - Keep the original legend-reading and diamond-filling rule.
#     - Scale the legend to 6-8 rows, which forces a taller canvas and a denser set of
#       marker/fill correspondences for the solver to track.
#   input_appearance:
#     - The lower-left still contains a 2-column vertical legend.
#     - Each legend row is (marker color, fill color), read bottom -> top.
#     - Away from the legend there are many more marker instances than before:
#       usually 6-8 legend rows, with 0-1 noop rows and 0-1 block rows.
#     - The canvas is enlarged to roughly 20-25 in each dimension so the taller legend
#       and the extra marker objects remain visually separable.
#   behavior_trace:
#     original_flow:
#       - A 10-19 grid held 2-5 legend rows.
#       - The generator only had to fit a small number of L-pairs / blocks, so almost
#         any bbox size worked.
#     variant_flow:
#       - Example: with 7 legend rows, the bottom-left legend occupies rows 24..18.
#       - The solver still reads those rows bottom -> top and processes each marker color.
#       - The difference is not in one marker's local transformation, but in the amount of
#         bookkeeping: more legend mappings, more protected future markers, and a larger
#         scene where several diamonds can coexist.
#     concrete_difference:
#       - On the same type of input structure, the original generator would have produced
#         at most five rows of legend, while this variant visibly shows a tall 6-8-row
#         legend and many more non-legend marker groups to match it.
#   complexity_calibration:
#     - Using 6-8 entries is already near the ARC palette limit because marker colors
#       must be distinct and different from the background.
#     - To avoid accidental simplification, noop entries remain possible but rare, so most
#       samples still show at least six active mappings.
#     - Fill colors are biased away from self-color and background-only mappings to keep
#       the transformation visually legible even with many legend entries.
#   implicit_assumption_audit:
#     - _find_legend assumes the first non-bg cells from the bottom form a vertical 2-cell
#       legend strip. The generator preserves that by reserving the left legend zone.
#     - main() groups all non-legend pixels of one marker color into one object. Therefore
#       each legend marker color must correspond to exactly one placed marker group or be
#       a noop. The generator still enforces one group per legend color.
#     - main() infers the fill diagonal from L-anchor scores. Ambiguous bboxes like (2,2),
#       (2,3), (3,2) remain forbidden in the generator.
#     - main() protects future marker colors by legend order. With 6-8 rows this becomes
#       more important, but the original logic still holds because later colors remain
#       disjoint in the input grid.
#   placement_legality:
#     - The legend is placed at columns 0-2 and rows near the bottom.
#     - Non-legend pixels are forbidden below the legend bottom and in the legend columns
#       above it, so legend scanning cannot be corrupted.
#     - Marker pixels never overlap each other or the legend; bboxes may overlap only
#       through empty space, which is safe because the solver reads marker pixels from
#       the input grid, not from the filled output.
#   visual_observability:
#     - The D10 change is immediately visible because the legend is tall (6-8 rows) in
#       every sample, and the grid must be materially larger to accommodate it.
#     - The rule stays inferable from 2-4 examples because the legend still explicitly
#       shows every color mapping; the variation is quantitative, not hidden.
#   visual_cleanliness:
#     - Larger canvases and moderated bbox sizes prevent the extra objects from collapsing
#       into a single noisy cluster.
#     - Distractors are kept rare because the main source of difficulty here should be
#       legend load, not unrelated clutter.
#   color_inference:
#     - Color changes remain fully supported by the visible 2-column legend.
#     - The generator limits background fills and self-fills so the examples do not
#       degenerate into many low-contrast rows that are hard to distinguish by eye.

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

    This checks full L anchors rather than corner occupancy alone, which avoids
    ambiguity in narrow bounding boxes.
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
    Mask for cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neighbor
    path between corners a and b.
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
    Use a tall color legend to drive many rectangle fills at once.

    Input structure:
    - The bottom-left corner contains a vertical 2-column legend, now typically 6-8 rows tall.
    - Each legend row stores a marker color in the left cell and its fill color in the right cell.
    - Elsewhere, each marker color appears either as one diagonal pair of L-corners or as one solid 2x2 block.

    Rule:
    - Read the legend from bottom to top and process entries in that order.
    - For an L-pair, collect that marker color's non-legend pixels, find which diagonal holds the L-anchors,
      and fill the opposite diagonal's diamond region with the legend's fill color.
    - For a solid 2x2 marker block, recolor the block itself to the legend's fill color.
    - Always preserve legend cells and marker pixels that belong to later legend entries.

    Key insight:
    - The legend tells you the color mapping, while the L-shapes tell you which diagonal is marked;
      the cells to fill are the shortest-path diamond on the opposite diagonal of the marker bbox.
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
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for future_marker in future_markers:
            protect |= (g == future_marker) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
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
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (D10) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (20, 25)
GRID_W_RANGE = (20, 25)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (6, 8)
MAX_BG_FILLS = 1
SELF_FILL_PROB = 0.08

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.15
BLOCK_PROB = 0.20

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 12
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.82, 0.10, 0.08]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.12

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 150
LPAIR_RETRIES = 260
DISTRACTOR_RETRIES = 100
GEN_RETRIES = 500

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR', 'TRBL'}."""
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
    """True iff the bbox avoids an L-anchor score tie."""
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
        max_w = min(BBOX_W_MAX, W)
        if max_w < 4:
            return None
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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place an L-pair of `color` within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _choose_fill_colors(bg: int, marker_colors: list[int]) -> list[int]:
    """Bias fills away from bg/self so the larger legend stays visually readable."""
    fills: list[int] = []
    used_non_bg: set[int] = set()
    bg_fill_count = 0

    for marker in marker_colors:
        use_bg = bg_fill_count < MAX_BG_FILLS and random.random() < 0.10
        use_self = random.random() < SELF_FILL_PROB

        if use_bg and bg != marker:
            fill = bg
            bg_fill_count += 1
        elif use_self:
            fill = marker
        else:
            fresh = [c for c in range(10) if c != bg and c != marker and c not in used_non_bg]
            pool = fresh if fresh else [c for c in range(10) if c != marker and c != bg]
            if not pool:
                pool = [c for c in range(10) if c != marker]
            fill = random.choice(pool)
            if fill != bg:
                used_non_bg.add(fill)
        fills.append(fill)

    return fills


def _build_legend(H: int, W: int, bg: int):
    """Create the legend grid, rows, mask, and marker colors."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)
    fill_colors = _choose_fill_colors(bg, marker_colors)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i, (marker, fill) in enumerate(zip(marker_colors, fill_colors)):
        r = legend_bot - i
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    """Protect the legend scan corridor from accidental extra pixels."""
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
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)

    if random.random() < NOOP_PROB and n_entries >= 7:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        block_idx = idx_pool.pop()
        if types[block_idx] == "L":
            types[block_idx] = "block"

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

    for i, (_, marker, _) in enumerate(legend_rows):
        marker_type = types[i]
        if marker_type == "noop":
            continue
        if marker_type == "block":
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
            if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place):
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

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
