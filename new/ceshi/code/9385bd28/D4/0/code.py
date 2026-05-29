# Variant of 9385bd28
# Dimension: D4 — 图例查找与颜色映射 (symbol_instructions)
# Change: 左下角垂直排列的颜色对图例：左列为标记色，右列为填充色；从下到上读取处理顺序... → 图例放置在网格边缘任意位置

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the D4 variant
# ----------------------------------
# concretization:
# - Keep the original marker vocabulary unchanged: each actionable color still appears either
#   as a diagonal L-pair or as a solid 2x2 block, and the filled region is still the diamond
#   between the opposite bbox corners.
# - Only the instruction lookup changes. The legend remains a vertical stack of 2-cell rows
#   whose local encoding is still "left cell = marker color, right cell = fill color".
# - To avoid introducing a second independent change, the legend is not rotated. Instead it can
#   slide to either vertical edge: columns (0,1) on the left or (W-2,W-1) on the right, with
#   its top row anywhere from 0 to H-n. Touching row 0 / row H-1 yields the visible corner cases
#   (left-up, left-down, right-up, right-down) mentioned in the task description.
# - Processing order stays bottom->top in absolute grid coordinates. Example:
#   if a right-edge legend occupies rows 1..3, the solver must read row 3 first, then row 2,
#   then row 1. Under the old logic, the same grid would be misread because the bottom-most
#   non-background pixel is no longer guaranteed to be the legend start.
#
# complexity calibration:
# - The visual change is strong enough to notice because the instruction strip may move from the
#   bottom-left to the top-left, top-right, bottom-right, or a mid-edge slot.
# - The rule stays learnable because all other mechanics are unchanged; the human only has to
#   add "find the edge legend first" rather than infer a new fill geometry or new color algebra.
# - A single fixed local encoding (marker on the left, fill on the right) avoids extra branches
#   such as orientation-dependent reading directions.
#
# hidden assumptions review:
# - Original `_find_legend()` assumed the first non-bg cell found while scanning upward from the
#   bottom belonged to the legend. This breaks as soon as the legend can sit near the top or on
#   the right edge.
# - Original generator protected a wedge-shaped area tailored to that scan path. Once the legend
#   can move, that protection is no longer meaningful.
# - The solver's later logic only needs a correct `(legend_col, entries, legend_mask)` and does
#   not care where the legend lives, so the rest of `main()` can remain almost unchanged.
#
# placement legality:
# - The generator reserves both candidate edge pairs `(0,1)` and `(W-2,W-1)` for legend usage.
#   Only the actual legend rows are released. This guarantees that the solver sees exactly one
#   contiguous edge run of marker cells, even when a fill color equals the background.
# - All non-legend objects are placed strictly in the interior columns `2..W-3`, so markers,
#   blocks, and distractors cannot collide with the legend or fake another edge legend.
#
# observability / cleanliness / color inference:
# - Because the legend still shows explicit marker->fill color pairs, the color mapping remains
#   human-readable from a small number of examples.
# - Reserving the edge pairs keeps the border visually tidy: the legend is the only structured
#   border object, and the L-pairs stay in the interior where their bbox relationship is clear.
# - The generator intentionally samples top / bottom / middle placements on both sides so the new
#   search requirement appears repeatedly instead of degenerating back to the original layout.


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive row runs where a 1D boolean mask is True."""
    runs: list[tuple[int, int]] = []
    h = int(mask.shape[0])
    r = 0
    while r < h:
        if not bool(mask[r]):
            r += 1
            continue
        r0 = r
        while r + 1 < h and bool(mask[r + 1]):
            r += 1
        runs.append((r0, r))
        r += 1
    return runs


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs attached to the left or right edge.
    Each legend row is still (marker at col c, fill at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    assert w >= 2, "grid must be at least 2 columns wide"

    candidate_cols = [0]
    if w - 2 != 0:
        candidate_cols.append(w - 2)

    candidates: list[tuple[int, list[tuple[int, int, int]]]] = []
    for c in candidate_cols:
        if c + 1 >= w:
            continue
        marker_mask = g[:, c] != bg
        for r0, r1 in _contiguous_runs(marker_mask):
            if r1 - r0 + 1 < 2:
                continue
            entries = [(r, int(g[r, c]), int(g[r, c + 1])) for r in range(r1, r0 - 1, -1)]
            candidates.append((c, entries))

    assert len(candidates) == 1, "legend must be the unique contiguous edge stack"
    return candidates[0]


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
    Recolor the diamond between diagonal corner markers by following an edge legend.

    Input structure:
    - The board has one background color and one vertical two-column legend attached to either the left edge or the right edge.
    - Each legend row still encodes a marker/fill pair from left to right, and the rows are interpreted from the legend's bottom upward.
    - Away from the legend, each marker color appears either as two opposite L-corners of a bbox or as a solid 2x2 block.

    Rule:
    - First locate the edge legend, then process its rows bottom->top.
    - For an L-pair, inspect which bbox diagonal actually carries the L-anchors; fill the opposite diagonal's diamond with the legend's fill color.
    - For a 2x2 block, recolor the block itself with the legend's fill color.
    - Never overwrite legend cells, and do not destroy marker pixels that belong to later legend entries.

    Key insight: the transformation logic for each marker color is unchanged; the only new challenge is that the instruction strip must be found on the grid edge before the original fill rule can be applied.
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
    marker_order = [m for _, m, _ in entries]

    # Process legend entries bottom -> top; earlier entries must not destroy later marker pixels.
    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Protect legend cells and all marker pixels for entries that appear later (higher) in the legend.
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        # Also protect current marker pixels in the common case (so markers remain visible),
        # except when they form a solid 2x2 block (treated as a filled marker that gets recolored).
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
        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = eff_fill

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries
LEGEND_SIDE_WEIGHTS = [0.4, 0.6] # left / right edge
LEGEND_POS_WEIGHTS = [0.40, 0.20, 0.40]  # top / bottom / middle

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry (marker with no pixels)
BLOCK_PROB = 0.25                # prob of one 2x2 block entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25           # prob of adding a non-legend L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for an L-pair
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


def _try_place_l_pair(
    H: int,
    usable_W: int,
    c_lo: int,
    c_hi: int,
    color: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, usable_W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(c_lo, c_hi - bw + 1)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _pick_legend_top(H: int, n_entries: int) -> int:
    """Sample a legend position so top / bottom / middle placements all occur."""
    max_top = H - n_entries
    mode = random.choices(["top", "bottom", "middle"], weights=LEGEND_POS_WEIGHTS)[0]
    if mode == "top":
        return 0
    if mode == "bottom":
        return max_top
    if max_top <= 1:
        return random.randint(0, max_top)
    return random.randint(1, max_top - 1)


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
    side = random.choices(["left", "right"], weights=LEGEND_SIDE_WEIGHTS)[0]
    legend_c = 0 if side == "left" else W - 2
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    legend_top = _pick_legend_top(H, n_entries)

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_top + n_entries - 1 - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Reserve both candidate edge pairs so only the actual legend can occupy them."""
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[:, :2] = True
    forbidden[:, W - 2 :] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
        forbidden[r, legend_c + 1] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_rows)
    occupied = legend_mask.copy()

    c_lo, c_hi = 2, W - 3
    usable_W = c_hi - c_lo + 1
    if usable_W < 2:
        return None

    # Assign a marker type per legend entry.
    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
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

    # Place markers for each legend entry.
    for i, (_, marker, _) in enumerate(legend_rows):
        typ = types[i]
        if typ == "noop":
            continue
        if typ == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(c_lo, c_hi - 1)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, usable_W, c_lo, c_hi, marker, LPAIR_RETRIES, can_place, do_place):
                return None

    # Optional distractor L-pair in a non-legend color.
    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(
                H,
                usable_W,
                c_lo,
                c_hi,
                random.choice(avail),
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
            )

    # Ensure bg remains the most common color (so get_bg picks it).
    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    # Oracle check: solver must run without errors.
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
