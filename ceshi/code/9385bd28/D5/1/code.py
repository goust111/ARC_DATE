# Variant of 9385bd28
# Dimension: D5 — 填充变换方式 (transform)
# Change: rectangle_fill — 用图例对应的填充色填充标记对之间的菱形区域；2x2 方块直接重新上色... → 条纹填充替代实心填充

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


# Design analysis for the D5 variant:
# a) Concrete behavior and trace
#    - The input still has the same two visible structures: a bottom-left legend of
#      (marker_color, fill_color) pairs and scattered non-legend marker shapes.
#    - Each L-pair still determines a bounding box and, via its anchored diagonal,
#      the opposite diagonal whose shortest-path cells form the target diamond.
#    - The change happens only after that diamond is found:
#        original task: every writable diamond cell becomes fill_color
#        new variant: writable diamond cells alternate between fill_color and bg
#                     in parallel bands.
#    - Stripe direction is determined by bbox aspect ratio, which is visible:
#        height >= width -> horizontal stripes (alternate by row from the bbox top)
#        width > height  -> vertical stripes (alternate by column from the bbox left)
#    - Example trace:
#        Suppose the fill diagonal is TL->BR inside a 5x3 bbox. The original solver
#        would set all writable diamond cells to fill_color. The new solver first
#        decides "5x3 is tall", so rows rmin, rmin+2, ... inside the diamond become
#        fill_color while rows rmin+1, rmin+3, ... are explicitly reset to bg.
#        The same markers therefore produce a striped diamond instead of a solid one.
#
# b) Complexity calibration
#    - The rule uses only two stripe orientations, avoiding excess hidden freedom.
#    - Orientation comes from geometry rather than color ids or processing order.
#    - The generator keeps at least two genuine L-pairs whenever possible and biases
#      them toward both tall and wide boxes so the training set can exhibit both
#      horizontal and vertical striping.
#
# c) Implicit-assumption audit of the original main()
#    - `_find_legend`, anchor detection, and diamond construction do not depend on
#      solid filling and remain valid unchanged.
#    - The original assignment `out[diamond & (~protect)] = fill` relies on the old
#      assumption that non-filled cells inside the target region should preserve the
#      previous canvas. That is false now: stripe gaps must become background even if
#      an earlier fill already painted them. The new code therefore writes bg first
#      on every writable diamond cell, then repaints the selected stripe subset with
#      fill_color.
#    - Current marker protection is still needed; otherwise stripe gaps could erase
#      the marker L-corners themselves.
#    - 2x2 blocks remain direct recolors, matching the variant statement exactly.
#
# d) Placement legality
#    - Markers still cannot overlap, cannot enter the forbidden legend-scan region,
#      and must stay in-bounds.
#    - The generator now rejects square L-pair boxes so stripe direction is visually
#      determined by the bbox aspect ratio instead of an arbitrary tie.
#    - At least one L-pair is required; otherwise the instance would not exercise
#      the new striped transform at all.
#
# e) Visual observability
#    - Stripe direction depends only on visible bbox shape: tall diamonds show row
#      bands, wide diamonds show column bands.
#    - Fill colors are chosen away from the background so the alternating pattern is
#      always visible.
#
# f) Visual cleanliness
#    - The legend and marker geometry are unchanged, so object detection remains the
#      same task. The extra complexity is localized to the fill texture itself.
#    - Overlaps are still legal as in the base task, but explicit bg stripe writing
#      keeps the pattern crisp instead of leaving accidental remnants from earlier
#      fills.
#
# g) Color-change inferability
#    - The legend still provides the fill color directly.
#    - The only new color behavior is that every other stripe cell stays/returns to
#      background, which is structurally signaled by the alternating row/column bands.
#
# New solver assumptions introduced by this variant and matched by generate_input():
#    - There is at least one non-block L-pair.
#    - L-pair bboxes are non-square, so the human-visible aspect ratio determines
#      stripe direction unambiguously.
#    - Fill colors for legend entries differ from background, ensuring the stripe
#      texture changes the output in a visible way.


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
    assert start_r is not None and start_c is not None and start_c < w - 1, \
        "legend must exist at bottom-left with at least 2 columns"

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


def _l_anchor_present(pts: set[tuple[int, int]], r: int, c: int, arm1: tuple[int, int], arm2: tuple[int, int]) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _diamond_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
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


def _stripe_orientation(rmin: int, rmax: int, cmin: int, cmax: int) -> str:
    """Choose horizontal stripes for tall/square boxes, vertical stripes for wide boxes."""
    height = rmax - rmin + 1
    width = cmax - cmin + 1
    return "horizontal" if height >= width else "vertical"


def _paint_striped_diamond(
    out: np.ndarray,
    diamond: np.ndarray,
    protect: np.ndarray,
    fill: int,
    bg: int,
    orientation: str,
    rmin: int,
    cmin: int,
) -> None:
    """Write alternating fill/bg stripes on the writable part of `diamond`."""
    paint_mask = diamond & (~protect)
    if not np.any(paint_mask):
        return

    out[paint_mask] = bg
    if orientation == "horizontal":
        stripe_fill = paint_mask & (((np.arange(out.shape[0]) - rmin) % 2 == 0)[:, None])
    else:
        stripe_fill = paint_mask & (((np.arange(out.shape[1]) - cmin) % 2 == 0)[None, :])
    out[stripe_fill] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the diamond opposite each marker diagonal using striped legend colors.

    Input structure:
    - A single-background grid contains a bottom-left vertical legend whose rows are (marker_color, fill_color).
    - Away from the legend, each marker color appears either as two opposite L-corners of a bbox or as a solid 2x2 block.
    - The L-corners still identify which axis-aligned bbox and opposite-corner diamond belong to that legend entry.

    Rule:
    - Read legend entries from bottom to top and collect the non-legend pixels of that marker color.
    - Detect which bbox diagonal carries the L-shaped anchors; the other diagonal defines the target diamond exactly as in the base task.
    - For L-pairs, fill the writable diamond cells with alternating fill-color and background stripes: tall boxes use horizontal row bands, wide boxes use vertical column bands.
    - Solid 2x2 marker blocks are recolored directly, while legend cells and later markers stay protected from earlier writes.

    Key insight: The marker geometry still picks the diamond, but the bbox aspect ratio determines whether that diamond is striped by rows or by columns.
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

        # Preserve L-markers, but allow 2x2 blocks to be recolored wholesale.
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
        orientation = _stripe_orientation(rmin, rmax, cmin, cmax)
        _paint_striped_diamond(out, diamond, protect, eff_fill, bg, orientation, rmin, cmin)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.2                  # prob of one noop entry, only when >=3 entries remain
BLOCK_PROB = 0.2                 # prob of one 2x2 block entry, only when >=2 L-pairs remain

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)
BBOX_PICK_RETRIES = 40           # retries to find a non-square bbox with preferred aspect

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
    """True if (h, w) avoids an L-anchor score tie.

    Fails for (2,2), (2,3), (3,2). Safe: h>=3 and w>=3, or h==2 and w>=4, or h>=4 and w==2.
    """
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int, preferred_orientation: str | None = None) -> tuple[int, int] | None:
    """Pick a valid non-square L-pair bbox, optionally matching a preferred stripe orientation."""
    for _ in range(BBOX_PICK_RETRIES):
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
                continue
            bw = random.randint(4, w_hi)
        else:
            h_hi = min(BBOX_H_MAX, H)
            if h_hi < 4:
                continue
            bh = random.randint(4, h_hi)
            bw = 2

        if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw) or bh == bw:
            continue
        orientation = "horizontal" if bh > bw else "vertical"
        if preferred_orientation is None or orientation == preferred_orientation:
            return bh, bw
    return None


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    preferred_orientation: str | None = None,
) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W, preferred_orientation)
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
        fill_pool = [c for c in range(10) if c != bg and c != m]
        if not fill_pool:
            return None
        fill = random.choice(fill_pool)
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
        forbidden[legend_bot + 1:, :] = True
    forbidden[:legend_bot + 1, :legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _assign_marker_types(n_entries: int) -> list[str]:
    """Choose legend entry types while preserving at least one, usually two, striped L-pairs."""
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)

    l_remaining = n_entries
    if n_entries >= 3 and random.random() < NOOP_PROB and l_remaining - 1 >= 2:
        types[idx_pool.pop()] = "noop"
        l_remaining -= 1
    if n_entries >= 3 and idx_pool and random.random() < BLOCK_PROB and l_remaining - 1 >= 2:
        types[idx_pool.pop()] = "block"
        l_remaining -= 1
    return types


def _assign_l_orientations(types: list[str]) -> dict[int, str]:
    """Assign preferred stripe orientations to L-entries for observability support."""
    l_indices = [i for i, t in enumerate(types) if t == "L"]
    prefs: dict[int, str] = {}
    random.shuffle(l_indices)
    if len(l_indices) >= 2:
        prefs[l_indices[0]] = "horizontal"
        prefs[l_indices[1]] = "vertical"
        for idx in l_indices[2:]:
            prefs[idx] = random.choice(["horizontal", "vertical"])
    elif len(l_indices) == 1:
        prefs[l_indices[0]] = random.choice(["horizontal", "vertical"])
    return prefs


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

    types = _assign_marker_types(len(legend_rows))
    if "L" not in types:
        return None
    orientation_prefs = _assign_l_orientations(types)

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

    for i, (_, m, _) in enumerate(legend_rows):
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
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(
                H,
                W,
                m,
                LPAIR_RETRIES,
                can_place,
                do_place,
                preferred_orientation=orientation_prefs.get(i),
            ):
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
                preferred_orientation=random.choice(["horizontal", "vertical"]),
            )

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
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
