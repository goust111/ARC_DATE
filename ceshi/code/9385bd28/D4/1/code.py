# Variant of 9385bd28
# Dimension: D4 — 图例查找与颜色映射 (symbol_instructions)
# Change: 左下角垂直排列的颜色对图例：左列为标记色，右列为填充色；从下到上读取处理顺序... → 图例为水平排列

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


# Design analysis for the D4 horizontal-legend variant
# ----------------------------------------------------
# concretization:
#   - The legend is no longer a vertical stack of 2-cell rows. Instead it is a 2-row horizontal strip
#     near the bottom-left corner.
#   - Top legend row: marker colors. Bottom legend row: fill colors. Each column is one instruction pair.
#   - Processing order is fixed to left -> right. This keeps the variant visually learnable from a few
#     examples; mixing left->right and right->left across samples would make the rule harder to infer.
#   - Example trace:
#       original: bottom-left legend rows [(2,3), (6,7)] mean "process marker 2, then marker 6".
#       variant: top row [2,6], bottom row [3,7] means the same mapping, but now the first processed
#       entry is the left column, not the bottom cell of a vertical stack.
#
# complexity calibration:
#   - Only the legend orientation and readout direction are changed. Marker geometry, diamond filling,
#     protection of later entries, 2x2 blocks, and optional distractors stay the same.
#   - Fills are restricted to non-background colors. This is a necessary adaptation: in a horizontal
#     legend a background-valued fill would disappear from the bottom row and make the legend ambiguous.
#   - The legend still has 2-5 entries, so repeated evidence for the horizontal mapping remains visible.
#
# hidden assumptions audited from the original main():
#   - Original _find_legend assumed the bottommost non-bg row contains a 2-column legend row and that
#     walking upward along one fixed column recovers all instructions. That assumption is invalid here.
#   - Original legend_mask assumed one column pair repeated over many rows. Here the mask must cover one
#     row pair repeated over many columns.
#   - Original processing order came from bottom->top enumeration. Here every place that depended on the
#     order must consume left->right enumeration instead.
#   - Original generator could allow fill == bg because the marker cell still made each legend row visible.
#     In this variant that would create holes in the fill row, so the generator forbids it.
#
# placement legality:
#   - The generator reserves the two legend rows across the whole width so no marker can sit on those rows.
#   - It also reserves the columns left of the legend and the legend span above it, ensuring the bottommost
#     visible object is a clean horizontal legend and the solver's scan cannot latch onto stray pixels.
#   - Marker placements still use the original non-overlap checks and bbox validity checks.
#
# visual observability and cleanliness:
#   - The horizontal legend is visually distinct from the source task because the color-pair structure is
#     now seen as aligned columns instead of stacked rows.
#   - Keeping both legend rows fully visible and contiguous avoids ambiguous "floating" colors.
#   - Restricting to one fixed read direction avoids invisible dependence on color ids or hidden tie-breaks.
#
# color inferability:
#   - Each legend column presents the before/after mapping directly as a vertical pair, which is easy to
#     read from a few examples.
#   - Fill colors may still coincide with marker colors from other entries, preserving the original task's
#     overlap behavior, but they are never background so every mapping remains visible in the legend itself.
#
# new main() assumptions checked against generate_input():
#   - Assumption: the legend occupies exactly two adjacent rows and both rows have the same contiguous span.
#     Guarantee: _build_legend writes exactly that, and _build_forbidden prevents extra pixels on those rows.
#   - Assumption: read order is left->right. Guarantee: entries are stored and processed in increasing column order.
#   - Assumption: the legend is the bottommost non-background structure. Guarantee: rows below legend_bot are forbidden.


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[tuple[int, int, int]]]:
    """
    Legend is a horizontal 2-row strip near the bottom-left.
    Top row holds marker colors, bottom row holds fill colors.
    Returns (legend_top, legend_bot, entries) where entries are ordered left->right as (col, marker, fill).
    """
    h, w = g.shape
    for legend_bot in range(h - 1, 0, -1):
        bot_nz = np.flatnonzero(g[legend_bot, :] != bg)
        if bot_nz.size == 0:
            continue
        legend_top = legend_bot - 1
        top_nz = np.flatnonzero(g[legend_top, :] != bg)
        if top_nz.size == 0:
            continue

        start_c = int(bot_nz[0])
        if int(top_nz[0]) != start_c:
            continue

        entries: list[tuple[int, int, int]] = []
        c = start_c
        while c < w and int(g[legend_top, c]) != bg and int(g[legend_bot, c]) != bg:
            entries.append((c, int(g[legend_top, c]), int(g[legend_bot, c])))
            c += 1

        if not entries:
            continue

        span = set(range(start_c, start_c + len(entries)))
        if set(map(int, top_nz.tolist())) != span or set(map(int, bot_nz.tolist())) != span:
            continue

        return legend_top, legend_bot, entries

    raise AssertionError("legend must exist as a 2-row horizontal strip near the bottom-left")


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles between diagonal marker pairs using a horizontal color legend.

    Input structure:
    - A background grid contains a 2-row legend near the bottom-left: marker colors on the upper row,
      matching fill colors directly beneath them on the lower row.
    - Elsewhere, each marker color appears either as two opposite L-shaped bbox corners or as a solid 2x2 block.
    - The legend columns are the instruction sequence and are read from left to right.

    Rule:
    - For each legend column, collect all non-legend pixels of that marker color and process entries left to right.
    - If the marker forms a solid 2x2 block, recolor that block to the legend's fill color.
    - Otherwise, detect which bbox diagonal contains the L-anchors and fill the opposite diagonal's diamond region.
    - Legend cells and marker pixels belonging to later legend entries stay protected from earlier fills.

    Key insight: The geometry rule is unchanged; only the instruction carrier moved from a vertical row-stack
    to a horizontal two-row strip, so the solver must parse columns of the legend instead of rows.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_top, legend_bot, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for c, _, _ in entries:
        if 0 <= legend_top < h and 0 <= c < w:
            legend_mask[legend_top, c] = True
        if 0 <= legend_bot < h and 0 <= c < w:
            legend_mask[legend_bot, c] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    # Process legend entries left -> right; earlier entries must not destroy later marker pixels.
    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Protect legend cells and all marker pixels for entries that appear later (to the right) in the legend.
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        # Keep current marker pixels visible, except for a solid 2x2 marker block which is recolored directly.
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


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_START_COL_RANGE = (0, 2)  # left edge of horizontal legend
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries / columns

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


def _try_place_l_pair(H: int, W: int, color: int, retries: int,
                      can_place, do_place) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts.

    Returns True on success, False if all attempts failed.
    """
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


def _build_legend(H: int, W: int, bg: int):
    """Create a 2-row horizontal legend and its mask.

    Returns (grid, legend_entries, legend_mask, legend_top, legend_bot, legend_c0, legend_c1, marker_colors)
    or None on failure.
    """
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    legend_c0_hi = min(LEGEND_START_COL_RANGE[1], W - n_entries)
    if legend_c0_hi < LEGEND_START_COL_RANGE[0]:
        return None
    legend_c0 = random.randint(LEGEND_START_COL_RANGE[0], legend_c0_hi)
    legend_c1 = legend_c0 + n_entries - 1
    legend_bot = random.choice([H - 1, H - 2])
    legend_top = legend_bot - 1
    if legend_top < 0:
        return None

    marker_pool = [c for c in range(10) if c != bg]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

    fill_pool = [c for c in range(10) if c != bg]
    grid = np.full((H, W), bg, dtype=int)
    legend_entries: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        c = legend_c0 + i
        marker = marker_colors[i]
        fill = random.choice(fill_pool)
        legend_entries.append((c, marker, fill))
        grid[legend_top, c] = marker
        grid[legend_bot, c] = fill
        legend_mask[legend_top, c] = True
        legend_mask[legend_bot, c] = True

    return grid, legend_entries, legend_mask, legend_top, legend_bot, legend_c0, legend_c1, marker_colors


def _build_forbidden(H: int, W: int, legend_c0: int, legend_c1: int, legend_top: int, legend_bot: int):
    """Build the forbidden-placement mask that protects horizontal legend parsing.

    Forbids non-bg placement on legend rows, below the legend, to the left of the legend,
    and directly above the legend span so the bottom-left horizontal strip stays unambiguous.
    """
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[legend_top : legend_bot + 1, :] = True
    if legend_c0 > 0:
        forbidden[: legend_bot + 1, :legend_c0] = True
    if legend_top > 0:
        forbidden[:legend_top, legend_c0 : legend_c1 + 1] = True
    forbidden[legend_top, legend_c0 : legend_c1 + 1] = False
    forbidden[legend_bot, legend_c0 : legend_c1 + 1] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_entries, legend_mask, legend_top, legend_bot, legend_c0, legend_c1, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c0, legend_c1, legend_top, legend_bot)
    occupied = legend_mask.copy()

    # Assign a marker type per legend entry.
    n_entries = len(legend_entries)
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
    for i, (_, marker, _) in enumerate(legend_entries):
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

    # Optional distractor L-pair in a non-legend color.
    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail),
                              DISTRACTOR_RETRIES, can_place, do_place)

    # Ensure bg remains the most common color (so get_bg picks it).
    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    # Oracle check: solver must run without errors, and the sample must visibly transform.
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
