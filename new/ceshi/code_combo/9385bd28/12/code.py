# Combo variant of 9385bd28 (index 12)
# combo_spec: D4/1+D10/1+D15/1
# Name: 水平图例+大量条目+大网格

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the fused combo variant
# -------------------------------------------
# concretization_and_behavior_trace:
#   - The source task used a vertical 2-column legend at bottom-left and read rows bottom -> top.
#   - This combo rotates the instruction carrier into a horizontal 2-row strip near the bottom-left:
#       top row = marker colors, bottom row = fill colors, each column = one mapping.
#   - Processing order becomes left -> right, and there are now 6-8 legend columns instead of 2-5 rows.
#   - The canvas is enlarged to 20-25 in both dimensions so the legend can be wide and the marker groups
#     can stay separated. Example trace:
#       original: legend rows [(marker 2 -> fill 3), (marker 6 -> fill 7)] at the lower-left.
#       fused:    top row [2, 6, 4, 8, 1, 9], bottom row [3, 7, 5, 2, 6, 4], read left -> right.
#       The local geometry rule is unchanged: each non-legend marker color either forms one L-pair bbox
#       whose opposite diagonal diamond gets filled, or forms one 2x2 block that gets recolored.
#
# complexity_calibration:
#   - The visual change must be obvious, so every sample uses a wide 6-8-column legend and a 20-25 grid.
#   - To keep the legend unambiguous, fill colors are never background-valued; otherwise the lower legend
#     row would develop holes and stop reading like a clean horizontal strip.
#   - Most entries stay active: noop rows are allowed only when there are at least 7 legend columns, which
#     preserves the "many mappings" effect instead of shrinking back toward the original difficulty.
#   - Output-side clutter is reduced by forcing marker bboxes to be spatially separated, which matches the
#     large-grid sparse-layout requirement and keeps the fused variant readable by eye.
#
# implicit_assumption_audit:
#   - Original _find_legend assumed the bottommost non-bg structure was a vertical stack in one column pair.
#     That breaks here; the solver now searches for two adjacent rows sharing one contiguous non-bg span.
#   - Original legend_mask protected repeated column cells across many rows. Here it must protect repeated
#     row cells across many columns.
#   - Original processing order came from bottom -> top enumeration. Every "future marker" protection step
#     must now follow the left -> right legend column order.
#   - main() still assumes each legend marker color corresponds to exactly one non-legend object or no object.
#     The generator preserves that by placing at most one marker group per legend color.
#   - main() still assumes L-pair bboxes avoid diagonal-score ties. The generator keeps rejecting (2,2),
#     (2,3), and (3,2) style ambiguous boxes.
#
# placement_legality_and_cleanliness:
#   - Non-legend pixels are forbidden on the two legend rows, below the legend, to the left of the legend,
#     and directly above the legend span. This guarantees that the first bottom-left visible structure is the
#     intended horizontal legend and that its span is contiguous in both rows.
#   - Marker pixels cannot overlap each other or the legend, and full marker bboxes are reserved so different
#     diamonds remain disjoint. This makes the large canvas look sparse rather than tangled.
#   - Distractors use colors outside the legend palette and obey the same placement legality, so they do not
#     corrupt legend parsing or accidentally merge with a real marker group.
#
# observability_and_inference:
#   - From 2-4 samples, a human can see a stable pattern: a wide horizontal strip at the lower-left always
#     encodes color mappings, and many scattered marker pairs elsewhere respond to those mappings.
#   - The rule never depends on hidden properties such as numeric color order. Read direction is fixed to
#     left -> right, and the horizontal strip itself provides the only ordering cue needed.
#   - The large grid is not cosmetic: it is necessary to display 6-8 mapping columns and multiple separated
#     marker bboxes at once, making the fused change visibly different from both the source task and the
#     single-axis horizontal-only variant.
#
# new_main_assumptions_checked_against_generate_input:
#   - Assumption: legend occupies exactly two adjacent rows with the same contiguous non-bg column span.
#     Guarantee: _build_legend writes that exact structure and _build_forbidden blocks extra pixels there.
#   - Assumption: read order is left -> right. Guarantee: legend entries are stored in increasing column order.
#   - Assumption: later marker colors are still present in the input when earlier fills execute.
#     Guarantee: the solver reads marker pixels from the original input grid, and generator colors are disjoint.
#   - Assumption: each marker color forms either one L-pair or one 2x2 block, or is a noop.
#     Guarantee: placement assigns exactly one type per legend entry and never duplicates a marker color.


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
    Fill many sparse marker rectangles using a wide horizontal legend.

    Input structure:
    - A large background grid contains a 2-row legend near the bottom-left.
    - The upper legend row lists 6-8 marker colors, and the lower row lists the matching fill colors
      directly underneath them, so each legend column is one instruction read left to right.
    - Away from the legend, each marker color appears either as one diagonal pair of L-corners or as one
      solid 2x2 block.

    Rule:
    - Parse the horizontal legend columns from left to right and collect each marker color's non-legend pixels.
    - If a marker color forms a solid 2x2 block, recolor that block to the legend's fill color.
    - Otherwise, find which bbox diagonal carries the L-anchors and fill the opposite diagonal's diamond region.
    - Keep the legend intact and protect marker pixels belonging to later legend entries while earlier entries fill.

    Key insight:
    - The only structural change from the source task is the instruction carrier: the color mapping now comes
      from columns of a horizontal two-row legend, while the L-corner geometry still tells you to fill the
      shortest-path diamond on the opposite diagonal of the marker bbox.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_top, legend_bot, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for c, _, _ in entries:
        legend_mask[legend_top, c] = True
        legend_mask[legend_bot, c] = True

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


# --------------- generator (combo) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (20, 25)
GRID_W_RANGE = (20, 25)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_START_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (6, 8)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.24
BLOCK_PROB = 0.30

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 10
BBOX_W_MAX = 10
BBOX_SHAPE_WEIGHTS = [0.86, 0.08, 0.06]
TLBR_PROB = 0.8
BBOX_PAD = 0

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.10

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 180
LPAIR_RETRIES = 450
DISTRACTOR_RETRIES = 160
GEN_RETRIES = 700

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
    """Pick a moderate valid L-pair bbox size that fits in a large sparse layout."""
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


def _choose_fill_colors(bg: int, marker_colors: list[int]) -> list[int]:
    """Choose visible fills; background is excluded to keep the lower legend row contiguous."""
    fills: list[int] = []
    used_pairs: set[tuple[int, int]] = set()
    for marker in marker_colors:
        preferred = [c for c in range(10) if c != bg and c != marker]
        fallback = [c for c in range(10) if c != bg]
        pool = preferred if preferred else fallback
        fill = random.choice(pool)
        if (marker, fill) in used_pairs and len(pool) > 1:
            alt_pool = [c for c in pool if (marker, c) not in used_pairs]
            if alt_pool:
                fill = random.choice(alt_pool)
        used_pairs.add((marker, fill))
        fills.append(fill)
    return fills


def _build_legend(H: int, W: int, bg: int):
    """Create a 2-row horizontal legend and its mask."""
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
    fill_colors = _choose_fill_colors(bg, marker_colors)

    grid = np.full((H, W), bg, dtype=int)
    legend_entries: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i, (marker, fill) in enumerate(zip(marker_colors, fill_colors)):
        c = legend_c0 + i
        legend_entries.append((c, marker, fill))
        grid[legend_top, c] = marker
        grid[legend_bot, c] = fill
        legend_mask[legend_top, c] = True
        legend_mask[legend_bot, c] = True

    return grid, legend_entries, legend_mask, legend_top, legend_bot, legend_c0, legend_c1, marker_colors


def _build_forbidden(H: int, W: int, legend_c0: int, legend_c1: int, legend_top: int, legend_bot: int) -> np.ndarray:
    """Protect the horizontal legend from stray pixels that would break parsing."""
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


def _rect_is_clear(rect_used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> bool:
    """True if the rectangle, optionally expanded by pad, is unused."""
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rect_used.shape[0] - 1, r1 + pad)
    cc1 = min(rect_used.shape[1] - 1, c1 + pad)
    return not rect_used[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_rect(rect_used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    """Reserve the rectangle, optionally expanded by pad."""
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rect_used.shape[0] - 1, r1 + pad)
    cc1 = min(rect_used.shape[1] - 1, c1 + pad)
    rect_used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    rect_is_clear,
    reserve_rect,
) -> bool:
    """Try to place a sparse L-pair of `color` within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        if not rect_is_clear(rmin, cmin, rmax, cmax):
            continue

        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if can_place(cells):
            do_place(cells, color)
            reserve_rect(rmin, cmin, rmax, cmax)
            return True
    return False


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
    rect_used = forbidden.copy()

    n_entries = len(legend_entries)
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

    def rect_is_clear(r0: int, c0: int, r1: int, c1: int) -> bool:
        return _rect_is_clear(rect_used, r0, c0, r1, c1, BBOX_PAD)

    def reserve_rect(r0: int, c0: int, r1: int, c1: int) -> None:
        _mark_rect(rect_used, r0, c0, r1, c1, BBOX_PAD)

    for i, (_, marker, _) in enumerate(legend_entries):
        marker_type = types[i]
        if marker_type == "noop":
            continue
        if marker_type == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                r1 = r0 + 1
                c1 = c0 + 1
                if not rect_is_clear(r0, c0, r1, c1):
                    continue
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    reserve_rect(r0, c0, r1, c1)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(
                H,
                W,
                marker,
                LPAIR_RETRIES,
                can_place,
                do_place,
                rect_is_clear,
                reserve_rect,
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
                rect_is_clear,
                reserve_rect,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
