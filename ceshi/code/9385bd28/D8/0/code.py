# Variant of 9385bd28
# Dimension: D8 — 轨迹：标记像素的去留 (trajectory)
# Change: 保留标记 — 标记 L 形角和图例在输出中保持不变... → 擦除标记（仅保留填充结果）

import sys
import random
from collections import Counter

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import get_bg


# Design analysis for the D8 variant
# ----------------------------------
# spec:
#   source_task: none
#   variant_core:
#     - The input vocabulary stays the same: a bottom-left legend, L-corner marker pairs, optional 2x2 marker blocks,
#       and optional non-legend distractor marker groups.
#     - The output trajectory changes: every non-legend colored pixel is treated as a temporary construction cue.
#       After all fills are computed, those temporary pixels are erased back to the background color.
#     - The legend remains visible because it is the only non-temporary colored structure in the input.
#   concrete_pixel_behavior:
#     - L-pair entries still define an axis-aligned bbox and the opposite diagonal still determines the rectangle fill.
#       Because every cell of a rectangle lies on some shortest Manhattan path between opposite bbox corners, the
#       `_diamond_mask` remains the correct fill mask and still produces the full rectangle.
#     - The visible difference from the original task is that the 6 L-corner pixels no longer remain on top of the
#       rectangle; they become background-colored holes after the rectangle is painted.
#     - A 2x2 marker block is no longer recolored. It is a pure temporary clue, so it vanishes entirely in the output.
#   trace_example:
#     - Suppose marker color A has TL+BR L-corners. Original logic: fill the TR-BL rectangle in the legend's fill
#       color, but protect A's six pixels so they stay visible. Variant logic: fill the same rectangle, then erase
#       A's six input pixels to the background. The rectangle survives everywhere else, and the old anchor cells turn
#       into six background holes.
#   complexity_calibration:
#     - The main rule change is simple, but if the generator allowed samples with only noop/block entries or only
#       background-colored fills, many outputs would collapse into "almost nothing happened", making the trajectory
#       change hard to infer. To keep D8 observable, generation rejects such weak samples and keeps at least one
#       visible rectangle plus at least one erased marker group.
#     - Distractors remain allowed, but since every non-legend colored input pixel is temporary in this variant,
#       distractors are erased too. This keeps the visual language consistent: only fills and the legend survive.
#   hidden_assumptions_in_original_main:
#     - `out = g.copy()` was harmless before because markers were supposed to remain. In the variant it is still safe,
#       but only if a final erase pass removes every non-legend input-colored cell.
#     - The original `protect` mask included current and future markers so earlier fills would not overwrite visible
#       clues. That assumption is invalid now: clues are temporary, so only the legend needs protection.
#     - The original 2x2 branch recolored the block in place. That directly contradicts the new trajectory and must be
#       replaced with "do nothing now, erase later".
#   placement_legality:
#     - The generator still forbids marker overlap and out-of-bounds placement using an occupied mask.
#     - Legend scan integrity is preserved with the same forbidden placement zone around the bottom-left legend.
#     - Since the solver now erases every non-legend colored input pixel, the generator must avoid injecting any other
#       non-legend colored structures that are meant to survive. It does so by only creating marker-like temporary
#       objects outside the legend.
#   visual_observability:
#     - Human-solvers can now compare input/output and notice a consistent pattern: legend stays, rectangles appear,
#       and every scattered clue shape disappears into background holes.
#     - The generator rejects samples with no visible rectangle growth, because those would not display the fill rule.
#   visual_cleanliness:
#     - Keeping the legend fixed and erasing all other clue pixels produces a cleaner final grid than preserving a mix
#       of rectangles plus marker overlays.
#   color_inference:
#     - Fill colors are still dictated by the adjacent legend chip, so color changes remain locally signposted rather
#       than tied to invisible color-id ordering.
#   new_solver_assumptions_to_match_in_generator:
#     - Every non-legend non-background input pixel is temporary and may be erased.
#     - At least one legend entry should yield an actual L-pair rectangle, otherwise the variant effect degenerates.
#     - The legend remains two cells wide and readable bottom-to-top, exactly as in the original task.


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
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
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
    Use the legend to paint each marker-defined rectangle, then remove the clues.

    Input structure:
    - The canvas has one dominant background color plus a 2-column legend stacked near the bottom-left.
    - Outside the legend, each referenced marker color appears either as two opposite L-corners or as a solid 2x2 block.
    - Extra non-legend marker-shaped groups may also appear as distractors, but they still count as temporary clue pixels.

    Rule:
    - Read the legend from bottom to top and gather the non-legend pixels of each marker color.
    - For L-corner markers, detect which bbox diagonal carries the L anchors; fill the full opposite-diagonal rectangle
      with the legend's fill color while keeping the legend itself untouched.
    - A 2x2 block contributes no lasting painted region in this variant: it is only a temporary marker.
    - After all fills are done, erase every non-legend colored input pixel back to the background color.

    Key insight: The geometry is unchanged from the original task, but the scattered colored clues are now disposable:
    they identify what to fill, then disappear so only the legend and the fill result remain.
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
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Only the legend survives across all steps in this variant.
        protect = legend_mask.copy()

        # Solid 2x2 blocks are temporary clues and vanish in the final erase pass.
        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
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
        assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"
        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    # Every non-legend colored input pixel was only a temporary clue.
    temporary_mask = (g != bg) & (~legend_mask)
    out[temporary_mask] = bg
    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
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


def _build_legend(H, W, bg):
    """Create legend entries and the corresponding grid/mask arrays.

    Returns (grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors) or None on failure.
    """
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
        fill_choices = [c for c in range(10) if c != m]
        fill = random.choice(fill_choices)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity.

    Forbids non-bg placement in rows below legend_bot and in columns <= legend_c
    above the legend (except legend marker cells themselves).
    """
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _variant_signal_ok(grid: np.ndarray, out: np.ndarray, legend_mask: np.ndarray, bg: int) -> bool:
    """Reject weak samples where the D8 trajectory change is hard to observe."""
    temporary_mask = (grid != bg) & (~legend_mask)
    erased_count = int(np.count_nonzero(temporary_mask & (out == bg)))
    visible_fill_count = int(np.count_nonzero((grid == bg) & (~legend_mask) & (out != bg)))
    return erased_count >= 6 and visible_fill_count >= 4


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

    # Assign a marker type per legend entry, but keep at least one true L-pair so the fill rule stays visible.
    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        candidate = idx_pool.pop()
        if types.count("L") > 1:
            types[candidate] = "block"

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

    placed_l_pairs = 0

    # Place markers for each legend entry.
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
            if not _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place):
                return None
            placed_l_pairs += 1

    if placed_l_pairs == 0:
        return None

    # Optional distractor L-pair in a non-legend color. It is also temporary in this variant.
    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not _variant_signal_ok(grid, out, legend_mask, bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
