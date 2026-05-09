# Variant of 9385bd28
# Dimension: D11 — 干扰项 (meta)
# Change: 0-1 个干扰 L 对（使用非图例颜色的 L 形角标记），解题者需忽略非图例色的标记... → 多个干扰 L 对（2-3 个）

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for D11 (multiple distractor L-pairs)
# - Concrete behavior: the legend at bottom-left is still the only source of valid marker colors.
#   Real work happens exactly as before for legend colors, but the input now also contains 2-3
#   extra L-pairs in distinct colors that do not appear anywhere in the legend. They look valid
#   locally, so the solver must filter them out by checking the legend rather than by shape alone.
# - Behavior trace: in the original distribution, one orange fake L-pair could appear and was
#   harmless because `main()` only iterated over legend markers. In the new variant, there may be
#   orange, maroon, and grey fake pairs at once. The fill output should still be computed only from
#   legend-listed marker colors; all three fake pairs remain unchanged in the output.
# - Complexity calibration: the distractors reuse the same L-pair geometry as real markers, but
#   their count is capped at 2-3 and each uses a distinct reserved color. This makes the visual
#   change obvious without creating palette chaos or ambiguous matching rules.
# - Hidden assumptions review:
#   `_find_legend()` assumes the bottom-left scan is uninterrupted by stray pixels, so generator
#   placement still forbids any non-bg pixels in the legend scan corridor.
#   `marker_pts = argwhere(g == marker)` only ignores distractors if distractor colors never equal
#   a legend marker color, so generator reserves distractor colors outside the legend marker set.
#   Human-facing D11 also needs distractor colors to stay outside legend fill colors; otherwise a
#   fake pair could reuse a visible legend color and blur the "consult the legend" cue. Generator
#   therefore reserves distractor colors outside both legend columns.
#   The diagonal-anchor assertions still rely on every processed color corresponding to either one
#   valid L-pair or one 2x2 block, so distractors must never share colors with true entries.
# - Placement legality: every object still uses `can_place()` against occupied cells and the
#   forbidden legend corridor, so no overlap, out-of-bounds, or legend corruption is allowed.
#   Because 2-3 distractors must always exist, generation retries until all of them are placed.
# - Visual observability and cleanliness: several fake L-pairs in different colors are easy to spot
#   in 2-4 examples, and because their colors never appear in the legend, the ignore rule is
#   directly inferable by eye. The count is limited so the grid remains legible.


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
    Recolor each legend-listed marker configuration while ignoring extra fake corner pairs.

    Input structure:
    - The lower-left corner contains a vertical 2-column legend: left cell = marker color, right cell = fill color.
    - Away from the legend, each valid marker color appears either as one diagonal pair of L-corners or as one solid 2x2 block.
    - The grid may also include several additional L-pairs in colors that never appear in the legend; these are distractors.

    Rule:
    - Read the legend from bottom to top and process only those marker colors.
    - For a valid L-pair, inspect which bbox diagonal actually carries the two L-corner anchors, then fill the opposite diagonal's diamond with that legend row's fill color.
    - For a valid 2x2 block, recolor the block itself to the row's fill color.
    - Preserve legend cells, preserve later legend markers during earlier fills, and leave all non-legend distractor pairs untouched.

    Key insight: shape alone is insufficient in this variant because several extra L-pairs are decoys; the decisive filter is whether the color is listed in the legend.
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

    # Only legend-listed marker colors participate; distractor colors are ignored by construction.
    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Earlier fills must not erase legend pixels or marker pixels for later legend rows.
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

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

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_COUNT_RANGE = (2, 3)  # number of non-legend distractor L-pairs

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for an L-pair
DISTRACTOR_RETRIES = 90          # placement attempts for each distractor
GEN_RETRIES = 400                # outer generate_input retry budget

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
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
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


def _build_legend(H, W, bg, n_distractors):
    """Create legend entries plus reserved distractor colors that stay absent from the legend."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries + n_distractors:
        return None
    marker_colors = random.sample(pool, n_entries)
    remaining = [c for c in pool if c not in marker_colors]
    distractor_colors = random.sample(remaining, n_distractors)
    fill_pool = [c for c in range(10) if c not in distractor_colors]

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = random.choice(fill_pool)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, distractor_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
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
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)

    legend_result = _build_legend(H, W, bg, n_distractors)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, distractor_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

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

    random.shuffle(distractor_colors)
    for color in distractor_colors:
        if not _try_place_l_pair(H, W, color, DISTRACTOR_RETRIES, can_place, do_place):
            return None

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
