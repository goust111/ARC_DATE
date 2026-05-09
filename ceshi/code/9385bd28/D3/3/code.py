# Variant of 9385bd28
# Dimension: D3 — 目标对象：填充区域的形状 (dst)
# Change: 标记对角线的对边对角线上所有最短路径像素（菱形/矩形填充区域）... → 填充区域为标记之间的对角带状
#
# Design analysis
# requested_reference:
#   task: 16de56c4
#   solver_found_in_workspace: false
#   note: The prompt requires reading the source-task solver first when available.
#     The workspace only contains `16de56c4.json`, so the solver-reading step is
#     skipped exactly under the "不存在就跳过" clause. The variant still borrows
#     the high-level `line_fill` idea: replace area fill by a thin drawn path.
# variant_spec:
#   preserved_structure:
#     - Background grid with a bottom-left vertical 2-column legend.
#     - Legend rows are read bottom-to-top as `(marker_color, fill_color)`.
#     - Each active marker color still appears as either two opposite L-corners
#       or a solid 2x2 block.
#     - The diagonal carrying the two L-corners still determines that the fill
#       belongs on the opposite diagonal of the bounding box.
#     - Later legend entries still protect their marker pixels from earlier fills.
#   D3_change:
#     old_dst: all cells of the opposite-corner rectangle / monotone shortest-path region
#     new_dst: only the diagonal band between the two opposite fill corners
#     concrete_shape:
#       - For the chosen opposite corners `a` and `b`, draw one digital line
#         segment from the center of `a` to the center of `b`.
#       - Fill every grid cell whose unit square is touched by that segment.
#       - This creates a 1-pixel diagonal when slope is neat, and a 2-pixel
#         staircase at corner-touching steps, matching the requested "1-2 px band".
#   behavior_trace:
#     example_bbox:
#       anchors: TL + BR markers on bbox rows 2..6, cols 4..9
#       fill_diagonal: TR=(2,9) to BL=(6,4)
#     original:
#       - `_diamond_mask` marks the whole 5x6 rectangle because every cell lies on
#         some monotone shortest path from TR to BL.
#       - The output becomes a solid block, except protected markers.
#     variant:
#       - `_diag_band_mask` marks only the supercover diagonal strip from TR to BL.
#       - The output shows a thin descending staircase, leaving most of the bbox blank.
#     observable_difference:
#       - Same legend, same markers, same chosen diagonal.
#       - Only the destination geometry changes, so the rule is visually attributable
#         to D3 instead of a different parsing or recoloring mechanism.
#   complexity_calibration:
#     - Keep 2-5 legend entries so the legend mapping remains learnable from few examples.
#     - Keep optional noop and 2x2-block cases, but force at least two genuine L-pairs
#       so the diagonal-band behavior is visible in every generated grid.
#     - Fill colors are chosen away from the background and away from legend marker
#       colors to keep the new line shape visible.
#     - At least one L-pair must have a bbox where `bbox_area - band_area >= 3`,
#       preventing degeneration into a shape that looks too close to the old rectangle fill.
#   implicit_assumption_audit:
#     - `_find_legend`: still valid; D3 does not alter legend geometry.
#     - `marker_pts = argwhere((g == marker) & ~legend_mask)`: still valid because
#       marker recognition remains color-based on the input only.
#     - Bounding-box extraction and corner-anchor tests still rely on markers occupying
#       opposite bbox corners; generator keeps this invariant unchanged.
#     - The old `_diamond_mask` assumed "destination is every monotone shortest-path cell".
#       That assumption no longer holds and is replaced by an explicit line-segment test.
#     - `protect` logic remains necessary: if a line crosses later markers, they must survive.
#     - The 2x2-block recolor branch remains necessary because those entries do not define
#       a meaningful diagonal path; generator still allows them as a separate subcase.
#   placement_legality:
#     - Inputs still forbid overlap with legend cells and the scan-critical left margin.
#     - Marker cells themselves never overlap.
#     - Output line overlaps are allowed; later legend rows overwrite earlier fills just as before.
#     - Generator validates the produced input by running `main()`, and rejects any case
#       where background would stop being the dominant color or where the solver asserts.
#   observability_and_cleanliness:
#     - Because the line occupies far fewer cells than the old rectangle, the new rule is
#       visible from 2-3 examples without relying on color IDs or hidden ordering cues.
#     - Fill colors avoid bg/marker colors so viewers can isolate the drawn bands quickly.
#     - Distractors keep using non-legend colors, so they remain easy to ignore.

import numpy as np

from arc_utils import get_bg
import random
from collections import Counter


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


def _segment_hits_cell(a: tuple[int, int], b: tuple[int, int], r: int, c: int) -> bool:
    """True iff the center-to-center segment from a to b intersects cell (r,c)'s unit square."""
    y0, x0 = float(a[0]), float(a[1])
    y1, x1 = float(b[0]), float(b[1])
    xmin, xmax = c - 0.5, c + 0.5
    ymin, ymax = r - 0.5, r + 0.5
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    eps = 1e-9
    for p, q in (
        (-dx, x0 - xmin),
        (dx, xmax - x0),
        (-dy, y0 - ymin),
        (dy, ymax - y0),
    ):
        if abs(p) < eps:
            if q < -eps:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 - t1 > eps:
            return False
    return True


def _diag_band_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int,
                    a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    """Mask for the 1-2 pixel digital diagonal band between opposite corners a and b."""
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _segment_hits_cell(a, b, r, c):
                m[r, c] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal bands between opposite marker pairs using a color legend.

    Input structure:
    - The background contains a bottom-left vertical legend, where each row stores a marker color
      in one column and its output color in the next column.
    - Outside the legend, each active marker color appears either as two diagonal L-shaped corners
      or as a solid 2x2 block.
    - The paired L-corners still define an axis-aligned bounding box and identify which diagonal is
      the marker diagonal versus the fill diagonal.

    Rule:
    - Read legend rows from bottom to top and find all non-legend pixels of that marker color.
    - For an L-pair, detect which bbox diagonal carries the L-anchors; the opposite diagonal is the
      one to draw on.
    - Instead of coloring the whole rectangle, draw only the thin diagonal band along that opposite
      corner-to-corner line using the legend's fill color.
    - Solid 2x2 marker blocks are still a special case and are simply recolored to the fill color.
    - Earlier rows must preserve legend cells and all marker pixels belonging to later rows.

    Key insight: The markers still tell you which *diagonal* is relevant, but the destination is no
    longer a filled area; it is the supercover line segment on the opposite diagonal.
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

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1:])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue
        else:
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

        band = _diag_band_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[band & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20
MIN_ACTIVE_LPAIRS = 2

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8
MIN_BAND_SAVINGS = 3

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
DISTRACTOR_RETRIES = 60
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


def _try_place_l_pair(H: int, W: int, color: int, retries: int,
                      can_place, do_place) -> dict[str, int | str] | None:
    """Try to place an L-pair of `color`; return bbox metadata on success."""
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
            return {
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmin + bh - 1,
                "cmax": cmin + bw - 1,
                "diag": diag,
            }
    return None


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
    fill_pool = [c for c in range(10) if c != bg and c not in marker_colors]
    if not fill_pool:
        return None

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


def _fill_corners_from_anchor_diag(rmin: int, rmax: int, cmin: int, cmax: int, anchor_diag: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if anchor_diag == "TLBR":
        return (rmin, cmax), (rmax, cmin)
    return (rmin, cmin), (rmax, cmax)


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
    if random.random() < NOOP_PROB and len(idx_pool) > MIN_ACTIVE_LPAIRS:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and len(idx_pool) > MIN_ACTIVE_LPAIRS:
        types[idx_pool.pop()] = "block"
    while sum(t == "L" for t in types) < MIN_ACTIVE_LPAIRS:
        for idx, t in enumerate(types):
            if t != "L":
                types[idx] = "L"
                break

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

    l_infos: list[dict[str, int | str]] = []
    for i, (_, m, fill) in enumerate(legend_rows):
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
            info = _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place)
            if info is None:
                return None
            info["fill"] = int(fill)
            l_infos.append(info)

    if len(l_infos) < MIN_ACTIVE_LPAIRS:
        return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    strong_band_count = 0
    for info in l_infos:
        rmin = int(info["rmin"])
        cmin = int(info["cmin"])
        rmax = int(info["rmax"])
        cmax = int(info["cmax"])
        a, b = _fill_corners_from_anchor_diag(rmin, rmax, cmin, cmax, str(info["diag"]))
        band = _diag_band_mask(H, W, rmin, rmax, cmin, cmax, a, b)
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        if bbox_area - int(band.sum()) >= MIN_BAND_SAVINGS:
            strong_band_count += 1
    if strong_band_count == 0:
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
