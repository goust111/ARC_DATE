# Combo variant of 9385bd28 (index 2)
# combo_spec: D3/1+D5/2+D8/1
# Name: 棋盘格边框填充+标记变色
#
# Design analysis:
# a) Concrete behavior and trace
#    - Input objects stay the same as the base task: a bottom-left vertical 2-column legend plus off-legend markers.
#    - Each active legend color still appears either as one diagonal L-pair or as one solid 2x2 block.
#    - For an L-pair with bbox rows 2..6 and cols 4..8:
#      old task -> detect the anchored diagonal, then fill the opposite-diagonal diamond while keeping the L pixels.
#      combo    -> detect the same anchored diagonal, but use the whole bbox perimeter as the target;
#                  perimeter cells alternate between fill color and bg by grid parity;
#                  the six original L pixels are then forced to the fill color, so they do not alternate away.
#      visible difference -> the output is a hollow rectangle with a checkerboard edge, and the original marker color
#      disappears from both the scattered markers and that legend row's marker cell.
#    - For a 2x2 block there is no rectangle pair to outline, so it remains the base special case: recolor the block solid.
#
# b) Complexity calibration
#    - The task still requires legend lookup, bottom-to-top overwrite order, diagonal disambiguation, optional no-op rows,
#      optional 2x2 blocks, and optional distractors.
#    - To keep the combo visually inferable, the generator forces at least one genuine L-pair with bbox height >= 3,
#      width >= 3, and fill color different from both bg and marker color. This guarantees:
#      1. a hollow interior is visible,
#      2. the checkerboard border differs from a solid outline,
#      3. marker recoloring is observable.
#
# c) Hidden assumption review
#    - `_find_legend` still assumes the legend is the lowest-leftmost non-bg structure and 2 columns wide; unchanged.
#    - `_l_anchor_present` still assumes each marker corner is a 3-pixel L; unchanged.
#    - The base fill mask (`_diamond_mask`) is no longer the right target geometry; it must be replaced by a bbox border mask.
#    - The base/current D3 behavior protects current marker pixels so they remain visible; that assumption is false here.
#      Current marker pixels must remain paintable and then be forced to the fill color.
#    - The D5 checkerboard logic cannot simply paint the full target mask by parity, because current marker pixels would
#      sometimes become bg. The combo therefore splits painting into:
#      1. non-marker border cells -> checkerboard fill/bg
#      2. current marker pixels   -> always fill
#    - Legend parsing must keep reading from the original grid `g`, not from `out`, because current legend marker cells
#      are recolored during processing.
#
# d) Placement legality
#    - Input markers may not overlap, may not corrupt the legend scan corridor, and must stay in-bounds.
#    - Valid L-pair bbox sizes still exclude tie-causing sizes (2,2), (2,3), and (3,2).
#    - The generator places one showcase L-pair first and keeps later objects' bboxes away from it, guaranteeing at least
#      one clean example of the fused behavior instead of letting every border be overwritten by later rectangles.
#
# e) Visual observability
#    - Fill colors are drawn from colors different from bg and the row's marker color.
#    - The showcase L-pair must survive in the final output with:
#      1. at least one border cell painted fill,
#      2. at least one border cell painted bg,
#      3. at least one original marker pixel recolored to fill.
#    - Those three cues are enough for a human to infer "legend picks the color, edge alternates with bg, markers get absorbed."
#
# f) Visual cleanliness
#    - Only one isolated showcase rectangle is guaranteed; other objects may still appear, including blocks and distractors,
#      so the task remains nontrivial without making the whole board cluttered.
#    - Using bg as the second checkerboard color keeps the pattern readable and avoids introducing a second arbitrary palette map.
#
# g) Color-change inferability
#    - The recolor target is always given locally by the legend row: left cell is the marker color, right cell is the fill color.
#    - Because the same fill color is used on both the edge pattern and the absorbed markers, the solver does not rely on any
#      hidden numeric property of colors.
#
# New solver assumptions that the generator now guarantees:
# - At least one legend entry is an L-pair rather than only block/no-op entries.
# - At least one L-pair has a non-degenerate hollow bbox and remains visibly checkerboarded after all overwrites.

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

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    dr1, dc1 = arm1
    dr2, dc2 = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _rect_border_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Mask for the perimeter of the axis-aligned rectangle [r0..r1] x [c0..c1]."""
    m = np.zeros((h, w), dtype=bool)
    m[r0, c0 : c1 + 1] = True
    m[r1, c0 : c1 + 1] = True
    m[r0 : r1 + 1, c0] = True
    m[r0 : r1 + 1, c1] = True
    return m


def _paint_checkerboard(
    out: np.ndarray,
    mask: np.ndarray,
    protect: np.ndarray,
    fill: int,
    bg: int,
    even_parity: np.ndarray,
    force_fill: np.ndarray | None = None,
) -> None:
    """Paint `mask` with a fill/bg checkerboard, optionally forcing some paintable cells to fill."""
    paintable = mask & (~protect)
    if not np.any(paintable):
        return

    forced = np.zeros_like(mask, dtype=bool) if force_fill is None else (force_fill & paintable)
    alt = paintable & (~forced)

    out[alt & even_parity] = int(fill)
    out[alt & (~even_parity)] = int(bg)
    out[forced] = int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw checkerboard rectangle borders from diagonal markers and absorb the markers into the result.

    Input structure:
    - The grid has one dominant background color and a bottom-left vertical legend with two cells per row.
    - Each legend row maps a marker color to one fill color, and rows are interpreted bottom to top.
    - Away from the legend, each active marker color appears either as two opposite 3-pixel L corners or as a solid 2x2 block.

    Rule:
    - Read the legend bottom-to-top and find all non-legend pixels of that marker color in the original grid.
    - A 2x2 block is still the no-rectangle special case: recolor all four block pixels to the legend fill color.
    - For an L-pair, detect which bbox diagonal contains the L anchors; the bbox itself is the rectangle to draw.
    - Paint only the rectangle border, using a fill/bg checkerboard on non-marker border cells, while turning the current
      marker pixels themselves into the fill color.
    - Protect legend cells and marker pixels belonging to later legend entries from earlier rows, but recolor the current
      legend marker cell to its row's fill color after the row has been read.

    Key insight:
    - The marker geometry still tells you which color row matches which rectangle, and the L diagonal still has to be
      recognized correctly; the combo change is that the output is now a hollow checkerboard outline whose original
      marker pixels are absorbed into the fill color instead of staying visible.
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

    rr, cc = np.indices((h, w))
    even_parity = ((rr + cc) % 2) == 0

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (legend_r, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        out[legend_r, legend_c] = eff_fill

        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        marker_mask = np.zeros_like(g, dtype=bool)
        for r, c in marker_pts:
            marker_mask[r, c] = True

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            out[marker_mask & (~protect)] = eff_fill
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

        border = _rect_border_mask(h, w, rmin, rmax, cmin, cmax)
        _paint_checkerboard(out, border, protect, eff_fill, bg, even_parity, force_fill=marker_mask)

    return out


# --------------- generator (C1) ---------------

# -- CFG: canvas & grid --------------------------------------------------------
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# -- CFG: legend ---------------------------------------------------------------
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# -- CFG: marker types ---------------------------------------------------------
NOOP_PROB = 0.3
BLOCK_PROB = 0.25

# -- CFG: L-pair bbox shape ----------------------------------------------------
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.5
VISIBLE_COMBO_MIN = 1

# -- CFG: distractor -----------------------------------------------------------
DISTRACTOR_PROB = 0.25

# -- CFG: retry budgets --------------------------------------------------------
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


def _pick_bbox_size(H: int, W: int, require_hollow: bool = False) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    if require_hollow:
        h_hi = min(BBOX_H_MAX, H)
        w_hi = min(BBOX_W_MAX, W)
        if h_hi < 3 or w_hi < 3:
            return None
        return random.randint(3, h_hi), random.randint(3, w_hi)

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


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True if two closed axis-aligned bboxes intersect."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _intersects_any(
    bbox: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]] | None,
) -> bool:
    """True if `bbox` intersects any bbox in `others`."""
    if not others:
        return False
    return any(_rectangles_intersect(bbox, other) for other in others)


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    require_hollow: bool = False,
    avoid_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int] | None:
    """Try to place an L-pair and return its bbox on success."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W, require_hollow=require_hollow)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        bbox = (rmin, rmin + bh - 1, cmin, cmin + bw - 1)
        if _intersects_any(bbox, avoid_bboxes):
            continue
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(bbox[0], bbox[2], bbox[1], bbox[3], diag)
        if can_place(cells):
            do_place(cells, color)
            return bbox
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

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill_pool = [c for c in range(10) if c != bg and c != marker]
        fill = random.choice(fill_pool)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _combo_metrics(grid: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """Measure whether the fused behavior is visibly present in the final output."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    visible_combo_pairs = 0
    recolored_marker_cells = 0
    recolored_legend_markers = 0

    for legend_r, marker, fill in entries:
        if out[legend_r, legend_c] == fill:
            recolored_legend_markers += 1

        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        marker_mask = np.zeros_like(g, dtype=bool)
        for r, c in marker_pts:
            marker_mask[r, c] = True

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            recolored_marker_cells += int(np.sum(out[marker_mask] == fill))
            continue

        if rmin == rmax or cmin == cmax:
            continue

        border = _rect_border_mask(h, w, rmin, rmax, cmin, cmax)
        non_marker_border = border & (~marker_mask)
        border_fill = int(np.sum(out[non_marker_border] == fill))
        border_bg = int(np.sum(out[non_marker_border] == bg))
        marker_fill = int(np.sum(out[marker_mask] == fill))
        recolored_marker_cells += marker_fill

        if (
            (rmax - rmin) >= 2
            and (cmax - cmin) >= 2
            and border_fill > 0
            and border_bg > 0
            and marker_fill > 0
        ):
            visible_combo_pairs += 1

    return {
        "visible_combo_pairs": visible_combo_pairs,
        "recolored_marker_cells": recolored_marker_cells,
        "recolored_legend_markers": recolored_legend_markers,
    }


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
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    if "L" not in types:
        candidates = [i for i, t in enumerate(types) if t == "block"]
        if candidates:
            types[random.choice(candidates)] = "L"
        else:
            types[random.randrange(n_entries)] = "L"

    showcase_choices = [i for i, t in enumerate(types) if t == "L"]
    if not showcase_choices:
        return None
    showcase_idx = random.choice(showcase_choices)

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

    showcase_bbox = _try_place_l_pair(
        H,
        W,
        legend_rows[showcase_idx][1],
        LPAIR_RETRIES,
        can_place,
        do_place,
        require_hollow=True,
        avoid_bboxes=None,
    )
    if showcase_bbox is None:
        return None
    placed_bboxes = [showcase_bbox]

    for i, (_, marker, _) in enumerate(legend_rows):
        if i == showcase_idx:
            continue

        t = types[i]
        if t == "noop":
            continue

        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                bbox = (r0, r0 + 1, c0, c0 + 1)
                if _intersects_any(bbox, placed_bboxes):
                    continue
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed_bboxes.append(bbox)
                    placed = True
                    break
            if not placed:
                return None
            continue

        bbox = _try_place_l_pair(
            H,
            W,
            marker,
            LPAIR_RETRIES,
            can_place,
            do_place,
            require_hollow=False,
            avoid_bboxes=placed_bboxes,
        )
        if bbox is None:
            return None
        placed_bboxes.append(bbox)

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
                require_hollow=False,
                avoid_bboxes=placed_bboxes,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    metrics = _combo_metrics(grid, out)
    if metrics["visible_combo_pairs"] < VISIBLE_COMBO_MIN:
        return None
    if metrics["recolored_legend_markers"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
