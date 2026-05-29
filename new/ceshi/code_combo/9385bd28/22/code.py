# Combo variant of 9385bd28 (index 22)
# combo_spec: D3/0+D5/1+D9/0
# Name: 条纹填充+固定对角线+完整矩形

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for combo D3/0 + D5/1 + D9/0
# ------------------------------------------------
# a) concretization_and_trace:
#    The visible input vocabulary stays close to the base task: a bottom-left
#    vertical legend maps marker colors to fill colors, and each active marker is
#    either one L-pair or one 2x2 block. The fused change applies only to L-pairs.
#    Example: suppose marker color 2 maps to fill color 7 and forms TL+BR anchors
#    on a 4x5 bbox. In the source task, the solver would infer the anchored
#    diagonal, switch to the opposite diagonal, and paint only that diamond while
#    preserving the six marker pixels. In this combo, the diagonal choice branch is
#    gone because every L-pair is always TL-BR; the output target is the whole 4x5
#    rectangle; and that rectangle is not solid but striped by TR-BL anti-diagonal
#    bands, alternating 7 and background. The six marker pixels are swallowed into
#    that stripe texture instead of surviving as visible corners.
# b) complexity_calibration:
#    The combo changes three linked parts of the same object, not three unrelated
#    rules: fixed TL-BR anchors tell the solver which family of rectangles it is
#    seeing; the rectangle itself becomes the destination; and the stripe direction
#    is globally tied to the fixed diagonal, so there is only one stripe regime to
#    learn. To keep the rule visible without adding hidden cases, generation keeps
#    at least one real L-pair (and usually two when 3+ legend entries exist),
#    samples fill colors away from both bg and marker color, and rejects inputs
#    whose final output barely differs from a solid-rectangle version.
# c) implicit_assumptions_audit:
#    - The original `score_tlbr/score_trbl` comparison assumed both diagonals were
#      plausible. D9/0 invalidates that, so the combo solver asserts TL and BR
#      anchors directly instead of inferring between two options.
#    - The original `protect |= (g == marker)` preserved current L-markers. D3/0
#      invalidates that; keeping the line would silently revert to the old task.
#    - The original solid write assumed untouched cells inside the target should
#      stay as they were unless painted. D5/1 invalidates that too: stripe gaps
#      must be actively reset to bg, even if an earlier fill had colored them.
#    - These edits are compatible: once the destination becomes the full rectangle,
#      the fixed diagonal no longer changes *where* to paint, but it still defines
#      the only legal marker geometry and the perpendicular stripe direction.
# d) placement_validity:
#    The generator keeps all non-legend placements in bounds, off the legend scan
#    corridor, and non-overlapping at the pixel level. Every non-block legend
#    marker that is placed as an L-pair uses the fixed TL-BR orientation, including
#    optional distractors in unused colors. Blocks remain 2x2 recolors exactly as
#    in the base task, because they do not participate in the diagonal/stripe rule.
# e) visual_observability:
#    A human can see the fused rule from a few examples because all L-pairs lean
#    the same way, their full bboxes get painted, and the resulting texture is a
#    repeated anti-diagonal alternation between fill color and background. The
#    generator enforces a minimum visible stripe delta in the final output so the
#    samples do not collapse into nearly solid rectangles after overlaps/protection.
# f) visual_cleanliness:
#    The legend remains the only color key, and the input stays sparse because only
#    marker pixels are drawn before transformation. Later markers are protected from
#    earlier fills, which keeps the input readable and prevents stripe erasure from
#    destroying evidence needed for later passes. Optional distractors use unused
#    colors, so they remain visually separable from legend-driven objects.
# g) color_inference:
#    No ordering by numeric color id is introduced. The legend still explains the
#    fill color directly; the only added color behavior is that alternating cells in
#    each rectangle revert to the globally dominant background. That alternation is
#    tied to visible anti-diagonal structure, not to hidden metadata.
# new_solver_assumptions_after_edit:
#    - Every active non-block legend marker forms exactly one TL-BR L-pair bbox.
#    - Fill colors for L-pairs differ from bg so the stripe pattern is visible.
#    - At least one real L-pair survives protection strongly enough that the final
#      striped-rectangle output differs measurably from both the solid-rectangle
#      version and the source-task diamond version; generate_input() enforces this.


FIXED_ANCHOR_DIAGONAL = "TLBR"


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

    This helper is still used to validate the fixed TL-BR anchor pattern.
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
    """Mask of shortest-path cells between corners `a` and `b` inside the bbox."""
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _paint_striped_rectangle(
    out: np.ndarray,
    protect: np.ndarray,
    fill: int,
    bg: int,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> None:
    """
    Paint the bbox with TR-BL anti-diagonal stripes, alternating fill and bg.

    Stripe lines are the sets of cells with constant (r-rmin)+(c-cmin), which is
    perpendicular to the fixed TL-BR marker diagonal.
    """
    sub_out = out[rmin : rmax + 1, cmin : cmax + 1]
    writable = ~protect[rmin : rmax + 1, cmin : cmax + 1]
    if not np.any(writable):
        return

    sub_out[writable] = bg
    rr = np.arange(rmax - rmin + 1)[:, None]
    cc = np.arange(cmax - cmin + 1)[None, :]
    stripe_fill = writable & (((rr + cc) % 2) == 0)
    sub_out[stripe_fill] = fill


def _solve(
    grid: np.ndarray,
    *,
    fill_mode: str,
    swallow_current_markers: bool,
    fixed_diagonal: bool,
) -> np.ndarray:
    """
    Shared solver core for the combo and for visibility checks.

    fill_mode:
      - 'diamond_solid': source-like fill on the opposite diagonal
      - 'rect_solid': full bbox fill
      - 'rect_striped': full bbox with anti-diagonal fill/bg stripes
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

        if not swallow_current_markers:
            protect |= (g == marker) & (~legend_mask)

        if rmin == rmax or cmin == cmax:
            continue

        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)

        if fixed_diagonal:
            assert _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0)), (
                "combo variant expects a TL anchor at the bbox top-left corner"
            )
            assert _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0)), (
                "combo variant expects a BR anchor at the bbox bottom-right corner"
            )
            diamond_endpoints = (tr, bl)
        else:
            tl_anchor = _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0))
            tr_anchor = _l_anchor_present(marker_pts, tr[0], tr[1], (0, -1), (1, 0))
            bl_anchor = _l_anchor_present(marker_pts, bl[0], bl[1], (0, 1), (-1, 0))
            br_anchor = _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0))

            score_tlbr = int(tl_anchor) + int(br_anchor)
            score_trbl = int(tr_anchor) + int(bl_anchor)
            assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
            assert score_tlbr != score_trbl, "diagonal scores must not tie"
            diamond_endpoints = (tr, bl) if score_tlbr > score_trbl else (tl, br)

        if fill_mode == "diamond_solid":
            a, b = diamond_endpoints
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
            out[diamond & (~protect)] = int(fill)
        elif fill_mode == "rect_solid":
            writable = ~protect[rmin : rmax + 1, cmin : cmax + 1]
            out[rmin : rmax + 1, cmin : cmax + 1][writable] = int(fill)
        elif fill_mode == "rect_striped":
            _paint_striped_rectangle(out, protect, int(fill), bg, rmin, rmax, cmin, cmax)
        else:
            raise ValueError(f"unknown fill_mode: {fill_mode}")

    return out


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a generated sample visibly exercises the fused behavior.

    stripe_delta:
      Cells where striped full-rectangle output differs from solid full-rectangle
      output on the same fixed-diagonal input. This isolates the stripe effect.
    legacy_delta:
      Cells where the combo output differs from the source-task output on the same
      input. This catches accidental regression toward the original behavior.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    l_pairs = 0
    for _, marker, _ in entries:
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue
        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        if max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1 and len(marker_pts) == 4:
            continue
        if max(rs) == min(rs) or max(cs) == min(cs):
            continue
        tl = (min(rs), min(cs))
        br = (max(rs), max(cs))
        assert _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0))
        assert _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0))
        l_pairs += 1

    combo = _solve(g, fill_mode="rect_striped", swallow_current_markers=True, fixed_diagonal=True)
    solid_rect = _solve(g, fill_mode="rect_solid", swallow_current_markers=True, fixed_diagonal=True)
    source_like = _solve(g, fill_mode="diamond_solid", swallow_current_markers=False, fixed_diagonal=False)
    return {
        "l_pairs": l_pairs,
        "stripe_delta": int(np.count_nonzero(combo != solid_rect)),
        "legacy_delta": int(np.count_nonzero(combo != source_like)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint fixed-diagonal marker rectangles with anti-diagonal stripes from the legend.

    Input structure:
    - One dominant background color fills the canvas, and a 2-column legend is stacked near the bottom-left.
    - Each legend row stores a marker color in the left cell and its fill color in the right cell, read bottom to top.
    - Away from the legend, active non-block markers are always TL-BR L-pairs, while some entries may still be solid 2x2 blocks.

    Rule:
    - Process legend entries from bottom to top and gather all non-legend pixels of the current marker color.
    - A 2x2 block is recolored directly to its legend fill color.
    - A TL-BR L-pair paints its entire bounding rectangle, including its own marker pixels, with alternating fill-color and background stripes whose bands run along TR-BL anti-diagonals.
    - Legend cells and markers belonging to later legend rows remain protected from earlier writes.

    Key insight: Once every L-pair uses the same TL-BR anchor diagonal, the solver no longer needs to infer direction per object; that fixed diagonal also determines the stripe geometry, so the only job is to find the bbox and repaint the whole rectangle with perpendicular anti-diagonal bands.
    """
    return _solve(
        grid,
        fill_mode="rect_striped",
        swallow_current_markers=True,
        fixed_diagonal=True,
    )


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.20
BLOCK_PROB = 0.20

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: visibility guards ─────────────────────────────────────────
MIN_STRIPE_DELTA = 4
MIN_LEGACY_DELTA = 6

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
    """True iff the bbox keeps the L-pair visually distinct from a 2x2 block."""
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
    """Try to place one fixed-diagonal TL-BR L-pair of `color`."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, FIXED_ANCHOR_DIAGONAL)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
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
        if not fill_pool:
            return None
        fill = random.choice(fill_pool)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows) -> np.ndarray:
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _assign_marker_types(n_entries: int) -> list[str]:
    """
    Keep the fixed-diagonal striped-rectangle rule visible.

    With 2 entries we require at least 1 real L-pair.
    With 3+ entries we require at least 2 real L-pairs.
    Spare legend rows are assigned deliberately so block/noop branches stay
    meaningfully exercised in the generated corpus.
    """
    min_l_pairs = 1 if n_entries == 2 else 2
    special_slots = max(0, n_entries - min_l_pairs)

    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)

    for slot_idx in range(special_slots):
        idx = idx_pool[slot_idx]
        if slot_idx == 0:
            replacement = random.choices(
                ["L", "block", "noop"],
                weights=[0.40, 0.35, 0.25],
            )[0]
        else:
            replacement = random.choices(
                ["L", "block", "noop"],
                weights=[0.35, 0.40, 0.25],
            )[0]
        types[idx] = replacement

    if special_slots > 0 and all(t == "L" for t in types):
        if random.random() < max(NOOP_PROB, BLOCK_PROB):
            types[idx_pool[0]] = random.choice(["block", "noop"])
    return types


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
        metrics = _combo_metrics(grid)
        _ = main(grid)
    except Exception:
        return None

    min_l_pairs = 1 if len(legend_rows) == 2 else 2
    if metrics["l_pairs"] < min_l_pairs:
        return None
    if metrics["stripe_delta"] < MIN_STRIPE_DELTA:
        return None
    if metrics["legacy_delta"] < MIN_LEGACY_DELTA:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
