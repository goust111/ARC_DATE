# Variant of 9385bd28
# Dimension: D2 — 源对象：标记的形状结构 (src)
# Change: L 形 3 像素角标记，位于外接矩形的对角两端（共 6 像素）；另有 2x2 实心方块变体... → 十字形标记

"""
design_analysis:
  variant_core:
    original_marker:
      Each active legend color used either:
      - two 3-pixel L-corners on one bbox diagonal, or
      - a 2x2 solid block special case.
    new_marker:
      Each active legend color now uses exactly two 5-pixel crosses.
      A cross is {center, up, down, left, right}.
      The two cross centers are the diagonal corners of the rectangle to fill.
      The crosses stay visible in the output; only the rectangle between their centers is recolored.

  behavior_trace:
    concrete_example:
      Suppose marker color 4 has crosses centered at (2,3) and (7,9).
      Input shows ten pixels of color 4:
      - around (2,3): (2,3),(1,3),(3,3),(2,2),(2,4)
      - around (7,9): (7,9),(6,9),(8,9),(7,8),(7,10)
      Original solver path:
      - expected L-shaped anchors at bbox corners
      - scored TL-BR vs TR-BL using corner-arm patterns
      - that logic would fail because a symmetric cross has no L orientation
      Variant path:
      - split same-color pixels into 4-neighbor components
      - recognize each 5-pixel component as a cross and read its center
      - use the two centers as opposite rectangle corners
      - fill rows 2..7 and cols 3..9, excluding protected marker/legend pixels
    difference:
      The changed step is not the legend or overwrite order, but the way marker geometry is decoded.
      The bounding box of all marker pixels is no longer the fill rectangle because each cross extends
      one cell beyond its center.

  complexity_calibration:
    visibility:
      Crosses are larger and more salient than L-corners, so the D2 change is visually obvious.
      Keeping exactly two identical crosses per active color limits free-form ambiguity.
    ambiguity_control:
      Cross centers must differ in both row and col by at least 2, so they are truly diagonal and
      their components do not merge under 4-connectivity.
      No 2x2 block variant is generated in this version; otherwise the source-shape dimension would
      still contain a second unrelated marker family and dilute the intended change.
    variety:
      Rectangles still vary in size and diagonal orientation (TL-BR more common, TR-BL also present),
      so the samples do not collapse into a single easy template.

  implicit_assumption_review:
    old_main_dependencies:
      - "_l_anchor_present(...)" assumed marker pixels sat exactly on bbox corners with oriented arms.
      - "score_tlbr / score_trbl" assumed one diagonal could be inferred from asymmetric corner shapes.
      - "2x2 block" branch assumed a non-pair shape could be solved by recoloring only its own cells.
    variant_effect:
      - Crosses are rotationally symmetric, so diagonal scoring by oriented arms is invalid.
      - The fill rectangle must come from detected cross centers, not from extremal marker pixels.
      - Since all active markers are now cross-pairs, the block branch is removed.
    new_assumptions:
      - Every non-noop legend marker color has exactly two disconnected 5-pixel cross components.
      - Each component contains a unique center whose four orthogonal neighbors are present.
      The generator enforces both assumptions directly.

  placement_legality:
    bounds:
      Centers are placed at least one cell away from the border so every arm stays in-bounds.
    collisions:
      Marker pixels cannot overlap legend cells, forbidden legend-scan columns, or other marker pixels.
      Fill regions may overlap, matching the original task's overwrite behavior.
    solver_compatibility:
      Because same-color cross components are disconnected, component-based center extraction is stable.
      Optional distractors use colors absent from the legend, so they are ignored by the solver.

  visual_observability:
    human_view:
      In 2-4 examples, a person can spot that each relevant color appears as two plus signs and that
      the rectangle spanning the two plus centers gets recolored according to the bottom-left legend.
      The rule does not depend on hidden color ordering or numeric color ids.

  visual_cleanliness:
    controls:
      - Crosses are kept disconnected.
      - The legend-protection zone is preserved from the original task.
      - Fill overlap is allowed, but marker overlap is disallowed, so the causal objects remain legible.

  color_inference:
    cue:
      The only color transformation remains the explicit legend mapping (marker color -> fill color).
      This is structurally signposted in the input, so even when multiple colors appear, the mapping is
      locally inferable rather than arbitrary.
"""

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


_CROSS_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def _split_components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Split a point set into 4-connected components."""
    remaining = set(pts)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _cross_center(comp: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Return the center of an exact 5-pixel cross component, else None."""
    if len(comp) != 5:
        return None
    for r, c in comp:
        expected = {(r + dr, c + dc) for dr, dc in _CROSS_OFFSETS}
        if expected == comp:
            return r, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rectangles whose opposite corners are indicated by same-color cross markers.

    Input structure:
    - The background is mostly uniform, with a two-column legend stacked near the bottom-left.
    - Each legend row gives a marker color in the left cell and its fill color in the right cell.
    - Away from the legend, an active marker color appears as two separate 5-pixel crosses.

    Rule:
    - Read legend rows from bottom to top.
    - For each marker color, locate its non-legend pixels and split them into 4-connected components.
    - If two exact crosses are present, use their centers as opposite corners of an axis-aligned rectangle.
    - Fill that rectangle with the legend's fill color while preserving legend cells, current marker cells,
      and marker pixels belonging to later legend entries.

    Key insight:
    - The relevant corners are not the extrema of the marker pixels themselves; they are the centers of the
      two cross shapes. Because each cross sticks out by one cell in four directions, center recovery is the
      essential shape-decoding step in this variant.
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
        if not marker_pts:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        comps = _split_components4(marker_pts)
        centers: list[tuple[int, int]] = []
        for comp in comps:
            center = _cross_center(comp)
            assert center is not None, "every active marker component must be an exact 5-pixel cross"
            centers.append(center)

        assert len(centers) == 2, "every active marker color must contain exactly two crosses"
        (r0, c0), (r1, c1) = centers
        rmin, rmax = sorted((r0, r1))
        cmin, cmax = sorted((c0, c1))
        assert rmin < rmax and cmin < cmax, "cross centers must be diagonal corners"

        out[rmin : rmax + 1, cmin : cmax + 1] = np.where(
            protect[rmin : rmax + 1, cmin : cmax + 1],
            out[rmin : rmax + 1, cmin : cmax + 1],
            int(fill),
        )

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3

# ── CFG: cross-pair rectangle shape ────────────────────────────────
RECT_H_MAX = 17
RECT_W_MAX = 12
RECT_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=4, h=3/w>=5, h>=5/w=3]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
CROSSPAIR_RETRIES = 150
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300


def _cross_cells(center_r: int, center_c: int) -> list[tuple[int, int]]:
    return [(center_r + dr, center_c + dc) for dr, dc in _CROSS_OFFSETS]


def _cross_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 10 cells for a pair of crosses whose centers sit on one rectangle diagonal."""
    if diag == "TLBR":
        centers = [(rmin, cmin), (rmax, cmax)]
    else:
        centers = [(rmin, cmax), (rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for r, c in centers:
        cells.extend(_cross_cells(r, c))
    return cells


def _pick_rect_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid fill-rectangle size for a cross pair."""
    h_hi = min(RECT_H_MAX, H - 2)
    w_hi = min(RECT_W_MAX, W - 2)
    if h_hi < 3 or w_hi < 3:
        return None

    options: list[tuple[str, float]] = []
    if h_hi >= 4 and w_hi >= 4:
        options.append(("both_ge4", RECT_SHAPE_WEIGHTS[0]))
    if h_hi >= 3 and w_hi >= 5:
        options.append(("h3_w_ge5", RECT_SHAPE_WEIGHTS[1]))
    if h_hi >= 5 and w_hi >= 3:
        options.append(("h_ge5_w3", RECT_SHAPE_WEIGHTS[2]))

    if not options:
        return random.randint(3, h_hi), random.randint(3, w_hi)

    shape_choice = random.choices(
        [name for name, _ in options],
        weights=[weight for _, weight in options],
    )[0]

    if shape_choice == "both_ge4":
        return random.randint(4, h_hi), random.randint(4, w_hi)
    if shape_choice == "h3_w_ge5":
        return 3, random.randint(5, w_hi)
    return random.randint(5, h_hi), 3


def _try_place_cross_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place a cross pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_rect_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin_lo, rmin_hi = 1, H - bh - 1
        cmin_lo, cmin_hi = 1, W - bw - 1
        if rmin_lo > rmin_hi or cmin_lo > cmin_hi:
            continue
        rmin = random.randint(rmin_lo, rmin_hi)
        cmin = random.randint(cmin_lo, cmin_hi)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _cross_pair_cells(rmin, cmin, rmax, cmax, diag)
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
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
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
    types = ["cross"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"

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
        if types[i] == "noop":
            continue
        if not _try_place_cross_pair(H, W, marker, CROSSPAIR_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_cross_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
