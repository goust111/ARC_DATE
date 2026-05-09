# Variant of 9385bd28
# Dimension: D14 — 标记对的基数关系 (multi_step)
# Change: 每个图例条目对应 1 个标记对（L 对或 2x2 方块），共 2-5 对... → 三角形标记组（三个标记定义三角形填充）
"""
Design analysis for the D14 variant.

spec:
  concretization:
    legend: keep the original 2-column vertical legend at bottom-left and keep bottom->top processing.
    marker_group: every legend color now owns exactly three single-pixel markers outside the legend.
    geometry: the three markers are the three visible corners of one axis-aligned bounding box; the missing
      fourth corner tells the viewer which half of that box is absent, and the solver fills the triangle
      whose vertices are the three markers.
    fill: the filled region is the discrete triangle interior including edges, with current marker pixels
      preserved and later legend entries protected exactly as before.
  behavior_trace:
    old:
      - collect all pixels of one marker color
      - detect which bbox diagonal contains two L-corner anchors
      - fill the opposite diagonal's diamond inside the rectangle
    new:
      - collect the three marker pixels of one legend color
      - treat those pixels as triangle vertices
      - fill every cell whose center lies inside/on that triangle
    difference:
      - cardinality changes from paired corner structures to 3-point groups
      - the output region is now a half-rectangle triangle rather than a rectangle/diamond derived from a pair
  complexity_calibration:
    keep only four missing-corner cases (TL/TR/BL/BR) instead of arbitrary free-form triangles.
    This is visually obvious, avoids underconstrained matching, and still forces the solver to use
    three-point geometry instead of the original diagonal-pair heuristic.
  implicit_assumptions_checked:
    - old `_l_anchor_present` and diagonal scoring relied on six-pixel L markers; these assumptions are removed.
    - old 2x2 special casing relied on an alternate cardinality branch; removed so D14 remains the only change.
    - protection logic for legend cells and future markers is still valid and is preserved.
    - `main()` now assumes each active legend entry has exactly 3 non-legend pixels and non-zero triangle area;
      generator guarantees this directly.
  placement_legality:
    markers never overlap the legend, never leave the grid, and never overlap one another.
    Only marker pixels are constrained at generation time; triangle interiors may overlap, which is safe because
    `main()` resolves conflicts by the same bottom->top overwrite policy as the original task.
  observability:
    every active entry visibly shows three same-color corner dots and the output shows a large triangular fill.
    Across 2-4 examples, humans can infer "three corners -> fill that triangle" without hidden numeric ordering.
  cleanliness:
    generator uses bbox sizes >= 3x3 so triangles are large enough to see, and avoids placing markers in the
    legend scan corridor on the left.
  color_inference:
    color behavior remains structured by the explicit legend; fill colors are never chosen by sorting or by
    invisible color-id arithmetic.
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


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    """Twice the signed area of triangle abc, using (row, col) coordinates."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _triangle_mask(h: int, w: int, vertices: list[tuple[int, int]]) -> np.ndarray:
    """Mask of all grid cells whose centers lie inside or on the triangle defined by `vertices`."""
    assert len(vertices) == 3, "triangle needs exactly three vertices"
    a, b, c = vertices
    area2 = abs(_cross(a, b, c))
    assert area2 != 0, "triangle markers must be non-collinear"

    rs = [r for r, _ in vertices]
    cs = [col for _, col in vertices]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    mask = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            p = (r, col)
            s = abs(_cross(p, a, b)) + abs(_cross(p, b, c)) + abs(_cross(p, c, a))
            if s == area2:
                mask[r, col] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill triangles indicated by three-pixel marker groups using a color legend.

    Input structure:
    - The grid has one background color and a bottom-left vertical legend with rows `(marker, fill)`.
    - For each active legend entry, the matching marker color appears exactly three times outside the legend.
    - Those three same-color pixels act as triangle vertices; generator examples place them on three corners of an axis-aligned box.

    Rule:
    - Read legend entries from bottom to top and gather the non-legend pixels of that marker color.
    - Interpret the three marker pixels as one triangle and fill every cell inside or on its boundary with the legend's fill color.
    - Keep legend pixels, current marker pixels, and all later marker groups visible while earlier fills are applied.

    Key insight: the important relation is no longer "which diagonal has the L-markers" but "which three points belong together";
    those three points directly define the filled region's geometry.
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
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = [tuple(map(int, pt)) for pt in np.argwhere((g == marker) & (~legend_mask))]
        assert len(marker_pts) == 3, "each active legend entry must have exactly three marker pixels"

        protect = legend_mask.copy()
        for future_marker in marker_order[i + 1 :]:
            protect |= (g == future_marker) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        tri = _triangle_mask(h, w, marker_pts)
        out[tri & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: triangle groups ────────────────────────────────────────────
TRI_BBOX_H_RANGE = (3, 9)
TRI_BBOX_W_RANGE = (3, 9)
TRIANGLE_RETRIES = 180
MISSING_CORNER_CHOICES = ["TL", "TR", "BL", "BR"]

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25
DISTRACTOR_RETRIES = 90

# ── CFG: retry budgets ─────────────────────────────────────────────
GEN_RETRIES = 400


def _triangle_vertices_from_bbox(
    rmin: int, cmin: int, rmax: int, cmax: int, missing_corner: str
) -> list[tuple[int, int]]:
    corners = {
        "TL": (rmin, cmin),
        "TR": (rmin, cmax),
        "BL": (rmax, cmin),
        "BR": (rmax, cmax),
    }
    return [pt for name, pt in corners.items() if name != missing_corner]


def _pick_triangle_bbox(H: int, W: int) -> tuple[int, int] | None:
    h_lo, h_hi = TRI_BBOX_H_RANGE
    w_lo, w_hi = TRI_BBOX_W_RANGE
    max_h = min(h_hi, H)
    max_w = min(w_hi, W)
    if max_h < h_lo or max_w < w_lo:
        return None
    return random.randint(h_lo, max_h), random.randint(w_lo, max_w)


def _try_place_triangle_group(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    for _ in range(retries):
        size = _pick_triangle_bbox(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        missing = random.choice(MISSING_CORNER_CHOICES)
        cells = _triangle_vertices_from_bbox(rmin, cmin, rmax, cmax, missing)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
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
        marker = marker_colors[i]
        fill = random.choice(fill_pool)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
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

    for _, marker, _ in legend_rows:
        if not _try_place_triangle_group(H, W, marker, TRIANGLE_RETRIES, can_place, do_place):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_triangle_group(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
