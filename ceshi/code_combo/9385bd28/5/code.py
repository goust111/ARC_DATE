# Combo variant of 9385bd28 (index 5)
# combo_spec: D3/3+D9/0+D14/1
# Name: 三角形标记组+固定对角线+对角带状填充
"""
Design analysis for the combo variant.

spec:
  concretization_and_behavior_trace:
    preserved:
      - Keep the original bottom-left 2-column legend and the bottom->top processing order.
      - Keep one dominant background color and color lookup purely via the legend.
      - Keep overwrite semantics: earlier legend rows may paint under later ones, but later markers
        and the legend itself stay protected.
    old_rule:
      - One legend color corresponded to an L-pair or a 2x2 block.
      - The solver inferred which bbox diagonal was marked, then filled the opposite rectangle-wide
        diamond or block.
    fused_rule:
      - Every active legend color now has exactly three single-pixel markers.
      - Those three pixels are three corners of one axis-aligned bbox, so they define a right triangle.
      - The fill direction is globally fixed to TL-BR, independent of which triangle corner is missing.
      - The output mask is the supercover TL-BR diagonal band of that bbox, clipped to the triangle.
    concrete_example:
      - Suppose the bbox is rows 2..6 and cols 4..8, with markers at TR=(2,8), BL=(6,4), BR=(6,8),
        so TL is the missing corner.
      - Original 9385bd28 logic would not even parse this object because there are no L-corners.
      - A plain D14 triangle variant would fill the whole lower-right triangle.
      - This combo instead draws only the TL-BR diagonal strip from (2,4) to (6,8), then keeps only
        the part of that strip lying inside the triangle; visually, only the lower half survives.
      - If the missing corner were TR or BL, the same TL-BR band would survive almost fully because
        that diagonal becomes a triangle edge. This makes the third marker materially affect the output.
  complexity_calibration:
    - Restrict marker groups to "three corners of one rectangle" rather than arbitrary triangles.
      This keeps the grouping learnable from 2-4 examples.
    - Use a single global diagonal family (TL-BR) so there is one stable geometric cue repeated across
      all legend entries.
    - Reject generated samples unless at least one group is genuinely triangle-clipped and at least one
      group has a triangle that is much larger than its painted band. This keeps both D14 and D3 visible.
    - Remove original no-op and 2x2 special cases because they would weaken the fused cue and reintroduce
      irrelevant branches from the source task.
  implicit_assumption_audit:
    - `_find_legend` remains valid because the legend geometry is unchanged.
    - The original solver assumed marker detection by color, which still holds.
    - The original diagonal-scoring and `_l_anchor_present` assumptions are invalid because objects are no
      longer 6-pixel L-pairs; those branches are removed entirely.
    - The original 2x2 recolor branch is invalid because every active entry now has exactly 3 pixels.
    - The new solver assumes each marker group is exactly three distinct bbox corners and therefore forms
      a non-collinear right triangle; the generator enforces this directly.
    - The new solver also assumes the relevant band is always the bbox TL-BR diagonal; the generator uses
      only triangle groups compatible with that fixed global choice.
  placement_legality:
    - Marker pixels must stay in bounds, avoid the legend footprint, avoid the left-side scan corridor,
      and avoid one another.
    - Triangle interiors and painted bands may overlap across different colors; overwrite order handles that.
    - Distractors, when present, use non-legend colors and obey the same geometric placement rules.
  observability_and_cleanliness:
    - Humans can see three same-color corner dots, infer an axis-aligned triangle, and observe that only
      a thin TL-BR stripe inside that triangle is recolored.
    - Fill colors are chosen away from the background and legend marker colors so the stripe remains visible.
    - Thin bands create much less clutter than full-triangle fills, which helps when multiple entries coexist.
  color_inference:
    - Colors are never sorted or interpreted numerically; the legend is the only source of marker->fill mapping.
    - Repeated TL-BR stripe orientation across entries gives enough structure to infer the geometric rule
      without relying on hidden color-id conventions.
"""

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


FIXED_BAND_DIAGONAL = "TLBR"


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
    """Mask of all cells whose centers lie inside or on the triangle defined by `vertices`."""
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


def _diag_band_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """Mask for the 1-2 pixel digital diagonal band between corners `a` and `b`."""
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _segment_hits_cell(a, b, r, c):
                m[r, c] = True
    return m


def _group_info(
    marker_pts: list[tuple[int, int]]
) -> tuple[int, int, int, int, dict[str, tuple[int, int]], str]:
    """Validate a 3-corner triangle group and return its bbox, corner map, and missing corner name."""
    pts = [tuple(map(int, pt)) for pt in marker_pts]
    assert len(pts) == 3, "each active legend entry must have exactly three marker pixels"
    pts_set = set(pts)
    assert len(pts_set) == 3, "triangle marker pixels must be distinct"

    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    assert rmin < rmax and cmin < cmax, "triangle group bbox must have positive height and width"

    corners = {
        "TL": (rmin, cmin),
        "TR": (rmin, cmax),
        "BL": (rmax, cmin),
        "BR": (rmax, cmax),
    }
    assert pts_set.issubset(set(corners.values())), "marker pixels must occupy bbox corners"
    missing = [name for name, pt in corners.items() if pt not in pts_set]
    assert len(missing) == 1, "triangle group must use exactly three of the four bbox corners"
    return rmin, cmin, rmax, cmax, corners, missing[0]


def _paint_triangle_bands(grid: np.ndarray, *, clip_to_triangle: bool) -> np.ndarray:
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
        rmin, cmin, rmax, cmax, corners, _missing = _group_info(marker_pts)

        protect = legend_mask.copy()
        for future_marker in marker_order[i + 1 :]:
            protect |= (g == future_marker) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        tri = _triangle_mask(h, w, marker_pts)
        assert FIXED_BAND_DIAGONAL == "TLBR"
        band = _diag_band_mask(h, w, rmin, rmax, cmin, cmax, corners["TL"], corners["BR"])
        mask = band & tri if clip_to_triangle else band
        out[mask & (~protect)] = int(fill)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint fixed-orientation diagonal bands inside three-corner triangle groups using a color legend.

    Input structure:
    - The grid has one dominant background color and a bottom-left vertical 2-column legend whose
      rows are `(marker_color, fill_color)` read from bottom to top.
    - Outside the legend, each active marker color appears exactly three times as single pixels.
    - Those three pixels occupy three corners of one axis-aligned rectangle, so they define one
      right triangle and implicitly identify the missing fourth corner.

    Rule:
    - Scan legend entries bottom-to-top and gather the three non-legend pixels of the current marker color.
    - Compute that group's bounding box and triangle region, but do not infer a per-object diagonal.
    - Always draw the bbox's TL-BR supercover diagonal band, then clip that band to the triangle
      determined by the three markers.
    - Color the surviving band with the legend's fill color while preserving legend cells, the current
      marker pixels, and all marker pixels belonging to later legend rows.

    Key insight: the three markers define *which triangular half* of a rectangle is relevant, while the
    direction of painting is fixed globally; the output is the TL-BR diagonal band restricted to that triangle.
    """
    return _paint_triangle_bands(grid, clip_to_triangle=True)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: triangle groups ────────────────────────────────────────────
TRI_BBOX_H_RANGE = (4, 9)
TRI_BBOX_W_RANGE = (4, 9)
TRIANGLE_RETRIES = 220
MISSING_CORNER_CHOICES = ["TL", "TR", "BL", "BR"]
CLIPPED_MISSING_CORNERS = ("TL", "BR")
EDGE_MISSING_CORNERS = ("TR", "BL")

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.20
DISTRACTOR_RETRIES = 120

# ── CFG: observability guards ──────────────────────────────────────
MIN_TRIANGLE_MINUS_BAND = 4
MIN_CLIP_LOSS = 2

# ── CFG: retry budgets ─────────────────────────────────────────────
GEN_RETRIES = 500


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
    missing_corner: str,
    retries: int,
    can_place,
    do_place,
) -> dict[str, int | str] | None:
    for _ in range(retries):
        size = _pick_triangle_bbox(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        cells = _triangle_vertices_from_bbox(rmin, cmin, rmax, cmax, missing_corner)
        if can_place(cells):
            do_place(cells, color)
            return {
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmax,
                "cmax": cmax,
                "missing": missing_corner,
            }
    return None


def _planned_missing_corners(n_entries: int) -> list[str]:
    """
    Pre-plan missing-corner branches so every sample exposes both:
    - clipped cases (missing TL/BR), where the fixed TL-BR band loses cells
    - edge-aligned cases (missing TR/BL), where the same band hugs a triangle edge
    """
    assert n_entries >= 2, "legend must contain at least two entries"

    seq = [
        random.choice(CLIPPED_MISSING_CORNERS),
        random.choice(EDGE_MISSING_CORNERS),
    ]
    remaining = [corner for corner in MISSING_CORNER_CHOICES if corner not in seq]
    random.shuffle(remaining)
    while len(seq) < n_entries and remaining:
        seq.append(remaining.pop())
    while len(seq) < n_entries:
        seq.append(random.choice(MISSING_CORNER_CHOICES))
    random.shuffle(seq)
    return seq


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


def _combo_metrics(
    H: int, W: int, group_infos: list[dict[str, int | str]]
) -> tuple[int, int, set[str], set[str]]:
    strong_band_count = 0
    clipped_group_count = 0
    missing_types: set[str] = set()
    missing_classes: set[str] = set()
    for info in group_infos:
        rmin = int(info["rmin"])
        cmin = int(info["cmin"])
        rmax = int(info["rmax"])
        cmax = int(info["cmax"])
        missing = str(info["missing"])
        vertices = _triangle_vertices_from_bbox(rmin, cmin, rmax, cmax, missing)
        tri = _triangle_mask(H, W, vertices)
        band = _diag_band_mask(H, W, rmin, rmax, cmin, cmax, (rmin, cmin), (rmax, cmax))
        clipped = band & tri
        if int(tri.sum()) - int(clipped.sum()) >= MIN_TRIANGLE_MINUS_BAND:
            strong_band_count += 1
        if int(band.sum()) - int(clipped.sum()) >= MIN_CLIP_LOSS:
            clipped_group_count += 1
        missing_types.add(missing)
        missing_classes.add("clipped" if missing in CLIPPED_MISSING_CORNERS else "edge")
    return strong_band_count, clipped_group_count, missing_types, missing_classes


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

    group_infos: list[dict[str, int | str]] = []
    planned_missing = _planned_missing_corners(len(legend_rows))
    for (_, marker, _), missing_corner in zip(legend_rows, planned_missing):
        info = _try_place_triangle_group(
            H, W, marker, missing_corner, TRIANGLE_RETRIES, can_place, do_place
        )
        if info is None:
            return None
        group_infos.append(info)

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_triangle_group(
                H,
                W,
                random.choice(avail),
                random.choice(MISSING_CORNER_CHOICES),
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
            )

    strong_band_count, clipped_group_count, missing_types, missing_classes = _combo_metrics(
        H, W, group_infos
    )
    if strong_band_count == 0:
        return None
    if clipped_group_count == 0:
        return None
    if len(missing_types) < 2:
        return None
    if missing_classes != {"clipped", "edge"}:
        return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        combo_out = main(grid)
        degenerate_out = _paint_triangle_bands(grid, clip_to_triangle=False)
    except Exception:
        return None
    if np.array_equal(combo_out, degenerate_out):
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
