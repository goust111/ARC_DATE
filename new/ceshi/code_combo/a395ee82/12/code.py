# Combo variant of a395ee82 (index 12)
# combo_spec: D4/3+D8/0
# Name: 旋转stamp边框原点

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# combo_variant:
#   fused_rule:
#     - Keep the original input grammar: one monochrome template in color T and a
#       disjoint singleton lattice with one T-colored origin plus several O-colored
#       non-origin markers.
#     - Output all non-origin copies as the original solid template in color T.
#     - Replace the original "origin becomes solid O" behavior with a fused rule:
#       the origin copy is first rotated 180 degrees inside the same bbox, then
#       only its border support is drawn in color O; its interior support is left
#       as background instead of T. The result is a rotated hollow outline versus
#       several upright solid copies.
#   concrete_behavior_trace:
#     - Example template: a 4x3 asymmetric shape with one true interior cell.
#     - Seed task:
#       input shows the template once in T and a singleton lattice elsewhere.
#       output stamps upright solid copies at every lattice offset and recolors the
#       origin copy entirely to O.
#     - D4-only change:
#       the origin copy would instead be a solid 180-degree-rotated T shape.
#     - D8-only change:
#       the origin copy would keep the original orientation but only its border
#       would be O, with interior still present.
#     - Fused combo here:
#       the origin copy uses the rotated silhouette, but only the rotated border is
#       drawn, in O. Any rotated interior cells are omitted completely, so the
#       origin reads as a hollow rotated outline rather than a filled stamp.
#   complexity_calibration:
#     - The output still uses only three colors {bg, T, O}; there is no arbitrary
#       color remapping beyond reusing O for the special outline.
#     - The generator requires repeated upright solid copies (5-6 lattice sites) so
#       the user sees a stable default pattern before identifying the lone exception.
#     - The template must have at least one interior cell; otherwise "border-only"
#       would collapse to the full silhouette and the combo would visually degenerate.
#     - The template must also change noticeably under 180-degree rotation, and its
#       border set must change under that rotation as well. This prevents samples
#       where the hollow O outline is visible but the rotation dimension contributes
#       nothing.
#   implicit_assumption_audit:
#     - `template = max(comps, key=area)` still assumes the only non-singleton
#       object is the template. The generator preserves that.
#     - `templ_col` still identifies both the template color and the unique lattice
#       origin marker. This remains valid.
#     - The seed solver used coverage counters because recolor decisions depended on
#       overlap and on whether a pixel belonged to the origin copy. The combo rule no
#       longer needs overlap accounting; it needs two stencils instead: the ordinary
#       template cells and the rotated-origin border cells.
#     - `off_r = dr * h` and `off_c = dc * w` still assume bbox-aligned stamping.
#       A 180-degree rotation keeps the bbox size unchanged, so this assumption
#       survives the fusion unchanged.
#     - New assumption introduced by the combo: the rotated origin border must be
#       distinguishable from the unrotated border. The generator enforces this via a
#       minimum symmetric-difference threshold on border masks.
#   placement_legality:
#     - Every stamped copy must fit within the 22x22 grid.
#     - The singleton cluster must remain disjoint from the template bbox with at
#       least one background cell gap, exactly as in the seed generator.
#     - For cleanliness, the generator forbids overlap not only between the visible
#       output pixels but between the full silhouettes of all stamps, using the
#       rotated full silhouette at the origin. This reserves the origin interior so
#       it stays hollow instead of being accidentally filled by another copy.
#   visual_observability:
#     - A human can infer the rule from a few examples because one upright T template
#       appears in the input, many upright solid T copies appear in the output, and
#       exactly one copy differs in two coupled ways: it is upside-down and hollow,
#       with only an O outline.
#     - The generator rejects 180-degree-symmetric borders, so the rotated outline
#       is visibly different from an unrotated outline on every accepted sample.
#   visual_cleanliness:
#     - No mixed overlays are allowed in generated samples, so the origin outline is
#       not obscured by other stamps.
#     - The O color is confined to the singleton hints in the input and to the one
#       hollow origin outline in the output, making its role easy to track.
#   inferability_of_changes:
#     - The combo does not depend on hidden color ordering or numeric color ids.
#     - The two special behaviors are structurally coupled at one repeated lattice
#       site: "the T singleton marks the copy that becomes a rotated O outline".
#       This keeps the inference space narrow enough for ARC-style reasoning.
#   solver_assumptions_after_edit:
#     - The solver assumes there is exactly one T-colored singleton, at least one
#       non-T singleton, and at least one non-zero lattice delta so the gcd-based
#       spacing recovery is defined. The generator explicitly builds that structure.
#     - The solver assumes the template border can be computed from 4-neighbor shape
#       adjacency. The generator only emits 4-connected polyomino templates, so this
#       border notion is stable and unambiguous.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
MIN_INTERIOR_CELLS = 1
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
ROTATION_DIFF_MIN = 4
BORDER_ROTATION_DIFF_MIN = 2

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 100
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


def _rot180_rel(cells: list[tuple[int, int]] | np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    return sorted((int(h - 1 - r), int(w - 1 - c)) for r, c in cells)


def _rotation_diff(cells: list[tuple[int, int]] | np.ndarray, h: int, w: int) -> int:
    base = {(int(r), int(c)) for r, c in cells}
    rot = {(int(h - 1 - r), int(w - 1 - c)) for r, c in cells}
    return int(len(base ^ rot))


def _partition_template_cells(
    cells: list[tuple[int, int]] | np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split template-relative cells into border cells and interior cells."""
    cell_set = {(int(r), int(c)) for r, c in cells}
    border: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for r, c in sorted(cell_set):
        if any((r + dr, c + dc) not in cell_set for dr, dc in ORTHO_DIRS):
            border.append((int(r), int(c)))
        else:
            interior.append((int(r), int(c)))
    return border, interior


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    outline_col: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    templ_rel: list[tuple[int, int]],
    rot_border_rel: list[tuple[int, int]],
    deltas: list[tuple[int, int, bool]],
) -> np.ndarray:
    """Render the fused output from parsed or generated layout data."""
    out = np.full((H, W), int(bg), dtype=int)
    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        rels = rot_border_rel if is_origin else templ_rel
        color = int(outline_col) if is_origin else int(templ_col)
        for rr, cc in rels:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton lattice.

    Input structure:
    - The dominant color is the background.
    - One large 4-connected monochrome component in color T is the template sprite.
    - Separate singleton pixels form a uniform-spacing lattice: exactly one singleton
      is also color T and marks the origin, while the remaining singletons share
      color O.

    Rule:
    - Recover each singleton's coarse lattice offset from the unique T-colored
      origin marker.
    - Convert those coarse offsets into template-sized translations.
    - At every non-origin offset, stamp the template unchanged in color T.
    - At the origin offset, rotate the template 180 degrees inside its bbox, then
      draw only the rotated border cells in color O; the rotated interior cells are
      left as background.

    Key insight: The special singleton still tells you which copy is exceptional, but
    the exception is now a coupled geometric-and-structural change: that one copy is
    upside-down and hollow, showing only an O-colored outline instead of a solid T
    fill.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]
    templ_rot_rel = _rot180_rel(templ_rel, h, w)
    rot_border_rel, _rot_interior_rel = _partition_template_cells(templ_rot_rel)

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    outline_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col) == int(templ_col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    coarse_deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in deltas]

    return _render_output(
        H=H,
        W=W,
        bg=bg,
        templ_col=templ_col,
        outline_col=outline_col,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        templ_rel=templ_rel,
        rot_border_rel=rot_border_rel,
        deltas=coarse_deltas,
    )


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier: list[tuple[int, int]] = []
    for dr, dc in ORTHO_DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in ORTHO_DIRS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    if len(cells) != n:
        return None
    return sorted((int(r), int(c)) for r, c in cells)


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Polyomino of n cells that spans exactly rows x cols."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(
    h: int, w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Template with true interior and a visibly different rotated border."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is None:
            continue
        border_cells, interior_cells = _partition_template_cells(cells)
        rot_cells = _rot180_rel(cells, h, w)
        rot_border_cells, _rot_interior_cells = _partition_template_cells(rot_cells)
        if len(interior_cells) < MIN_INTERIOR_CELLS:
            continue
        if _rotation_diff(cells, h, w) < ROTATION_DIFF_MIN:
            continue
        if _rotation_diff(border_cells, h, w) < BORDER_ROTATION_DIFF_MIN:
            continue
        return cells, border_cells, rot_cells, rot_border_cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _all_stamp_points(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    templ_rot_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    """Reserve full stamp silhouettes so the origin outline stays hollow and isolated."""
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        rels = templ_rot_cells if (int(dr) == 0 and int(dc) == 0) else templ_cells
        for tr, tc in rels:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)
    return occupied


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    rot_border_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Try to place the singleton lattice and keep the input valid for the solver."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)
    rendered_deltas = [(int(dr), int(dc), int(dr) == 0 and int(dc) == 0) for dr, dc in deltas]

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * GUNIT)
        sr1 = int(orr + max_dr * GUNIT)
        sc0 = int(orc + min_dc * GUNIT)
        sc1 = int(orc + max_dc * GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((int(orr + dr * GUNIT), int(orc + dc * GUNIT)) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for dr, dc in deltas:
            grid[int(orr + dr * GUNIT), int(orc + dc * GUNIT)] = int(T if (dr == 0 and dc == 0) else O)

        expected = _render_output(
            H=GRID_SIZE,
            W=GRID_SIZE,
            bg=bg,
            templ_col=T,
            outline_col=O,
            r0=r0,
            c0=c0,
            h=h,
            w=w,
            templ_rel=templ_cells,
            rot_border_rel=rot_border_cells,
            deltas=rendered_deltas,
        )
        origin_patch = expected[r0 : r0 + h, c0 : c0 + w]
        if int(np.count_nonzero(origin_patch == int(T))) != 0:
            continue
        if int(np.count_nonzero(origin_patch == int(O))) != len(rot_border_cells):
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template and lattice, then place them."""
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ = _gen_template(h, w)
    if templ is None:
        return None
    templ_cells, _border_cells, rot_cells, rot_border_cells = templ

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied = _all_stamp_points(deltas, templ_cells, rot_cells, r0, c0, h, w)
    if occupied is None:
        return None

    return _place_singletons(
        deltas=deltas,
        min_dr=min_dr,
        max_dr=max_dr,
        min_dc=min_dc,
        max_dc=max_dc,
        templ_cells=templ_cells,
        rot_border_cells=rot_border_cells,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        bg=bg,
        T=T,
        O=O,
        occupied=occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
