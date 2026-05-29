# Combo variant of a395ee82 (index 16)
# combo_spec: D8/1+D14/0
# Name: 补集象限旋转

from __future__ import annotations

import math
import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# a) Concrete behavior and trace
# - Keep the original input decomposition: one large monochrome template in
#   color T, plus a separated singleton lattice where the unique T singleton
#   marks the origin and the remaining singletons use color O.
# - The fused output now combines two changes at once.
#   1) The origin copy no longer draws the template support at all. Inside the
#      template bbox, it draws the complement of the template in O, leaving the
#      template cells as background.
#   2) Every non-origin lattice point still creates a normal T-colored stamp,
#      but its orientation depends on the offset quadrant relative to the
#      origin: upper-right 0, upper-left 90 CCW, lower-left 180, lower-right
#      270.
# - Small trace: suppose the template lives in a 4x4 bbox and misses three
#   cells near one corner. In the source task the origin would be a same-shape
#   recolor; in D14-only it would still be the positive shape but rotated by
#   quadrant elsewhere; in this combo the origin becomes those three missing
#   cells in O, while the four diagonal neighbors show four quarter-turns of
#   the positive T shape.
#
# b) Complexity calibration
# - The rule space stays compact: there is one special origin behavior and four
#   fixed quadrant-to-rotation categories.
# - Every generated lattice includes all four diagonal quadrants, with an
#   optional sixth singleton repeating one quadrant. This gives repeated visual
#   support for the rotation mapping in each training sample.
# - The template bbox is forced to 4x4 and rotationally asymmetric, so quarter
#   turns are visible; the template is also non-full, so the origin complement
#   is always visible.
#
# c) Implicit-assumption audit of the original solver
# - Original `templ_rel` translation without modification is invalid because
#   non-origin copies may rotate. Each stamp now needs a bbox-relative rotation
#   before translation.
# - Original `off_r = dr * h` / `off_c = dc * w` only works unchanged when
#   quarter turns do not swap dimensions. The fused generator therefore makes
#   the template bbox square and `main()` asserts that.
# - Original `center` logic assumed the origin copy shares the template support.
#   That assumption is broken: the origin support is the bbox complement, so the
#   origin cannot be counted inside the same positive-shape accumulator.
# - Axis-aligned non-origin deltas would make quadrant selection ambiguous.
#   `main()` treats non-origin deltas as strict diagonal-quadrant points, and
#   the generator constructs them that way.
# - Rendering the complement after the positive copies is safe only if stamp
#   bboxes are disjoint. The generator enforces box-aligned spacing so this is
#   deterministic.
#
# d) Placement legality
# - All stamp anchors differ by integer multiples of the common 4x4 box size,
#   so distinct stamp bboxes do not overlap and all copies fit inside 22x22.
# - The singleton cluster bbox is kept disjoint from the template bbox with at
#   least one background cell gap, preserving clean component recovery.
# - Singleton pixels are also kept away from the eventual non-background output
#   support to avoid clutter between the input cue and the output stamp field.
#
# e) Visual observability
# - The unique T singleton still visibly identifies the origin in the input.
# - The output shows one high-contrast negative image at the origin and normal
#   copies elsewhere, so the origin transformation is easy to notice.
# - Because all four diagonal quadrants are always present and the template has
#   four distinct rotations, the quadrant-rotation mapping is inferable from a
#   few examples instead of being an ungrounded guess.
#
# f) Visual cleanliness
# - The scene contains only three semantic elements: background, one template,
#   and one singleton lattice. The output contains one negative bbox plus clean
#   non-overlapping positive stamps.
# - Using one special origin box instead of mixing many transform types inside
#   the same area keeps the composition readable.
#
# g) Change inferability
# - The task still uses exactly three colors: bg, T, O.
# - O keeps a structural role: it marks non-origin singletons in the input and
#   becomes the origin-complement foreground in the output. No hidden ordering
#   of color ids is used.
# - To prevent degeneration, the generator rejects any sample whose fused output
#   matches either single-variant output on the same input scaffold.
#
# Post-edit assumption audit
# - `main()` assumes the largest component is the template, the template bbox is
#   square and not full, exactly one singleton matches the template color, every
#   non-origin singleton lies in a strict diagonal quadrant, and all translated
#   stamp boxes are disjoint. `generate_input()` directly constructs inputs that
#   satisfy those assumptions.


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """Upper-right -> 0, upper-left -> 1, lower-left -> 2, lower-right -> 3."""
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return 0
    assert dr != 0 and dc != 0
    if dr < 0 and dc > 0:
        return 0
    if dr < 0 and dc < 0:
        return 1
    if dr > 0 and dc < 0:
        return 2
    return 3


def _complement_cells(cells: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Return the bbox-level complement of a shape inside a square box."""
    cell_set = {(int(r), int(c)) for r, c in cells}
    return [(r, c) for r in range(size) for c in range(size) if (r, c) not in cell_set]


def _render_positive_copies(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    *,
    rotate_by_quadrant: bool,
    include_origin: bool,
    recolor_origin: bool,
) -> np.ndarray:
    """Render positive-shape copies, optionally rotating by quadrant and recoloring the origin."""
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        is_origin = int(dr) == 0 and int(dc) == 0
        if is_origin and not include_origin:
            continue
        turns = _quadrant_turns(dr, dc) if rotate_by_quadrant else 0
        rot_cells = _rotate_cells(templ_cells, size, turns)
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
        for rr, cc in rot_cells:
            total[int(off_r + rr), int(off_c + cc)] += 1
            if is_origin and recolor_origin:
                center[int(off_r + rr), int(off_c + cc)] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _paint_origin_complement(
    out: np.ndarray,
    bg: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
) -> None:
    """Overwrite the origin box with the complement of the canonical template."""
    templ_set = {(int(r), int(c)) for r, c in templ_cells}
    for rr in range(size):
        for cc in range(size):
            out[int(base_r + rr), int(base_c + cc)] = int(bg) if (rr, cc) in templ_set else int(overlap_col)


def _render_combo_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    """Render the fused rule: quadrant-rotated positives plus an origin complement."""
    out = _render_positive_copies(
        H,
        W,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        size,
        templ_cells,
        deltas,
        rotate_by_quadrant=True,
        include_origin=False,
        recolor_origin=False,
    )
    _paint_origin_complement(out, bg, overlap_col, base_r, base_c, size, templ_cells)
    return out


def _render_d8_only_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    """Single-variant baseline: unrotated positives plus an origin complement."""
    out = _render_positive_copies(
        H,
        W,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        size,
        templ_cells,
        deltas,
        rotate_by_quadrant=False,
        include_origin=False,
        recolor_origin=False,
    )
    _paint_origin_complement(out, bg, overlap_col, base_r, base_c, size, templ_cells)
    return out


def _render_d14_only_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    """Single-variant baseline: quadrant rotations with a recolored positive origin."""
    return _render_positive_copies(
        H,
        W,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        size,
        templ_cells,
        deltas,
        rotate_by_quadrant=True,
        include_origin=True,
        recolor_origin=True,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated copies of one template, while rendering the origin as a negative image.

    Input structure:
    - One large monochrome component in color T is the canonical template, shown inside its own square bbox.
    - Separate singleton pixels form a uniform-spacing lattice; exactly one singleton also has color T and marks the origin.
    - All remaining singletons share color O, which will be reused in the special origin output.

    Rule:
    - Recover coarse lattice offsets from the singleton spacing, centered at the unique T-colored singleton.
    - For every non-origin offset, stamp the template translated by whole bbox sizes and rotated by quadrant: upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees.
    - The origin does not receive a normal template copy. Instead, inside the template bbox draw the complement of the canonical template in color O, leaving template cells as background.
    - If positive-shape copies were ever to overlap, the overlap color is O.

    Key insight:
    The singleton lattice now encodes two things at once: the unique T singleton identifies the one bbox that turns into a negative image, while every other singleton's diagonal quadrant determines which quarter-turn of the normal template should appear there.
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
    assert h == w
    size = int(h)

    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]
    templ_set = set(templ_rel)
    assert len(templ_set) < size * size

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
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    non_origin_total = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        if int(col) == int(templ_col):
            continue
        turns = _quadrant_turns(dr, dc)
        rot_cells = _rotate_cells(templ_rel, size, turns)
        off_r = int(r0 + dr * size)
        off_c = int(c0 + dc * size)
        for rr, cc in rot_cells:
            non_origin_total[int(off_r + rr), int(off_c + cc)] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[non_origin_total > 0] = int(templ_col)
    out[non_origin_total >= 2] = int(overlap_col)

    for rr in range(size):
        for cc in range(size):
            out[int(r0 + rr), int(c0 + cc)] = int(bg) if (rr, cc) in templ_set else int(overlap_col)

    return out


# --------------- generator ---------------

GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 150
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 500

BASE_DELTAS = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
EXTRA_DELTAS = [
    (-2, -1),
    (-1, -2),
    (-2, 1),
    (-1, 2),
    (2, -1),
    (1, -2),
    (2, 1),
    (1, 2),
]


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols box."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))

    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Random 4-connected polyomino that touches all four sides of the bbox."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _is_rotation_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    """Require four visually distinct quarter-turns inside the square bbox."""
    seen = {frozenset(_rotate_cells(cells, size, turns)) for turns in range(4)}
    return len(seen) == 4


def _is_complement_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    """Reject templates whose complement matches a positive rotation exactly."""
    comp = frozenset(_complement_cells(cells, size))
    return all(comp != frozenset(_rotate_cells(cells, size, turns)) for turns in range(4))


def _gen_template(size: int) -> list[tuple[int, int]] | None:
    """Square, spanning, rotationally asymmetric template with a visible complement."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, size, size)
        if cells is None:
            continue
        if not _is_rotation_distinct(cells, size):
            continue
        if not _is_complement_distinct(cells, size):
            continue
        return cells
    return None


def _gen_lattice() -> list[tuple[int, int]]:
    """All four diagonal quadrants are present; an optional sixth point repeats one quadrant."""
    deltas = list(BASE_DELTAS)
    if random.randint(*SINGLETON_COUNT_RANGE) == 6:
        deltas.append(random.choice(EXTRA_DELTAS))
    return deltas


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_dr, max_dr, min_dc, max_dc) for coarse lattice offsets."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    expected: np.ndarray,
) -> np.ndarray | None:
    """Place the singleton lattice away from the template and the eventual output marks."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)
    occupied = {tuple(map(int, pt)) for pt in np.argwhere(expected != int(bg))}

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * GUNIT)
        sr1 = int(orr + max_dr * GUNIT)
        sc0 = int(orc + min_dc * GUNIT)
        sc1 = int(orc + max_dc * GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        singleton_pixels = {(int(orr + dr * GUNIT), int(orc + dc * GUNIT)) for dr, dc in deltas}
        if singleton_pixels & occupied:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for dr, dc in deltas:
            rr = int(orr + dr * GUNIT)
            cc = int(orc + dc * GUNIT)
            grid[rr, cc] = int(templ_col if (dr, dc) == (0, 0) else overlap_col)

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Build the fused output first, then back out a valid input."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template(BOX_SIZE)
    if templ_cells is None:
        return None

    deltas = _gen_lattice()
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = -min_dr * BOX_SIZE
    base_r_hi = GRID_SIZE - max_dr * BOX_SIZE - BOX_SIZE
    base_c_lo = -min_dc * BOX_SIZE
    base_c_hi = GRID_SIZE - max_dc * BOX_SIZE - BOX_SIZE
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    expected = _render_combo_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        BOX_SIZE,
        templ_cells,
        deltas,
    )
    d8_only = _render_d8_only_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        BOX_SIZE,
        templ_cells,
        deltas,
    )
    d14_only = _render_d14_only_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        BOX_SIZE,
        templ_cells,
        deltas,
    )
    if np.array_equal(expected, d8_only) or np.array_equal(expected, d14_only):
        return None

    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        base_r,
        base_c,
        bg,
        templ_col,
        overlap_col,
        expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
