# Combo variant of a395ee82 (index 20)
# combo_spec: D11/0+D14/0
# Name: 重心基准象限旋转

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
# -------------------------------------
# a) Concrete behavior and trace
#    - Input still has two object families:
#      1. one large monochrome template component in color T;
#      2. a separated singleton lattice with one T-colored origin and the rest O-colored.
#    - Original task:
#        each lattice delta (dr,dc) produced a plain translated copy with top-left
#        at (r0 + dr*h, c0 + dc*w).
#    - D11 alone changed the anchor:
#        (r0 + dr*h, c0 + dc*w) became the template centroid position, so the copy
#        top-left had to be shifted by the template centroid.
#    - D14 alone changed the orientation:
#        each diagonal quadrant picked a quarter-turn, but the placement anchor
#        remained the template bbox top-left.
#    - This combo must do both in sequence:
#        1. use the lattice delta to choose a quadrant turn;
#        2. rotate the template inside its square bbox;
#        3. compute that rotated version's own pixel centroid;
#        4. treat (r0 + dr*size, c0 + dc*size) as that centroid's target location;
#        5. round back to an integer top-left before stamping.
#    - Example with a 4x4 template:
#        suppose the canonical local centroid is (1.75, 1.00).
#        The 90-degree version then has centroid (1.00, 1.25).
#        For lattice delta (-1,-1), old D14 top-left would be (r0-4, c0-4),
#        but the combo top-left becomes
#        (round(r0-4-1.00), round(c0-4-1.25)) = (r0-5, c0-5).
#        So rotation changes both the silhouette and the offset correction.
# b) Complexity calibration
#    - The rotation rule stays low-branching: exactly four quadrant categories.
#    - The generator always includes all four diagonal quadrants plus the origin,
#      and optionally one extra diagonal point, so the mapping is repeated inside
#      each sample instead of being a one-off coincidence.
#    - To make the fusion visible, the template is forced to live in a 4x4 square
#      bbox, be rotationally asymmetric, and have quadrant-dependent centroid shifts
#      after rounding; otherwise the sample is rejected as visually degenerate.
# c) Hidden assumption review against the original solver
#    - `template = max(comps, key=area)`:
#      still valid because generator keeps exactly one large component and the rest
#      are singletons.
#    - `templ_rel = coords - bbox_topleft`:
#      still the right canonical representation, but it can no longer be stamped
#      directly for every delta. Each delta must first choose a rotation.
#    - `off_r = dr * h`, `off_c = dc * w`:
#      these are no longer final top-left shifts. They now define integer anchor
#      positions for the rotated centroid, so an additional centroid-dependent
#      correction is required.
#    - fixed template bbox dimensions:
#      once 90-degree turns are introduced, a non-square bbox would swap h and w
#      under rotation and make the original spacing invalid. The combo therefore
#      assumes a square bbox and the generator enforces `h == w == 4`.
#    - quadrant lookup:
#      non-origin lattice points may not lie on an axis, otherwise "which quadrant"
#      would be ambiguous. The generator only emits strict diagonal deltas and
#      `main()` asserts that assumption.
#    - output recoloring:
#      the original origin-copy recolor and overlap recolor rules remain intact.
# d) Placement legality
#    - Generator enumerates template origins only where every rotated-centroid stamp
#      stays fully inside the 22x22 grid.
#    - It rejects any candidate where two output stamps overlap.
#    - It also keeps the input singleton cluster bbox disjoint from the input
#      template bbox with at least one background cell gap, and forbids singleton
#      pixels from landing on any future output stamp pixel.
# e) Visual observability
#    - A solver can still read the scene as "template + marked lattice", but now
#      must infer two coupled facts from the output:
#      1. quadrant controls the turn;
#      2. copies are not aligned by bbox top-left, but by the rotated shape centroid.
#    - Because all four diagonal quadrants appear in each sample, the quadrant map
#      can be inferred from repeated evidence rather than one isolated example.
# f) Visual cleanliness
#    - The output contains a small set of non-overlapping stamps, so each rotated
#      silhouette remains legible and the centroid shifts are not hidden by large
#      overlap regions.
#    - Colors stay at the original three-color budget: bg, T, O.
# g) Change inferability
#    - No rule depends on color ids or other human-invisible ordering.
#    - Generator rejects cases where the combo output collapses to the original,
#      centroid-only, or rotation-only output, ensuring both control dimensions
#      affect the visible result.
# Post-edit assumption audit
#    - `main()` assumes:
#      * the largest component is the template;
#      * the template bbox is square;
#      * the template has one visible color T;
#      * exactly one singleton shares that color and marks the origin;
#      * every non-origin singleton has nonzero row and column offset from the origin;
#      * all non-origin singletons share a common color O.
#    - `generate_input()` constructs inputs that satisfy each assumption directly.


def _round_nearest(x: float) -> int:
    """Round to nearest integer using half-up behavior for positive half cases."""
    return int(math.floor(float(x) + 0.5))


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _template_centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map an origin-relative lattice offset to a quadrant rotation.

    Upper-right -> 0, upper-left -> 1, lower-left -> 2, lower-right -> 3.
    The origin keeps the canonical orientation.
    """
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


def _rotation_cache(templ_cells: list[tuple[int, int]], size: int):
    cache: dict[int, tuple[list[tuple[int, int]], float, float]] = {}
    for turns in range(4):
        rot_cells = _rotate_cells(templ_cells, size, turns)
        centroid_r, centroid_c = _template_centroid(rot_cells)
        cache[turns] = (rot_cells, centroid_r, centroid_c)
    return cache


def _stamp_shift(
    base_r: int,
    base_c: int,
    dr: int,
    dc: int,
    size: int,
    centroid_r: float,
    centroid_c: float,
    *,
    use_centroid: bool,
) -> tuple[int, int]:
    anchor_r = int(base_r + dr * size)
    anchor_c = int(base_c + dc * size)
    if not use_centroid:
        return anchor_r, anchor_c
    return _round_nearest(anchor_r - centroid_r), _round_nearest(anchor_c - centroid_c)


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    rotations: dict[int, tuple[list[tuple[int, int]], float, float]],
    deltas: list[tuple[int, int]],
    *,
    use_centroid: bool,
    use_quadrant_rotation: bool,
) -> np.ndarray:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if use_quadrant_rotation else 0
        rot_cells, centroid_r, centroid_c = rotations[turns]
        shift_r, shift_c = _stamp_shift(
            base_r,
            base_c,
            dr,
            dc,
            size,
            centroid_r,
            centroid_c,
            use_centroid=use_centroid,
        )
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in rot_cells:
            R = int(shift_r + rr)
            C = int(shift_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _occupied_stamp_pixels(
    base_r: int,
    base_c: int,
    size: int,
    rotations: dict[int, tuple[list[tuple[int, int]], float, float]],
    deltas: list[tuple[int, int]],
    *,
    use_centroid: bool,
    use_quadrant_rotation: bool,
) -> set[tuple[int, int]] | None:
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if use_quadrant_rotation else 0
        rot_cells, centroid_r, centroid_c = rotations[turns]
        shift_r, shift_c = _stamp_shift(
            base_r,
            base_c,
            dr,
            dc,
            size,
            centroid_r,
            centroid_c,
            use_centroid=use_centroid,
        )
        for rr, cc in rot_cells:
            pt = (int(shift_r + rr), int(shift_c + cc))
            if not (0 <= pt[0] < GRID_SIZE and 0 <= pt[1] < GRID_SIZE):
                return None
            if pt in occupied:
                return None
            occupied.add(pt)
    return occupied


def _solver_output(
    grid: np.ndarray,
    *,
    use_centroid: bool,
    use_quadrant_rotation: bool,
) -> np.ndarray:
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
    rotations = _rotation_cache(templ_rel, size)

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

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        turns = _quadrant_turns(dr, dc) if use_quadrant_rotation else 0
        rot_cells, centroid_r, centroid_c = rotations[turns]
        shift_r, shift_c = _stamp_shift(
            r0,
            c0,
            dr,
            dc,
            size,
            centroid_r,
            centroid_c,
            use_centroid=use_centroid,
        )
        is_origin = int(col) == int(templ_col)
        for rr, cc in rot_cells:
            R = int(shift_r + rr)
            C = int(shift_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated copies of a monochrome template, but align each copy by the rotated template centroid.

    Input structure:
    - The background is one dominant color.
    - One large monochrome component in color T is the canonical template.
    - Elsewhere, singleton pixels form a uniformly spaced lattice: exactly one singleton is color T and the others are color O.

    Rule:
    - Use the unique T-colored singleton as the lattice origin and convert all singleton displacements into coarse lattice deltas.
    - For each delta, choose a quarter-turn by quadrant around the origin: upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees; the origin keeps 0 degrees.
    - Turn `(r0 + dr*size, c0 + dc*size)` into the target location of that rotated stamp's pixel centroid, then round back to an integer top-left and place the whole rotated copy there.
    - Paint the origin copy in O instead of T, and paint every overlapped pixel in O as well.

    Key insight:
    - The singleton lattice controls both which copies appear and how they are oriented, but the placement anchor is no longer the bbox corner; it is the centroid of the quadrant-selected rotated shape.
    """
    return _solver_output(grid, use_centroid=True, use_quadrant_rotation=True)


# --------------- generator ---------------

GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 160
SINGLETON_RETRIES = 100
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
    seen = set()
    for turns in range(4):
        seen.add(frozenset(_rotate_cells(cells, size, turns)))
    return len(seen) == 4


def _has_quadrant_dependent_centroid_shift(cells: list[tuple[int, int]], size: int) -> bool:
    offsets = set()
    for turns in range(4):
        rot_cells = _rotate_cells(cells, size, turns)
        centroid_r, centroid_c = _template_centroid(rot_cells)
        offsets.add((_round_nearest(-centroid_r), _round_nearest(-centroid_c)))
    return len(offsets) >= 2


def _gen_template(size: int) -> list[tuple[int, int]] | None:
    """Square, 4-connected, spanning, asymmetric template with visible combo shifts."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, size, size)
        if cells is None:
            continue
        if not _is_rotation_distinct(cells, size):
            continue
        if not _has_quadrant_dependent_centroid_shift(cells, size):
            continue
        return cells
    return None


def _gen_lattice() -> list[tuple[int, int]]:
    deltas = list(BASE_DELTAS)
    if random.randint(*SINGLETON_COUNT_RANGE) == 6:
        deltas.append(random.choice(EXTRA_DELTAS))
    return deltas


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _valid_template_origins(
    deltas: list[tuple[int, int]],
    rotations: dict[int, tuple[list[tuple[int, int]], float, float]],
    size: int,
) -> tuple[list[int], list[int]]:
    valid_r0: list[int] = []
    valid_c0: list[int] = []

    for r0 in range(GRID_SIZE - size + 1):
        ok = True
        for dr, dc in deltas:
            turns = _quadrant_turns(dr, dc)
            _rot_cells, centroid_r, _centroid_c = rotations[turns]
            shift_r = _round_nearest(r0 + dr * size - centroid_r)
            if shift_r < 0 or shift_r + size - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_r0.append(r0)

    for c0 in range(GRID_SIZE - size + 1):
        ok = True
        for dr, dc in deltas:
            turns = _quadrant_turns(dr, dc)
            _rot_cells, _centroid_r, centroid_c = rotations[turns]
            shift_c = _round_nearest(c0 + dc * size - centroid_c)
            if shift_c < 0 or shift_c + size - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_c0.append(c0)

    return valid_r0, valid_c0


def _variant_diffs(grid: np.ndarray) -> tuple[int, int, int]:
    combo = main(grid)
    centroid_only = _solver_output(grid, use_centroid=True, use_quadrant_rotation=False)
    rotation_only = _solver_output(grid, use_centroid=False, use_quadrant_rotation=True)
    original = _solver_output(grid, use_centroid=False, use_quadrant_rotation=False)
    diff_vs_centroid = int(np.count_nonzero(combo != centroid_only))
    diff_vs_rotation = int(np.count_nonzero(combo != rotation_only))
    diff_vs_original = int(np.count_nonzero(combo != original))
    return diff_vs_centroid, diff_vs_rotation, diff_vs_original


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
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Place the singleton lattice away from the input template and future stamp pixels."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)

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
        if not np.array_equal(out, expected):
            continue

        diff_vs_centroid, diff_vs_rotation, diff_vs_original = _variant_diffs(grid)
        if diff_vs_centroid < 4:
            continue
        if diff_vs_rotation < 4:
            continue
        if diff_vs_original < max(6, len(templ_cells)):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Build the combo output first, then back out a valid input grid."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template(BOX_SIZE)
    if templ_cells is None:
        return None
    rotations = _rotation_cache(templ_cells, BOX_SIZE)

    deltas = _gen_lattice()
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    valid_r0, valid_c0 = _valid_template_origins(deltas, rotations, BOX_SIZE)
    if not valid_r0 or not valid_c0:
        return None

    for _ in range(80):
        base_r = random.choice(valid_r0)
        base_c = random.choice(valid_c0)
        occupied = _occupied_stamp_pixels(
            base_r,
            base_c,
            BOX_SIZE,
            rotations,
            deltas,
            use_centroid=True,
            use_quadrant_rotation=True,
        )
        if occupied is None:
            continue

        expected = _render_output(
            GRID_SIZE,
            GRID_SIZE,
            bg,
            templ_col,
            overlap_col,
            base_r,
            base_c,
            BOX_SIZE,
            rotations,
            deltas,
            use_centroid=True,
            use_quadrant_rotation=True,
        )
        grid = _place_singletons(
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
            occupied,
        )
        if grid is not None:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
