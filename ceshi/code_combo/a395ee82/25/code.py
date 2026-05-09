# Combo variant of a395ee82 (index 25)
# combo_spec: D10/0+D14/0
# Name: 象限旋转紧凑裁剪

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
# =====================================
# a) Concrete behavior and trace
#    - Keep the original input vocabulary: one large monochrome template in
#      color T plus a separated singleton layout with exactly one T-colored
#      origin and the remaining singletons in color O.
#    - The first change is geometric: each stamped copy rotates by quadrant
#      relative to the marked origin. We use the mapping upper-right -> 0
#      turns, upper-left -> 1 turn, lower-left -> 2 turns, lower-right -> 3
#      turns. The origin copy keeps 0 turns and is still recolored to O.
#    - The second change is output shape: after rendering all rotated copies on
#      the original 22x22 canvas, crop to the minimal bbox containing every
#      stamped pixel.
#    - Important interaction: if we kept the D14 square 4x4 template box, each
#      rotation would still occupy a shape touching all four sides of that box,
#      so the crop would usually collapse to the same outer square-box hull and
#      D10 would become visually weak. The combo therefore uses a non-square
#      4x3 canonical template. Its 90/270-degree rotations become 3x4, so the
#      occupied width/height of extreme stamps changes with quadrant and the
#      final crop really depends on rotation.
#    - Example trace with base deltas {(-1,-1), (-1,1), (1,-1), (1,1), (0,0)}:
#      the source task would stamp the same 4x3 template at offsets (-4,-3),
#      (-4,+3), (+4,-3), (+4,+3), (0,0) and return the full 22x22 canvas.
#      This combo instead rotates those four non-origin copies by quadrant, so
#      the left side may be set by a 4-wide rotated copy while the top side is
#      set by a 4-tall unrotated copy; only then do we tight-crop away the
#      empty border. The output size is therefore part of the rule, not just
#      the internal pixel pattern.
# b) Complexity calibration
#    - The rotation rule uses only four fixed categories, one per quadrant.
#      That keeps the hypothesis space small enough for ARC while still making
#      the transform visibly richer than the base task.
#    - Every generated lattice includes all four diagonal quadrants, so each
#      training example shows the full quadrant-to-rotation mapping rather than
#      hiding some cases. A sixth singleton, when present, repeats one quadrant
#      to provide redundancy without adding new categories.
#    - The crop is not optional noise: generator-side checks reject samples
#      where the rotated-and-cropped answer matches the no-rotation cropped
#      answer, so both D10 and D14 remain active.
# c) Hidden assumptions audit in main()
#    - `template = max(comps, key=area)` still works because the input contains
#      exactly one multi-cell component and all singletons remain isolated.
#    - `templ_col` from the template remains valid because the template is still
#      monochrome.
#    - Origin detection by `col == templ_col` still works because exactly one
#      singleton reuses T.
#    - The original solver assumed one immutable orientation and translated
#      `templ_rel` unchanged. That assumption is broken, so `main()` now rotates
#      the template per delta before plotting each copy.
#    - The original step `off_c = dc * w` is kept, but now `w` is the canonical
#      template width while some rotated stamps are wider. This is intentional:
#      copies are still anchored on the original coarse lattice, and the varying
#      occupied bbox is what makes the crop depend on rotation.
#    - Non-origin axis points would sit on quadrant boundaries and make the turn
#      assignment ambiguous. `main()` therefore asserts that every non-origin
#      delta has both coordinates nonzero, and the generator constructs only
#      diagonal-quadrant layouts.
#    - Returning the full `(H, W)` output is no longer correct. `main()` now
#      renders on the full canvas first, then crops to the foreground bbox.
# d) Placement legality
#    - The generator chooses anchors so every rotated stamp stays in-bounds even
#      though widths/heights alternate between 4x3 and 3x4.
#    - Candidate singleton clusters must stay disjoint from the visible input
#      template bbox with at least one background cell gap, and singleton pixels
#      are also kept away from every future stamped output pixel for cleanliness.
#    - Stamp overlap is checked explicitly from occupied output pixels; invalid
#      layouts are rejected before an input is returned.
# e) Visual observability
#    - The canonical template is forced to have four distinct quarter-turns, so
#      the quadrant mapping is human-visible from shape alone.
#    - Because all four quadrants appear in every sample, the solver can infer
#      the mapping by comparison across repeated copies rather than from a lone
#      exceptional case.
#    - Tight crop is visible because the output dimensions vary and because the
#      crop boundary shifts with which rotated copies sit on the outer edges.
# f) Visual cleanliness
#    - Output stamps are generated without overlap, so the rotated silhouettes
#      remain readable.
#    - The singleton cluster is separated from the template and future output
#      footprint, avoiding mixed cues in the input.
#    - Only three colors are used, as in the source task; the combo adds no
#      extra annotation objects.
# g) Change inferability
#    - Color semantics stay simple: T means template/non-origin stamp, O means
#      origin stamp and overlap color. No hidden dependence on color ids exists.
#    - The new inferential burden is geometric only: identify the unique origin,
#      read quadrant-conditioned rotation, then notice that the answer is the
#      minimal crop around the rotated stamp union.
# Post-edit assumption audit
#    - `main()` now assumes: largest component is the template; exactly one
#      singleton shares the template color; every non-origin singleton lies in a
#      strict diagonal quadrant; the template rotations are meaningful; and at
#      least one stamped pixel exists after rendering. `generate_input()` builds
#      inputs satisfying each condition and validates them by re-running `main()`.


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_once(
    cells: list[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """Rotate bbox-relative cells by one 90-degree CCW turn and renormalize."""
    rotated = [(int(c), int(height - 1 - r)) for r, c in cells]
    return rotated, int(width), int(height)


def _rotate_cells(
    cells: list[tuple[int, int]],
    height: int,
    width: int,
    turns: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """Rotate bbox-relative cells by `turns` quarter-turns inside their minimal bbox."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    cur_h, cur_w = int(height), int(width)
    for _ in range(turns):
        out, cur_h, cur_w = _rotate_once(out, cur_h, cur_w)
    return out, int(cur_h), int(cur_w)


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map an origin-relative lattice offset to the quadrant rotation.

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


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    height: int,
    width: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    *,
    use_quadrant_rotation: bool,
) -> np.ndarray:
    """Render the uncropped output on the original canvas."""
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if use_quadrant_rotation else 0
        rot_cells, _rot_h, _rot_w = _rotate_cells(templ_cells, height, width, turns)
        off_r = int(base_r + dr * height)
        off_c = int(base_c + dc * width)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in rot_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _foreground_bbox(grid != int(bg))
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated copies of a monochrome template from a singleton lattice, then tight-crop the union.

    Input structure:
    - The dominant color is background `bg`.
    - One large monochrome component is the template in color `T`, shown in its canonical orientation.
    - Separate singleton markers define a uniformly spaced lattice; exactly one singleton is also `T` and marks the origin, while the others are color `O`.

    Rule:
    - Recover coarse lattice offsets from the singleton spacing, using the unique `T` singleton as `(0, 0)`.
    - For each offset, place one translated template copy using the canonical template bbox height and width as the lattice step.
    - Rotate each copy by quadrant around the origin: upper-right `0°`, upper-left `90°`, lower-left `180°`, lower-right `270°`; recolor the origin copy to `O`, and recolor any overlapped pixels to `O`.
    - After rendering all copies on the full canvas, remove every all-background border row and column so the answer is the tight bounding box of the stamped figure.

    Key insight:
    - The unique singleton in the template color does double duty: it identifies the lattice origin and therefore tells you both which copy becomes `O` and how every other singleton falls into a rotation quadrant; the final output is then the tight crop of that rotated stamp arrangement, not the full 22x22 scene.
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

    raw_deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas: list[tuple[int, int]] = []
    for dr, dc, col in raw_deltas:
        sdr = int(dr // gunit)
        sdc = int(dc // gunit)
        if int(col) != int(templ_col):
            assert sdr != 0 and sdc != 0
        deltas.append((sdr, sdc))

    out = _render_output(
        H,
        W,
        bg,
        templ_col,
        overlap_col,
        r0,
        c0,
        h,
        w,
        templ_rel,
        deltas,
        use_quadrant_rotation=True,
    )
    return _crop_foreground(out, bg)


# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H = 4
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (8, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2
OUTPUT_MARGIN = 2
MAX_STAMP_EXTENT = max(TEMPLATE_H, TEMPLATE_W)

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 160
SINGLETON_RETRIES = 100
GENERATE_RETRIES = 500

BASE_DELTAS = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
EXTRA_DELTAS = [(-2, -1), (-2, 1), (2, -1), (2, 1)]


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


def _is_rotation_distinct(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    """Require four visually distinct quarter-turns after normalization."""
    seen = set()
    for turns in range(4):
        rot_cells, _rh, _rw = _rotate_cells(cells, h, w, turns)
        seen.add(frozenset(rot_cells))
    return len(seen) == 4


def _gen_template() -> list[tuple[int, int]] | None:
    """4x3, 4-connected, spanning, rotationally asymmetric monochrome template."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, TEMPLATE_H, TEMPLATE_W)
        if cells is None:
            continue
        if _is_rotation_distinct(cells, TEMPLATE_H, TEMPLATE_W):
            return cells
    return None


def _gen_lattice() -> list[tuple[int, int]]:
    """All four quadrants are present; optional sixth point repeats one quadrant safely."""
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


def _occupied_stamp_pixels(
    templ_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    deltas: list[tuple[int, int]],
    *,
    use_quadrant_rotation: bool,
) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if use_quadrant_rotation else 0
        rot_cells, _rh, _rw = _rotate_cells(templ_cells, TEMPLATE_H, TEMPLATE_W, turns)
        off_r = int(base_r + dr * TEMPLATE_H)
        off_c = int(base_c + dc * TEMPLATE_W)
        for rr, cc in rot_cells:
            pt = (int(off_r + rr), int(off_c + cc))
            if pt in occupied:
                return set()
            occupied.add(pt)
    return occupied


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
    """Place the singleton cluster away from the template and away from future stamps."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + TEMPLATE_H - 1)
    tc0, tc1 = int(base_c), int(base_c + TEMPLATE_W - 1)

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
    """Build output first, then back out a valid input with template + singleton lattice."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template()
    if templ_cells is None:
        return None

    deltas = _gen_lattice()
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = -min_dr * TEMPLATE_H
    base_r_hi = GRID_SIZE - MAX_STAMP_EXTENT - max_dr * TEMPLATE_H
    base_c_lo = -min_dc * TEMPLATE_W
    base_c_hi = GRID_SIZE - MAX_STAMP_EXTENT - max_dc * TEMPLATE_W
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    occupied = _occupied_stamp_pixels(
        templ_cells,
        base_r,
        base_c,
        deltas,
        use_quadrant_rotation=True,
    )
    if not occupied:
        return None

    expected_uncropped = _render_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        TEMPLATE_H,
        TEMPLATE_W,
        templ_cells,
        deltas,
        use_quadrant_rotation=True,
    )
    rr0, cc0, rr1, cc1 = _foreground_bbox(expected_uncropped != int(bg))
    if rr0 < OUTPUT_MARGIN or cc0 < OUTPUT_MARGIN:
        return None
    if rr1 > GRID_SIZE - 1 - OUTPUT_MARGIN or cc1 > GRID_SIZE - 1 - OUTPUT_MARGIN:
        return None
    expected = _crop_foreground(expected_uncropped, bg)
    if expected.shape == (GRID_SIZE, GRID_SIZE):
        return None

    plain_uncropped = _render_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        TEMPLATE_H,
        TEMPLATE_W,
        templ_cells,
        deltas,
        use_quadrant_rotation=False,
    )
    plain = _crop_foreground(plain_uncropped, bg)
    if expected.shape == plain.shape:
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
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
