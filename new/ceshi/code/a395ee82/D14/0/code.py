# Variant of a395ee82
# Dimension: D14 — stamp几何变换 (transform)
# Change: 所有stamp使用相同朝向(无旋转/翻转)，仅有平移和重着色... → 晶格象限决定旋转

from __future__ import annotations

import math
import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior
#    - Keep the original two-part input: one large monochrome template component
#      plus a separated singleton lattice with exactly one T-colored origin and
#      the rest colored O.
#    - Change only the stamping transform: the lattice offset still selects where
#      a copy goes, but the copy's orientation now depends on which quadrant the
#      offset occupies relative to the origin.
#    - To make 90-degree turns physically well defined, use a square 4x4 template
#      bbox. The template itself stays monochrome, but it must be rotationally
#      asymmetric so the quadrant-conditioned turns are visible.
#    - Example: if a singleton sits at (-1,+1) from the origin, its stamp keeps
#      the input orientation; at (-1,-1) it becomes the 90-degree CCW version;
#      at (+1,-1) it becomes the 180-degree version; at (+1,+1) it becomes the
#      270-degree version. The origin stamp remains at 0 degrees and is recolored O.
# b) Complexity calibration
#    - The rotation rule uses exactly four fixed categories, one per quadrant,
#      instead of arbitrary per-stamp transforms.
#    - Every generated lattice contains all four diagonal quadrants, with an
#      optional sixth point repeating one quadrant. This gives repeated visual
#      support without exploding the hypothesis space.
# c) Hidden assumptions from the original solver
#    - Original code assumed one immutable template orientation and translated
#      `templ_rel` unchanged for every lattice point. That is no longer valid.
#    - Original `off_r = dr * h` and `off_c = dc * w` also relied on a fixed bbox.
#      After introducing 90-degree rotation, a non-square bbox would swap width
#      and height and break placement. The generator therefore guarantees `h == w`.
#    - Axis-aligned singleton offsets would sit on quadrant boundaries and make
#      the rule ambiguous. The generator explicitly avoids non-origin axis points,
#      and `main()` asserts that assumption instead of silently picking a side.
# d) Placement legality
#    - All stamps live in disjoint 4x4 boxes whose anchors are spaced by integer
#      multiples of the same box size, so rotated copies cannot overlap by accident.
#    - The singleton cluster bbox remains disjoint from the template bbox with at
#      least one background cell gap, as in the original task.
# e) Visual observability
#    - The template is forced to have four distinct rotations, so the quadrant
#      mapping is visible from the output alone.
#    - Each sample exposes all four quadrant cases; the extra singleton, when
#      present, repeats one of them to provide redundancy.
# f) Visual cleanliness
#    - No output-stamp overlap is generated, so the scene stays readable: one
#      template, one small singleton cluster, and a clean array of rotated copies.
# g) Color inference
#    - Colors keep the original semantics: T is the template/stamp color, O is
#      both the origin-copy color and the overlap color. No color-id ordering is used.
# Post-edit assumption audit
#    - `main()` now assumes: largest component is the template, template bbox is
#      square, exactly one singleton matches template color, every non-origin
#      singleton lies in a strict diagonal quadrant, and all non-origin singletons
#      share one visible color O. `generate_input()` constructs inputs that satisfy
#      each of those constraints directly.


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map an origin-relative lattice offset to the quadrant rotation.

    Right-up -> 0, left-up -> 1, left-down -> 2, right-down -> 3.
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
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    """Render the expected output from a template and coarse lattice deltas."""
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, size, _quadrant_turns(dr, dc))
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of a monochrome template using a singleton lattice with a marked origin.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome component is the template (color T), shown in its canonical orientation.
    - A separated singleton lattice marks copy locations: exactly one singleton is color T (the origin), and the others are color O.

    Rule:
    - Recover coarse lattice offsets from the singleton spacing, using the T-colored singleton as (0,0).
    - Place one template copy per offset, translating by whole template-box widths/heights as before.
    - Rotate each non-origin copy by quadrant around the origin: upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees.
    - Recolor the origin copy to O, and recolor any overlapped pixels to O as well.

    Key insight: The singleton layout does not merely tell you where to stamp; after you identify the unique origin, each offset's diagonal quadrant determines which quarter-turn of the template should appear there.
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
        rot_cells = _rotate_cells(templ_rel, size, _quadrant_turns(dr, dc))
        off_r = int(dr * size)
        off_c = int(dc * size)
        is_origin = int(col) == int(templ_col)
        for rr, cc in rot_cells:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator ---------------

GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 120
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 400

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
    seen = set()
    for turns in range(4):
        rot = frozenset(_rotate_cells(cells, size, turns))
        seen.add(rot)
    return len(seen) == 4


def _gen_template(size: int) -> list[tuple[int, int]] | None:
    """Square, 4-connected, spanning, rotationally asymmetric monochrome template."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, size, size)
        if cells is None:
            continue
        if _is_rotation_distinct(cells, size):
            return cells
    return None


def _gen_lattice() -> list[tuple[int, int]]:
    """All four diagonal quadrants are present; an optional sixth point repeats one quadrant."""
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
    size: int,
    base_r: int,
    base_c: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """All occupied output pixels after stamping the rotated copies."""
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, size, _quadrant_turns(dr, dc))
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
        for rr, cc in rot_cells:
            occupied.add((int(off_r + rr), int(off_c + cc)))
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
) -> np.ndarray | None:
    """Place the singleton cluster away from the template and away from output-stamp pixels."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)
    occupied = _occupied_stamp_pixels(templ_cells, BOX_SIZE, base_r, base_c, deltas)

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

    expected = _render_output(
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
