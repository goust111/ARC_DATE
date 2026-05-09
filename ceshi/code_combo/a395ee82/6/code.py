# Combo variant of a395ee82 (index 6)
# combo_spec: D5/1+D7/1+D14/0
# Name: 八连通旋转消失

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# ---------------
# a) Concretization and behavior trace
#    - Input still has two clean parts: one large monochrome template in color T, plus a separated
#      singleton lattice with exactly one T-colored origin and 4-5 O-colored non-origin markers.
#    - The fused change is not "translate only" anymore:
#      1. Normalize singleton offsets by their common spacing gunit=2.
#      2. Those coarse offsets must form an 8-connected but not 4-connected lattice, so diagonal
#         adjacency is essential rather than optional.
#      3. Each offset chooses a quadrant-conditioned quarter-turn of the template:
#         upper-right 0, upper-left 90, lower-left 180, lower-right 270.
#      4. To make D5 visible, stamps use stride (box_size - 1) per axis instead of full box_size.
#         With diagonal neighbors this creates corner interference. Covered-once pixels stay visible
#         (origin-only -> O, non-origin-only -> T), while covered-2+ pixels disappear back to bg.
#    - Concrete example with size=4 and deltas {(-1,-1),(-1,1),(0,0),(1,-1),(1,1)}:
#      original task would place five unrotated copies on a 4-step grid and keep overlaps colored O.
#      Here the four diagonal copies are rotated by quadrant and shifted by 3, so each diagonal copy
#      meets the origin copy at one corner. If both rotated shapes occupy that corner, the corner cell
#      vanishes to bg instead of turning O. The output therefore shows four differently oriented stamps
#      around an O-colored center stamp, with corner notches erased.
# b) Complexity calibration
#    - Rotation is limited to exactly four fixed quadrant cases, and the lattice always includes all
#      four diagonal quadrants. This keeps the rule learnable while giving repeated evidence.
#    - The lattice family is restricted to a diagonal 5-point core plus one optional 8-neighbor extra
#      point. That is enough to expose 8-connectivity and extra interference without making placement
#      chaotic.
#    - The template is forced into a 4x4 square bbox with four distinct rotations and all four corners
#      occupied. Square boxes keep quarter-turn placement stable; occupied corners make interference
#      reliably visible under the reduced stride.
# c) Implicit assumption review
#    - Original `off_r = dr * h`, `off_c = dc * w` assumed non-overlapping full-box translations.
#      The fused variant breaks that: solver and generator both switch to stride size-1.
#    - Original solver reused one fixed `templ_rel` for every stamp. Now each non-origin stamp must use
#      `_rotate_cells(..., _quadrant_turns(dr, dc))`.
#    - Original code tolerated axis-aligned non-origin offsets. Quadrant rotation would be ambiguous on
#      axes, so the fused solver asserts every non-origin coarse delta has both coordinates nonzero.
#    - Original task never required lattice topology checks. The fused solver now explicitly requires
#      8-connectivity, strict interior origin, and at least one diagonal-only connection (`not 4-conn`).
# d) Placement legality
#    - All stamp anchors are chosen so every rotated 4x4 box stays inside 22x22 after using stride=3.
#    - The singleton-cluster bbox stays disjoint from the input template bbox with at least one bg gap.
#    - Singleton pixels are also kept off the entire eventual output footprint so the input remains clean.
# e) Visual observability
#    - Every sample shows all four quadrants, so all four rotation classes appear in one grid.
#    - The lattice is 8-connected but not 4-connected, making diagonal organization visible from the
#      singleton markers alone.
#    - Disappearance is visible because corner overlaps revert to bg rather than O; this is impossible
#      to confuse with the original "overlap becomes O" behavior once a few missing corner cells appear.
# f) Visual cleanliness
#    - The scene remains a two-object layout: one input template plus one compact singleton cluster.
#      Output complexity comes from repeated rotated copies, not from extra marker systems or colors.
#    - The generator rejects samples with too little surviving T/O area or too few canceled cells.
# g) Color inference
#    - Only three colors are used. T still means template / non-origin visible stamp pixels, O still
#      means the visible part of the origin stamp, and bg is both background and interference result.
#      No branch depends on numeric color identity.
# Post-edit assumption audit
#    - `main()` assumes: largest component is the template; the template bbox is square; there is exactly
#      one T-colored singleton origin; all other non-template components are singletons; normalized
#      singleton deltas are 8-connected but not 4-connected; non-origin deltas are strict diagonals; the
#      origin sits strictly inside the coarse lattice bbox. `generate_input()` constructs each of these.

GRID_SIZE = 22
BOX_SIZE = 4
STAMP_STEP = BOX_SIZE - 1
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2

MIN_CANCEL_CELLS = 4
MIN_ORIGIN_VISIBLE = 3
MIN_TEMPLATE_VISIBLE = 10

SINGLETON_RETRIES = 100
GENERATE_RETRIES = 600

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

BASE_DELTAS = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
EXTRA_DELTAS = [
    (-2, -1),
    (-1, -2),
    (-2, 1),
    (-1, 2),
    (1, -2),
    (2, -1),
    (1, 2),
    (2, 1),
]


def _dirs(connectivity: int) -> list[tuple[int, int]]:
    return DIR4 if int(connectivity) == 4 else DIR8


def _is_connected(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], connectivity: int) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in _dirs(connectivity):
            nxt = (int(r + dr), int(c + dc))
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _touches_all_edges(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], size: int) -> bool:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return min(rs) == 0 and max(rs) == size - 1 and min(cs) == 0 and max(cs) == size - 1


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map origin-relative coarse offsets to quadrant rotations.

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


def _coverage_maps(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    size: int,
    deltas: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return total coverage and origin-copy coverage under quadrant rotations and overlap stride."""
    H, W = map(int, shape)
    total = np.zeros((H, W), dtype=int)
    origin = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, size, _quadrant_turns(dr, dc))
        off_r = int(base_r + dr * STAMP_STEP)
        off_c = int(base_c + dc * STAMP_STEP)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in rot_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin[R, C] += 1
    return total, origin


def _render_output(bg: int, templ_col: int, overlap_col: int, total: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Render visible pixels; overlapped cells vanish to background."""
    out = np.full(total.shape, int(bg), dtype=int)
    out[(total == 1) & (origin == 0)] = int(templ_col)
    out[(total == 1) & (origin > 0)] = int(overlap_col)
    return out


def _variant_stats(
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> dict[str, int]:
    total, origin = _coverage_maps((GRID_SIZE, GRID_SIZE), 8, 8, templ_cells, BOX_SIZE, deltas)
    return {
        "cancel_cells": int(np.sum(total >= 2)),
        "origin_visible": int(np.sum((total == 1) & (origin > 0))),
        "templ_visible": int(np.sum((total == 1) & (origin == 0))),
        "footprint_cells": int(np.sum(total > 0)),
    }


def _occupied_output_pixels(
    templ_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, BOX_SIZE, _quadrant_turns(dr, dc))
        off_r = int(base_r + dr * STAMP_STEP)
        off_c = int(base_c + dc * STAMP_STEP)
        for rr, cc in rot_cells:
            pts.add((int(off_r + rr), int(off_c + cc)))
    return pts


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _is_rotation_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    seen = {frozenset(_rotate_cells(cells, size, turns)) for turns in range(4)}
    return len(seen) == 4


def _corners_filled(cells: list[tuple[int, int]], size: int) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    corners = {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}
    return corners <= pts


def _enumerate_template_candidates() -> list[list[tuple[int, int]]]:
    coords = [(r, c) for r in range(BOX_SIZE) for c in range(BOX_SIZE)]
    out: list[list[tuple[int, int]]] = []
    for area in range(TEMPLATE_AREA_RANGE[0], TEMPLATE_AREA_RANGE[1] + 1):
        for cells in itertools.combinations(coords, area):
            if not _touches_all_edges(cells, BOX_SIZE):
                continue
            if not _is_connected(cells, connectivity=4):
                continue
            cand = [(int(r), int(c)) for r, c in cells]
            if not _corners_filled(cand, BOX_SIZE):
                continue
            if not _is_rotation_distinct(cand, BOX_SIZE):
                continue
            stats = _variant_stats(cand, BASE_DELTAS)
            if stats["cancel_cells"] < MIN_CANCEL_CELLS:
                continue
            if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
                continue
            out.append(cand)
    return out


def _enumerate_lattice_candidates() -> dict[int, list[list[tuple[int, int]]]]:
    out: dict[int, list[list[tuple[int, int]]]] = {5: [list(BASE_DELTAS)], 6: []}
    for extra in EXTRA_DELTAS:
        deltas = list(BASE_DELTAS) + [tuple(map(int, extra))]
        rs = [int(r) for r, _c in deltas]
        cs = [int(c) for _r, c in deltas]
        if not (min(rs) < 0 < max(rs) and min(cs) < 0 < max(cs)):
            continue
        if any((dr, dc) != (0, 0) and (dr == 0 or dc == 0) for dr, dc in deltas):
            continue
        if not _is_connected(deltas, connectivity=8):
            continue
        if _is_connected(deltas, connectivity=4):
            continue
        out[6].append(deltas)
    return out


TEMPLATE_CANDIDATES = _enumerate_template_candidates()
LATTICE_CANDIDATES = _enumerate_lattice_candidates()
assert TEMPLATE_CANDIDATES
assert LATTICE_CANDIDATES[5]
assert LATTICE_CANDIDATES[6]


def _build_lattice_schedule() -> list[list[tuple[int, int]]]:
    """
    Interleave 5-point and 6-point lattice families so repeated generation covers every branch.

    A pure random chooser is unbiased in expectation, but over only 50 samples it can still miss one
    of the 6-point extra branches. Cycling a fixed interleaving keeps branch coverage balanced and
    makes the fused behavior easier to audit from a small batch of generated examples.
    """
    base = [tuple(map(int, cell)) for cell in BASE_DELTAS]
    schedule: list[list[tuple[int, int]]] = []
    for deltas in LATTICE_CANDIDATES[6]:
        schedule.append(list(base))
        schedule.append([tuple(map(int, cell)) for cell in deltas])
    return schedule


LATTICE_SCHEDULE = _build_lattice_schedule()
_lattice_schedule_index = 0


def _next_lattice_branch() -> list[tuple[int, int]]:
    global _lattice_schedule_index
    branch = LATTICE_SCHEDULE[_lattice_schedule_index % len(LATTICE_SCHEDULE)]
    _lattice_schedule_index += 1
    return [tuple(map(int, cell)) for cell in branch]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated template copies from an 8-neighbor singleton lattice, erasing overlaps.

    Input structure:
    - The background is uniform. One large monochrome component in color T is the canonical template.
    - Elsewhere are 5-6 isolated singleton pixels on a fixed-step coarse lattice; exactly one singleton
      is also T and marks the lattice origin, while the others are color O.
    - After dividing singleton offsets by the common spacing, the coarse lattice is 8-connected, not
      4-connected, and every non-origin marker lies in a strict diagonal quadrant of the origin.

    Rule:
    - Recover coarse offsets from the unique T-colored singleton and use them as stamp anchors.
    - Translate by a reduced stride of one less than the square template size so neighboring diagonal
      stamps can overlap, and rotate each stamp by quadrant: upper-right 0, upper-left 90, lower-left
      180, lower-right 270.
    - Pixels covered only by the origin copy become O; pixels covered only by non-origin copies stay T.
    - Any pixel covered by two or more rotated copies disappears to the background color.

    Key insight: The singleton lattice now encodes three things at once: diagonal 8-neighbor topology,
    quadrant-conditioned template orientation, and destructive interference where rotated copies overlap.
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
        assert int(comp["area"]) == 1
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert SINGLETON_COUNT_RANGE[0] <= len(singletons) <= SINGLETON_COUNT_RANGE[1]

    origin = [(r, c) for r, c, col in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for _r, _c, col in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas_pix = [(int(r - orr), int(c - orc), int(col)) for r, c, col in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _col in deltas_pix for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas_pix]

    coarse_cells = [(int(dr), int(dc)) for dr, dc, _col in deltas]
    assert (0, 0) in coarse_cells
    assert _is_connected(coarse_cells, connectivity=8)
    assert not _is_connected(coarse_cells, connectivity=4)
    rs = [int(dr) for dr, _dc in coarse_cells]
    cs = [int(dc) for _dr, dc in coarse_cells]
    assert min(rs) < 0 < max(rs)
    assert min(cs) < 0 < max(cs)
    for dr, dc in coarse_cells:
        if (dr, dc) == (0, 0):
            continue
        assert dr != 0 and dc != 0

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        rot_cells = _rotate_cells(templ_rel, size, _quadrant_turns(dr, dc))
        off_r = int(r0 + dr * (size - 1))
        off_c = int(c0 + dc * (size - 1))
        is_origin = int(col) == int(templ_col)
        for rr, cc in rot_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    return _render_output(bg, templ_col, overlap_col, total, origin_cover)


def _choose_template() -> list[tuple[int, int]]:
    return list(random.choice(TEMPLATE_CANDIDATES))


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
    """Place the singleton cluster away from the input template and future stamp footprint."""
    orr_lo = int(-min_dr * GUNIT)
    orr_hi = int(GRID_SIZE - 1 - max_dr * GUNIT)
    orc_lo = int(-min_dc * GUNIT)
    orc_hi = int(GRID_SIZE - 1 - max_dc * GUNIT)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)
    occupied = _occupied_output_pixels(templ_cells, base_r, base_c, deltas)

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

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for dr, dc in deltas:
            rr = int(orr + dr * GUNIT)
            cc = int(orc + dc * GUNIT)
            grid[rr, cc] = int(templ_col if (dr, dc) == (0, 0) else overlap_col)

        try:
            produced = main(grid)
        except Exception:
            continue
        if np.array_equal(produced, expected):
            return grid

    return None


def _gen_attempt(deltas: list[tuple[int, int]]) -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _choose_template()
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = int(-min_dr * STAMP_STEP)
    base_r_hi = int(GRID_SIZE - max_dr * STAMP_STEP - BOX_SIZE)
    base_c_lo = int(-min_dc * STAMP_STEP)
    base_c_hi = int(GRID_SIZE - max_dc * STAMP_STEP - BOX_SIZE)
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    total, origin = _coverage_maps((GRID_SIZE, GRID_SIZE), base_r, base_c, templ_cells, BOX_SIZE, deltas)
    stats = {
        "cancel_cells": int(np.sum(total >= 2)),
        "origin_visible": int(np.sum((total == 1) & (origin > 0))),
        "templ_visible": int(np.sum((total == 1) & (origin == 0))),
    }
    if stats["cancel_cells"] < MIN_CANCEL_CELLS:
        return None
    if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
        return None
    if stats["templ_visible"] < MIN_TEMPLATE_VISIBLE:
        return None

    expected = _render_output(bg, templ_col, overlap_col, total, origin)
    return _place_singletons(
        deltas=deltas,
        min_dr=min_dr,
        max_dr=max_dr,
        min_dc=min_dc,
        max_dc=max_dc,
        templ_cells=templ_cells,
        base_r=base_r,
        base_c=base_c,
        bg=bg,
        templ_col=templ_col,
        overlap_col=overlap_col,
        expected=expected,
    )


def generate_input() -> np.ndarray:
    deltas = _next_lattice_branch()
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(deltas)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
