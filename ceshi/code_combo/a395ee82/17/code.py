# Combo variant of a395ee82 (index 17)
# combo_spec: D3/1+D14/1
# Name: 孔洞模板方向翻转

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg


# Design analysis
# ---------------
# a) Concrete behavior / trace:
# - Preserve the original backbone: a 22x22 grid contains one large template component
#   in color T and a detached singleton lattice with one T-colored origin and the rest
#   colored O. Output still stamps one copy per lattice delta using bbox-sized steps.
# - D3 changes what is being copied: the template bbox now contains 1-2 enclosed bg
#   pixels, so stamping must use the exact filled mask and leave holes empty.
# - D14 changes how each copy is oriented: if a lattice point is left of the origin
#   (dc < 0), mirror horizontally; if it is above (dr < 0), mirror vertically; if it
#   is upper-left, apply both mirrors.
# - Concrete 4x3 example:
#     ###
#     #.#
#     ###
#     #..
#   Old task: delta (-1, 0) copies the same shape upward.
#   Fused task: delta (-1, 0) copies
#     #..
#     ###
#     #.#
#     ###
#   because the stamp is vertically flipped and the hole remains a hole. Likewise
#   delta (0, -1) flips left-right and delta (-1, -1) flips both axes.
#
# b) Complexity calibration:
# - A holey template alone is not enough if it is reflection-symmetric; the flip rule
#   would become invisible. The generator therefore keeps only templates whose filled
#   masks change under H, V, and HV reflection, and in practice all four orientations
#   are distinct.
# - A flip rule alone is not enough if samples omit left / up / upper-left witnesses.
#   The lattice is therefore constrained so every sample contains those three cases plus
#   at least one right-side and one lower witness for contrast.
# - Hole count is limited to 1 or 2 and lattice size is fixed at 6, keeping the number
#   of latent cases small enough for 2-4 ARC examples to teach the rule.
#
# c) Implicit-assumption audit of the original main():
# - Original `find_components(..., monochromatic=False)` merged all foreground into one
#   color-agnostic component family. With hole-aware reasoning we instead want the large
#   monochrome T component plus area-1 singleton components, so this variant uses
#   `monochromatic=True`.
# - Original `templ_rel` was reusable for every stamp. That assumption breaks once each
#   delta can request a different reflection, so we precompute four transformed masks.
# - Original code only needed foreground coords. In the fused task that is still correct
#   for painting, but only after asserting the bbox contains enclosed bg; otherwise the
#   generator could silently collapse back to the non-hole family.
# - Original placement math using multiples of template height/width still holds because
#   reflections stay inside the same bbox.
# - Coverage counting remains necessary: origin recoloring and overlap recoloring still
#   cannot be implemented safely by sequential painting.
#
# d) Placement legality:
# - Every transformed copy stays inside an h x w bbox anchored at the same translated
#   top-left, so in-bounds placement is checked using the old bbox stepping logic.
# - The generator computes occupied output pixels using the reflected masks and keeps the
#   singleton cluster disjoint from both the template bbox and all stamped pixels.
# - Because lattice offsets differ by integer bbox steps, copy bboxes do not overlap; we
#   still compute actual occupied pixels as a guard against implementation drift.
#
# e) Visual observability:
# - Each sample visibly contains a template hole, so the solver must distinguish interior
#   void from outside background.
# - Each sample also contains left, up, and upper-left lattice witnesses around the same
#   origin, making the sign-to-flip mapping directly observable.
# - No hidden ordering by color id or coordinate sort is used; all cues are visible in
#   geometry and in the unique T singleton that marks the origin.
#
# f) Visual cleanliness:
# - The grid still contains only two object families: one large template and isolated
#   singletons. The singleton cluster is separated from the template by at least one bg
#   cell, and output stamp bboxes do not overlap, so silhouettes remain readable.
# - Using a 3x3 lattice extent with 6 cells gives enough directional evidence without
#   filling the board with too many copies.
#
# g) Change inferability:
# - Color logic stays minimal and structured: T marks the template and non-origin copies,
#   O marks the origin copy and any overlaps. Geometry carries the new burden of
#   inference, so generator constraints ensure repeated evidence for both "holes persist"
#   and "negative deltas trigger reflections".


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bg cells inside the bbox that are not 4-connected to the bbox edge."""
    h, w = filled_mask.shape
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if touches_edge:
            continue
        holes[coords[:, 0], coords[:, 1]] = True
    return holes


def _transform_cells(
    cells: list[tuple[int, int]], h: int, w: int, flip_v: bool, flip_h: bool
) -> list[tuple[int, int]]:
    """Reflect relative template cells inside the template bbox."""
    out: list[tuple[int, int]] = []
    for rr, cc in cells:
        nr = h - 1 - rr if flip_v else rr
        nc = w - 1 - cc if flip_h else cc
        out.append((int(nr), int(nc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp reflected copies of a holey template using a singleton lattice as the guide.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome template of color T appears once; its bbox is 3-4 by 3 and
      contains 1-2 enclosed bg holes.
    - Elsewhere, isolated singleton pixels form a uniform-spacing lattice; exactly one
      singleton is color T and marks the lattice origin, while the others are color O.

    Rule:
    - Read singleton offsets relative to the unique T-colored origin and normalize them
      by the lattice spacing.
    - Convert each lattice offset into a translation scaled by the template bbox height
      and width.
    - Stamp only the template's filled pixels, preserving holes as bg.
    - For stamps left of the origin, flip horizontally; for stamps above the origin,
      flip vertically; for stamps in the upper-left region, apply both flips.
    - Paint the origin copy in O instead of T, and recolor any overlap area to O.

    Key insight: The lattice still determines where copies go, but the sign of each
    delta also selects which reflected version of the holey template to place.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col)
    hole_pixels = int(_enclosed_bg_mask(templ_mask).sum())
    assert 1 <= hole_pixels <= 2

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]
    transformed = {
        (flip_v, flip_h): _transform_cells(templ_rel, h, w, flip_v, flip_h)
        for flip_v in (False, True)
        for flip_h in (False, True)
    }

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

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
        off_r = int(dr * h)
        off_c = int(dc * w)
        stamp_cells = transformed[(int(dr) < 0, int(dc) < 0)]
        is_origin = int(col) == int(templ_col)
        for rr, cc in stamp_cells:
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
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)
SINGLETON_COUNT_RANGE = (6, 6)
LATTICE_EXTENT_RANGE = (3, 3)
GUNIT = 2

SPANNING_RETRIES = 120
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 500


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _cell_set(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
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
    """Polyomino of n cells that spans exactly rows x cols (touches all 4 edges)."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _template_candidates(h: int, w: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate legal holey templates whose reflections are all visibly different."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    grouped: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}

    for bits in range(1, 1 << total):
        cells = [(idx // w, idx % w) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < lo or area > hi:
            continue

        mask = _cells_to_mask(cells, h, w)
        comps = components_mask(mask, connectivity=4)
        if len(comps) != 1:
            continue
        if not mask[0, :].any() or not mask[h - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, w - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels not in grouped:
            continue

        base = _cell_set(cells)
        variants = {
            frozenset(_transform_cells(cells, h, w, flip_v, flip_h))
            for flip_v in (False, True)
            for flip_h in (False, True)
        }
        if len(variants) != 4:
            continue
        if any(
            _cell_set(_transform_cells(cells, h, w, flip_v, flip_h)) == base
            for flip_v, flip_h in [(False, True), (True, False), (True, True)]
        ):
            continue

        grouped[hole_pixels].append(cells)

    return grouped


TEMPLATE_LIBRARY = {
    h: _template_candidates(h, TEMPLATE_W)
    for h in TEMPLATE_H_CHOICES
}


def _gen_template() -> tuple[int, int, list[tuple[int, int]], int] | None:
    """Sample a legal holey, reflection-visible template."""
    hole_choices = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_choices)
    for desired_holes in hole_choices:
        valid_heights = [h for h in TEMPLATE_H_CHOICES if TEMPLATE_LIBRARY[h][desired_holes]]
        if not valid_heights:
            continue
        h = random.choice(valid_heights)
        cells = random.choice(TEMPLATE_LIBRARY[h][desired_holes])
        return h, TEMPLATE_W, cells, desired_holes
    return None


def _choose_origin(cells: list[tuple[int, int]], rows: int, cols: int) -> tuple[int, int] | None:
    """Pick an interior origin with left/up/upper-left evidence and contrast on right/down."""
    cell_set = _cell_set(cells)
    candidates: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 < r < rows - 1 and 0 < c < cols - 1):
            continue
        if (r, c - 1) not in cell_set:
            continue
        if (r - 1, c) not in cell_set:
            continue
        if (r - 1, c - 1) not in cell_set:
            continue
        if not any(rr == r and cc > c for rr, cc in cell_set):
            continue
        if not any(rr > r and cc == c for rr, cc in cell_set):
            continue
        candidates.append((int(r), int(c)))
    return random.choice(candidates) if candidates else None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with a reflection-demonstrating interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    origin = _choose_origin(cells, ext_r, ext_c)
    if origin is None:
        return None
    return cells, origin


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _stamp_points(
    templ_cells: list[tuple[int, int]], h: int, w: int, r0: int, c0: int, deltas: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Real occupied output pixels for all reflected copies."""
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        stamp = _transform_cells(templ_cells, h, w, dr < 0, dc < 0)
        for tr, tc in stamp:
            occupied.add((int(r0 + dr * h + tr), int(c0 + dc * w + tc)))
    return occupied


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Find a valid singleton cluster position and build the input grid."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * GUNIT
        sr1 = orr + max_dr * GUNIT
        sc0 = orc + min_dc * GUNIT
        sc1 = orc + max_dc * GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((orr + dr * GUNIT, orc + dc * GUNIT) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = T if (dr == 0 and dc == 0) else O

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: choose colors, build a holey asymmetric template, then place the lattice."""
    bg, T, O = random.sample(range(10), 3)

    templ = _gen_template()
    if templ is None:
        return None
    h, w, templ_cells, _hole_pixels = templ

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    occupied = _stamp_points(templ_cells, h, w, r0, c0, deltas)

    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        r0,
        c0,
        h,
        w,
        bg,
        T,
        O,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
