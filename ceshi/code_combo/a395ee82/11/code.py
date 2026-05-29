# Combo variant of a395ee82 (index 11)
# combo_spec: D3/1+D4/4
# Name: 孔洞模板擦除

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
# - Preserved backbone from the base task:
#   one large monochrome template of color T plus a detached singleton lattice with
#   one T-colored origin and 4-5 O-colored witnesses.
# - D3/1 changes the destination template itself:
#   the template bbox is still 3-4 by 3 and still touches all four bbox sides, but
#   it now contains 1-2 enclosed bg pixels. Only the filled mask is part of the
#   object; enclosed bg cells are structural holes.
# - D4/4 is fused as erase semantics rather than recolor:
#   conceptually draw every translated copy in T, then apply the origin-position copy
#   as an eraser. Every filled pixel of that origin copy is cleared back to bg, while
#   hole pixels are left untouched as transparent windows.
# - Small example with a 4x3 template:
#     ###      Base stamping would replicate the eight filled cells at every lattice
#     #.#      offset. The fused rule then removes the eight filled cells at the
#     ###      origin position, but the central hole cell is not touched by the
#     #..      eraser because it is not part of the template mask.
# - Important family-level consequence:
#   the preserved row/col translation uses full bbox height/width, so translated
#   bboxes are normally disjoint. That means the "window" usually preserves bg rather
#   than another copy's paint, producing a clean missing holey sprite. The solver is
#   still implemented with true erase semantics (`total > 0` and `origin_cover == 0`)
#   so the logic stays correct if overlaps ever appear.
#
# b) Complexity calibration:
# - Keep the same 3-color palette, singleton count range, lattice spacing, and
#   origin-inside-lattice rule, so the new burden is localized to two visible facts:
#   the template has genuine holes, and the origin copy disappears by erasure.
# - Hole count is limited to 1 or 2 because the bbox is tiny; this keeps the task
#   human-inferable while still making hole-awareness necessary.
# - The generator keeps the origin strictly interior so the missing copy is supported
#   by surrounding repetitions instead of becoming an ambiguous edge omission.
#
# c) Implicit-assumption audit against the original solver:
# - "Largest component is the template" still holds, but hole reasoning is cleaner
#   with monochromatic component parsing. The generator guarantees all other objects
#   are singleton pixels.
# - The old code could iterate template foreground coordinates directly, but only for
#   a solid-mask interpretation. Here we must explicitly validate that the bbox also
#   contains enclosed bg holes; rectangle-based stamping would be wrong.
# - The old recolor logic (`origin -> O`, overlaps -> O) is invalid after fusion.
#   Erase mode depends on whether a cell belongs to the origin filled mask, so the
#   solver tracks `origin_cover` separately from total coverage.
# - No step may depend on hidden color-id ordering. The only semantic color relation
#   is "singleton color == template color" marks the origin.
#
# d) Placement legality:
# - All translated bboxes fit inside 22x22 because generator placement is computed
#   from the lattice delta bounds and the chosen bbox height/width.
# - The singleton cluster bbox stays disjoint from the template bbox with at least
#   one bg cell gap, matching the base family and avoiding accidental mergers.
# - The generator keeps filled-pixel overlaps between translated copies forbidden.
#   That preserves visual cleanliness and matches the preserved stride assumptions.
#
# e) Visual observability:
# - Every sample contains at least one enclosed hole pixel, usually in the center
#   column of the template bbox, so the source object is visibly not a solid sprite.
# - The output always contains an interior missing copy position because the origin is
#   chosen from a strictly interior lattice cell.
# - The missing copy is shaped by the exact holey mask, so a solver must recover both
#   the lattice motion and the hole-aware foreground mask.
#
# f) Visual cleanliness:
# - Inputs still contain only two object families: one large template and isolated
#   singleton markers. There is no extra clutter or auxiliary legend object.
# - Outputs stay crisp because translated copies do not overlap on filled pixels and
#   the erased origin produces one clean gap rather than a noisy recolor field.
#
# g) Rule inferability / non-degeneration:
# - Compared with the original task, the same input no longer yields an O-colored
#   origin copy; it yields bg there, so the new transform is visibly different.
# - Compared with the base solid-template family, hole pixels are mandatory and are
#   preserved as bg in every translated copy.
# - The generator rejects any template without real enclosed holes and any instance
#   where the erased output would collapse to a single remaining copy.
#
# h) New main() assumptions and generator guarantees:
# - main() assumes the largest monochromatic component has 1-2 enclosed bg holes.
#   The template library enforces that exactly.
# - main() assumes exactly one singleton shares the template color and all other
#   singletons share one marker color. The generator constructs that directly.
# - main() assumes lattice spacing can be recovered by a gcd over singleton deltas.
#   The generator preserves the original uniform spacing `GUNIT = 2`.


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SPANNING_RETRIES = 40
SINGLETON_RETRIES = 60
GENERATE_RETRIES = 400


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bg cells in the bbox that are not 4-connected to the bbox edge."""
    h, w = map(int, filled_mask.shape)
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if not touches_edge:
            holes[coords[:, 0], coords[:, 1]] = True
    return holes


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a holey template from a singleton lattice, then erase the origin copy.

    Input structure:
    - `bg` is the dominant background color.
    - One large monochrome component of color `T` is the template; its bbox is 3-4 by 3.
    - Inside that bbox there are 1-2 enclosed `bg` holes, so the copied object is a
      foreground mask with internal empty pixels rather than a solid rectangle.
    - Elsewhere there is a uniform-spacing lattice of singleton pixels: exactly one is
      color `T` (the origin marker) and the others share color `O`.

    Rule:
    - Read singleton offsets relative to the unique `T` singleton and normalize them by
      the lattice spacing.
    - Convert each lattice offset into a translated template position using bbox height
      and width as the row/col step sizes.
    - Draw the template's filled pixels for every translated copy, but then erase the
      origin-position copy by clearing its filled pixels back to `bg`.
    - Hole cells are never painted and never erased, so they remain transparent windows.

    Key insight: The repeated object is the template's exact filled mask. The origin
    marker does not mean "recolor this copy"; it identifies which translated instance
    acts as an eraser stencil, while enclosed holes stay outside that stencil.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = map(int, g.shape)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    assert h in TEMPLATE_H_CHOICES and w == TEMPLATE_W

    templ_mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col)
    hole_mask = _enclosed_bg_mask(templ_mask)
    hole_pixels = int(hole_mask.sum())
    assert hole_pixels in TEMPLATE_HOLE_PIXELS

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

    assert SINGLETON_COUNT_RANGE[0] <= len(singletons) <= SINGLETON_COUNT_RANGE[1]

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    marker_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert marker_cols
    marker_col = int(marker_cols[0])
    assert all(int(col) == int(marker_col) for col in marker_cols)

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[(total > 0) & (origin_cover == 0)] = int(templ_col)
    return out


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier: list[tuple[int, int]] = []
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


def _template_candidates(h: int, w: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate all legal holey templates, grouped by enclosed-hole pixel count."""
    total = int(h * w)
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))

    grouped: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}
    for bits in range(1, 1 << total):
        cells = [(idx // w, idx % w) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < lo or area > hi:
            continue

        mask = _cells_to_mask(cells, h, w)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not mask[0, :].any() or not mask[h - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, w - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels in grouped:
            grouped[hole_pixels].append(cells)
    return grouped


TEMPLATE_LIBRARY = {
    h: _template_candidates(h, TEMPLATE_W)
    for h in TEMPLATE_H_CHOICES
}


def _gen_template() -> tuple[int, int, list[tuple[int, int]], int] | None:
    """Sample a legal holey template with 1-2 enclosed bg pixels."""
    hole_choices = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_choices)
    for desired_holes in hole_choices:
        valid_heights = [h for h in TEMPLATE_H_CHOICES if TEMPLATE_LIBRARY[h][desired_holes]]
        if not valid_heights:
            continue
        h = int(random.choice(valid_heights))
        cells = random.choice(TEMPLATE_LIBRARY[h][desired_holes])
        return h, TEMPLATE_W, cells, desired_holes
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
    return min_dr, max_dr, min_dc, max_dc


def _baseline_like_output(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    templ_col: int,
    marker_col: int,
) -> np.ndarray:
    """Original-family output on the same input: origin recolored, not erased."""
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for dr, dc in deltas:
        is_origin = int(dr) == 0 and int(dc) == 0
        col = int(marker_col) if is_origin else int(templ_col)
        off_r = int(r0 + dr * h)
        off_c = int(c0 + dc * w)
        for rr, cc in templ_cells:
            out[off_r + rr, off_c + cc] = int(col)
    return out


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
    templ_col: int,
    marker_col: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Try to place the singleton lattice and build a valid input grid."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)

    expected_gap = {(int(r0 + rr), int(c0 + cc)) for rr, cc in templ_cells}

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

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = int(templ_col)
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = int(templ_col) if (dr == 0 and dc == 0) else int(marker_col)

        try:
            out = main(grid)
        except Exception:
            continue

        # The erased origin must produce a real full-mask gap at the template site.
        if any(int(out[r, c]) != int(bg) for r, c in expected_gap):
            continue

        # There must still be more than one surviving copy so the missing origin is inferable.
        if int((out == int(templ_col)).sum()) <= len(templ_cells):
            continue

        # Quantified anti-degeneration guard:
        # compared with the original family on the same input, the origin copy is erased,
        # not recolored. The difference count should equal the filled template area.
        baseline = _baseline_like_output(deltas, templ_cells, r0, c0, h, w, bg, templ_col, marker_col)
        diff = int(np.count_nonzero(out != baseline))
        if diff != len(templ_cells):
            continue

        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: choose colors, build a holey template, then place the lattice."""
    bg, templ_col, marker_col = random.sample(range(10), 3)

    templ = _gen_template()
    if templ is None:
        return None
    h, w, templ_cells, _hole_pixels = templ

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

    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)

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
        templ_col,
        marker_col,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
