# Variant of a395ee82
# Dimension: D5 — stamp重叠处理规则 (transform)
# Change: 重叠像素(被两个或以上stamp覆盖)统一着O色... → stamp不允许重叠

# Design analysis
# spec:
#   core_variant:
#     keep the original input language: one monochrome template component plus a
#     singleton lattice with exactly one T-colored origin marker and the rest O.
#     The output still places one template copy per lattice point and recolors
#     the origin copy to O.
#   minimal_supporting_adjustment:
#     with the seed rule `offset = (dr*h, dc*w)`, two different template bboxes
#     never overlap at all, so D5 would be invisible. To make overlap handling
#     observable, reduce the translation stride to `(h-1, w-1)`. This preserves
#     the original "template-sized coarse shift" idea while creating only a
#     1-row / 1-col overlap band between neighboring lattice points.
#   overwrite_priority:
#     stamps are painted in singleton reading order: sort lattice markers by
#     `(row, col)` in the input. Later stamps overwrite earlier ones normally.
#
# a) concretization_and_trace:
#   input still shows:
#   - a large T-colored sprite component;
#   - a detached lattice of singleton pixels spaced by `gunit=2`;
#   - exactly one singleton uses T and marks lattice origin.
#   output now uses stride `(h-1, w-1)`, so horizontally adjacent lattice cells
#   make two copies share one template column, and vertically adjacent cells
#   share one template row.
#   Example trace:
#   - template bbox = 4x3, hence stride = 3x2;
#   - origin delta (0,0), right neighbor delta (0,1), left neighbor delta (0,-1);
#   - row-major order is left, origin, right.
#   Original logic on an overlap pixel:
#     any overlap count >= 2 => force O.
#   New logic on that same overlap pixel:
#     left/origin overlap only => origin is later, so final O;
#     origin/right overlap => right is later, so final T.
#   The difference is visible inside the same origin copy: some overlap bands stay
#   O and others revert to T.
#
# b) complexity_calibration:
#   only one priority rule is used globally, avoiding many-case ambiguity. The
#   generator requires both an "origin wins" overlap and an "origin loses"
#   overlap, so the overwrite rule is supported by repeated evidence instead of a
#   single accidental collision.
#
# c) old_assumptions_that_break:
#   - `off_r = dr * h` and `off_c = dc * w` assumed disjoint bboxes. Under D5
#     this makes overlap impossible, so these lines must change.
#   - `total` / `center` coverage counting assumed overlap resolution depends only
#     on multiplicity, not order. That no longer holds; sequential painting is
#     required.
#   - `out[total >= 2] = O` was harmless before but would destroy the whole new
#     variant by erasing ordering effects.
#
# d) placement_legality:
#   generator places every future stamp fully inside the 22x22 grid using the
#   reduced stride. Singleton cluster stays detached from the template bbox with a
#   background gap, and singleton cells are forbidden from landing on any future
#   stamped pixel so the input remains visually clean.
#
# e) visual_observability:
#   priority depends only on visible singleton positions, never on hidden facts
#   like color ids or generation order. The reading-order rule can be inferred
#   from a few examples because the origin overlap produces both O-preserved and
#   T-overwritten regions.
#
# f) visual_tidiness:
#   overlaps are limited to thin bands caused by stride `(h-1, w-1)`, so the
#   output stays legible: recognizable copies with local overwriting, not a noisy
#   pile-up.
#
# g) color_change_inference:
#   the palette stays at three colors. O still means "origin-colored copy", but
#   now O survives only where no later T-stamp repaints it. This keeps the color
#   story simple enough for ARC-style induction.

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def _stamp_stride(h: int, w: int) -> tuple[int, int]:
    """Minimal stride reduction that makes neighboring stamps overlap."""
    return max(1, int(h) - 1), max(1, int(w) - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint translated template copies from a singleton lattice using overwrite order.

    Input structure:
    - The largest non-background component is a monochrome template in color T.
    - Detached singleton pixels form a regular lattice; exactly one singleton is
      also color T and marks the lattice origin.
    - All remaining singletons share color O, the recolor used for the origin copy.

    Rule:
    - Normalize singleton offsets by the lattice spacing and convert each lattice
      step into a template shift of `(template_height-1, template_width-1)`.
    - Sort lattice markers in reading order: top-to-bottom, then left-to-right.
    - Stamp the template at every lattice offset; the origin copy is painted O and
      every other copy is painted T.
    - If stamps overlap, later stamps simply overwrite earlier paint instead of
      turning the shared pixels into a special overlap color.

    Key insight: The singleton layout encodes both where to place each copy and
    which copy is painted later, so overlap resolution comes from visible lattice
    order rather than from counting how many copies touch a pixel.
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
    step_r, step_c = _stamp_stride(h, w)

    templ_rel = [(int(r) - r0, int(c) - c0) for r, c in templ_coords]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template or int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(g[rr, cc])))

    origin = [(r, c) for (r, c, col) in singletons if col == templ_col]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [col for (_r, _c, col) in singletons if col != templ_col]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(r - orr, c - orc, col, r, c) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _col, _r, _c in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))

    ordered = []
    for dr, dc, col, rr, cc in deltas:
        ordered.append((int(dr // gunit), int(dc // gunit), int(col), int(rr), int(cc)))
    ordered.sort(key=lambda item: (item[3], item[4]))

    out = np.full((H, W), int(bg), dtype=int)
    for dr, dc, col, _rr, _cc in ordered:
        off_r = int(dr * step_r)
        off_c = int(dc * step_c)
        paint_col = int(overlap_col) if int(col) == int(templ_col) else int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = paint_col
    return out


# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 120
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 800


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of *n* cells in a rows x cols grid."""
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
    """Polyomino of *n* cells that spans exactly rows x cols."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _template_overlap_ready(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    """Require visible horizontal and vertical overlap channels."""
    cellset = set(cells)
    horiz = any((r, 0) in cellset and (r, w - 1) in cellset for r in range(h))
    vert = any((0, c) in cellset and (h - 1, c) in cellset for c in range(w))
    return horiz and vert


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template in h x w, spanning bbox, overlap-compatible."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None and _template_overlap_ready(cells, h, w):
            return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of *n* cells with a strictly interior origin."""
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


def _stamp_pixels(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Relative pixel footprints for each lattice delta."""
    step_r, step_c = _stamp_stride(h, w)
    pixels: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for dr, dc in deltas:
        pixels[(dr, dc)] = {
            (int(dr * step_r + tr), int(dc * step_c + tc))
            for tr, tc in templ_cells
        }
    return pixels


def _origin_overlap_profile(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict[tuple[int, int], set[tuple[int, int]]]]:
    """
    Return (origin_wins, origin_loses, pixels_by_delta).

    origin_wins: overlap pixels where the origin stamp is the latest covering copy.
    origin_loses: overlap pixels where some later non-origin stamp repaints origin.
    """
    ordered = sorted(deltas)
    assert (0, 0) in ordered
    origin_idx = ordered.index((0, 0))
    pixels = _stamp_pixels(deltas, templ_cells, h, w)

    pixel_cover: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, delta in enumerate(ordered):
        for pt in pixels[delta]:
            pixel_cover[pt].append(idx)

    origin_wins: set[tuple[int, int]] = set()
    origin_loses: set[tuple[int, int]] = set()
    for pt in pixels[(0, 0)]:
        cover = pixel_cover.get(pt, [])
        if len(cover) < 2:
            continue
        has_earlier = any(idx < origin_idx for idx in cover)
        has_later = any(idx > origin_idx for idx in cover)
        if has_later:
            origin_loses.add(pt)
        elif has_earlier:
            origin_wins.add(pt)

    return origin_wins, origin_loses, pixels


def _shift_pixels(
    pixels: dict[tuple[int, int], set[tuple[int, int]]],
    r0: int,
    c0: int,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Translate relative pixel footprints to absolute grid positions."""
    shifted: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for delta, pts in pixels.items():
        shifted[delta] = {(r0 + r, c0 + c) for r, c in pts}
    return shifted


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
    """Find a legal singleton cluster position and build the input grid."""
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
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build a template+lattice pair that exposes overwrite order."""
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]

    origin_wins, origin_loses, rel_pixels = _origin_overlap_profile(deltas, templ_cells, h, w)
    if not origin_wins or not origin_loses:
        return None

    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)
    step_r, step_c = _stamp_stride(h, w)
    r0_lo, r0_hi = -min_dr * step_r, GRID_SIZE - max_dr * step_r - h
    c0_lo, c0_hi = -min_dc * step_c, GRID_SIZE - max_dc * step_c - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    abs_pixels = _shift_pixels(rel_pixels, r0, c0)
    occupied = set()
    for pts in abs_pixels.values():
        occupied.update(pts)

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
