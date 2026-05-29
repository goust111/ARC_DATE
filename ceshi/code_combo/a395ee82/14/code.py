# Combo variant of a395ee82 (index 14)
# combo_spec: D3/0+D5/0
# Name: 多色模板覆盖优先级

"""
Design analysis for the D3/0 + D5/0 combo
=========================================

a) Concrete behavior and trace
spec_variant:
  grid: 22x22, one bg color
  template:
    one 4-connected multicolor component
    bbox: 3-4 rows x 3 cols
    area: 6-10
    fill ratio: 0.65-0.85
    touches all four bbox sides
    palette size: 2-3 colors, every color repeated
  singletons:
    count: 5-6
    arranged on the same gunit=2 lattice as the source task
    exactly one singleton uses a color from the template palette; this marks the origin
    all remaining singletons share O, and O is outside the template palette
  output:
    stamp one translated copy of the multicolor template per lattice point
    stride is reduced to (h-1, w-1) so neighboring copies can overlap visibly
    stamps are painted in visible singleton reading order: top-to-bottom, then left-to-right
    origin copy paints every template cell as O
    non-origin copies preserve the template's per-cell colors
    later stamps overwrite earlier ones normally

Trace on a small example:
- Suppose the template bbox is 4x3, so the stamp stride becomes 3x2.
- Suppose three lattice points on the same row are at deltas (-1,0), (0,0), (0,1),
  hence painting order is left copy, origin copy, right copy.
- In the source task, overlap multiplicity alone decided the color, so every shared
  pixel would become O no matter which copy arrived last.
- In this combo, the left copy first paints multicolor cells, then the origin paints
  O over its overlap strip, then the right copy repaints part of that strip with its
  own multicolor cells.
- The changed stages are therefore:
  1. template extraction must preserve per-cell colors instead of flattening them;
  2. stamping uses the reduced stride `(h-1, w-1)` instead of `(h, w)`;
  3. overlap resolution is sequential overwrite, not "any overlap => O".

b) Complexity calibration
- The palette is limited to 2-3 colors, which is enough to expose overwrite priority
  without turning the sprite into noise.
- The generator requires repeated evidence:
  origin wins on some overlap pixels, origin loses on others, and at least one overlap
  between non-origin stamps involves different template colors.
- This keeps the combo visibly distinct from both the original task and the single-D3
  variant where multicolor copies never interact.

c) Old assumptions that break
- `templ_col = one sampled cell` no longer represents the whole template; the solver
  must track `(dr, dc, color)` for every template pixel and infer a palette.
- `origin singleton color == template color` changes into `origin singleton color is in
  the template palette`; generator enforces uniqueness.
- `off_r = dr*h`, `off_c = dc*w` would keep stamp bboxes disjoint, making D5 invisible;
  the combo intentionally changes the stride to `(h-1, w-1)`.
- Coverage-count logic such as `total >= 2 => O` is invalid now because the final color
  depends on paint order, not just multiplicity.
- Sorting by any hidden property would make the task unreadable to humans, so overwrite
  order is tied only to visible singleton positions.

d) Placement legality
- The generator places the template so every shifted copy stays inside the 22x22 grid
  under the reduced stride.
- The singleton cluster bbox remains disjoint from the template bbox with at least one
  background cell gap, as in the source task.
- Singleton pixels are also forbidden from landing on any future stamped output pixel,
  preventing ambiguous input clutter.

e) Visual observability
- The origin marker is the only singleton whose color appears inside the template, so
  the solver can visibly recover the special copy.
- Reading order is visible from singleton geometry alone; no color-id ordering or hidden
  generation state is used.
- Because the output contains both O-preserved overlap bands and O-overwritten bands,
  a human can infer that later stamps overwrite earlier ones.

f) Visual cleanliness
- Overlaps are only one row and/or one column thick because the stride reduction is
  minimal. The output remains readable as repeated sprites with local covering, not as
  a dense pile-up.
- Template colors are assigned in contiguous chunks, which keeps the multicolor motif
  legible after stamping.

g) Change inferability
- There is a single color rule for the special copy: origin cells always paint as O.
- Non-origin copies keep their template colors exactly.
- Overwrite priority is supported by multiple pixels, not by a single accidental clash,
  and the palette stays small enough that order-induced color changes are still easy to
  trace across examples.

Post-change solver assumptions and generator guarantees
- Solver assumes exactly one singleton color belongs to the template palette.
  Generator guarantees that by choosing one origin marker from the template colors and
  giving every other singleton the shared color O.
- Solver assumes the reduced stride produces meaningful overlap.
  Generator guarantees this by requiring origin-win pixels, origin-lose pixels, and at
  least one overlap with conflicting non-origin template colors.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def _stamp_stride(h: int, w: int) -> tuple[int, int]:
    """Use the smallest stride reduction that makes overlap visible."""
    return max(1, int(h) - 1), max(1, int(w) - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint overlapping copies of a multicolor template from a singleton lattice.

    Input structure:
    - The largest non-background component is one 4-connected multicolor template with 2-3 colors.
    - Detached singleton pixels form a regular lattice with uniform spacing.
    - Exactly one singleton uses a color from the template palette and marks the lattice origin;
      every other singleton shares color O.

    Rule:
    - Recover lattice offsets from the singleton positions and normalize them by the singleton spacing.
    - Convert each lattice step into a translated stamp shift of `(template_height-1, template_width-1)`.
    - Sort the singleton markers in reading order and stamp one template copy per marker.
    - Non-origin copies keep the template's cell colors, while the origin copy paints all template cells as O.
      When copies overlap, later stamps simply overwrite earlier ones.

    Key insight: The lattice still encodes the copy locations, but now the internal template colors must be
    preserved and the final overlap colors are determined by visible singleton order rather than by counting
    how many copies touched a pixel.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    step_r, step_c = _stamp_stride(h, w)

    templ_rel: list[tuple[int, int, int]] = []
    templ_palette: set[int] = set()
    for r, c in templ_coords:
        rr = int(r)
        cc = int(c)
        col = int(g[rr, cc])
        templ_rel.append((rr - r0, cc - c0, col))
        templ_palette.add(col)
    assert 2 <= len(templ_palette) <= 3

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template or int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(g[rr, cc])))
    assert len(singletons) >= 2

    origin_candidates = [(r, c) for (r, c, col) in singletons if int(col) in templ_palette]
    assert len(origin_candidates) == 1
    orr, orc = origin_candidates[0]

    other_cols = [int(col) for (r, c, col) in singletons if not (int(r) == orr and int(c) == orc)]
    assert other_cols
    assert all(col not in templ_palette for col in other_cols)
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(r), int(c)) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _r, _c in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))

    ordered: list[tuple[int, int, bool, int, int]] = []
    for dr, dc, rr, cc in deltas:
        ordered.append((int(dr // gunit), int(dc // gunit), int(rr) == orr and int(cc) == orc, int(rr), int(cc)))
    ordered.sort(key=lambda item: (item[3], item[4]))

    out = np.full((H, W), int(bg), dtype=int)
    for dr, dc, is_origin, _rr, _cc in ordered:
        off_r = int(dr * step_r)
        off_c = int(dc * step_c)
        for tr, tc, col in templ_rel:
            R = int(r0 + off_r + tr)
            C = int(c0 + off_c + tc)
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = int(overlap_col if is_origin else col)
    return out


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_COLOR_COUNT_CHOICES = [2, 3]
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SPANNING_RETRIES = 80
TEMPLATE_RETRIES = 180
COLORING_RETRIES = 80
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 1200


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
    """Polyomino of n cells that touches all four sides of the bbox."""
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
    """Require visible horizontal and vertical overlap channels in the template."""
    cellset = set(cells)
    horiz = any((r, 0) in cellset and (r, w - 1) in cellset for r in range(h))
    vert = any((0, c) in cellset and (h - 1, c) in cellset for c in range(w))
    return horiz and vert


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template shape with source-task size constraints and visible overlap channels."""
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


def _chunk_sizes(total: int, parts: int) -> list[int]:
    """Split total into positive chunks with a minimum size of two per chunk."""
    sizes = [2] * parts
    for _ in range(total - 2 * parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _gen_template_colors(
    templ_cells: list[tuple[int, int]], bg: int, overlap_col: int
) -> dict[tuple[int, int], int] | None:
    """Assign a compact 2-3 color pattern to the template cells."""
    area = len(templ_cells)
    max_colors = min(max(TEMPLATE_COLOR_COUNT_CHOICES), area // 2)
    if max_colors < 2:
        return None

    key_functions = [
        lambda cell: (cell[0], cell[1]),
        lambda cell: (cell[1], cell[0]),
        lambda cell: (cell[0] + cell[1], cell[0], cell[1]),
        lambda cell: (cell[0] - cell[1], cell[0], cell[1]),
    ]
    available = [c for c in range(10) if c not in {bg, overlap_col}]

    for _ in range(COLORING_RETRIES):
        k = random.choice([x for x in TEMPLATE_COLOR_COUNT_CHOICES if 2 <= x <= max_colors])
        palette = random.sample(available, k)
        ordered = sorted(templ_cells, key=random.choice(key_functions))
        if random.random() < 0.5:
            ordered = list(reversed(ordered))
        sizes = _chunk_sizes(area, k)
        color_map: dict[tuple[int, int], int] = {}
        idx = 0
        for col, size in zip(palette, sizes):
            for cell in ordered[idx : idx + size]:
                color_map[cell] = int(col)
            idx += size
        counts = Counter(color_map.values())
        if len(counts) != k or min(counts.values()) < 2:
            continue
        return color_map
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with a strictly interior origin."""
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
    """Return (min_dr, max_dr, min_dc, max_dc) for lattice offsets."""
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
    """Relative output footprints for each lattice delta under the reduced stride."""
    step_r, step_c = _stamp_stride(h, w)
    pixels: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for dr, dc in deltas:
        pixels[(dr, dc)] = {
            (int(dr * step_r + tr), int(dc * step_c + tc))
            for tr, tc in templ_cells
        }
    return pixels


def _structural_overlap_profile(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int, int, dict[tuple[int, int], set[tuple[int, int]]]]:
    """
    Measure overlap structure without looking at colors.

    Returns (origin_wins, origin_loses, nonorigin_overlap_pixels, relative_pixel_sets).
    """
    ordered = sorted(deltas)
    assert (0, 0) in ordered
    pixels = _stamp_pixels(deltas, templ_cells, h, w)

    pixel_cover: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    for idx, delta in enumerate(ordered):
        is_origin = delta == (0, 0)
        for pt in pixels[delta]:
            pixel_cover[pt].append((idx, is_origin))

    origin_wins = 0
    origin_loses = 0
    nonorigin_overlap_pixels = 0
    for cover in pixel_cover.values():
        if len(cover) < 2:
            continue
        if any(is_origin for _idx, is_origin in cover):
            if cover[-1][1]:
                origin_wins += 1
            else:
                origin_loses += 1
        if sum(1 for _idx, is_origin in cover if not is_origin) >= 2:
            nonorigin_overlap_pixels += 1

    return origin_wins, origin_loses, nonorigin_overlap_pixels, pixels


def _behavior_profile(
    deltas: list[tuple[int, int]],
    templ_colors: dict[tuple[int, int], int],
    h: int,
    w: int,
    overlap_col: int,
) -> dict[str, int]:
    """Quantify whether the generated sample actually exposes the combo behavior."""
    ordered = sorted(deltas)
    step_r, step_c = _stamp_stride(h, w)
    histories: dict[tuple[int, int], list[tuple[int, bool, int]]] = defaultdict(list)

    for idx, (dr, dc) in enumerate(ordered):
        is_origin = (int(dr), int(dc)) == (0, 0)
        for (tr, tc), col in templ_colors.items():
            pt = (int(dr * step_r + tr), int(dc * step_c + tc))
            paint = int(overlap_col if is_origin else col)
            histories[pt].append((idx, is_origin, paint))

    overlap_pixels = 0
    origin_wins = 0
    origin_loses = 0
    palette_priority = 0
    for events in histories.values():
        if len(events) < 2:
            continue
        overlap_pixels += 1
        if any(is_origin for _idx, is_origin, _paint in events):
            if events[-1][1]:
                origin_wins += 1
            else:
                origin_loses += 1
        nonorigin_colors = [paint for _idx, is_origin, paint in events if not is_origin]
        if len(nonorigin_colors) >= 2 and len(set(nonorigin_colors)) >= 2 and not events[-1][1]:
            palette_priority += 1

    return {
        "overlap_pixels": int(overlap_pixels),
        "origin_wins": int(origin_wins),
        "origin_loses": int(origin_loses),
        "palette_priority": int(palette_priority),
    }


def _render_output(
    deltas: list[tuple[int, int]],
    templ_colors: dict[tuple[int, int], int],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    overlap_col: int,
) -> np.ndarray:
    """Render the intended output using sequential overwrite order."""
    step_r, step_c = _stamp_stride(h, w)
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for dr, dc in sorted(deltas):
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(dr * step_r)
        off_c = int(dc * step_c)
        for (tr, tc), col in templ_colors.items():
            out[r0 + off_r + tr, c0 + off_c + tc] = int(overlap_col if is_origin else col)
    return out


def _shift_pixels(
    pixels: dict[tuple[int, int], set[tuple[int, int]]],
    r0: int,
    c0: int,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Translate relative stamp footprints into absolute grid positions."""
    shifted: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for delta, pts in pixels.items():
        shifted[delta] = {(int(r0 + r), int(c0 + c)) for r, c in pts}
    return shifted


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_colors: dict[tuple[int, int], int],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    origin_col: int,
    overlap_col: int,
    occupied: set[tuple[int, int]],
    expected: np.ndarray,
) -> np.ndarray | None:
    """Find a legal singleton cluster placement and build the input grid."""
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

        if any((int(orr + dr * GUNIT), int(orc + dc * GUNIT)) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for (tr, tc), col in templ_colors.items():
            grid[r0 + tr, c0 + tc] = int(col)
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = int(origin_col if (dr == 0 and dc == 0) else overlap_col)

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the combo variant."""
    bg, overlap_col = random.sample(range(10), 2)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lattice = _gen_lattice(n_sing)
    if lattice is None:
        return None
    lattice_cells, origin = lattice
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]

    origin_wins, origin_loses, nonorigin_overlap, rel_pixels = _structural_overlap_profile(deltas, templ_cells, h, w)
    if origin_wins <= 0 or origin_loses <= 0 or nonorigin_overlap <= 0:
        return None

    templ_colors = None
    behavior = None
    for _ in range(COLORING_RETRIES):
        candidate = _gen_template_colors(templ_cells, bg, overlap_col)
        if candidate is None:
            continue
        profile = _behavior_profile(deltas, candidate, h, w, overlap_col)
        if profile["origin_wins"] > 0 and profile["origin_loses"] > 0 and profile["palette_priority"] > 0:
            templ_colors = candidate
            behavior = profile
            break
    if templ_colors is None or behavior is None:
        return None

    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)
    step_r, step_c = _stamp_stride(h, w)
    r0_lo, r0_hi = -min_dr * step_r, GRID_SIZE - max_dr * step_r - h
    c0_lo, c0_hi = -min_dc * step_c, GRID_SIZE - max_dc * step_c - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    origin_col = random.choice(list(templ_colors.values()))

    abs_pixels = _shift_pixels(rel_pixels, r0, c0)
    occupied: set[tuple[int, int]] = set()
    for pts in abs_pixels.values():
        occupied.update(pts)

    expected = _render_output(deltas, templ_colors, r0, c0, h, w, bg, overlap_col)
    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_colors,
        r0,
        c0,
        h,
        w,
        bg,
        origin_col,
        overlap_col,
        occupied,
        expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
