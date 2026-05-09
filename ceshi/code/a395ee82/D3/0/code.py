# Variant of a395ee82
# Dimension: D3 — 目标模板结构 (dst)
# Change: 单连通单色精灵(色T)，外接矩形3-4行x3列，面积6-10，填充率0.65-0.85，触及外接矩形四边... → 多色模板

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior and trace
spec_variant:
  grid: 22x22, one bg color
  template:
    one 4-connected multicolor component
    same shape constraints as the source task:
      bbox: 3-4 rows x 3 cols
      area: 6-10
      fill ratio: 0.65-0.85
      touches all four bbox sides
    colors: 2-3 distinct non-bg, non-O colors
  singletons:
    count: 5-6
    one origin singleton uses one color already present in the template palette
    all other singletons use O
    lattice geometry stays unchanged from the source task
  output:
    stamp translated copies of the multicolor template at the lattice offsets
    non-origin copies preserve each template cell color exactly
    origin copy recolors every template cell to O
    any accidental overlap is also recolored to O

Trace on a small example:
- Suppose the template bbox is 3x3 and its occupied cells are
  {(0,0):A, (0,1):A, (1,1):B, (1,2):B, (2,0):C, (2,1):C}.
- Suppose the singleton lattice deltas are {(0,0), (0,1), (1,0)} and the
  singleton at (0,0) is color B while the other two are O.
- Original task behavior would stamp three monochrome copies and recolor the
  origin copy to O, so all non-origin copies lose the internal A/B/C pattern.
- Variant behavior stamps the same geometry, but the copy at (0,1) and (1,0)
  stays A/B/C cell-by-cell, while only the origin copy becomes solid O.
- The changed step is therefore the rendering stage: geometry inference from
  the lattice is unchanged, but per-cell template colors must now be preserved.

b) Complexity calibration
- The template uses exactly 2 or 3 colors, not an arbitrary large palette.
- Every template color appears at least twice, so the multicolor nature is
  visually obvious and not reducible to a single outlier pixel.
- The singleton system remains the same 5-6 point lattice, so the task still
  has one clear geometric mechanism instead of introducing a second puzzle.

c) Old assumptions that break under D3
- `templ_col = ...` used one sampled template pixel as if it represented the
  whole component. That is only valid when the template is monochrome.
- `templ_rel` stored coordinates only. The variant needs `(dr, dc, color)` for
  every occupied template cell.
- `origin = ... if singleton color == templ_col` assumed the unique template
  color also marked the origin singleton. In the variant the template has a
  palette, so the correct predicate is “singleton color belongs to the template
  palette”; generator guarantees exactly one such singleton.
- `out[total > 0] = templ_col` flattened every non-origin copy to one color.
  That would silently destroy the D3 change, so the solver now records the
  first stamped color at each cell and only overrides with O for origin cells
  or true overlaps.

d) Placement legality
- Template palette is chosen disjoint from both bg and O, so the origin marker
  color can be recognized as “belongs to template palette” and O remains a
  separate recolor signal.
- Template remains the largest connected foreground component; all other
  connected components are singletons, so the solver can still identify the
  template by area.
- The singleton cluster bbox stays disjoint from the template bbox with at
  least one bg cell gap, exactly as in the source task.
- All translated stamp bboxes are kept in-bounds; the generator also rejects
  any accidental cell overlap before building the input.

e) Visual observability
- Across 2-4 examples, a human sees the same lattice-copying mechanism as the
  source task, but now the repeated copies visibly preserve a multicolor motif.
- The origin singleton is the only singleton whose color also appears inside
  the template, which is a visible cue for which copy becomes O.

f) Visual cleanliness
- The template is still small and compact, and colors are assigned in chunked
  bands rather than pure per-pixel noise, so copied sprites remain legible.
- Because the generator rejects stamp overlaps, the output does not become a
  pile of conflicting colors; only the origin copy is uniformly recolored.

g) Color-rule inferability
- There is only one recolor rule: every cell of the origin copy becomes O.
- All non-origin copies keep the exact template colors. This is easier to infer
  than introducing a hidden per-color remapping table from only a few samples.

Post-change solver assumptions and generator guarantees
- Solver assumes exactly one singleton color is in the template palette.
  Generator enforces this by choosing one origin marker from the template
  colors and coloring all remaining singletons with O only.
- Solver assumes template palette size is at least 2. Generator enforces 2-3
  template colors with at least two cells per color.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a multicolor template guided by a singleton lattice.

    Input structure:
    - bg = background. One large 4-connected template component carries 2-3 colors.
    - Singleton pixels elsewhere form the same uniform-spacing lattice as the source task.
    - Exactly one singleton uses a color from the template palette; that singleton is the lattice origin.
      All remaining singletons share color O.

    Rule:
    - Read each singleton offset from the unique palette-matching singleton and convert those offsets into
      coarse lattice units.
    - Stamp translated copies of the template at those offsets, using the template bbox height/width as the
      translation scale.
    - Non-origin copies keep the template's per-cell colors. The origin copy recolors every stamped cell to O.
    - If two copies ever cover the same output cell, that cell is also recolored to O.

    Key insight: Geometry recovery is unchanged, but color handling is different: the template must be copied
    cell-by-cell with its internal colors preserved, and the unique singleton whose color belongs to the
    template palette identifies the one copy that is fully recolored.
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
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(g[rr, cc])))
    assert len(singletons) >= 2

    origin_candidates = [(r, c, col) for (r, c, col) in singletons if int(col) in templ_palette]
    assert len(origin_candidates) == 1
    orr, orc, _origin_col = origin_candidates[0]

    overlap_cols = [int(col) for (r, c, col) in singletons if not (int(r) == orr and int(c) == orc)]
    assert overlap_cols
    overlap_cols = [col for col in overlap_cols if col not in templ_palette]
    assert overlap_cols
    overlap_col = int(Counter(overlap_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(r) == orr and int(c) == orc) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in deltas]

    total = np.zeros((H, W), dtype=int)
    first_color = np.full((H, W), int(bg), dtype=int)

    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc, col in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if total[R, C] == 1:
                    first_color[R, C] = int(overlap_col if is_origin else col)

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = first_color[total > 0]
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator (D3) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_COLOR_COUNT_CHOICES = [2, 3]
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
COLORING_RETRIES = 30
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


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
    """Polyomino of *n* cells that spans exactly rows x cols (touches all 4 edges)."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template shape in h x w, spanning bbox, within area/fill bounds."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _chunk_sizes(total: int, parts: int) -> list[int]:
    """Split total into `parts` positive chunks with a minimum size of 2."""
    sizes = [2] * parts
    for _ in range(total - 2 * parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _gen_template_colors(
    templ_cells: list[tuple[int, int]], bg: int, overlap_col: int
) -> dict[tuple[int, int], int] | None:
    """Assign 2-3 visible colors to template cells while keeping the overall shape unchanged."""
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
    """Lattice polyomino of *n* cells with strictly interior origin."""
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
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


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
    """Render the intended output using the D3 multicolor stamping rule."""
    total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    first_color = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for dr, dc in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(dr) == 0 and int(dc) == 0
        for (tr, tc), col in templ_colors.items():
            R = int(r0 + off_r + tr)
            C = int(c0 + off_c + tc)
            total[R, C] += 1
            if total[R, C] == 1:
                first_color[R, C] = int(overlap_col if is_origin else col)
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    out[total > 0] = first_color[total > 0]
    out[total >= 2] = int(overlap_col)
    return out


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
    """Try to find a valid singleton cluster position and build the input grid."""
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
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
    bg, overlap_col = random.sample(range(10), 2)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    templ_colors = _gen_template_colors(templ_cells, bg, overlap_col)
    if templ_colors is None:
        return None
    origin_col = random.choice(list(templ_colors.values()))

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

    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (r0 + dr * h + tr, c0 + dc * w + tc)
            if pt in occupied:
                return None
            occupied.add(pt)

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
