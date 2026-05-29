# Combo variant of 1ae2feb7 (index 13)
# combo_spec: D3/2+D8/3
# Name: 矩形色块GCD间隔

from __future__ import annotations

from math import gcd

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

"""
Design analysis for the combo variant
=====================================

spec:
  grid:
    size: 9-13 rows x 15-20 cols
    bg: 0
  separator:
    one vertical monochrome column, present in at least H-1 rows
    source objects all lie on one side; the opposite side is blank in input
  source objects:
    solid monochrome rectangles instead of 1D bars
    exactly 3 horizontal bands, with >=1 empty row between adjacent bands
    band roles:
      - one single 2x3 rectangle
      - one single 3x2 rectangle
      - one 2-rectangle band of height 2, widths {2,3} in either near/far order
    nearest rectangle in a band may leave a 1-cell gap to the separator
  transform:
    rectangles are still processed nearest-to-separator first, never overwriting
    the periodic step is no longer per rectangle
    for each row, step = gcd(area of all rectangles touching that row)
    each rectangle paints dots across every row it occupies using that row step

a) Concrete behavior and trace
  Original task:
    row r with bars of lengths 4 and 6 paints with per-bar steps 4 and 6.
  D3-only rectangle variant:
    a 2x3 rectangle paints both of its rows with step 6 = area.
  This combo:
    suppose a height-2 band has two adjacent rectangles with widths 2 and 3.
    Their areas are 4 and 6, so every row in that band uses gcd(4,6)=2.
    The near rectangle paints positions 0,2,4,... on both rows; the far rectangle
    targets the same cadence and only fills misses because overwrite is forbidden.
    In another band, a lone 2x3 rectangle still paints with step 6, and a lone
    3x2 rectangle also paints with step 6. That pair shows that area, not width
    or height alone, controls the period; the 2-rectangle band then shows the
    extra GCD rule when multiple rectangles share a row.

b) Complexity calibration
  The design keeps only one geometric code: area. It avoids overfitting by using
  repeated evidence:
  - 2x3 and 3x2 single-rectangle bands share the same period 6
  - one multi-rectangle band forces a non-trivial shared step 2 from areas 4 and 6
  This is enough to distinguish area from width/height and to distinguish shared
  GCD spacing from per-rectangle area spacing without adding many free variables.

c) Hidden assumptions audit
  Original row-run logic breaks in two places:
  - `_iter_runs_1d` only sees width, so it cannot recover rectangle area.
  - per-object `step = length` is invalid because the step becomes a row property.
  New solver assumptions:
  - every source connected component is a solid axis-aligned rectangle
  - rectangles that share a row are exactly the rectangles whose areas enter that
    row's GCD
  - the target side starts blank
  The generator enforces these assumptions by using separated horizontal bands,
  solid rectangles, unique rectangle colors, and output-first validation.

d) Placement legality
  The generator uses 3 bands with heights [2,3,2] in random vertical order and
  distributes remaining slack as extra blank margins/gaps. Rectangles are packed
  from the separator outward on the source side, so they never cross the separator,
  overlap, or leave the canvas. The 2-rectangle band needs width 5 plus optional
  1-cell separator gap, so source-side width is constrained accordingly.

e) Visual observability
  Human-solvable evidence comes from two repeated patterns:
  - same-area rectangles (2x3 and 3x2) yield the same cadence
  - the 2-rectangle band is visibly denser than either area-4 or area-6 would be
    alone, forcing the solver to infer a shared arithmetic rule
  This can be seen directly from shapes and output spacing; no color-id arithmetic
  or hidden ordering is used.

f) Visual cleanliness
  Bands are separated by blank rows, so each rectangle cluster maps cleanly to its
  own output rows. The target side only contains one dot layer, and "no overwrite"
  keeps overlaps readable: the nearer rectangle wins where cadences coincide.

g) Change inferability
  Colors are preserved and only label rectangles. The critical information is fully
  visible in geometry: rectangle height/width implies area, shared rows imply which
  areas enter the GCD, and repeated periodic dots reveal the resulting step.

h) Quantitative anti-degeneracy target
  A valid generated sample must contain at least one row with >=2 rectangles whose
  area GCD is strictly between 1 and the minimum area on that row, and the combo
  output must differ from the D3-only "per-rectangle area step" output on the same
  input. The generator explicitly checks both conditions.

i) Post-change solver assumptions checked against generation
  - `_find_sep_col` still expects a near-full monochrome separator column.
  - `_find_rectangles` expects each source component to be a solid rectangle.
  - row steps are computed from all rectangles covering each row; generation makes
    the row memberships visually unambiguous by using full-height band rectangles.
  - rectangles are sorted only by geometric distance from the separator; generation
    never relies on color values or any invisible metadata.
"""


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _gcd_many(values: list[int]) -> int:
    step = 0
    for value in values:
        step = value if step == 0 else gcd(step, value)
    return step


def _collect_component(
    g: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    c_lo: int,
    c_hi: int,
    seen: np.ndarray,
) -> list[tuple[int, int]]:
    h, _w = g.shape
    color = int(g[r0, c0])
    stack = [(r0, c0)]
    seen[r0, c0] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= rr < h and c_lo <= cc < c_hi):
                continue
            if seen[rr, cc] or int(g[rr, cc]) != color or int(g[rr, cc]) == bg:
                continue
            seen[rr, cc] = True
            stack.append((rr, cc))
    return cells


def _find_rectangles(
    g: np.ndarray,
    sep_col: int,
    src_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    h, w = g.shape
    c_lo, c_hi = (0, sep_col) if src_on_left else (sep_col + 1, w)
    seen = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int, int, int, int]] = []

    for r in range(h):
        for c in range(c_lo, c_hi):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            cells = _collect_component(g, r, c, bg, c_lo, c_hi, seen)
            rows = [rr for rr, _cc in cells]
            cols = [cc for _rr, cc in cells]
            rr0, rr1 = min(rows), max(rows)
            cc0, cc1 = min(cols), max(cols)
            color = int(g[r, c])
            block = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if not np.all(block == color):
                continue
            dist = (sep_col - 1) - cc1 if src_on_left else cc0 - (sep_col + 1)
            area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
            rects.append((dist, rr0, rr1, cc0, cc1, area, color))

    rects.sort(key=lambda t: (t[0], t[1], t[3]))
    return rects


"""
Overview:
  Detect source-side rectangles, infer a shared row-wise spacing from rectangle
  areas, and paint the opposite side nearest-first without overwriting.

Input structure:
  The grid contains one mostly full vertical separator column. All non-separator
  source objects lie on exactly one side of it and are solid monochrome
  rectangles. The other side is empty in the input.

Rule:
  For each row, collect the areas of every source rectangle that touches that
  row. Their GCD is the painting step for that row. Then process rectangles from
  nearest to farthest from the separator; each rectangle paints periodic dots on
  every row it occupies using that row's shared step, leaving existing non-bg
  pixels unchanged.

Key insight:
  Width alone is insufficient because 2x3 and 3x2 rectangles must behave the
  same when isolated. The repeat interval is encoded by rectangle area, and when
  multiple rectangles share a row the interval is their area GCD.
"""
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    src_on_left = left_mass >= right_mass

    rects = _find_rectangles(g, sep_col, src_on_left, bg)
    row_steps: dict[int, int] = {}
    for r in range(h):
        areas = [area for _dist, r0, r1, _c0, _c1, area, _color in rects if r0 <= r <= r1]
        if areas:
            row_steps[r] = _gcd_many(areas)

    for _dist, r0, r1, _c0, _c1, _area, color in rects:
        for r in range(r0, r1 + 1):
            step = row_steps.get(r, 0)
            if step <= 0:
                continue
            if src_on_left:
                positions = range(sep_col + 1, w, step)
            else:
                positions = range(sep_col - 1, -1, -step)
            for c in positions:
                if int(out[r, c]) == bg:
                    out[r, c] = color

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 9, 13
W_MIN, W_MAX = 15, 20

SRC_SIDE_MIN, SRC_SIDE_MAX = 6, 8
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 13

PALETTE_MIN, PALETTE_MAX = 5, 8

P_SEP_GAP = 0.2
P_SWAP_MULTI_WIDTHS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(pool)) + 1))
    palette = list(map(int, rng.choice(pool, size=size, replace=False)))
    return sep_color, palette


def _place_bands(rng: np.random.Generator, h: int, heights: list[int]) -> list[tuple[int, int]]:
    band_count = len(heights)
    mandatory = sum(heights) + (band_count - 1)
    slack = h - mandatory
    if slack < 0:
        raise ValueError("band layout does not fit the canvas height")
    extras = rng.multinomial(slack, np.full(band_count + 1, 1.0 / (band_count + 1)))

    bands: list[tuple[int, int]] = []
    cursor = int(extras[0])
    for idx, height in enumerate(heights):
        r0 = cursor
        r1 = r0 + height - 1
        bands.append((r0, r1))
        cursor = r1 + 1
        if idx < band_count - 1:
            cursor += 1 + int(extras[idx + 1])
    return bands


def _build_band_specs(rng: np.random.Generator) -> list[dict[str, object]]:
    specs = [
        {"height": 2, "widths": [3], "kind": "anchor_2x3"},
        {"height": 3, "widths": [2], "kind": "anchor_3x2"},
        {
            "height": 2,
            "widths": [2, 3] if rng.random() < P_SWAP_MULTI_WIDTHS else [3, 2],
            "kind": "gcd_band",
        },
    ]
    rng.shuffle(specs)
    for spec in specs:
        spec["sep_gap"] = 1 if rng.random() < P_SEP_GAP else 0
    return specs


def _place_rectangles(
    out: np.ndarray,
    bands: list[tuple[int, int]],
    band_specs: list[dict[str, object]],
    colors: list[int],
    sep_col: int,
    src_on_left: bool,
) -> list[tuple[int, int, int, int, int]]:
    rects: list[tuple[int, int, int, int, int]] = []
    color_idx = 0

    for (r0, r1), spec in zip(bands, band_specs):
        widths = list(spec["widths"])
        sep_gap = int(spec["sep_gap"])
        if src_on_left:
            cursor = sep_col - sep_gap
            for width in widths:
                color = int(colors[color_idx])
                color_idx += 1
                c1 = cursor - 1
                c0 = c1 - int(width) + 1
                out[r0 : r1 + 1, c0 : c1 + 1] = color
                rects.append((r0, r1, c0, c1, color))
                cursor = c0
        else:
            cursor = sep_col + 1 + sep_gap
            for width in widths:
                color = int(colors[color_idx])
                color_idx += 1
                c0 = cursor
                c1 = c0 + int(width) - 1
                out[r0 : r1 + 1, c0 : c1 + 1] = color
                rects.append((r0, r1, c0, c1, color))
                cursor = c1 + 1

    return rects


def _row_steps_from_rects(h: int, rects: list[tuple[int, int, int, int, int]]) -> dict[int, int]:
    row_steps: dict[int, int] = {}
    for r in range(h):
        areas = []
        for r0, r1, c0, c1, _color in rects:
            if r0 <= r <= r1:
                areas.append((r1 - r0 + 1) * (c1 - c0 + 1))
        if areas:
            row_steps[r] = _gcd_many(areas)
    return row_steps


def _paint_extension_combo(
    out: np.ndarray,
    rects: list[tuple[int, int, int, int, int]],
    sep_col: int,
    src_on_left: bool,
) -> None:
    h, w = out.shape
    row_steps = _row_steps_from_rects(h, rects)
    ordered = []
    for r0, r1, c0, c1, color in rects:
        dist = (sep_col - 1) - c1 if src_on_left else c0 - (sep_col + 1)
        ordered.append((dist, r0, c0, r1, c1, color))
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))

    for _dist, r0, _c0, r1, _c1, color in ordered:
        for r in range(r0, r1 + 1):
            step = row_steps[r]
            if src_on_left:
                positions = range(sep_col + 1, w, step)
            else:
                positions = range(sep_col - 1, -1, -step)
            for c in positions:
                if out[r, c] == 0:
                    out[r, c] = color


def _paint_extension_area_only(
    out: np.ndarray,
    rects: list[tuple[int, int, int, int, int]],
    sep_col: int,
    src_on_left: bool,
) -> None:
    h, w = out.shape
    ordered = []
    for r0, r1, c0, c1, color in rects:
        dist = (sep_col - 1) - c1 if src_on_left else c0 - (sep_col + 1)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        ordered.append((dist, r0, c0, r1, area, color))
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))

    for _dist, r0, _c0, r1, area, color in ordered:
        for r in range(r0, r1 + 1):
            if src_on_left:
                positions = range(sep_col + 1, w, area)
            else:
                positions = range(sep_col - 1, -1, -area)
            for c in positions:
                if out[r, c] == 0:
                    out[r, c] = color


def _nontrivial_gcd_row_count(h: int, rects: list[tuple[int, int, int, int, int]]) -> int:
    count = 0
    for r in range(h):
        areas = []
        for r0, r1, c0, c1, _color in rects:
            if r0 <= r <= r1:
                areas.append((r1 - r0 + 1) * (c1 - c0 + 1))
        if len(areas) >= 2:
            step = _gcd_many(areas)
            if 1 < step < min(areas):
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(SRC_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(SRC_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    src_side_len = int(rng.integers(lo, hi + 1))
    src_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = src_side_len if src_on_left else (w - 1 - src_side_len)

    sep_color, palette = _sample_palette(rng)
    band_specs = _build_band_specs(rng)
    heights = [int(spec["height"]) for spec in band_specs]
    bands = _place_bands(rng, h, heights)

    rect_count = sum(len(spec["widths"]) for spec in band_specs)
    if rect_count > len(palette):
        return None

    out_base = np.zeros((h, w), dtype=int)
    out_base[:, sep_col] = sep_color

    active_rows = {r for r0, r1 in bands for r in range(r0, r1 + 1)}
    inactive_rows = [r for r in range(h) if r not in active_rows]
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out_base[gap_row, sep_col] = 0

    rect_colors = list(map(int, rng.permutation(palette)[:rect_count]))
    rects = _place_rectangles(out_base, bands, band_specs, rect_colors, sep_col, src_on_left)

    out_combo = out_base.copy()
    _paint_extension_combo(out_combo, rects, sep_col, src_on_left)

    out_area_only = out_base.copy()
    _paint_extension_area_only(out_area_only, rects, sep_col, src_on_left)

    inp = out_base.copy()
    if src_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if _nontrivial_gcd_row_count(h, rects) == 0:
        return None
    if np.array_equal(out_combo, out_area_only):
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out_combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
