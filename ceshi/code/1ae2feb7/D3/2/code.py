# Variant of 1ae2feb7
# Dimension: D3 — 射线源对象类型 (src)
# Change: 水平单色色条——分隔条一侧的水平连续同色像素段，每行1-2个色条，紧邻分隔条排列... → 矩形色块

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

"""
Design analysis for the D3 variant
==================================

spec:
  grid:
    size: 8-13 rows x 15-20 cols
    bg: 0
  separator:
    same as the source task: one vertical column of a single non-bg color,
    present in at least H-1 rows
  source side:
    all source objects live on one side of the separator
    objects are solid monochrome rectangles instead of 1D bars
    active layout is organized as 2-3 horizontal bands with >=1 empty row
    between bands
    each band has height 2 or 3
    each band contains 1-2 adjacent rectangles ordered near->far from the
    separator; the nearest rectangle may leave a 1-cell gap to the separator
  transform:
    process rectangles from closest to separator outward, never overwriting
    step/period = rectangle area = height * width
    each rectangle paints periodic dots across every row it occupies on the
    opposite side of the separator

a) Concrete behavior and trace
  Original rule:
    a 1xL bar on row r painted cells on the opposite side at offsets
    0, L, 2L, ...
  Variant rule:
    an HxW rectangle paints on each of its H rows at offsets
    0, H*W, 2*H*W, ...
  Example:
    separator at c=5, source on the left.
    A 2x3 rectangle occupying rows 1-2 and cols 2-4 touches the separator.
    Original bar logic would infer step 3 from each row-run.
    Variant logic infers step 6 from the 2D object, so rows 1 and 2 receive
    paint at cols 6, 12, ... instead of 6, 9, 12, ...
    This is the core D3 difference.

b) Complexity calibration
  The variant uses only one new controlling attribute: area. To make area
  visibly inferable, the generator deliberately includes both a 2x3 rectangle
  and a 3x2 rectangle in most samples. They have different width/height but the
  same area, so they cast the same period. This keeps the rule compact while
  making the 2D source type matter.

c) Hidden assumptions from the original main()
  - Row-wise run detection no longer works: a single row only reveals width,
    not area, so the variant must recover connected 2D rectangles.
  - "length = c1-c0+1" was safe only because the source objects were bars.
    Here it would silently collapse the task back to the original behavior.
  - Painting only one row per source object is no longer valid; each rectangle
    must project across all rows in its height.
  - Sorting by distance from the separator remains valid and is preserved.

d) Placement legality
  The generator guarantees:
  - rectangles are axis-aligned solid bboxes
  - bands do not overlap vertically and have a blank row between them
  - rectangles inside a band do not overlap and remain within the source side
  - the separator color is disjoint from rectangle colors
  - the target side is blank in the input

e) Visual observability
  The change is visible because source objects are clearly 2D blocks rather
  than 1D runs, and their output repeats as vertical stacks of dots across
  several rows. The 2x3 vs 3x2 pair makes "area controls period" observable
  without relying on color ids or hidden metadata.

f) Visual cleanliness
  Only one object layer exists on the source side plus one dot layer on the
  target side. Band spacing prevents source-side clutter; non-overwrite keeps
  target-side overlaps readable.

g) Color reasoning
  Colors are preserved; there is no color mapping. Distinct rectangle colors
  are used only to separate objects visually.

Post-change solver assumptions reviewed against the generator
-------------------------------------------------------------
  - _find_sep_col assumes one mostly full monochrome vertical column.
    The generator always creates that.
  - _find_rectangles assumes each non-bg connected component on the source side
    is a solid rectangle. The generator enforces solid rectangles and unique
    colors per rectangle so no two components merge accidentally.
  - main() assumes the target side is blank in the input; the generator clears
    that side after constructing the output-first example.
  - main() paints by area. The generator limits rectangle areas to {2,3,4,6},
    so each rectangle usually leaves at least two marks on the target side.
"""


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


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
    for _dist, r0, r1, _c0, _c1, area, color in rects:
        if area <= 0:
            continue
        if src_on_left:
            positions = range(sep_col + 1, w, area)
        else:
            positions = range(sep_col - 1, -1, -area)
        for r in range(r0, r1 + 1):
            for c in positions:
                if int(out[r, c]) == bg:
                    out[r, c] = color

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SRC_SIDE_MIN, SRC_SIDE_MAX = 5, 7
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 14

PALETTE_MIN, PALETTE_MAX = 6, 8

P_THIRD_BAND = 0.6
P_TWO_RECTS = 0.55
P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(pool)) + 1))
    palette = list(map(int, rng.choice(pool, size=size, replace=False)))
    return sep_color, palette


def _allowed_widths(height: int, cap: int) -> list[int]:
    if height == 2:
        base = [1, 2, 3]
    elif height == 3:
        base = [1, 2]
    else:
        base = list(range(1, min(3, cap) + 1))
    return [w for w in base if w <= cap]


def _sample_band_heights(rng: np.random.Generator, h: int) -> list[int]:
    heights = [2, 3]
    if h >= 10 and rng.random() < P_THIRD_BAND:
        heights.append(int(rng.choice([2, 3])))
    rng.shuffle(heights)
    return heights


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


def _sample_band_rect_specs(
    rng: np.random.Generator,
    heights: list[int],
    src_side_len: int,
) -> list[tuple[int, list[int], int]]:
    specs: list[tuple[int, list[int], int]] = []
    used_anchor_2 = False
    used_anchor_3 = False

    for height in heights:
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        cap = src_side_len - sep_gap
        if cap <= 0:
            raise ValueError("source side capacity vanished")

        if height == 2 and not used_anchor_2:
            widths = [3]
            used_anchor_2 = True
        elif height == 3 and not used_anchor_3:
            widths = [2]
            used_anchor_3 = True
        else:
            widths = [int(rng.choice(_allowed_widths(height, cap)))]

        if rng.random() < P_TWO_RECTS:
            remaining = cap - sum(widths)
            choices = _allowed_widths(height, remaining)
            if choices:
                widths.append(int(rng.choice(choices)))

        specs.append((height, widths, sep_gap))

    if not (used_anchor_2 and used_anchor_3):
        raise ValueError("failed to create the 2x3 and 3x2 evidence pair")
    return specs


def _place_rectangles(
    out: np.ndarray,
    bands: list[tuple[int, int]],
    band_specs: list[tuple[int, list[int], int]],
    colors: list[int],
    sep_col: int,
    src_on_left: bool,
) -> list[tuple[int, int, int, int, int]]:
    rects: list[tuple[int, int, int, int, int]] = []
    color_idx = 0

    for (r0, r1), (_height, widths, sep_gap) in zip(bands, band_specs):
        if src_on_left:
            cursor = sep_col - sep_gap
            for width in widths:
                color = int(colors[color_idx])
                color_idx += 1
                c1 = cursor - 1
                c0 = c1 - width + 1
                out[r0 : r1 + 1, c0 : c1 + 1] = color
                rects.append((r0, r1, c0, c1, color))
                cursor = c0
        else:
            cursor = sep_col + 1 + sep_gap
            for width in widths:
                color = int(colors[color_idx])
                color_idx += 1
                c0 = cursor
                c1 = c0 + width - 1
                out[r0 : r1 + 1, c0 : c1 + 1] = color
                rects.append((r0, r1, c0, c1, color))
                cursor = c1 + 1

    return rects


def _paint_extension(
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
        if src_on_left:
            positions = range(sep_col + 1, w, area)
        else:
            positions = range(sep_col - 1, -1, -area)
        for r in range(r0, r1 + 1):
            for c in positions:
                if out[r, c] == 0:
                    out[r, c] = color


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
    heights = _sample_band_heights(rng, h)
    bands = _place_bands(rng, h, heights)
    band_specs = _sample_band_rect_specs(rng, heights, src_side_len)
    rect_count = sum(len(widths) for _height, widths, _sep_gap in band_specs)
    if rect_count > len(palette):
        return None

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if rng.random() < P_SEP_COLUMN_GAP:
        inactive_rows = [r for r in range(h) if all(not (r0 <= r <= r1) for r0, r1 in bands)]
        if inactive_rows:
            gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
            out[gap_row, sep_col] = 0

    rect_colors = list(map(int, rng.permutation(palette)[:rect_count]))
    rects = _place_rectangles(out, bands, band_specs, rect_colors, sep_col, src_on_left)
    _paint_extension(out, rects, sep_col, src_on_left)

    inp = out.copy()
    if src_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
