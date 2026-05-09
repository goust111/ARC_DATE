# Combo variant of 136b0064 (index 23)
# combo_spec: D6/1+D7/3
# Name: 无分隔线的单区域设计

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the fused variant
# a) Concrete behavior and trace
#    - Original task:
#      input is Hx15, the left 7 columns hold the glyph bank, col 7 is a full-height divider,
#      and the lone gray start sits on the right. The solver first removes the divider and shifts
#      the gray point back into the left 7-column canvas.
#    - Fused task:
#      input is directly the left canvas, so the divider disappears and there is no right-side
#      region at all. Glyphs and the gray start now coexist in one Hx7 field.
#    - Small example with N=2, H=7:
#      original layout might place the gray point at input (0,10), which maps to output start
#      (0,2) after cutting away the divider/right offset. In the fused version we instead place
#      the gray point directly at (1,3) inside the same 7-column glyph canvas. The glyph order
#      and area->segment decoding stay unchanged, but the solver must now detect "one singleton
#      gray marker" and "several 3x3 monochrome glyph components" from a single region.
# b) Complexity calibration
#    - The fusion changes the global layout strongly but keeps the instruction alphabet small:
#      the same four glyph templates, the same strict shape-color binding, and the same
#      column-major reading order remain in force.
#    - To keep the shared-region effect visible without making the input messy, the generator
#      always embeds the gray point into the middle blank column c=3 and chooses a row inside a
#      glyph band (row mod 4 in {0,1,2}). Visually, the start point sits between the two glyph
#      stacks instead of being isolated in a separate panel.
# c) Hidden-assumption audit of the original solver
#    - `find_full_height_constant_col(...)` and every use of `div_c` depended on a unique divider.
#      That assumption is invalid now and must be removed entirely.
#    - `c_cur = gray_col - (div_c + 1)` assumed the start point lives in a translated right panel.
#      That is invalid now; the gray coordinate is already the true drawing coordinate.
#    - Slicing `left = g[:, :div_c]` assumed every instruction lies strictly left of the divider.
#      That is invalid now; glyph detection must scan the whole single-region canvas.
#    - The useful assumption that every instruction glyph is still a monochrome 3x3, 8-connected
#      component remains valid and is preserved by the generator.
# d) Placement legality
#    - Glyph placement still uses the original non-overlapping 3x3 tile scaffold at c0 in {0,4}
#      and r0 spaced by 4, so glyph components stay easy to recover.
#    - The generator is path-first: it samples an instruction sequence that stays within the 7-column
#      canvas, then chooses a non-top start row whose shifted path still fits under the bottom edge.
#    - The gray point is always placed at c=3, which is the permanent spacer column between glyph
#      banks, so it never overlaps any glyph cell.
# e) Visual observability
#    - From a few examples, a human sees that the divider is gone, the whole input is only 7 columns
#      wide, and the gray point is embedded inside the glyph field rather than off to the side.
#    - Because the gray point remains unique and every true instruction is still a repeated 3x3
#      pattern, the new parsing rule is visually recoverable.
# f) Visual cleanliness
#    - The input keeps the original sparse two-column tile grid, with only one extra singleton in the
#      central spacer column. No overlapping markers or dense clutter are introduced.
# g) Change inferability
#    - No hidden color ordering or numeric-color trick is added. The only new inference is structural:
#      identify the lone gray singleton as the start and the 3x3 colored components as instructions.
# 9) New solver assumptions and generator guarantees
#    - `main()` assumes exactly one gray cell exists and every actionable glyph is a monochrome 3x3
#      component whose area is one of {4,5,6,7}.
#    - `generate_input()` guarantees those conditions, guarantees width 7 with no divider column,
#      and guarantees the start row is non-top so the sample distribution does not collapse back to
#      the original "top-row start in a separate panel" appearance.


LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
START_COL = 3

N_MIN, N_MAX = 2, 5
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 500
AREA_SAMPLE_WEIGHTS = {4: 1, 5: 1, 6: 3, 7: 1}
PATH_ATTEMPTS_PER_N = 80

AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_DCOL = {4: 0, 5: 1, 6: -1, 7: -1}
AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
AREA_TO_ROW_ADVANCE = {4: 2, 5: 1, 6: 1, 7: 1}
AREAS = (4, 5, 6, 7)

TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}


def _extract_glyphs(grid: np.ndarray, *, bg: int) -> list[dict[str, int]]:
    """Extract all instruction glyphs from the single shared canvas."""
    glyphs: list[dict[str, int]] = []
    for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {int(bg), GRAY}:
            continue
        bbox = comp["bbox"]
        area = int(comp["area"])
        if bbox_h(bbox) != 3 or bbox_w(bbox) != 3 or area not in AREA_TO_LEN:
            continue
        r0, c0, _r1, _c1 = map(int, bbox)
        glyphs.append({"color": color, "r0": r0, "c0": c0, "area": area})
    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def _draw_trace(
    height: int,
    width: int,
    start_row: int,
    start_col: int,
    glyphs: list[dict[str, int]],
) -> np.ndarray:
    """Shared tracer for both solver execution and generator validation."""
    out = np.zeros((height, width), dtype=int)
    r_cur = int(start_row)
    c_cur = int(start_col)
    out[r_cur, c_cur] = GRAY

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN[area])

        r_cur += 1
        assert r_cur < height, f"segment exits bottom: r_cur={r_cur}, H={height}"
        out[r_cur, c_cur] = color

        if area == 4:
            for _ in range(seg_len - 1):
                r_cur += 1
                assert r_cur < height, f"vertical segment exits bottom: r_cur={r_cur}, H={height}"
                out[r_cur, c_cur] = color
        else:
            dc = int(AREA_TO_DCOL[area])
            for _ in range(seg_len - 1):
                c_cur += dc
                assert 0 <= c_cur < width, f"horizontal segment exits canvas: c_cur={c_cur}, W={width}"
                out[r_cur, c_cur] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the colored polyline by decoding 3x3 glyphs from a single shared canvas.

    Input structure:
    - The whole input is already the 7-column drawing canvas; there is no divider and no separate right panel.
    - Colored 3x3 glyphs occupy the usual tile slots, and exactly one gray (5) cell marks the start directly inside this same region.

    Rule:
    - Find the lone gray start cell and keep it at its in-canvas coordinate.
    - Collect every non-gray monochrome 3x3 component, read them in column-major order (left column top→bottom, then right column top→bottom).
    - Decode each glyph by pixel area: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2; before every segment, step down by one row.

    Key insight:
    - The start marker is the only singleton gray pixel, while every instruction is a full 3x3 monochrome component, so the solver can separate "where to start" from "what to draw" even though they share one undivided region.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    gray_cells = np.argwhere(g == GRAY)
    assert gray_cells.shape[0] == 1, f"expected exactly one gray start, found {gray_cells.shape[0]}"
    start_row, start_col = map(int, gray_cells[0])

    glyphs = _extract_glyphs(g, bg=bg)
    return _draw_trace(h, w, start_row, start_col, glyphs)


def _sample_areas(n: int) -> list[int] | None:
    """Sample a valid area sequence from the canonical center start column."""
    h = 4 * n - 1
    r = 0
    c = START_COL
    areas: list[int] = []

    for _ in range(2 * n):
        options = [
            area
            for area in AREAS
            if r + AREA_TO_ROW_ADVANCE[area] < h and 0 <= c + AREA_TO_DCOL[area] * (AREA_TO_LEN[area] - 1) < LEFT_W
        ]
        if not options:
            return None
        area = random.choices(options, weights=[AREA_SAMPLE_WEIGHTS[area] for area in options], k=1)[0]
        areas.append(area)
        r += AREA_TO_ROW_ADVANCE[area]
        c += AREA_TO_DCOL[area] * (AREA_TO_LEN[area] - 1)

    types = [area == 4 for area in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if r > h - 2:
        return None
    return areas


def _choose_start_row(height: int, areas: list[int]) -> int | None:
    """Choose a non-top embedded row for the in-canvas gray start."""
    total_advance = sum(AREA_TO_ROW_ADVANCE[area] for area in areas)
    max_start_row = height - 1 - total_advance
    if max_start_row < 1:
        return None

    preferred = [row for row in range(1, max_start_row + 1) if row % TILE_ROW_STRIDE != 3]
    if not preferred:
        return None
    return int(random.choice(preferred))


def _render_grid(n: int, start_row: int, areas: list[int]) -> np.ndarray:
    """Stamp the single-region input grid with embedded gray start and tiled glyphs."""
    h = 4 * n - 1
    grid = np.zeros((h, LEFT_W), dtype=int)
    grid[start_row, START_COL] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = TEMPLATES[area] * AREA_TO_COLOR[area]
    return grid


def _gray_is_visibly_embedded(grid: np.ndarray) -> bool:
    gray = np.argwhere(grid == GRAY)
    if gray.shape[0] != 1:
        return False
    r, c = map(int, gray[0])
    return (
        c == START_COL
        and r > 0
        and r % TILE_ROW_STRIDE != 3
        and np.any(grid[r, :START_COL] != 0)
        and np.any(grid[r, START_COL + 1 :] != 0)
    )


def _gen_attempt(n: int) -> np.ndarray | None:
    for _ in range(PATH_ATTEMPTS_PER_N):
        areas = _sample_areas(n)
        if areas is None:
            continue

        h = 4 * n - 1
        start_row = _choose_start_row(h, areas)
        if start_row is None:
            continue

        grid = _render_grid(n, start_row, areas)
        if _gray_is_visibly_embedded(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            n = random.choice(tuple(range(N_MIN, N_MAX + 1)))
            inp = _gen_attempt(n)
            if inp is None:
                continue
            out = main(inp)
            if out.shape != inp.shape:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
