# Combo variant of 136b0064 (index 5)
# combo_spec: D4/0+D8/1
# Name: 底部紧凑向上折线

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
#    - The input layout stays structurally familiar: 3x3 glyphs live in two tile
#      columns on the left of a full-height divider, and one gray start marker
#      sits alone on the right. The two fused changes are entirely about how the
#      decoded path is drawn.
#    - D4 moves the anchor to the bottom and reverses vertical growth. D8 then
#      removes the inter-glyph row step, so every segment starts exactly on the
#      previous endpoint.
#    - Concrete trace with shifted start at (10, 3) and areas [5, 4, 7]:
#        original task: step to (11,3), draw right-3 to (11,5);
#                       step to (12,5), draw down-2 to (13,5);
#                       step to (14,5), draw left-2 to (14,4)
#        D4 only:       step to (9,3),  draw right-3 to (9,5);
#                       step to (8,5),  draw up-2 to (7,5);
#                       step to (6,5),  draw left-2 to (6,4)
#        combo:         draw right-3 from (10,3) to (10,5);
#                       draw up-2    from (10,5) to (9,5);
#                       draw left-2  from (9,5)  to (9,4)
#      The fused result is visibly tighter than D4 alone and clearly inverted
#      relative to the original downward staircase.
#
# b) Complexity calibration
#    - Keep the same four glyph templates and strict shape->color binding. That
#      preserves repeated local evidence and prevents the combo from becoming a
#      new symbolic lookup task.
#    - The generator requires at least one vertical glyph, at least one
#      horizontal glyph, and at least two vertical/horizontal alternations. That
#      makes both "upward growth" and "shared-endpoint compactness" visible in
#      each sample instead of allowing nearly straight degenerate traces.
#
# c) Implicit-assumption audit for main()
#    - The original `r_cur += 1` encoded a mandatory downward step. The combo
#      invalidates it twice over: growth is upward and the step disappears, so no
#      row advance should happen before a glyph.
#    - The original area-4 branch extended downward. Under D4 it must flip to
#      upward extension while horizontal branches keep their old left/right
#      meaning.
#    - The original output could leave the gray pixel visible because drawing
#      began on the next row. Under D8 the first segment starts on the anchor, so
#      `out[r_cur, c_cur]` is immediately recolored.
#    - Bounds now rely on a shared-endpoint walk whose total deltas are
#      area4:(-1,0), area5:(0,+2), area6:(0,-3), area7:(0,-1), not the source
#      task's stepped deltas.
#
# d) Placement legality
#    - Glyph tiles are still stamped into disjoint 3x3 slots, so component
#      extraction remains unambiguous.
#    - Path generation backtracks inside the 7-column output canvas and forbids
#      revisiting any painted cell except the intentional shared junction cell at
#      the start of the next segment. This prevents self-overlap and out-of-
#      bounds traces.
#
# e) Visual observability
#    - A human can read the combo from a few examples because the gray input
#      marker is always on the bottom row, while the output path stays compact
#      around shared corner cells instead of leaving blank spacer rows.
#    - Ordering still comes only from visible tile placement (column-major over
#      the two glyph columns); no hidden color-id sorting or metadata is used.
#
# f) Visual cleanliness
#    - The right side remains almost empty except for the single gray start, and
#      the left glyph bank keeps its original spacing. The output polyline is
#      compact but simple because only segment endpoints may touch existing path
#      cells.
#
# g) Change inferability
#    - Colors do not acquire any new meaning. Each color is still tied to one
#      visible glyph shape, so the examples only need to teach the spatial fusion
#      rule: start from the bottom and connect segments directly.
#
# Post-change assumptions guaranteed by the generator
#    - exactly one full-height non-background divider column exists;
#    - all decoded glyphs are monochromatic 3x3 components in the two tile
#      columns and are sortable by visible position alone;
#    - exactly one gray start exists on the right and lies on the bottom row;
#    - every compact upward segment is in-bounds and the combo output differs
#      from the stepped-upward baseline on the same input.
# ---------------------------------------------------------------------------


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5

# --- shape grammar ---
AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_HDIR = {5: 1, 6: -1, 7: -1}
AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREAS = (4, 5, 6, 7)

# Shared-endpoint total displacement per glyph for the upward combo walk.
_AREA_DELTA = {4: (-1, 0), 5: (0, 2), 6: (0, -3), 7: (0, -1)}

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 3x3 glyph instructions and draw a compact polyline that rises from a bottom anchor.

    Input structure:
    - Left of a full-height divider: monochromatic 3x3 glyphs placed in two tile columns on background.
    - Right of the divider: one gray (5) start pixel located on the bottom row.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read the glyphs in visible column-major order (left column top→bottom, then right column top→bottom).
    - Each glyph's pixel area encodes one segment: area 4 → up 2, area 5 → right 3, area 6 → left 4,
      area 7 → left 2.
    - Draw the segments with no inter-glyph step: each segment starts on the current endpoint, so the
      first segment overwrites the gray anchor cell.

    Key insight: The glyph shape still determines the move, but the cursor logic has fused two changes:
    the reference frame is bottom-to-top and the endpoint is reused directly, producing a tight upward
    turning path instead of a spaced descending staircase.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((h, int(div_c)), bg, dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": int(col), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN.get(area, 2))

        out[int(r_cur), int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur - (seg_len - 1))
            assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur - 1)
                out[int(r_cur), int(c_cur)] = color
        else:
            dir_h = int(AREA_TO_HDIR.get(area, 1))
            end_c = int(c_cur + dir_h * (seg_len - 1))
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[int(r_cur), int(c_cur)] = color

    return out


def _segment_cells_up(area: int, r: int, c: int) -> list[tuple[int, int]]:
    seg_len = int(AREA_TO_LEN[area])
    if area == 4:
        return [(int(r - i), int(c)) for i in range(seg_len)]
    dir_h = int(AREA_TO_HDIR[area])
    return [(int(r), int(c + dir_h * i)) for i in range(seg_len)]


def _render_output_compact_up(h: int, start_col: int, areas: list[int]) -> np.ndarray:
    out = np.zeros((h, LEFT_W), dtype=int)
    r_cur, c_cur = h - 1, start_col
    out[int(r_cur), int(c_cur)] = GRAY

    for area in areas:
        color = int(AREA_TO_COLOR[area])
        cells = _segment_cells_up(area, r_cur, c_cur)
        for rr, cc in cells:
            out[int(rr), int(cc)] = color
        r_cur, c_cur = map(int, cells[-1])

    return out


def _render_output_stepped_up(h: int, start_col: int, areas: list[int]) -> np.ndarray | None:
    """Counterfactual D4-only rendering kept for generator anti-degeneracy checks."""
    out = np.zeros((h, LEFT_W), dtype=int)
    r_cur, c_cur = h - 1, start_col
    out[int(r_cur), int(c_cur)] = GRAY

    for area in areas:
        color = int(AREA_TO_COLOR[area])
        seg_len = int(AREA_TO_LEN[area])

        r_cur = int(r_cur - 1)
        if area == 4:
            cells = [(int(r_cur - i), int(c_cur)) for i in range(seg_len)]
        else:
            dir_h = int(AREA_TO_HDIR[area])
            cells = [(int(r_cur), int(c_cur + dir_h * i)) for i in range(seg_len)]

        if any(not (0 <= rr < h and 0 <= cc < LEFT_W) for rr, cc in cells):
            return None
        for rr, cc in cells:
            out[int(rr), int(cc)] = color
        r_cur, c_cur = map(int, cells[-1])

    return out


def _count_direction_changes(areas: list[int]) -> int:
    kinds = [area == 4 for area in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _sample_path(n: int, start_col: int, required_area: int) -> list[int] | None:
    """Backtrack a simple upward shared-endpoint walk of length 2*n."""
    h = 4 * n - 1
    target_len = 2 * n
    painted: set[tuple[int, int]] = {(h - 1, start_col)}
    areas: list[int] = []

    def rec(r: int, c: int) -> bool:
        if len(areas) == target_len:
            if 4 not in areas:
                return False
            if all(area == 4 for area in areas):
                return False
            if required_area not in areas:
                return False
            return _count_direction_changes(areas) >= MIN_DIRECTION_CHANGES

        opts = list(_AREAS)
        random.shuffle(opts)
        for area in opts:
            cells = _segment_cells_up(area, r, c)
            if any(not (0 <= rr < h and 0 <= cc < LEFT_W) for rr, cc in cells):
                continue
            new_cells = cells[1:]
            if any(cell in painted for cell in new_cells):
                continue

            areas.append(int(area))
            for cell in new_cells:
                painted.add(cell)

            r_next, c_next = map(int, cells[-1])
            if rec(r_next, c_next):
                return True

            for cell in new_cells:
                painted.remove(cell)
            areas.pop()
        return False

    if rec(h - 1, start_col):
        return list(areas)
    return None


def _render_grid(n: int, start_col: int, areas: list[int]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[h - 1, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[area] * AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = random.randint(N_MIN, N_MAX)
    h = 4 * n - 1
    required_area = random.choice(_AREAS)

    start_cols = list(range(LEFT_W))
    random.shuffle(start_cols)
    for start_col in start_cols:
        areas = _sample_path(n, start_col, required_area)
        if areas is None:
            continue
        inp = _render_grid(n, start_col, areas)
        expected = _render_output_compact_up(h, start_col, areas)
        stepped = _render_output_stepped_up(h, start_col, areas)
        if stepped is not None and np.array_equal(expected, stepped):
            continue
        return inp, expected, stepped
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected, stepped = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            if stepped is not None and np.array_equal(out, stepped):
                continue
            gray = np.argwhere(inp == GRAY)
            if gray.shape != (1, 2) or int(gray[0, 0]) != inp.shape[0] - 1:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
