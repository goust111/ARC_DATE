# Variant of 136b0064
# Dimension: D8 — 字形间的步进规则 (multi_step)
# Change: 每处理一个字形时，先向下移动1行（固定步进），再执行该字形对应的线段指令... → 无步进——线段直接首尾相连

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# a) Concretization and trace
#    - Input stays the same: a 7-column left glyph panel, a full-height divider
#      at col 7, and one gray start pixel on the right.
#    - Output still lives on the 7-column left canvas, but the polyline is now
#      drawn as a continuous chain. The key physical change is that glyph k+1
#      starts on glyph k's endpoint instead of on the row below it.
#    - Concrete trace from start (0,3), glyph areas [5,4,7]:
#        old rule:  right-3 from (1,3) -> endpoint (1,5),
#                   then down-2 from (2,5) -> endpoint (3,5),
#                   then left-2 from (4,5) -> endpoint (4,4)
#        new rule:  right-3 from (0,3) -> endpoint (0,5),
#                   then down-2 from (0,5) -> endpoint (1,5),
#                   then left-2 from (1,5) -> endpoint (1,4)
#      The difference is visible as a compact turning path rather than a
#      staircase with forced blank-row drops between segments.
#
# b) Complexity calibration
#    - Keep the original four glyph templates and strict area->color binding so
#      the task still has repeated local evidence and does not introduce new
#      ambiguity.
#    - To make D8 visible, generator requires both vertical and horizontal
#      instructions and at least two direction changes. This avoids nearly
#      straight traces that would hide the new connection rule.
#
# c) Old assumptions that break under D8
#    - `r_cur += 1` before every glyph encoded the fixed downward step; this
#      must be removed.
#    - The old solver could safely leave the gray start visible forever because
#      the first colored segment began one row below it. Under direct
#      connection, the first segment starts at the gray cell and overwrites it.
#    - Old generator deltas assumed every glyph consumed one extra row; those
#      deltas must shrink to shared-endpoint motion:
#        area 4: (+1,0), area 5: (0,+2), area 6: (0,-3), area 7: (0,-1).
#
# d) Placement legality
#    - Glyph placement is unchanged, so 3x3 components still never overlap.
#    - Path sampling now enforces that every new segment stays inside the
#      7-column output canvas and that only the segment's first cell may touch
#      an already painted cell. This guarantees a simple polyline with clean
#      junctions and no accidental self-overwrite loops.
#
# e) Visual observability
#    - Humans can see the new rule because the output row span is much smaller
#      than in the original task and turns happen at shared corner cells.
#    - The generator does not rely on hidden sorting by color id or any
#      non-visible metadata; order is still read from the visible tile layout.
#
# f) Visual cleanliness
#    - The no-self-overlap constraint keeps the compact trace readable even
#      though junction cells are recolored by the next segment.
#    - Input remains neatly partitioned: glyphs left, divider middle, one gray
#      marker right.
#
# g) Color inferability
#    - Colors are unchanged and still tied to visible glyph shapes, so examples
#      provide repeated support for every color/shape pair.
#
# Post-change solver assumptions checked against the generator:
#    - exactly one full-height non-bg divider column;
#    - glyph components are monochromatic 3x3 objects in the two tile columns;
#    - one gray start exists on the right and shifts into col range 0..6;
#    - every decoded segment remains in-bounds when drawn with shared endpoints.
# ---------------------------------------------------------------------------


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a compact polyline by decoding 3x3 glyph instructions across a divided grid.

    Input structure:
    - Left of a full-height divider: colored 3x3 glyphs arranged in two tile columns on background.
    - Right of the divider: one gray (5) start pixel that anchors the walk.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read glyphs in visible column-major order (left column top→bottom, then right column top→bottom).
    - Each glyph's pixel area encodes one segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - Draw segments consecutively with no extra step between glyphs: each new segment starts on the previous endpoint.

    Key insight: The glyph shape still determines the segment, but the path state has changed — the cursor is reused directly, so junction cells become the shared hinges of the polyline.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
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

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        out[int(r_cur), int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur + (seg_len - 1))
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[int(r_cur), int(c_cur)] = color
        else:
            dir_h = int(area_to_dir.get(area, 1))
            end_c = int(c_cur + dir_h * (seg_len - 1))
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[int(r_cur), int(c_cur)] = color

    return out


# --------------- generator ---------------

# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)

# Shared-endpoint total displacement per glyph.
_AREA_DELTA = {4: (1, 0), 5: (0, 2), 6: (0, -3), 7: (0, -1)}

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 300


def _segment_cells(area: int, r: int, c: int) -> list[tuple[int, int]]:
    seg_len = int(_AREA_TO_LEN[area])
    if area == 4:
        return [(int(r + i), int(c)) for i in range(seg_len)]
    dir_h = int(_AREA_TO_DIR[area])
    return [(int(r), int(c + dir_h * i)) for i in range(seg_len)]


def _render_output(H: int, start_col: int, areas: list[int]) -> np.ndarray:
    out = np.zeros((H, LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, start_col
    out[int(r_cur), int(c_cur)] = GRAY

    for area in areas:
        color = int(_AREA_TO_COLOR[area])
        cells = _segment_cells(area, r_cur, c_cur)
        for rr, cc in cells:
            out[int(rr), int(cc)] = color
        r_cur, c_cur = map(int, cells[-1])

    return out


def _count_direction_changes(areas: list[int]) -> int:
    kinds = [area == 4 for area in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Backtrack a simple shared-endpoint walk of length 2*N inside the left canvas."""
    H = 4 * N - 1
    target_len = 2 * N
    painted: set[tuple[int, int]] = {(GRAY_START_ROW, start_col)}
    areas: list[int] = []

    def rec(r: int, c: int) -> bool:
        if len(areas) == target_len:
            if 4 not in areas or all(area == 4 for area in areas):
                return False
            return _count_direction_changes(areas) >= MIN_DIRECTION_CHANGES

        opts = list(_AREAS)
        random.shuffle(opts)
        for area in opts:
            cells = _segment_cells(area, r, c)
            if any(not (0 <= rr < H and 0 <= cc < LEFT_W) for rr, cc in cells):
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

    if rec(GRAY_START_ROW, start_col):
        return list(areas)
    return None


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1

    start_cols = list(range(LEFT_W))
    random.shuffle(start_cols)
    for start_col in start_cols:
        areas = _sample_path(N, start_col)
        if areas is None:
            continue
        inp = _render_grid(N, start_col, areas)
        expected = _render_output(H, start_col, areas)
        return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
