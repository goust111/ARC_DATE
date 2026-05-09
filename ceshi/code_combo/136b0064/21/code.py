# Combo variant of 136b0064 (index 21)
# combo_spec: D1/0+D9/3
# Name: 三列行主序遍历

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior:
#    The original task had a 7-column left canvas with 2 tile columns at c0=0 and
#    c0=4. Glyphs were decoded column-major. This combo widens the left side to
#    11 columns and uses 3 tile columns at c0=0, 4, 8. The decoding order changes
#    at the same time: scan row-major, so each tile row contributes three glyphs
#    from left to right before moving down to the next tile row. Example with M=3:
#      tile rows:
#        row 0: A B C
#        row 1: D E F
#        row 2: G H I
#      3-column column-major order would be: A D G B E H C F I
#      this combo order is instead:          A B C D E F G H I
#    The output polyline is still drawn from the gray start after removing the
#    divider, and each 3x3 glyph still contributes one colored segment determined
#    only by its pixel area.
#
# b) Complexity calibration:
#    Changing only the traversal order can become visually weak if the layout stays
#    at 2 columns or if the sampled instruction list accidentally produces the same
#    trace under both orders. The 3-column layout amplifies the difference because
#    one tile row now bundles three successive instructions, and generation rejects
#    any sample whose row-major decoding matches the old 3-column column-major
#    decoding. This keeps the fusion observable without adding unrelated symbols.
#
# c) Old implicit assumptions audited:
#    - `find_full_height_constant_col(...)` still works because the divider remains
#      a single full-height non-background column.
#    - Gray-start shifting still works, but the valid shifted range is now 0..10
#      because the left canvas widened from 7 to 11 columns.
#    - Component extraction of 3x3 monochromatic glyphs still works because tiles
#      remain separated by one blank row and one blank column.
#    - The original sort key `(c0, r0)` encoded column-major traversal. Under this
#      combo that assumption is false; the solver must sort by `(r0, c0)`.
#    - The original generator assumption "split the instruction list into 2 equal
#      columns" is also false. The new generator must place instruction i at tile
#      column `i % 3` and tile row `i // 3`.
#    - Height can no longer ignore instruction count: there are 3*M glyphs now, so
#      generation must bound how many vertical area-4 glyphs appear to keep the walk
#      inside `H = 4*M - 1`.
#
# d) Placement legality:
#    Tile columns are fixed at 0, 4, 8, each with width 3, so the left glyph region
#    occupies exactly columns 0..10. A divider at column 11 separates it from an
#    11-column right region carrying the single gray start. The path is sampled in
#    output space first with explicit row/column bounds, and the maximum number of
#    vertical instructions is capped so no segment exits the bottom.
#
# e) Visual observability:
#    The changed behavior is visible from spatial arrangement alone: humans can see
#    three glyphs aligned in each tile row and infer that the examples consume them
#    left-to-right within each row. No hidden feature such as color id sorting or
#    component discovery order is used. The rejection test against 3-column
#    column-major decoding guarantees that the new row-wise traversal matters.
#
# f) Visual cleanliness:
#    The wider 11-column left canvas leaves one blank separator column between every
#    pair of 3x3 tiles, so glyphs remain isolated and easy to parse. The right side
#    still contains only one gray marker, and the output remains a single polyline
#    plus that marker, with no extra overlays or legends.
#
# g) Change inferability:
#    Color semantics stay strict and low-entropy: each fixed 3x3 shape keeps the same
#    color and the same segment type as in the original task. The only fused changes
#    are spatial layout and traversal order, both recoverable from a few examples
#    because the three glyphs in a row are visibly grouped and reused consistently.
#
# New solver assumptions checked against generation:
# - Exactly one full-height divider column exists at the boundary of the 11-column
#   left canvas.
# - Every intended glyph is a 3x3 monochromatic 8-connected component.
# - The gray start appears on the right and shifts into columns 0..10 after divider
#   removal.
# - Glyphs are stamped into every slot of a 3-column tile grid and decoded row-major.
# - Generated samples satisfy `main(inp) != _solve_with_order(inp, row_major=False)`,
#   so the fusion does not collapse back to the old traversal behavior.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        box = comp["bbox"]
        if bbox_h(box) == 3 and bbox_w(box) == 3:
            r0, c0, _r1, _c1 = map(int, box)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


def _solve_with_order(grid: np.ndarray, *, row_major: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)
    glyphs.sort(key=(lambda d: (int(d["r0"]), int(d["c0"]))) if row_major else (lambda d: (int(d["c0"]), int(d["r0"]))))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(area_to_dir.get(area, 1))
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 3-column array of 3x3 glyphs into a colored polyline on the left canvas.

    Input structure:
    - Left of a full-height divider: monochromatic 3x3 glyphs packed into three tile columns on a black background.
    - Right of the divider: exactly one gray (5) start pixel whose shifted position marks the walk origin after the divider is removed.

    Rule:
    - Remove the divider and move the gray start into the 11-column left canvas.
    - Read the glyphs in row-major tile order: for each tile row, process the left, middle, and right glyph before advancing to the next row.
    - Interpret each glyph by its pixel area: area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
    - For every decoded glyph, first step down one row and color that cell, then extend the rest of the segment in the decoded direction.

    Key insight:
    - The glyph shape still determines the segment type, but the instruction sequence now comes from scanning a visibly wider three-column tile grid row by row rather than finishing one glyph column at a time.
    """
    return _solve_with_order(grid, row_major=True)


# --------------- generator (3-column row-major combo variant) ---------------

M_MIN, M_MAX = 3, 5
NUM_TILE_COLS = 3
GRID_W = 23
DIVIDER_COL = 11
DIVIDER_COLOR = 4
LEFT_W = 11
TILE_COLS = (0, 4, 8)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 600


def _sample_path(tile_rows: int, start_col: int) -> list[int] | None:
    """Randomized depth-first construction of a valid 3*tile_rows instruction list."""
    H = 4 * tile_rows - 1
    total_steps = NUM_TILE_COLS * tile_rows
    max_vertical = tile_rows - 2

    def dfs(
        idx: int,
        r: int,
        c: int,
        areas: list[int],
        vertical_count: int,
        prev_vertical: bool | None,
        changes: int,
    ) -> list[int] | None:
        remaining = total_steps - idx
        if remaining == 0:
            return list(areas) if changes >= MIN_DIRECTION_CHANGES else None

        # Even the cheapest future steps (all horizontal) must fit vertically.
        if r + remaining > H - 1:
            return None

        # Not enough remaining steps left to realize the required vertical/horizontal flips.
        if changes + remaining < MIN_DIRECTION_CHANGES:
            return None

        options = list(_AREAS)
        random.shuffle(options)
        for area in options:
            is_vertical = area == 4
            if is_vertical and vertical_count + 1 > max_vertical:
                continue

            dr, dc = _AREA_DELTA[area]
            new_r = r + dr
            new_c = c + dc
            if new_r >= H or not (0 <= new_c < LEFT_W):
                continue

            new_changes = changes
            if prev_vertical is not None and is_vertical != prev_vertical:
                new_changes += 1

            areas.append(area)
            result = dfs(
                idx + 1,
                new_r,
                new_c,
                areas,
                vertical_count + int(is_vertical),
                is_vertical,
                new_changes,
            )
            areas.pop()
            if result is not None:
                return result
        return None

    return dfs(0, GRAY_START_ROW, start_col, [], 0, None, 0)


def _render_grid(tile_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint divider, gray start, and all row-major tiled glyphs into the input grid."""
    H = 4 * tile_rows - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i % NUM_TILE_COLS]
        r0 = TILE_ROW_STRIDE * (i // NUM_TILE_COLS)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _column_major_permutation(areas: list[int], tile_rows: int) -> list[int]:
    return [areas[NUM_TILE_COLS * r + c] for c in range(NUM_TILE_COLS) for r in range(tile_rows)]


def _variant_observable(inp: np.ndarray, areas: list[int], tile_rows: int) -> bool:
    permuted = _column_major_permutation(areas, tile_rows)
    if areas == permuted:
        return False
    return not np.array_equal(main(inp), _solve_with_order(inp, row_major=False))


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(M_MIN, M_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(tile_rows, start_col)
    if areas is None:
        return None

    inp = _render_grid(tile_rows, start_col, areas)
    if not _variant_observable(inp, areas, tile_rows):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
