# Combo variant of 136b0064 (index 14)
# combo_spec: D2/4+D13/0
# Name: 负形编码与灰色关节

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The scene keeps the original macro-layout: a 15-column input, a full-height divider
#    at col 7, two left-side columns of 3x3 glyph slots, and one gray start pixel on the
#    right. The fused change is entirely in how each glyph is decoded and how each segment
#    begins in the output. A glyph now carries two independent cues:
#      - the unique empty side-center cell in its 3x3 bbox selects direction
#        (top/bottom/left/right hole => up/down/left/right motion),
#      - the filled-pixel count selects length (areas 5/6/7/8 => lengths 2/3/4/5).
#    After reading one glyph, the tracer still first moves down by one row, but that
#    connector cell is now always gray 5 instead of the glyph color. The remaining
#    segment body cells use the glyph color. Example: a color-2 area-7 glyph with a
#    right-side hole produces a gray joint one row below the current endpoint, then
#    three red cells extending right from that joint.
# b) Complexity calibration:
#    The codebook is 4 directions x 4 lengths = 16 visible glyph types, which is still
#    small enough for ARC. The generator forces repeated areas across multiple directions
#    and repeated directions across multiple areas so that solvers cannot fall back to
#    "color alone" or "area alone" shortcuts and the negative-space rule is visible from
#    only a few examples.
# c) Hidden assumptions broken from the seed solver:
#    - Area no longer determines direction; solver must inspect the missing side pixel.
#    - The original vertical-vs-horizontal split is invalid because every area can now
#      point in any of the four directions.
#    - The old colored connector assumption is wrong; the first cell of every segment
#      must be painted gray, with only the body painted in the glyph color.
#    - Because upward segments are now legal, the generator cannot assume monotone
#      downward progress and must explicitly validate bounds and overlap behavior.
# d) Placement legality:
#    Glyph placement still uses the original fixed 3x3 tile slots, so left-side objects
#    never overlap. The generator is output-first: it searches for a legal instruction
#    sequence inside the 7-column canvas, rejects moves that leave bounds or revisit old
#    cells, and only allows the special one-cell overlap inherent in an upward segment
#    stepping back onto the previous endpoint. This keeps solver assertions safe.
# e) Visual observability:
#    The change is meant to be legible by eye. Same-area glyphs can point in different
#    directions because their holes move, and same-direction glyphs can have different
#    body lengths because their fill counts change. Gray joints create a separate visual
#    layer, making "connector" versus "segment body" easy to distinguish.
# f) Visual cleanliness:
#    The tiled glyph layout stays sparse because the original blank row and blank column
#    separators are preserved. On the output side the generator avoids self-crossings
#    except for the unavoidable upward back-step onto the previous endpoint, so later
#    segments do not generally overwrite large earlier regions.
# g) Color inference and new solver assumptions:
#    Colors are still not hidden keys: each area keeps one stable non-gray color bucket,
#    while direction is recoverable from hole position alone. The new solver assumes
#    every glyph is a monochrome 3x3 component with exactly one missing side-center cell
#    and area in {5,6,7,8}; the generator constructs exactly that family.


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
N_MIN, N_MAX = 3, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- glyph grammar ---
AREAS = (5, 6, 7, 8)
AREA_TO_LEN = {5: 2, 6: 3, 7: 4, 8: 5}
AREA_TO_COLOR = {5: 1, 6: 2, 7: 3, 8: 6}
DIRS = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_POS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
CORNER_PRIORITY = {
    "up": [(0, 0), (0, 2), (2, 0), (2, 2)],
    "down": [(2, 0), (2, 2), (0, 0), (0, 2)],
    "left": [(0, 0), (2, 0), (0, 2), (2, 2)],
    "right": [(0, 2), (2, 2), (0, 0), (2, 0)],
}
MAX_ATTEMPTS = 700
SEARCH_ATTEMPTS = 160
MIN_DIR_CHANGES = 3


def _build_template(direction: str, area: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=int)
    mask[1, 1] = 1
    for name, pos in SIDE_POS.items():
        if name != direction:
            mask[pos] = 1
    for rr, cc in CORNER_PRIORITY[direction][: int(area - 4)]:
        mask[rr, cc] = 1
    return mask


TEMPLATES = {(direction, area): _build_template(direction, area) for direction in DIRS for area in AREAS}


def _decode_glyph(tile: np.ndarray) -> tuple[int, str, tuple[int, int]]:
    filled = tile != 0
    area = int(filled.sum())
    assert area in AREA_TO_LEN, f"unexpected glyph area: {area}"

    missing_sides = [name for name, (rr, cc) in SIDE_POS.items() if not bool(filled[rr, cc])]
    assert len(missing_sides) == 1, f"glyph must have exactly one side hole, got {missing_sides}"

    direction = missing_sides[0]
    return int(AREA_TO_LEN[area]), direction, DIR_TO_DELTA[direction]


def _instruction_cells(r: int, c: int, direction: str, seg_len: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rr = int(r + 1)
    cc = int(c)
    cells = [(rr, cc)]
    dr, dc = DIR_TO_DELTA[direction]
    for _ in range(int(seg_len) - 1):
        rr += int(dr)
        cc += int(dc)
        cells.append((int(rr), int(cc)))
    return cells, (int(rr), int(cc))


def _cells_in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= int(rr) < int(h) and 0 <= int(cc) < int(w) for rr, cc in cells)


def _instructions_ok(instructions: list[tuple[str, int]]) -> bool:
    dirs = [direction for direction, _area in instructions]
    areas = [area for _direction, area in instructions]
    dir_set = set(dirs)
    dir_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
    repeated_area_multi_dir = any(
        len({direction for direction, area in instructions if area == target_area}) >= 2 for target_area in set(areas)
    )
    repeated_dir_multi_area = any(
        len({area for direction, area in instructions if direction == target_direction}) >= 2
        for target_direction in set(dirs)
    )
    return (
        len(dir_set) == 4
        and len(set(areas)) == 4
        and dir_changes >= MIN_DIR_CHANGES
        and repeated_area_multi_dir
        and repeated_dir_multi_area
    )


def _search_path(
    h: int,
    r: int,
    c: int,
    steps_left: int,
    start_cell: tuple[int, int],
    visited: set[tuple[int, int]],
) -> list[tuple[str, int]] | None:
    if steps_left == 0:
        return []

    options: list[tuple[str, int, list[tuple[int, int]], tuple[int, int]]] = []
    for direction in DIRS:
        for area in AREAS:
            seg_len = AREA_TO_LEN[area]
            cells, end = _instruction_cells(r, c, direction, seg_len)
            if not _cells_in_bounds(cells, h, LEFT_W):
                continue
            if start_cell in cells:
                continue
            allowed_overlap = {(int(r), int(c))} if direction == "up" else set()
            if any(cell in visited and cell not in allowed_overlap for cell in cells):
                continue
            options.append((direction, area, cells, end))

    random.shuffle(options)
    for direction, area, cells, end in options:
        new_cells = [cell for cell in cells if cell not in visited]
        for cell in new_cells:
            visited.add(cell)
        tail = _search_path(h, end[0], end[1], steps_left - 1, start_cell, visited)
        if tail is not None:
            return [(direction, area), *tail]
        for cell in new_cells:
            visited.remove(cell)
    return None


def _sample_path(n: int, start_col: int) -> list[tuple[str, int]] | None:
    h = 4 * n - 1
    start_cell = (GRAY_START_ROW, start_col)
    for _ in range(SEARCH_ATTEMPTS):
        visited = {start_cell}
        instructions = _search_path(h, GRAY_START_ROW, start_col, 2 * n, start_cell, visited)
        if instructions is not None and _instructions_ok(instructions):
            return instructions
    return None


def _simulate_output(h: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    out = np.zeros((h, LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, start_col
    out[r_cur, c_cur] = GRAY
    for direction, area in instructions:
        color = AREA_TO_COLOR[area]
        seg_len = AREA_TO_LEN[area]
        cells, end = _instruction_cells(r_cur, c_cur, direction, seg_len)
        joint_r, joint_c = cells[0]
        out[joint_r, joint_c] = GRAY
        for rr, cc in cells[1:]:
            out[rr, cc] = color
        r_cur, c_cur = end
    return out


def _render_grid(n: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (direction, area) in enumerate(instructions):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = TEMPLATES[(direction, area)] * AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    instructions = _sample_path(n, start_col)
    if instructions is None:
        return None
    inp = _render_grid(n, start_col, instructions)
    expected = _simulate_output(4 * n - 1, start_col, instructions)
    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a segmented path by decoding hole-marked 3x3 glyphs across a divided board.

    Input structure:
    - Left of a full-height non-background divider: monochrome 3x3 glyphs placed in two tile columns.
    - Right of the divider: one gray (5) start cell whose shifted column becomes the left-canvas start.

    Rule:
    - Remove the divider, shift the gray start into the 7-column output canvas, and read glyphs in
      column-major order (left column top→bottom, then right column top→bottom).
    - For each glyph, move one row down and paint that connector cell gray; then continue the segment
      in the direction indicated by the glyph's unique side-hole position, using the glyph color for
      the remaining cells. The glyph's filled-pixel count determines the total segment length.

    Key insight:
    - Each instruction is fused from positive and negative space: areas 5/6/7/8 encode lengths 2/3/4/5,
      while the lone missing side-center pixel chooses up/down/left/right. Gray cells are not segment
      colors here; they are dedicated joints separating successive colored segment bodies.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[r_cur, c_cur] = GRAY

    left = g[:, : int(div_c)]
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": int(r0), "c0": int(c0)})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        r0 = int(glyph["r0"])
        c0 = int(glyph["c0"])
        tile = left[r0 : r0 + 3, c0 : c0 + 3]
        seg_len, _direction, (dr, dc) = _decode_glyph(tile)

        r_cur = int(r_cur + 1)
        assert 0 <= r_cur < h, f"segment joint exits vertically: r_cur={r_cur}, H={h}"
        out[r_cur, c_cur] = GRAY

        for _ in range(seg_len - 1):
            r_cur = int(r_cur + dr)
            c_cur = int(c_cur + dc)
            assert 0 <= r_cur < h and 0 <= c_cur < int(div_c), (
                f"segment exits canvas: r_cur={r_cur}, c_cur={c_cur}, H={h}, div_c={div_c}"
            )
            out[r_cur, c_cur] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            generated = _gen_attempt()
            if generated is None:
                continue
            inp, expected = generated
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
