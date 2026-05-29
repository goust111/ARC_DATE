# Variant of 136b0064
# Dimension: D2 — 字形编码方式——面积到指令的映射 (transform)
# Change: 字形像素面积唯一决定线段：area 4→垂直向下2格，area 5→水平向右3格，area 6→水平向左4格，area ... → 字形外接矩形内的空洞位置编码方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    The input layout stays the same: a 15-column grid, divider at col 7, one gray
#    start pixel on the right, and 2*N colored 3x3 glyphs tiled on the left in two
#    columns. The change is local to glyph decoding. Each glyph now has exactly one
#    empty side-center cell in its 3x3 bbox:
#      - empty at (0,1) => segment goes up
#      - empty at (2,1) => segment goes down
#      - empty at (1,0) => segment goes left
#      - empty at (1,2) => segment goes right
#    The filled-pixel count is now the length code. Areas 5/6/7/8 map to lengths
#    2/3/4/5. Example delta trace:
#      original: area 6 glyph always meant "left 4", regardless of where its holes sat
#      variant : area 6 glyph with right-side hole means "right 3"; area 6 glyph with
#                top-side hole means "up 3". Same area, different direction.
# b) Complexity calibration:
#    Direction has only 4 categories and length has only 4 categories, so the grammar
#    stays small enough for ARC. At the same time, same-area glyphs can point in
#    different directions, which makes the D2 change visible and blocks the old
#    area-only shortcut.
# c) Hidden assumptions in the original main():
#    - `glyph["area"]` used to determine both length and direction. That assumption is
#      invalid here; solver must inspect the 3x3 negative space.
#    - The `if area == 4 else horizontal` split depended on "vertical only for area 4".
#      That is no longer true because every area can encode any of the four directions.
#    - The old fallback `area_to_dir.get(area, 1)` would silently decode new area-8
#      glyphs as rightward. This variant removes that fallback and asserts the new code.
#    - The original generator could never revisit the gray start because paths only went
#      down or sideways. Upward segments can now do that, so the new generator forbids
#      touching the gray start after the initial placement.
# d) Placement legality:
#    Every glyph remains a monochromatic 8-connected 3x3 component placed in the same
#    tile slots as the seed task. The generator samples a path first, rejects any move
#    that exits the 7-column canvas or revisits a painted cell, then stamps the matching
#    glyph templates. This guarantees no object overlap and no solver asserts.
# e) Visual observability:
#    A human can compare same-color / same-area glyphs whose only salient difference is
#    the side notch and see that output direction follows the notch, not the ink amount.
#    The generator explicitly requires repeated areas across multiple directions in a
#    sample so the new rule is visible in 2-4 examples.
# f) Visual cleanliness:
#    Left-side glyphs stay separated by the original blank row/blank column structure.
#    Output polylines avoid self-overlap, so later colors do not obscure earlier ones.
# g) Color inference:
#    Color is not used as a hidden key. It is still just copied from glyph to segment,
#    with one stable color per area/length bucket. Direction is recoverable purely from
#    the visible hole position.


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
MAX_ATTEMPTS = 600
MIN_DIR_CHANGES = 3


def _build_template(direction: str, area: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=int)
    hole = SIDE_POS[direction]
    mask[1, 1] = 1
    for name, pos in SIDE_POS.items():
        if name != direction:
            mask[pos] = 1
    for rr, cc in CORNER_PRIORITY[direction][: int(area - 4)]:
        mask[rr, cc] = 1
    return mask


TEMPLATES = {(direction, area): _build_template(direction, area) for direction in DIRS for area in AREAS}


def _decode_glyph(tile: np.ndarray) -> tuple[int, tuple[int, int]]:
    filled = (tile != 0)
    area = int(filled.sum())
    assert area in AREA_TO_LEN, f"unexpected glyph area: {area}"

    missing_sides = [name for name, (rr, cc) in SIDE_POS.items() if not bool(filled[rr, cc])]
    assert len(missing_sides) == 1, f"glyph must have exactly one side hole, got {missing_sides}"

    direction = missing_sides[0]
    return int(AREA_TO_LEN[area]), DIR_TO_DELTA[direction]


def _segment_cells(r: int, c: int, direction: str, seg_len: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rr = int(r + 1)
    cc = int(c)
    cells = [(rr, cc)]
    dr, dc = DIR_TO_DELTA[direction]
    for _ in range(int(seg_len) - 1):
        rr += int(dr)
        cc += int(dc)
        cells.append((int(rr), int(cc)))
    return cells, (int(rr), int(cc))


def _all_in_bounds(cells: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(rr) < int(H) and 0 <= int(cc) < int(W) for rr, cc in cells)


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
        "up" in dir_set
        and "down" in dir_set
        and len(dir_set) >= 3
        and len(set(areas)) >= 3
        and dir_changes >= MIN_DIR_CHANGES
        and repeated_area_multi_dir
        and repeated_dir_multi_area
    )


def _search_path(
    H: int,
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
            cells, end = _segment_cells(r, c, direction, seg_len)
            if not _all_in_bounds(cells, H, LEFT_W):
                continue
            if start_cell in cells:
                continue
            if any(cell in visited for cell in cells):
                continue
            options.append((direction, area, cells, end))

    random.shuffle(options)
    for direction, area, cells, end in options:
        for cell in cells:
            visited.add(cell)
        tail = _search_path(H, end[0], end[1], steps_left - 1, start_cell, visited)
        if tail is not None:
            return [(direction, area), *tail]
        for cell in cells:
            visited.remove(cell)
    return None


def _sample_path(N: int, start_col: int) -> list[tuple[str, int]] | None:
    H = 4 * N - 1
    start_cell = (GRAY_START_ROW, start_col)
    for _ in range(120):
        visited = {start_cell}
        instructions = _search_path(H, GRAY_START_ROW, start_col, 2 * N, start_cell, visited)
        if instructions is not None and _instructions_ok(instructions):
            return instructions
    return None


def _simulate_output(H: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    out = np.zeros((H, LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, start_col
    out[r_cur, c_cur] = GRAY
    for direction, area in instructions:
        color = AREA_TO_COLOR[area]
        seg_len = AREA_TO_LEN[area]
        cells, end = _segment_cells(r_cur, c_cur, direction, seg_len)
        for rr, cc in cells:
            out[rr, cc] = color
        r_cur, c_cur = end
    return out


def _render_grid(N: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (direction, area) in enumerate(instructions):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = TEMPLATES[(direction, area)] * AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    instructions = _sample_path(N, start_col)
    if instructions is None:
        return None
    inp = _render_grid(N, start_col, instructions)
    expected = _simulate_output(4 * N - 1, start_col, instructions)
    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a colored polyline by reading 3x3 glyphs separated from a start marker by a divider.

    Input structure:
    - Left of a full-height vertical divider: 3x3 monochrome glyphs arranged in two tile columns.
    - Right of the divider: one gray (5) start pixel whose shifted column seeds the path.

    Rule:
    - Remove the divider and shift the gray start into the left 7-column canvas.
    - Read glyphs in column-major order (left tile column top→bottom, then right tile column).
    - For each glyph, step down one row, then draw a segment in the direction indicated by the
      glyph's unique side-hole position; the glyph's filled-pixel count sets the segment length.

    Key insight:
    - The instruction is split across positive and negative space: area 5/6/7/8 means length
      2/3/4/5, while the empty side-center cell in the 3x3 bbox selects up/down/left/right.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
        seg_len, (dr, dc) = _decode_glyph(tile)

        r_cur = int(r_cur + 1)
        assert 0 <= r_cur < H, f"segment start exits vertically: r_cur={r_cur}, H={H}"
        out[r_cur, c_cur] = color

        for _ in range(seg_len - 1):
            r_cur = int(r_cur + dr)
            c_cur = int(c_cur + dc)
            assert 0 <= r_cur < H and 0 <= c_cur < int(div_c), (
                f"segment exits canvas: r_cur={r_cur}, c_cur={c_cur}, H={H}, div_c={div_c}"
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
