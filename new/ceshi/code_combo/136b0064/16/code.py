# Combo variant of 136b0064 (index 16)
# combo_spec: D2/4+D3/2
# Name: 空心框架的负形方向编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The seed task used fixed 3x3 solid glyphs whose filled area directly selected one
#    of four segments. That is not enough for the combo, because D2/4 wants direction
#    to come from a visible hole position and D3/2 wants the glyph itself to be a hollow
#    frame. A true "frame with an internal hole pointing up/down/left/right" needs more
#    room than 3x3, so the tile geometry is enlarged to 8x8 slots.
#    Each slot now contains one monochrome square ring of outer side 3/4/5/6. The ring
#    is flush against one side of the 8x8 slot and centered on the orthogonal axis. Its
#    empty interior therefore sits visibly high / low / left / right inside the slot:
#      - 4x4 ring at rows 0..3, cols 2..5 -> hole sits high -> segment is up, length 3
#      - 4x4 ring at rows 2..5, cols 4..7 -> hole sits right -> segment is right, length 3
#    Compared with the seed task, two same-size rings can now point in different
#    directions, and two same-direction rings can have different lengths.
# b) Complexity calibration:
#    The grammar still has only 4 lengths and 4 directions. Length comes from outer side
#    3/4/5/6 -> segment length 2/3/4/5. Direction comes from one of four cavity offsets.
#    That is small enough for ARC, but visibly different from the original because the
#    solver must combine a positive-space cue (frame size) with a negative-space cue
#    (where the cavity sits inside the slot).
# c) Hidden-assumption audit against the original main():
#    - Original assumption: connected-component area alone determines the instruction.
#      Broken. Here the ring size determines only the length; direction must be decoded
#      from the cavity location inside the fixed tile slot.
#    - Original assumption: every non-area-4 glyph is horizontal. Broken. The combo
#      allows all four directions for every supported length.
#    - Original assumption: glyph bbox is always exactly 3x3 and can be found by generic
#      component scanning. Broken. Sizes now vary, so the solver reads fixed tile slots
#      implied by the layout instead of assuming same-size components.
#    - Original assumption: stepping down before the segment never revisits old paint.
#      Broken by upward segments. In this combo, an up-segment necessarily repaints the
#      previous endpoint; the generator allows only that one seam overlap and forbids all
#      other self-intersections so the output stays legible.
# d) Placement legality:
#    The generator is output-first. It samples a path in the 17-column left canvas, checks
#    every segment cell stays in bounds, and rejects any overlap except the one-cell seam
#    caused by moving up from the previous endpoint. Only after a valid instruction list is
#    found are the hollow-frame glyphs stamped into the left slots. Slots are disjoint by
#    construction, so glyphs never touch each other or the divider.
# e) Visual observability:
#    A human sees repeated colors / sizes with different cavity offsets, so size cannot be
#    the whole rule. The generator requires at least one repeated size used with multiple
#    directions, plus repeated directions across multiple sizes, so the fused behavior is
#    visible within a small number of examples.
# f) Visual cleanliness:
#    The left side keeps a strict grid of two glyph columns with one blank column and one
#    blank row between slots. The output path forbids self-crossing except the controlled
#    up-seam repaint, which keeps the trace readable instead of becoming a dense scribble.
# g) Color inference:
#    Color is still not a hidden key. It is a stable function of frame size only, so the
#    human can treat it as a secondary cue for length. Direction remains recoverable from
#    the visible negative space even when two glyphs share the same color.


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
N_MIN, N_MAX = 3, 4
GRID_W = 35
DIVIDER_COL = 17
DIVIDER_COLOR = 4
LEFT_W = 17
TILE_COLS = (0, 9)
SLOT_SIZE = 8
TILE_ROW_STRIDE = 9
GRAY = 5
GRAY_START_ROW = 0

# --- hollow-frame grammar ---
FRAME_SIZES = (3, 4, 5, 6)
SIZE_TO_LEN = {3: 2, 4: 3, 5: 4, 6: 5}
SIZE_TO_COLOR = {3: 1, 4: 2, 5: 3, 6: 6}
DIRS = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# --- generation constraints ---
MAX_ATTEMPTS = 700
MIN_DIR_CHANGES = 4


def _row_starts(h: int) -> list[int]:
    assert (int(h) + 1) % TILE_ROW_STRIDE == 0, f"unexpected height: {h}"
    n = (int(h) + 1) // TILE_ROW_STRIDE
    return [int(i * TILE_ROW_STRIDE) for i in range(int(n))]


def _frame_origin(direction: str, size: int) -> tuple[int, int]:
    offset = (SLOT_SIZE - int(size)) // 2
    if direction == "up":
        return 0, int(offset)
    if direction == "down":
        return int(SLOT_SIZE - int(size)), int(offset)
    if direction == "left":
        return int(offset), 0
    if direction == "right":
        return int(offset), int(SLOT_SIZE - int(size))
    raise AssertionError(f"bad direction: {direction}")


def _frame_mask(size: int) -> np.ndarray:
    mask = np.ones((int(size), int(size)), dtype=int)
    mask[1:-1, 1:-1] = 0
    return mask


FRAME_MASKS = {int(size): _frame_mask(int(size)) for size in FRAME_SIZES}


def _build_tile(direction: str, size: int) -> np.ndarray:
    tile = np.zeros((SLOT_SIZE, SLOT_SIZE), dtype=int)
    color = int(SIZE_TO_COLOR[int(size)])
    r0, c0 = _frame_origin(direction, int(size))
    tile[r0 : r0 + int(size), c0 : c0 + int(size)] = FRAME_MASKS[int(size)] * color
    return tile


TEMPLATES = {(direction, int(size)): _build_tile(direction, int(size)) for direction in DIRS for size in FRAME_SIZES}


def _decode_tile(tile: np.ndarray) -> tuple[int, tuple[int, int]]:
    filled = np.argwhere(tile != 0)
    assert filled.size > 0, "empty glyph slot"

    colors = np.unique(tile[tile != 0])
    assert len(colors) == 1, f"glyph must be monochrome, got {colors.tolist()}"

    r0, c0 = filled.min(axis=0)
    r1, c1 = filled.max(axis=0)
    size_h = int(r1 - r0 + 1)
    size_w = int(c1 - c0 + 1)
    assert size_h == size_w, f"frame must be square, got {size_h}x{size_w}"
    size = int(size_h)
    assert size in SIZE_TO_LEN, f"unexpected frame size: {size}"

    local = tile[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert (local[0, :] != 0).all() and (local[-1, :] != 0).all(), "frame missing horizontal border"
    assert (local[:, 0] != 0).all() and (local[:, -1] != 0).all(), "frame missing vertical border"
    assert (local[1:-1, 1:-1] == 0).all(), "frame interior must be hollow"

    hole = np.argwhere(local == 0)
    hole_r = float(r0) + float(hole[:, 0].mean())
    hole_c = float(c0) + float(hole[:, 1].mean())
    center = (SLOT_SIZE - 1) / 2.0
    dr = hole_r - center
    dc = hole_c - center
    assert abs(dr) != abs(dc), f"hole offset ambiguous: dr={dr}, dc={dc}"

    if abs(dr) > abs(dc):
        direction = "up" if dr < 0 else "down"
    else:
        direction = "left" if dc < 0 else "right"
    return int(SIZE_TO_LEN[size]), DIR_TO_DELTA[direction]


def _decode_instruction_slots(grid: np.ndarray, div_c: int) -> list[tuple[int, tuple[int, int], int]]:
    left = np.asarray(grid[:, : int(div_c)], dtype=int)
    instructions: list[tuple[int, tuple[int, int], int]] = []
    for tile_col in TILE_COLS:
        for r0 in _row_starts(left.shape[0]):
            tile = left[int(r0) : int(r0) + SLOT_SIZE, int(tile_col) : int(tile_col) + SLOT_SIZE]
            assert tile.shape == (SLOT_SIZE, SLOT_SIZE), f"bad tile shape at {(r0, tile_col)}: {tile.shape}"
            colors = np.unique(tile[tile != 0])
            assert len(colors) == 1, f"slot {(r0, tile_col)} has {len(colors)} colors"
            seg_len, delta = _decode_tile(tile)
            instructions.append((int(seg_len), delta, int(colors[0])))
    return instructions


def _segment_cells(r: int, c: int, direction: str, seg_len: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rr = int(r + 1)
    cc = int(c)
    cells = [(int(rr), int(cc))]
    dr, dc = DIR_TO_DELTA[direction]
    for _ in range(int(seg_len) - 1):
        rr += int(dr)
        cc += int(dc)
        cells.append((int(rr), int(cc)))
    return cells, (int(rr), int(cc))


def _all_in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= int(rr) < int(h) and 0 <= int(cc) < int(w) for rr, cc in cells)


def _allowed_overlap(r: int, c: int, direction: str, start_cell: tuple[int, int]) -> set[tuple[int, int]]:
    if direction == "up" and (int(r), int(c)) != start_cell:
        return {(int(r), int(c))}
    return set()


def _instructions_ok(instructions: list[tuple[str, int]]) -> bool:
    dirs = [direction for direction, _size in instructions]
    sizes = [size for _direction, size in instructions]
    dir_set = set(dirs)
    size_set = set(sizes)
    dir_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
    size_multi_dir = any(len({direction for direction, size in instructions if size == target}) >= 2 for target in size_set)
    dir_multi_size = any(len({size for direction, size in instructions if direction == target}) >= 2 for target in dir_set)
    has_vertical = any(direction in {"up", "down"} for direction in dirs)
    has_horizontal = any(direction in {"left", "right"} for direction in dirs)
    return (
        "up" in dir_set
        and "down" in dir_set
        and len(dir_set) >= 3
        and len(size_set) >= 3
        and has_vertical
        and has_horizontal
        and dir_changes >= MIN_DIR_CHANGES
        and size_multi_dir
        and dir_multi_size
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
        for size in FRAME_SIZES:
            seg_len = int(SIZE_TO_LEN[int(size)])
            cells, end = _segment_cells(int(r), int(c), direction, seg_len)
            if not _all_in_bounds(cells, int(h), LEFT_W):
                continue
            allow = _allowed_overlap(int(r), int(c), direction, start_cell)
            if start_cell in cells and start_cell not in allow:
                continue
            if any(cell in visited and cell not in allow for cell in cells):
                continue
            options.append((direction, int(size), cells, end))

    random.shuffle(options)
    for direction, size, cells, end in options:
        for cell in cells:
            visited.add(cell)
        tail = _search_path(int(h), int(end[0]), int(end[1]), int(steps_left) - 1, start_cell, visited)
        if tail is not None:
            return [(direction, int(size)), *tail]
        for cell in cells:
            if cell not in _allowed_overlap(int(r), int(c), direction, start_cell):
                visited.remove(cell)
    return None


def _sample_path(n: int, start_col: int) -> list[tuple[str, int]] | None:
    h = TILE_ROW_STRIDE * int(n) - 1
    start_cell = (GRAY_START_ROW, int(start_col))
    for _ in range(180):
        visited = {start_cell}
        instructions = _search_path(h, GRAY_START_ROW, int(start_col), 2 * int(n), start_cell, visited)
        if instructions is not None and _instructions_ok(instructions):
            return instructions
    return None


def _simulate_output(h: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    out = np.zeros((int(h), LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    out[r_cur, c_cur] = GRAY
    start_cell = (GRAY_START_ROW, int(start_col))
    for direction, size in instructions:
        color = int(SIZE_TO_COLOR[int(size)])
        seg_len = int(SIZE_TO_LEN[int(size)])
        cells, end = _segment_cells(int(r_cur), int(c_cur), direction, seg_len)
        allow = _allowed_overlap(int(r_cur), int(c_cur), direction, start_cell)
        for rr, cc in cells:
            assert 0 <= int(rr) < int(h) and 0 <= int(cc) < LEFT_W, f"segment exits canvas: {(rr, cc)}"
            if (int(rr), int(cc)) == start_cell and (int(rr), int(cc)) not in allow:
                raise AssertionError("segment touches gray start")
            out[int(rr), int(cc)] = color
        r_cur, c_cur = int(end[0]), int(end[1])
    out[GRAY_START_ROW, int(start_col)] = GRAY
    return out


def _render_grid(n: int, start_col: int, instructions: list[tuple[str, int]]) -> np.ndarray:
    h = TILE_ROW_STRIDE * int(n) - 1
    grid = np.zeros((int(h), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, (direction, size) in enumerate(instructions):
        tile_col = int(TILE_COLS[i // int(n)])
        r0 = int(TILE_ROW_STRIDE * (i % int(n)))
        grid[r0 : r0 + SLOT_SIZE, tile_col : tile_col + SLOT_SIZE] = TEMPLATES[(direction, int(size))]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]] | None:
    n = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    instructions = _sample_path(int(n), int(start_col))
    if instructions is None:
        return None
    inp = _render_grid(int(n), int(start_col), instructions)
    expected = _simulate_output(TILE_ROW_STRIDE * int(n) - 1, int(start_col), instructions)
    return inp, expected, instructions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a colored polyline by decoding hollow-frame glyphs arranged beside a divider.

    Input structure:
    - Left of a full-height divider: two columns of fixed 8x8 glyph slots, each slot holding
      one monochrome hollow square frame on a black background.
    - Right of the divider: one gray (5) start pixel whose shifted column chooses the path origin.

    Rule:
    - Remove the divider and shift the gray start into the 17-column left canvas.
    - Read the glyph slots in column-major order: left column top→bottom, then right column.
    - For each slot, step down one row and draw a segment in the glyph's color. The outer frame
      side length 3/4/5/6 gives segment length 2/3/4/5, and the frame's empty interior sitting
      high / low / left / right inside the slot gives the segment direction.

    Key insight:
    - The instruction is genuinely fused: positive space says how long the segment is, while the
      negative space says where it goes. Same-size frames can point different ways because the
      cavity position, not the ink amount alone, determines direction.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, _div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None, "divider not found"

    out = np.zeros((int(h), int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    start_col = int(c_cur)
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    instructions = _decode_instruction_slots(g, int(div_c))
    for seg_len, (dr, dc), color in instructions:
        r_cur = int(r_cur + 1)
        assert 0 <= int(r_cur) < int(h), f"segment start exits vertically: r_cur={r_cur}, h={h}"
        out[int(r_cur), int(c_cur)] = int(color)

        for _ in range(int(seg_len) - 1):
            r_cur = int(r_cur + int(dr))
            c_cur = int(c_cur + int(dc))
            assert 0 <= int(r_cur) < int(h) and 0 <= int(c_cur) < int(div_c), (
                f"segment exits canvas: r_cur={r_cur}, c_cur={c_cur}, h={h}, div_c={div_c}"
            )
            out[int(r_cur), int(c_cur)] = int(color)

    out[GRAY_START_ROW, int(start_col)] = GRAY
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            generated = _gen_attempt()
            if generated is None:
                continue
            inp, expected, _instructions = generated
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
