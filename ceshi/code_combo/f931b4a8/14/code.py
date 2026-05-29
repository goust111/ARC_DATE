# Combo variant of f931b4a8 (index 14)
# combo_spec: D2/2+D8/0
# Name: 分隔线划分+边缘编码控制

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - The border now carries the two control counts: the left column (excluding the
#     corner) encodes output height and the top row (excluding the corner) encodes
#     output width.
#   - The remaining body contains one solid horizontal divider row and one solid
#     vertical divider column of a dedicated separator color. They split the body into
#     four rectangles with an off-center vertical cut.
#   - To keep every body region meaningful, the left pair of regions show the same
#     fill tile and the right pair show the same stencil tile. Example: with a 7x10
#     body, divider row at 3 and divider column at 4, the four regions have shapes
#     3x4, 3x5, 3x4, 3x5. Six non-black pixels on the outer left border mean
#     output height 6, and eight non-black pixels on the top border mean output
#     width 8. The solver must detect the divider cross to know that the 3x4 left
#     block is fill data and the 3x5 right block is stencil data.
#   - Original task: controls were 2D upper quadrants and the split was the midpoint.
#     Combo task: controls moved to 1D edges and the midpoint is no longer usable,
#     because the body widths differ and the separator column marks the true cut.
#
# b) complexity_calibration:
#   - The two fused signals are both visually strong: one border row/column of count
#     marks, plus one internal separator cross.
#   - The body keeps only two semantic materials, fill on the left and stencil on the
#     right, repeated vertically. This avoids introducing four unrelated regions while
#     still making the D8 partition necessary.
#   - The generator always makes the vertical cut off-center (`left_w != right_w`), so
#     a midpoint-based parser is systematically wrong on the body.
#
# c) implicit_assumption_audit:
#   - `assert h == w and h % 2 == 0` from the source solver no longer holds. The combo
#     layout is usually rectangular because the border control lengths and off-center
#     divider widths are chosen independently.
#   - `qh, qw = h // 2, w // 2` is invalid because the true boundaries are the detected
#     divider row/column inside `grid[1:, 1:]`.
#   - `upper_left` and `upper_right` are no longer control regions. Counting must read
#     `grid[1:, 0]` and `grid[0, 1:]` instead.
#   - The coarse fallback lookup used `fill_tile[r // qh, c // qw]` only because the
#     old stencil and fill widths matched. Here the fallback column index must divide
#     by the stencil width, not the fill width.
#   - New assumption introduced by the combo design: the upper and lower body regions
#     on each side are repeated copies. The generator enforces that so the body looks
#     like two materials split by a separator rather than four unrelated patches.
#
# d) placement_legality:
#   - The top row and left column are reserved for controls and never overlap the body.
#   - The divider row/column are inserted inside the body after the fill/stencil tiles
#     are built, so they occupy dedicated cells and cannot overwrite controls.
#   - The fill tile contains no black cells. The stencil may be empty or may contain
#     holes. Output dimensions are filtered so all coarse fill indices remain in bounds.
#
# e) visual_observability:
#   - A human can see three cues at once: edge count marks, an internal grey cross, and
#     left-vs-right material regions. Across a few examples, the only stable reading is
#     "count border cells for size, then read fill/stencil from the separated body".
#   - Because the separator column is off-center, the body cannot be parsed by simply
#     halving the interior at its midpoint. That makes the D8 contribution observable.
#
# f) visual_cleanliness:
#   - Grey is reserved for the separator only.
#   - Control colors are excluded from fill/stencil colors, so count marks stay distinct
#     from pattern content.
#   - Body regions use the same simple stripe/checker/frame patterns as the source task,
#     which keeps the examples readable.
#
# g) change inferability:
#   - No behavior depends on hidden color ordering or numeric color ids.
#   - Color changes are not remapped at all; colors merely identify controls, fill, or
#     stencil roles through position and separator structure.
#
# h) degeneration_check_plan:
#   - Quantify how often the separator column is off-center.
#   - Compare the combo solver output against a naive D2-only midpoint parser on the
#     same generated inputs. If they almost never match, the D8 signal is active.
#   - Compare border counts against body-region counts. If they differ almost always,
#     the D2 signal is also active and the task is not collapsing back to the source.
#
# i) new_solver_assumptions_and_generator_guarantees:
#   solver_assumptions:
#     - `grid[0, 0]` is black so the two edge controls remain separate.
#     - Inside `grid[1:, 1:]`, exactly one full non-black row and one full non-black
#       column share the same color; they are the separator cross.
#     - The upper and lower left body blocks are identical fill copies, and the upper
#       and lower right blocks are identical stencil copies.
#   generator_guarantees:
#     - Grey never appears outside the separator cross.
#     - The vertical cut is always off-center.
#     - Output dimensions do not exceed the edge-control capacities.

from math import ceil
import random

import numpy as np

from arc_utils import Color


BLACK = int(Color.BLACK)
DIVIDER = int(Color.GREY)
COLORS = tuple(c for c in range(1, 10) if c != DIVIDER)

STENCIL_EMPTY_PROB = 0.16
FILL_SOLID_PROB = 0.70
FILL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_SIDE = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_SEGMENT_PROB = 0.85
LEFT_MAX_COLORS = 3
TOP_MAX_COLORS = 2
TILE_HEIGHT_RANGE = (3, 8)
TILE_WIDTH_RANGE = (3, 8)
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _find_dividers(body: np.ndarray) -> tuple[int, int, int]:
    """Return (divider_row, divider_col, divider_color) inside the border-free body."""
    bh, bw = body.shape
    row_candidates = [
        (r, int(body[r, 0]))
        for r in range(bh)
        if int(body[r, 0]) != BLACK and np.all(body[r, :] == body[r, 0])
    ]
    col_candidates = [
        (c, int(body[0, c]))
        for c in range(bw)
        if int(body[0, c]) != BLACK and np.all(body[:, c] == body[0, c])
    ]
    colors = {
        color
        for _, color in row_candidates
        for _, col_color in col_candidates
        if color == col_color
    }
    assert len(colors) == 1
    divider_color = next(iter(colors))
    rows = [r for r, color in row_candidates if color == divider_color]
    cols = [c for c, color in col_candidates if color == divider_color]
    assert len(rows) == 1 and len(cols) == 1
    divider_r = rows[0]
    divider_c = cols[0]
    assert 0 < divider_r < bh - 1 and 0 < divider_c < bw - 1
    return divider_r, divider_c, divider_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode size from border counts, then use an internal separator cross to locate the fill and stencil tiles.

    Input structure:
    - The top row and left column are 1D control lines, with the top-left corner kept black.
    - Removing that border leaves a body containing one full separator row and one full separator column of a dedicated color.
    - The separator cross splits the body into four rectangles; the two left rectangles are matching fill copies and the two right rectangles are matching stencil copies.

    Rule:
    - Output height is the number of non-black cells in the left border column, excluding the corner.
    - Output width is the number of non-black cells in the top border row, excluding the corner.
    - Read the fill tile from the upper-left body region and the stencil tile from the upper-right body region. If the stencil is empty, tile the fill directly.
    - Otherwise tile the stencil, and whenever a stencil cell is black, replace it with the fill color selected by the coarse stencil-repeat position.

    Key insight:
    - The task now combines two different parsing steps: the size signal lives on the outer border, while the pattern source lives inside a separator-defined body whose true split is not the midpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 8 and w >= 8
    assert int(g[0, 0]) == BLACK

    out_h = int(np.count_nonzero(g[1:, 0] != BLACK))
    out_w = int(np.count_nonzero(g[0, 1:] != BLACK))
    assert out_h > 0 and out_w > 0

    body = g[1:, 1:]
    divider_r, divider_c, _divider_color = _find_dividers(body)

    fill_tile = body[:divider_r, :divider_c]
    stencil = body[:divider_r, divider_c + 1:]
    fill_copy = body[divider_r + 1:, :divider_c]
    stencil_copy = body[divider_r + 1:, divider_c + 1:]

    assert fill_tile.size > 0 and stencil.size > 0
    assert fill_tile.shape == fill_copy.shape
    assert stencil.shape == stencil_copy.shape
    assert np.array_equal(fill_tile, fill_copy)
    assert np.array_equal(stencil, stencil_copy)
    assert np.all(fill_tile != BLACK)

    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    assert fill_h == stencil_h

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % fill_h, c % fill_w])
                assert value != BLACK
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % stencil_h, c % stencil_w])
            if value == BLACK:
                fill_r = r // stencil_h
                fill_c = c // stencil_w
                assert fill_r < fill_h and fill_c < fill_w
                value = int(fill_tile[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = value

    return out


def _palette(k: int, pool: tuple[int, ...] = COLORS) -> list[int]:
    pool_list = list(pool)
    random.shuffle(pool_list)
    return pool_list[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_positions(length: int, mode: str) -> list[int]:
    positions = list(range(length))
    if mode == "compact":
        anchor = random.randrange(length)
        random.shuffle(positions)
        return sorted(positions, key=lambda idx: (abs(idx - anchor), random.random()))
    if mode == "ends":
        return positions[::2] + positions[1::2]
    random.shuffle(positions)
    return positions


def _make_fill_tile(
    shape: tuple[int, int],
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build an all-non-black fill tile using simple regular patterns."""
    height, width = shape
    k = len(palette)
    tile = np.empty((height, width), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(height):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(width):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:height, :width]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [color for color in palette if color not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(height) for c in range(width)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    """Build a stencil tile where black cells become coarse fill lookups."""
    height, width = shape
    tile = np.zeros((height, width), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if min(height, width) >= STENCIL_FRAME_MIN_SIDE:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        rr, cc = np.mgrid[:height, :width]
        mask = (rr + cc) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        thickness = random.randint(1, max(1, min(height, width) // 4))
        mask = np.ones((height, width), dtype=bool)
        mask[thickness:height - thickness, thickness:width - thickness] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(height):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((width,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(width):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((height,), palette)
    else:
        area = height * width
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _make_control_line(length: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    palette = _palette(k, tuple(pool))
    line = np.full(length, BLACK, dtype=int)

    if random.random() < CONTROL_SEGMENT_PROB:
        start = random.randint(0, length - count)
        line[start:start + count] = _random_color_array((count,), palette)
        return line

    mode = random.choice(["compact", "ends", "scatter"])
    positions = _ordered_positions(length, mode)[:count]
    for idx in positions:
        line[idx] = random.choice(palette)
    return line


def _assemble_body(fill_tile: np.ndarray, stencil: np.ndarray) -> np.ndarray:
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    assert fill_h == stencil_h

    body = np.zeros((2 * fill_h + 1, fill_w + 1 + stencil_w), dtype=int)
    body[fill_h, :] = DIVIDER
    body[:, fill_w] = DIVIDER
    body[:fill_h, :fill_w] = fill_tile
    body[fill_h + 1:, :fill_w] = fill_tile
    body[:stencil_h, fill_w + 1:] = stencil
    body[fill_h + 1:, fill_w + 1:] = stencil
    return body


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % fill_h, cc % fill_w]
    tiled = stencil[rr % stencil_h, cc % stencil_w].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // stencil_h, cc[bg_mask] // stencil_w]
    return tiled


def _midpoint_output(grid: np.ndarray) -> np.ndarray:
    """Naive baseline that guesses the body split from the geometric midpoint."""
    g = np.asarray(grid, dtype=int)
    out_h = int(np.count_nonzero(g[1:, 0] != BLACK))
    out_w = int(np.count_nonzero(g[0, 1:] != BLACK))
    body = g[1:, 1:]
    divider_r = body.shape[0] // 2
    divider_c = body.shape[1] // 2
    fill_tile = body[:divider_r, :divider_c]
    stencil = body[:divider_r, divider_c + 1:]
    return _expected_output(fill_tile, stencil, out_h, out_w)


def _sample_layout() -> tuple[int, int, int]:
    while True:
        fill_h = random.randint(*TILE_HEIGHT_RANGE)
        fill_w = random.randint(*TILE_WIDTH_RANGE)
        stencil_w = random.randint(*TILE_WIDTH_RANGE)
        if stencil_w == fill_w:
            continue
        body_w = fill_w + 1 + stencil_w
        if fill_w == body_w // 2:
            continue
        return fill_h, fill_w, stencil_w


def _gen_attempt() -> np.ndarray | None:
    fill_h, fill_w, stencil_w = _sample_layout()
    body_h = 2 * fill_h + 1
    body_w = fill_w + 1 + stencil_w
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    fill_k = 1 if random.random() < FILL_SOLID_PROB else random.randint(*FILL_MULTI_COLOR_RANGE)
    fill_palette = _palette(fill_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        stencil_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(stencil_k)

    output_colors = set(fill_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    max_h = min(OUTPUT_DIM_MAX, body_h)
    max_w = min(OUTPUT_DIM_MAX, body_w)
    valid_dims: list[tuple[int, int]] = []
    for out_h in range(OUTPUT_DIM_MIN, max_h + 1):
        for out_w in range(OUTPUT_DIM_MIN, max_w + 1):
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if not stencil_empty:
                used_rows = ceil(out_h / fill_h)
                used_cols = ceil(out_w / stencil_w)
                if used_rows > fill_h or used_cols > fill_w:
                    continue
                if used_rows * used_cols < fill_k:
                    continue
            valid_dims.append((out_h, out_w))
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = None
    if not stencil_empty:
        used_fill = [
            (r, c)
            for r in range(ceil(out_h / fill_h))
            for c in range(ceil(out_w / stencil_w))
        ]

    fill_tile = _make_fill_tile((fill_h, fill_w), fill_palette, used_fill)
    stencil = (
        np.zeros((fill_h, stencil_w), dtype=int)
        if stencil_empty
        else _make_stencil((fill_h, stencil_w), stencil_palette)
    )
    body = _assemble_body(fill_tile, stencil)

    top_ctrl = _make_control_line(body_w, out_w, TOP_MAX_COLORS, control_pool)
    left_ctrl = _make_control_line(body_h, out_h, LEFT_MAX_COLORS, control_pool)

    grid = np.zeros((body_h + 1, body_w + 1), dtype=int)
    grid[0, 1:] = top_ctrl
    grid[1:, 0] = left_ctrl
    grid[1:, 1:] = body

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    try:
        midpoint_output = _midpoint_output(grid)
    except Exception:
        midpoint_output = None
    if midpoint_output is not None and np.array_equal(midpoint_output, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
