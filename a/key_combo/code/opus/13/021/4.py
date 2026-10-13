from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
TILE_SIZE = 3
GLYPH_ROWS = 2
GLYPH_WIDTH = 2
START_ROW = 3
MIN_GLYPHS = 3
MAX_GLYPHS = 5
HEIGHT_MAX = 30
WIDTH_MAX = 30
GEN_ATTEMPTS = 400
GEN_PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)

VERTICAL = 0
LEFT = -1
RIGHT = 1

VERTICAL_GLYPH = np.array([[1, 0], [1, 0]], dtype=int)
LEFT_GLYPH = np.array([[1, 0], [1, 1]], dtype=int)
RIGHT_GLYPH = np.array([[0, 1], [1, 1]], dtype=int)

GLYPH_TO_STEP = {
    ((1, 0), (1, 0)): VERTICAL,
    ((0, 1), (0, 1)): VERTICAL,
    ((1, 0), (1, 1)): LEFT,
    ((0, 1), (1, 1)): RIGHT,
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_tile(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = (grid != BG) & (grid != RED)
    box = _bbox(mask)
    assert box is not None, "expected one colored 3x3 tile"
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 == TILE_SIZE and c1 - c0 + 1 == TILE_SIZE
    tile = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(tile != BG) and np.all(tile != RED)
    assert np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], np.ones((TILE_SIZE, TILE_SIZE), dtype=bool))
    return tile.copy(), box


def _decode_glyphs(grid: np.ndarray, tile_left: int) -> list[int]:
    glyphs: list[int] = []
    started = False
    max_col = max(0, tile_left - 2)

    for c0 in range(0, max_col + 1, GLYPH_WIDTH):
        block = (grid[:GLYPH_ROWS, c0 : c0 + GLYPH_WIDTH] == RED).astype(int)
        if block.shape != (GLYPH_ROWS, GLYPH_WIDTH):
            break
        if not block.any():
            if started:
                break
            continue
        started = True
        key = tuple(tuple(int(v) for v in row) for row in block)
        assert key in GLYPH_TO_STEP, f"unknown glyph pattern: {key}"
        glyphs.append(GLYPH_TO_STEP[key])

    assert MIN_GLYPHS <= len(glyphs) <= MAX_GLYPHS
    return glyphs


def _stamp(canvas: np.ndarray, tile: np.ndarray, top: int, left: int) -> None:
    bottom = top + tile.shape[0]
    right = left + tile.shape[1]
    assert 0 <= top and 0 <= left and bottom <= canvas.shape[0] and right <= canvas.shape[1]
    canvas[top:bottom, left:right] = tile


def _render_output(shape: tuple[int, int], tile: np.ndarray, glyphs: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    row = START_ROW
    col = shape[1] // 2 - 1
    _stamp(out, tile, row, col)
    for step in glyphs:
        row += TILE_SIZE
        col += TILE_SIZE * step
        _stamp(out, tile, row, col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the top red glyph strip into a sequence of 3x3 tile stamps on a downward path.

    Input structure:
    - The grid is mostly black, with a red two-row glyph strip anchored on the left and a single non-black/non-red 3x3 tile in the upper-right area.
    - The glyph strip is read in contiguous 2-column chunks; each chunk is either a vertical two-pixel segment or one of two L-shaped elbows.
    - The colored tile contains no black or red cells and is copied as the stamp pattern.

    Rule:
    - Read the 3x3 tile from the upper-right object.
    - Decode the red glyph chunks from left to right: the vertical glyph means move down by 3, the elbow missing its upper-right corner means move down by 3 and left by 3, and the elbow missing its upper-left corner means move down by 3 and right by 3.
    - Build a black output of the same size as the input, place the first tile with top-left corner at row 3 and column `(width // 2) - 1`, then stamp the same tile again after each decoded move, with later stamps overwriting earlier cells if needed.

    Key insight:
    - The red strip is not copied as an image; it is a left-to-right instruction stream that determines where the colored 3x3 tile gets stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    tile, tile_box = _extract_tile(g)
    glyphs = _decode_glyphs(g, tile_box[1])
    return _render_output(g.shape, tile, glyphs)


def _sample_shape(n_glyphs: int) -> tuple[int, int]:
    min_height = START_ROW + TILE_SIZE * n_glyphs + TILE_SIZE
    height = random.randint(min_height, HEIGHT_MAX)
    width = random.randint(18, WIDTH_MAX)
    return height, width


def _sequence_mode(n_glyphs: int) -> list[int]:
    if n_glyphs == 3 and random.random() < 0.6:
        seq = [VERTICAL, LEFT, RIGHT]
        random.shuffle(seq)
        return seq

    modes = ("balanced", "straight_heavy", "turn_heavy")
    mode = random.choice(modes)
    seq: list[int]
    if mode == "balanced":
        seq = [VERTICAL, LEFT, RIGHT]
    elif mode == "straight_heavy":
        seq = [VERTICAL, VERTICAL, random.choice((LEFT, RIGHT))]
    else:
        seq = [LEFT, RIGHT, random.choice((LEFT, RIGHT, VERTICAL))]

    while len(seq) < n_glyphs:
        seq.append(random.choice((VERTICAL, LEFT, RIGHT)))
    random.shuffle(seq)
    return seq


def _path_in_bounds(width: int, glyphs: list[int]) -> bool:
    col = width // 2 - 1
    if not (0 <= col <= width - TILE_SIZE):
        return False
    for step in glyphs:
        col += TILE_SIZE * step
        if not (0 <= col <= width - TILE_SIZE):
            return False
    return True


def _sample_glyphs(width: int, n_glyphs: int) -> list[int] | None:
    for _ in range(80):
        glyphs = _sequence_mode(n_glyphs)
        if glyphs.count(LEFT) == 0 and random.random() < 0.5:
            glyphs[random.randrange(n_glyphs)] = LEFT
        if glyphs.count(RIGHT) == 0 and random.random() < 0.5:
            glyphs[random.randrange(n_glyphs)] = RIGHT
        if glyphs.count(VERTICAL) == 0 and random.random() < 0.5:
            glyphs[random.randrange(n_glyphs)] = VERTICAL
        if len(set(glyphs)) == 1:
            continue
        if _path_in_bounds(width, glyphs):
            return glyphs
    return None


def _sample_tile() -> np.ndarray:
    n_colors = random.choice((2, 3))
    colors = random.sample(GEN_PALETTE, n_colors)
    while True:
        tile = np.array([random.choice(colors) for _ in range(TILE_SIZE * TILE_SIZE)], dtype=int).reshape(TILE_SIZE, TILE_SIZE)
        present = sorted(set(map(int, tile.ravel())))
        if len(present) < 2:
            continue
        if len(present) == 2:
            counts = [int(np.count_nonzero(tile == color)) for color in present]
            if min(counts) == 1:
                return tile
        if len(present) == 3:
            return tile


def _glyph_pattern(step: int) -> np.ndarray:
    if step == VERTICAL:
        return VERTICAL_GLYPH
    if step == LEFT:
        return LEFT_GLYPH
    assert step == RIGHT
    return RIGHT_GLYPH


def _build_input(shape: tuple[int, int], tile: np.ndarray, glyphs: list[int]) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width), dtype=int)

    strip_width = len(glyphs) * GLYPH_WIDTH
    tile_left = width - TILE_SIZE
    assert strip_width + 2 < tile_left

    for idx, step in enumerate(glyphs):
        c0 = idx * GLYPH_WIDTH
        pattern = _glyph_pattern(step)
        grid[:GLYPH_ROWS, c0 : c0 + GLYPH_WIDTH][pattern == 1] = RED

    grid[:TILE_SIZE, tile_left:width] = tile
    return grid


def _attempt_generate() -> np.ndarray | None:
    n_glyphs = random.randint(MIN_GLYPHS, MAX_GLYPHS)
    shape = _sample_shape(n_glyphs)
    glyphs = _sample_glyphs(shape[1], n_glyphs)
    if glyphs is None:
        return None

    tile = _sample_tile()
    grid = _build_input(shape, tile, glyphs)
    expected = _render_output(shape, tile, glyphs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
