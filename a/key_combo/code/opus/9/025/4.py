import random

import numpy as np


INPUT_SHAPE = (12, 15)
OUTPUT_SHAPE = (12, 12)
BLOCK_SIZE = 3
MACRO_SIDE = 4
BACKGROUND = 0
GREY = 5
ROW_TO_COLOR = {0: 1, 1: 2, 2: 3, 3: 4}

MAX_GENERATION_ATTEMPTS = 800
_SEEN_INPUTS: set[bytes] = set()
_SEEN_OUTPUTS: set[bytes] = set()
_COUNTER_USAGE = {1: 0, 2: 0, 3: 0}
_ROW_USAGE = {0: 0, 1: 0, 2: 0, 3: 0}


def _first_active(pattern: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(pattern)
    assert coords.size > 0, "expected at least one active macro block"
    r, c = coords[0]
    return int(r), int(c)


def _extract_counter_height(grid: np.ndarray) -> int:
    counter = grid[:, 12:15]
    assert counter.shape == (12, 3), "expected a 12x3 counter strip"
    assert np.all(counter == counter[:, [0]]), "all counter columns must match"
    col = counter[:, 0]
    assert set(map(int, np.unique(col))).issubset({BACKGROUND, GREY}), "counter must use only black and grey"
    n = int(np.count_nonzero(col == GREY))
    assert 1 <= n <= 3, "counter height must be 1..3"
    assert np.all(col[:n] == GREY), "counter must start with a grey prefix"
    assert np.all(col[n:] == BACKGROUND), "counter grey cells must be contiguous from the top"
    return n


def _extract_pattern(grid: np.ndarray) -> np.ndarray:
    left = grid[:, :12]
    pattern = np.zeros((MACRO_SIDE, MACRO_SIDE), dtype=bool)
    for br in range(MACRO_SIDE):
        for bc in range(MACRO_SIDE):
            block = left[
                br * BLOCK_SIZE : (br + 1) * BLOCK_SIZE,
                bc * BLOCK_SIZE : (bc + 1) * BLOCK_SIZE,
            ]
            center = int(block[1, 1])
            ring = block.copy()
            ring[1, 1] = BACKGROUND
            assert np.all(ring == BACKGROUND), "non-center block pixels must stay black"
            assert center in (BACKGROUND, GREY), "macro blocks may only contain grey center markers"
            pattern[br, bc] = center == GREY
    return pattern


def _render_output(pattern: np.ndarray, color: int) -> np.ndarray:
    out = np.full(OUTPUT_SHAPE, BACKGROUND, dtype=int)
    for br in range(MACRO_SIDE):
        for bc in range(MACRO_SIDE):
            if pattern[br, bc]:
                r0 = br * BLOCK_SIZE
                c0 = bc * BLOCK_SIZE
                out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(color)
    return out


def _render_input(pattern: np.ndarray, n: int) -> np.ndarray:
    grid = np.full(INPUT_SHAPE, BACKGROUND, dtype=int)
    for br in range(MACRO_SIDE):
        for bc in range(MACRO_SIDE):
            if pattern[br, bc]:
                grid[br * BLOCK_SIZE + 1, bc * BLOCK_SIZE + 1] = GREY
    grid[:n, 12:15] = GREY
    return grid


def _least_used(pool: dict[int, int]) -> int:
    lowest = min(pool.values())
    choices = [key for key, value in pool.items() if value == lowest]
    return int(random.choice(choices))


def _choose_counter_value() -> int:
    if 0 in _COUNTER_USAGE.values():
        return _least_used(_COUNTER_USAGE)
    return int(random.randint(1, 3))


def _choose_target_row() -> int:
    if 0 in _ROW_USAGE.values():
        return _least_used(_ROW_USAGE)
    return int(random.randint(0, 3))


def _semantic_pattern_ok(pattern: np.ndarray, n: int, target_row: int | None = None) -> bool:
    if pattern.shape != (MACRO_SIDE, MACRO_SIDE):
        return False
    active_count = int(np.count_nonzero(pattern))
    if not 3 <= active_count <= 6:
        return False
    if not any(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
        return False

    rotated = np.rot90(pattern, n)
    if np.array_equal(rotated, pattern):
        return False

    base_first = _first_active(pattern)
    rotated_first = _first_active(rotated)
    if target_row is not None and rotated_first[0] != target_row:
        return False
    if base_first[0] == rotated_first[0]:
        return False

    rotated_shapes = {np.rot90(pattern, k).tobytes() for k in (1, 2, 3)}
    output_signatures = set()
    output_rows = set()
    for k in (1, 2, 3):
        candidate = np.rot90(pattern, k)
        first_r, _ = _first_active(candidate)
        output_rows.add(first_r)
        output_signatures.add(_render_output(candidate, ROW_TO_COLOR[first_r]).tobytes())
    if len(rotated_shapes) < 2:
        return False
    if len(output_signatures) < 2:
        return False
    if len(output_rows) < 2:
        return False
    return True


def _validate_case(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.shape == INPUT_SHAPE
    assert set(map(int, np.unique(inp))).issubset({BACKGROUND, GREY}), "input should use only black and grey"

    n = _extract_counter_height(inp)
    pattern = _extract_pattern(inp)
    assert _semantic_pattern_ok(pattern, n), "generator produced a degenerate macro pattern"

    out = main(inp)
    rotated = np.rot90(pattern, n)
    first_r, _ = _first_active(rotated)
    expected = _render_output(rotated, ROW_TO_COLOR[first_r])
    assert np.array_equal(out, expected), "main must follow the stated rule"
    assert out.shape == OUTPUT_SHAPE
    assert set(map(int, np.unique(out))).issubset({BACKGROUND, 1, 2, 3, 4}), "unexpected output colors"
    used = sorted(int(color) for color in np.unique(out) if color != BACKGROUND)
    assert used == [ROW_TO_COLOR[first_r]], "output must use exactly one fill color"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the marked 4x4 macro-block pattern by the grey counter value and repaint the active blocks with a row-coded color.

    Input structure:
    - The input is a 12x15 black grid.
    - Its left 12x12 region is a 4x4 macro-grid of 3x3 blocks where an active block is marked only by a grey center pixel.
    - Its rightmost three columns are identical grey counters whose top 1-3 cells are grey and whose remaining cells are black.

    Rule:
    - Count the grey height N in one counter column.
    - Read the 4x4 macro activation mask from the grey block centers and rotate that boolean mask N quarter-turns counterclockwise.
    - Scan the rotated 4x4 mask in row-major order, use the row index of the first active block to choose the fill color
      (row 0 -> blue, row 1 -> red, row 2 -> green, row 3 -> yellow), then draw every active macro block as a solid 3x3 block in that color.
    - Remove the counter area and return only the 12x12 repainted macro-grid.

    Key insight:
    - The grey counter does not recolor pixels directly; it changes the macro-level orientation, and that rotated orientation determines which active block becomes the top-leftmost witness that selects the final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, "expected a 12x15 input grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, GREY}), "input should use only black and grey"

    n = _extract_counter_height(g)
    pattern = _extract_pattern(g)
    active_count = int(np.count_nonzero(pattern))
    assert 3 <= active_count <= 6, "expected 3-6 active macro blocks"
    assert any(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)), "pattern must not be fully rotation-symmetric"

    rotated = np.rot90(pattern, n)
    first_r, _ = _first_active(rotated)
    color = ROW_TO_COLOR[first_r]
    return _render_output(rotated, color)


def _sample_rotated_pattern(target_row: int) -> np.ndarray:
    cells = [(r, c) for r in range(target_row, MACRO_SIDE) for c in range(MACRO_SIDE)]
    max_cells = min(6, len(cells))
    count = random.randint(3, max_cells)

    pattern = np.zeros((MACRO_SIDE, MACRO_SIDE), dtype=bool)
    must_use = random.choice([(target_row, c) for c in range(MACRO_SIDE)])
    pattern[must_use] = True
    others = [cell for cell in cells if cell != must_use]
    for r, c in random.sample(others, count - 1):
        pattern[r, c] = True
    return pattern


def _generation_attempt() -> np.ndarray | None:
    n = _choose_counter_value()
    target_row = _choose_target_row()

    rotated = _sample_rotated_pattern(target_row)
    base_pattern = np.rot90(rotated, -n)
    if not _semantic_pattern_ok(base_pattern, n, target_row=target_row):
        return None

    inp = _render_input(base_pattern, n)
    out = main(inp)
    input_sig = inp.tobytes()
    output_sig = out.tobytes()
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None

    _validate_case(inp)
    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    _COUNTER_USAGE[n] += 1
    _ROW_USAGE[target_row] += 1
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generation_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a non-degenerate input for puzzle 4")
