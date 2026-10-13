import random

import numpy as np


BLOCK = 3

BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

BOARD_COLORS = {GREEN, YELLOW}
MARKER_COLORS = {RED, BLUE}

BLOCK_ROWS_RANGE = (3, 10)
BLOCK_COLS_RANGE = (3, 10)
MAX_MARKERS = 10
GEN_RETRIES = 300

_TEMPLATE_MASKS = (
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
)


def _block_origin(block_row: int, block_col: int) -> tuple[int, int]:
    return block_row * BLOCK, block_col * BLOCK


def _read_block(grid: np.ndarray, block_row: int, block_col: int) -> np.ndarray:
    r0, c0 = _block_origin(block_row, block_col)
    return grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK]


def _write_block(grid: np.ndarray, block_row: int, block_col: int, pattern: np.ndarray) -> None:
    r0, c0 = _block_origin(block_row, block_col)
    grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = pattern


def _corner_blocks(block_rows: int, block_cols: int) -> set[tuple[int, int]]:
    return {
        (0, 0),
        (0, block_cols - 1),
        (block_rows - 1, 0),
        (block_rows - 1, block_cols - 1),
    }


def _outer_ring(block: np.ndarray) -> np.ndarray:
    flat = block.reshape(-1)
    return flat[[0, 1, 2, 3, 5, 6, 7, 8]]


def _sample_template(rng: random.Random) -> np.ndarray:
    mask = _TEMPLATE_MASKS[rng.randrange(len(_TEMPLATE_MASKS))]
    on_color = GREEN if rng.random() < 0.5 else YELLOW
    off_color = YELLOW if on_color == GREEN else GREEN
    return np.where(mask == 1, on_color, off_color).astype(int)


def _build_checkerboard(block_rows: int, block_cols: int, phase: int) -> np.ndarray:
    grid = np.empty((block_rows * BLOCK, block_cols * BLOCK), dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            color = GREEN if (br + bc + phase) % 2 == 0 else YELLOW
            r0, c0 = _block_origin(br, bc)
            grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = color
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace center-marked checkerboard cells with the corner legend patterns.

    Input structure:
    - The grid is partitioned into 3x3 blocks that alternate between solid green and solid yellow like a checkerboard.
    - Some non-corner 3x3 blocks replace only their center pixel with a red or blue marker.
    - The top-left 3x3 block is a green/yellow template for red markers, and the bottom-right 3x3 block is a green/yellow template for blue markers.

    Rule:
    - Read the full 3x3 pattern from the top-left block as the replacement for red markers and from the bottom-right block as the replacement for blue markers.
    - Scan every non-corner 3x3 block: if its center is red, replace the whole block with the red template; if its center is blue, replace the whole block with the blue template.
    - Leave unmarked checkerboard blocks and the two corner legend blocks unchanged.

    Key insight:
    - The unusual corner blocks are legends: their 3x3 green/yellow layouts define how each scattered center-marker color expands into a full replacement tile.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % BLOCK == 0 and w % BLOCK == 0

    block_rows, block_cols = h // BLOCK, w // BLOCK
    assert block_rows >= 2 and block_cols >= 2

    red_pattern = _read_block(g, 0, 0).copy()
    blue_pattern = _read_block(g, block_rows - 1, block_cols - 1).copy()
    assert set(map(int, np.unique(red_pattern))) <= BOARD_COLORS
    assert set(map(int, np.unique(blue_pattern))) <= BOARD_COLORS
    assert len(np.unique(red_pattern)) == 2
    assert len(np.unique(blue_pattern)) == 2

    out = g.copy()
    corners = _corner_blocks(block_rows, block_cols)

    for br in range(block_rows):
        for bc in range(block_cols):
            block = _read_block(g, br, bc)
            colors = set(map(int, np.unique(block)))
            if (br, bc) == (0, 0) or (br, bc) == (block_rows - 1, block_cols - 1):
                continue

            center = int(block[1, 1])
            outer_colors = set(map(int, np.unique(_outer_ring(block))))
            if center in MARKER_COLORS:
                assert outer_colors <= BOARD_COLORS
                assert len(outer_colors) == 1
                if (br, bc) not in corners:
                    _write_block(out, br, bc, red_pattern if center == RED else blue_pattern)
            else:
                assert colors <= BOARD_COLORS
                if (br, bc) in corners:
                    assert len(colors) == 1
                else:
                    assert len(colors) == 1

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    block_rows = rng.randint(*BLOCK_ROWS_RANGE)
    block_cols = rng.randint(*BLOCK_COLS_RANGE)
    grid = _build_checkerboard(block_rows, block_cols, phase=rng.randint(0, 1))

    red_pattern = _sample_template(rng)
    blue_pattern = _sample_template(rng)
    if np.array_equal(red_pattern, blue_pattern):
        return None

    _write_block(grid, 0, 0, red_pattern)
    _write_block(grid, block_rows - 1, block_cols - 1, blue_pattern)

    corners = _corner_blocks(block_rows, block_cols)
    eligible = [(br, bc) for br in range(block_rows) for bc in range(block_cols) if (br, bc) not in corners]
    max_markers = min(MAX_MARKERS, len(eligible) - 1)
    if max_markers < 2:
        return None

    total_markers = rng.randint(2, max_markers)
    marker_blocks = rng.sample(eligible, total_markers)
    red_count = rng.randint(1, total_markers - 1)
    marker_colors = [RED] * red_count + [BLUE] * (total_markers - red_count)
    rng.shuffle(marker_colors)

    for (br, bc), color in zip(marker_blocks, marker_colors):
        r0, c0 = _block_origin(br, bc)
        grid[r0 + 1, c0 + 1] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    for (br, bc), color in zip(marker_blocks, marker_colors):
        expected = red_pattern if color == RED else blue_pattern
        if not np.array_equal(_read_block(out, br, bc), expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-legend input")
