import numpy as np

from arc_utils import Color, bbox


def _shell_color(depth: int, max_depth: int, frame_color: int, bg: int) -> int:
    if max_depth == 4 and depth == max_depth:
        return bg
    phase = depth % 4
    if phase == 1:
        return Color.RED
    if phase == 3:
        return bg
    return frame_color


def main(grid: np.ndarray) -> np.ndarray:
    """Fill a hollow grey square frame with nested square shells cycling grey, red, grey, and black.

    Input structure
    - The grid contains black background and one complete one-cell-thick grey square frame.
    - The frame interior and any exterior padding are black.

    Rule
    - Preserve the frame bbox and recolor every cell in it by its distance from the frame border.
    - Shell depths repeat the color cycle grey, red, grey, black while moving inward.
    - In the 9x9 small-frame case, the lone center cell remains in the black center left by the first blank shell.

    Key insight
    - The output is a distance-shell rewrite of the single square frame, not a movement or matching task.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    present = set(map(int, g.ravel()))
    assert bg in present
    frame_colors = present - {bg}
    assert frame_colors == {Color.GREY}
    frame_color = Color.GREY

    frame_box = bbox(g == frame_color)
    assert frame_box is not None
    r0, c0, r1, c1 = frame_box
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side >= 3

    expected_frame = np.zeros_like(g, dtype=bool)
    expected_frame[r0 : r1 + 1, c0] = True
    expected_frame[r0 : r1 + 1, c1] = True
    expected_frame[r0, c0 : c1 + 1] = True
    expected_frame[r1, c0 : c1 + 1] = True
    assert np.array_equal(g == frame_color, expected_frame)

    out = g.copy()
    max_depth = side // 2
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = _shell_color(depth, max_depth, frame_color, bg)
    return out

# --------------- generator (o5x) ---------------

# Canvas/frame sampling bounds.
GRID_SIZE_MIN = 6
GRID_SIZE_MAX_EXCLUSIVE = 31
FRAME_SIDE_MIN = 6
FRAME_SIDE_MAX = 25
MAX_SIDE_PADDING = 4
MAX_TOTAL_PADDING = 2 * MAX_SIDE_PADDING

# Shell color cycle and retry budget.
GEN_BG, GEN_FRAME, GEN_RED = Color.BLACK, Color.GREY, Color.RED
SHELL_PERIOD = 4
RED_PHASE, BG_PHASE = 1, 3
SPECIAL_CENTER_DEPTH = 4
GENERATION_RETRIES = 300


def _gen_expected_output(grid: np.ndarray, r0: int, c0: int, side: int) -> np.ndarray:
    out = grid.copy()
    rows = np.arange(side)[:, None]
    cols = np.arange(side)[None, :]
    edge_dist = np.minimum(
        np.minimum(rows, side - 1 - rows),
        np.minimum(cols, side - 1 - cols),
    )
    phase = edge_dist % SHELL_PERIOD
    patch = np.full((side, side), GEN_FRAME, dtype=int)
    patch[phase == RED_PHASE] = GEN_RED
    patch[phase == BG_PHASE] = GEN_BG
    max_depth = side // 2
    if max_depth == SPECIAL_CENTER_DEPTH:
        patch[edge_dist == max_depth] = GEN_BG
    out[r0 : r0 + side, c0 : c0 + side] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_size = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE))
    min_pad_sum = max(0, grid_size - FRAME_SIDE_MAX)
    max_pad_sum = min(MAX_TOTAL_PADDING, grid_size - FRAME_SIDE_MIN)
    if min_pad_sum > max_pad_sum:
        return None

    pad_sum = int(np.random.randint(min_pad_sum, max_pad_sum + 1))
    side = grid_size - pad_sum
    split_lo = max(0, pad_sum - MAX_SIDE_PADDING)
    split_hi = min(MAX_SIDE_PADDING, pad_sum)
    r0 = int(np.random.randint(split_lo, split_hi + 1))
    c0 = int(np.random.randint(split_lo, split_hi + 1))
    r1 = r0 + side - 1
    c1 = c0 + side - 1

    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[r0 : r1 + 1, c0] = GEN_FRAME
    grid[r0 : r1 + 1, c1] = GEN_FRAME
    grid[r0, c0 : c1 + 1] = GEN_FRAME
    grid[r1, c0 : c1 + 1] = GEN_FRAME

    expected = _gen_expected_output(grid, r0, c0, side)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a3f84088 input")
