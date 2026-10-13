from collections import Counter

import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the one quadrant pattern that is not repeated among the four separated quadrants.

    Input structure:
    - The grid has odd height and width.
    - The middle row and middle column are the background color, forming a separator cross.
    - The separator cross divides the grid into four same-sized rectangular quadrant regions.

    Rule:
    - Compare the four quadrant regions as complete arrays.
    - Exactly one quadrant pattern occurs once while the other three share another pattern.
    - Output a copy of the uniquely occurring quadrant region.

    Key insight:
    - The large grid is a four-choice set of small patterns; the answer is the odd one out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and width % 2 == 1, "expected odd grid dimensions"

    mid_r, mid_c = height // 2, width // 2
    assert mid_r > 0 and mid_c > 0, "expected nonempty quadrant regions"
    bg = get_bg(g)
    assert np.all(g[mid_r, :] == bg), "expected a background separator row"
    assert np.all(g[:, mid_c] == bg), "expected a background separator column"

    quadrants = [
        g[:mid_r, :mid_c],
        g[:mid_r, mid_c + 1 :],
        g[mid_r + 1 :, :mid_c],
        g[mid_r + 1 :, mid_c + 1 :],
    ]
    shapes = {q.shape for q in quadrants}
    assert len(shapes) == 1, "expected four same-sized quadrant regions"

    keys = [tuple(map(tuple, q.tolist())) for q in quadrants]
    counts = Counter(keys)
    unique_indices = [idx for idx, key in enumerate(keys) if counts[key] == 1]
    assert len(unique_indices) == 1, "expected exactly one unique quadrant pattern"

    return quadrants[unique_indices[0]].copy()

# --------------- generator (o5x) ---------------

# Canvas/palette bounds.
QUAD_SIDE_MIN, QUAD_SIDE_MAX, SEPARATOR_WIDTH = 2, 5, 1
QUADRANTS_PER_AXIS, QUADRANT_SLOT_COUNT = 2, 4
BG_COLOR, FG_COLOR_MIN, FG_COLOR_MAX = 0, 1, 9

# Quadrant fill/mutation and retry controls.
MIN_CELLS_PER_COLOR, MIN_FG_NUM, MIN_FG_DEN = 1, 1, 4
MAX_FG_NUM, MAX_FG_DEN = 3, 4
UNIQUE_DIFF_COUNTS = (1, 2)
DIFF_DELTA_STEP, DIFF_BALANCE_DEN = 2, 2
MAX_GRID_FG_DEN, BG_MAJORITY_MARGIN = 2, 1
MAX_GENERATION_ATTEMPTS = 300


def _mutation_choices(area: int, height: int, width: int) -> list[tuple[int, int, int]]:
    """List valid (majority_fg, bg_to_fg, fg_to_bg) mutation options."""
    min_quad_fg = max(MIN_CELLS_PER_COLOR, (MIN_FG_NUM * area + MIN_FG_DEN - 1) // MIN_FG_DEN)
    max_quad_fg = min(area - MIN_CELLS_PER_COLOR, MAX_FG_NUM * area // MAX_FG_DEN)
    max_total_fg = (height * width - BG_MAJORITY_MARGIN) // MAX_GRID_FG_DEN
    choices = []
    for majority_fg in range(min_quad_fg, max_quad_fg + 1):
        for diff_count in UNIQUE_DIFF_COUNTS:
            for delta in range(-diff_count, diff_count + 1, DIFF_DELTA_STEP):
                bg_to_fg = (diff_count + delta) // DIFF_BALANCE_DEN
                fg_to_bg = (diff_count - delta) // DIFF_BALANCE_DEN
                unique_fg = majority_fg + delta
                fg_ok = min_quad_fg <= unique_fg <= max_quad_fg
                cells_ok = bg_to_fg <= area - majority_fg and fg_to_bg <= majority_fg
                total_ok = QUADRANT_SLOT_COUNT * majority_fg + delta <= max_total_fg
                if fg_ok and cells_ok and total_ok:
                    choices.append((majority_fg, bg_to_fg, fg_to_bg))
    return choices


def _make_quadrants(qh: int, qw: int, height: int, width: int, fg: int):
    """Build the repeated quadrant and its 1-2 cell variant."""
    area = qh * qw
    majority_fg, bg_to_fg, fg_to_bg = random.choice(_mutation_choices(area, height, width))
    flat = np.full(area, BG_COLOR, dtype=int)
    flat[random.sample(range(area), majority_fg)] = fg

    majority = flat.reshape(qh, qw)
    unique = majority.copy()
    changes = (
        ([idx for idx, value in enumerate(flat) if value == fg], fg_to_bg, BG_COLOR),
        ([idx for idx, value in enumerate(flat) if value == BG_COLOR], bg_to_fg, fg),
    )
    for cells, count, color in changes:
        for idx in random.sample(cells, count):
            unique.flat[idx] = color
    return majority, unique


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(QUAD_SIDE_MIN, QUAD_SIDE_MAX)
    qw = random.randint(QUAD_SIDE_MIN, QUAD_SIDE_MAX)
    height = QUADRANTS_PER_AXIS * qh + SEPARATOR_WIDTH
    width = QUADRANTS_PER_AXIS * qw + SEPARATOR_WIDTH
    fg = random.randint(FG_COLOR_MIN, FG_COLOR_MAX)
    majority, unique = _make_quadrants(qh, qw, height, width, fg)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    unique_slot = random.randrange(QUADRANT_SLOT_COUNT)
    bottom_r, right_c = qh + SEPARATOR_WIDTH, qw + SEPARATOR_WIDTH
    slots = ((0, 0), (0, right_c), (bottom_r, 0), (bottom_r, right_c))
    for slot_idx, (r0, c0) in enumerate(slots):
        grid[r0 : r0 + qh, c0 : c0 + qw] = unique if slot_idx == unique_slot else majority
    if not np.array_equal(main(grid), unique):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 88a62173 input")
