import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_w, crop_bbox
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden 5x5 patch by rotating the scene 180 degrees and cropping the blue-mask location.

    Input structure
    A scene contains exactly one solid 5x5 blue mask over a larger rotationally symmetric pattern.

    Rule
    Locate the blue mask, rotate the whole input grid 180 degrees, and return the 5x5 patch at the mask's original bbox.

    Key insight
    The blue block is an occluder; the hidden pixels are supplied by the opposite 180-degree-corresponding part of the same pattern.
    """
    g = np.asarray(grid, dtype=int)

    mask = g == Color.BLUE
    b = bbox(mask)
    assert b is not None
    assert bbox_h(b) == 5 and bbox_w(b) == 5
    assert int(mask.sum()) == 25
    assert np.all(crop_bbox(mask, b))

    out = crop_bbox(np.rot90(g, 2), b)
    assert out.shape == (5, 5)
    assert not np.any(out == Color.BLUE)
    return out

# --------------- generator (o5x) ---------------

# Canvas, block, and blue-mask geometry.
BLOCK_SIZE, GRID_SIZE, BLUE_SIZE = 6, 24, 5
BLUE_TOP_LEFTS = range(GRID_SIZE - BLUE_SIZE + 1)
OPPOSITE_TOP_LEFT_BASE = GRID_SIZE - BLUE_SIZE
HALF_TURN = 2
# Transpose-symmetric 6x6 block foreground counts.
BLOCK_ON_COUNT_RANGE = (16, 29)
DIAGONAL_CELLS = list(range(BLOCK_SIZE))
TRANSPOSE_PAIRS = [(r, c) for r in range(BLOCK_SIZE) for c in range(r + 1, BLOCK_SIZE)]
# Color sharing modes for A, B, C blocks.
COLOR_POOL = [c for c in range(10) if c not in (Color.BLACK, Color.BLUE)]
COLOR_SHARE_MODES = ("ab", "bc", "abc")
# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_transpose_mask() -> np.ndarray:
    target = random.randint(*BLOCK_ON_COUNT_RANGE)
    diagonal_choices = [
        d for d in range(BLOCK_SIZE + 1)
        if (target - d) % 2 == 0 and 0 <= (target - d) // 2 <= len(TRANSPOSE_PAIRS)
    ]
    diag_count = random.choice(diagonal_choices)
    pair_count = (target - diag_count) // 2

    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    diag_cells = random.sample(DIAGONAL_CELLS, diag_count)
    for i in diag_cells:
        mask[i, i] = True

    for r, c in random.sample(TRANSPOSE_PAIRS, pair_count):
        mask[r, c] = True
        mask[c, r] = True
    return mask


def _gen_color_assignment() -> tuple[int, int, int]:
    mode = random.choice(COLOR_SHARE_MODES)
    if mode == "ab":
        shared, other = random.sample(COLOR_POOL, 2)
        return shared, shared, other
    if mode == "bc":
        other, shared = random.sample(COLOR_POOL, 2)
        return other, shared, shared
    return tuple(random.sample(COLOR_POOL, 3))


def _gen_colored_block(color: int) -> np.ndarray:
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    block[_gen_transpose_mask()] = color
    return block


def _gen_pattern() -> np.ndarray:
    color_a, color_b, color_c = _gen_color_assignment()
    block_a = _gen_colored_block(color_a)
    block_b = _gen_colored_block(color_b)
    block_c = _gen_colored_block(color_c)
    q1 = np.block([[block_a, block_b], [block_b, block_c]])
    top = np.concatenate([q1, np.fliplr(q1)], axis=1)
    bottom = np.flipud(top)
    return np.concatenate([top, bottom], axis=0)


def _gen_blue_position(pattern: np.ndarray) -> tuple[int, int] | None:
    positions = []
    for r in BLUE_TOP_LEFTS:
        for c in BLUE_TOP_LEFTS:
            rr, cc = OPPOSITE_TOP_LEFT_BASE - r, OPPOSITE_TOP_LEFT_BASE - c
            row_overlap = max(r, rr) <= min(r + BLUE_SIZE - 1, rr + BLUE_SIZE - 1)
            col_overlap = max(c, cc) <= min(c + BLUE_SIZE - 1, cc + BLUE_SIZE - 1)
            if row_overlap and col_overlap:
                continue
            opposite_patch = pattern[rr : rr + BLUE_SIZE, cc : cc + BLUE_SIZE]
            if np.any(opposite_patch != Color.BLACK):
                positions.append((r, c))
    if not positions:
        return None
    return random.choice(positions)


def _gen_attempt() -> np.ndarray | None:
    pattern = _gen_pattern()
    pos = _gen_blue_position(pattern)
    if pos is None:
        return None

    r, c = pos
    inp = pattern.copy()
    inp[r : r + BLUE_SIZE, c : c + BLUE_SIZE] = Color.BLUE
    expected = np.rot90(pattern, HALF_TURN)[r : r + BLUE_SIZE, c : c + BLUE_SIZE]
    got = main(inp)
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ff805c23 input")
