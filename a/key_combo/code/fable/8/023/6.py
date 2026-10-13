from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


CELL_SIZE = 3
GAP_SIZE = 1
STEP = CELL_SIZE + GAP_SIZE
MIN_SEED_SIDE = 2
MAX_SEED_SIDE = 5
MAX_GENERATION_ATTEMPTS = 300


def _extract_seed(input_grid: np.ndarray) -> np.ndarray:
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    active_bbox = bbox(g != Color.BLACK)
    assert active_bbox is not None, "expected a visible macro grid"
    body = crop_bbox(g, active_bbox)
    h, w = map(int, body.shape)

    assert h % STEP == CELL_SIZE and w % STEP == CELL_SIZE, "macro grid body must be 3x3 cells with 1-cell gaps"
    seed_h = (h + 1) // STEP
    seed_w = (w + 1) // STEP
    assert MIN_SEED_SIDE <= seed_h <= MAX_SEED_SIDE
    assert MIN_SEED_SIDE <= seed_w <= MAX_SEED_SIDE

    for r in range(h):
        if r % STEP == CELL_SIZE:
            assert np.all(body[r, :] == Color.BLACK), "horizontal gaps must be black"
    for c in range(w):
        if c % STEP == CELL_SIZE:
            assert np.all(body[:, c] == Color.BLACK), "vertical gaps must be black"

    seed = np.zeros((seed_h, seed_w), dtype=int)
    for mr in range(seed_h):
        for mc in range(seed_w):
            r0 = mr * STEP
            c0 = mc * STEP
            cell = body[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            assert cell.shape == (CELL_SIZE, CELL_SIZE)
            center = int(cell[1, 1])
            assert center in {Color.BLACK, Color.BLUE}, "cell center must be black or blue"

            frame = cell.copy()
            frame[1, 1] = Color.RED
            assert np.all(frame == Color.RED), "every non-center cell must be red"
            seed[mr, mc] = 1 if center == Color.BLUE else 0

    return seed


def _expand_seed(seed: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    expanded = np.kron(seed, seed)
    return np.where(expanded == 1, Color.BLUE, Color.BLACK).astype(int)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Read the hidden binary seed from macro-cell centers and expand it into a self-similar output.

    Input structure:
    - The input is a rectangular macro grid of red 3x3 cells separated by one-cell black gaps.
    - Every cell is identical except for its center pixel, which is either black or blue.
    - The red frames and black gaps only package the hidden information; the meaningful data is the matrix of center colors.

    Rule:
    - Read a binary seed matrix where a blue center means 1 and a black center means 0.
    - Produce an output of shape (m^2, n^2), where m x n is the seed size.
    - For each seed entry equal to 1, replace that position by a full blue/black copy of the entire seed matrix; for each 0, place an equally sized all-black block.

    Key insight:
    - The visible macro structure is only a wrapper; the true object is the hidden center-pixel matrix, and the output is its self-replacement product rather than a crop or resize.
    """
    seed = _extract_seed(input_grid)
    return _expand_seed(seed)


def _render_input(seed: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    h, w = map(int, seed.shape)
    grid = np.full((h * STEP - GAP_SIZE, w * STEP - GAP_SIZE), Color.BLACK, dtype=int)

    for mr in range(h):
        for mc in range(w):
            r0 = mr * STEP
            c0 = mc * STEP
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = Color.RED
            grid[r0 + 1, c0 + 1] = Color.BLUE if int(seed[mr, mc]) else Color.BLACK

    return grid


def _single_hot_seed(h: int, w: int) -> np.ndarray:
    seed = np.zeros((h, w), dtype=int)
    r = random.randrange(h)
    c = random.randrange(w)
    seed[r, c] = 1
    return seed


def _single_hole_seed(h: int, w: int) -> np.ndarray:
    seed = np.ones((h, w), dtype=int)
    r = random.randrange(h)
    c = random.randrange(w)
    seed[r, c] = 0
    return seed


def _stripe_seed(h: int, w: int) -> np.ndarray:
    seed = np.zeros((h, w), dtype=int)
    if random.random() < 0.5:
        cols = random.sample(range(w), k=random.randint(1, max(1, min(w, 2))))
        seed[:, cols] = 1
        extra = random.randint(0, max(0, h * w // 4))
    else:
        rows = random.sample(range(h), k=random.randint(1, max(1, min(h, 2))))
        seed[rows, :] = 1
        extra = random.randint(0, max(0, h * w // 4))

    zeros = np.argwhere(seed == 0).tolist()
    random.shuffle(zeros)
    for r, c in zeros[:extra]:
        seed[r, c] = 1
    return seed


def _checker_seed(h: int, w: int) -> np.ndarray:
    phase_r = random.randrange(2)
    phase_c = random.randrange(2)
    return np.fromfunction(lambda r, c: ((r + phase_r + c + phase_c) % 2 == 0).astype(int), (h, w), dtype=int)


def _random_seed(h: int, w: int) -> np.ndarray:
    density = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    return (np.random.random((h, w)) < density).astype(int)


def _block_seed(h: int, w: int) -> np.ndarray:
    seed = np.zeros((h, w), dtype=int)
    block_h = random.randint(1, h)
    block_w = random.randint(1, w)
    r0 = random.randint(0, h - block_h)
    c0 = random.randint(0, w - block_w)
    seed[r0 : r0 + block_h, c0 : c0 + block_w] = 1

    if random.random() < 0.5:
        for _ in range(random.randint(0, max(1, (h * w) // 5))):
            rr = random.randrange(h)
            cc = random.randrange(w)
            seed[rr, cc] = 1
    return seed


def _has_profile_variation(seed: np.ndarray) -> bool:
    seed = np.asarray(seed, dtype=int)
    row_patterns = {tuple(map(int, row)) for row in seed}
    col_patterns = {tuple(map(int, col)) for col in seed.T}
    return len(row_patterns) > 1 or len(col_patterns) > 1


def _valid_seed(seed: np.ndarray) -> bool:
    seed = np.asarray(seed, dtype=int)
    h, w = map(int, seed.shape)
    if seed.shape[0] * seed.shape[0] > 30 or seed.shape[1] * seed.shape[1] > 30:
        return False
    ones = int(seed.sum())
    if ones == 0 or ones == h * w:
        return False
    if not _has_profile_variation(seed):
        return False
    return True


def _sample_seed() -> np.ndarray:
    h = random.randint(MIN_SEED_SIDE, MAX_SEED_SIDE)
    w = random.randint(MIN_SEED_SIDE, MAX_SEED_SIDE)

    generators = [
        _single_hot_seed,
        _single_hole_seed,
        _stripe_seed,
        _checker_seed,
        _random_seed,
        _block_seed,
    ]

    for _ in range(80):
        seed = random.choice(generators)(h, w).astype(int)
        if not _valid_seed(seed):
            continue
        out = _expand_seed(seed)
        if np.count_nonzero(out == Color.BLUE) < 2:
            continue
        return seed

    raise RuntimeError("failed to sample a non-degenerate seed")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed = _sample_seed()
        inp = _render_input(seed)
        out = main(inp)
        expected = _expand_seed(seed)
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
