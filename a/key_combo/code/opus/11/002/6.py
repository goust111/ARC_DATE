from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SPRITE_COLORS = (BLUE, GREEN, YELLOW, 5, 6, 7, 8, 9)

MACRO_MIN_N = 3
MACRO_MAX_N = 5
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sorted_centers(centers: np.ndarray) -> np.ndarray:
    counts = centers.sum(axis=0)
    assert len(set(map(int, counts))) == centers.shape[1], "macro-column fill counts must be unique"
    order = np.argsort(-counts, kind="stable")
    return centers[:, order]


def _stamp_slots(mask: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    n = int(mask.shape[0])
    out = np.zeros((n * 3, n * 4 - 1), dtype=int)
    for r in range(n):
        for c in range(n):
            if bool(mask[r, c]):
                rs = r * 3
                cs = c * 4
                out[rs : rs + 3, cs : cs + 3] = sprite
    return out


def _render_macro_grid(centers: np.ndarray) -> np.ndarray:
    n = int(centers.shape[0])
    h = 2 * n + 1
    w = 4 * n - 1
    grid = np.full((h, w), RED, dtype=int)
    gutter_cols = np.arange(3, w, 4)
    center_rows = np.arange(1, h, 2)
    center_cols = np.arange(1, w, 4)
    grid[:, gutter_cols] = BLACK
    grid[np.ix_(center_rows, center_cols)] = np.where(centers, RED, BLACK)
    return grid


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    macro_box = _bbox(g == RED)
    assert macro_box is not None, "expected a red macro grid"
    mr0, mc0, mr1, mc1 = macro_box
    macro = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
    mh, mw = macro.shape

    assert mh % 2 == 1 and (mw + 1) % 4 == 0, "invalid macro-grid geometry"
    n = (mh - 1) // 2
    assert MACRO_MIN_N <= n <= MACRO_MAX_N and mw == 4 * n - 1, "expected an NxN macro grid"
    assert set(map(int, np.unique(macro))).issubset({BLACK, RED}), "macro area should use only black and red"
    assert int(np.count_nonzero(g == RED)) == int(np.count_nonzero(macro == RED)), "red must belong only to the macro grid"

    center_rows = np.arange(1, mh, 2)
    center_cols = np.arange(1, mw, 4)
    gutter_cols = np.arange(3, mw, 4)

    frame_mask = np.ones((mh, mw), dtype=bool)
    frame_mask[:, gutter_cols] = False
    frame_mask[np.ix_(center_rows, center_cols)] = False

    assert np.all(macro[:, gutter_cols] == BLACK), "expected black gutter columns"
    assert np.all(macro[frame_mask] == RED), "expected red macro-cell frames"

    centers = macro[np.ix_(center_rows, center_cols)] == RED

    sprite_mask = (g != BLACK) & (g != RED)
    sprite_box = _bbox(sprite_mask)
    assert sprite_box is not None, "expected a lower 3x3 sprite"
    sr0, sc0, sr1, sc1 = sprite_box
    assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (3, 3), "sprite must be 3x3"
    assert sr0 > mr1, "sprite must be below the macro grid"
    assert np.all(g[mr1 + 1 : sr0, :] == BLACK), "macro grid and sprite must be separated by blank rows"

    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert np.all((sprite != BLACK) & (sprite != RED)), "sprite cannot contain black or red"
    assert int(np.count_nonzero(sprite_mask)) == 9, "sprite should be the only non-black/non-red object"

    covered = np.zeros_like(g, dtype=bool)
    covered[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
    covered[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    assert np.all(g[~covered] == BLACK), "unexpected extra objects outside the macro grid and sprite"

    return centers, sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Sort macro-center columns by fill count, then use the sorted center matrix as a sprite-stamping mask.

    Input structure:
    - The upper object is a square red macro grid made from 3x3 cells with black vertical gutters; each cell has a fixed red frame and a center that is either red or black.
    - A separate 3x3 multicolor sprite sits below the macro grid, with 1-2 blank separator rows and no black or red inside the sprite.

    Rule:
    - Read the macro-cell centers as an NxN binary matrix, where red means filled and black means empty.
    - Count filled centers in each macro-column and sort whole columns from highest count to lowest, preserving each column's internal top-to-bottom pattern.
    - Build an NxN array of 3x3 output slots with 1-column black gutters between slots, and stamp the lower 3x3 sprite into every slot whose sorted center is filled.

    Key insight:
    - The macro grid is not copied directly; its center-column sort only rearranges the binary placement mask that decides where sprite copies appear.
    """
    centers, sprite = _parse_input(grid)
    sorted_mask = _sorted_centers(centers)
    return _stamp_slots(sorted_mask, sprite)


def _random_sprite() -> np.ndarray:
    for _ in range(200):
        n_colors = random.choice((3, 4, 5))
        palette = random.sample(SPRITE_COLORS, n_colors)
        cells = np.array([random.choice(palette) for _ in range(9)], dtype=int).reshape(3, 3)
        values, counts = np.unique(cells, return_counts=True)
        if len(values) != n_colors:
            continue
        if counts.max() > 5:
            continue
        if any(len(set(map(int, row))) < 2 for row in cells):
            continue
        if any(len(set(map(int, col))) < 2 for col in cells.T):
            continue
        return cells
    raise RuntimeError("failed to build a diverse sprite")


def _random_sorted_centers(n: int) -> np.ndarray:
    counts = sorted(random.sample(range(n + 1), n), reverse=True)
    centers = np.zeros((n, n), dtype=bool)
    for c, count in enumerate(counts):
        if count:
            rows = random.sample(range(n), count)
            centers[rows, c] = True
    assert 0 < int(centers.sum()) < n * n
    return centers


def _non_identity_permutation(n: int) -> list[int]:
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    return perm


def _compose_input(macro: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    macro_h, macro_w = map(int, macro.shape)
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    right_margin = random.randint(0, 4)
    macro_left = random.randint(0, 4)
    sprite_left_max = min(22, macro_left + macro_w + 2)
    sprite_left = random.randint(0, sprite_left_max)
    gap_rows = random.choice((1, 2))

    height = top_margin + macro_h + gap_rows + 3 + bottom_margin
    width = max(macro_left + macro_w, sprite_left + 3) + right_margin
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    grid[top_margin : top_margin + macro_h, macro_left : macro_left + macro_w] = macro
    sprite_top = top_margin + macro_h + gap_rows
    grid[sprite_top : sprite_top + 3, sprite_left : sprite_left + 3] = sprite
    return grid


def _validate_candidate(grid: np.ndarray, input_centers: np.ndarray, sprite: np.ndarray) -> bool:
    parsed_centers, parsed_sprite = _parse_input(grid)
    if not np.array_equal(parsed_centers, input_centers):
        return False
    if not np.array_equal(parsed_sprite, sprite):
        return False

    sorted_centers = _sorted_centers(parsed_centers)
    if np.array_equal(sorted_centers, parsed_centers):
        return False

    out = main(grid)
    expected = _stamp_slots(sorted_centers, sprite)
    if not np.array_equal(out, expected):
        return False

    unsorted_out = _stamp_slots(parsed_centers, sprite)
    if np.array_equal(out, unsorted_out):
        return False

    n = parsed_centers.shape[0]
    if out.shape != (n * 3, n * 4 - 1):
        return False
    if np.count_nonzero(sorted_centers) < 2 or np.count_nonzero(~sorted_centers) < 2:
        return False

    for c in range(n - 1):
        if np.any(out[:, 3 + 4 * c] != BLACK):
            return False
    for r in range(n):
        for c in range(n):
            block = out[r * 3 : (r + 1) * 3, c * 4 : c * 4 + 3]
            if sorted_centers[r, c]:
                if not np.array_equal(block, sprite):
                    return False
            else:
                if np.any(block != BLACK):
                    return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(MACRO_MIN_N, MACRO_MAX_N)
    sorted_centers = _random_sorted_centers(n)
    perm = _non_identity_permutation(n)
    input_centers = sorted_centers[:, perm]
    sprite = _random_sprite()
    grid = _compose_input(_render_macro_grid(input_centers), sprite)
    if _validate_candidate(grid, input_centers, sprite):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
