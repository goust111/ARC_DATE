from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PINK = 6

PANEL_SIZE = 3
INPUT_SHAPE = (PANEL_SIZE, PANEL_SIZE * 2)
OUTPUT_SHAPE = (PANEL_SIZE * PANEL_SIZE, PANEL_SIZE * PANEL_SIZE)
GEN_RETRY_LIMIT = 400

# (left_only, right_only, overlap)
VALID_CATEGORY_TRIPLETS = (
    (2, 2, 1),
    (2, 3, 1),
    (3, 2, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 1, 2),
    (2, 2, 2),
    (1, 1, 3),
)


def _parse_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, f"expected a 3x6 grid, got {g.shape}"

    left = g[:, :PANEL_SIZE]
    right = g[:, PANEL_SIZE:]
    assert set(map(int, np.unique(left))).issubset({BLACK, BLUE})
    assert set(map(int, np.unique(right))).issubset({BLACK, RED})

    left_mask = left == BLUE
    right_mask = right == RED
    return left_mask, right_mask


def _render_output_from_masks(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    overlap = left_mask & right_mask
    union_mask = left_mask | right_mask

    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    for outer_mask, color in (
        (left_only, BLUE),
        (right_only, RED),
        (overlap, PINK),
    ):
        out[np.kron(outer_mask, union_mask)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a color-coded Kronecker self-product from the three-way union of two aligned side-by-side masks.

    Input structure:
    - The grid is 3x6 with black background, split into two aligned 3x3 panels at column 3.
    - The left panel uses blue foreground cells and the right panel uses red foreground cells.
    - Corresponding cells across the two panels define four per-position states: left-only, right-only, overlap, or empty.

    Rule:
    - Compute the 3x3 boolean union mask of the two panels, marking every left-only, right-only, and overlap position as foreground.
    - Produce a 9x9 output by taking the Kronecker self-product of that union mask with itself.
    - For each foreground cell of the outer 3x3 union mask, place one 3x3 copy of the union mask in the corresponding output block.
    - Color the foreground cells inside that copied block by the outer cell's source class: blue for left-only, red for right-only, and pink for overlap; copied background stays black.

    Key insight:
    - The small union mask controls the geometry of every repeated block, while the source category of each outer union cell controls the color of that block.
    """
    left_mask, right_mask = _parse_masks(grid)
    return _render_output_from_masks(left_mask, right_mask)


def _build_input_from_categories(category_grid: np.ndarray) -> np.ndarray:
    left_mask = (category_grid == 1) | (category_grid == 3)
    right_mask = (category_grid == 2) | (category_grid == 3)

    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, :PANEL_SIZE][left_mask] = BLUE
    grid[:, PANEL_SIZE:][right_mask] = RED
    return grid


def _validate_input_semantics(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_mask, right_mask = _parse_masks(grid)

    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    overlap = left_mask & right_mask
    union_mask = left_mask | right_mask

    assert 3 <= left_count <= 4
    assert 3 <= right_count <= 4
    assert int(left_only.sum()) >= 1
    assert int(right_only.sum()) >= 1
    assert int(overlap.sum()) >= 1
    assert 3 <= int(union_mask.sum()) <= 6
    assert not np.array_equal(union_mask, left_mask)
    assert not np.array_equal(union_mask, right_mask)
    return left_mask, right_mask


def _independent_oracle(grid: np.ndarray) -> np.ndarray:
    left_mask, right_mask = _validate_input_semantics(grid)
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    overlap = left_mask & right_mask
    union_mask = left_mask | right_mask

    expected = np.zeros(OUTPUT_SHAPE, dtype=int)
    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            category = 0
            if left_only[r, c]:
                category = BLUE
            elif right_only[r, c]:
                category = RED
            elif overlap[r, c]:
                category = PINK
            if category == 0:
                continue

            block = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
            block[union_mask] = category
            rr = r * PANEL_SIZE
            cc = c * PANEL_SIZE
            expected[rr : rr + PANEL_SIZE, cc : cc + PANEL_SIZE] = block

    union_count = int(union_mask.sum())
    assert int((expected == BLUE).sum()) == union_count * int(left_only.sum())
    assert int((expected == RED).sum()) == union_count * int(right_only.sum())
    assert int((expected == PINK).sum()) == union_count * int(overlap.sum())
    return expected


def _sample_category_grid(rng: np.random.Generator) -> np.ndarray:
    left_only_count, right_only_count, overlap_count = VALID_CATEGORY_TRIPLETS[
        int(rng.integers(len(VALID_CATEGORY_TRIPLETS)))
    ]
    union_count = left_only_count + right_only_count + overlap_count

    cells = rng.choice(PANEL_SIZE * PANEL_SIZE, size=union_count, replace=False)
    rng.shuffle(cells)

    category_grid = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    split_a = left_only_count
    split_b = left_only_count + right_only_count
    category_grid.flat[cells[:split_a]] = 1
    category_grid.flat[cells[split_a:split_b]] = 2
    category_grid.flat[cells[split_b:]] = 3
    return category_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        category_grid = _sample_category_grid(rng)
        grid = _build_input_from_categories(category_grid)

        expected = _independent_oracle(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        if set(map(int, np.unique(actual))) != {BLACK, BLUE, RED, PINK}:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
