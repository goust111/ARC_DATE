from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = np.arange(1, 10, dtype=int)
GEN_RETRIES = 400
MIN_QUAD_SIDE = 5
MAX_QUAD_SIDE = 14

ONE_PRESENT_PATTERNS = np.eye(4, dtype=int)
ONE_ABSENT_PATTERNS = 1 - ONE_PRESENT_PATTERNS
TWO_TWO_PATTERNS = np.array(
    [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ],
    dtype=int,
)
ALL_PATTERNS = np.vstack(
    [
        ONE_PRESENT_PATTERNS,
        ONE_ABSENT_PATTERNS,
        TWO_TWO_PATTERNS,
        np.zeros((1, 4), dtype=int),
        np.ones((1, 4), dtype=int),
    ]
)
ALL_PATTERN_WEIGHTS = np.array([4] * 4 + [4] * 4 + [2] * 6 + [1, 1], dtype=float)
ALL_PATTERN_WEIGHTS /= ALL_PATTERN_WEIGHTS.sum()


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)
    row_candidates = [
        r
        for r in range(1, height - 1)
        if int(grid[r, 0]) != BLACK and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(1, width - 1)
        if int(grid[0, c]) != BLACK and np.all(grid[:, c] == grid[0, c])
    ]

    matches = []
    for r in row_candidates:
        for c in col_candidates:
            color = int(grid[r, 0])
            if int(grid[0, c]) != color:
                continue
            if height != 2 * r + 1 or width != 2 * c + 1:
                continue
            matches.append((r, c, color))

    assert len(matches) == 1, "expected one solid non-black cross that splits the grid into equal quadrants"
    return matches[0]


def _extract_quadrants(grid: np.ndarray, cross_r: int, cross_c: int) -> list[np.ndarray]:
    quadrants = [
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 :],
        grid[cross_r + 1 :, :cross_c],
        grid[cross_r + 1 :, cross_c + 1 :],
    ]
    shape = quadrants[0].shape
    assert shape[0] > 0 and shape[1] > 0, "quadrants must be non-empty"
    assert all(q.shape == shape for q in quadrants), "quadrants must be equal-sized"
    return quadrants


def _quadrant_colors(quadrants: list[np.ndarray], separator: int) -> np.ndarray:
    colors = []
    for quadrant in quadrants:
        present = _colors_present(quadrant, exclude={BLACK})
        assert len(present) == 1, "each quadrant must contain exactly one dedicated foreground color"
        color = int(present[0])
        assert color != separator, "separator color cannot be reused inside a quadrant"
        assert np.all((quadrant == BLACK) | (quadrant == color)), "quadrant must be a black-and-one-color mask"
        colors.append(color)
    assert len(set(colors)) == 4, "the four quadrant colors must be distinct"
    return np.array(colors, dtype=int)


def _masks_to_output(masks: np.ndarray, colors: np.ndarray) -> np.ndarray:
    counts = masks.sum(axis=0)
    out = np.zeros(counts.shape, dtype=int)

    single_present = counts == 1
    if np.any(single_present):
        winners = np.argmax(masks, axis=0)
        out[single_present] = colors[winners[single_present]]

    single_absent = counts == 3
    if np.any(single_absent):
        losers = np.argmin(masks, axis=0)
        out[single_absent] = colors[losers[single_absent]]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the four quadrant masks cell by cell and color only the unique dissenter.

    Input structure:
    - A full solid non-black row and a full solid non-black column form a cross that
      splits the grid into four equal quadrants on a black background.
    - Each quadrant contains exactly one dedicated foreground color, used as a sparse
      mask on black, and the four quadrant colors are all distinct.

    Rule:
    - Remove the separator cross conceptually and align the four quadrants cellwise.
    - For each aligned cell, compare only whether each quadrant has its own color there
      or is black there.
    - If exactly one quadrant disagrees with the other three, paint the output cell with
      that quadrant's dedicated color: the lone present quadrant for a 1-vs-3 split, or
      the lone absent quadrant for a 3-vs-1 split. Leave all other cases black.

    Key insight:
    - The odd-one-out test is symmetric between presence and absence, so a cell with
      three colored quadrants and one empty quadrant must output the missing quadrant's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    cross_r, cross_c, separator = _find_separator_cross(g)
    quadrants = _extract_quadrants(g, cross_r, cross_c)
    colors = _quadrant_colors(quadrants, separator)
    masks = np.stack([quadrant == color for quadrant, color in zip(quadrants, colors)], axis=0)
    return _masks_to_output(masks, colors)


def _sample_patterns(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    patterns = np.zeros((height, width, 4), dtype=int)
    positions = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(positions)

    forced = np.vstack(
        [
            ONE_PRESENT_PATTERNS,
            ONE_ABSENT_PATTERNS,
            TWO_TWO_PATTERNS[[int(rng.integers(0, len(TWO_TWO_PATTERNS)))]],
        ]
    )
    for (r, c), pattern in zip(positions, forced):
        patterns[r, c] = pattern

    remaining = positions[len(forced) :]
    if remaining:
        sampled = ALL_PATTERNS[
            rng.choice(len(ALL_PATTERNS), size=len(remaining), p=ALL_PATTERN_WEIGHTS)
        ]
        for (r, c), pattern in zip(remaining, sampled):
            patterns[r, c] = pattern

    return patterns


def _render_input(patterns: np.ndarray, separator: int, colors: np.ndarray) -> np.ndarray:
    height, width, _ = patterns.shape
    grid = np.zeros((2 * height + 1, 2 * width + 1), dtype=int)
    grid[height, :] = separator
    grid[:, width] = separator

    quadrants = [np.where(patterns[:, :, i].astype(bool), int(colors[i]), BLACK) for i in range(4)]
    grid[:height, :width] = quadrants[0]
    grid[:height, width + 1 :] = quadrants[1]
    grid[height + 1 :, :width] = quadrants[2]
    grid[height + 1 :, width + 1 :] = quadrants[3]
    return grid


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = np.rot90(grid, k=int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _semantic_oracle(grid: np.ndarray) -> np.ndarray:
    cross_r, cross_c, separator = _find_separator_cross(grid)
    quadrants = _extract_quadrants(grid, cross_r, cross_c)
    colors = _quadrant_colors(quadrants, separator)
    masks = np.stack([quadrant == color for quadrant, color in zip(quadrants, colors)], axis=0)

    out = np.zeros(quadrants[0].shape, dtype=int)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            present = [bool(masks[idx, r, c]) for idx in range(4)]
            count = sum(present)
            if count == 1:
                out[r, c] = int(colors[present.index(True)])
            elif count == 3:
                out[r, c] = int(colors[present.index(False)])
    return out


def _validate_sample(grid: np.ndarray) -> bool:
    try:
        cross_r, cross_c, separator = _find_separator_cross(grid)
        quadrants = _extract_quadrants(grid, cross_r, cross_c)
        colors = _quadrant_colors(quadrants, separator)
        masks = np.stack([quadrant == color for quadrant, color in zip(quadrants, colors)], axis=0)
        counts = masks.sum(axis=0)

        assert grid.shape[0] <= 30 and grid.shape[1] <= 30
        assert all(0 < int(mask.sum()) < mask.size for mask in masks), "each quadrant mask should show both color and black"
        assert int(np.count_nonzero(counts == 1)) >= 4, "must include 1-vs-3 cells"
        assert int(np.count_nonzero(counts == 3)) >= 4, "must include 3-vs-1 cells"
        assert int(np.count_nonzero(counts == 2)) >= 1, "must include a 2-vs-2 counterexample"
        assert all(0.2 <= float(mask.mean()) <= 0.8 for mask in masks), "quadrant masks should stay visually sparse"

        expected = _semantic_oracle(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected), "solver must match the semantic oracle"

        present_output_colors = set(map(int, np.unique(actual))) - {BLACK}
        assert present_output_colors == set(map(int, colors)), "all quadrant colors should appear in the output"
        return True
    except AssertionError:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray:
    quad_h = int(rng.integers(MIN_QUAD_SIDE, MAX_QUAD_SIDE + 1))
    quad_w = int(rng.integers(MIN_QUAD_SIDE, MAX_QUAD_SIDE + 1))
    colors = rng.choice(PALETTE, size=5, replace=False)
    separator = int(colors[0])
    quadrant_colors = colors[1:].astype(int)

    patterns = _sample_patterns(quad_h, quad_w, rng)
    grid = _render_input(patterns, separator, quadrant_colors)
    return _apply_random_symmetry(grid, rng)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        candidate = _generate_attempt(rng)
        if _validate_sample(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 4")
