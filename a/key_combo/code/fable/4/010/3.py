from __future__ import annotations

import random

import numpy as np


# Gray is the visible separator color named in the puzzle statement.
_SEPARATOR_COLOR = 5
_WINDOW_COUNTS = (2, 3, 4, 5)
_WINDOW_COLORS = tuple(color for color in range(10) if color != _SEPARATOR_COLOR)
_GENERATION_ATTEMPTS = 200


def _split_windows(grid: np.ndarray) -> list[list[np.ndarray]]:
    """Validate the square divider layout and return its N by N window views."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    height, width = map(int, g.shape)
    assert height == width and height <= 30, "expected a square ARC grid of size at most 30"

    gray_rows = np.flatnonzero(np.all(g == _SEPARATOR_COLOR, axis=1))
    gray_cols = np.flatnonzero(np.all(g == _SEPARATOR_COLOR, axis=0))
    assert gray_rows.size == gray_cols.size and gray_rows.size >= 1, (
        "expected the same positive number of horizontal and vertical gray separators"
    )

    n = int(gray_rows.size + 1)
    expected_size = n * n + n - 1
    assert height == expected_size, "windows must be N by N in an N by N arrangement"
    expected_separators = np.arange(n, expected_size, n + 1, dtype=int)
    assert np.array_equal(gray_rows, expected_separators), (
        "horizontal separators must be single-cell lines between equal windows"
    )
    assert np.array_equal(gray_cols, expected_separators), (
        "vertical separators must be single-cell lines between equal windows"
    )

    starts = range(0, expected_size, n + 1)
    windows = [
        [g[row : row + n, col : col + n] for col in starts]
        for row in starts
    ]
    assert len(windows) == n and all(len(row) == n for row in windows)
    assert all(window.shape == (n, n) for row in windows for window in row)
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sample one address-matched cell from every window of a gray-divided grid.

    Input structure:
    - Full gray separator lines divide a square grid into N rows by N columns of windows.
    - Every window is itself an N by N multicolor pattern, using local (row, column) coordinates.

    Rule:
    - Produce an N by N grid. For output position (i,j), select the window in window-row i
      and window-column j, then copy the color at local coordinate (i,j) inside that window.

    Key insight:
    - A window's address is also its sampling coordinate; other cells, including fixed corners,
      centers, and majority-colored cells, are distractors.
    """
    windows = _split_windows(grid)
    n = len(windows)
    return np.array(
        [[int(windows[i][j][i, j]) for j in range(n)] for i in range(n)],
        dtype=int,
    )


def _coordinate_transform(index: int, r: int, c: int, n: int) -> tuple[int, int]:
    """Apply one of the eight square symmetries to a local (r,c) coordinate."""
    transforms = (
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    )
    return transforms[index]


def _sample_target(n: int, palette: list[int]) -> np.ndarray:
    """Choose a varied, asymmetric target so outputs do not collapse to one pattern."""
    cell_count = n * n
    required_colors = random.randint(3, min(6, cell_count, len(palette)))
    target = np.array(
        [palette[k] for k in range(required_colors)]
        + [random.choice(palette) for _ in range(cell_count - required_colors)],
        dtype=int,
    )
    np.random.shuffle(target)
    target = target.reshape(n, n)

    if any(np.unique(target[r]).size < 2 for r in range(n)):
        raise ValueError("constant target row")
    if any(np.unique(target[:, c]).size < 2 for c in range(n)):
        raise ValueError("constant target column")
    for transform_index in range(1, 8):
        transformed = np.empty_like(target)
        for r in range(n):
            for c in range(n):
                rr, cc = _coordinate_transform(transform_index, r, c, n)
                transformed[r, c] = target[rr, cc]
        if np.array_equal(transformed, target):
            raise ValueError("symmetric target")
    return target


def _make_window(n: int, sample: tuple[int, int], target: int, palette: list[int]) -> np.ndarray:
    """Make a multicolor window whose majority and fixed-position cues are distractors."""
    base = random.choice([color for color in palette if color != target])
    accent = random.choice([color for color in palette if color not in {target, base}])
    window = np.full((n, n), base, dtype=int)
    window[sample] = target

    free = [(r, c) for r in range(n) for c in range(n) if (r, c) != sample]
    random.shuffle(free)
    window[free.pop()] = accent

    # At N=2, two base cells form a strict majority and the two singleton colors
    # prevent "take the unique cell" from identifying the sampled cell. Larger
    # windows repeat the sampled color once and add sparse, visibly varied noise.
    if n >= 3:
        window[free.pop()] = target
        extra_count = random.randint(1, n - 1)
        extra_colors = [color for color in palette if color != base]
        for _ in range(extra_count):
            window[free.pop()] = random.choice(extra_colors)

    values, counts = np.unique(window, return_counts=True)
    majority = int(values[int(np.argmax(counts))])
    assert int(counts.max()) > int(np.partition(counts, -2)[-2])
    assert majority == base and majority != target
    assert np.unique(window).size >= 3
    return window


def _render_windows(windows: list[list[np.ndarray]]) -> np.ndarray:
    """Render N by N windows with one-cell gray separator lines."""
    n = len(windows)
    size = n * n + n - 1
    grid = np.full((size, size), _SEPARATOR_COLOR, dtype=int)
    for i in range(n):
        for j in range(n):
            r0, c0 = i * (n + 1), j * (n + 1)
            grid[r0 : r0 + n, c0 : c0 + n] = windows[i][j]
    return grid


def _direct_expected(grid: np.ndarray, n: int) -> np.ndarray:
    """Independent oracle using generated global geometry, without divider parsing."""
    out = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            out[i, j] = grid[i * (n + 1) + i, j * (n + 1) + j]
    return out


def _sample_with_maps(
    windows: list[list[np.ndarray]], source_map: int, coordinate_map: int
) -> np.ndarray:
    """Evaluate a common wrong rule that transforms window and/or local addresses."""
    n = len(windows)
    out = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            si, sj = _coordinate_transform(source_map, i, j, n)
            ri, rj = _coordinate_transform(coordinate_map, i, j, n)
            out[i, j] = windows[si][sj][ri, rj]
    return out


def _semantics_are_explicit(
    grid: np.ndarray,
    windows: list[list[np.ndarray]],
    expected: np.ndarray,
    palette: list[int],
) -> bool:
    """Check the stated rule, uniqueness cues, and sampled-cell dependencies."""
    n = len(windows)
    if grid.shape != (n * n + n - 1, n * n + n - 1):
        return False
    if max(grid.shape) > 30 or np.unique(expected).size < 3:
        return False

    try:
        actual = main(grid)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(_direct_expected(grid, n), expected):
        return False

    # Every panel is genuinely multicolor, and its strict majority is wrong.
    majority_answer = np.empty_like(expected)
    for i in range(n):
        for j in range(n):
            window = windows[i][j]
            values, counts = np.unique(window, return_counts=True)
            if values.size < 3:
                return False
            order = np.argsort(counts)
            if counts[order[-1]] <= counts[order[-2]]:
                return False
            majority_answer[i, j] = values[order[-1]]
    if np.array_equal(majority_answer, expected):
        return False

    # No one fixed local coordinate, nor any non-identity square symmetry of
    # either address, may reproduce the intended output.
    for fixed_r in range(n):
        for fixed_c in range(n):
            fixed = np.array(
                [[windows[i][j][fixed_r, fixed_c] for j in range(n)] for i in range(n)]
            )
            if np.array_equal(fixed, expected):
                return False
    for transform_index in range(1, 8):
        if np.array_equal(_sample_with_maps(windows, 0, transform_index), expected):
            return False
        if np.array_equal(_sample_with_maps(windows, transform_index, 0), expected):
            return False
        if np.array_equal(
            _sample_with_maps(windows, transform_index, transform_index), expected
        ):
            return False

    # Counterfactual checks establish that all N^2 selected cells are necessary:
    # changing one changes exactly its matching output cell. Conversely, changing
    # one unselected distractor in every window leaves the output unchanged.
    replacement_choices = [color for color in palette if color != _SEPARATOR_COLOR]
    distractor_grid = grid.copy()
    for i in range(n):
        for j in range(n):
            global_r = i * (n + 1) + i
            global_c = j * (n + 1) + j
            old = int(grid[global_r, global_c])
            replacement = next(color for color in replacement_choices if color != old)
            changed_grid = grid.copy()
            changed_grid[global_r, global_c] = replacement
            changed_expected = expected.copy()
            changed_expected[i, j] = replacement
            try:
                if not np.array_equal(main(changed_grid), changed_expected):
                    return False
            except (AssertionError, IndexError, ValueError):
                return False

            local_r, local_c = i, (j + 1) % n
            distractor_r = i * (n + 1) + local_r
            distractor_c = j * (n + 1) + local_c
            distractor_old = int(distractor_grid[distractor_r, distractor_c])
            distractor_grid[distractor_r, distractor_c] = next(
                color for color in replacement_choices if color != distractor_old
            )
    try:
        return bool(np.array_equal(main(distractor_grid), expected))
    except (AssertionError, IndexError, ValueError):
        return False


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate address-as-coordinate sampling grid."""
    for _ in range(_GENERATION_ATTEMPTS):
        n = random.choice(_WINDOW_COUNTS)
        palette = random.sample(_WINDOW_COLORS, random.randint(5, 7))
        try:
            expected = _sample_target(n, palette)
            windows = [
                [
                    _make_window(n, (i, j), int(expected[i, j]), palette)
                    for j in range(n)
                ]
                for i in range(n)
            ]
        except (AssertionError, ValueError):
            continue

        grid = _render_windows(windows)
        if _semantics_are_explicit(grid, windows, expected, palette):
            return grid
    raise RuntimeError("failed to generate a valid address-as-coordinate input")
