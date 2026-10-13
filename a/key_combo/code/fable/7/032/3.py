from __future__ import annotations

from fractions import Fraction
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    """Return half-open bands between one-cell separator lines."""
    starts = [0] + [position + 1 for position in separators]
    ends = separators + [size]
    bands = list(zip(starts, ends))
    if len(bands) != 3 or any(start >= end for start, end in bands):
        raise ValueError("expected two separators making three nonempty bands")
    return bands


def _paths() -> list[tuple[tuple[int, int], ...]]:
    """The four undirected 3x3 row/column serpentine path classes."""
    row_left = tuple(
        (r, c if r % 2 == 0 else 2 - c)
        for r in range(3)
        for c in range(3)
    )
    row_right = tuple((r, 2 - c) for r, c in row_left)
    col_top = tuple(
        (r if c % 2 == 0 else 2 - r, c)
        for c in range(3)
        for r in range(3)
    )
    col_bottom = tuple((2 - r, c) for r, c in col_top)
    return [row_left, row_right, col_top, col_bottom]


PATHS = _paths()


def _read_tiles(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int], list[int]]:
    """Parse separator bands and the nine centered, bottom-aligned bars."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a two-dimensional grid")

    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    if len(row_seps) != 2 or len(col_seps) != 2:
        raise ValueError("expected exactly two full grey separator rows and columns")

    row_bands = _bands(row_seps, int(g.shape[0]))
    col_bands = _bands(col_seps, int(g.shape[1]))
    heights: list[int] = []
    colors: list[int] = []
    bar_columns: list[int] = []

    for macro_r, (r0, r1) in enumerate(row_bands):
        for macro_c, (c0, c1) in enumerate(col_bands):
            tile = g[r0:r1, c0:c1]
            foreground = np.argwhere(tile != BACKGROUND)
            if foreground.size == 0:
                raise ValueError("every tile must contain one nonempty bar")
            if np.any(tile == SEPARATOR):
                raise ValueError("grey is reserved for separator lines")

            used_cols = np.unique(foreground[:, 1])
            if len(used_cols) != 1:
                raise ValueError("each tile must contain one one-cell-wide bar")
            local_c = int(used_cols[0])
            expected_c = (c1 - c0) // 2
            if local_c != expected_c:
                raise ValueError("bars must be horizontally centered in their tiles")

            rows = foreground[:, 0]
            height = int(len(rows))
            if not np.array_equal(rows, np.arange((r1 - r0) - height, r1 - r0)):
                raise ValueError("bars must be solid and bottom-aligned")
            values = np.unique(tile[rows, local_c])
            if len(values) != 1 or int(values[0]) in (BACKGROUND, SEPARATOR):
                raise ValueError("each bar must be monochromatic")

            heights.append(height)
            colors.append(int(values[0]))
            bar_columns.append(c0 + local_c)

    return row_bands, col_bands, heights, colors, bar_columns


def _quadratic_from_points(
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int] | None:
    """Recover integer (a, d, q) for a+i*d+i*(i-1)/2*q."""
    (i0, y0), (i1, y1), (i2, y2) = points
    slope01 = Fraction(y1 - y0, i1 - i0)
    slope12 = Fraction(y2 - y1, i2 - i1)
    c = (slope12 - slope01) / (i2 - i0)
    b = slope01 - c * (i0 + i1)
    a = Fraction(y0) - b * i0 - c * i0 * i0
    q = 2 * c
    d = b + c
    if any(value.denominator != 1 for value in (a, d, q)):
        return None
    result = (int(a), int(d), int(q))
    return result if result[2] != 0 else None


def _sequence(params: tuple[int, int, int]) -> tuple[int, ...]:
    a, d, q = params
    return tuple(a + i * d + i * (i - 1) // 2 * q for i in range(9))


def _find_solutions(
    heights: list[int], row_bands: list[tuple[int, int]]
) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """Find (path index, bad sequence index, expected height, sequence)."""
    solutions: set[tuple[int, int, int, tuple[int, ...]]] = set()
    tile_caps = [row_bands[r][1] - row_bands[r][0] for r in range(3) for _ in range(3)]

    for path_index, path in enumerate(PATHS):
        observed = [heights[r * 3 + c] for r, c in path]
        caps = [tile_caps[r * 3 + c] for r, c in path]
        for omitted in range(9):
            kept = [i for i in range(9) if i != omitted]
            # Any three correct terms determine a quadratic uniquely.  Under
            # this omitted-position hypothesis the first three kept terms are
            # known-correct, so trying further triples would be redundant.
            indices = kept[:3]
            params = _quadratic_from_points(
                tuple((i, observed[i]) for i in indices)  # type: ignore[arg-type]
            )
            if params is None:
                continue
            sequence = _sequence(params)
            mismatches = [i for i in range(9) if observed[i] != sequence[i]]
            if mismatches != [omitted]:
                continue
            if any(value < 1 or value > cap for value, cap in zip(sequence, caps)):
                continue
            solutions.add((path_index, omitted, sequence[omitted], sequence))

    return sorted(solutions)


def main(grid: np.ndarray) -> np.ndarray:
    """Correct the one bar that breaks a hidden serpentine second-order progression.

    Input structure:
    - Two full grey rows and columns divide a black canvas into a 3x3 tile grid.
      Tiles in one macro row have equal height, while different macro rows may
      have different heights (the minimal 30x30-feasible version of the puzzle).
    - Every tile contains one centered, bottom-aligned, monochrome vertical bar;
      bar colors carry no numerical meaning.

    Rule:
    - Test the four row-first or column-first serpentine paths, identifying a
      path with its reverse.  Along exactly one path, eight heights agree with
      an integer sequence whose first differences change by one nonzero constant.
    - Replace the sole mismatching bar by the predicted height, retaining its
      color, center column, bottom alignment, separators, and all other pixels.

    Key insight:
    - Both the boustrophedon reading order and the quadratic height law are
      hidden; only their joint interpretation isolates a single anomalous tile.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, _col_bands, heights, colors, bar_columns = _read_tiles(g)
    solutions = _find_solutions(heights, row_bands)
    if len(solutions) != 1:
        raise ValueError(f"expected one hidden path/progression solution, found {len(solutions)}")

    path_index, bad_index, expected_height, _sequence_values = solutions[0]
    macro_r, macro_c = PATHS[path_index][bad_index]
    flat_index = macro_r * 3 + macro_c
    r0, r1 = row_bands[macro_r]
    column = bar_columns[flat_index]

    out = g.copy()
    out[r0:r1, column] = BACKGROUND
    out[r1 - expected_height:r1, column] = colors[flat_index]
    return out


def _render(
    row_heights: list[int], tile_width: int, path_index: int,
    sequence: tuple[int, ...], bad_index: int, bad_height: int,
) -> np.ndarray:
    height = sum(row_heights) + 2
    width = 3 * tile_width + 2
    grid = np.full((height, width), BACKGROUND, dtype=int)

    row_seps = [row_heights[0], row_heights[0] + 1 + row_heights[1]]
    col_seps = [tile_width, 2 * tile_width + 1]
    grid[row_seps, :] = SEPARATOR
    grid[:, col_seps] = SEPARATOR
    row_bands = _bands(row_seps, height)
    col_bands = _bands(col_seps, width)

    tile_values = [0] * 9
    for i, (macro_r, macro_c) in enumerate(PATHS[path_index]):
        tile_values[macro_r * 3 + macro_c] = bad_height if i == bad_index else sequence[i]

    palette = list(BAR_COLORS)
    random.shuffle(palette)
    colors = [random.choice(palette) for _ in range(9)]
    for macro_r, (r0, r1) in enumerate(row_bands):
        for macro_c, (c0, c1) in enumerate(col_bands):
            flat_index = macro_r * 3 + macro_c
            bar_height = tile_values[flat_index]
            column = c0 + (c1 - c0) // 2
            grid[r1 - bar_height:r1, column] = colors[flat_index]
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance of the feasible puzzle."""
    for _ in range(1000):
        path_index = random.randrange(len(PATHS))
        d = random.choice((-4, -3))
        base_a = 11 if d == -4 else 7
        base_sequence = _sequence((base_a, d, 1))

        row_needs = [1, 1, 1]
        for value, (macro_r, _macro_c) in zip(base_sequence, PATHS[path_index]):
            row_needs[macro_r] = max(row_needs[macro_r], value)
        max_shift = (28 - sum(row_needs)) // 3
        if max_shift < 0:
            continue
        shift = random.randint(0, min(2, max_shift))
        sequence = tuple(value + shift for value in base_sequence)
        row_needs = [value + shift for value in row_needs]

        spare = 28 - sum(row_needs)
        extra_total = random.randint(0, spare)
        extras = [0, 0, 0]
        for _ in range(extra_total):
            extras[random.randrange(3)] += 1
        row_heights = [need + extra for need, extra in zip(row_needs, extras)]
        tile_width = random.randint(3, 9)

        bad_index = random.randrange(9)
        bad_macro_r, _bad_macro_c = PATHS[path_index][bad_index]
        choices = [
            value for value in range(1, row_heights[bad_macro_r] + 1)
            if value != sequence[bad_index]
        ]
        if not choices:
            continue
        bad_height = random.choice(choices)
        grid = _render(
            row_heights, tile_width, path_index, sequence, bad_index, bad_height
        )

        try:
            parsed_rows, _cols, observed, _colors, _bar_cols = _read_tiles(grid)
            solutions = _find_solutions(observed, parsed_rows)
            if len(solutions) != 1:
                continue
            solution = solutions[0]
            if solution[0] != path_index or solution[1] != bad_index:
                continue
            # The sole accepted candidate predicts a different height because
            # bad_height was sampled outside the expected value.
        except (ValueError, IndexError):
            continue
        return grid

    raise RuntimeError("failed to generate an unambiguous puzzle instance")
