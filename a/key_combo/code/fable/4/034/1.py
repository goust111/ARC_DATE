"""ARC task: grey pixels fall toward a bar and take cyclic depth colors."""

from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
GREY = 5
ARC_COLORS = tuple(color for color in range(1, 10) if color != GREY)
MAX_GENERATION_ATTEMPTS = 200


def _read_structure(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    """Return (bar row, bar color, palette from the bar outwards)."""
    g = np.asarray(grid)
    full_rows = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != BACKGROUND and np.all(g[r] == g[r, 0])
    ]
    assert len(full_rows) == 1, "expected exactly one non-black full-width bar"
    bar_row = full_rows[0]
    bar_color = int(g[bar_row, 0])

    palette: list[int] = []
    r = bar_row - 1
    while r >= 0 and int(g[r, 0]) not in (BACKGROUND, GREY):
        palette.append(int(g[r, 0]))
        r -= 1

    assert palette, "expected a palette column immediately above the bar"
    return bar_row, bar_color, tuple(palette)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide grey pixels toward the full-width bar, then recolor them by cyclic stack depth.

    Input structure:
    - A black grid contains one monochrome full-width horizontal bar, a contiguous
      palette in the leftmost column immediately above it, and isolated grey pixels
      scattered on both sides of the bar.

    Rule:
    - In each column, upper grey pixels fall downward and lower grey pixels rise
      upward until they form stacks against the bar; nearer pixels settle first.
    - Recolor stack layer k with palette color (k - 1) modulo the palette length,
      counting independently away from the bar on either side. Preserve the bar
      and palette.

    Key insight:
    - A grey pixel's color is determined by its new collision-created layer depth,
      not by its original distance; the attached palette is a cyclic depth ruler.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"
    assert g.size > 0, "expected a non-empty grid"

    bar_row, _bar_color, palette = _read_structure(g)
    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.size, "expected grey pixels"
    assert not np.any(grey_coords[:, 0] == bar_row), "grey pixels cannot occupy the bar"
    assert not np.any(grey_coords[:, 1] == 0), "grey pixels cannot overlap the palette column"

    out = g.copy()
    out[g == GREY] = BACKGROUND

    for c in range(g.shape[1]):
        upper_count = int(np.count_nonzero(g[:bar_row, c] == GREY))
        lower_count = int(np.count_nonzero(g[bar_row + 1 :, c] == GREY))
        assert upper_count <= bar_row, "upper stack does not fit"
        assert lower_count < g.shape[0] - bar_row, "lower stack does not fit"

        for depth in range(1, upper_count + 1):
            dest = (bar_row - depth, c)
            assert int(out[dest]) == BACKGROUND, "upper stack collides with a fixed object"
            out[dest] = palette[(depth - 1) % len(palette)]

        for depth in range(1, lower_count + 1):
            dest = (bar_row + depth, c)
            assert int(out[dest]) == BACKGROUND, "lower stack collides with a fixed object"
            out[dest] = palette[(depth - 1) % len(palette)]

    return out


def _spaced_columns(width: int, count: int) -> tuple[int, ...]:
    """Choose columns away from the palette, with a black separator between them."""
    choices = [
        cols
        for cols in itertools.combinations(range(2, width), count)
        if all(b - a >= 2 for a, b in zip(cols, cols[1:]))
    ]
    assert choices
    return random.choice(choices)


def _spaced_distances(max_distance: int, count: int) -> tuple[int, ...]:
    """Choose nonadjacent initial distances, never initially touching the bar."""
    choices = [
        ds
        for ds in itertools.combinations(range(2, max_distance + 1), count)
        if all(b - a >= 2 for a, b in zip(ds, ds[1:]))
    ]
    assert choices
    return random.choice(choices)


def _independent_expected(
    grid: np.ndarray, bar_row: int, palette: tuple[int, ...]
) -> np.ndarray:
    """Small declarative oracle used only to check generated examples."""
    expected = grid.copy()
    expected[expected == GREY] = BACKGROUND
    for c in range(grid.shape[1]):
        above = int(np.sum(grid[:bar_row, c] == GREY))
        below = int(np.sum(grid[bar_row + 1 :, c] == GREY))
        for side, count in ((-1, above), (1, below)):
            for layer in range(1, count + 1):
                expected[bar_row + side * layer, c] = palette[(layer - 1) % len(palette)]
    return expected


def _semantically_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    bar_row: int,
    palette: tuple[int, ...],
    sources: dict[tuple[int, int], tuple[int, ...]],
) -> bool:
    """Reject ambiguous examples and examples admitting the intended shortcuts."""
    try:
        found_row, _bar_color, found_palette = _read_structure(grid)
    except AssertionError:
        return False
    if found_row != bar_row or found_palette != palette:
        return False

    grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    if not grey or any(c == 0 or r == bar_row for r, c in grey):
        return False
    # The sources are genuinely isolated pixels rather than pre-made bars/stacks.
    if any((r + dr, c + dc) in grey for r, c in grey for dr, dc in ((1, 0), (0, 1))):
        return False

    above = [(key, ds) for key, ds in sources.items() if key[0] == -1]
    below = [(key, ds) for key, ds in sources.items() if key[0] == 1]
    if not above or not below:
        return False
    counts = [len(ds) for _key, ds in above + below]
    if max(counts) <= len(palette):
        return False  # the cyclic branch must be visible
    if not any(count >= 2 for count in counts):
        return False  # collision-created depth must matter

    # At least one ordered source would get a different color under the tempting
    # but wrong rule "use original distance modulo palette length".
    defeats_initial_distance = any(
        any((distance - layer) % len(palette) != 0 for layer, distance in enumerate(ds, 1))
        for ds in sources.values()
    )
    if not defeats_initial_distance:
        return False

    if not any(int(expected[r, c]) == BACKGROUND for r, c in grey) or np.array_equal(grid, expected):
        return False  # some old locations must visibly clear

    # A full cycle is exercised, so every visible palette entry controls output.
    moving_colors = {
        int(expected[bar_row + side * layer, c])
        for (side, c), distances in sources.items()
        for layer in range(1, len(distances) + 1)
    }
    if not set(palette).issubset(moving_colors):
        return False
    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(11, 24)
    bar_row = random.randint(10, height - 11)
    palette_len = random.randint(2, 4)

    chosen_colors = random.sample(ARC_COLORS, palette_len + 1)
    bar_color = chosen_colors[0]
    palette = tuple(chosen_colors[1:])

    grid = np.zeros((height, width), dtype=int)
    grid[bar_row, :] = bar_color
    for depth, color in enumerate(palette, 1):
        grid[bar_row - depth, 0] = color

    upper_ncols = random.randint(2, min(4, (width - 1) // 2))
    lower_ncols = random.randint(2, min(4, (width - 1) // 2))
    upper_cols = _spaced_columns(width, upper_ncols)
    lower_cols = _spaced_columns(width, lower_ncols)

    deep_side = random.choice((-1, 1))
    sources: dict[tuple[int, int], tuple[int, ...]] = {}
    for side, columns in ((-1, upper_cols), (1, lower_cols)):
        forced_col = random.choice(columns)
        for c in columns:
            if side == deep_side and c == forced_col:
                count = palette_len + 1
            elif c == forced_col:
                count = random.randint(2, min(3, palette_len + 1))
            else:
                count = random.randint(1, min(3, palette_len + 1))

            max_distance = bar_row if side == -1 else height - 1 - bar_row
            distances = _spaced_distances(max_distance, count)
            sources[(side, c)] = distances
            for distance in distances:
                grid[bar_row + side * distance, c] = GREY

    expected = _independent_expected(grid, bar_row, palette)
    if _semantically_valid(grid, expected, bar_row, palette, sources):
        return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate instance that exposes every rule branch."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid falling-depth instance")
