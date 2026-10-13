import itertools
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (7, 15)
GRID_WIDTH_RANGE = (8, 18)
HORIZONTAL_COUNT_RANGE = (2, 4)
VERTICAL_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300


def _detect_horizontal_rows(g: np.ndarray) -> tuple[list[tuple[int, int]], set[int]]:
    height, width = g.shape
    assert width >= 2, "expected at least two columns"

    horizontals: list[tuple[int, int]] = []
    sinking_rows: set[int] = set()
    for r in range(height):
        body = g[r, 1:]
        values = np.unique(body)
        if len(values) != 1:
            continue
        color = int(values[0])
        if color == BG:
            continue
        horizontals.append((r, color))
        if int(g[r, 0]) != color:
            sinking_rows.add(r)
            assert int(g[r, 0]) != BG, "markers must be non-background"

    assert horizontals, "expected at least one horizontal stripe"
    colors = [color for _row, color in horizontals]
    assert len(set(colors)) == len(colors), "horizontal stripe colors must be unique"
    return horizontals, sinking_rows


def _detect_vertical_columns(g: np.ndarray, horizontals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    height, width = g.shape
    exposed_mask = np.ones(height, dtype=bool)
    for row, _color in horizontals:
        exposed_mask[row] = False
    exposed_rows = np.flatnonzero(exposed_mask)
    assert len(exposed_rows) > 0, "expected at least one row without a horizontal stripe"

    verticals: list[tuple[int, int]] = []
    for c in range(width):
        values = np.unique(g[exposed_rows, c])
        if len(values) == 1 and int(values[0]) == BG:
            continue
        assert len(values) == 1, "vertical stripes must be uniform on exposed rows"
        color = int(values[0])
        assert color != BG, "vertical stripe color must be non-background"
        verticals.append((c, color))

    assert verticals, "expected at least one vertical stripe"
    colors = [color for _col, color in verticals]
    assert len(set(colors)) == len(colors), "vertical stripe colors must be unique"
    return verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reweave horizontal and vertical full-length stripes, using left-edge markers to decide which crossings sink below.

    Input structure:
    - The background is black.
    - Some rows are full-width 1-pixel horizontal stripes, each with its own color.
    - Some columns are full-height 1-pixel vertical stripes, with colors distinct from the horizontal stripes.
    - In the input, every stripe crossing currently shows the horizontal stripe color.
    - Some horizontal rows have a non-stripe marker pixel in the leftmost column; these are the sinking rows.

    Rule:
    - Keep all non-crossing stripe pixels unchanged.
    - For an unmarked horizontal row, every crossing stays the horizontal stripe color.
    - For a marked horizontal row, every crossing on that row is repainted to the corresponding vertical stripe color.
    - Remove the left-edge markers by restoring the stripe pattern underneath.

    Key insight:
    - Horizontal stripes appear as nearly solid colored rows, while the vertical stripe colors are revealed on rows without horizontals; the left-edge markers only flip the over-under order on their own rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    horizontals, sinking_rows = _detect_horizontal_rows(g)
    verticals = _detect_vertical_columns(g, horizontals)

    horizontal_colors = {color for _row, color in horizontals}
    vertical_colors = {color for _col, color in verticals}
    assert horizontal_colors.isdisjoint(vertical_colors), "stripe directions must use disjoint colors"

    out = np.full(g.shape, BG, dtype=int)
    for col, color in verticals:
        out[:, col] = color

    for row, color in horizontals:
        out[row, :] = color

    for row in sinking_rows:
        for col, color in verticals:
            out[row, col] = color

    return out


def _spaced_sample(start: int, stop: int, count: int, *, min_gap: int) -> list[int]:
    values = list(range(start, stop))
    options = [
        combo
        for combo in itertools.combinations(values, count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    if not options:
        options = list(itertools.combinations(values, count))
    return list(random.choice(options))


def _render_scene(
    height: int,
    width: int,
    row_to_color: dict[int, int],
    col_to_color: dict[int, int],
    sinking_rows: set[int],
    marker_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)

    for col, color in col_to_color.items():
        inp[:, col] = color
        out[:, col] = color

    for row, color in row_to_color.items():
        inp[row, :] = color
        out[row, :] = color

    for row in sinking_rows:
        inp[row, 0] = marker_color
        for col, color in col_to_color.items():
            out[row, col] = color

    return inp, out


def _passes_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_h = min(HORIZONTAL_COUNT_RANGE[1], height - 1)
    if max_h < HORIZONTAL_COUNT_RANGE[0]:
        return None
    horizontal_count = random.randint(HORIZONTAL_COUNT_RANGE[0], max_h)

    max_v = min(VERTICAL_COUNT_RANGE[1], width - 2, len(PALETTE) - horizontal_count - 1)
    if max_v < VERTICAL_COUNT_RANGE[0]:
        return None
    vertical_count = random.randint(VERTICAL_COUNT_RANGE[0], max_v)

    horizontal_rows = _spaced_sample(0, height, horizontal_count, min_gap=2)
    vertical_cols = _spaced_sample(1, width, vertical_count, min_gap=2)

    stripe_colors = random.sample(PALETTE, horizontal_count + vertical_count)
    marker_choices = [color for color in PALETTE if color not in stripe_colors]
    marker_color = random.choice(marker_choices)

    row_to_color = dict(zip(horizontal_rows, stripe_colors[:horizontal_count]))
    col_to_color = dict(zip(vertical_cols, stripe_colors[horizontal_count:]))

    sinking_count = random.randint(1, horizontal_count - 1)
    sinking_rows = set(random.sample(horizontal_rows, sinking_count))

    inp, expected = _render_scene(height, width, row_to_color, col_to_color, sinking_rows, marker_color)
    if not _passes_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
