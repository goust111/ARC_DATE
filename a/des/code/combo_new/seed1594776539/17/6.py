from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
ROW_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

ROW_COUNT_MIN = 2
ROW_COUNT_MAX = 6
BEAT_COUNT_MIN = 2
BEAT_COUNT_MAX = 6
GAP_MIN = 1
GAP_MAX = 4
LEFT_MARGIN_MAX = 3
RIGHT_MARGIN_MAX = 4
BOTTOM_PADDING_MAX = 3
ARC_MAX_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and bool(np.all(grid[r] == grid[r, 0]))
    ]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    row = int(rows[0])
    return row, int(grid[row, 0])


def _row_pattern(row: np.ndarray, bg: int, sep_color: int) -> tuple[int, np.ndarray, int]:
    cols = np.flatnonzero(row != bg)
    assert cols.size >= 2, "each rhythm row must contain at least two beats"
    colors = np.unique(row[cols])
    assert len(colors) == 1, "each rhythm row must be monochrome"
    color = int(colors[0])
    assert color != sep_color, "separator color must be unique"

    gaps = np.diff(cols) - 1
    assert bool(np.all(gaps >= 1)), "beats must be separated by visible black gaps"
    assert bool(np.all(row[np.setdiff1d(np.arange(row.size), cols)] == bg)), "only beats may be non-background"
    return color, cols.astype(int), int(gaps.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Generate vertical echoes below each rhythm row, with echo length equal to that row's total black-gap count.

    Input structure:
    - The background is black.
    - The upper band contains several left-aligned horizontal rhythm rows; each such row
      consists of same-color single pixels separated by black gaps.
    - A single full-width grey row separates the rhythm rows from a blank lower area.

    Rule:
    - Keep the rhythm rows and the grey separator unchanged.
    - For each upper-row beat, note its column and compute the total number of black cells
      across all gaps between consecutive beats in that row.
    - Starting from the first row below the grey separator, draw a vertical segment in each
      beat column using that row's color; every segment for that row has the same length,
      namely the row's total gap count.

    Key insight:
    - The horizontal beat positions determine where echoes appear, while the summed black
      intervals between beats determine how far each echo extends downward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    sep_row, sep_color = _separator_row(g, bg)
    assert sep_row >= 1, "expected at least one rhythm row"
    assert sep_row + 1 < g.shape[0], "expected blank space below the separator"
    assert bool(np.all(g[sep_row + 1 :] == bg)), "lower band must start blank"

    out = g.copy()
    seen_colors: set[int] = set()
    for r in range(sep_row):
        color, cols, echo_len = _row_pattern(g[r], bg, sep_color)
        assert color not in seen_colors, "rhythm-row colors must be distinct"
        seen_colors.add(color)
        assert sep_row + echo_len < g.shape[0], "echo must fit below the separator"
        out[sep_row + 1 : sep_row + 1 + echo_len, cols] = color

    return out


def _positions_from_gaps(start_col: int, gaps: np.ndarray) -> tuple[int, ...]:
    positions = [int(start_col)]
    cur = int(start_col)
    for gap in gaps:
        cur += int(gap) + 1
        positions.append(cur)
    return tuple(positions)


def _render_grid(
    row_specs: list[dict[str, object]],
    *,
    width: int,
    height: int,
    separator_row: int,
    with_echoes: bool,
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for r, spec in enumerate(row_specs):
        color = int(spec["color"])
        cols = np.array(spec["cols"], dtype=int)
        grid[r, cols] = color
        if with_echoes:
            echo_len = int(spec["echo_len"])
            grid[separator_row + 1 : separator_row + 1 + echo_len, cols] = color
    grid[separator_row, :] = GREY
    return grid


def _background_is_unique_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    if values.size < 2:
        return False
    ordered = np.sort(counts)
    return int(ordered[-1]) > int(ordered[-2])


def _gen_attempt() -> np.ndarray | None:
    row_count = int(_RNG.integers(ROW_COUNT_MIN, ROW_COUNT_MAX + 1))
    start_col = int(_RNG.integers(0, LEFT_MARGIN_MAX + 1))
    right_pad = int(_RNG.integers(0, RIGHT_MARGIN_MAX + 1))
    bottom_pad = int(_RNG.integers(0, BOTTOM_PADDING_MAX + 1))
    colors = _RNG.choice(np.array(ROW_COLOR_POOL, dtype=int), size=row_count, replace=False)

    row_specs: list[dict[str, object]] = []
    beat_counts: list[int] = []
    echo_lengths: list[int] = []
    max_col = 0
    for color in colors:
        beat_count = int(_RNG.integers(BEAT_COUNT_MIN, BEAT_COUNT_MAX + 1))
        gaps = _RNG.integers(GAP_MIN, GAP_MAX + 1, size=beat_count - 1)
        cols = _positions_from_gaps(start_col, gaps)
        echo_len = int(np.sum(gaps))
        beat_counts.append(beat_count)
        echo_lengths.append(echo_len)
        max_col = max(max_col, cols[-1])
        row_specs.append({"color": int(color), "cols": cols, "echo_len": echo_len})

    if max(beat_counts) < 3:
        return None
    if len(set(echo_lengths)) < 2 or max(echo_lengths) < 2:
        return None
    if len({tuple(spec["cols"]) for spec in row_specs}) < 2:
        return None

    separator_row = row_count
    width = max_col + 1 + right_pad
    height = separator_row + 1 + max(echo_lengths) + bottom_pad
    if max(width, height) > ARC_MAX_SIZE:
        return None

    inp = _render_grid(
        row_specs,
        width=width,
        height=height,
        separator_row=separator_row,
        with_echoes=False,
    )
    expected = _render_grid(
        row_specs,
        width=width,
        height=height,
        separator_row=separator_row,
        with_echoes=True,
    )

    if not _background_is_unique_dominant(inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rhythm-echo grid")
