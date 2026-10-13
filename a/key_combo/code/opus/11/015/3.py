from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BG = 0


def _find_separator_row(grid: np.ndarray) -> int:
    for r in range(1, grid.shape[0] - 1):
        if np.all(grid[r] == BG) and np.any(grid[:r] != BG) and np.all(grid[r + 1] != BG):
            return int(r)
    raise AssertionError("expected one all-black separator row above the palette")


def _prefix_length(row: np.ndarray) -> int:
    length = 0
    while length < row.size and int(row[length]) != BG:
        length += 1
    if length < 3:
        raise AssertionError("row prefix must be at least length 3")
    if not np.all(row[length:] == BG):
        raise AssertionError("row must contain a single left non-background prefix")
    return int(length)


def _extend_row(row: np.ndarray) -> tuple[np.ndarray, int]:
    width = int(row.size)
    prefix_len = _prefix_length(row)
    prefix = row[:prefix_len].copy()
    boundary = int(prefix[0])
    if boundary != int(prefix[-1]):
        raise AssertionError("prefix must start and end with the same boundary color")
    suffix_len = prefix_len - 1
    if width < 2 * prefix_len - 1:
        raise AssertionError("row is too narrow for closed-prefix extension")

    out = row.copy()
    out[prefix_len : width - suffix_len] = boundary
    out[width - suffix_len :] = prefix[1:]
    return out, boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each row's left closed prefix, then recolor the resulting boundary pixels column-by-column from the palette strip.

    Input structure:
    - The upper block is the main pattern area on black background.
    - Each main-area row starts with one contiguous non-black prefix; its first and last cells share the same boundary color, while the cells after the prefix are black.
    - One all-black separator row sits directly above a full-width non-black palette row.

    Rule:
    - Split the grid into main area, separator row, and palette row.
    - For each main-area row, keep the left prefix in place, fill the middle gap with the prefix boundary color, and copy the prefix without its first cell onto the right edge.
    - After all rows are extended, recolor every pixel equal to the shared boundary color in column c with the palette color from column c; leave interior non-boundary colors unchanged.

    Key insight:
    The palette does not recolor an existing mask; it only replaces the boundary-color pixels created or preserved by the row-extension step, so the interior motif survives untouched.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    sep = _find_separator_row(g)
    main_area = g[:sep].copy()
    palette = g[sep + 1].copy()
    if main_area.size == 0 or sep + 1 >= g.shape[0]:
        raise AssertionError("missing main area or palette row")
    if main_area.shape[1] != palette.size:
        raise AssertionError("palette width must match the main area width")

    extended = np.zeros_like(main_area)
    boundary_colors = set()
    for r in range(main_area.shape[0]):
        row_out, boundary = _extend_row(main_area[r])
        extended[r] = row_out
        boundary_colors.add(int(boundary))

    if len(boundary_colors) != 1:
        raise AssertionError("all rows must share one boundary color")
    boundary_color = next(iter(boundary_colors))

    for c, color in enumerate(palette):
        mask = extended[:, c] == boundary_color
        extended[mask, c] = int(color)
    return extended


_WIDTH_RANGE = (8, 12)
_HEIGHT_RANGE = (5, 9)
_PREFIX_MIN = 3
_PREFIX_MAX = 5
_BOUNDARY_COLOR = 2
_INTERIOR_POOL = (1, 3, 4)
_PALETTE_POOL = (5, 6, 7, 8, 9)
_RUN_COUNT_OPTIONS = (3, 4, 5)
_RUN_LENGTH_RANGE = (1, 3)
_GEN_TRIES = 300


def _valid_run_lengths(width: int, n_runs: int) -> list[tuple[int, ...]]:
    lo, hi = _RUN_LENGTH_RANGE
    return [
        tuple(int(x) for x in lengths)
        for lengths in itertools.product(range(lo, hi + 1), repeat=int(n_runs))
        if sum(lengths) == int(width)
    ]


def _sample_run_colors(
    rng: np.random.Generator, colors: Iterable[int], n_runs: int
) -> list[int]:
    palette = [int(c) for c in colors]
    distinct = int(rng.integers(3, min(len(palette), n_runs) + 1))
    chosen = [int(c) for c in rng.choice(palette, size=distinct, replace=False)]
    order: list[int] = []
    unused = set(chosen)
    for i in range(n_runs):
        remaining_slots = n_runs - i
        candidates = [c for c in chosen if not order or c != order[-1]]
        must_cover = [c for c in candidates if c in unused]
        if len(unused) >= remaining_slots and must_cover:
            candidates = must_cover
        elif must_cover and rng.random() < 0.7:
            candidates = must_cover
        color = int(rng.choice(candidates))
        order.append(color)
        unused.discard(color)
    return order


def _generate_palette_row(rng: np.random.Generator, width: int) -> np.ndarray:
    valid_counts = [n for n in _RUN_COUNT_OPTIONS if _valid_run_lengths(width, n)]
    n_runs = int(rng.choice(valid_counts))
    lengths = _valid_run_lengths(width, n_runs)
    run_lengths = lengths[int(rng.integers(len(lengths)))]
    run_colors = _sample_run_colors(rng, _PALETTE_POOL, n_runs)
    return np.repeat(np.asarray(run_colors, dtype=int), np.asarray(run_lengths, dtype=int))


def _generate_prefix(
    rng: np.random.Generator, prefix_len: int, interior_colors: tuple[int, ...]
) -> np.ndarray:
    interiors = rng.choice(np.asarray(interior_colors, dtype=int), size=prefix_len - 2, replace=True)
    prefix = np.empty(prefix_len, dtype=int)
    prefix[0] = _BOUNDARY_COLOR
    prefix[-1] = _BOUNDARY_COLOR
    prefix[1:-1] = interiors
    return prefix


def _generate_rows(
    rng: np.random.Generator, height: int, width: int
) -> np.ndarray:
    max_prefix = min(_PREFIX_MAX, (width + 1) // 2)
    prefix_choices = list(range(_PREFIX_MIN, max_prefix + 1))
    if not prefix_choices:
        raise AssertionError("width too small for any valid prefix")

    interior_count = int(rng.integers(1, len(_INTERIOR_POOL) + 1))
    interior_colors = tuple(int(c) for c in rng.choice(_INTERIOR_POOL, size=interior_count, replace=False))

    rows: list[np.ndarray] = []
    anchor_lengths = prefix_choices.copy()
    rng.shuffle(anchor_lengths)
    while len(rows) < min(height, len(anchor_lengths)):
        rows.append(_generate_prefix(rng, anchor_lengths[len(rows)], interior_colors))
    while len(rows) < height:
        rows.append(_generate_prefix(rng, int(rng.choice(prefix_choices)), interior_colors))

    if len({tuple(map(int, row)) for row in rows}) < 2:
        rows[-1] = _generate_prefix(rng, prefix_choices[-1], interior_colors)
        if len({tuple(map(int, row)) for row in rows}) < 2 and len(prefix_choices) > 1:
            rows[0] = _generate_prefix(rng, prefix_choices[0], interior_colors)

    grid = np.zeros((height, width), dtype=int)
    for r, prefix in enumerate(rows):
        grid[r, : prefix.size] = prefix
    return grid


def _is_interesting(inp: np.ndarray, out: np.ndarray) -> bool:
    sep = _find_separator_row(inp)
    main_area = inp[:sep]
    palette = inp[sep + 1]
    if np.array_equal(main_area, out):
        return False
    if len(set(map(int, palette))) < 3:
        return False
    if len({tuple(map(int, row)) for row in main_area}) < 2:
        return False
    if not np.any((main_area != BG) & (main_area != _BOUNDARY_COLOR)):
        return False
    if len({_prefix_length(row) for row in main_area}) < 2 and main_area.shape[1] >= 9:
        return False
    return True


def _generate_one(rng: np.random.Generator) -> np.ndarray:
    width = int(rng.integers(_WIDTH_RANGE[0], _WIDTH_RANGE[1] + 1))
    height = int(rng.integers(_HEIGHT_RANGE[0], _HEIGHT_RANGE[1] + 1))
    main_area = _generate_rows(rng, height, width)
    palette = _generate_palette_row(rng, width)
    inp = np.vstack([main_area, np.zeros((1, width), dtype=int), palette[np.newaxis, :]])
    out = main(inp)
    if not _is_interesting(inp, out):
        raise AssertionError("degenerate sample")
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_TRIES):
        try:
            return _generate_one(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
