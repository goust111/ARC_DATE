"""ARC task: use an edge palette bar, an anchor, and an oriented scan order."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


Coord = tuple[int, int]


@dataclass(frozen=True)
class _Bar:
    orientation: str
    coords: tuple[Coord, ...]
    colors: tuple[int, ...]


def _nonzero_runs(coords: list[Coord], grid: np.ndarray) -> list[tuple[Coord, ...]]:
    """Return maximal non-black runs along one already ordered edge."""
    runs: list[tuple[Coord, ...]] = []
    start = 0
    while start < len(coords):
        if int(grid[coords[start]]) == 0:
            start += 1
            continue
        stop = start + 1
        while stop < len(coords) and int(grid[coords[stop]]) != 0:
            stop += 1
        runs.append(tuple(coords[start:stop]))
        start = stop
    return runs


def _find_bar(grid: np.ndarray) -> _Bar:
    """Find the unique length-at-least-three colored run on a grid edge."""
    height, width = map(int, grid.shape)
    edges = (
        ("horizontal", [(0, c) for c in range(width)]),
        ("horizontal", [(height - 1, c) for c in range(width)]),
        ("vertical", [(r, 0) for r in range(height)]),
        ("vertical", [(r, width - 1) for r in range(height)]),
    )
    candidates: list[tuple[str, tuple[Coord, ...]]] = []
    for orientation, edge in edges:
        candidates.extend(
            (orientation, run) for run in _nonzero_runs(edge, grid) if len(run) >= 3
        )

    assert len(candidates) == 1, "expected one unique colored edge bar"
    orientation, coords = candidates[0]
    colors = tuple(int(grid[coord]) for coord in coords)
    return _Bar(orientation, coords, colors)


def _ordered_scatter(coords: list[Coord], orientation: str) -> list[Coord]:
    if orientation == "horizontal":
        return sorted(coords)  # (row, column): row-major order
    assert orientation == "vertical"
    return sorted(coords, key=lambda point: (point[1], point[0]))


def _phase_for_anchor(
    ordered: list[Coord], sequence: tuple[int, ...], anchor: Coord, anchor_color: int
) -> int:
    anchor_rank = ordered.index(anchor)
    matching_indices = [i for i, color in enumerate(sequence) if color == anchor_color]
    assert len(matching_indices) == 1, "anchor color must occur exactly once in the bar"
    return (matching_indices[0] - anchor_rank) % len(sequence)


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the edge palette bar and cyclically recolor isolated pixels in its oriented scan order.

    Input structure:
    - A black-background grid contains one colored bar of length at least three,
      flush with one edge, plus separated singleton pixels away from the bar.
    - Exactly one singleton has a color from the bar, and that anchor color occurs
      exactly once in the bar; all other singleton colors are absent from the bar.

    Rule:
    - Read a horizontal bar left-to-right and scan singletons in row-major order;
      read a vertical bar top-to-bottom and scan them in column-major order.
    - Cyclically assign the bar sequence to that ordering, choosing the unique
      phase that assigns the anchor its original color. Erase the entire bar to
      black and recolor every singleton with its assigned color.

    Key insight:
    - The bar simultaneously supplies the scan compass and ordered palette, while
      the sole shared-color singleton fixes a non-default cyclic phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a two-dimensional grid"

    bar = _find_bar(g)
    bar_coords = set(bar.coords)
    scatter = [
        (int(r), int(c))
        for r, c in np.argwhere(g != 0)
        if (int(r), int(c)) not in bar_coords
    ]
    assert len(scatter) >= 2, "expected multiple isolated pixels"

    # The singleton objects must remain spatially distinct from one another and
    # from the bar. Chebyshev distance > 1 makes even diagonal touching invalid.
    for i, first in enumerate(scatter):
        assert all(
            max(abs(first[0] - second[0]), abs(first[1] - second[1])) > 1
            for second in scatter[i + 1 :]
        ), "scattered pixels must be isolated"
        assert all(
            max(abs(first[0] - cell[0]), abs(first[1] - cell[1])) > 1
            for cell in bar.coords
        ), "scattered pixels must be separated from the bar"

    palette = set(bar.colors)
    anchors = [coord for coord in scatter if int(g[coord]) in palette]
    assert len(anchors) == 1, "expected exactly one shared-color anchor"
    anchor = anchors[0]
    anchor_color = int(g[anchor])
    assert bar.colors.count(anchor_color) == 1
    assert all(int(g[coord]) not in palette for coord in scatter if coord != anchor)

    ordered = _ordered_scatter(scatter, bar.orientation)
    phase = _phase_for_anchor(ordered, bar.colors, anchor, anchor_color)

    out = g.copy()
    for coord in bar.coords:
        out[coord] = 0
    for rank, coord in enumerate(ordered):
        out[coord] = bar.colors[(rank + phase) % len(bar.colors)]
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_RNG = random.Random()
_COLORS = tuple(range(1, 10))
_MAX_ATTEMPTS = 400


def _make_palette(length: int) -> tuple[tuple[int, ...], int]:
    """Make an aperiodic sequence; longer bars always show visible repetition."""
    anchor_color = _RNG.choice(_COLORS)
    available = [color for color in _COLORS if color != anchor_color]
    # With a three-cell bar, [anchor, x, x] is equivalent under reversal after
    # anchor alignment. Use two distinct non-anchor colors at this boundary so
    # left-to-right/top-to-bottom reading remains observably necessary.
    nonanchor_count = 2 if length == 3 else _RNG.randint(1, min(3, length - 2))
    nonanchor_colors = _RNG.sample(available, nonanchor_count)

    fillers = list(nonanchor_colors)
    while len(fillers) < length - 1:
        fillers.append(_RNG.choice(nonanchor_colors))
    sequence = fillers + [anchor_color]
    _RNG.shuffle(sequence)

    assert sequence.count(anchor_color) == 1
    assert length == 3 or len(set(sequence)) < len(sequence)
    return tuple(sequence), anchor_color


def _bar_coords(
    height: int, width: int, orientation: str, side: str, length: int
) -> tuple[Coord, ...]:
    axis_length = width if orientation == "horizontal" else height
    if _RNG.random() < 0.35:
        start = _RNG.choice((0, axis_length - length))
    else:
        start = _RNG.randint(0, axis_length - length)

    if orientation == "horizontal":
        row = 0 if side == "first" else height - 1
        return tuple((row, c) for c in range(start, start + length))
    col = 0 if side == "first" else width - 1
    return tuple((r, col) for r in range(start, start + length))


def _choose_scatter(
    height: int, width: int, bar: tuple[Coord, ...], count: int
) -> list[Coord] | None:
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if all(max(abs(r - br), abs(c - bc)) > 1 for br, bc in bar)
    ]
    _RNG.shuffle(candidates)
    chosen: list[Coord] = []
    for point in candidates:
        if all(
            max(abs(point[0] - other[0]), abs(point[1] - other[1])) > 1
            for other in chosen
        ):
            chosen.append(point)
            if len(chosen) == count:
                return chosen
    return None


def _assignment_grid(
    shape: tuple[int, int],
    scatter: list[Coord],
    orientation: str,
    sequence: tuple[int, ...],
    anchor: Coord,
    anchor_color: int,
) -> np.ndarray:
    ordered = _ordered_scatter(scatter, orientation)
    phase = _phase_for_anchor(ordered, sequence, anchor, anchor_color)
    out = np.zeros(shape, dtype=int)
    for rank, coord in enumerate(ordered):
        out[coord] = sequence[(rank + phase) % len(sequence)]
    return out


def _default_phase_grid(
    shape: tuple[int, int], scatter: list[Coord], orientation: str, sequence: tuple[int, ...]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rank, coord in enumerate(_ordered_scatter(scatter, orientation)):
        out[coord] = sequence[rank % len(sequence)]
    return out


def _semantically_strong(
    inp: np.ndarray,
    expected: np.ndarray,
    bar_coords: tuple[Coord, ...],
    sequence: tuple[int, ...],
    scatter: list[Coord],
    orientation: str,
    anchor: Coord,
    anchor_color: int,
) -> bool:
    """Reject valid-looking samples that fail to distinguish common shortcuts."""
    try:
        detected = _find_bar(inp)
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    if detected.coords != bar_coords or detected.colors != sequence:
        return False
    if not np.array_equal(actual, expected):
        return False

    # Anchor must change the phase relative to naively starting at palette[0].
    default = _default_phase_grid(inp.shape, scatter, orientation, sequence)
    if np.array_equal(default, expected):
        return False

    # The bar orientation must materially change the scan order and result.
    other_orientation = "vertical" if orientation == "horizontal" else "horizontal"
    wrong_scan = _assignment_grid(
        inp.shape, scatter, other_orientation, sequence, anchor, anchor_color
    )
    if np.array_equal(wrong_scan, expected):
        return False

    # Reading the palette backwards must not be an equivalent explanation.
    reversed_palette = tuple(reversed(sequence))
    wrong_read = _assignment_grid(
        inp.shape, scatter, orientation, reversed_palette, anchor, anchor_color
    )
    if np.array_equal(wrong_read, expected):
        return False

    scatter_values = {int(expected[coord]) for coord in scatter}
    if scatter_values != set(sequence):
        return False
    if any(int(expected[coord]) != 0 for coord in bar_coords):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = _RNG.randint(13, 23)
    width = _RNG.randint(13, 23)
    orientation = _RNG.choice(("horizontal", "vertical"))
    side = _RNG.choice(("first", "last"))
    axis_length = width if orientation == "horizontal" else height
    length = _RNG.randint(3, min(8, axis_length - 2))
    sequence, anchor_color = _make_palette(length)
    bar = _bar_coords(height, width, orientation, side, length)

    # More than two full periods make the cyclic repetition directly visible.
    scatter_count = 2 * length + _RNG.randint(1, 4)
    scatter = _choose_scatter(height, width, bar, scatter_count)
    if scatter is None:
        return None

    ordered = _ordered_scatter(scatter, orientation)
    anchor_bar_index = sequence.index(anchor_color)
    anchor_rank_choices = [
        rank
        for rank in range(1, len(ordered) - 1)
        if rank % length != anchor_bar_index
    ]
    if not anchor_rank_choices:
        return None
    anchor = ordered[_RNG.choice(anchor_rank_choices)]

    inp = np.zeros((height, width), dtype=int)
    for coord, color in zip(bar, sequence):
        inp[coord] = color

    distractor_pool = [color for color in _COLORS if color not in set(sequence)]
    distractor_colors = _RNG.sample(
        distractor_pool, _RNG.randint(2, min(4, len(distractor_pool)))
    )
    for coord in scatter:
        inp[coord] = anchor_color if coord == anchor else _RNG.choice(distractor_colors)

    expected = _assignment_grid(
        inp.shape, scatter, orientation, sequence, anchor, anchor_color
    )
    if not _semantically_strong(
        inp,
        expected,
        bar,
        sequence,
        scatter,
        orientation,
        anchor,
        anchor_color,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse valid compass-palette instance, rejecting degeneracies."""
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate compass palette input")
