from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent input color."""
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(counts.max())
    assert int(np.count_nonzero(counts == largest)) == 1, "background must be unique"
    return int(colors[int(np.argmax(counts))])


def _marked_corners(grid: np.ndarray, background: int) -> list[tuple[Coord, int]]:
    """Read all non-background corner pixels and validate the sparse canvas."""
    h, w = map(int, grid.shape)
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    corner_set = set(corners)
    non_background = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    assert non_background, "expected at least one marked corner"
    assert non_background <= corner_set, "only corners may be marked"

    markers = [(corner, int(grid[corner])) for corner in corners if int(grid[corner]) != background]
    marker_colors = [color for _corner, color in markers]
    assert len(marker_colors) == len(set(marker_colors)), "corner marker colors must be distinct"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating Chebyshev-distance waves from marked corners, leaving ties as frozen background seams.

    Input structure:
    - A rectangular background canvas has one-pixel markers at some or all of its four corners.
    - Every marked corner has a distinct color, and every non-corner cell is background.

    Rule:
    - Assign each cell to its uniquely nearest marked corner using Chebyshev distance in full-grid row/column coordinates.
    - Leave cells tied between two or more nearest corners in the background color.
    - In each uniquely owned territory, paint even-distance cells with that corner's marker color and leave odd-distance cells as background.

    Key insight:
    - First resolve multi-source nearest-corner competition; only then apply distance parity, so frozen seams are exactly the unresolved ties rather than separately drawn dividers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "the four grid corners must be distinct"

    bg = _background(g)
    markers = _marked_corners(g, bg)

    rows, cols = np.indices((h, w))
    distances = np.stack(
        [np.maximum(np.abs(rows - r), np.abs(cols - c)) for (r, c), _color in markers],
        axis=0,
    )
    minimum = distances.min(axis=0)
    unique_nearest = np.count_nonzero(distances == minimum[None, :, :], axis=0) == 1
    winner = distances.argmin(axis=0)

    out = np.full((h, w), bg, dtype=int)
    even_ring = (minimum % 2) == 0
    for index, (_corner, color) in enumerate(markers):
        paint = unique_nearest & even_ring & (winner == index)
        out[paint] = color
    return out


# The six-phase schedule guarantees that every three consecutive generated examples
# contain 2, 3, and 4 sources.  The two-source phase alternates adjacent/opposite
# corner geometry; all other choices remain random.
_GENERATION_PHASE = random.randrange(6)
_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 400


def _corner_coords(shape: tuple[int, int]) -> tuple[Coord, Coord, Coord, Coord]:
    h, w = shape
    return (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)


def _choose_marker_indices(phase: int) -> tuple[int, ...]:
    count = (2, 3, 4)[phase % 3]
    if count == 4:
        return (0, 1, 2, 3)
    if count == 3:
        missing = random.randrange(4)
        return tuple(i for i in range(4) if i != missing)

    if phase < 3:
        # Adjacent corners: top, bottom, left, or right edge.
        return random.choice(((0, 1), (2, 3), (0, 2), (1, 3)))
    # Opposite corners exercise diagonal competition.
    return random.choice(((0, 3), (1, 2)))


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    """Small loop-based oracle, intentionally separate from main's vectorized code."""
    bg = _background(grid)
    markers = _marked_corners(grid, bg)
    h, w = map(int, grid.shape)
    expected = np.full((h, w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            distances = [max(abs(r - mr), abs(c - mc)) for (mr, mc), _ in markers]
            nearest = min(distances)
            winners = [i for i, distance in enumerate(distances) if distance == nearest]
            if len(winners) == 1 and nearest % 2 == 0:
                expected[r, c] = markers[winners[0]][1]
    return expected


def _manhattan_shortcut(grid: np.ndarray) -> np.ndarray:
    """Deliberately wrong oracle used to reject metric-ambiguous examples."""
    bg = _background(grid)
    markers = _marked_corners(grid, bg)
    h, w = map(int, grid.shape)
    result = np.full((h, w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            distances = [abs(r - mr) + abs(c - mc) for (mr, mc), _ in markers]
            nearest = min(distances)
            winners = [i for i, distance in enumerate(distances) if distance == nearest]
            if len(winners) == 1 and nearest % 2 == 0:
                result[r, c] = markers[winners[0]][1]
    return result


def _semantically_strong(grid: np.ndarray) -> bool:
    """Check the stated rule and reject samples compatible with common shortcuts."""
    try:
        bg = _background(grid)
        markers = _marked_corners(grid, bg)
        actual = main(grid)
        expected = _independent_expected(grid)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False

    h, w = map(int, grid.shape)
    rows, cols = np.indices((h, w))
    distances = np.stack(
        [np.maximum(np.abs(rows - r), np.abs(cols - c)) for (r, c), _ in markers],
        axis=0,
    )
    minimum = distances.min(axis=0)
    nearest_count = np.count_nonzero(distances == minimum[None, :, :], axis=0)
    winner = distances.argmin(axis=0)
    tied = nearest_count >= 2
    unique = nearest_count == 1

    # A visible frozen seam must survive both parity classes; even-distance ties
    # specifically prove that the seam cannot be explained by odd rings alone.
    if int(tied.sum()) < min(h, w):
        return False
    if not bool(np.any(tied & (minimum % 2 == 0)) and np.any(tied & (minimum % 2 == 1))):
        return False
    if not bool(np.all(actual[tied] == bg)):
        return False
    if float(tied.mean()) > 0.40:
        return False

    # Every source must visibly contribute even rings and odd background gaps.
    for index, (_corner, color) in enumerate(markers):
        owned = unique & (winner == index)
        owned_even = owned & (minimum % 2 == 0)
        owned_odd = owned & (minimum % 2 == 1)
        if int(owned_even.sum()) < 4 or int(owned_odd.sum()) < 4:
            return False
        if not bool(np.all(actual[owned_even] == color) and np.all(actual[owned_odd] == bg)):
            return False

        # There must be an even wave cell of this source captured by another
        # territory, so unrestricted corner rings are not an equivalent rule.
        captured_even = unique & (winner != index) & (distances[index] % 2 == 0)
        if not bool(np.any(captured_even)):
            return False

    if int(np.count_nonzero(actual != grid)) < 10:
        return False
    if int(np.count_nonzero(actual != _manhattan_shortcut(grid))) < 6:
        return False
    return True


def _generate_attempt(phase: int) -> np.ndarray | None:
    # Use both square and rectangular canvases and all parity combinations.
    h = random.randint(11, 25)
    w = random.randint(11, 25)
    if random.random() < 0.75 and h == w:
        w = w + 1 if w < 25 else w - 1

    background, *marker_colors = random.sample(_PALETTE, 5)
    marker_indices = _choose_marker_indices(phase)
    corners = _corner_coords((h, w))

    grid = np.full((h, w), background, dtype=int)
    for index, color in zip(marker_indices, marker_colors):
        grid[corners[index]] = color

    if not _semantically_strong(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate marked-corner competition canvas."""
    global _GENERATION_PHASE
    phase = _GENERATION_PHASE
    _GENERATION_PHASE = (_GENERATION_PHASE + 1) % 6

    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt(phase)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a semantically strong corner-wave input")
