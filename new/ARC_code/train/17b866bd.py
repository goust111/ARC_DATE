from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


_INTERIOR_OFFSETS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
_INTERIOR = set(_INTERIOR_OFFSETS)


def _lattice_color(g: np.ndarray, empty: int) -> int:
    vals, counts = np.unique(g, return_counts=True)
    assert vals.size >= 2, "expected an empty color and a lattice color"
    max_count = int(counts.max())
    winners = vals[counts == max_count]
    assert winners.size == 1, "lattice color must be unique"
    lattice = int(winners[0])
    assert lattice != empty, "the lattice color must differ from the empty color"
    return lattice


def _matches_anchor_template(g: np.ndarray, r: int, c: int, empty: int, lattice: int) -> bool:
    for dr in range(5):
        for dc in range(5):
            if (dr, dc) == (0, 0):
                continue
            expected = empty if (dr, dc) in _INTERIOR else lattice
            if int(g[r + dr, c + dc]) != expected:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker from a lattice corner gap into the adjacent enclosed 12-pixel cavity.

    Input structure:
    - The grid is a repeated teal lattice around black 12-pixel cavities.
    - A marker is a non-black pixel placed at the northwest corner gap of one
      complete 5x5 lattice tile; the marker may itself be teal.

    Rule:
    - For every marked northwest corner gap, clear that marker to black.
    - Fill the 12 black pixels in the tile immediately down-right of the marker
      with the marker color.
    - Leave the rest of the lattice unchanged.

    Key insight:
    - The apparent isolated markers are control pixels for the adjacent cavity;
      teal markers are recognized as extra teal pixels occupying a gap that the
      periodic lattice template says should be black.
    """
    g = np.asarray(grid, dtype=int)
    empty = Color.BACKGROUND
    lattice = _lattice_color(g, empty)
    h, w = g.shape

    anchors: list[tuple[int, int, int]] = []
    for r in range(h - 4):
        for c in range(w - 4):
            if _matches_anchor_template(g, r, c, empty, lattice) and int(g[r, c]) != empty:
                anchors.append((r, c, int(g[r, c])))

    assert anchors, "expected at least one marked lattice anchor"

    out = g.copy()
    for r, c, color in anchors:
        out[r, c] = empty
        for dr, dc in _INTERIOR_OFFSETS:
            assert int(g[r + dr, c + dc]) == empty, "anchor cavity must be empty before filling"
            out[r + dr, c + dc] = color

    return out

# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
_LATTICE = 8                       # lattice / frame color
_EMPTY = 0                        # background / cavity color
_TILE_RANGE = (3, 4)              # rows/cols of tiles (N, M each sampled from this)
_MAX_MARKERS = 9                  # upper bound on marker count per grid
_MAX_DISTINCT_NON_LATTICE = 3     # max distinct non-lattice marker colors
_LATTICE_MARKER_PROB = 0.35       # probability of including lattice-color markers
_NON_LATTICE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
_GEN_RETRIES = 300                # max attempts before giving up

# Reuse cavity offsets from the solver section (identical to _INTERIOR_OFFSETS).
_CAVITY_OFFSETS = _INTERIOR_OFFSETS


def _base_lattice(n_tiles: int, m_tiles: int) -> np.ndarray:
    """Build an empty lattice grid: all lattice cells = _LATTICE, corners and cavities = _EMPTY."""
    h, w = 5 * n_tiles + 1, 5 * m_tiles + 1
    grid = np.full((h, w), _LATTICE, dtype=int)
    rs, cs = np.mgrid[:h, :w]
    dr, dc = rs % 5, cs % 5
    corner = (dr == 0) & (dc == 0)
    cavity = np.zeros((h, w), dtype=bool)
    for ro, co in _CAVITY_OFFSETS:
        cavity |= (dr == ro) & (dc == co)
    grid[corner | cavity] = _EMPTY
    return grid


def _choose_marker_colors(marker_count: int) -> list[int]:
    """Pick and assign marker colors: optionally include lattice color, plus 1-3 non-lattice colors."""
    include_lattice = random.random() < _LATTICE_MARKER_PROB
    if include_lattice:
        non_count = random.randint(0, min(_MAX_DISTINCT_NON_LATTICE, marker_count - 1))
    else:
        non_count = random.randint(1, min(_MAX_DISTINCT_NON_LATTICE, marker_count))

    colors = random.sample(_NON_LATTICE_COLORS, non_count)
    if include_lattice:
        colors.append(_LATTICE)

    assignments = colors[:]
    assignments.extend(random.choice(colors) for _ in range(marker_count - len(assignments)))
    random.shuffle(assignments)
    return assignments


def _gen_attempt() -> np.ndarray | None:
    """Try to generate one valid input grid. Returns None on oracle mismatch."""
    n_tiles = random.randint(*_TILE_RANGE)
    m_tiles = random.randint(*_TILE_RANGE)
    total_tiles = n_tiles * m_tiles
    marker_count = random.randint(1, min(_MAX_MARKERS, total_tiles - 1))

    marked_tiles = [(i, j) for i in range(n_tiles) for j in range(m_tiles)]
    random.shuffle(marked_tiles)
    marked_tiles = marked_tiles[:marker_count]
    marker_colors = _choose_marker_colors(marker_count)

    out = _base_lattice(n_tiles, m_tiles)
    for (i, j), color in zip(marked_tiles, marker_colors):
        r0, c0 = 5 * i, 5 * j
        for dr, dc in _CAVITY_OFFSETS:
            out[r0 + dr, c0 + dc] = color

    inp = out.copy()
    for (i, j), color in zip(marked_tiles, marker_colors):
        r0, c0 = 5 * i, 5 * j
        for dr, dc in _CAVITY_OFFSETS:
            inp[r0 + dr, c0 + dc] = _EMPTY
        inp[r0, c0] = color

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 17b866bd input")
