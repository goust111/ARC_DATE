import numpy as np

from arc_utils import Color, find_components, get_bg, in_bounds, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Add the four colored diagonal corner markers around every grey 2x2 square.

    Input structure
    - The grid has black background pixels and one or more 4-connected grey
      foreground objects.
    - Every grey object is a solid 2x2 square with a one-cell black margin at
      each diagonal outside corner.

    Rule
    - Preserve the input grid.
    - For each grey square, paint blue at the northwest outside corner, red at
      the northeast outside corner, green at the southwest outside corner, and
      yellow at the southeast outside corner.

    Key insight
    - The grey squares are fixed anchors; the output is a local decoration of
      each square's bounding box, not a movement or matching problem.
    """
    g = np.asarray(grid, dtype=int)

    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.GREY})

    out = g.copy()
    squares = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.GREY])
    assert squares

    for square in squares:
        assert square["color"] == Color.GREY
        assert square["area"] == 4
        r0, c0, r1, c1 = square["bbox"]
        assert r1 - r0 + 1 == 2
        assert c1 - c0 + 1 == 2
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY)

        markers = [
            (r0 - 1, c0 - 1, Color.BLUE),
            (r0 - 1, c1 + 1, Color.RED),
            (r1 + 1, c0 - 1, Color.GREEN),
            (r1 + 1, c1 + 1, Color.YELLOW),
        ]
        for r, c, color in markers:
            assert in_bounds(g.shape, r, c)
            assert g[r, c] == bg
            assert out[r, c] == bg
            out[r, c] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas and square bounds.
GRID_SIZE_RANGE = (10, 15)
MAX_SQUARES = 6
SQUARE_SIZE = 2
SQUARE_COLOR = 5
# Placement geometry and retry budget.
PLACEMENT_START = 1
ZONE_SPACING = 4
GENERATION_RETRIES = 300
# Marker offsets/colors from each square top-left.
MARKER_OFFSETS = ((-1, -1, 1), (-1, 2, 2), (2, -1, 3), (2, 2, 4))


def _gen_95990924_zones_disjoint(pos, placed):
    r, c = pos
    return all(abs(r - pr) >= ZONE_SPACING or abs(c - pc) >= ZONE_SPACING for pr, pc in placed)


def _gen_95990924_expected(grid, positions):
    out = grid.copy()
    for r, c in positions:
        for dr, dc, color in MARKER_OFFSETS:
            out[r + dr, c + dc] = color
    return out


def _gen_95990924_max_squares(n):
    row_slots = (n - ZONE_SPACING) // ZONE_SPACING + 1
    return min(MAX_SQUARES, row_slots * row_slots)

def _gen_95990924_sample_positions(n, k):
    """Sample square top-lefts using the original greedy viability heuristic."""
    fits = _gen_95990924_zones_disjoint
    slots = range(PLACEMENT_START, n - SQUARE_SIZE)
    candidates = [(r, c) for r in slots for c in slots]
    placed = []
    for _ in range(k):
        available = [pos for pos in candidates if fits(pos, placed)]
        if not available:
            return None
        if len(placed) + 1 < k:
            remaining = k - len(placed) - 1
            viable = []
            for pos in available:
                next_placed = placed + [pos]
                future_count = sum(fits(other, next_placed) for other in candidates)
                if future_count >= remaining:
                    viable.append(pos)
            if viable:
                available = viable
        placed.append(random.choice(available))
    return placed

def _gen_attempt():
    n = random.randint(*GRID_SIZE_RANGE)
    k = random.randint(1, _gen_95990924_max_squares(n))
    placed = _gen_95990924_sample_positions(n, k)
    if placed is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    square = np.full((SQUARE_SIZE, SQUARE_SIZE), SQUARE_COLOR, dtype=int)
    for r, c in placed:
        paste(grid, square, r, c)

    expected = _gen_95990924_expected(grid, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 95990924 input")
