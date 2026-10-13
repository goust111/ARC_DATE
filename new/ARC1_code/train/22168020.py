import numpy as np

from arc_utils import find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored outline by painting the horizontal row spans enclosed by its paired boundary pixels.

    Input structure
    - The grid has a dominant background and one or more monochromatic 8-connected foreground outlines.
    - Each outline has exactly two pixels on every row it occupies; those pixels are the left and right boundary of that row.

    Rule
    - For each foreground outline, process only the rows touched by that outline.
    - On each touched row, recolor the whole interval between the two boundary pixels to the outline color.
    - Keep all other pixels unchanged.

    Key insight
    - The sparse diagonal sides already mark the row-by-row horizontal fill limits of the solid shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground outline"

    out = g.copy()
    for obj in objects:
        color = int(obj["color"])
        coords = obj["coords"]
        rows = sorted({int(r) for r in coords[:, 0]})
        assert rows, "expected each object to occupy at least one row"

        for r in rows:
            row_cols = coords[coords[:, 0] == r, 1]
            assert row_cols.size == 2, "expected exactly two boundary pixels per occupied row"
            c0 = int(row_cols.min())
            c1 = int(row_cols.max())

            span = g[r, c0 : c1 + 1]
            assert np.all((span == bg) | (span == color)), "row span must contain only background and the outline color"
            out[r, c0 : c1 + 1] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, sampling bounds, and retry limit.
CANVAS_SIZE = 10
BACKGROUND = 0
COLOR_RANGE = range(1, 10)
OUTLINE_COUNT_BOUNDS = (1, 3)
HEIGHT_BOUNDS = (3, 5)
GENERATION_ATTEMPTS = 300

def _gen_outline_cells(r0, c0, height):
    """Return boundary cells and filled row spans for one V outline."""
    width = 2 * (height - 1)
    boundary = []
    filled = []
    for k in range(height):
        side_k = min(k, height - 2)
        r = r0 + k
        c_left = c0 + side_k
        c_right = c0 + width - 1 - side_k
        boundary.append((r, c_left))
        boundary.append((r, c_right))
        for c in range(c_left, c_right + 1):
            filled.append((r, c))
    return boundary, filled

def _gen_valid_positions(height, used_filled):
    """Return shuffled placements whose filled spans do not overlap used cells."""
    width = 2 * (height - 1)
    positions = []
    for r0 in range(CANVAS_SIZE + 1 - height):
        for c0 in range(CANVAS_SIZE + 1 - width):
            boundary, filled = _gen_outline_cells(r0, c0, height)
            if all(not used_filled[r, c] for r, c in filled):
                positions.append((r0, c0, boundary, filled))
    random.shuffle(positions)
    return positions

def _place_outlines(heights, order):
    """Backtrack over shuffled valid positions; return placements or None."""
    placements = [None] * len(heights)
    used_filled = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    def place_from(pos):
        if pos == len(order):
            return True

        idx = order[pos]
        for r0, c0, boundary, filled in _gen_valid_positions(heights[idx], used_filled):
            for r, c in filled:
                used_filled[r, c] = True
            placements[idx] = (r0, c0, boundary, filled)

            if place_from(pos + 1):
                return True

            placements[idx] = None
            for r, c in filled:
                used_filled[r, c] = False

        return False

    return placements if place_from(0) else None

def _render_grids(colors, placements):
    """Render sparse input and its filled oracle target."""
    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), BACKGROUND, dtype=int)
    expected = np.full((CANVAS_SIZE, CANVAS_SIZE), BACKGROUND, dtype=int)
    for color, placement in zip(colors, placements):
        _, _, boundary, filled = placement
        for r, c in boundary:
            inp[r, c] = color
        for r, c in filled:
            expected[r, c] = color
    return inp, expected

def _gen_attempt():
    n_outlines = random.randint(*OUTLINE_COUNT_BOUNDS)
    colors = random.sample(COLOR_RANGE, n_outlines)
    heights = [random.randint(*HEIGHT_BOUNDS) for _ in range(n_outlines)]
    order = list(range(n_outlines))
    random.shuffle(order)

    placements = _place_outlines(heights, order)
    if placements is None:
        return None

    inp, expected = _render_grids(colors, placements)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp

def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 22168020 input")
