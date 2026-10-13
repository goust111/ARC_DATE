import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the interior of each solid grey rectangle red while leaving its border unchanged.

    Input structure
    - The grid has black background and one or more 4-connected grey foreground objects.
    - Each grey object is a filled axis-aligned rectangle with height and width at least 3.

    Rule
    - Preserve the black background and the one-cell grey border of every rectangle.
    - Recolor every grey cell strictly inside a rectangle's border to red.

    Key insight
    - The selected cells are not based on absolute position; they are exactly the non-border cells of each filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY}, "expected only black and grey input colors"

    components = find_components(g, bg=Color.BLACK, connectivity=4, colors=[Color.GREY])
    assert components, "expected at least one grey rectangle"

    out = g.copy()
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        height = bbox_h(component["bbox"])
        width = bbox_w(component["bbox"])
        assert height >= 3 and width >= 3, "grey rectangles must have interiors"
        assert component["area"] == height * width, "grey foreground objects must be filled rectangles"
        out[r0 + 1 : r1, c0 + 1 : c1] = Color.RED

    return out

# --------------- generator (o5x) ---------------
# Canvas, rectangle size ranges, placement gap, and retry budget.
GRID_SIZE = 10
RECT_HEIGHT_RANGE = (3, 7)
RECT_WIDTH_RANGE = (3, 8)
MIN_BLACK_GAP = 1
MAX_GENERATION_ATTEMPTS = 300


def _gen_rects_have_gap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + MIN_BLACK_GAP < br0 or br1 + MIN_BLACK_GAP < ar0 or (
        ac1 + MIN_BLACK_GAP < bc0 or bc1 + MIN_BLACK_GAP < ac0
    )


def _gen_attempt():
    h1 = int(np.random.randint(*RECT_HEIGHT_RANGE))
    w1 = int(np.random.randint(*RECT_WIDTH_RANGE))
    h2 = int(np.random.randint(*RECT_HEIGHT_RANGE))
    w2 = int(np.random.randint(*RECT_WIDTH_RANGE))

    placements = []
    for r1 in range(GRID_SIZE - h1 + 1):
        for c1 in range(GRID_SIZE - w1 + 1):
            b1 = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
            for r2 in range(GRID_SIZE - h2 + 1):
                for c2 in range(GRID_SIZE - w2 + 1):
                    b2 = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
                    if _gen_rects_have_gap(b1, b2):
                        placements.append((r1, c1, r2, c2))

    if not placements:
        return None

    r1, c1, r2, c2 = placements[int(np.random.randint(len(placements)))]
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = grid.copy()
    rects = [(r1, c1, h1, w1), (r2, c2, h2, w2)]
    for r, c, h, w in rects:
        rect = np.full((h, w), Color.GREY, dtype=int)
        paste(grid, rect, r, c)
        paste(expected, rect, r, c)
        expected[r + 1 : r + h - 1, c + 1 : c + w - 1] = Color.RED

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bb43febb input")
