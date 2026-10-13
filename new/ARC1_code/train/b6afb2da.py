import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid gray rectangle by marking its corners blue, border yellow, and interior red.

    Input structure
    - The grid contains black background cells and one or more gray foreground objects.
    - Each gray object is a 4-connected solid rectangle with height and width at
      least 2.

    Rule
    - For every gray rectangle, recolor its four corner cells blue.
    - Recolor its remaining border cells yellow.
    - Recolor all strict interior cells red.
    - Leave black background cells unchanged.

    Key insight
    - The output colors encode each rectangle's local geometry, not its position
      or size: corners, edges, and interiors receive fixed roles independently
      for every gray component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.GREY}, "expected only black and gray"

    out = g.copy()
    components = find_components(
        g,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
    )
    assert components, "expected at least one gray rectangle"

    gray_area = 0
    for comp in components:
        assert comp["color"] == Color.GREY, "expected gray foreground components"
        r0, c0, r1, c1 = comp["bbox"]
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        assert h >= 2 and w >= 2, "gray rectangles must have corners"
        assert comp["area"] == h * w, "gray component must fill its bounding box"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY), "gray rectangle must be solid"

        gray_area += int(comp["area"])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_horizontal_edge = r in (r0, r1)
                on_vertical_edge = c in (c0, c1)
                if on_horizontal_edge and on_vertical_edge:
                    out[r, c] = Color.BLUE
                elif on_horizontal_edge or on_vertical_edge:
                    out[r, c] = Color.YELLOW
                else:
                    out[r, c] = Color.RED

    assert gray_area == int(np.sum(g == Color.GREY)), "all gray pixels must belong to rectangles"
    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle bounds.
GRID_SIZE = 10
RECT_HEIGHT_OPTIONS = tuple(range(3, 7))
RECT_WIDTH_OPTIONS = tuple(range(4, 7))
MIN_EMPTY_GAP = 1

# Retry limits.
MAX_GENERATION_ATTEMPTS = 300


def _gen_rectangles_separated(rect_a, rect_b):
    ar, ac, ah, aw = rect_a
    br, bc, bh, bw = rect_b
    ar1, ac1 = ar + ah - 1, ac + aw - 1
    br1, bc1 = br + bh - 1, bc + bw - 1
    row_gap = ar1 + MIN_EMPTY_GAP < br or br1 + MIN_EMPTY_GAP < ar
    col_gap = ac1 + MIN_EMPTY_GAP < bc or bc1 + MIN_EMPTY_GAP < ac
    return row_gap or col_gap


def _gen_dimension_options():
    options = []
    for h1 in RECT_HEIGHT_OPTIONS:
        for w1 in RECT_WIDTH_OPTIONS:
            for h2 in RECT_HEIGHT_OPTIONS:
                for w2 in RECT_WIDTH_OPTIONS:
                    if h1 + h2 + MIN_EMPTY_GAP <= GRID_SIZE or w1 + w2 + MIN_EMPTY_GAP <= GRID_SIZE:
                        options.append((h1, w1, h2, w2))
    return tuple(options)


DIMENSION_OPTIONS = _gen_dimension_options()


def _gen_position_pairs(h1, w1, h2, w2):
    rects1 = [(r, c, h1, w1) for r in range(GRID_SIZE + 1 - h1) for c in range(GRID_SIZE + 1 - w1)]
    rects2 = [(r, c, h2, w2) for r in range(GRID_SIZE + 1 - h2) for c in range(GRID_SIZE + 1 - w2)]
    return [
        (rect1, rect2)
        for rect1 in rects1
        for rect2 in rects2
        if _gen_rectangles_separated(rect1, rect2)
    ]


def _gen_render_input(rects):
    grid = np.full((GRID_SIZE, GRID_SIZE), Color.BLACK, dtype=int)
    for r0, c0, h, w in rects:
        grid[r0 : r0 + h, c0 : c0 + w] = Color.GREY
    return grid


def _gen_expected_output(grid, rects):
    out = grid.copy()
    for r0, c0, h, w in rects:
        r1, c1 = r0 + h - 1, c0 + w - 1
        out[r0 : r1 + 1, c0 : c1 + 1] = Color.RED
        out[r0, c0 : c1 + 1] = Color.YELLOW
        out[r1, c0 : c1 + 1] = Color.YELLOW
        out[r0 : r1 + 1, c0] = Color.YELLOW
        out[r0 : r1 + 1, c1] = Color.YELLOW
        out[(r0, r0, r1, r1), (c0, c1, c0, c1)] = Color.BLUE
    return out


def _gen_attempt():
    rng = np.random.default_rng()
    h1, w1, h2, w2 = DIMENSION_OPTIONS[int(rng.integers(len(DIMENSION_OPTIONS)))]
    position_pairs = _gen_position_pairs(h1, w1, h2, w2)
    if not position_pairs:
        return None

    rects = position_pairs[int(rng.integers(len(position_pairs)))]
    inp = _gen_render_input(rects)
    expected = _gen_expected_output(inp, rects)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b6afb2da input")
