import numpy as np

from arc_utils import Color, find_components


def _assert_solid_rect(g: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(region == obj["color"]), "each foreground object must be a solid rectangle"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every third monochromatic foreground bar in top-to-bottom order red, starting with the first.

    Input structure:
    - Black cells are the background.
    - The foreground consists of solid blue/teal horizontal rectangles.
    - Each rectangle is a maximal 4-connected monochromatic foreground object.

    Rule:
    - Keep the grid shape and black background unchanged.
    - Order the foreground objects by top row, then left column.
    - Recolor objects with ranks 1, 4, 7, ... to red; leave all other foreground objects unchanged.

    Key insight:
    - The selected objects are not chosen by their color or size; only their ordinal position in the vertical object sequence matters.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, g.flat))
    assert Color.BLACK in colors, "expected black background cells"
    assert colors - {Color.BLACK} == {Color.BLUE, Color.TEAL}, "expected blue and teal foreground objects"

    objects = find_components(g, bg=Color.BLACK, connectivity=4, monochromatic=True)
    assert objects, "expected foreground objects"
    for obj in objects:
        _assert_solid_rect(g, obj)

    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    out = g.copy()
    for i, obj in enumerate(ordered):
        if i % 3 == 0:
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Canvas and object bounds.
GRID_HEIGHT_MIN, GRID_HEIGHT_MAX = 14, 22
GRID_WIDTH_MIN, GRID_WIDTH_MAX = 3, 5
OBJECT_COUNT_MIN, OBJECT_COUNT_MAX = 8, 16
# Rectangle geometry.
RECT_HEIGHT_MIN, RECT_HEIGHT_MID, RECT_HEIGHT_MAX = 1, 2, 3
RECT_WIDTH_MIN = 2
# Sampling weights and colors.
HEIGHT_THREE_WEIGHTS = (1.0, 0.35)
COLOR_ALTERNATION_PERIOD, FOREGROUND_COLORS = 2, [int(Color.BLUE), int(Color.TEAL)]
# Validation and transform constants.
BACKGROUND_COLOR, RECOLOR_COLOR = int(Color.BLACK), int(Color.RED)
RECOLOR_PERIOD, GENERATION_RETRIES = 3, 300


def _gen_heights(n):
    min_total = max(GRID_HEIGHT_MIN, RECT_HEIGHT_MIN * n)
    max_total = min(GRID_HEIGHT_MAX, RECT_HEIGHT_MAX * n)
    if min_total > max_total:
        return None

    total = int(np.random.randint(min_total, max_total + 1))
    extra = total - RECT_HEIGHT_MIN * n
    mid_extra = RECT_HEIGHT_MID - RECT_HEIGHT_MIN
    max_extra = RECT_HEIGHT_MAX - RECT_HEIGHT_MIN
    min_threes = max(0, extra - mid_extra * n)
    max_threes = extra // max_extra
    three_choices = np.arange(min_threes, max_threes + 1)
    if three_choices.size == 1:
        n_threes = int(three_choices[0])
    else:
        weights = np.linspace(*HEIGHT_THREE_WEIGHTS, three_choices.size)
        weights = weights / weights.sum()
        n_threes = int(np.random.choice(three_choices, p=weights))

    n_twos = extra - max_extra * n_threes
    heights = np.array(
        [RECT_HEIGHT_MAX] * n_threes
        + [RECT_HEIGHT_MID] * n_twos
        + [RECT_HEIGHT_MIN] * (n - n_twos - n_threes),
        dtype=int,
    )
    np.random.shuffle(heights)
    return heights.tolist()


def _alternating_colors(n_objects):
    start = int(np.random.choice(FOREGROUND_COLORS))
    other = FOREGROUND_COLORS[1] if start == FOREGROUND_COLORS[0] else FOREGROUND_COLORS[0]
    return [start if i % COLOR_ALTERNATION_PERIOD == 0 else other for i in range(n_objects)]


def _render_stacked_rectangles(width, heights, colors):
    grid = np.full((sum(heights), width), BACKGROUND_COLOR, dtype=int)
    narrow_i = int(np.random.randint(len(heights)))

    r = 0
    for i, h in enumerate(heights):
        if i == narrow_i:
            rect_w = int(np.random.randint(RECT_WIDTH_MIN, width))
        else:
            rect_w = int(np.random.randint(RECT_WIDTH_MIN, width + 1))
        c0 = int(np.random.randint(0, width - rect_w + 1))
        grid[r : r + h, c0 : c0 + rect_w] = colors[i]
        r += h
    return grid


def _gen_expected_output(inp):
    out = inp.copy()
    r = obj_i = 0

    while r < inp.shape[0]:
        cols = np.flatnonzero(inp[r] != BACKGROUND_COLOR)
        if cols.size == 0:
            return None

        c0, c1 = int(cols[0]), int(cols[-1])
        color = int(inp[r, c0])
        if (
            not np.all(inp[r, c0 : c1 + 1] == color)
            or np.any(inp[r, :c0] != BACKGROUND_COLOR)
            or np.any(inp[r, c1 + 1 :] != BACKGROUND_COLOR)
        ):
            return None

        r1 = r + 1
        while r1 < inp.shape[0] and np.array_equal(inp[r1], inp[r]):
            r1 += 1

        if obj_i % RECOLOR_PERIOD == 0:
            out[r:r1, c0 : c1 + 1] = RECOLOR_COLOR

        r = r1
        obj_i += 1

    return out


def _gen_attempt():
    width = int(np.random.randint(GRID_WIDTH_MIN, GRID_WIDTH_MAX + 1))
    n_objects = int(np.random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))
    heights = _gen_heights(n_objects)
    if heights is None:
        return None
    colors = _alternating_colors(n_objects)
    return _render_stacked_rectangles(width, heights, colors)


def generate_input():
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue

        expected = _gen_expected_output(inp)
        if expected is None:
            continue

        try:
            actual = main(inp)
        except Exception:
            continue

        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid 22a4bbc2 input")
