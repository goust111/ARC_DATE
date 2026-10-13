import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Build a right-aligned histogram of foreground colors by their object counts.

    Input structure
    A background canvas contains multiple solid monochromatic foreground objects;
    objects are counted separately per color using 4-connectivity.

    Rule
    Count the 4-connected objects of each non-background color. Sort the colors by
    decreasing object count, requiring those counts to be unique, and create one
    output row per color. The output width is the largest count; each row is
    background on the left and that row's color repeated count times on the right.

    Key insight
    The small output does not preserve object positions or object sizes; it is a
    ranked histogram of how many separate objects each color has.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    count_by_color: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
        )
        count = len(comps)
        assert count > 0
        count_by_color.append((count, color))

    assert count_by_color
    counts = [count for count, _color in count_by_color]
    assert len(set(counts)) == len(counts)

    count_by_color.sort(key=lambda item: item[0], reverse=True)
    height = len(count_by_color)
    width = count_by_color[0][0]
    out = np.full((height, width), bg, dtype=int)
    for row, (count, color) in enumerate(count_by_color):
        out[row, width - count :] = color

    return out

# --------------- generator (o5x) ---------------

from itertools import product

# Canvas, palette, and retries.
CANVAS_SHAPE = (16, 16)
CANVAS_AREA = CANVAS_SHAPE[0] * CANVAS_SHAPE[1]
BACKGROUND_COLOR = 0
FOREGROUND_COLORS = range(1, 10)
GENERATION_RETRIES = 300

# Color/object count constraints.
COLOR_COUNT_OPTIONS = [3, 4]
COUNT_VALUE_RANGE = range(1, 6)
MIN_TOTAL_OBJECTS, MAX_TOTAL_OBJECTS = 6, 11

# Square size sampling and placement.
SQUARE_SIZES = [2, 3, 4, 5]
SIZE_WEIGHTS_BY_TOTAL = ((7, [2, 3, 3, 2]), (9, [3, 3, 2, 1]), (None, [4, 4, 2, 1]))
MIN_PLACEMENT_SIZE = 2
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 8
PLACEMENT_KWARGS = dict(
    transparent=BACKGROUND_COLOR,
    padding=PLACEMENT_PADDING,
    padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
)


def _gen_count_choices(k):
    choices = {
        tuple(sorted(counts))
        for counts in product(COUNT_VALUE_RANGE, repeat=k)
        if len(set(counts)) == k
        and MIN_TOTAL_OBJECTS <= sum(counts) <= MAX_TOTAL_OBJECTS
    }
    return sorted(choices)

def _build_expected_output(counts, colors):
    ranked = sorted(zip(counts, colors), key=lambda item: item[0], reverse=True)
    max_count = ranked[0][0]
    out = np.zeros((len(colors), max_count), dtype=int)
    for row, (count, color) in enumerate(ranked):
        out[row, max_count - count :] = color
    return out

def _gen_size(total_objects):
    for cutoff, weights in SIZE_WEIGHTS_BY_TOTAL:
        if cutoff is None or total_objects <= cutoff:
            return random.choices(SQUARE_SIZES, weights=weights, k=1)[0]

def _make_objects(counts, colors):
    total_objects = sum(counts)
    color_area = {color: 0 for color in colors}
    objects = []
    for count, color in zip(counts, colors):
        for _ in range(count):
            size = _gen_size(total_objects)
            color_area[color] += size * size
            objects.append((size, color, random.random()))
    return objects, color_area

def _render_canvas(objects):
    """Place largest squares first; return None if any object cannot fit."""
    canvas = np.zeros(CANVAS_SHAPE, dtype=int)
    objects.sort(key=lambda item: (-item[0], item[2]))
    for size, color, _tie in objects:
        placed = False
        for s in range(size, MIN_PLACEMENT_SIZE - 1, -1):
            patch = np.full((s, s), color, dtype=int)
            try:
                r, c = random_free_location(canvas, patch, **PLACEMENT_KWARGS)
            except ValueError:
                continue
            paste(canvas, patch, r, c)
            placed = True
            break
        if not placed:
            return None
    return canvas

def _gen_attempt():
    k = random.choice(COLOR_COUNT_OPTIONS)
    counts = list(random.choice(_gen_count_choices(k)))
    random.shuffle(counts)
    colors = random.sample(FOREGROUND_COLORS, k)

    expected = _build_expected_output(counts, colors)
    objects, color_area = _make_objects(counts, colors)
    if CANVAS_AREA - sum(color_area.values()) <= max(color_area.values()):
        return None

    canvas = _render_canvas(objects)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 2753e76c input")
