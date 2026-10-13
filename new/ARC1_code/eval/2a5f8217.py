import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, is_contiguous, paste, random_free_location
import itertools
import random


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    origin = coords.min(axis=0)
    shifted = coords - origin
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue placeholder object with the color of the non-blue object that has the same shape.

    Input structure
    - A black-background grid contains several 4-connected foreground objects.
    - Blue objects are uncolored placeholders.
    - Every non-blue foreground object is a colored shape template.

    Rule
    - Compare objects by their normalized 4-connected pixel mask inside their bounding box.
    - For each blue object, find the unique non-blue template object with the same normalized mask.
    - Recolor all pixels of the blue object to that matched template color; leave templates and background unchanged.

    Key insight
    - Position is irrelevant: each placeholder is identified by shape equality with a colored template elsewhere in the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.BLUE, "blue must be a placeholder foreground color"

    colors = colors_present(g, exclude={bg})
    assert Color.BLUE in colors, "expected at least one blue placeholder object"
    template_colors = [color for color in colors if color != Color.BLUE]
    assert template_colors, "expected at least one non-blue colored template"

    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for color in template_colors:
        components = find_components(g, bg=bg, colors=[color], connectivity=4, monochromatic=True)
        assert components, "expected each template color to appear as an object"
        for component in components:
            key = _normalized_shape(component["coords"])
            assert key not in shape_to_color, "template shapes must be unique"
            shape_to_color[key] = int(color)

    out = g.copy()
    blue_components = find_components(g, bg=bg, colors=[Color.BLUE], connectivity=4, monochromatic=True)
    assert blue_components, "expected at least one blue placeholder object"
    for component in blue_components:
        key = _normalized_shape(component["coords"])
        assert key in shape_to_color, "each blue object must match a colored template shape"
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = shape_to_color[key]

    assert np.any(out != g), "expected at least one placeholder recolor"
    return out

# --------------- generator (o5x) ---------------

# Shape library bounds.
GEN_SHAPE_BBOX, GEN_SHAPE_CELLS_MIN, GEN_SHAPE_CELLS_MAX = 3, 2, 6

# Pair counts and canvas sizing.
GEN_PAIR_MIN, GEN_PAIR_MAX = 1, 5
GEN_CANVAS_MIN, GEN_CANVAS_MAX = 6, 13
GEN_CANVAS_FIT_BASE, GEN_CANVAS_FIT_STEP = 5, 2

# Template color sampling.
GEN_TEMPLATE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_MAX_DISTINCT_COLORS = 4

# Placement spacing and retry limit.
GEN_PLACEMENT_PADDING, GEN_RETRY_LIMIT = 1, 300


def _gen_normalized_coords(coords):
    arr = np.array(coords, dtype=int)
    origin = arr.min(axis=0)
    shifted = arr - origin
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _gen_shapes_by_size():
    shapes = {size: [] for size in range(GEN_SHAPE_CELLS_MIN, GEN_SHAPE_CELLS_MAX + 1)}
    seen = set()
    cells = list(itertools.product(range(GEN_SHAPE_BBOX), range(GEN_SHAPE_BBOX)))
    for size in range(GEN_SHAPE_CELLS_MIN, GEN_SHAPE_CELLS_MAX + 1):
        for combo in itertools.combinations(cells, size):
            mask = np.zeros((GEN_SHAPE_BBOX, GEN_SHAPE_BBOX), dtype=int)
            for r, c in combo:
                mask[r, c] = 1
            if not is_contiguous(mask, background=0, connectivity=4):
                continue
            key = _gen_normalized_coords(combo)
            if key in seen:
                continue
            seen.add(key)
            shapes[size].append(key)
    return shapes


_GEN_SHAPES_BY_SIZE = _gen_shapes_by_size()


def _gen_choose_shapes(k):
    by_size = {size: vals[:] for size, vals in _GEN_SHAPES_BY_SIZE.items()}
    shapes = []
    for _ in range(k):
        usable_sizes = [size for size, vals in by_size.items() if vals]
        size = random.choice(usable_sizes)
        idx = random.randrange(len(by_size[size]))
        shapes.append(by_size[size].pop(idx))
    return shapes


def _gen_choose_colors(k):
    n_colors = random.randint(1, min(GEN_MAX_DISTINCT_COLORS, k))
    chosen = random.sample(GEN_TEMPLATE_COLORS, n_colors)
    colors = chosen[:] + [random.choice(chosen) for _ in range(k - n_colors)]
    random.shuffle(colors)
    return colors


def _gen_shape_patch(shape, color):
    h, w = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _gen_pair_objects(shapes, colors):
    objects = [
        {"pair_id": pair_id, "is_placeholder": is_placeholder, "shape": shape, "color": color}
        for pair_id, (shape, color) in enumerate(zip(shapes, colors))
        for is_placeholder in (False, True)
    ]
    random.shuffle(objects)
    objects.sort(key=lambda obj: (len(obj["shape"]), random.random()), reverse=True)
    return objects


def _gen_place_objects(output, objects):
    """Place objects with the task's one-cell Manhattan clearance."""
    placements = []
    for obj in objects:
        patch = _gen_shape_patch(obj["shape"], obj["color"])
        try:
            r0, c0 = random_free_location(
                output,
                patch,
                transparent=0,
                border=0,
                padding=GEN_PLACEMENT_PADDING,
                padding_connectivity=4,
            )
        except ValueError:
            return None
        paste(output, patch, r0, c0, transparent=0)
        coords = [(r0 + r, c0 + c) for r, c in obj["shape"]]
        placements.append((obj["is_placeholder"], coords))
    return placements


def _gen_attempt():
    k = random.randint(GEN_PAIR_MIN, GEN_PAIR_MAX)
    min_n = min(GEN_CANVAS_MAX, GEN_CANVAS_FIT_BASE + GEN_CANVAS_FIT_STEP * (k - 1))
    n = random.randint(max(GEN_CANVAS_MIN, min_n), GEN_CANVAS_MAX)
    shapes = _gen_choose_shapes(k)
    colors = _gen_choose_colors(k)
    output = np.zeros((n, n), dtype=int)

    placements = _gen_place_objects(output, _gen_pair_objects(shapes, colors))
    if placements is None:
        return None
    inp = output.copy()
    for is_placeholder, coords in placements:
        if is_placeholder:
            for r, c in coords:
                inp[r, c] = Color.BLUE

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2a5f8217 input")
