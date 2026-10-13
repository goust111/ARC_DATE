import numpy as np

from arc_utils import Color, colors_present, crop_bbox, find_components, get_bg, label_mask, paste, random_free_location
import random


HOLE_COLOR = {
    1: Color.BLUE,
    2: Color.RED,
    3: Color.GREEN,
    4: Color.ORANGE,
}


def _hole_count(mask: np.ndarray) -> int:
    padded_bg = np.pad(~mask, 1, constant_values=True)
    labels, n_labels = label_mask(padded_bg, connectivity=4)
    outside = int(labels[0, 0])
    return sum(1 for k in range(1, n_labels + 1) if k != outside)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each foreground object according to how many enclosed background regions it contains.

    Input structure
    - A background canvas with one foreground color forming several 4-connected objects.
    - Each object is a filled rectangular blob with one to four enclosed background holes.

    Rule
    - Preserve the background.
    - Count the 4-connected enclosed background components inside each object's bounding box.
    - Recolor the whole object by the learned hole palette: one hole -> blue, two -> red, three -> green, four -> orange.

    Key insight
    - The internal holes are the only varying symbol; object size and absolute position do not determine the output color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    src_color = fg_colors[0]

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[src_color],
    )
    assert len(components) > 0

    out = np.full_like(g, bg)
    for comp in components:
        assert comp["color"] == src_color
        obj_mask = crop_bbox(g == src_color, comp["bbox"])
        holes = _hole_count(obj_mask)
        assert holes in HOLE_COLOR
        rr, cc = comp["coords"].T
        out[rr, cc] = HOLE_COLOR[holes]

    return out

# --------------- generator (o5x) ---------------

_GEN_ROW_RANGE = (16, 19)
_GEN_COL_RANGE = (16, 18)
_GEN_OBJECT_COUNT_RANGE = (4, 6)
_GEN_HOLE_COUNT_RANGE = (1, 4)
_GEN_FOREGROUND_COLOR = Color.TEAL
_GEN_BACKGROUND_COLOR = Color.BLACK
_GEN_OBJECT_HEIGHT_RANGE = (3, 9)
_GEN_OBJECT_WIDTH_RANGE = (3, 7)
_GEN_HOLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
_GEN_INTERIOR_MARGIN = 1
_GEN_MIN_HOLE_DISTANCE = 2
_GEN_MAX_HOLE_AREA = max(h * w for h, w in _GEN_HOLE_SHAPES)
_GEN_MIN_FG_AREA_BY_HOLE_COUNT = {1: 8, 2: 12, 3: 17, 4: 20}
_GEN_FG_BUDGET_RATIO = 0.46
_GEN_HOLE_SEARCH_TRIES = 8
_GEN_PLACEMENT_PADDING = 1
_GEN_PLACEMENT_CONNECTIVITY = 8
_GEN_MAX_ATTEMPTS = 300
_GEN_HOLE_COLOR = {1: Color.BLUE, 2: Color.RED, 3: Color.GREEN, 4: Color.ORANGE}


def _gen_hole_options(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    options = []
    for hh, ww in _GEN_HOLE_SHAPES:
        r_stop = height - _GEN_INTERIOR_MARGIN - hh + 1
        c_stop = width - _GEN_INTERIOR_MARGIN - ww + 1
        for r0 in range(_GEN_INTERIOR_MARGIN, r_stop):
            for c0 in range(_GEN_INTERIOR_MARGIN, c_stop):
                cells = tuple((r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww))
                options.append(cells)
    return options


def _gen_separated(cells: tuple[tuple[int, int], ...], used: tuple[tuple[int, int], ...]) -> bool:
    for r0, c0 in cells:
        for r1, c1 in used:
            if abs(r0 - r1) + abs(c0 - c1) < _GEN_MIN_HOLE_DISTANCE:
                return False
    return True


def _gen_search_holes(
    options: list[tuple[tuple[int, int], ...]],
    remaining: int,
    used: tuple[tuple[int, int], ...],
    *,
    shuffled: bool,
) -> list[tuple[tuple[int, int], ...]] | None:
    if remaining == 0:
        return []

    pool = options[:]
    if shuffled:
        random.shuffle(pool)

    for cells in pool:
        if not _gen_separated(cells, used):
            continue
        tail = _gen_search_holes(options, remaining - 1, used + cells, shuffled=shuffled)
        if tail is not None:
            return [cells] + tail
    return None


def _gen_feasible_dims(n_holes: int) -> list[tuple[int, int]]:
    cache = getattr(_gen_feasible_dims, "_cache", {})
    if n_holes not in cache:
        dims = []
        min_height, max_height = _GEN_OBJECT_HEIGHT_RANGE
        min_width, max_width = _GEN_OBJECT_WIDTH_RANGE
        for height in range(min_height, max_height + 1):
            for width in range(min_width, max_width + 1):
                options = _gen_hole_options(height, width)
                if _gen_search_holes(options, n_holes, (), shuffled=False) is not None:
                    dims.append((height, width))
        cache[n_holes] = dims
        _gen_feasible_dims._cache = cache
    return cache[n_holes]

def _gen_make_object(n_holes: int, fg: int, max_fg_area: int | None = None) -> np.ndarray | None:
    dims = _gen_feasible_dims(n_holes)[:]
    random.shuffle(dims)

    for height, width in dims:
        if max_fg_area is not None and height * width - _GEN_MAX_HOLE_AREA * n_holes > max_fg_area:
            continue

        options = _gen_hole_options(height, width)
        for _ in range(_GEN_HOLE_SEARCH_TRIES):
            holes = _gen_search_holes(options, n_holes, (), shuffled=True)
            if holes is None:
                break

            obj = np.full((height, width), int(fg), dtype=int)
            for cells in holes:
                for r, c in cells:
                    obj[r, c] = _GEN_BACKGROUND_COLOR

            if max_fg_area is not None and int(np.count_nonzero(obj == int(fg))) > max_fg_area:
                continue
            if _hole_count(obj == int(fg)) != n_holes:
                continue
            return obj
    return None


def _gen_build_object_specs(hole_counts: list[int], fg: int, remaining_fg_budget: int) -> list[tuple[np.ndarray, int]] | None:
    """Build all objects while reserving enough foreground budget for later ones."""
    specs = []
    for i, n_holes in enumerate(hole_counts):
        min_future = sum(_GEN_MIN_FG_AREA_BY_HOLE_COUNT[int(count)] for count in hole_counts[i + 1 :])
        obj = _gen_make_object(n_holes, fg, max_fg_area=remaining_fg_budget - min_future)
        if obj is None:
            return None
        remaining_fg_budget -= int(np.count_nonzero(obj == int(fg)))
        specs.append((obj, n_holes))
    return specs


def _gen_render_scene(rows: int, cols: int, specs: list[tuple[np.ndarray, int]], fg: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Place objects and return the generated input plus its expected solver output."""
    inp = np.full((rows, cols), _GEN_BACKGROUND_COLOR, dtype=int)
    expected = np.full((rows, cols), _GEN_BACKGROUND_COLOR, dtype=int)
    occupied_bboxes = np.full((rows, cols), _GEN_BACKGROUND_COLOR, dtype=int)

    for obj, n_holes in specs:
        bbox_patch = np.ones(obj.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied_bboxes,
                bbox_patch,
                transparent=0,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(occupied_bboxes, bbox_patch, r0, c0, transparent=0)
        paste(inp, obj, r0, c0, transparent=Color.BLACK)
        out_obj = np.where(obj == int(fg), int(_GEN_HOLE_COLOR[n_holes]), _GEN_BACKGROUND_COLOR)
        paste(expected, out_obj, r0, c0, transparent=_GEN_BACKGROUND_COLOR)
    return inp, expected


def _gen_passes_oracle(inp: np.ndarray, expected: np.ndarray, fg: int) -> bool:
    """Reject samples whose expected output does not match the solver exactly."""
    if int(np.count_nonzero(inp == _GEN_BACKGROUND_COLOR)) <= int(np.count_nonzero(inp == int(fg))):
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_ROW_RANGE)
    cols = random.randint(*_GEN_COL_RANGE)
    fg = _GEN_FOREGROUND_COLOR
    n_objects = random.randint(*_GEN_OBJECT_COUNT_RANGE)
    hole_counts = [random.randint(*_GEN_HOLE_COUNT_RANGE) for _ in range(n_objects)]
    remaining_fg_budget = int(rows * cols * _GEN_FG_BUDGET_RATIO)

    specs = _gen_build_object_specs(hole_counts, fg, remaining_fg_budget)
    if specs is None:
        return None
    random.shuffle(specs)
    specs.sort(key=lambda item: item[0].size, reverse=True)

    rendered = _gen_render_scene(rows, cols, specs, fg)
    if rendered is None:
        return None
    inp, expected = rendered
    if not _gen_passes_oracle(inp, expected, fg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid 37d3e8b2 input")
