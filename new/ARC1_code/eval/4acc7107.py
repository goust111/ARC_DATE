import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, paste


def _component_patch(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(component["color"])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Pack each color's foreground objects into bottom-aligned columns while preserving each object's shape.

    Input structure
    The grid has a dominant background color and two foreground color groups.
    Each foreground color group contains exactly two monochromatic 4-connected
    objects.

    Rule
    Order the color groups by their leftmost occupied column, creating output
    columns from left to right with one background column between groups. Within
    each color group, order its objects by decreasing bbox left column, crop
    each object mask to its bbox, and stack those objects top-to-bottom in that
    order with one background row between them. Bottom-align each color stack on
    the original-size output canvas.

    Key insight
    Original position is used as ordering information: color groups sort
    left-to-right, while objects inside a color sort right-to-left before being
    repacked at the bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color = {color: [] for color in colors}
    for component in components:
        by_color[int(component["color"])].append(component)
    assert all(len(group) == 2 for group in by_color.values()), "each color must have two objects"

    color_groups = []
    for group in by_color.values():
        left_edges = [component["bbox"][1] for component in group]
        assert len(set(left_edges)) == len(left_edges), "object left-edge order must be unique"
        color_groups.append(
            {
                "components": group,
                "left": min(left_edges),
                "width": max(bbox_w(component["bbox"]) for component in group),
            }
        )

    assert len({group["left"] for group in color_groups}) == len(color_groups), "color column order must be unique"
    color_groups.sort(key=lambda group: group["left"])
    packed_width = sum(group["width"] for group in color_groups) + len(color_groups) - 1
    assert packed_width <= w, "packed columns must fit in the input width"

    out = np.full_like(g, bg)
    c0 = 0
    for group in color_groups:
        ordered = sorted(group["components"], key=lambda component: component["bbox"][1], reverse=True)
        stack_h = sum(bbox_h(component["bbox"]) for component in ordered) + len(ordered) - 1
        assert stack_h <= h, "packed stack must fit in the input height"

        r = h - stack_h
        for i, component in enumerate(ordered):
            patch = _component_patch(component, bg)
            paste(out, patch, r, c0, transparent=bg)
            r += patch.shape[0]
            if i != len(ordered) - 1:
                r += 1
        assert r == h
        c0 += group["width"] + 1

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette constraints.
GRID_H = 10
GRID_W = 10
GRID_SHAPE = (GRID_H, GRID_W)
BACKGROUND = 0
FOREGROUND_COLORS = np.arange(1, 10)
COLOR_GROUP_COUNT = 2

# Sprite geometry constraints.
SPRITE_HEIGHTS = range(1, 4)
SPRITE_WIDTHS = range(2, 6)
EXCLUDED_SPRITE_DIMS = {(1, 2)}
MIN_SPRITE_CELLS = 3
MAX_SPRITE_CELLS = 8

# Packing and ordering constraints.
OBJECT_GAP = 1
GROUP_GAP = 1
LEFT_EDGE_ORDER_STEP = 1
MIN_STACK_HEIGHT = 4
MAX_STACK_HEIGHT = 7
MIN_MANHATTAN_GAP = 2

# Retry limits.
GENERATION_RETRIES = 300


def _gen_choice(options):
    return options[int(np.random.randint(len(options)))]


def _gen_group_buckets():
    buckets = {}
    for h0 in SPRITE_HEIGHTS:
        for w0 in SPRITE_WIDTHS:
            if (h0, w0) in EXCLUDED_SPRITE_DIMS:
                continue
            for h1 in SPRITE_HEIGHTS:
                for w1 in SPRITE_WIDTHS:
                    if (h1, w1) in EXCLUDED_SPRITE_DIMS:
                        continue
                    stack_h = h0 + h1 + OBJECT_GAP
                    if not MIN_STACK_HEIGHT <= stack_h <= MAX_STACK_HEIGHT:
                        continue
                    group_w = max(w0, w1)
                    buckets.setdefault((group_w, stack_h), []).append(((h0, w0), (h1, w1)))
    return buckets


GROUP_DIM_BUCKETS = _gen_group_buckets()
GROUP_WIDTH_PAIRS = [(a, b) for a in SPRITE_WIDTHS for b in SPRITE_WIDTHS if a + b + GROUP_GAP <= GRID_W]


def _gen_sprite(h, w, color):
    color = int(color)
    cells = {(int(np.random.randint(h)), c) for c in range(w)}
    spine_c = int(np.random.randint(w))
    cells.update((r, spine_c) for r in range(h))
    area = h * w
    min_cells = max(MIN_SPRITE_CELLS, (area + 1) // 2, len(cells))
    max_cells = min(MAX_SPRITE_CELLS, area)
    target = int(np.random.randint(min_cells, max_cells + 1))

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(_gen_choice(frontier))

    patch = np.full((h, w), BACKGROUND, dtype=int)
    coords = np.asarray(list(cells), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = color
    return patch


def _gen_make_groups():
    group_widths = _gen_choice(GROUP_WIDTH_PAIRS)
    colors = np.random.choice(FOREGROUND_COLORS, size=COLOR_GROUP_COUNT, replace=False)
    groups = []

    for color, group_w in zip(colors, group_widths):
        stack_heights = [s for (w, s) in GROUP_DIM_BUCKETS if w == group_w]
        stack_h = _gen_choice(stack_heights)
        dims = _gen_choice(GROUP_DIM_BUCKETS[(group_w, stack_h)])
        patches = [_gen_sprite(h, w, int(color)) for h, w in dims]
        groups.append({"color": int(color), "patches": patches, "width": group_w})
    return groups


def _gen_build_output(groups):
    out = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    c0 = 0
    for group in groups:
        stack_h = sum(patch.shape[0] for patch in group["patches"]) + OBJECT_GAP
        r = GRID_H - stack_h
        for i, patch in enumerate(group["patches"]):
            paste(out, patch, r, c0, transparent=BACKGROUND)
            r += patch.shape[0]
            if i != len(group["patches"]) - 1:
                r += OBJECT_GAP
        c0 += group["width"] + GROUP_GAP
    return out


def _gen_abs_coords(patch, r0, c0):
    coords = np.argwhere(patch != BACKGROUND)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _gen_far_enough(coords, placed):
    if not placed:
        return True
    other = np.vstack([item["coords"] for item in placed])
    d = np.abs(coords[:, None, :] - other[None, :, :]).sum(axis=2)
    return bool(np.all(d >= MIN_MANHATTAN_GAP))


def _gen_place_items(items):
    def rec(remaining, placed):
        if not remaining:
            return placed

        best_i = None
        best_choices = None
        for i, item in enumerate(remaining):
            patch = item["patch"]
            choices = []
            for r in range(0, GRID_H - patch.shape[0] + 1):
                coords = _gen_abs_coords(patch, r, item["c"])
                if _gen_far_enough(coords, placed):
                    choices.append((r, coords))
            if best_choices is None or len(choices) < len(best_choices):
                best_i = i
                best_choices = choices
        if not best_choices:
            return None

        order = np.random.permutation(len(best_choices))
        item = remaining[best_i]
        rest = remaining[:best_i] + remaining[best_i + 1 :]
        for choice_i in order:
            r, coords = best_choices[int(choice_i)]
            placed_item = {"patch": item["patch"], "r": int(r), "c": item["c"], "coords": coords}
            result = rec(rest, placed + [placed_item])
            if result is not None:
                return result
        return None

    return rec(items, [])


def _gen_column_options(patches):
    top0, bottom0, top1, bottom1 = patches
    column_options = []
    for left0 in range(0, GRID_W - bottom0.shape[1] + 1):
        for right0 in range(left0 + LEFT_EDGE_ORDER_STEP, GRID_W - top0.shape[1] + 1):
            for left1 in range(left0 + LEFT_EDGE_ORDER_STEP, GRID_W - bottom1.shape[1] + 1):
                for right1 in range(left1 + LEFT_EDGE_ORDER_STEP, GRID_W - top1.shape[1] + 1):
                    column_options.append((left0, right0, left1, right1))
    return column_options


def _gen_items_for_columns(patches, columns):
    top0, bottom0, top1, bottom1 = patches
    left0, right0, left1, right1 = columns
    return [
        {"patch": top0, "c": right0},
        {"patch": bottom0, "c": left0},
        {"patch": top1, "c": right1},
        {"patch": bottom1, "c": left1},
    ]


def _gen_scatter_input(groups):
    patches = (*groups[0]["patches"], *groups[1]["patches"])
    column_options = _gen_column_options(patches)
    for option_i in np.random.permutation(len(column_options)):
        items = _gen_items_for_columns(patches, column_options[int(option_i)])
        placed = _gen_place_items(items)
        if placed is None:
            continue

        inp = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
        for item in placed:
            paste(inp, item["patch"], item["r"], item["c"], transparent=BACKGROUND)
        return inp
    return None


def _gen_attempt():
    groups = _gen_make_groups()
    out = _gen_build_output(groups)
    inp = _gen_scatter_input(groups)
    if inp is None:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid 4acc7107 input")
