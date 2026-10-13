from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, is_contiguous, paste, random_free_location
import itertools
import random


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0])
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the object whose shape matches the colored template inside the grey L-frame.

    Input structure:
    - The black background contains one grey L-frame, with its vertical leg on
      the right and horizontal leg on the bottom of the frame bbox.
    - A single colored template object lies in the area enclosed above-left by
      that L-frame.
    - Other colored objects are scattered outside the framed template region.

    Rule:
    - Extract the normalized mask of the colored template inside the grey
      L-frame bbox.
    - Find the unique outside object with the same normalized mask, ignoring
      its color.
    - Recolor that matched outside object to grey and leave all other pixels
      unchanged.

    Key insight:
    The grey L-frame is a shape selector: it marks the template to search for
    elsewhere in the scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color = Color.GREY
    assert bg != frame_color, "grey is the frame color, not the background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    frames = [comp for comp in comps if int(comp["color"]) == frame_color]
    assert len(frames) == 1, "expected exactly one grey L-frame"

    frame = frames[0]
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
    height = fr1 - fr0 + 1
    width = fc1 - fc0 + 1
    assert height >= 2 and width >= 2, "frame bbox must enclose a template area"

    frame_region = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    expected_frame = np.zeros((height, width), dtype=bool)
    expected_frame[-1, :] = True
    expected_frame[:, -1] = True
    assert np.array_equal(frame_region == frame_color, expected_frame), "expected a bottom-right L-frame"

    template_region = frame_region[:-1, :-1]
    template_colors = set(map(int, np.unique(template_region))) - {bg}
    assert len(template_colors) == 1, "expected one template color inside the frame"
    template_color = template_colors.pop()

    local_template_coords = np.argwhere(template_region == template_color)
    assert local_template_coords.size > 0, "template must contain pixels"
    template_shape = _normalized_shape(local_template_coords)

    matches = []
    frame_bbox = (fr0, fc0, fr1, fc1)
    for comp in comps:
        color = int(comp["color"])
        if color in (bg, frame_color):
            continue
        coords = comp["coords"]
        if np.any(_inside_bbox(coords, frame_bbox)):
            continue
        if _normalized_shape(coords) == template_shape:
            matches.append(comp)

    assert len(matches) == 1, "expected one outside object matching the framed template"

    out = g.copy()
    coords = matches[0]["coords"]
    out[coords[:, 0], coords[:, 1]] = frame_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and frame geometry.
GEN_GRID_SHAPE = (10, 10)
GEN_BACKGROUND = 0
GEN_FRAME_COLOR = Color.GREY
GEN_FRAME_SPAN = 4
GEN_TEMPLATE_SPAN = 3
# Shape and outside-object count ranges.
GEN_SHAPE_SIZE_MIN = 3
GEN_SHAPE_SIZE_MAX = 5
GEN_DISTRACTOR_COUNT_MIN = 2
GEN_DISTRACTOR_COUNT_MAX = 4
# Placement and retry policy.
GEN_PLACEMENT_PADDING = 1
GEN_PLACEMENT_PADDING_CONNECTIVITY = 8
GEN_RETRY_LIMIT = 300
# Color pools.
GEN_NON_FRAME_COLORS = [c for c in range(1, 10) if c != GEN_FRAME_COLOR]


def _gen_all_shapes_by_size():
    """Enumerate normalized contiguous shapes that fit the template box."""
    cells = [(r, c) for r in range(GEN_TEMPLATE_SPAN) for c in range(GEN_TEMPLATE_SPAN)]
    by_size = {}
    for size in range(GEN_SHAPE_SIZE_MIN, GEN_SHAPE_SIZE_MAX + 1):
        seen = set()
        shapes = []
        for combo in itertools.combinations(cells, size):
            patch = np.zeros((GEN_TEMPLATE_SPAN, GEN_TEMPLATE_SPAN), dtype=int)
            for r, c in combo:
                patch[r, c] = 1
            if not is_contiguous(patch, background=GEN_BACKGROUND, connectivity=4):
                continue
            key = _normalized_shape(np.array(combo, dtype=int))
            if key not in seen:
                seen.add(key)
                shapes.append(key)
        by_size[size] = shapes
    return by_size


_GEN_SHAPES_BY_SIZE = _gen_all_shapes_by_size()


def _gen_shape_patch(shape, color):
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _gen_paint_frame(grid):
    grid[GEN_TEMPLATE_SPAN, 0:GEN_FRAME_SPAN] = GEN_FRAME_COLOR
    grid[0:GEN_FRAME_SPAN, GEN_TEMPLATE_SPAN] = GEN_FRAME_COLOR


def _gen_place_template(grid, template_shape, template_color):
    template = _gen_shape_patch(template_shape, template_color)
    tr = random.randint(0, GEN_TEMPLATE_SPAN - template.shape[0])
    tc = random.randint(0, GEN_TEMPLATE_SPAN - template.shape[1])
    paste(grid, template, tr, tc, transparent=GEN_BACKGROUND)


def _gen_distractor_objects(used_shapes, outside_colors):
    """Return unique-shape distractors, or None if the shape pool is exhausted."""
    objects = []
    for _ in range(random.randint(GEN_DISTRACTOR_COUNT_MIN, GEN_DISTRACTOR_COUNT_MAX)):
        shape = _gen_choose_unused_shape(used_shapes)
        if shape is None:
            return None
        objects.append((shape, random.choice(outside_colors), None, False))
    return objects


def _gen_choose_unused_shape(used_shapes):
    """Sample and reserve one normalized shape that is not in used_shapes."""
    possible_sizes = [
        size
        for size in range(GEN_SHAPE_SIZE_MIN, GEN_SHAPE_SIZE_MAX + 1)
        if any(shape not in used_shapes for shape in _GEN_SHAPES_BY_SIZE[size])
    ]
    if not possible_sizes:
        return None
    size = random.choice(possible_sizes)
    choices = [shape for shape in _GEN_SHAPES_BY_SIZE[size] if shape not in used_shapes]
    shape = random.choice(choices)
    used_shapes.add(shape)
    return shape


def _gen_place_outside_objects(output_grid, objects):
    """Render outside objects and return match coords/color, or None on placement failure."""
    occupancy = np.zeros(GEN_GRID_SHAPE, dtype=int)
    occupancy[0:GEN_FRAME_SPAN, 0:GEN_FRAME_SPAN] = 1
    match_coords = None
    match_input_color = None

    for shape, out_color, input_color, is_match in objects:
        patch = _gen_shape_patch(shape, out_color)
        try:
            r0, c0 = random_free_location(
                occupancy,
                patch,
                transparent=GEN_BACKGROUND,
                padding=GEN_PLACEMENT_PADDING,
                padding_connectivity=GEN_PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(output_grid, patch, r0, c0, transparent=GEN_BACKGROUND)
        occ_patch = _gen_shape_patch(shape, 1)
        paste(occupancy, occ_patch, r0, c0, transparent=GEN_BACKGROUND)
        if is_match:
            local = np.argwhere(occ_patch != GEN_BACKGROUND)
            match_coords = local + np.array([r0, c0])
            match_input_color = int(input_color)

    if match_coords is None:
        return None
    return match_coords, match_input_color


def _gen_attempt():
    """Build one validated input grid from the output-first construction."""
    output_grid = np.zeros(GEN_GRID_SHAPE, dtype=int)
    _gen_paint_frame(output_grid)

    template_size = random.randint(GEN_SHAPE_SIZE_MIN, GEN_SHAPE_SIZE_MAX)
    template_shape = random.choice(_GEN_SHAPES_BY_SIZE[template_size])
    template_color = random.choice(GEN_NON_FRAME_COLORS)
    outside_colors = [c for c in GEN_NON_FRAME_COLORS if c != template_color]
    _gen_place_template(output_grid, template_shape, template_color)

    used_shapes = {template_shape}
    objects = _gen_distractor_objects(used_shapes, outside_colors)
    if objects is None:
        return None
    objects.append((template_shape, GEN_FRAME_COLOR, random.choice(outside_colors), True))
    random.shuffle(objects)

    placed_match = _gen_place_outside_objects(output_grid, objects)
    if placed_match is None:
        return None
    match_coords, match_input_color = placed_match

    input_grid = output_grid.copy()
    input_grid[match_coords[:, 0], match_coords[:, 1]] = match_input_color
    try:
        solved = main(input_grid)
    except Exception:
        return None
    if np.array_equal(solved, output_grid):
        return input_grid
    return None


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 63613498 input")
