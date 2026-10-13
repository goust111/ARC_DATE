import math
from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords):
    r0, c0, r1, c1 = _bbox(coords)
    shape = frozenset((r - r0, c - c0) for r, c in coords)
    return shape, r1 - r0 + 1, c1 - c0 + 1


def _shape_key(shape, height, width):
    return height, width, tuple(sorted(shape))


def _component_cells(grid, start_r, start_c, seen):
    color = int(grid[start_r, start_c])
    queue = deque([(start_r, start_c)])
    seen.add((start_r, start_c))
    coords = []
    height, width = grid.shape

    while queue:
        r, c = queue.popleft()
        coords.append((r, c))
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))

    r0, c0, r1, c1 = _bbox(coords)
    shape = frozenset((r - r0, c - c0) for r, c in coords)
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "shape": shape,
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "area": len(coords),
    }


def _find_components(grid):
    seen = set()
    components = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) == BG or (r, c) in seen:
                continue
            components.append(_component_cells(grid, r, c, seen))
    return components


def _is_vertical_symmetric(shape, height, width):
    return all((r, width - 1 - c) in shape for r, c in shape)


def _is_horizontal_symmetric(shape, height, width):
    return all((height - 1 - r, c) in shape for r, c in shape)


def _has_axis_symmetry(shape, height, width):
    return _is_vertical_symmetric(shape, height, width) or _is_horizontal_symmetric(shape, height, width)


def _draw_shape(grid, top, left, shape, color):
    for r, c in shape:
        grid[top + r, left + c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Use the unique mirror-symmetric object as a stamp template and repaint every other equal-area object with it.

    Input structure:
    - The background is black.
    - The grid contains several separated 4-connected monochrome foreground objects, each in a distinct color.
    - All foreground objects have the same pixel count.
    - Exactly one object's shape is mirror-symmetric across its own horizontal or vertical bbox axis.

    Rule:
    - Extract every foreground object's normalized shape relative to its bbox top-left corner.
    - Identify the unique object whose shape is horizontally or vertically mirror-symmetric; this is the template.
    - Keep that template object unchanged.
    - For every other object, erase its original pixels and redraw the template shape at that object's original bbox top-left anchor, using the object's original color.

    Key insight:
    - Color and size do not distinguish the special object; the only reliable selector is shape symmetry, after which the task becomes shape replacement while preserving each object's own color and anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components

    areas = {component["area"] for component in components}
    assert len(areas) == 1

    symmetric = [
        component
        for component in components
        if _has_axis_symmetry(component["shape"], component["height"], component["width"])
    ]
    assert len(symmetric) == 1
    template = symmetric[0]

    out = g.copy()
    for component in components:
        if component is template:
            continue
        for r, c in component["coords"]:
            out[r, c] = BG

    for component in components:
        if component is template:
            continue
        top, left, _, _ = component["bbox"]
        _draw_shape(out, top, left, template["shape"], component["color"])

    return out


def _reflect(axis, height, width, cell):
    r, c = cell
    if axis == "vertical":
        return r, width - 1 - c
    return height - 1 - r, c


def _grow_connected_shape(area, max_height, max_width):
    if area > max_height * max_width:
        return None

    cells = {(int(RNG.integers(max_height)), int(RNG.integers(max_width)))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < max_height and 0 <= nc < max_width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        frontier_list = list(frontier)
        cells.add(frontier_list[int(RNG.integers(len(frontier_list)))])

    return _normalize_shape(cells)


def _grow_symmetric_shape(axis, canvas_height, canvas_width, target_area):
    if axis == "vertical" and canvas_width % 2 == 0:
        return None
    if axis == "horizontal" and canvas_height % 2 == 0:
        return None
    if target_area > canvas_height * canvas_width:
        return None

    if axis == "vertical":
        axis_cells = [(r, canvas_width // 2) for r in range(canvas_height)]
    else:
        axis_cells = [(canvas_height // 2, c) for c in range(canvas_width)]

    cells = {axis_cells[int(RNG.integers(len(axis_cells)))]}

    while len(cells) < target_area:
        orbit_choices = {}
        for r, c in list(cells):
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < canvas_height and 0 <= nc < canvas_width):
                    continue
                orbit = frozenset({(nr, nc), _reflect(axis, canvas_height, canvas_width, (nr, nc))})
                new_cells = orbit - cells
                if not new_cells:
                    continue
                size = len(new_cells)
                if len(cells) + size <= target_area:
                    orbit_choices[orbit] = new_cells
        if not orbit_choices:
            return None
        orbits = list(orbit_choices.items())
        new_cells = orbits[int(RNG.integers(len(orbits)))][1]
        cells |= set(new_cells)

    shape, height, width = _normalize_shape(cells)
    if len(shape) != target_area:
        return None
    if axis == "vertical":
        if not _is_vertical_symmetric(shape, height, width) or _is_horizontal_symmetric(shape, height, width):
            return None
    else:
        if not _is_horizontal_symmetric(shape, height, width) or _is_vertical_symmetric(shape, height, width):
            return None
    if height < 2 or width < 2:
        return None
    return shape, height, width


def _generate_template(axis):
    for _ in range(200):
        if axis == "vertical":
            canvas_height = int(RNG.integers(3, 7))
            canvas_width = int(RNG.choice([5, 7]))
        else:
            canvas_height = int(RNG.choice([5, 7]))
            canvas_width = int(RNG.integers(3, 7))
        min_area = max(5, (canvas_height + canvas_width) // 2)
        max_area = min(canvas_height * canvas_width - 1, 10)
        if min_area > max_area:
            continue
        target_area = int(RNG.integers(min_area, max_area + 1))
        shape_data = _grow_symmetric_shape(axis, canvas_height, canvas_width, target_area)
        if shape_data is not None:
            return shape_data
    return None


def _generate_nonsymmetric_shape(area, max_height, max_width, forbidden):
    for _ in range(300):
        shape_data = _grow_connected_shape(area, max_height, max_width)
        if shape_data is None:
            continue
        shape, height, width = shape_data
        if height < 2 or width < 2:
            continue
        if _has_axis_symmetry(shape, height, width):
            continue
        key = _shape_key(shape, height, width)
        if key in forbidden:
            continue
        return shape_data
    return None


def _layout_candidates(slot_height, slot_width, count):
    candidates = []
    gap = int(RNG.integers(2, 4))
    margin = int(RNG.integers(1, 3))
    for cols in range(2, min(3, count) + 1):
        rows = math.ceil(count / cols)
        height = 2 * margin + rows * slot_height + (rows - 1) * gap
        width = 2 * margin + cols * slot_width + (cols - 1) * gap
        if height <= 30 and width <= 30:
            candidates.append((rows, cols, gap, margin, height, width))
    return candidates


def _place_shape(grid, slot_top, slot_left, slot_height, slot_width, shape_data, template_height, template_width, color):
    shape, height, width = shape_data
    max_off_r = slot_height - max(height, template_height)
    max_off_c = slot_width - max(width, template_width)
    if max_off_r < 0 or max_off_c < 0:
        return False
    off_r = int(RNG.integers(max_off_r + 1))
    off_c = int(RNG.integers(max_off_c + 1))
    _draw_shape(grid, slot_top + off_r, slot_left + off_c, shape, color)
    return True


def _validate_generated(grid):
    components = _find_components(grid)
    if len(components) < 3:
        return False
    if len({component["color"] for component in components}) != len(components):
        return False
    if len({component["area"] for component in components}) != 1:
        return False
    symmetric_count = sum(
        _has_axis_symmetry(component["shape"], component["height"], component["width"])
        for component in components
    )
    return symmetric_count == 1


def _gen_attempt():
    axis = str(RNG.choice(["vertical", "horizontal"]))
    template_data = _generate_template(axis)
    if template_data is None:
        return None
    template_shape, template_height, template_width = template_data
    area = len(template_shape)

    slot_height = template_height + int(RNG.integers(0, 3))
    slot_width = template_width + int(RNG.integers(0, 3))
    slot_height = max(slot_height, 4)
    slot_width = max(slot_width, 4)
    if slot_height > 8 or slot_width > 8:
        return None

    object_count = int(RNG.integers(3, 7))
    if object_count > len(PALETTE):
        return None

    forbidden = {_shape_key(template_shape, template_height, template_width)}
    shapes = [template_data]
    for _ in range(object_count - 1):
        shape_data = _generate_nonsymmetric_shape(area, slot_height, slot_width, forbidden)
        if shape_data is None:
            return None
        shape, height, width = shape_data
        forbidden.add(_shape_key(shape, height, width))
        shapes.append(shape_data)

    layout = _layout_candidates(slot_height, slot_width, object_count)
    if not layout:
        return None
    rows, cols, gap, margin, grid_height, grid_width = layout[int(RNG.integers(len(layout)))]

    grid = np.zeros((grid_height, grid_width), dtype=int)
    colors = list(RNG.choice(PALETTE, size=object_count, replace=False))
    template_index = int(RNG.integers(object_count))
    shape_order = [None] * object_count
    shape_order[template_index] = template_data

    other_shapes = list(shapes[1:])
    other_shapes = [other_shapes[i] for i in RNG.permutation(len(other_shapes))]
    fill_index = 0
    for idx in range(object_count):
        if shape_order[idx] is None:
            shape_order[idx] = other_shapes[fill_index]
            fill_index += 1

    for idx, (shape_data, color) in enumerate(zip(shape_order, colors)):
        row = idx // cols
        col = idx % cols
        slot_top = margin + row * (slot_height + gap)
        slot_left = margin + col * (slot_width + gap)
        ok = _place_shape(
            grid,
            slot_top,
            slot_left,
            slot_height,
            slot_width,
            shape_data,
            template_height,
            template_width,
            int(color),
        )
        if not ok:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        assert out.shape == grid.shape
        return grid
    raise RuntimeError("failed to generate a valid input")
