from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "arc_utils_051_3", str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py"))
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w

GRID_SIZE = 16
QUAD_SIZE = GRID_SIZE // 2
BG = Color.BLACK
CONTROL = Color.RED
OBJECT_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_ATTEMPTS = 400

RECT_DIMS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
IRREGULAR_MASKS = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
]


def _quadrant_slices(shape: tuple[int, int], index: int) -> tuple[slice, slice]:
    half_r = shape[0] // 2
    half_c = shape[1] // 2
    row_slice = slice(0, half_r) if index < 2 else slice(half_r, shape[0])
    col_slice = slice(0, half_c) if index % 2 == 0 else slice(half_c, shape[1])
    return row_slice, col_slice


def _quadrant_index_from_coord(shape: tuple[int, int], row: int, col: int) -> int:
    half_r = shape[0] // 2
    half_c = shape[1] // 2
    return (2 if row >= half_r else 0) + (1 if col >= half_c else 0)


def _opposite_quadrant(index: int) -> int:
    return 3 - int(index)


def _solid_rectangle(obj: dict) -> bool:
    box = obj["bbox"]
    return int(obj["area"]) == bbox_h(box) * bbox_w(box)


def _non_control_objects(grid: np.ndarray) -> list[dict]:
    colors = [color for color in np.unique(grid) if int(color) not in {BG, CONTROL}]
    return find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[int(color) for color in colors],
        return_format="dict",
    )


def _expected_output(grid: np.ndarray) -> np.ndarray:
    control = np.argwhere(grid == CONTROL)
    assert control.shape == (1, 2)
    source_idx = _quadrant_index_from_coord(grid.shape, int(control[0, 0]), int(control[0, 1]))
    row_slice, col_slice = _quadrant_slices(grid.shape, source_idx)
    source = grid[row_slice, col_slice]
    out = np.full_like(grid, BG)
    for obj in _non_control_objects(source):
        if not _solid_rectangle(obj):
            continue
        rows = obj["coords"][:, 0] + row_slice.start
        cols = obj["coords"][:, 1] + col_slice.start
        out[grid.shape[0] - 1 - rows, grid.shape[1] - 1 - cols] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the control-pixel quadrant, keep only its solid rectangles, and rotate them 180 degrees into the opposite quadrant.

    Input structure:
    - The grid has an even height and width and is split into four equal quadrants by the canvas center.
    - Exactly one isolated red pixel marks the source quadrant.
    - Each quadrant may contain several separated monochrome 4-connected objects on a black background.

    Rule:
    - Find the quadrant containing the red control pixel.
    - Within that quadrant, test each non-red object for being a filled axis-aligned rectangle.
    - Erase every object that fails the rectangle test, erase the original source contents, and erase all other quadrants.
    - Rotate every surviving rectangle 180 degrees around the full-grid center, mapping each pixel `(r, c)` to `(H-1-r, W-1-c)` without changing color.

    Key insight:
    - The red singleton chooses where to look, the rectangle test chooses what survives, and the 180-degree global rotation chooses where the survivors reappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0

    return _expected_output(g)


def _make_rect_patch(color: int) -> np.ndarray:
    height, width = random.choice(RECT_DIMS)
    return np.full((height, width), int(color), dtype=int)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    turns = random.randrange(4)
    return np.rot90(out, turns)


def _make_irregular_patch(color: int) -> np.ndarray:
    mask = _transform_mask(random.choice(IRREGULAR_MASKS))
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _place_patch(quad: np.ndarray, occupancy: np.ndarray, patch: np.ndarray) -> bool:
    try:
        row, col = random_free_location(
            occupancy,
            patch,
            transparent=BG,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return False
    paste(quad, patch, row, col, transparent=BG)
    paste(occupancy, patch, row, col, transparent=BG)
    return True


def _build_quadrant(
    rect_count: int,
    irregular_count: int,
    colors: list[int],
    *,
    add_control: bool = False,
    blocked_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    for _ in range(120):
        quad = np.full((QUAD_SIZE, QUAD_SIZE), BG, dtype=int)
        occupancy = np.full_like(quad, BG)
        if blocked_mask is not None:
            occupancy[blocked_mask] = Color.PAD

        kinds = ["rect"] * rect_count + ["irregular"] * irregular_count
        random.shuffle(kinds)
        ok = True
        for kind, color in zip(kinds, colors):
            patch = _make_rect_patch(color) if kind == "rect" else _make_irregular_patch(color)
            if not _place_patch(quad, occupancy, patch):
                ok = False
                break
        if not ok:
            continue

        if add_control:
            control_patch = np.array([[CONTROL]], dtype=int)
            if not _place_patch(quad, occupancy, control_patch):
                continue

        return quad
    return None


def _source_rectangle_layer(grid: np.ndarray) -> np.ndarray:
    control = np.argwhere(grid == CONTROL)
    source_idx = _quadrant_index_from_coord(grid.shape, int(control[0, 0]), int(control[0, 1]))
    row_slice, col_slice = _quadrant_slices(grid.shape, source_idx)
    source = grid[row_slice, col_slice]
    layer = np.full_like(source, BG)
    for obj in _non_control_objects(source):
        if _solid_rectangle(obj):
            coords = obj["coords"]
            layer[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return layer


def _valid_control_isolation(grid: np.ndarray) -> bool:
    row, col = map(int, np.argwhere(grid == CONTROL)[0])
    for rr in range(max(0, row - 1), min(grid.shape[0], row + 2)):
        for cc in range(max(0, col - 1), min(grid.shape[1], col + 2)):
            if rr == row and cc == col:
                continue
            if int(grid[rr, cc]) != BG:
                return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if np.count_nonzero(grid == CONTROL) != 1:
        return False
    if not _valid_control_isolation(grid):
        return False

    objects = _non_control_objects(grid)
    if not objects:
        return False
    for obj in objects:
        if not (3 <= int(obj["area"]) <= 8):
            return False
        rows = obj["coords"][:, 0]
        cols = obj["coords"][:, 1]
        quadrants = {_quadrant_index_from_coord(grid.shape, int(r), int(c)) for r, c in zip(rows, cols)}
        if len(quadrants) != 1:
            return False

    control = np.argwhere(grid == CONTROL)[0]
    source_idx = _quadrant_index_from_coord(grid.shape, int(control[0]), int(control[1]))
    target_idx = _opposite_quadrant(source_idx)

    source_slice = _quadrant_slices(grid.shape, source_idx)
    target_slice = _quadrant_slices(grid.shape, target_idx)
    source_objects = _non_control_objects(grid[source_slice[0], source_slice[1]])
    if not source_objects:
        return False
    source_rects = [obj for obj in source_objects if _solid_rectangle(obj)]
    source_nonrects = [obj for obj in source_objects if not _solid_rectangle(obj)]
    if not source_rects or not source_nonrects:
        return False

    outside_rects = []
    for idx in range(4):
        if idx == source_idx:
            continue
        row_slice, col_slice = _quadrant_slices(grid.shape, idx)
        quad_objects = _non_control_objects(grid[row_slice, col_slice])
        if not quad_objects:
            return False
        outside_rects.extend(obj for obj in quad_objects if _solid_rectangle(obj))
    if not outside_rects:
        return False

    rect_layer = _source_rectangle_layer(grid)
    if np.array_equal(rect_layer, np.rot90(rect_layer, 2)):
        return False

    expected = _expected_output(grid)
    if not np.any(expected != BG):
        return False
    target_input = grid[target_slice[0], target_slice[1]]
    target_output = expected[target_slice[0], target_slice[1]]
    if np.any((target_output != BG) & (target_input != BG)):
        return False
    if np.any(expected[: target_slice[0].start, :] != BG):
        return False
    if np.any(expected[target_slice[0].stop :, :] != BG):
        return False
    if np.any(expected[:, : target_slice[1].start] != BG) and target_slice[1].start > 0:
        return False
    if np.any(expected[:, target_slice[1].stop :] != BG) and target_slice[1].stop < grid.shape[1]:
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    source_idx = random.randrange(4)
    target_idx = _opposite_quadrant(source_idx)
    distractor_rect_idx = random.choice([idx for idx in range(4) if idx != source_idx])
    palette = OBJECT_COLORS.copy()
    random.shuffle(palette)
    color_cursor = 0

    source_total = random.choice([2, 3])
    if source_total == 2:
        source_rects, source_irregulars = 1, 1
    else:
        source_rects, source_irregulars = random.choice([(2, 1), (1, 2)])
    source_colors = palette[color_cursor : color_cursor + source_total]
    color_cursor += source_total
    source_quad = _build_quadrant(source_rects, source_irregulars, source_colors, add_control=True)
    if source_quad is None:
        return None

    source_rect_layer = np.full((QUAD_SIZE, QUAD_SIZE), BG, dtype=int)
    for obj in _non_control_objects(source_quad):
        if _solid_rectangle(obj):
            coords = obj["coords"]
            source_rect_layer[coords[:, 0], coords[:, 1]] = int(obj["color"])
    if np.array_equal(source_rect_layer, np.rot90(source_rect_layer, 2)):
        return None

    blocked_target = np.rot90(source_rect_layer != BG, 2)

    quadrants: list[np.ndarray | None] = [None, None, None, None]
    quadrants[source_idx] = source_quad
    non_source_indices = [idx for idx in range(4) if idx != source_idx]
    remaining_slots = len(OBJECT_COLORS) - source_total
    for pos, idx in enumerate(non_source_indices):
        remaining_quadrants = len(non_source_indices) - pos
        max_total = min(2, remaining_slots - (remaining_quadrants - 1))
        if max_total < 1:
            return None
        total = random.randint(1, max_total)
        if idx == distractor_rect_idx:
            rects = 1
            irregulars = total - 1
        else:
            choices = [(0, 1), (1, 0), (1, 1)]
            rects, irregulars = random.choice([pair for pair in choices if sum(pair) == total] or [(0, 1)])
        needed = rects + irregulars
        colors = palette[color_cursor : color_cursor + needed]
        if len(colors) < needed:
            return None
        color_cursor += needed
        remaining_slots -= needed
        blocked = blocked_target if idx == target_idx else None
        quadrants[idx] = _build_quadrant(rects, irregulars, colors, blocked_mask=blocked)
        if quadrants[idx] is None:
            return None

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for idx, quad in enumerate(quadrants):
        assert quad is not None
        row_slice, col_slice = _quadrant_slices(grid.shape, idx)
        grid[row_slice, col_slice] = quad
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid control-quadrant rectangle-rotation input")
