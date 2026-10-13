import random

import numpy as np


BLACK = 0
ALL_COLORS = tuple(range(1, 10))

LEGEND_LENGTH_CHOICES = (6, 7)
LEGEND_LENGTH_WEIGHTS = (3, 2)
GROUP_COUNT_WEIGHTS = {
    1: (1,),
    2: (1, 3),
    3: (1, 3, 2),
}
GROUP_PIXEL_COUNT_RANGE = (3, 6)
BBOX_HEIGHT_RANGE = (3, 8)
BBOX_WIDTH_RANGE = (3, 8)
GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
PLACEMENT_PADDING = 1
PLACEMENT_TRIES = 160
POINT_SAMPLE_TRIES = 200
GENERATION_ATTEMPTS = 400


def _perimeter_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(width):
        cells.append((0, c))
        if height > 1:
            cells.append((height - 1, c))
    for r in range(1, height - 1):
        cells.append((r, 0))
        if width > 1:
            cells.append((r, width - 1))
    return cells


def _sample_constellation_points(
    box_h: int, box_w: int, count: int
) -> list[tuple[int, int]] | None:
    perimeter = _perimeter_cells(box_h, box_w)
    if len(perimeter) < count:
        return None

    for _ in range(POINT_SAMPLE_TRIES):
        points = random.sample(perimeter, count)
        rows = [r for r, _ in points]
        cols = [c for _, c in points]
        if min(rows) != 0 or max(rows) != box_h - 1:
            continue
        if min(cols) != 0 or max(cols) != box_w - 1:
            continue
        return points

    corners = [(0, 0), (0, box_w - 1), (box_h - 1, 0), (box_h - 1, box_w - 1)]
    if count == 3:
        missing = random.randrange(4)
        return [corner for i, corner in enumerate(corners) if i != missing]

    extras = [cell for cell in perimeter if cell not in corners]
    if len(extras) < count - 4:
        return None
    return corners + random.sample(extras, count - 4)


def _boxes_too_close(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _draw_outline(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _legend_colors(grid: np.ndarray) -> list[int]:
    assert grid.shape[0] >= 3, "expected legend row, separator row, and main area"
    legend: list[int] = []
    for value in grid[0]:
        color = int(value)
        if color == BLACK:
            break
        legend.append(color)
    assert legend, "expected a non-empty legend on the top row"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    assert np.all(grid[0, len(legend) :] == BLACK), "legend must be a left-aligned band"
    assert np.all(grid[1] == BLACK), "expected a black separator row below the legend"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """Draw each same-color constellation's bounding box and fill it by count via the legend.

    Input structure:
    - Background is black.
    - The top row contains a left-aligned legend band of distinct 1-pixel colors, followed by black cells.
    - The second row is entirely black and separates the legend from the scene.
    - Below the separator, each non-legend color appears as one scattered constellation of 3 to 6 pixels.

    Rule:
    - Treat all pixels of the same non-legend color as one constellation, even if they are disconnected.
    - Compute that constellation's minimal axis-aligned bounding box and draw its rectangle outline in the constellation's original color.
    - Count the constellation's pixels; if the count is K, use the K-th legend color as the fill color.
    - Fill every black cell strictly inside the rectangle with that legend color, while leaving the legend unchanged.

    Key insight:
    - The scattered pixels do not define lines between themselves; they define a bounding rectangle, and the interior color is determined indirectly by pixel count through the visible legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend = _legend_colors(g)
    scene = g[2:, :]
    scene_colors = sorted(set(map(int, np.unique(scene))) - {BLACK} - set(legend))
    assert scene_colors, "expected at least one constellation below the legend"

    out = g.copy()
    for color in scene_colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        count = int(coords.shape[0])
        assert 1 <= count <= len(legend), "constellation size must index into the legend"

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        _draw_outline(out, int(r0), int(c0), int(r1), int(c1), int(color))

        if r1 - r0 >= 2 and c1 - c0 >= 2:
            fill_color = int(legend[count - 1])
            interior_mask = g[r0 + 1 : r1, c0 + 1 : c1] == BLACK
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior_mask] = fill_color

    return out


def _sample_group_count(max_groups: int) -> int:
    choices = list(range(1, max_groups + 1))
    weights = GROUP_COUNT_WEIGHTS[max_groups]
    return random.choices(choices, weights=weights, k=1)[0]


def _place_specs(
    grid_h: int, grid_w: int, specs: list[dict]
) -> list[dict] | None:
    placed: list[dict] = []
    occupied_boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: item["box_h"] * item["box_w"], reverse=True):
        box_h = int(spec["box_h"])
        box_w = int(spec["box_w"])
        row_min = 3
        row_max = grid_h - box_h - 1
        col_min = 1
        col_max = grid_w - box_w - 1
        if row_max < row_min:
            row_max = grid_h - box_h
        if col_max < col_min:
            col_max = grid_w - box_w
        if row_max < row_min or col_max < col_min:
            return None

        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(row_min, row_max)
            c0 = random.randint(col_min, col_max)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_boxes_too_close(bbox, other, PLACEMENT_PADDING) for other in occupied_boxes):
                continue
            placed_spec = dict(spec)
            placed_spec["bbox"] = bbox
            placed.append(placed_spec)
            occupied_boxes.append(bbox)
            break
        else:
            return None

    return placed


def _validate_generated_case(grid: np.ndarray, specs: list[dict], legend: list[int]) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[:2], grid[:2]):
        return False

    for spec in specs:
        color = int(spec["color"])
        count = int(spec["count"])
        r0, c0, r1, c1 = spec["bbox"]
        fill = int(legend[count - 1])

        border = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        if not np.all(border[0, :] == color):
            return False
        if not np.all(border[-1, :] == color):
            return False
        if not np.all(border[:, 0] == color):
            return False
        if not np.all(border[:, -1] == color):
            return False

        if r1 - r0 >= 2 and c1 - c0 >= 2:
            original_interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            output_interior = out[r0 + 1 : r1, c0 + 1 : c1]
            if not np.all(output_interior[original_interior == BLACK] == fill):
                return False
            if not np.array_equal(
                output_interior[original_interior != BLACK],
                original_interior[original_interior != BLACK],
            ):
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    legend_len = random.choices(LEGEND_LENGTH_CHOICES, weights=LEGEND_LENGTH_WEIGHTS, k=1)[0]
    max_groups = min(3, len(ALL_COLORS) - legend_len)
    if max_groups < 1:
        return None
    group_count = _sample_group_count(max_groups)

    specs: list[dict] = []
    for _ in range(group_count):
        count = random.randint(*GROUP_PIXEL_COUNT_RANGE)
        box_h = random.randint(*BBOX_HEIGHT_RANGE)
        box_w = random.randint(*BBOX_WIDTH_RANGE)
        points = _sample_constellation_points(box_h, box_w, count)
        if points is None:
            return None
        specs.append(
            {
                "count": count,
                "box_h": box_h,
                "box_w": box_w,
                "points": points,
            }
        )

    max_box_h = max(int(spec["box_h"]) for spec in specs)
    max_box_w = max(int(spec["box_w"]) for spec in specs)
    min_h = max(GRID_HEIGHT_RANGE[0], 5 + max_box_h, 11 + 2 * group_count)
    min_w = max(GRID_WIDTH_RANGE[0], legend_len + 3, max_box_w + 3, 11 + 2 * group_count)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    grid_w = random.randint(min_w, GRID_WIDTH_RANGE[1])
    placed_specs = _place_specs(grid_h, grid_w, specs)
    if placed_specs is None:
        return None

    legend = random.sample(list(ALL_COLORS), legend_len)
    scene_palette = [color for color in ALL_COLORS if color not in legend]
    if len(scene_palette) < group_count:
        return None
    group_colors = random.sample(scene_palette, group_count)
    random.shuffle(group_colors)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[0, :legend_len] = legend

    for spec, color in zip(placed_specs, group_colors):
        r0, c0, _r1, _c1 = spec["bbox"]
        spec["color"] = int(color)
        for dr, dc in spec["points"]:
            grid[r0 + dr, c0 + dc] = int(color)

    if not _validate_generated_case(grid, placed_specs, legend):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid constellation-legend input")
