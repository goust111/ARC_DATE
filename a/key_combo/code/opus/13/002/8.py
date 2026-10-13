from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TURN_CLOCKWISE = {
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
    (-1, 0): (0, 1),
}
COLOR_POOL = tuple(range(1, 10))
OBJECT_COUNT_RANGE = (2, 4)
CELL_COUNT_RANGE = (3, 8)
INPUT_HEIGHT_RANGE = (13, 19)
INPUT_WIDTH_RANGE = (13, 19)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox = _bbox_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "height": bbox[2] - bbox[0] + 1,
                    "width": bbox[3] - bbox[1] + 1,
                }
            )

    return components


def _draw_spiral(height: int, width: int, color: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=int)
    r = c = 0
    dr, dc = 0, 1
    canvas[r, c] = int(color)

    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < height and 0 <= nc < width):
            dr, dc = TURN_CLOCKWISE[(dr, dc)]
            continue

        if canvas[nr, nc] != 0:
            break

        rr, cc = r + 2 * dr, c + 2 * dc
        if 0 <= rr < height and 0 <= cc < width and canvas[rr, cc] != 0:
            dr, dc = TURN_CLOCKWISE[(dr, dc)]
            continue

        r, c = nr, nc
        canvas[r, c] = int(color)

    return canvas


def _compose_spirals(components: list[dict]) -> np.ndarray:
    assert components, "expected at least one foreground component"

    items = []
    seen_areas = set()
    for comp in components:
        area = int(comp["height"]) * int(comp["width"])
        assert area not in seen_areas, "bounding-box areas must be unique"
        seen_areas.add(area)
        items.append(
            (
                area,
                _draw_spiral(int(comp["height"]), int(comp["width"]), int(comp["color"])),
            )
        )

    items.sort(key=lambda item: item[0])
    max_height = max(canvas.shape[0] for _area, canvas in items)
    total_width = sum(canvas.shape[1] for _area, canvas in items) + len(items) - 1
    out = np.zeros((max_height, total_width), dtype=int)

    cursor = 0
    for index, (_area, canvas) in enumerate(items):
        h, w = canvas.shape
        out[:h, cursor : cursor + w] = canvas
        cursor += w
        if index != len(items) - 1:
            cursor += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each foreground object's bounding-box size and render its color as a spiral strip.

    Input structure:
    - The grid is a black background containing 2-4 separated monochrome foreground objects.
    - Each foreground object is a 4-connected component with one non-zero color.
    - The component's exact pixel shape is arbitrary, but its bounding-box height and width are meaningful.

    Rule:
    - Find every 4-connected monochrome foreground component and measure its tight bounding box.
    - For each component, create a black canvas with that bounding-box height and width, then draw a clockwise spiral in the component's color starting from the top-left corner.
    - Sort those spiral canvases by bounding-box area from small to large and concatenate them left-to-right with one black separator column, top-aligning shorter canvases.

    Key insight:
    - The input objects are only carriers for color plus bounding-box dimensions; the original object shapes themselves are discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    return _compose_spirals(components)


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    normalized = np.array([(r - min_r, c - min_c) for r, c in coords], dtype=int)
    order = np.lexsort((normalized[:, 1], normalized[:, 0]))
    return normalized[order]


def _sample_shape_coords(cell_count: int) -> np.ndarray | None:
    coords = {(0, 0)}
    while len(coords) < cell_count:
        frontier = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier.difference_update(coords)
        coords.add(random.choice(tuple(frontier)))

    normalized = _normalize_coords(coords)
    r0, c0, r1, c1 = _bbox_coords(normalized)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    bbox_area = height * width

    if not (2 <= height <= 5 and 2 <= width <= 5):
        return None
    if bbox_area > 25 or cell_count == bbox_area:
        return None
    return normalized


def _coords_to_patch(coords: np.ndarray, color: int) -> np.ndarray:
    _r0, _c0, r1, c1 = _bbox_coords(coords)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _sample_object_specs(count: int) -> list[dict] | None:
    specs: list[dict] = []
    used_bbox_areas: set[int] = set()
    colors = random.sample(COLOR_POOL, count)

    for color in colors:
        shape = None
        for _ in range(100):
            coords = _sample_shape_coords(random.randint(*CELL_COUNT_RANGE))
            if coords is None:
                continue
            bbox = _bbox_coords(coords)
            height = bbox[2] - bbox[0] + 1
            width = bbox[3] - bbox[1] + 1
            bbox_area = height * width
            if bbox_area in used_bbox_areas:
                continue
            shape = coords
            used_bbox_areas.add(bbox_area)
            break
        if shape is None:
            return None

        patch = _coords_to_patch(shape, color)
        specs.append(
            {
                "color": int(color),
                "patch": patch,
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "bbox_area": int(patch.shape[0] * patch.shape[1]),
            }
        )

    return specs


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_objects(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray | None:
    height, width = grid_shape
    grid = np.zeros(grid_shape, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: item["bbox_area"], reverse=True):
        patch = spec["patch"]
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(height - ph + 1)
            for c0 in range(width - pw + 1)
        ]
        random.shuffle(candidates)

        location = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_have_gap(box, other) for other in placed_boxes):
                location = (r0, c0)
                placed_boxes.append(box)
                break

        if location is None:
            return None

        r0, c0 = location
        mask = patch != 0
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

    return grid


def _expected_output_from_specs(specs: list[dict]) -> np.ndarray:
    components = [
        {
            "color": spec["color"],
            "height": spec["height"],
            "width": spec["width"],
        }
        for spec in specs
    ]
    return _compose_spirals(components)


def _generate_candidate() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    specs = _sample_object_specs(count)
    if specs is None:
        return None

    grid_shape = (
        random.randint(*INPUT_HEIGHT_RANGE),
        random.randint(*INPUT_WIDTH_RANGE),
    )
    grid = _place_objects(specs, grid_shape)
    if grid is None:
        return None

    expected = _expected_output_from_specs(specs)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
