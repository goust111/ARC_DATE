from __future__ import annotations

import math
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 400
MIN_COLOR_COUNT = 2
MAX_COLOR_COUNT = 5
MIN_OBJECT_AREA = 2
MAX_OBJECT_AREA = 5
MIN_COUNT_VALUE = 1
COUNT_PADDING = 3
PLACEMENT_GAP = 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for row in range(height):
        for col in range(width):
            if visited[row, col] or int(grid[row, col]) == bg:
                continue

            color = int(grid[row, col])
            stack = [(row, col)]
            visited[row, col] = True
            cells: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if visited[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    visited[rr, cc] = True
                    stack.append((rr, cc))

            min_pos = min(cells)
            components.append(
                {
                    "color": color,
                    "area": len(cells),
                    "min_pos": min_pos,
                }
            )

    return components


def _ordered_colors(grid: np.ndarray) -> list[int]:
    bg = _dominant_color(grid)
    components = _find_components(grid, bg)
    assert components, "expected at least one foreground component"

    by_color: dict[int, dict[str, object]] = {}
    for comp in components:
        color = int(comp["color"])
        if color not in by_color:
            by_color[color] = {"count": 0, "area": 0, "min_pos": comp["min_pos"]}
        by_color[color]["count"] = int(by_color[color]["count"]) + 1
        by_color[color]["area"] = int(by_color[color]["area"]) + int(comp["area"])
        by_color[color]["min_pos"] = min(by_color[color]["min_pos"], comp["min_pos"])

    ordered = sorted(
        by_color.items(),
        key=lambda item: (
            -int(item[1]["count"]),
            -int(item[1]["area"]),
            int(item[1]["min_pos"][0]),
            int(item[1]["min_pos"][1]),
        ),
    )
    return [int(color) for color, _info in ordered]


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count monochromatic 4-connected foreground objects for each color and
    redraw the colors as nested square rings from most frequent to least.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = _ordered_colors(g)
    return _ring_output(colors)


def _random_connected_mask(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}

    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True

    mask = np.rot90(mask, random.randrange(4))
    if random.choice([False, True]):
        mask = np.flipud(mask)
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    return np.asarray(mask, dtype=bool)


def _valid_positions(occupied: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = occupied.shape
    mask_h, mask_w = mask.shape
    valid: list[tuple[int, int]] = []

    for row in range(height - mask_h + 1):
        for col in range(width - mask_w + 1):
            r0 = max(0, row - PLACEMENT_GAP)
            r1 = min(height, row + mask_h + PLACEMENT_GAP)
            c0 = max(0, col - PLACEMENT_GAP)
            c1 = min(width, col + mask_w + PLACEMENT_GAP)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((row, col))

    return valid


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    valid = _valid_positions(occupied, mask)
    if not valid:
        return False

    row, col = random.choice(valid)
    mask_h, mask_w = mask.shape
    window = grid[row : row + mask_h, col : col + mask_w]
    window[mask] = int(color)
    occupied[row : row + mask_h, col : col + mask_w][mask] = True
    return True


def _sample_counts(color_count: int) -> list[int]:
    upper = max(MIN_COUNT_VALUE + color_count + COUNT_PADDING, MIN_COUNT_VALUE + color_count)
    counts = random.sample(range(MIN_COUNT_VALUE, upper + 1), color_count)
    return counts


def _canvas_shape(total_area: int, object_total: int) -> tuple[int, int]:
    target_area = max(6 * total_area, (object_total * (MAX_OBJECT_AREA + 2)) ** 2 // 5)
    aspect_num = random.randint(7, 14)
    aspect = aspect_num / 10.0
    height = max(10, math.ceil(math.sqrt(target_area * aspect)))
    width = max(10, math.ceil(target_area / height))
    height += random.randint(0, 4)
    width += random.randint(0, 4)
    return height, width


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    non_bg = [color for color in PALETTE if color != bg]

    color_count = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    colors = random.sample(non_bg, color_count)
    counts = _sample_counts(color_count)
    shared_area = random.randint(MIN_OBJECT_AREA, MAX_OBJECT_AREA)

    total_objects = sum(counts)
    total_area = total_objects * shared_area
    height, width = _canvas_shape(total_area, total_objects)

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    placement_queue: list[int] = []
    for color, count in zip(colors, counts):
        placement_queue.extend([color] * count)
    random.shuffle(placement_queue)

    for color in placement_queue:
        mask = _random_connected_mask(shared_area)
        if not _place_mask(grid, occupied, mask, color):
            return None

    ordered = [color for count, color in sorted(zip(counts, colors), reverse=True)]
    expected = _ring_output(ordered)

    if _dominant_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
