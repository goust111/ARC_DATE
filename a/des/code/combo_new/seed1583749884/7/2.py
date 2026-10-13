import random
from collections import deque

import numpy as np


AZURE = 8
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [color for color in range(10) if color != AZURE]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            seen[row, col] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(coords)

    return components


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    objects: list[dict[str, object]] = []

    for coords in _component_coords(g != bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "template": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "area": len(coords),
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each cropped foreground object as many times as its own area.

    Input structure:
    - The background is azure.
    - The grid contains 2-4 separated 4-connected monochrome foreground objects in distinct colors.
    - Each object has area 2-5, and its bounding box may include background cells around the shape.

    Rule:
    - Create a new canvas filled with the input background color.
    - For each foreground object, crop its bounding box to obtain a stamp template.
    - Process the objects from left to right according to their input positions.
    - If an object has N foreground pixels, stamp that cropped template N times side by side with no gap inside the group.
    - Leave exactly two background columns between consecutive object groups and bottom-align every stamp.

    Key insight:
    - The object's area is self-referential: the same pixels define both the stamp's shape and the number of times that stamp must be repeated.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _extract_objects(g)
    if not objects:
        return np.full((0, 0), bg, dtype=int)

    out_height = max(int(obj["template"].shape[0]) for obj in objects)
    out_width = sum(int(obj["area"]) * int(obj["template"].shape[1]) for obj in objects) + 2 * (len(objects) - 1)
    out = np.full((out_height, out_width), bg, dtype=int)

    cursor = 0
    for index, obj in enumerate(objects):
        template = np.asarray(obj["template"], dtype=int)
        repeats = int(obj["area"])
        height, width = template.shape
        top = out_height - height
        for _ in range(repeats):
            out[top : top + height, cursor : cursor + width] = template
            cursor += width
        if index != len(objects) - 1:
            cursor += 2

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    normalized = {(row - min_row, col - min_col) for row, col in cells}
    height = max(row for row, _ in normalized) + 1
    width = max(col for _, col in normalized) + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, col in normalized:
        mask[row, col] = True
    return mask


def _random_connected_mask(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for row, col in cells:
            for dr, dc in DIR4:
                nbr = (row + dr, col + dc)
                if nbr not in cells:
                    frontier.add(nbr)
        cells.add(random.choice(tuple(frontier)))
    return _mask_from_cells(cells)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    current = mask.copy()
    for _ in range(4):
        current = _trim_mask(current)
        signature = (tuple(map(int, current.shape)), current.tobytes())
        if signature not in seen:
            rotations.append(current.copy())
            seen.add(signature)
        current = np.rot90(current)
    return rotations


def _sample_oriented_mask(area: int, max_contribution: int) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for _ in range(40):
        base = _random_connected_mask(area)
        for rotated in _unique_rotations(base):
            if area * int(rotated.shape[1]) <= max_contribution:
                candidates.append(rotated.copy())
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_objects() -> list[dict[str, object]] | None:
    count = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, count)
    remaining_width_budget = MAX_GRID_SIZE - 2 * (count - 1)
    objects: list[dict[str, object]] = []

    for index, color in enumerate(colors):
        remaining_objects = count - index - 1
        min_future_budget = 2 * remaining_objects
        max_contribution = remaining_width_budget - min_future_budget
        if max_contribution < 2:
            return None

        area_options = list(range(2, min(5, max_contribution) + 1))
        random.shuffle(area_options)
        candidates: list[tuple[int, np.ndarray]] = []

        for area in area_options:
            mask = _sample_oriented_mask(area, max_contribution)
            if mask is not None:
                candidates.append((area, mask))

        if not candidates:
            return None

        area, mask = random.choice(candidates)
        template = np.where(mask, color, AZURE).astype(int)
        objects.append({"area": area, "template": template})
        remaining_width_budget -= area * int(template.shape[1])

    return objects


def _expected_output(objects: list[dict[str, object]], bg: int) -> np.ndarray:
    out_height = max(int(obj["template"].shape[0]) for obj in objects)
    out_width = sum(int(obj["area"]) * int(obj["template"].shape[1]) for obj in objects) + 2 * (len(objects) - 1)
    out = np.full((out_height, out_width), bg, dtype=int)

    cursor = 0
    for index, obj in enumerate(objects):
        template = np.asarray(obj["template"], dtype=int)
        repeats = int(obj["area"])
        height, width = template.shape
        top = out_height - height
        for _ in range(repeats):
            out[top : top + height, cursor : cursor + width] = template
            cursor += width
        if index != len(objects) - 1:
            cursor += 2

    return out


def _build_input_grid(objects: list[dict[str, object]], bg: int) -> np.ndarray | None:
    widths = [int(obj["template"].shape[1]) for obj in objects]
    heights = [int(obj["template"].shape[0]) for obj in objects]
    total_area = sum(int(obj["area"]) for obj in objects)

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gaps = [random.randint(2, 4) for _ in range(len(objects) - 1)]
    min_width = left_margin + sum(widths) + sum(gaps) + right_margin
    if min_width > MAX_GRID_SIZE:
        return None

    slack = random.randint(0, min(6, MAX_GRID_SIZE - min_width))
    slots = [left_margin, *gaps, right_margin]
    for _ in range(slack):
        slots[random.randrange(len(slots))] += 1

    grid_width = sum(widths) + sum(slots)
    max_height = max(heights)
    grid_height = random.randint(max_height + 2, min(MAX_GRID_SIZE, max_height + 8))
    if grid_width > MAX_GRID_SIZE or grid_height > MAX_GRID_SIZE:
        return None

    if grid_height * grid_width - total_area <= total_area:
        return None

    grid = np.full((grid_height, grid_width), bg, dtype=int)
    cursor = slots[0]

    for index, obj in enumerate(objects):
        template = np.asarray(obj["template"], dtype=int)
        height, width = template.shape
        row = random.randint(0, grid_height - height)
        grid[row : row + height, cursor : cursor + width] = template
        cursor += width
        if index < len(objects) - 1:
            cursor += slots[index + 1]

    if np.count_nonzero(grid != bg) != total_area:
        return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = AZURE
    objects = _sample_objects()
    if not objects:
        return None

    expected = _expected_output(objects, bg)
    if max(expected.shape) > MAX_GRID_SIZE:
        return None

    grid = _build_input_grid(objects, bg)
    if grid is None:
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
