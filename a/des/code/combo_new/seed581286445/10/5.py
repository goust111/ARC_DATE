from __future__ import annotations

import random

import numpy as np


FENCE_COLOR = 5
GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
STRIPE_COUNT_RANGE = (2, 7)
INTERIOR_HEIGHT_RANGE = (2, 6)
SHAPE_GROWTH_ATTEMPTS = 40


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(cells, dtype=int))

    return components


def _frame_sequence(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]] | None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    border_size = 2 * (height + width) - 4
    if int(coords.shape[0]) != border_size:
        return None

    cells = {tuple(map(int, cell)) for cell in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = map(int, interior.shape)
    if inner_h < 1 or inner_w < 1:
        return None

    sequence: list[int] = []
    seen_colors: set[int] = set()
    for c in range(inner_w):
        column = interior[:, c]
        if np.any(column != int(column[0])):
            return None
        stripe_color = int(column[0])
        if stripe_color == int(color) or stripe_color in seen_colors:
            return None
        seen_colors.add(stripe_color)
        sequence.append(stripe_color)

    return (r0, c0, r1, c1), sequence


def _find_fence_sequence(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int], list[int]]] = []
    for color in np.unique(grid):
        for coords in _components4(grid == int(color)):
            found = _frame_sequence(grid, int(color), coords)
            if found is None:
                continue
            bbox, sequence = found
            r0, c0, r1, c1 = bbox
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            candidates.append((area, len(sequence), bbox, sequence))

    assert candidates, "expected one fenced stripe legend"
    candidates.sort(key=lambda item: (int(item[0]), int(item[1])))
    _, _, bbox, sequence = candidates[-1]
    return bbox, sequence


def _rings_from_sequence(sequence: list[int]) -> np.ndarray:
    size = 2 * len(sequence) - 1
    out = np.empty((size, size), dtype=int)
    for layer, color in enumerate(sequence):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the fenced stripe legend into concentric square rings.

    Input structure:
    - The background is a single dominant color.
    - One monochrome rectangular fence forms a 1-pixel-thick closed border around a compact legend area.
    - Inside that fence, each interior column is a solid vertical stripe, and the stripe colors are all distinct.
    - Outside the fence, there is exactly one connected object in each stripe color, acting only as a distractor.

    Rule:
    - Detect the rectangular fence and read the stripe colors inside it from left to right.
    - Let N be the number of stripes and create an output grid of size (2N-1)×(2N-1).
    - Paint nested square layers from outside to inside using the stripe sequence order.
    - The last stripe color fills the single center cell.

    Key insight:
    - The fence is a legend container: only the left-to-right order of its internal stripes determines the nesting order, while the outer objects are irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bbox, sequence = _find_fence_sequence(g)
    return _rings_from_sequence(sequence)


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(c), int(-r)) for r, c in cells}


def _random_blob(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(area - 1):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (int(r + dr), int(c + dc))
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return cells


def _base_shape() -> set[tuple[int, int]]:
    kind = random.choice(("rectangle", "l", "t", "plus", "stair", "blob"))

    if kind == "rectangle":
        height = random.randint(2, 3)
        width = random.randint(2, 4)
        return {(r, c) for r in range(height) for c in range(width)}

    if kind == "l":
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        return {(r, 0) for r in range(height)} | {(0, c) for c in range(width)}

    if kind == "t":
        width = random.choice((3, 5))
        stem = random.randint(2, 3)
        center = width // 2
        return {(0, c) for c in range(width)} | {(r, center) for r in range(stem)}

    if kind == "plus":
        return {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}

    if kind == "stair":
        steps = random.randint(2, 4)
        cells: set[tuple[int, int]] = set()
        for k in range(steps):
            cells.add((int(k), int(k)))
            cells.add((int(k), int(k + 1)))
        return cells

    area = random.randint(4, 7)
    return _random_blob(area)


def _is_rectangular_frame(cells: tuple[tuple[int, int], ...]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    height = max(rows) + 1
    width = max(cols) + 1
    if height < 3 or width < 3:
        return False
    border = {
        (0, c) for c in range(width)
    } | {
        (height - 1, c) for c in range(width)
    } | {
        (r, 0) for r in range(height)
    } | {
        (r, width - 1) for r in range(height)
    }
    return set(cells) == border


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(SHAPE_GROWTH_ATTEMPTS):
        cells = _base_shape()
        if random.random() < 0.5:
            cells = {(int(r), int(-c)) for r, c in cells}
        for _ in range(random.randrange(4)):
            cells = _rotate(cells)
        shape = _normalize(cells)
        if len(shape) >= 3 and not _is_rectangular_frame(shape):
            return shape
    return ((0, 0), (0, 1), (1, 0))


def _reserve_with_padding(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = mask.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - 1), min(height, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(width, int(c) + 2)):
                mask[int(rr), int(cc)] = True


def _place_objects(
    shape: tuple[int, int],
    fence_bbox: tuple[int, int, int, int],
    colors: list[int],
) -> list[tuple[int, list[tuple[int, int]]]] | None:
    height, width = map(int, shape)
    r0, c0, r1, c1 = fence_bbox
    reserved = np.zeros((height, width), dtype=bool)
    reserved[max(0, r0 - 1) : min(height, r1 + 2), max(0, c0 - 1) : min(width, c1 + 2)] = True

    specs = [(int(color), _random_shape()) for color in colors]
    specs.sort(key=lambda item: len(item[1]), reverse=True)

    placements: list[tuple[int, list[tuple[int, int]]]] = []
    for color, cells in specs:
        shape_h = max(r for r, _ in cells) + 1
        shape_w = max(c for _, c in cells) + 1
        candidates: list[list[tuple[int, int]]] = []
        for top in range(1, height - shape_h):
            for left in range(1, width - shape_w):
                shifted = [(int(top + r), int(left + c)) for r, c in cells]
                if any(reserved[r, c] for r, c in shifted):
                    continue
                candidates.append(shifted)

        if not candidates:
            return None

        shifted = random.choice(candidates)
        placements.append((int(color), shifted))
        _reserve_with_padding(reserved, shifted)

    return placements


def _validate_generated_grid(grid: np.ndarray, fence_bbox: tuple[int, int, int, int], sequence: list[int]) -> bool:
    bbox, detected = _find_fence_sequence(grid)
    if bbox != fence_bbox or detected != sequence:
        return False

    r0, c0, r1, c1 = fence_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    for color in sequence:
        if len(_components4((grid == int(color)) & outside)) != 1:
            return False

    expected = _rings_from_sequence(sequence)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([color for color in range(10) if color != FENCE_COLOR])
    stripe_pool = [color for color in range(10) if color not in (bg, FENCE_COLOR)]

    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    if stripe_count > len(stripe_pool):
        return None

    inner_h = random.randint(*INTERIOR_HEIGHT_RANGE)
    outer_h = inner_h + 2
    outer_w = stripe_count + 2

    min_h = max(GRID_H_RANGE[0], outer_h + 6)
    min_w = max(GRID_W_RANGE[0], outer_w + 8)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_H_RANGE[1])
    width = random.randint(min_w, GRID_W_RANGE[1])

    top = random.randint(2, height - outer_h - 2)
    left = random.randint(2, width - outer_w - 2)
    fence_bbox = (int(top), int(left), int(top + outer_h - 1), int(left + outer_w - 1))

    sequence = random.sample(stripe_pool, stripe_count)
    grid = np.full((height, width), int(bg), dtype=int)

    r0, c0, r1, c1 = fence_bbox
    grid[r0, c0 : c1 + 1] = FENCE_COLOR
    grid[r1, c0 : c1 + 1] = FENCE_COLOR
    grid[r0 : r1 + 1, c0] = FENCE_COLOR
    grid[r0 : r1 + 1, c1] = FENCE_COLOR
    for idx, color in enumerate(sequence):
        grid[r0 + 1 : r1, c0 + 1 + idx] = int(color)

    placements = _place_objects(grid.shape, fence_bbox, sequence)
    if placements is None:
        return None

    for color, cells in placements:
        for r, c in cells:
            grid[int(r), int(c)] = int(color)

    if not _validate_generated_grid(grid, fence_bbox, sequence):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-stripe input")
