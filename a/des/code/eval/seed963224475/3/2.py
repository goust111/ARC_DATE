from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_BG = 0
_DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_DIRECTIONS = tuple(_DIRECTION_VECTORS)

_GRID_HEIGHT_RANGE = (10, 18)
_GRID_WIDTH_RANGE = (10, 18)
_OBJECT_COUNT_RANGE = (2, 4)
_SHAPE_SIZE_RANGE = (1, 5)
_GEN_ATTEMPTS = 400
_OBJECT_ATTEMPTS = 160


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == _BG).any()):
        return _BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in _DIR8:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _detect_arrow(coords: list[tuple[int, int]], h: int, w: int) -> str | None:
    if len(coords) != 3:
        return None
    pts = set(coords)

    top_heads = [c for r, c in pts if r == 0]
    if len(top_heads) == 1:
        c = int(top_heads[0])
        if 0 < c < w - 1 and pts == {(0, c), (1, c - 1), (1, c + 1)}:
            return "up"

    bottom_heads = [c for r, c in pts if r == h - 1]
    if len(bottom_heads) == 1:
        c = int(bottom_heads[0])
        if 0 < c < w - 1 and pts == {(h - 1, c), (h - 2, c - 1), (h - 2, c + 1)}:
            return "down"

    left_heads = [r for r, c in pts if c == 0]
    if len(left_heads) == 1:
        r = int(left_heads[0])
        if 0 < r < h - 1 and pts == {(r, 0), (r - 1, 1), (r + 1, 1)}:
            return "left"

    right_heads = [r for r, c in pts if c == w - 1]
    if len(right_heads) == 1:
        r = int(right_heads[0])
        if 0 < r < h - 1 and pts == {(r, w - 1), (r - 1, w - 2), (r + 1, w - 2)}:
            return "right"

    return None


def _parse_objects(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    objects: dict[int, dict[str, object]] = {}
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)

    for color in colors:
        components = _connected_components(g == int(color))
        arrow_parts: list[tuple[list[tuple[int, int]], str]] = []
        seed_parts: list[list[tuple[int, int]]] = []
        for comp in components:
            direction = _detect_arrow(comp, h, w)
            if direction is None:
                seed_parts.append(comp)
            else:
                arrow_parts.append((comp, direction))
        assert len(arrow_parts) == 1, "each color must have exactly one edge arrow"
        assert len(seed_parts) == 1, "each color must have exactly one seed component"
        seed = seed_parts[0]
        assert 1 <= len(seed) <= 5, "seed size must be between 1 and 5"
        objects[int(color)] = {
            "arrow": arrow_parts[0][0],
            "direction": arrow_parts[0][1],
            "seed": seed,
        }

    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge arrows to choose a growth direction for each colored seed cluster,
    then extend the cluster by its own size along that direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, objects = _parse_objects(g)
    out = g.copy()

    for color in sorted(objects):
        seed = list(objects[color]["seed"])
        direction = str(objects[color]["direction"])
        steps = len(seed)
        dr, dc = _DIRECTION_VECTORS[direction]
        for r, c in seed:
            for step in range(1, int(steps) + 1):
                rr = int(r + dr * step)
                cc = int(c + dc * step)
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                current = int(out[rr, cc])
                if current not in (int(bg), int(color)):
                    break
                out[rr, cc] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((int(r - min_r), int(c - min_c)) for r, c in cells)
    return tuple(normalized)


def _random_seed_shape(size: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in _DIR8:
                frontier.add((int(r + dr), int(c + dc)))
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    return int(height), int(width)


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in shape}


def _dilate8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _growth_cells(
    seed_cells: set[tuple[int, int]],
    direction: str,
    steps: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], bool]:
    dr, dc = _DIRECTION_VECTORS[direction]
    grown: set[tuple[int, int]] = set()
    clipped = False
    for r, c in seed_cells:
        for step in range(1, int(steps) + 1):
            rr = int(r + dr * step)
            cc = int(c + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                clipped = True
                break
            grown.add((rr, cc))
    return grown, bool(clipped)


def _anchor_ranges(
    h: int,
    w: int,
    shape_h: int,
    shape_w: int,
    steps: int,
    direction: str,
    clip: bool,
) -> tuple[range, range] | None:
    max_r0 = h - shape_h - 1
    max_c0 = w - shape_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    if direction == "up":
        row_lo = 1
        row_hi = min(steps - 1, max_r0) if clip else max_r0
        if not clip:
            row_lo = max(row_lo, steps)
    elif direction == "down":
        row_lo = max(1, h - shape_h - steps + 1) if clip else 1
        row_hi = max_r0 if clip else min(max_r0, h - shape_h - steps)
    else:
        row_lo = 1
        row_hi = max_r0

    if direction == "left":
        col_lo = 1
        col_hi = min(steps - 1, max_c0) if clip else max_c0
        if not clip:
            col_lo = max(col_lo, steps)
    elif direction == "right":
        col_lo = max(1, w - shape_w - steps + 1) if clip else 1
        col_hi = max_c0 if clip else min(max_c0, w - shape_w - steps)
    else:
        col_lo = 1
        col_hi = max_c0

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(int(row_lo), int(row_hi) + 1), range(int(col_lo), int(col_hi) + 1)


def _arrow_cells(direction: str, anchor: int, h: int, w: int) -> set[tuple[int, int]]:
    if direction == "up":
        return {(0, int(anchor)), (1, int(anchor - 1)), (1, int(anchor + 1))}
    if direction == "down":
        return {(h - 1, int(anchor)), (h - 2, int(anchor - 1)), (h - 2, int(anchor + 1))}
    if direction == "left":
        return {(int(anchor), 0), (int(anchor - 1), 1), (int(anchor + 1), 1)}
    return {(int(anchor), w - 1), (int(anchor - 1), w - 2), (int(anchor + 1), w - 2)}


def _place_one_object(
    grid: np.ndarray,
    color: int,
    input_forbidden: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(_OBJECT_ATTEMPTS):
        size = random.randint(*_SHAPE_SIZE_RANGE)
        shape = _random_seed_shape(size)
        shape_h, shape_w = _shape_dims(shape)
        direction = random.choice(_DIRECTIONS)
        clip = bool(size > 1 and random.choice([False, True]))

        ranges = _anchor_ranges(h, w, shape_h, shape_w, size, direction, clip)
        if ranges is None:
            continue
        row_range, col_range = ranges
        anchors = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(anchors)

        placed = None
        for r0, c0 in anchors:
            seed_cells = _translate(shape, int(r0), int(c0))
            if seed_cells & input_forbidden:
                continue
            growth, clipped = _growth_cells(seed_cells, direction, size, h, w)
            if clipped != clip or not growth:
                continue
            seed_and_growth = seed_cells | growth
            if seed_and_growth & reserved_output:
                continue
            placed = (seed_cells, growth, direction)
            break
        if placed is None:
            continue

        seed_cells, growth, direction = placed
        local_forbidden = input_forbidden | _dilate8(seed_cells, h, w)
        anchors_1d = list(range(1, w - 1)) if direction in ("up", "down") else list(range(1, h - 1))
        random.shuffle(anchors_1d)

        for anchor in anchors_1d:
            arrow = _arrow_cells(direction, int(anchor), h, w)
            if arrow & local_forbidden:
                continue
            if arrow & reserved_output:
                continue
            if arrow & (seed_cells | growth):
                continue
            for r, c in seed_cells | arrow:
                grid[r, c] = int(color)
            return seed_cells, growth, arrow

    return None


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*_GRID_HEIGHT_RANGE)
    w = random.randint(*_GRID_WIDTH_RANGE)
    grid = np.full((h, w), _BG, dtype=int)

    max_objects = min(_OBJECT_COUNT_RANGE[1], 9, max(2, (h + w) // 8))
    min_objects = min(_OBJECT_COUNT_RANGE[0], max_objects)
    num_objects = random.randint(min_objects, max_objects)
    colors = random.sample(list(range(1, 10)), num_objects)

    input_cells: set[tuple[int, int]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()

    for color in colors:
        placed = _place_one_object(grid, int(color), input_forbidden, reserved_output)
        if placed is None:
            return None
        seed_cells, growth, arrow_cells = placed
        current_input = seed_cells | arrow_cells
        input_cells |= current_input
        input_forbidden = _dilate8(input_cells, h, w)
        reserved_output |= seed_cells | growth | arrow_cells

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            bg, objects = _parse_objects(grid)
            assert bg == _BG
            assert objects
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
