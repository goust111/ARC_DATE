from __future__ import annotations

import math
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each foreground color by a solid square whose side is sqrt(pixel_count)."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    specs: list[tuple[int, int]] = []
    for color in colors:
        count = int(np.count_nonzero(g == color))
        side = math.isqrt(count)
        assert side * side == count, "foreground counts must be perfect squares"
        specs.append((side, color))

    sides = [side for side, _color in specs]
    assert len(set(sides)) == len(sides), "square side lengths must be unique"

    ordered = sorted(specs)
    out_h = max(side for side, _color in ordered)
    out_w = sum(side for side, _color in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for idx, (side, color) in enumerate(ordered):
        out[out_h - side : out_h, col : col + side] = color
        col += side
        if idx != len(ordered) - 1:
            col += 1

    return out


ALL_COLORS = tuple(range(10))
SIDE_POOL = (2, 3, 4, 5, 6)
OBJECT_COUNT_RANGE = (2, 5)
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 28
MAX_GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _is_filled_rectangle(cells: set[tuple[int, int]]) -> bool:
    h, w = _shape_dims(cells)
    return len(cells) == h * w


def _transform_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rot = random.randrange(4)
    flip = random.random() < 0.5
    transformed = set()
    for r, c in cells:
        rr, cc = r, c
        if flip:
            cc = -cc
        for _ in range(rot):
            rr, cc = cc, -rr
        transformed.add((rr, cc))
    return _normalize_shape(transformed)


def _grow_irregular_shape(area: int, nominal_side: int) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))

            if not frontier:
                break

            center_r = sum(r for r, _c in cells) / len(cells)
            center_c = sum(c for _r, c in cells) / len(cells)
            frontier_list = list(frontier)
            weights = []
            for r, c in frontier_list:
                adjacency = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
                distance = abs(r - center_r) + abs(c - center_c)
                weight = (adjacency + 1) ** 3 / (1.0 + distance)
                weights.append(weight)

            cells.add(random.choices(frontier_list, weights=weights, k=1)[0])

        if len(cells) != area:
            continue

        cells = _transform_shape(_normalize_shape(cells))
        h, w = _shape_dims(cells)
        if _is_filled_rectangle(cells):
            continue
        if max(h, w) > nominal_side * 2 + 1:
            continue
        if min(h, w) == 1 and area >= 9:
            continue
        return cells

    return None


def _starts(limit: int, span: int) -> range:
    slack = limit - span
    if slack >= 2:
        return range(1, slack)
    return range(0, slack + 1)


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _place_objects(
    bg: int,
    objects: list[dict[str, object]],
) -> np.ndarray | None:
    total_fg = sum(int(obj["side"]) ** 2 for obj in objects)
    min_h = max(GRID_SIDE_MIN, max(int(obj["height"]) for obj in objects) + 2)
    min_w = max(GRID_SIDE_MIN, max(int(obj["width"]) for obj in objects) + 2)

    candidate_dims = [
        (height, width)
        for height in range(min_h, GRID_SIDE_MAX + 1)
        for width in range(min_w, GRID_SIDE_MAX + 1)
        if 2 * total_fg < height * width <= min(GRID_SIDE_MAX * GRID_SIDE_MAX, 5 * total_fg + 80)
    ]
    random.shuffle(candidate_dims)

    for height, width in candidate_dims[:PLACEMENT_ATTEMPTS]:
        grid = np.full((height, width), bg, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        seeded = objects[:]
        random.shuffle(seeded)
        seeded.sort(key=lambda obj: (-(int(obj["height"]) * int(obj["width"])), -int(obj["side"])))

        success = True
        for obj in seeded:
            shape = obj["shape"]
            color = int(obj["color"])
            shape_h = int(obj["height"])
            shape_w = int(obj["width"])
            positions = []
            row_starts = list(_starts(height, shape_h))
            col_starts = list(_starts(width, shape_w))
            random.shuffle(row_starts)
            random.shuffle(col_starts)

            for top in row_starts:
                for left in col_starts:
                    coords = [(top + r, left + c) for r, c in shape]
                    if all(not blocked[r, c] for r, c in coords):
                        positions.append(coords)

            if not positions:
                success = False
                break

            coords = random.choice(positions)
            for r, c in coords:
                grid[r, c] = color
            _mark_blocked(blocked, coords)

        if success:
            return grid

    return None


def _is_single_component(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    start = tuple(map(int, points[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == len(points)


def _mask_is_filled_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(mask.sum()) == int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _expected_output(bg: int, color_to_side: dict[int, int]) -> np.ndarray:
    ordered = sorted((side, color) for color, side in color_to_side.items())
    out_h = max(side for side, _color in ordered)
    out_w = sum(side for side, _color in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for idx, (side, color) in enumerate(ordered):
        out[out_h - side : out_h, col : col + side] = color
        col += side
        if idx != len(ordered) - 1:
            col += 1

    return out


def _is_valid_scene(grid: np.ndarray, bg: int, color_to_side: dict[int, int]) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(value): int(count) for value, count in zip(values, counts)}
    if bg not in count_map:
        return False
    if any(count_map[bg] <= count for color, count in count_map.items() if color != bg):
        return False
    if {int(color) for color in values if int(color) != bg} != set(color_to_side):
        return False

    for color, side in color_to_side.items():
        mask = grid == color
        if int(mask.sum()) != side * side:
            return False
        if not _is_single_component(mask):
            return False
        if _mask_is_filled_rectangle(mask):
            return False

    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, _expected_output(bg, color_to_side))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    sides = random.sample(SIDE_POOL, n_objects)
    colors = random.sample([color for color in ALL_COLORS if color != bg], n_objects)

    objects = []
    color_to_side = {}
    for color, side in zip(colors, sides):
        shape = _grow_irregular_shape(side * side, side)
        if shape is None:
            return None
        shape_h, shape_w = _shape_dims(shape)
        objects.append(
            {
                "color": color,
                "side": side,
                "shape": tuple(sorted(shape)),
                "height": shape_h,
                "width": shape_w,
            }
        )
        color_to_side[color] = side

    placed = _place_objects(bg, objects)
    if placed is None:
        return None
    if not _is_valid_scene(placed, bg, color_to_side):
        return None
    return placed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
