from __future__ import annotations

from collections import Counter
import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
_COLORS = tuple(range(10))
_GRID_HEIGHT_RANGE = (10, 24)
_GRID_WIDTH_RANGE = (10, 24)
_SPRITE_HEIGHT_RANGE = (1, 5)
_SPRITE_WIDTH_RANGE = (1, 6)
_MAX_OBJECTS = 9
_GEN_ATTEMPTS = 400
_SHAPE_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": int(color),
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _component_signature(component: dict) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = component["bbox"]
    return frozenset(
        (int(r) - int(r0), int(c) - int(c0)) for r, c in component["coords"]
    )


def _component_sprite(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in component["coords"]:
        sprite[int(r) - int(r0), int(c) - int(c0)] = int(component["color"])
    return sprite


def _row_capacity(input_width: int, sprite_width: int) -> int:
    return max(1, (int(input_width) + 1) // (int(sprite_width) + 1))


def _pack_identical_sprites(
    count: int,
    input_width: int,
    sprite_cells: frozenset[tuple[int, int]],
    sprite_shape: tuple[int, int],
    bg: int,
    color: int,
) -> np.ndarray:
    sprite_h, sprite_w = map(int, sprite_shape)
    per_row = _row_capacity(int(input_width), int(sprite_w))
    rows = (int(count) + int(per_row) - 1) // int(per_row)
    cols = min(int(count), int(per_row))
    out_h = rows * int(sprite_h) + max(0, rows - 1)
    out_w = cols * int(sprite_w) + max(0, cols - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for idx in range(int(count)):
        base_r = (idx // int(per_row)) * (int(sprite_h) + 1)
        base_c = (idx % int(per_row)) * (int(sprite_w) + 1)
        for dr, dc in sprite_cells:
            out[base_r + int(dr), base_c + int(dc)] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort identical components by bbox center and repack them into a tight array."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    colors = {int(component["color"]) for component in components}
    assert len(colors) == 1, "all foreground objects must share one color"

    first = components[0]
    sprite = _component_sprite(first, bg)
    sprite_shape = tuple(map(int, sprite.shape))
    sprite_cells = _component_signature(first)
    color = int(first["color"])

    for component in components[1:]:
        assert int(component["color"]) == int(color)
        assert _component_signature(component) == sprite_cells
        bbox = component["bbox"]
        assert (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1) == sprite_shape

    ordered = sorted(
        components,
        key=lambda component: (
            int(component["bbox"][0]) + int(component["bbox"][2]),
            int(component["bbox"][1]) + int(component["bbox"][3]),
        ),
    )
    assert len(ordered) == len(components)

    return _pack_identical_sprites(
        count=len(ordered),
        input_width=int(g.shape[1]),
        sprite_cells=sprite_cells,
        sprite_shape=sprite_shape,
        bg=int(bg),
        color=int(color),
    )


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r) - int(min_r), int(c) - int(min_c)) for r, c in cells}


def _grow_shape(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    cells = _normalize(cells)
    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1
    if max_r != int(height) or max_c != int(width):
        return None
    return cells


def _sample_shape() -> tuple[set[tuple[int, int]], int, int] | None:
    for _ in range(_SHAPE_ATTEMPTS):
        height = random.randint(*_SPRITE_HEIGHT_RANGE)
        width = random.randint(*_SPRITE_WIDTH_RANGE)
        if height == 1 and width == 1:
            continue
        min_area = max(2, height + width - 1)
        if min_area > height * width:
            continue
        area = random.randint(min_area, height * width)
        cells = _grow_shape(height, width, area)
        if cells is not None:
            return cells, int(height), int(width)
    return None


def _rect_separated(
    placed_rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int
) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for pr0, pc0, pr1, pc1 in placed_rects:
        if not (
            int(r1) < int(pr0) - 1
            or int(pr1) < int(r0) - 1
            or int(c1) < int(pc0) - 1
            or int(pc1) < int(c0) - 1
        ):
            return False
    return True


def _place_objects(
    grid_shape: tuple[int, int], sprite_shape: tuple[int, int], count: int
) -> list[tuple[int, int]] | None:
    grid_h, grid_w = map(int, grid_shape)
    sprite_h, sprite_w = map(int, sprite_shape)
    placed_rects: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []

    for _ in range(int(count)):
        candidates = [
            (r0, c0)
            for r0 in range(grid_h - sprite_h + 1)
            for c0 in range(grid_w - sprite_w + 1)
            if _rect_separated(placed_rects, r0, c0, sprite_h, sprite_w)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed_rects.append((r0, c0, r0 + sprite_h - 1, c0 + sprite_w - 1))
        placements.append((r0, c0))

    return placements


def _render_input(
    grid_shape: tuple[int, int],
    bg: int,
    color: int,
    sprite_cells: set[tuple[int, int]],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(tuple(map(int, grid_shape)), int(bg), dtype=int)
    for base_r, base_c in placements:
        for dr, dc in sprite_cells:
            out[int(base_r) + int(dr), int(base_c) + int(dc)] = int(color)
    return out


def _choose_object_count(
    grid_shape: tuple[int, int], sprite_shape: tuple[int, int], area: int
) -> int | None:
    grid_h, grid_w = map(int, grid_shape)
    sprite_h, sprite_w = map(int, sprite_shape)
    per_row = _row_capacity(grid_w, sprite_w)
    max_rows = max(1, (grid_h + 1) // (sprite_h + 1))
    max_slots = min(_MAX_OBJECTS, per_row * max_rows)
    max_bg_safe = (grid_h * grid_w - 1) // (2 * int(area))
    max_objects = min(max_slots, max_bg_safe)
    if max_objects < 2:
        return None

    want_multi_row = random.random() < 0.5
    if want_multi_row:
        if max_objects <= per_row:
            return None
        return random.randint(per_row + 1, max_objects)

    if per_row < 2:
        return None
    return random.randint(2, min(per_row, max_objects))


def _make_expected_output(
    input_width: int,
    bg: int,
    color: int,
    sprite_cells: set[tuple[int, int]],
    sprite_shape: tuple[int, int],
    count: int,
) -> np.ndarray:
    return _pack_identical_sprites(
        count=int(count),
        input_width=int(input_width),
        sprite_cells=frozenset((int(r), int(c)) for r, c in sprite_cells),
        sprite_shape=tuple(map(int, sprite_shape)),
        bg=int(bg),
        color=int(color),
    )


def _gen_attempt() -> np.ndarray | None:
    bg, color = random.sample(_COLORS, 2)
    grid_h = random.randint(*_GRID_HEIGHT_RANGE)
    grid_w = random.randint(*_GRID_WIDTH_RANGE)

    shape_data = _sample_shape()
    if shape_data is None:
        return None
    sprite_cells, sprite_h, sprite_w = shape_data
    area = len(sprite_cells)

    count = _choose_object_count((grid_h, grid_w), (sprite_h, sprite_w), area)
    if count is None:
        return None

    placements = _place_objects((grid_h, grid_w), (sprite_h, sprite_w), count)
    if placements is None:
        return None

    inp = _render_input((grid_h, grid_w), bg, color, sprite_cells, placements)
    expected = _make_expected_output(
        input_width=grid_w,
        bg=bg,
        color=color,
        sprite_cells=sprite_cells,
        sprite_shape=(sprite_h, sprite_w),
        count=count,
    )

    if np.array_equal(inp, expected):
        return None
    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {_GEN_ATTEMPTS} attempts")
