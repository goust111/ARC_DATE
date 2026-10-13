from __future__ import annotations

import random

import numpy as np


Box = tuple[int, int, int, int]
Cell = tuple[int, int]


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = exclude or set()
    return [int(color) for color in np.unique(grid) if int(color) not in excluded]


def _find_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    mask = np.asarray(grid == color, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[Cell] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in coords]
            cols = [cell[1] for cell in coords]
            components.append(
                {
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "coords": coords,
                    "area": len(coords),
                }
            )

    return components


def _assert_solid_rectangle(grid: np.ndarray, color: int, component: dict[str, object]) -> Box:
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert int(component["area"]) == height * width, "foreground objects must be solid rectangles"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be fully filled"
    return r0, c0, r1, c1


def _facing_corners(upper: Box, lower: Box) -> tuple[Cell, Cell]:
    _, uc0, ur1, uc1 = upper
    lr0, lc0, _, lc1 = lower
    assert ur1 < lr0, "same-color rectangles must be vertically separated"

    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("same-color rectangles must be diagonally separated")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color diagonal rectangle pair with its open 45-degree diagonal segment.

    Input structure:
    - The background is the dominant color.
    - Each foreground color appears in exactly two separate solid rectangles.
    - The upper rectangle and lower rectangle of a color have facing corners on a
      45-degree diagonal.

    Rule:
    - Keep both rectangles unchanged.
    - Connect the lower facing corner of the upper rectangle to the upper facing
      corner of the lower rectangle.
    - Paint only the background cells strictly between those endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    out = g.copy()

    for color in _colors_present(g, exclude={bg}):
        components = _find_components(g, color)
        assert len(components) == 2, "each foreground color must form exactly two rectangles"

        boxes = [_assert_solid_rectangle(g, color, comp) for comp in components]
        boxes.sort(key=lambda box: (box[0], box[1]))
        upper, lower = boxes
        start, end = _facing_corners(upper, lower)

        row_delta = end[0] - start[0]
        col_delta = end[1] - start[1]
        assert row_delta > 0, "the lower rectangle must lie below the upper one"
        assert abs(row_delta) == abs(col_delta), "facing corners must lie on a 45-degree diagonal"

        step_c = 1 if col_delta > 0 else -1
        for step in range(1, row_delta):
            r = start[0] + step
            c = start[1] + step * step_c
            assert int(out[r, c]) == bg, "the connector path must pass through background"
            out[r, c] = color

    return out


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))
PAIR_COUNT_CHOICES = (1, 2, 3, 4)
PAIR_COUNT_WEIGHTS = (2, 5, 4, 2)
RECT_SIDE_VALUES = (1, 2, 3, 4)
RECT_SIDE_WEIGHTS = (3, 4, 4, 2)
GAP_VALUES = (1, 2, 3, 4, 5, 6)
GRID_MAX = 24
GRID_BASE_MIN = 10
PLACEMENT_RESTARTS = 24
MAX_GENERATION_ATTEMPTS = 400


def _rect_cells(box: Box) -> list[Cell]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _paint_rect(grid: np.ndarray, box: Box, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_side() -> int:
    return random.choices(RECT_SIDE_VALUES, weights=RECT_SIDE_WEIGHTS, k=1)[0]


def _sample_pair_specs(pair_count: int, bg: int) -> list[dict[str, int]]:
    palette = [color for color in COLOR_VALUES if color != bg]
    colors = random.sample(palette, pair_count)
    specs = []
    for color in colors:
        specs.append(
            {
                "color": color,
                "upper_h": _sample_side(),
                "upper_w": _sample_side(),
                "lower_h": _sample_side(),
                "lower_w": _sample_side(),
                "gap": random.choice(GAP_VALUES),
                "direction": random.choice((-1, 1)),
            }
        )
    return specs


def _footprint_height(spec: dict[str, int]) -> int:
    return int(spec["upper_h"] + spec["gap"] + spec["lower_h"])


def _footprint_width(spec: dict[str, int]) -> int:
    return int(spec["upper_w"] + spec["gap"] + spec["lower_w"])


def _sample_grid_shape(specs: list[dict[str, int]]) -> tuple[int, int] | None:
    fg_area = sum(spec["upper_h"] * spec["upper_w"] + spec["lower_h"] * spec["lower_w"] for spec in specs)
    min_h = max(GRID_BASE_MIN, 8 + 2 * len(specs), max(_footprint_height(spec) for spec in specs) + 2)
    min_w = max(GRID_BASE_MIN, 8 + 2 * len(specs), max(_footprint_width(spec) for spec in specs) + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    valid_shapes = [
        (height, width)
        for height in range(min_h, GRID_MAX + 1)
        for width in range(min_w, GRID_MAX + 1)
        if height * width > 2 * fg_area
    ]
    if not valid_shapes:
        return None
    return random.choice(valid_shapes)


def _pair_candidate(spec: dict[str, int], top: int, left: int, height: int, width: int) -> tuple[Box, Box, list[Cell], list[Cell]] | None:
    upper_h = int(spec["upper_h"])
    upper_w = int(spec["upper_w"])
    lower_h = int(spec["lower_h"])
    lower_w = int(spec["lower_w"])
    gap = int(spec["gap"])
    direction = int(spec["direction"])
    delta = gap + 1

    upper = (top, left, top + upper_h - 1, left + upper_w - 1)
    if direction == 1:
        start = (upper[2], upper[3])
        end = (start[0] + delta, start[1] + delta)
        lower = (end[0], end[1], end[0] + lower_h - 1, end[1] + lower_w - 1)
    else:
        start = (upper[2], upper[1])
        end = (start[0] + delta, start[1] - delta)
        lower = (end[0], end[1] - lower_w + 1, end[0] + lower_h - 1, end[1])

    if upper[0] < 0 or upper[1] < 0 or lower[0] < 0 or lower[1] < 0:
        return None
    if upper[2] >= height or upper[3] >= width or lower[2] >= height or lower[3] >= width:
        return None

    path = [(start[0] + step, start[1] + direction * step) for step in range(1, gap + 1)]
    cells = _rect_cells(upper) + _rect_cells(lower) + path
    return upper, lower, path, cells


def _mark_blocked(blocked: np.ndarray, cells: list[Cell]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _candidate_positions(spec: dict[str, int], blocked: np.ndarray) -> list[tuple[Box, Box, list[Cell], list[Cell]]]:
    height, width = blocked.shape
    candidates = []
    for top in range(height - int(spec["upper_h"]) + 1):
        for left in range(width - int(spec["upper_w"]) + 1):
            candidate = _pair_candidate(spec, top, left, height, width)
            if candidate is None:
                continue
            if any(blocked[r, c] for r, c in candidate[3]):
                continue
            candidates.append(candidate)
    random.shuffle(candidates)
    return candidates


def _place_pairs(specs: list[dict[str, int]], shape: tuple[int, int]) -> list[dict[str, object]] | None:
    height, width = shape
    order = sorted(
        range(len(specs)),
        key=lambda idx: (_footprint_height(specs[idx]) * _footprint_width(specs[idx])),
        reverse=True,
    )

    for _ in range(PLACEMENT_RESTARTS):
        blocked = np.zeros((height, width), dtype=bool)
        placed: list[dict[str, object] | None] = [None] * len(specs)
        success = True

        for idx in order:
            candidates = _candidate_positions(specs[idx], blocked)
            if not candidates:
                success = False
                break
            upper, lower, path, cells = random.choice(candidates)
            _mark_blocked(blocked, cells)
            placed[idx] = {
                "color": int(specs[idx]["color"]),
                "upper": upper,
                "lower": lower,
                "path": path,
            }

        if success:
            return [item for item in placed if item is not None]

    return None


def _render_example(shape: tuple[int, int], bg: int, placed: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)

    for item in placed:
        color = int(item["color"])
        upper = item["upper"]
        lower = item["lower"]
        path = item["path"]
        _paint_rect(inp, upper, color)
        _paint_rect(inp, lower, color)
        _paint_rect(out, upper, color)
        _paint_rect(out, lower, color)
        for r, c in path:
            out[r, c] = color

    return inp, out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    return int(np.count_nonzero(grid == bg)) > int(np.count_nonzero(grid != bg))


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    bg = random.choice(COLOR_VALUES)
    specs = _sample_pair_specs(pair_count, bg)
    shape = _sample_grid_shape(specs)
    if shape is None:
        return None

    placed = _place_pairs(specs, shape)
    if placed is None:
        return None

    inp, expected = _render_example(shape, bg, placed)
    if not _background_is_dominant(inp, bg):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
