from __future__ import annotations

import numpy as np


GREY = 5
PALETTE = tuple(c for c in range(10) if c != GREY)
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECTS = 5
MIN_OBJECTS = 2
MAX_STRIPE_THICKNESS = 12
MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 30
PLACEMENT_ATTEMPTS = 300

_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_bar_lengths(top_row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = int(top_row.shape[0])
    while c < width and int(top_row[c]) != bg:
        color = int(top_row[c])
        start = c
        while c < width and int(top_row[c]) == color:
            c += 1
        if color in mapping:
            raise ValueError("top bar colors must be unique")
        mapping[color] = int(c - start)
    if c < width and np.any(top_row[c:] != bg):
        raise ValueError("top bar must be contiguous and may only have trailing background")
    if not mapping:
        raise ValueError("top bar is empty")
    return mapping


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    interior = np.asarray(grid[1:], dtype=int)
    height, width = interior.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(interior[r, c]) == bg:
                continue
            color = int(interior[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr + 1), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(interior[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read stripe periods from the top-edge color bar and fill matching objects with alternating grey bands.

    Input structure:
    - One dominant background color fills most of the grid.
    - The top row starts with a contiguous bar made of differently colored segments; each segment has a unique length from 1 to 5, and any remaining cells in that row are background.
    - Below the top row sit several solid monochrome 4-connected objects, one per bar color.

    Rule:
    - For each interior object, find the top-row bar segment with the same color and read its segment length L.
    - Compute the object's bounding box. If width >= height, treat it as horizontal; otherwise treat it as vertical.
    - Horizontal objects are striped by rows: row offsets inside the bounding box are grouped in runs of L. Vertical objects are striped by columns: column offsets inside the bounding box are grouped in runs of L.
    - Even-indexed groups keep the object color, odd-indexed groups become grey. The top bar row stays unchanged.

    Key insight:
    - The matching color only identifies which object uses which period; the real transformation parameter is the geometric length of that color's segment in the edge bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    bar_lengths = _parse_bar_lengths(g[0], bg)
    objects = _find_objects(g, bg)

    out = g.copy()
    for obj in objects:
        color = int(obj["color"])
        if color not in bar_lengths:
            raise ValueError("every object color must appear in the top bar")
        length = int(bar_lengths[color])
        r0, c0, r1, c1 = obj["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        horizontal = width >= height

        for r, c in obj["coords"]:
            offset = int(r - r0) if horizontal else int(c - c0)
            if (offset // length) % 2 == 1:
                out[r, c] = GREY

    return out


def _sample_object_dims(length: int, horizontal: bool, allow_square: bool) -> tuple[int, int]:
    stripe_span_max = min(MAX_STRIPE_THICKNESS, 3 * length + 3)
    stripe_span = int(_RNG.integers(length + 1, stripe_span_max + 1))
    long_side_max = min(MAX_GRID_SIZE - 4, stripe_span + 6)

    if horizontal:
        height = stripe_span
        if allow_square:
            width = height
        else:
            width = int(_RNG.integers(height, long_side_max + 1))
        return int(height), int(width)

    width = stripe_span
    height_min = width + 1
    height_max = max(height_min, long_side_max)
    height = int(_RNG.integers(height_min, height_max + 1))
    return int(height), int(width)


def _object_specs(colors: list[int], lengths: dict[int, int]) -> list[dict[str, int | bool]]:
    count = len(colors)
    orientations = [True, False]
    while len(orientations) < count:
        orientations.append(bool(_RNG.integers(0, 2)))
    _RNG.shuffle(orientations)

    square_index = int(_RNG.integers(0, count)) if bool(_RNG.integers(0, 3) == 0) else -1

    specs: list[dict[str, int | bool]] = []
    for idx, color in enumerate(colors):
        allow_square = idx == square_index and orientations[idx]
        h, w = _sample_object_dims(int(lengths[color]), bool(orientations[idx]), allow_square)
        specs.append(
            {
                "color": int(color),
                "length": int(lengths[color]),
                "horizontal": bool(orientations[idx]),
                "height": int(h),
                "width": int(w),
            }
        )
    return specs


def _place_objects(grid: np.ndarray, specs: list[dict[str, int | bool]]) -> bool:
    height, width = grid.shape
    blocked = np.zeros_like(grid, dtype=bool)
    order = sorted(specs, key=lambda spec: (int(spec["height"]) * int(spec["width"]), int(spec["height"]) + int(spec["width"])), reverse=True)

    for spec in order:
        obj_h = int(spec["height"])
        obj_w = int(spec["width"])
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            if height - obj_h <= 2 or width - obj_w < 0:
                return False
            r0 = int(_RNG.integers(2, height - obj_h + 1))
            c0 = int(_RNG.integers(0, width - obj_w + 1))
            rr0 = max(1, r0 - 1)
            rr1 = min(height, r0 + obj_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + obj_w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = int(spec["color"])
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _build_grid() -> np.ndarray | None:
    object_count = int(_RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    bg = int(_RNG.choice(PALETTE))
    usable_colors = [c for c in PALETTE if c != bg]
    colors = [int(c) for c in _RNG.choice(usable_colors, size=object_count, replace=False)]
    segment_lengths = [int(v) for v in _RNG.choice(np.arange(1, 6), size=object_count, replace=False)]
    length_map = {color: length for color, length in zip(colors, segment_lengths)}
    specs = _object_specs(colors, length_map)

    max_h = max(int(spec["height"]) for spec in specs)
    max_w = max(int(spec["width"]) for spec in specs)
    bar_total = int(sum(segment_lengths))

    min_h = max(MIN_GRID_SIZE, max_h + 3)
    min_w = max(MIN_GRID_SIZE, max(max_w + 2, bar_total))
    if min_h > MAX_GRID_SIZE or min_w > MAX_GRID_SIZE:
        return None

    height = int(_RNG.integers(min_h, MAX_GRID_SIZE + 1))
    width = int(_RNG.integers(min_w, MAX_GRID_SIZE + 1))
    grid = np.full((height, width), bg, dtype=int)

    bar_colors = colors.copy()
    _RNG.shuffle(bar_colors)
    cursor = 0
    for color in bar_colors:
        length = int(length_map[color])
        grid[0, cursor : cursor + length] = int(color)
        cursor += length

    if not _place_objects(grid, specs):
        return None
    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    try:
        bar_lengths = _parse_bar_lengths(grid[0], bg)
    except ValueError:
        return False

    objects = _find_objects(grid, bg)
    if not (MIN_OBJECTS <= len(objects) <= MAX_OBJECTS):
        return False
    if len(objects) != len(bar_lengths):
        return False

    colors = [int(obj["color"]) for obj in objects]
    if len(set(colors)) != len(colors) or set(colors) != set(bar_lengths):
        return False

    has_horizontal = False
    has_vertical = False
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if width >= height:
            has_horizontal = True
            stripe_depth = height
        else:
            has_vertical = True
            stripe_depth = width
        if stripe_depth <= int(bar_lengths[color]):
            return False

    if not (has_horizontal and has_vertical):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out[0] != grid[0]):
        return False
    if not np.any(out == GREY):
        return False
    if max(grid.shape) > MAX_GRID_SIZE or max(out.shape) > MAX_GRID_SIZE:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
