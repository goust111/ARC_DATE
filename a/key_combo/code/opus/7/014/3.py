from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


Coord = tuple[int, int]
ObjectSpec = dict[str, object]

DRAW_COLORS = [Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _clockwise(direction: Coord) -> Coord:
    dr, dc = map(int, direction)
    return int(dc), int(-dr)


def _inverse_clockwise(ray_direction: Coord) -> Coord:
    dr, dc = map(int, ray_direction)
    return int(-dc), int(dr)


def _segment_cells(pivot: Coord, direction: Coord, length: int) -> list[Coord]:
    pr, pc = map(int, pivot)
    dr, dc = map(int, direction)
    return [(pr + step * dr, pc + step * dc) for step in range(1, int(length) + 1)]


def _input_cells(obj: ObjectSpec) -> list[Coord]:
    pivot = obj["pivot"]
    direction = obj["direction"]
    length = obj["length"]
    return [pivot] + _segment_cells(pivot, direction, int(length))


def _rotated_cells(obj: ObjectSpec) -> list[Coord]:
    pivot = obj["pivot"]
    direction = _clockwise(obj["direction"])
    length = obj["length"]
    return _segment_cells(pivot, direction, int(length))


def _sorted_pivots(grid: np.ndarray) -> list[Coord]:
    return sorted((int(r), int(c)) for r, c in np.argwhere(grid == Color.GREY))


def _parse_palette(grid: np.ndarray) -> tuple[list[int], tuple[int, int]]:
    top = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(top != Color.BLACK)
    assert 3 <= len(cols) <= 4, "expected a 3-4 cell palette run on the top row"
    assert int(cols[-1] - cols[0] + 1) == len(cols), "palette cells must be contiguous"
    palette = [int(top[c]) for c in cols]
    assert all(color not in (Color.BLACK, Color.GREY, Color.GREEN) for color in palette)
    return palette, (int(cols[0]), int(cols[-1]))


def _follow_segment(grid: np.ndarray, pivot: Coord, near: Coord, color: int) -> list[Coord]:
    pr, pc = map(int, pivot)
    nr, nc = map(int, near)
    dr = nr - pr
    dc = nc - pc
    assert abs(dr) + abs(dc) == 1

    coords: list[Coord] = []
    r, c = nr, nc
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == int(color):
        coords.append((int(r), int(c)))
        r += dr
        c += dc

    assert 2 <= len(coords) <= 4, "line length must be 2-4"
    expected = _segment_cells(pivot, (dr, dc), len(coords))
    assert coords == expected, "line segment must be a solid straight bar"

    for idx, (r, c) in enumerate(coords):
        allowed = set()
        if idx > 0:
            allowed.add(coords[idx - 1])
        if idx + 1 < len(coords):
            allowed.add(coords[idx + 1])
        if idx == 0:
            allowed.add((pr, pc))
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(color):
                assert (rr, cc) in allowed, "segment must not branch"
    return coords


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for pivot in _sorted_pivots(grid):
        pr, pc = pivot
        colored_neighbors = []
        for nr, nc in _neighbors4(pr, pc):
            if _in_bounds(grid.shape, nr, nc):
                value = int(grid[nr, nc])
                if value not in (Color.BLACK, Color.GREY):
                    colored_neighbors.append((nr, nc, value))
        assert len(colored_neighbors) == 1, "each pivot must touch exactly one colored segment"
        nr, nc, color = colored_neighbors[0]
        segment = _follow_segment(grid, pivot, (nr, nc), color)
        direction = (nr - pr, nc - pc)
        objects.append(
            {
                "pivot": pivot,
                "direction": direction,
                "length": len(segment),
                "color": int(color),
                "segment": segment,
                "rotated_direction": _clockwise(direction),
            }
        )
    assert 2 <= len(objects) <= 4, "expected 2-4 pivoted segments"
    return objects


def _apply_transform(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    palette, _ = _parse_palette(g)
    objects = _extract_objects(g)

    out = g.copy()
    traces: list[dict[str, object]] = []
    for obj in objects:
        color = int(obj["color"])
        for r, c in _rotated_cells(obj):
            assert int(out[r, c]) == Color.BLACK, "rotated segment must land on background"
            out[r, c] = color

        for r, c in obj["segment"]:
            out[r, c] = Color.GREEN

        ray_dr, ray_dc = obj["rotated_direction"]
        free_end = _rotated_cells(obj)[-1]
        ray_len = 0
        stop_kind = "boundary"
        stop_cell: Coord | None = None
        r = free_end[0] + ray_dr
        c = free_end[1] + ray_dc
        while _in_bounds(g.shape, r, c):
            if int(out[r, c]) != Color.BLACK:
                stop_cell = (int(r), int(c))
                stop_kind = "drawn_obstacle" if int(g[r, c]) == Color.BLACK else "input_obstacle"
                break
            out[r, c] = int(palette[ray_len % len(palette)])
            ray_len += 1
            r += ray_dr
            c += ray_dc

        traces.append(
            {
                "pivot": obj["pivot"],
                "color": color,
                "length": int(obj["length"]),
                "direction": obj["direction"],
                "ray_direction": obj["rotated_direction"],
                "ray_len": ray_len,
                "stop_kind": stop_kind,
                "stop_cell": stop_cell,
            }
        )

    if with_trace:
        return out, traces, objects, palette
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pivot-attached segment clockwise, recolor the original bar, then extend a palette-cycled ray from the rotated endpoint until blocked.

    Input structure:
    - The background is black, and the top row contains one contiguous 3-4 cell palette band that defines a left-to-right color cycle.
    - Below the palette are 2-4 grey single-cell pivots, each touching exactly one straight colored segment of length 2-4.
    - Every segment is horizontal or vertical, attached to its pivot at one end, and different pivots are processed in row-major order.

    Rule:
    - Read the palette colors from the top row.
    - For each pivot, find its adjacent segment, rotate that segment 90 degrees clockwise around the pivot, and draw the rotated copy in the segment's original color.
    - Recolor the original input segment green while keeping the grey pivot and palette unchanged.
    - From the rotated segment's free endpoint, extend a ray outward in the same direction, coloring successive cells by the repeating palette until the next step would leave the grid or hit any non-black cell; do not draw the blocking cell.

    Key insight:
    - The ray's start point and heading are not visible directly in the input: they are created by the clockwise rotation, and later rays must treat earlier rotated bars and rays as real obstacles.
    """
    return _apply_transform(grid, with_trace=False)


def _render_grid(height: int, width: int, palette_start: int, palette: list[int], objects: list[ObjectSpec]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, palette_start : palette_start + len(palette)] = np.array(palette, dtype=int)
    for obj in objects:
        pr, pc = obj["pivot"]
        grid[pr, pc] = Color.GREY
        for r, c in _segment_cells(obj["pivot"], obj["direction"], int(obj["length"])):
            grid[r, c] = int(obj["color"])
    return grid


def _cross_touches(cells_a: set[Coord], cells_b: set[Coord]) -> bool:
    for r, c in cells_a:
        if (r, c) in cells_b:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells_b:
                return True
    return False


def _basic_input_ok(
    obj: ObjectSpec,
    existing_objects: list[ObjectSpec],
    shape: tuple[int, int],
    palette_cols: tuple[int, int],
) -> bool:
    input_cells = set(_input_cells(obj))
    rotated_cells = set(_rotated_cells(obj))
    if min(r for r, _ in input_cells) < 2:
        return False
    if any(not _in_bounds(shape, r, c) for r, c in input_cells | rotated_cells):
        return False

    palette_cells = {(0, c) for c in range(int(palette_cols[0]), int(palette_cols[1]) + 1)}
    if _cross_touches(input_cells, palette_cells):
        return False
    if rotated_cells & (input_cells | palette_cells):
        return False

    existing_input = set()
    for other in existing_objects:
        existing_input |= set(_input_cells(other))
    if _cross_touches(input_cells, existing_input):
        return False
    if rotated_cells & existing_input:
        return False

    ray_dr, ray_dc = _clockwise(obj["direction"])
    free_end = _rotated_cells(obj)[-1]
    blockers = existing_input | palette_cells | input_cells
    ray_space = 0
    r = free_end[0] + ray_dr
    c = free_end[1] + ray_dc
    while _in_bounds(shape, r, c) and (r, c) not in blockers:
        ray_space += 1
        r += ray_dr
        c += ray_dc
    return ray_space >= 3


def _build_anchor_pair(
    height: int,
    width: int,
    palette_cols: tuple[int, int],
    palette_len: int,
    colors: list[int],
    bottom_room: int,
) -> list[ObjectSpec] | None:
    min_long_ray = palette_len + 1
    for _ in range(300):
        a_len = random.randint(2, 4)
        b_len = random.randint(2, 4)
        mode = random.choice(("east", "west"))

        if mode == "east":
            a_dir = (-1, 0)
            c_a_min = 2
            c_a_max = width - a_len - min_long_ray - 2
        else:
            a_dir = (1, 0)
            c_a_min = a_len + min_long_ray + 1
            c_a_max = width - 3
        if c_a_min > c_a_max:
            continue

        r_a_min = a_len + 2 if mode == "east" else 2
        r_a_max = min(height - bottom_room - b_len - 8, 10)
        if r_a_min > r_a_max:
            continue

        r_a = random.randint(r_a_min, r_a_max)
        c_a = random.randint(c_a_min, c_a_max)

        if mode == "east":
            target_min = max(c_a + a_len + 3, b_len + 2)
            target_max = width - 2
        else:
            target_min = b_len + 2
            target_max = c_a - a_len - 3
        if target_min > target_max:
            continue
        target_col = random.randint(target_min, target_max)

        r_b_max = height - bottom_room - 2
        free_row_min = r_a + 4
        free_row_max = r_b_max - b_len
        if free_row_min > free_row_max:
            continue

        free_row = random.randint(free_row_min, free_row_max)
        r_b = free_row + b_len

        objects = [
            {"pivot": (r_a, c_a), "direction": a_dir, "length": a_len, "color": int(colors[0])},
            {"pivot": (r_b, target_col), "direction": (0, -1), "length": b_len, "color": int(colors[1])},
        ]
        if not _basic_input_ok(objects[0], [], (height, width), palette_cols):
            continue
        if not _basic_input_ok(objects[1], [objects[0]], (height, width), palette_cols):
            continue

        grid = _render_grid(height, width, int(palette_cols[0]), colors[:palette_len], objects)
        try:
            _, traces, _, _ = _apply_transform(grid, with_trace=True)
        except AssertionError:
            continue
        if traces[0]["stop_kind"] != "boundary":
            continue
        if traces[1]["stop_kind"] != "drawn_obstacle":
            continue
        if traces[1]["stop_cell"] != (r_a, target_col):
            continue
        return objects
    return None


def _sample_extra_object(
    height: int,
    width: int,
    palette_cols: tuple[int, int],
    color: int,
    existing_objects: list[ObjectSpec],
    min_pivot_row: int,
) -> ObjectSpec | None:
    if min_pivot_row > height - 2:
        return None
    for _ in range(400):
        obj = {
            "pivot": (random.randint(min_pivot_row, height - 2), random.randint(1, width - 2)),
            "direction": random.choice(DIRS),
            "length": random.randint(2, 4),
            "color": int(color),
        }
        if _basic_input_ok(obj, existing_objects, (height, width), palette_cols):
            return obj
    return None


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, traces, objects, palette = _apply_transform(grid, with_trace=True)
    except AssertionError:
        return False

    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.any(out != grid):
        return False
    if len(set(palette)) != len(palette):
        return False

    ray_lengths = [int(trace["ray_len"]) for trace in traces]
    if min(ray_lengths) < 3:
        return False
    if max(ray_lengths) <= len(palette):
        return False

    stop_kinds = {str(trace["stop_kind"]) for trace in traces}
    if "boundary" not in stop_kinds or "drawn_obstacle" not in stop_kinds:
        return False

    directions = [tuple(map(int, obj["direction"])) for obj in objects]
    if not any(dr != 0 for dr, _ in directions):
        return False
    if not any(dc != 0 for _, dc in directions):
        return False

    if len({int(trace["length"]) for trace in traces}) < 2 and len(traces) < 3:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 4)
    height = random.randint(max(20, 16 + 2 * object_count), 26)
    width = random.randint(18, 26)
    palette_len = random.randint(3, 4)
    palette_start = random.randint(1, width - palette_len - 2)
    palette = random.sample(DRAW_COLORS, palette_len)
    object_colors = random.sample(DRAW_COLORS, object_count)
    palette_cols = (palette_start, palette_start + palette_len - 1)
    bottom_room = 4 * max(0, object_count - 2)

    anchors = _build_anchor_pair(height, width, palette_cols, palette_len, palette, bottom_room)
    if anchors is None:
        return None

    objects = [
        {**anchors[0], "color": int(object_colors[0])},
        {**anchors[1], "color": int(object_colors[1])},
    ]

    min_row = int(objects[-1]["pivot"][0]) + 2
    for idx in range(2, object_count):
        extra = _sample_extra_object(height, width, palette_cols, object_colors[idx], objects, min_row)
        if extra is None:
            return None
        objects.append(extra)
        min_row = int(extra["pivot"][0]) + 2

    grid = _render_grid(height, width, palette_start, palette, objects)
    if not _validate_candidate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
