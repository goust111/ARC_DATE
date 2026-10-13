from __future__ import annotations

import random

import numpy as np


BG = 0
WALL_COLOR = 8
SEPARATOR_COLOR = 9
TEMPLATE_COLORS = (1, 2, 3, 4, 5, 6, 7)
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400
RNG = random.Random()


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    height = int(grid.shape[0])
    for row in range(height - 1):
        value = int(grid[row, 0])
        if value == BG:
            continue
        if np.all(grid[row] == value) and np.any(grid[:row] != BG) and np.any(grid[row + 1 :] != BG):
            candidates.append((row, value))
    assert candidates, "expected one full-width separator row"
    return candidates[-1]


def _parse_templates(template_area: np.ndarray, separator_color: int) -> list[np.ndarray]:
    height, width = map(int, template_area.shape)
    separator_cols = [col for col in range(width) if np.all(template_area[:, col] == separator_color)]
    templates: list[np.ndarray] = []
    start = 0
    for stop in separator_cols + [width]:
        if start < stop:
            block = template_area[:, start:stop]
            if np.any(block != BG):
                assert not np.any(block == separator_color), "separator color must only appear in full separator columns"
                assert int(block.shape[0]) == int(block.shape[1]), "template blocks must be square"
                templates.append(block.copy())
        start = stop + 1
    assert len(templates) == 3, "expected exactly three templates below the separator"
    return templates


def _room_from_bbox(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, wall_color: int
) -> dict | None:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    if (
        int(grid[r0, c0]) != wall_color
        or int(grid[r0, c1]) != wall_color
        or int(grid[r1, c0]) != wall_color
        or int(grid[r1, c1]) != wall_color
    ):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != BG):
        return None

    gap_count = 0
    for segment in (
        grid[r0, c0 + 1 : c1],
        grid[r1, c0 + 1 : c1],
        grid[r0 + 1 : r1, c0],
        grid[r0 + 1 : r1, c1],
    ):
        if np.any((segment != wall_color) & (segment != BG)):
            return None
        gap_idx = np.flatnonzero(segment == BG)
        if gap_idx.size and np.any(np.diff(gap_idx) == 1):
            return None
        gap_count += int(gap_idx.size)

    if not (1 <= gap_count <= 3):
        return None

    return {"bbox": (r0, c0, r1, c1), "doors": gap_count}


def _find_rooms(top_area: np.ndarray, wall_color: int) -> list[dict]:
    height, width = map(int, top_area.shape)
    wall_mask = top_area == wall_color
    rows_by_col = [set(map(int, np.flatnonzero(wall_mask[:, col]))) for col in range(width)]
    seen: set[tuple[int, int, int, int]] = set()
    rooms: list[dict] = []

    for r0 in range(height - 3):
        top_cols = list(map(int, np.flatnonzero(wall_mask[r0])))
        for idx, c0 in enumerate(top_cols):
            for c1 in top_cols[idx + 1 :]:
                if c1 - c0 < 3:
                    continue
                common_rows = sorted(row for row in rows_by_col[c0].intersection(rows_by_col[c1]) if row >= r0 + 3)
                for r1 in common_rows:
                    bbox = (r0, c0, r1, c1)
                    if bbox in seen:
                        continue
                    room = _room_from_bbox(top_area, r0, c0, r1, c1, wall_color)
                    if room is None:
                        continue
                    seen.add(bbox)
                    rooms.append(room)

    assert rooms, "expected at least one room in the top area"
    rooms.sort(key=lambda room: tuple(map(int, room["bbox"])))

    covered = np.zeros_like(top_area, dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = map(int, room["bbox"])
        covered[r0, c0 : c1 + 1] = True
        covered[r1, c0 : c1 + 1] = True
        covered[r0 : r1 + 1, c0] = True
        covered[r0 : r1 + 1, c1] = True
    assert np.all(~wall_mask | covered), "all wall pixels must belong to detected room borders"

    return rooms


def _scale_nearest(template: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    src_rows = (np.arange(out_height) * template.shape[0]) // out_height
    src_cols = (np.arange(out_width) * template.shape[1]) // out_width
    return template[src_rows[:, None], src_cols[None, :]]


def _fill_rooms(grid: np.ndarray, rooms: list[dict], templates: list[np.ndarray]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for room in rooms:
        r0, c0, r1, c1 = map(int, room["bbox"])
        template = templates[int(room["doors"]) - 1]
        scaled = _scale_nearest(template, int(r1 - r0 - 1), int(c1 - c0 - 1))
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the scaled bottom template selected by its door count.

    Input structure:
    - Black is the background.
    - The upper area contains one-pixel-thick rectangular room walls in one non-black color; each room has an otherwise black interior and 1 to 3 one-pixel wall gaps acting as doors.
    - A full-width non-black separator row sits above a bottom template strip that contains three square template blocks from left to right, with full-height separator columns between neighboring templates.

    Rule:
    - Detect every rectangular room in the upper area and count how many one-pixel gaps appear on its perimeter.
    - Use that count N to choose the Nth template block from the bottom strip, reading templates left to right.
    - Scale the whole chosen square template to the room's interior size with nearest-neighbor sampling and overwrite the room interior with that scaled pattern.
    - Leave the room walls, the door gaps, the separator row, and the template strip unchanged.

    Key insight:
    - The missing wall cells are not filled directly; they serve as the index that selects which separate bottom template must be expanded into the room.
    """
    g = np.asarray(grid, dtype=int)
    separator_row, separator_color = _find_separator_row(g)
    templates = _parse_templates(g[separator_row + 1 :], separator_color)

    top_area = g[:separator_row]
    wall_values = [int(value) for value in np.unique(top_area) if int(value) != BG]
    assert len(wall_values) == 1, "top area must contain a single wall color"
    rooms = _find_rooms(top_area, wall_values[0])

    return _fill_rooms(g, rooms, templates)


def _template_diag(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    for idx in range(size):
        template[idx, idx] = colors[0]
        template[idx, size - 1 - idx] = colors[1]
    template[size // 2, size // 2] = colors[2]
    return template


def _template_cross(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    mid = size // 2
    template[mid, :] = colors[0]
    template[:, mid] = colors[0]
    template[0, 0] = colors[1]
    template[0, -1] = colors[1]
    template[-1, 0] = colors[1]
    template[-1, -1] = colors[1]
    template[mid, mid] = colors[2]
    return template


def _template_frame(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    template[0, :] = colors[0]
    template[-1, :] = colors[0]
    template[:, 0] = colors[0]
    template[:, -1] = colors[0]
    for idx in range(1, size - 1):
        template[idx, idx] = colors[1]
    if size > 3:
        template[1:-1, size // 2] = colors[2]
    return template


def _template_checker(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    for row in range(size):
        for col in range(size):
            if (row + col) % 2 == 0:
                template[row, col] = colors[0]
    for idx in range(size):
        template[idx, size - 1 - idx] = colors[1]
    template[size // 2, size // 2] = colors[2]
    return template


def _template_bands(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    template[:, ::2] = colors[0]
    template[size // 2, :] = colors[1]
    if size > 3:
        template[1:-1, size // 2] = colors[2]
    return template


def _template_diamond(size: int, colors: tuple[int, int, int]) -> np.ndarray:
    template = np.zeros((size, size), dtype=int)
    mid = (size - 1) / 2.0
    radius = max(1, size // 2)
    for row in range(size):
        for col in range(size):
            dist = abs(row - mid) + abs(col - mid)
            if dist <= radius - 0.25:
                template[row, col] = colors[0]
            if abs(dist - radius) < 0.35:
                template[row, col] = colors[1]
    template[size // 2, size // 2] = colors[2]
    return template


TEMPLATE_BUILDERS = (
    _template_diag,
    _template_cross,
    _template_frame,
    _template_checker,
    _template_bands,
    _template_diamond,
)


def _transform_template(template: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(template, copy=True)
    out = np.rot90(out, int(rng.randrange(4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _make_templates(rng: random.Random, size: int) -> list[np.ndarray]:
    builders = list(TEMPLATE_BUILDERS)
    rng.shuffle(builders)
    templates: list[np.ndarray] = []
    used: list[np.ndarray] = []
    for builder in builders:
        colors = tuple(rng.sample(TEMPLATE_COLORS, 3))
        template = _transform_template(builder(size, colors), rng)
        if not np.any(template != BG):
            continue
        if any(np.array_equal(template, other) for other in used):
            continue
        used.append(template)
        templates.append(template)
        if len(templates) == 3:
            break
    assert len(templates) == 3, "failed to build three distinct templates"
    return templates


def _door_positions(
    rng: random.Random, bbox: tuple[int, int, int, int], door_count: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    positions = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    chosen_sides = list(rng.sample(SIDES, door_count))
    return [rng.choice(positions[side]) for side in chosen_sides]


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int], door_count: int, rng: random.Random) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR
    for row, col in _door_positions(rng, bbox, door_count):
        grid[row, col] = BG


def _layout_rooms(
    rng: random.Random, room_specs: list[dict], template_width: int, template_size: int
) -> tuple[int, int, list[dict]] | None:
    specs = [dict(spec) for spec in room_specs]
    rng.shuffle(specs)

    minimal_row_width = sum(int(spec["box_w"]) for spec in specs) + max(0, len(specs) - 1)
    if len(specs) >= 4 or minimal_row_width > 24:
        row_count = 2
    else:
        row_count = 1 if rng.random() < 0.55 else 2

    rows: list[list[dict]] = [[] for _ in range(row_count)]
    row_weight = [0] * row_count
    for spec in sorted(specs, key=lambda item: int(item["box_w"]), reverse=True):
        row_index = min(range(row_count), key=lambda idx: (row_weight[idx], len(rows[idx])))
        rows[row_index].append(spec)
        row_weight[row_index] += int(spec["box_w"])

    for row in rows:
        rng.shuffle(row)

    horizontal_gaps = [[rng.randint(1, 3) for _ in range(max(0, len(row) - 1))] for row in rows]
    row_widths = [
        sum(int(spec["box_w"]) for spec in row) + sum(gaps) for row, gaps in zip(rows, horizontal_gaps)
    ]
    row_heights = [max(int(spec["box_h"]) for spec in row) for row in rows]

    outer_left = rng.randint(1, 2)
    outer_right = rng.randint(1, 2)
    width_min = max(template_width, max(row_widths) + outer_left + outer_right)
    width_slack = 30 - width_min
    if width_slack < 0:
        return None
    width = width_min + rng.randint(0, min(4, width_slack))

    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    vertical_gaps = [rng.randint(1, 3) for _ in range(max(0, row_count - 1))]
    top_height = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    if top_height + 1 + template_size > 30:
        return None

    placements: list[dict] = []
    current_top = top_margin
    for row_index, row in enumerate(rows):
        row_height = row_heights[row_index]
        row_width = row_widths[row_index]
        start_col = rng.randint(outer_left, width - outer_right - row_width)
        current_col = start_col
        for spec_index, spec in enumerate(row):
            y_offset = rng.randint(0, row_height - int(spec["box_h"]))
            bbox = (
                current_top + y_offset,
                current_col,
                current_top + y_offset + int(spec["box_h"]) - 1,
                current_col + int(spec["box_w"]) - 1,
            )
            placements.append({"bbox": bbox, "doors": int(spec["doors"])})
            current_col += int(spec["box_w"])
            if spec_index < len(row) - 1:
                current_col += int(horizontal_gaps[row_index][spec_index])
        current_top += row_height
        if row_index < len(vertical_gaps):
            current_top += int(vertical_gaps[row_index])

    return width, top_height, placements


def _paint_template_strip(
    width: int, template_size: int, templates: list[np.ndarray], rng: random.Random
) -> np.ndarray:
    strip = np.full((template_size, width), SEPARATOR_COLOR, dtype=int)
    block_width = 3 * template_size + 2
    start = rng.randint(0, width - block_width)
    for idx, template in enumerate(templates):
        col0 = start + idx * (template_size + 1)
        strip[:, col0 : col0 + template_size] = template
        if idx < 2:
            strip[:, col0 + template_size] = SEPARATOR_COLOR
    return strip


def _expected_output(grid: np.ndarray, rooms: list[dict], templates: list[np.ndarray]) -> np.ndarray:
    return _fill_rooms(grid, rooms, templates)


def _gen_attempt() -> np.ndarray | None:
    template_size = RNG.randint(3, 5)
    templates = _make_templates(RNG, template_size)

    room_count = RNG.randint(3, 5)
    door_counts = [1, 2, 3] + [RNG.choice((1, 2, 3)) for _ in range(room_count - 3)]
    RNG.shuffle(door_counts)

    room_specs = []
    for door_count in door_counts:
        inner_height = RNG.randint(3, 6)
        inner_width = RNG.randint(3, 6)
        room_specs.append(
            {
                "doors": door_count,
                "box_h": inner_height + 2,
                "box_w": inner_width + 2,
            }
        )

    layout = _layout_rooms(RNG, room_specs, 3 * template_size + 2, template_size)
    if layout is None:
        return None
    width, top_height, rooms = layout

    top = np.zeros((top_height, width), dtype=int)
    for room in rooms:
        _draw_room(top, tuple(room["bbox"]), int(room["doors"]), RNG)

    separator = np.full((1, width), SEPARATOR_COLOR, dtype=int)
    template_strip = _paint_template_strip(width, template_size, templates, RNG)
    grid = np.vstack((top, separator, template_strip))

    expected = _expected_output(grid, rooms, templates)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-template ARC input")
