from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
AZURE = 8
PALETTE_COLORS = (1, 2, 3, 6, 7, 9)

# Generator bounds.
ROOM_ROWS_RANGE = (1, 4)
ROOM_COLS_RANGE = (1, 4)
ROOM_HEIGHT_RANGE = (2, 5)
ROOM_WIDTH_RANGE = (2, 5)
MIN_ROOMS = 2
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_row(g: np.ndarray) -> tuple[int, int]:
    for r in range(g.shape[0]):
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != BLACK:
            return int(r), int(g[r, 0])
    raise AssertionError("expected a full non-black separator row")


def _palette_colors_from_row(row: np.ndarray) -> list[int]:
    palette = [int(value) for value in row if int(value) != BLACK]
    assert palette, "palette row must contain non-black colors"
    return palette


def _find_wall_layout(region: np.ndarray) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for color in np.unique(region):
        color = int(color)
        if color == BLACK:
            continue
        wall_rows = [int(r) for r in range(region.shape[0]) if np.all(region[r, :] == color)]
        wall_cols = [int(c) for c in range(region.shape[1]) if np.all(region[:, c] == color)]
        if len(wall_rows) >= 2 and len(wall_cols) >= 2:
            score = len(wall_rows) * len(wall_cols)
            candidates.append((score, color, wall_rows, wall_cols))

    assert candidates, "expected a wall color spanning full rows and columns"
    _score, wall_color, wall_rows, wall_cols = max(candidates, key=lambda item: item[0])
    wall_rows = sorted(set(wall_rows))
    wall_cols = sorted(set(wall_cols))
    assert wall_rows[0] == 0 and wall_rows[-1] == region.shape[0] - 1, "rooms need an outer wall"
    assert wall_cols[0] == 0 and wall_cols[-1] == region.shape[1] - 1, "rooms need an outer wall"
    return int(wall_color), wall_rows, wall_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the yellow objects inside each closed room, use that count to look up a
    color in the palette row, and fill the room's black floor with the looked-up
    color while preserving the yellow objects, the azure walls, and the top bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row, _separator_color = _find_separator_row(g)
    assert separator_row > 0, "separator row must sit below the palette row"
    palette = _palette_colors_from_row(g[separator_row - 1, :])

    region = g[separator_row + 1 :, :]
    assert region.size > 0, "expected a room region below the separator"
    wall_color, wall_rows, wall_cols = _find_wall_layout(region)

    object_colors = sorted(
        int(color) for color in np.unique(region) if int(color) not in (BLACK, wall_color)
    )
    assert len(object_colors) == 1, "expected one object color inside the rooms"
    object_color = object_colors[0]

    out = g.copy()
    out_region = out[separator_row + 1 :, :]

    for r0, r1 in zip(wall_rows, wall_rows[1:], strict=False):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in zip(wall_cols, wall_cols[1:], strict=False):
            if c1 - c0 <= 1:
                continue
            room = region[r0 + 1 : r1, c0 + 1 : c1]
            yellow_count = int(np.count_nonzero(room == object_color))
            assert 1 <= yellow_count <= len(palette), "room object count must index the palette"
            fill_color = int(palette[yellow_count - 1])

            room_out = out_region[r0 + 1 : r1, c0 + 1 : c1]
            room_out[room == BLACK] = fill_color

    return out


def _choose_room_shape() -> tuple[int, int]:
    while True:
        room_rows = random.randint(*ROOM_ROWS_RANGE)
        room_cols = random.randint(*ROOM_COLS_RANGE)
        if room_rows * room_cols >= MIN_ROOMS:
            return room_rows, room_cols


def _line_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _choose_room_counts(room_areas: list[int], palette_len: int) -> list[int]:
    max_counts = [min(int(area) - 1, int(palette_len)) for area in room_areas]
    assert all(count >= 1 for count in max_counts), "each room must have space for floor and objects"

    counts: list[int | None] = [None] * len(room_areas)

    if len(room_areas) >= 3 and random.random() < 0.6:
        shared = random.randint(1, min(max_counts))
        shared_slots = random.sample(range(len(room_areas)), random.randint(2, len(room_areas)))
        for idx in shared_slots:
            counts[idx] = shared

    if random.random() < 0.7:
        target = max(max_counts)
        capable = [idx for idx, limit in enumerate(max_counts) if limit == target]
        counts[random.choice(capable)] = target

    if random.random() < 0.7:
        target = 1
        capable = [idx for idx, limit in enumerate(max_counts) if limit >= target]
        counts[random.choice(capable)] = target

    for idx, limit in enumerate(max_counts):
        if counts[idx] is None:
            if random.random() < 0.35:
                counts[idx] = random.choice((1, limit))
            else:
                counts[idx] = random.randint(1, limit)

    final_counts = [int(value) for value in counts]
    if len(room_areas) > 1 and len(set(final_counts)) == 1:
        alternatives = [
            idx for idx, limit in enumerate(max_counts) if any(v != final_counts[idx] for v in range(1, limit + 1))
        ]
        if alternatives:
            idx = random.choice(alternatives)
            choices = [value for value in range(1, max_counts[idx] + 1) if value != final_counts[idx]]
            final_counts[idx] = random.choice(choices)
    return final_counts


def _room_top_lefts(room_heights: list[int], room_widths: list[int]) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    row_start = 1
    for height in room_heights:
        col_start = 1
        for width in room_widths:
            starts.append((row_start, col_start))
            col_start += int(width) + 1
        row_start += int(height) + 1
    return starts


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    room_rows, room_cols = _choose_room_shape()
    room_heights = [random.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    region_height = sum(room_heights) + room_rows + 1
    region_width = sum(room_widths) + room_cols + 1

    palette_max = min(len(PALETTE_COLORS), (region_width + 1) // 2)
    if palette_max < 2:
        return None
    palette_len = random.randint(2, palette_max)
    palette = random.sample(PALETTE_COLORS, palette_len)

    room_areas = [
        int(height) * int(width)
        for height in room_heights
        for width in room_widths
    ]
    counts = _choose_room_counts(room_areas, palette_len)

    inp = np.full((region_height + 2, region_width), BLACK, dtype=int)
    out = inp.copy()
    inp[1, :] = GREY
    out[1, :] = GREY

    palette_span = 2 * palette_len - 1
    palette_start = random.randint(0, region_width - palette_span)
    for idx, color in enumerate(palette):
        col = palette_start + 2 * idx
        inp[0, col] = int(color)
        out[0, col] = int(color)

    region_in = inp[2:, :]
    region_out = out[2:, :]
    region_in[:, :] = BLACK
    region_out[:, :] = BLACK

    wall_rows = _line_positions(room_heights)
    wall_cols = _line_positions(room_widths)
    region_in[wall_rows, :] = AZURE
    region_in[:, wall_cols] = AZURE
    region_out[wall_rows, :] = AZURE
    region_out[:, wall_cols] = AZURE

    for (top, left), room_h, room_w, count in zip(
        _room_top_lefts(room_heights, room_widths),
        [height for height in room_heights for _ in room_widths],
        room_widths * room_rows,
        counts,
        strict=False,
    ):
        cells = [(r, c) for r in range(top, top + int(room_h)) for c in range(left, left + int(room_w))]
        yellow_cells = random.sample(cells, int(count))
        for r, c in yellow_cells:
            region_in[r, c] = YELLOW
            region_out[r, c] = YELLOW
        fill_color = int(palette[int(count) - 1])
        for r, c in cells:
            if region_out[r, c] == BLACK:
                region_out[r, c] = fill_color

    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
