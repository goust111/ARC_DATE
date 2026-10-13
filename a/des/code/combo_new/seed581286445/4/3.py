import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE_COLORS = (1, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
ROOM_COUNT_RANGE = (2, 6)
INNER_HEIGHT_RANGE = (2, 4)
INNER_WIDTH_RANGE = (2, 4)
H_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (1, 2)
TOP_GAP_RANGE = (1, 3)
H_GAP_RANGE = (1, 3)
V_GAP_RANGE = (1, 2)


def neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            component: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                component.append((int(r), int(c)))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(component)
    return components


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_palette(grid: np.ndarray) -> tuple[list[int], list[int]]:
    top_row = grid[0]
    palette_cols = [int(c) for c in np.flatnonzero(top_row != BLACK)]
    assert palette_cols, "expected palette pixels on the top row"
    assert all(palette_cols[i + 1] - palette_cols[i] >= 2 for i in range(len(palette_cols) - 1)), (
        "palette pixels must be separated by at least one background cell"
    )

    palette_colors = [int(top_row[c]) for c in palette_cols]
    assert len(palette_colors) == len(set(palette_colors)), "palette colors must be unique"
    return palette_cols, palette_colors


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    rooms: list[dict[str, object]] = []

    for coords in components_mask(grid != GREY):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in coords):
            continue

        r0, c0, r1, c1 = bbox_coords(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(coords) == area, "room interior must be a solid rectangle"
        assert r0 >= 1 and c0 >= 1 and r1 + 1 < height and c1 + 1 < width, "room must have a wall border"

        if not np.all(grid[r0 - 1, c0 - 1 : c1 + 2] == GREY):
            continue
        if not np.all(grid[r1 + 1, c0 - 1 : c1 + 2] == GREY):
            continue
        if not np.all(grid[r0 - 1 : r1 + 2, c0 - 1] == GREY):
            continue
        if not np.all(grid[r0 - 1 : r1 + 2, c1 + 1] == GREY):
            continue

        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert set(np.unique(window)).issubset({BLACK, RED}), "room interior should only contain background and red markers"

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "markers": int(np.count_nonzero(window == RED)),
            }
        )

    assert rooms, "expected at least one enclosed room"
    marker_counts = sorted(int(room["markers"]) for room in rooms)
    assert marker_counts == list(range(1, len(rooms) + 1)), "room marker counts must be exactly 1..N"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the palette color whose left-to-right rank matches the room's red-marker count rank.

    Input structure:
    - Black background surrounds several rectangular rooms enclosed by one-pixel-thick Grey walls.
    - Each room interior contains only Black cells and a unique number of Red marker pixels.
    - The top row contains a left-to-right ordered palette of distinct single-pixel colors separated by Black gaps.

    Rule:
    - Identify every enclosed room by its Grey rectangular border and count the Red markers inside it.
    - Read the palette colors from left to right as ranks 1..N, where N is the number of rooms.
    - Sort the rooms by increasing marker count and fill each room interior with the palette color of the same rank.
    - Overwrite both the Red markers and the Black interior cells, keep Grey walls unchanged, and erase the palette pixels to Black.

    Key insight:
    - The red markers determine an ordering over rooms, while the top-row palette provides a separate spatial ordering of colors; matching these two ranks links each room to its fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_cols, palette_colors = _extract_palette(g)
    rooms = _find_rooms(g)
    assert len(rooms) == len(palette_colors), "palette length must equal the number of rooms"

    out = g.copy()
    ordered_rooms = sorted(rooms, key=lambda room: int(room["markers"]))
    for room, color in zip(ordered_rooms, palette_colors):
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    out[0, palette_cols] = BLACK
    return out


def _random_composition(total: int, parts: int, rng: random.Random, max_part: int) -> list[int] | None:
    for _ in range(100):
        cuts = sorted(rng.sample(range(1, total), parts - 1)) if parts > 1 else []
        values = []
        prev = 0
        for cut in cuts + [total]:
            values.append(cut - prev)
            prev = cut
        if max(values) <= max_part:
            return values
    return None


def _choose_row_count(room_count: int, rng: random.Random) -> int:
    min_rows = max(1, (room_count + 2) // 3)
    options = list(range(min_rows, min(3, room_count) + 1))
    return rng.choice(options)


def _choose_inner_size(room_count: int, rng: random.Random) -> tuple[int, int]:
    for _ in range(100):
        inner_h = rng.randint(*INNER_HEIGHT_RANGE)
        inner_w = rng.randint(*INNER_WIDTH_RANGE)
        if inner_h * inner_w >= room_count:
            return inner_h, inner_w
    raise RuntimeError("failed to sample a valid room size")


def _palette_columns(width: int, count: int, rng: random.Random) -> list[int] | None:
    min_span = 2 * count - 1
    if width < min_span:
        return None

    extra = width - min_span
    gaps = [0] * (count + 1)
    for _ in range(extra):
        gaps[rng.randrange(count + 1)] += 1

    cols: list[int] = []
    col = gaps[0]
    cols.append(col)
    for idx in range(1, count):
        col += 2 + gaps[idx]
        cols.append(col)
    return cols


def _layout_rooms(room_count: int, rng: random.Random) -> tuple[int, int, list[dict[str, int]]] | None:
    row_count = _choose_row_count(room_count, rng)
    counts_per_row = _random_composition(room_count, row_count, rng, max_part=3)
    if counts_per_row is None:
        return None

    room_specs = []
    for room_idx in range(room_count):
        inner_h, inner_w = _choose_inner_size(room_count, rng)
        room_specs.append(
            {
                "id": room_idx,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    rng.shuffle(room_specs)

    rows: list[list[dict[str, int]]] = []
    cursor = 0
    for count in counts_per_row:
        rows.append(room_specs[cursor : cursor + count])
        cursor += count

    left_margin = rng.randint(*H_MARGIN_RANGE)
    right_margin = rng.randint(*H_MARGIN_RANGE)
    top_gap = rng.randint(*TOP_GAP_RANGE)
    bottom_margin = rng.randint(*BOTTOM_MARGIN_RANGE)
    row_band_heights = [max(room["outer_h"] for room in row) for row in rows]
    vertical_gaps = [rng.randint(*V_GAP_RANGE) for _ in range(len(rows) - 1)]

    total_height = 1 + top_gap + sum(row_band_heights) + sum(vertical_gaps) + bottom_margin
    if total_height > 30:
        return None

    row_widths = []
    row_gaps: list[list[int]] = []
    for row in rows:
        gaps = [rng.randint(*H_GAP_RANGE) for _ in range(len(row) - 1)]
        row_gaps.append(gaps)
        row_widths.append(sum(room["outer_w"] for room in row) + sum(gaps))

    total_width = left_margin + max(row_widths) + right_margin
    if total_width > 30:
        return None

    if total_width < 2 * room_count - 1:
        return None

    placements: list[dict[str, int]] = []
    band_top = 1 + top_gap
    for row_idx, row in enumerate(rows):
        band_h = row_band_heights[row_idx]
        used_width = row_widths[row_idx]
        gaps = row_gaps[row_idx]
        row_offset = rng.randint(0, total_width - left_margin - right_margin - used_width)
        left = left_margin + row_offset
        for idx, room in enumerate(row):
            room_top = band_top + rng.randint(0, band_h - room["outer_h"])
            placements.append(
                {
                    "top": room_top,
                    "left": left,
                    "inner_h": room["inner_h"],
                    "inner_w": room["inner_w"],
                }
            )
            left += room["outer_w"]
            if idx < len(gaps):
                left += gaps[idx]
        band_top += band_h
        if row_idx < len(vertical_gaps):
            band_top += vertical_gaps[row_idx]

    return total_height, total_width, placements


def _draw_room(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left : right + 1] = GREY
    grid[top + 1 : bottom, left + 1 : right] = BLACK
    return top + 1, left + 1, bottom - 1, right - 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    layout = _layout_rooms(room_count, rng)
    if layout is None:
        return None

    height, width, placements = layout
    grid = np.full((height, width), BLACK, dtype=int)

    palette_colors = rng.sample(PALETTE_COLORS, room_count)
    palette_cols = _palette_columns(width, room_count, rng)
    if palette_cols is None:
        return None
    for col, color in zip(palette_cols, palette_colors):
        grid[0, col] = color

    marker_counts = list(range(1, room_count + 1))
    rng.shuffle(marker_counts)

    for placement, marker_count in zip(placements, marker_counts):
        r0, c0, r1, c1 = _draw_room(
            grid,
            placement["top"],
            placement["left"],
            placement["inner_h"],
            placement["inner_w"],
        )
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        assert len(cells) >= marker_count
        for r, c in rng.sample(cells, marker_count):
            grid[r, c] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == RED) != room_count * (room_count + 1) // 2:
        return None
    if np.any(out[0] != BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
