import random

import numpy as np


WALL_COLOR = 1
SEED_COLOR = 2
BG_COLOR = 0
PALETTE_POOL = [3, 4, 5, 6, 7, 8, 9]

ROOM_LAYOUTS = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2)]
ROOM_MIN_SIZE = 4
ROOM_MAX_SIZE = 10
OUTER_MARGIN_MIN = 2
WALL_THICKNESS = 1
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_palette(grid: np.ndarray, bg: int) -> list[int]:
    mask = (grid != bg) & (grid != WALL_COLOR) & (grid != SEED_COLOR)
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one palette strip"
    comp = comps[0]
    rows = {r for r, _ in comp}
    assert len(rows) == 1, "palette must be horizontal"
    cols = sorted(c for _, c in comp)
    assert cols == list(range(cols[0], cols[-1] + 1)), "palette must be contiguous"
    row = next(iter(rows))
    palette = [int(grid[row, c]) for c in cols]
    assert 2 <= len(palette) <= 5
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette


def _extract_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    passable = grid != WALL_COLOR
    rooms: list[dict[str, object]] = []
    for comp in _components(passable):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        values = {int(grid[r, c]) for r, c in comp}
        assert values <= {bg, SEED_COLOR}, "rooms may only contain background and red seeds"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(comp) == area, "each room must be a solid rectangle"
        seeds = [(r, c) for r, c in comp if int(grid[r, c]) == SEED_COLOR]
        assert 1 <= len(seeds) <= 5
        rooms.append({"bbox": (r0, r1, c0, c1), "seeds": seeds})
    assert rooms, "expected at least one enclosed room"
    return sorted(rooms, key=lambda room: room["bbox"])


def _spiral_coords(height: int, width: int, loops: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = height - 1, width - 1
    rings = 0
    while top <= bottom and left <= right and rings < loops:
        for c in range(left, right + 1):
            coords.append((top, c))
        for r in range(top + 1, bottom + 1):
            coords.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                coords.append((r, left))
        top += 1
        left += 1
        bottom -= 1
        right -= 1
        rings += 1
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with a palette-selected spiral whose color and ring count are both determined by the number of red seeds inside that room.

    Input structure:
    - The background is the most common color, with blue wall pixels splitting the canvas into multiple closed rectangular rooms.
    - Each room contains only background cells plus between one and five red single-pixel seed markers.
    - A horizontal palette strip of two to five distinct colors sits in one corner outside the rooms.

    Rule:
    - For each room, count its red seeds as N and select the N-th palette color from left to right.
    - Starting at the room's top-left cell, draw a clockwise rectangular spiral with arm width one and N loops, truncating naturally if the room is too small.
    - Do not overwrite the original red seed cells; walls and the palette remain unchanged.

    Key insight:
    - The same visible quantity, seed count, simultaneously chooses both the spiral's color via palette indexing and the spiral's loop count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    palette = _extract_palette(g, bg)
    rooms = _extract_rooms(g, bg)

    out = g.copy()
    for room in rooms:
        r0, r1, c0, c1 = room["bbox"]
        seed_count = len(room["seeds"])
        assert seed_count <= len(palette), "palette must cover all seed counts"
        color = palette[seed_count - 1]
        for dr, dc in _spiral_coords(r1 - r0 + 1, c1 - c0 + 1, seed_count):
            rr, cc = r0 + dr, c0 + dc
            if int(g[rr, cc]) == SEED_COLOR:
                continue
            out[rr, cc] = color
    return out


def _random_partition(total: int, count: int, min_value: int, max_value: int, rng: random.Random) -> list[int] | None:
    parts: list[int] = []
    remaining = total
    for idx in range(count):
        slots_left = count - idx - 1
        low = max(min_value, remaining - slots_left * max_value)
        high = min(max_value, remaining - slots_left * min_value)
        if low > high:
            return None
        value = rng.randint(low, high)
        parts.append(value)
        remaining -= value
    rng.shuffle(parts)
    return parts


def _random_sizes(count: int, rng: random.Random, max_total: int) -> list[int] | None:
    min_total = count * ROOM_MIN_SIZE
    max_total = min(count * ROOM_MAX_SIZE, max_total)
    if min_total > max_total:
        return None
    total = rng.randint(min_total, max_total)
    return _random_partition(total, count, ROOM_MIN_SIZE, ROOM_MAX_SIZE, rng)


def _build_room_boxes(
    top_margin: int,
    left_margin: int,
    room_heights: list[int],
    room_widths: list[int],
) -> list[tuple[int, int, int, int]]:
    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []

    cursor = top_margin + WALL_THICKNESS
    for height in room_heights:
        row_spans.append((cursor, cursor + height - 1))
        cursor += height + WALL_THICKNESS

    cursor = left_margin + WALL_THICKNESS
    for width in room_widths:
        col_spans.append((cursor, cursor + width - 1))
        cursor += width + WALL_THICKNESS

    boxes = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            boxes.append((r0, r1, c0, c1))
    return boxes


def _draw_walls(grid: np.ndarray, top_margin: int, left_margin: int, room_heights: list[int], room_widths: list[int]) -> None:
    outer_h = sum(room_heights) + (len(room_heights) + 1) * WALL_THICKNESS
    outer_w = sum(room_widths) + (len(room_widths) + 1) * WALL_THICKNESS

    row = top_margin
    grid[row, left_margin : left_margin + outer_w] = WALL_COLOR
    for height in room_heights:
        row += WALL_THICKNESS + height
        grid[row, left_margin : left_margin + outer_w] = WALL_COLOR

    col = left_margin
    grid[top_margin : top_margin + outer_h, col] = WALL_COLOR
    for width in room_widths:
        col += WALL_THICKNESS + width
        grid[top_margin : top_margin + outer_h, col] = WALL_COLOR


def _choose_seed_cells(room_box: tuple[int, int, int, int], count: int, rng: random.Random) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = room_box
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in corners]
    if len(candidates) < count:
        return []
    return rng.sample(candidates, count)


def _place_palette(grid: np.ndarray, palette: list[int], corner: str) -> None:
    h, w = grid.shape
    if corner == "top_left":
        row, start = 0, 0
    elif corner == "top_right":
        row, start = 0, w - len(palette)
    elif corner == "bottom_left":
        row, start = h - 1, 0
    else:
        row, start = h - 1, w - len(palette)
    grid[row, start : start + len(palette)] = palette


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows, room_cols = rng.choice(ROOM_LAYOUTS)
    room_count = room_rows * room_cols
    palette_len = rng.randint(2, min(5, room_count))

    max_room_height_total = 30 - 2 * OUTER_MARGIN_MIN - (room_rows + 1) * WALL_THICKNESS
    max_room_width_total = 30 - 2 * OUTER_MARGIN_MIN - (room_cols + 1) * WALL_THICKNESS
    room_heights = _random_sizes(room_rows, rng, max_room_height_total)
    room_widths = _random_sizes(room_cols, rng, max_room_width_total)
    if room_heights is None or room_widths is None:
        return None

    outer_h = sum(room_heights) + (room_rows + 1) * WALL_THICKNESS
    outer_w = sum(room_widths) + (room_cols + 1) * WALL_THICKNESS
    max_extra_h = 30 - outer_h
    max_extra_w = 30 - outer_w
    if max_extra_h < 2 * OUTER_MARGIN_MIN or max_extra_w < 2 * OUTER_MARGIN_MIN:
        return None

    total_extra_h = rng.randint(2 * OUTER_MARGIN_MIN, max_extra_h)
    total_extra_w = rng.randint(2 * OUTER_MARGIN_MIN, max_extra_w)
    top_margin = rng.randint(OUTER_MARGIN_MIN, total_extra_h - OUTER_MARGIN_MIN)
    bottom_margin = total_extra_h - top_margin
    left_margin = rng.randint(OUTER_MARGIN_MIN, total_extra_w - OUTER_MARGIN_MIN)
    right_margin = total_extra_w - left_margin

    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin
    grid = np.full((height, width), BG_COLOR, dtype=int)
    _draw_walls(grid, top_margin, left_margin, room_heights, room_widths)

    room_boxes = _build_room_boxes(top_margin, left_margin, room_heights, room_widths)
    counts = list(range(1, palette_len + 1))
    while len(counts) < len(room_boxes):
        counts.append(rng.randint(1, palette_len))
    rng.shuffle(counts)

    palette = rng.sample(PALETTE_POOL, palette_len)
    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    _place_palette(grid, palette, corner)

    for room_box, seed_count in zip(room_boxes, counts):
        seeds = _choose_seed_cells(room_box, seed_count, rng)
        if len(seeds) != seed_count:
            return None
        for r, c in seeds:
            grid[r, c] = SEED_COLOR

    try:
        bg = _most_common_color(grid)
        palette_check = _extract_palette(grid, bg)
        rooms = _extract_rooms(grid, bg)
        assert palette_check == palette
        assert sorted(len(room["seeds"]) for room in rooms) == sorted(counts)
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-spiral ARC input")
