from __future__ import annotations

from collections import deque
from itertools import combinations_with_replacement
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

WINDOW_COLORS = (BLUE, GREEN, RED)
PAINT_COLORS = (YELLOW, ORANGE, MAGENTA, AZURE)
DISPLAY_ORDER = {BLUE: 0, GREEN: 1, RED: 2}

ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_SIZE_RANGES = {
    2: (5, 8),
    3: (4, 7),
    4: (4, 6),
}
RIGHT_WIDTH_RANGE = (6, 9)
GEN_RETRIES = 400

ALL_KEYS_BY_SIZE = {
    size: [tuple(combo) for combo in combinations_with_replacement(WINDOW_COLORS, size)]
    for size in (1, 2, 3)
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _multiset_key(colors: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(color) for color in colors))


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if not np.any(grid[:, :c] != color):
            continue
        if not np.any(grid[:, c + 1 :] != color):
            continue
        candidates.append((c, color))
    assert len(candidates) == 1
    return candidates[0]


def _extract_palette(right: np.ndarray, bg: int) -> dict[tuple[int, ...], int]:
    mapping: dict[tuple[int, ...], int] = {}
    h, _w = right.shape
    for r in range(h):
        cols = np.flatnonzero(right[r] != bg)
        if len(cols) == 0:
            continue
        groups: list[tuple[int, int]] = []
        start = prev = int(cols[0])
        for col in cols[1:]:
            col = int(col)
            if col == prev + 1:
                prev = col
            else:
                groups.append((start, prev))
                start = prev = col
        groups.append((start, prev))

        assert len(groups) == 2
        key_start, key_end = groups[0]
        fill_start, fill_end = groups[1]
        assert fill_start == fill_end
        assert fill_start - key_end == 2

        key_colors = [int(v) for v in right[r, key_start : key_end + 1].tolist()]
        assert 1 <= len(key_colors) <= 3
        fill_color = int(right[r, fill_start])
        key = _multiset_key(key_colors)
        assert key not in mapping
        mapping[key] = fill_color

    assert mapping
    return mapping


def _extract_rooms(left: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int, int, int, tuple[int, ...]]]:
    rooms: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    for pts in _components(left != bg):
        r0, c0, r1, c1 = _bbox(pts)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        hh, ww = sub.shape
        assert hh >= 3 and ww >= 3

        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(np.count_nonzero(sub != bg)) == int(border.sum())
        assert np.all(sub[border] != bg)
        assert np.all(sub[~border] == bg)

        windows = [int(v) for v in sub[border].tolist() if int(v) != frame_color]
        assert 1 <= len(windows) <= 3
        rooms.append((r0, c0, r1, c1, _multiset_key(windows)))

    assert rooms
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each rectangular room floor with the palette color keyed by its border-window multiset.

    Input structure:
    - The background is black.
    - A full-height grey separator column splits the grid into a left room area and a right palette area.
    - The left area contains several non-overlapping grey rectangular room frames whose interiors are black; each frame has 1-3 colored border pixels replacing grey border cells.
    - The right area contains palette rows; each occupied row shows 1-3 contiguous colored key pixels, one black gap, and one paint-color pixel.

    Rule:
    - For each room, collect the colors of its border window pixels as a multiset, ignoring their positions.
    - Find the palette row whose left key pixels have the exact same color multiset.
    - Fill every black interior cell of that room with the palette row's paint color.
    - Leave the room borders, window pixels, separator, and palette unchanged.

    Key insight:
    - The colored border pixels on each room are not decoration; they form a lookup key that must be matched exactly against the palette's key rows to recover the room's fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, frame_color = _find_separator(g, bg)

    mapping = _extract_palette(g[:, sep_col + 1 :], bg)
    rooms = _extract_rooms(g[:, :sep_col], bg, frame_color)

    out = g.copy()
    for r0, c0, r1, c1, key in rooms:
        assert key in mapping
        fill_color = mapping[key]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = fill_color
    return out


def _display_key(key: tuple[int, ...]) -> list[int]:
    return sorted((int(color) for color in key), key=lambda color: DISPLAY_ORDER[color])


def _border_cycle(height: int, width: int) -> list[tuple[int, int]]:
    cells = [(0, c) for c in range(width)]
    cells.extend((r, width - 1) for r in range(1, height - 1))
    cells.extend((height - 1, c) for c in range(width - 1, -1, -1))
    cells.extend((r, 0) for r in range(height - 2, 0, -1))
    return cells


def _sample_window_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cycle = _border_cycle(height, width)
    indices = list(range(len(cycle)))
    for _ in range(50):
        pick = random.sample(indices, count)
        ok = True
        for i, a in enumerate(pick):
            for b in pick[i + 1 :]:
                dist = abs(a - b)
                if min(dist, len(cycle) - dist) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return [cycle[i] for i in pick]
    return [cycle[i] for i in random.sample(indices, count)]


def _draw_room(grid: np.ndarray, top: int, left: int, height: int, width: int, key: tuple[int, ...]) -> None:
    grid[top, left : left + width] = GREY
    grid[top + height - 1, left : left + width] = GREY
    grid[top : top + height, left] = GREY
    grid[top : top + height, left + width - 1] = GREY

    colors = list(key)
    random.shuffle(colors)
    for color, (dr, dc) in zip(colors, _sample_window_positions(height, width, len(colors)), strict=False):
        grid[top + dr, left + dc] = int(color)


def _draw_palette_row(
    grid: np.ndarray,
    row: int,
    right_left: int,
    right_width: int,
    key: tuple[int, ...],
    fill_color: int,
) -> None:
    shown = _display_key(key)
    used = len(shown) + 2
    start = random.randint(0, right_width - used)
    base = right_left + start
    for i, color in enumerate(shown):
        grid[row, base + i] = int(color)
    grid[row, base + len(shown) + 1] = int(fill_color)


def _choose_room_keys(room_count: int) -> list[tuple[int, ...]]:
    if room_count == 2:
        sizes = random.sample([1, 2, 3], 2)
    elif room_count == 3:
        sizes = [1, 2, 3]
        random.shuffle(sizes)
    else:
        sizes = [1, 2, 3, random.choice([1, 2, 3])]
        random.shuffle(sizes)

    used: set[tuple[int, ...]] = set()
    keys: list[tuple[int, ...]] = []
    for size in sizes:
        choices = [key for key in ALL_KEYS_BY_SIZE[size] if key not in used]
        key = random.choice(choices)
        used.add(key)
        keys.append(key)
    random.shuffle(keys)
    return keys


def _place_rooms(height: int, left_width: int, rooms: list[dict[str, object]]) -> list[dict[str, object]] | None:
    occupied = np.zeros((height, left_width), dtype=bool)
    placed: list[dict[str, object]] = []
    order = sorted(
        rooms,
        key=lambda room: int(room["height"]) * int(room["width"]),
        reverse=True,
    )

    for room in order:
        room_h = int(room["height"])
        room_w = int(room["width"])
        choices: list[tuple[int, int]] = []
        for r in range(1, height - room_h):
            for c in range(1, left_width - room_w):
                rr0 = r - 1
                cc0 = c - 1
                rr1 = r + room_h + 1
                cc1 = c + room_w + 1
                if not np.any(occupied[rr0:rr1, cc0:cc1]):
                    choices.append((r, c))
        if not choices:
            return None
        top, left = random.choice(choices)
        occupied[top - 1 : top + room_h + 1, left - 1 : left + room_w + 1] = True
        placed.append(
            {
                "top": top,
                "left": left,
                "height": room_h,
                "width": room_w,
                "key": room["key"],
                "fill": room["fill"],
            }
        )
    return placed


def _expected_output(grid: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        top = int(room["top"])
        left = int(room["left"])
        height = int(room["height"])
        width = int(room["width"])
        fill = int(room["fill"])
        out[top + 1 : top + height - 1, left + 1 : left + width - 1] = fill
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        room_count = random.choice(ROOM_COUNT_CHOICES)
        key_list = _choose_room_keys(room_count)
        fill_list = random.sample(PAINT_COLORS, room_count)

        size_lo, size_hi = ROOM_SIZE_RANGES[room_count]
        rooms: list[dict[str, object]] = []
        for key, fill in zip(key_list, fill_list, strict=False):
            rooms.append(
                {
                    "height": random.randint(size_lo, size_hi),
                    "width": random.randint(size_lo, size_hi),
                    "key": key,
                    "fill": fill,
                }
            )

        right_width = random.randint(*RIGHT_WIDTH_RANGE)
        left_width_max = 30 - 1 - right_width
        left_width_min = max(max(int(room["width"]) for room in rooms) + 2, 10)
        if left_width_min > left_width_max:
            continue
        left_width = random.randint(left_width_min, left_width_max)

        height_min = max(
            max(int(room["height"]) for room in rooms) + 2,
            2 * room_count + 1,
            10,
        )
        if height_min > 30:
            continue
        height = random.randint(height_min, 30)

        placed_rooms = _place_rooms(height, left_width, rooms)
        if placed_rooms is None:
            continue

        grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
        grid[:, left_width] = GREY

        for room in placed_rooms:
            _draw_room(
                grid,
                int(room["top"]),
                int(room["left"]),
                int(room["height"]),
                int(room["width"]),
                room["key"],
            )

        palette_rows = list(range(1, height - 1, 2))
        if len(palette_rows) < room_count:
            continue
        palette_rows = sorted(random.sample(palette_rows, room_count))
        palette_entries = placed_rooms[:]
        random.shuffle(palette_entries)
        for row, room in zip(palette_rows, palette_entries, strict=False):
            _draw_palette_row(
                grid,
                row,
                left_width + 1,
                right_width,
                room["key"],
                int(room["fill"]),
            )

        expected = _expected_output(grid, placed_rooms)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert not np.array_equal(grid, expected)
        assert max(grid.shape) <= 30
        return grid

    raise RuntimeError("failed to generate a valid sample")
