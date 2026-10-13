from __future__ import annotations

import numpy as np


PATTERNS = ("cross", "x", "fill", "border")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    candidates: set[int] = set()
    for r in range(grid.shape[0]):
        if np.unique(grid[r, :]).size == 1:
            candidates.add(int(grid[r, 0]))
    for c in range(grid.shape[1]):
        if np.unique(grid[:, c]).size == 1:
            candidates.add(int(grid[0, c]))
    candidates.discard(bg)
    assert len(candidates) == 1, "expected one separator color"
    return next(iter(candidates))


def _interior_spans(lines: list[int], length: int) -> list[tuple[int, int]]:
    lines = sorted(lines)
    assert lines and lines[0] == 0 and lines[-1] == length - 1, "expected an outer wall border"
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        assert b - a >= 2, "adjacent wall lines must leave room interior"
        spans.append((a + 1, b - 1))
    return spans


def _pattern_mask(shape: tuple[int, int], seed: tuple[int, int], pattern: str) -> np.ndarray:
    h, w = shape
    rs, cs = seed
    rr, cc = np.indices((h, w))
    if pattern == "cross":
        return (rr == rs) | (cc == cs)
    if pattern == "x":
        return np.abs(rr - rs) == np.abs(cc - cs)
    if pattern == "fill":
        return np.ones((h, w), dtype=bool)
    if pattern == "border":
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        mask[rs, cs] = True
        return mask
    raise AssertionError(f"unknown pattern: {pattern}")


def _room_infos(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == wall)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == wall)]
    row_spans = _interior_spans(row_lines, grid.shape[0])
    col_spans = _interior_spans(col_lines, grid.shape[1])

    rooms: list[dict] = []
    for room_r, (r0, r1) in enumerate(row_spans):
        for room_c, (c0, c1) in enumerate(col_spans):
            room = grid[r0 : r1 + 1, c0 : c1 + 1]
            fg_mask = (room != bg) & (room != wall)
            fg_coords = np.argwhere(fg_mask)
            rooms.append(
                {
                    "room_row": room_r,
                    "room_col": room_c,
                    "slice": (r0, r1, c0, c1),
                    "room": room,
                    "fg_mask": fg_mask,
                    "fg_coords": fg_coords,
                }
            )
    return rooms


def _palette_pattern(room: np.ndarray, fg_mask: np.ndarray) -> tuple[int, str]:
    colors = np.unique(room[fg_mask])
    assert colors.size == 1, "palette room must use one key color"
    color = int(colors[0])
    h, w = room.shape
    assert h % 2 == 1 and w % 2 == 1, "palette examples need a unique center"
    center = (h // 2, w // 2)

    matches = []
    color_mask = room == color
    for pattern in PATTERNS:
        if np.array_equal(color_mask, _pattern_mask(room.shape, center, pattern)):
            matches.append(pattern)
    assert len(matches) == 1, "palette room must match exactly one pattern"
    return color, matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom palette examples as color-keyed growth rules, then grow each seed within its room.

    Input structure
    - The dominant background is split by one-pixel separator walls into rectangular rooms.
    - Some bottom rooms are palette examples: each uses one color to display a growth pattern.
    - Every non-palette room contains exactly one non-background, non-wall seed pixel.

    Rule
    - Infer a color -> pattern mapping from the palette rooms.
    - For each seed room, apply that pattern using the seed color and room boundaries.
    - Keep the palette rooms unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    wall = _wall_color(g, bg)
    rooms = _room_infos(g, bg, wall)

    color_to_pattern: dict[int, str] = {}
    seed_jobs: list[tuple[tuple[int, int, int, int], tuple[int, int], int]] = []

    for info in rooms:
        room = info["room"]
        fg_coords = info["fg_coords"]
        assert fg_coords.size > 0, "every room must contain either a seed or a palette example"
        colors = np.unique(room[info["fg_mask"]])
        assert colors.size == 1, "room contents must be monochrome aside from background"
        color = int(colors[0])

        if len(fg_coords) > 1:
            key_color, pattern = _palette_pattern(room, info["fg_mask"])
            if key_color in color_to_pattern:
                assert color_to_pattern[key_color] == pattern, "repeated palette examples must agree"
            color_to_pattern[key_color] = pattern
        else:
            seed_jobs.append((info["slice"], tuple(map(int, fg_coords[0])), color))

    assert color_to_pattern, "expected at least one palette example"

    out = g.copy()
    for (r0, r1, c0, c1), seed, color in seed_jobs:
        assert color in color_to_pattern, "every seed color must appear in the palette"
        room = out[r0 : r1 + 1, c0 : c1 + 1]
        room[_pattern_mask(room.shape, seed, color_to_pattern[color])] = color
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 9
GEN_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_SIZE_CHOICES = (5, 7)
GEN_MAIN_ROWS_RANGE = (2, 4)
GEN_MAX_ATTEMPTS = 300


def _room_spans_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for size in sizes:
        spans.append((start, start + size - 1))
        start += size + 1
    return spans


def _build_empty_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), GEN_BG, dtype=int)

    row_lines = [0]
    cursor = 0
    for size in row_sizes:
        cursor += size + 1
        row_lines.append(cursor)
    col_lines = [0]
    cursor = 0
    for size in col_sizes:
        cursor += size + 1
        col_lines.append(cursor)

    grid[row_lines, :] = GEN_WALL
    grid[:, col_lines] = GEN_WALL
    return grid, _room_spans_from_sizes(row_sizes), _room_spans_from_sizes(col_sizes)


def _paint_room(room: np.ndarray, color: int, pattern: str, seed: tuple[int, int]) -> None:
    room[_pattern_mask(room.shape, seed, pattern)] = color


def _palette_positions(palette_rows: int, col_count: int, row_count: int) -> list[tuple[int, int]]:
    if palette_rows == 1:
        return [(row_count - 1, c) for c in range(col_count)]
    return [(row_count - 2 + dr, c) for dr in range(2) for c in range(col_count)]


def _seed_position(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))


def _expected_output(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    ordinary_jobs: list[tuple[int, int, int, tuple[int, int]]],
    color_to_pattern: dict[int, str],
) -> np.ndarray:
    out = inp.copy()
    for room_r, room_c, color, seed in ordinary_jobs:
        r0, r1 = row_spans[room_r]
        c0, c1 = col_spans[room_c]
        room = out[r0 : r1 + 1, c0 : c1 + 1]
        _paint_room(room, color, color_to_pattern[color], seed)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette_rows = int(rng.choice((1, 2)))
    col_count = 4 if palette_rows == 1 else 2
    main_rows = int(rng.integers(*GEN_MAIN_ROWS_RANGE))
    row_count = main_rows + palette_rows

    row_sizes = [int(rng.choice(GEN_SIZE_CHOICES)) for _ in range(row_count)]
    col_sizes = [int(rng.choice(GEN_SIZE_CHOICES)) for _ in range(col_count)]

    inp, row_spans, col_spans = _build_empty_grid(row_sizes, col_sizes)
    palette_rooms = set(_palette_positions(palette_rows, col_count, row_count))
    ordinary_rooms = [
        (r, c)
        for r in range(row_count)
        for c in range(col_count)
        if (r, c) not in palette_rooms
    ]
    if len(ordinary_rooms) < len(PATTERNS):
        return None

    colors = [int(x) for x in rng.choice(GEN_COLOR_POOL, size=len(PATTERNS), replace=False)]
    shuffled_patterns = list(PATTERNS)
    rng.shuffle(shuffled_patterns)
    color_to_pattern = dict(zip(colors, shuffled_patterns, strict=False))

    palette_room_list = list(palette_rooms)
    rng.shuffle(palette_room_list)
    for (room_r, room_c), color in zip(palette_room_list, colors, strict=False):
        r0, r1 = row_spans[room_r]
        c0, c1 = col_spans[room_c]
        room = inp[r0 : r1 + 1, c0 : c1 + 1]
        seed = (room.shape[0] // 2, room.shape[1] // 2)
        _paint_room(room, color, color_to_pattern[color], seed)

    seed_colors = colors.copy()
    extra = len(ordinary_rooms) - len(seed_colors)
    if extra > 0:
        seed_colors.extend(int(x) for x in rng.choice(colors, size=extra, replace=True))
    rng.shuffle(seed_colors)

    ordinary_jobs: list[tuple[int, int, int, tuple[int, int]]] = []
    for (room_r, room_c), color in zip(ordinary_rooms, seed_colors, strict=False):
        r0, r1 = row_spans[room_r]
        c0, c1 = col_spans[room_c]
        room = inp[r0 : r1 + 1, c0 : c1 + 1]
        seed = _seed_position(rng, room.shape)
        room[seed] = color
        ordinary_jobs.append((room_r, room_c, color, seed))

    expected = _expected_output(inp, row_spans, col_spans, ordinary_jobs, color_to_pattern)

    if _most_common_color(inp) != GEN_BG:
        return None
    if _wall_color(inp, GEN_BG) != GEN_WALL:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
