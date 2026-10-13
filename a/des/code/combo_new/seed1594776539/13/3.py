from __future__ import annotations

import random

import numpy as np

BLACK = 0
RED = 2
GRAY = 5
PALETTE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_SIDE = 30
GEN_RETRIES = 400

Room = tuple[int, int, int, int]


def _detect_palette_height(grid: np.ndarray) -> int:
    count = 0
    for r in range(grid.shape[0] - 1, -1, -1):
        row = grid[r]
        if np.any(np.isin(row, (BLACK, RED, GRAY))):
            break
        count += 1
        if count == 2:
            break
    assert 1 <= count <= 2, "palette area must occupy the bottom one or two rows"
    return count


def _find_rooms(upper: np.ndarray) -> list[Room]:
    h, w = upper.shape
    wall_rows = [int(r) for r in range(h) if bool(np.all(upper[r, :] == GRAY))]
    wall_cols = [int(c) for c in range(w) if bool(np.all(upper[:, c] == GRAY))]

    gray_mask = upper == GRAY
    for r, c in np.argwhere(gray_mask):
        assert int(r) in wall_rows or int(c) in wall_cols, "gray pixels must belong to full grid lines"

    row_bounds = [-1] + wall_rows + [h]
    col_bounds = [-1] + wall_cols + [w]

    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []
    for a, b in zip(row_bounds, row_bounds[1:]):
        if b - a > 1:
            row_spans.append((int(a + 1), int(b - 1)))
    for a, b in zip(col_bounds, col_bounds[1:]):
        if b - a > 1:
            col_spans.append((int(a + 1), int(b - 1)))

    rooms: list[Room] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = upper[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch != GRAY), "room interior must not contain walls"
            rooms.append((r0, c0, r1, c1))

    assert rooms, "expected at least one room"
    return rooms


def _paint_room(out: np.ndarray, room: Room, palette_row: np.ndarray) -> None:
    r0, c0, r1, c1 = room
    palette = np.asarray(palette_row, dtype=int)
    plen = int(palette.size)
    assert plen > 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = int(palette[depth % plen])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a palette row by seed count, then recolor each partitioned room in concentric wall-distance layers.

    Input structure:
    - The background is black.
    - Gray one-cell-thick full rows and columns partition the upper part of the
      grid into rectangular rooms.
    - The bottom one or two rows form an unsplit palette area whose colors are
      distinct from the structural black/gray/red colors.
    - Each room interior contains one or more red seed pixels on black.

    Rule:
    - Count the red seeds inside each room; if a room has N seeds, choose the
      Nth palette row counted from the bottom.
    - Recolor the room interior by wall-distance layers: the outer ring gets the
      first color of that palette row, the next ring gets the second color, and
      so on, cycling if needed.
    - Seeds are overwritten by the layer colors, while gray walls and the palette
      area stay unchanged.

    Key insight:
    - The red pixels are not colors to preserve; they are lookup counters that
      select which bottom palette row should drive the concentric room fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert max(g.shape) <= MAX_SIDE

    palette_h = _detect_palette_height(g)
    upper_h = int(h - palette_h)
    assert upper_h >= 1

    upper = g[:upper_h, :]
    assert np.all(np.isin(upper, (BLACK, RED, GRAY))), "upper area must contain only structure colors"

    palettes = [g[r, :].copy() for r in range(upper_h, h)]
    out = g.copy()

    for room in _find_rooms(upper):
        r0, c0, r1, c1 = room
        seed_count = int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == RED))
        assert 1 <= seed_count <= palette_h, "seed count must index an existing palette row"
        _paint_room(out, room, palettes[-seed_count])

    return out


def _sample_palette_row(width: int, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(100):
        motif_len = random.randint(2, min(6, width))
        motif_colors = random.sample(PALETTE_COLORS, random.randint(2, min(4, len(PALETTE_COLORS), motif_len)))
        motif = [random.choice(motif_colors) for _ in range(motif_len)]
        if len(set(motif)) < 2:
            continue

        row = [motif[i % motif_len] for i in range(width)]
        for _ in range(random.randint(0, max(1, width // 8))):
            idx = random.randrange(width)
            row[idx] = random.choice(motif_colors)

        key = tuple(int(v) for v in row)
        if key in used:
            continue
        used.add(key)
        return np.array(row, dtype=int)
    raise RuntimeError("failed to sample a distinct palette row")


def _layout_axis(sizes: list[int], border_before: bool, border_after: bool) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    walls: list[int] = []
    cursor = 0
    if border_before:
        walls.append(0)
        cursor = 1

    for i, size in enumerate(sizes):
        start = int(cursor)
        end = int(start + size - 1)
        spans.append((start, end))
        cursor = end + 1
        if i < len(sizes) - 1:
            walls.append(cursor)
            cursor += 1

    if border_after:
        walls.append(cursor)
        cursor += 1

    return spans, walls, int(cursor)


def _choose_layout(palette_h: int) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], list[int], int, int]:
    room_shape_choices = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(200):
        n_rows, n_cols = random.choice(room_shape_choices)
        row_sizes = [random.randint(2, 7) for _ in range(n_rows)]
        col_sizes = [random.randint(2, 7) for _ in range(n_cols)]

        top_border = bool(random.getrandbits(1))
        bottom_border = bool(random.getrandbits(1))
        left_border = bool(random.getrandbits(1))
        right_border = bool(random.getrandbits(1))

        row_spans, wall_rows, upper_h = _layout_axis(row_sizes, top_border, bottom_border)
        col_spans, wall_cols, width = _layout_axis(col_sizes, left_border, right_border)
        height = upper_h + palette_h

        if height > MAX_SIDE or width > MAX_SIDE:
            continue
        if n_rows * n_cols < 2:
            continue
        return row_spans, wall_rows, col_spans, wall_cols, upper_h, width
    raise RuntimeError("failed to choose a valid room layout")


def _seed_counts(room_count: int, palette_h: int) -> list[int]:
    if palette_h == 1:
        return [1] * room_count

    counts = [1, 2]
    counts.extend(random.randint(1, 2) for _ in range(room_count - 2))
    random.shuffle(counts)
    return counts


def _build_expected(inp: np.ndarray, palette_h: int, rooms: list[Room], counts: list[int]) -> np.ndarray:
    out = inp.copy()
    palettes = [inp[r, :].copy() for r in range(inp.shape[0] - palette_h, inp.shape[0])]
    for room, seed_count in zip(rooms, counts):
        _paint_room(out, room, palettes[-seed_count])
    return out


def _gen_attempt() -> np.ndarray | None:
    palette_h = random.choice([1, 2])
    row_spans, wall_rows, col_spans, wall_cols, upper_h, width = _choose_layout(palette_h)
    height = upper_h + palette_h
    grid = np.full((height, width), BLACK, dtype=int)

    for r in wall_rows:
        grid[r, :] = GRAY
    for c in wall_cols:
        grid[:upper_h, c] = GRAY

    used_rows: set[tuple[int, ...]] = set()
    palette_rows = [_sample_palette_row(width, used_rows) for _ in range(palette_h)]
    for offset, row in enumerate(reversed(palette_rows), start=1):
        grid[height - offset, :] = row

    rooms = [(r0, c0, r1, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    counts = _seed_counts(len(rooms), palette_h)

    for room, seed_count in zip(rooms, counts):
        r0, c0, r1, c1 = room
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if len(cells) < seed_count:
            return None
        for r, c in random.sample(cells, seed_count):
            grid[r, c] = RED

    expected = _build_expected(grid, palette_h, rooms, counts)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-palette puzzle")
