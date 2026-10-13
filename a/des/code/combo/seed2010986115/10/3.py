from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
MAGENTA = 6
WALL_COLORS = (1, 2, 3, 4, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _floor_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == BLACK) | (grid == MAGENTA)


def _room_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    room_ids = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or room_ids[r, c] != -1:
                continue
            rid = len(rooms)
            queue = deque([(int(r), int(c))])
            room_ids[r, c] = rid
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and room_ids[nr, nc] == -1:
                        room_ids[nr, nc] = rid
                        queue.append((nr, nc))

            rooms.append(np.array(coords, dtype=int))

    return room_ids, rooms


def _add_wall_evidence(
    evidence: dict[tuple[int, int], Counter[int]],
    a: int,
    b: int,
    color: int,
) -> None:
    if a < 0 or b < 0 or a == b:
        return
    pair = (a, b) if a < b else (b, a)
    evidence.setdefault(pair, Counter())[int(color)] += 1


def _room_graph(grid: np.ndarray, room_ids: np.ndarray, room_count: int) -> list[dict[int, int]]:
    h, w = grid.shape
    evidence: dict[tuple[int, int], Counter[int]] = {}

    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BLACK, MAGENTA):
                continue

            if 0 < c < w - 1:
                _add_wall_evidence(
                    evidence,
                    int(room_ids[r, c - 1]),
                    int(room_ids[r, c + 1]),
                    value,
                )
            if 0 < r < h - 1:
                _add_wall_evidence(
                    evidence,
                    int(room_ids[r - 1, c]),
                    int(room_ids[r + 1, c]),
                    value,
                )

    graph = [dict() for _ in range(room_count)]
    for (a, b), counts in evidence.items():
        wall_color = int(counts.most_common(1)[0][0])
        graph[a][b] = wall_color
        graph[b][a] = wall_color
    return graph


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify black-floor rooms separated by colored walls, start from rooms
    containing magenta seeds, and propagate floor colors inward using the
    shared wall color between already-filled and unfilled rooms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    room_ids, rooms = _room_components(_floor_mask(g))
    assert rooms, "expected at least one room"

    graph = _room_graph(g, room_ids, len(rooms))

    seed_rooms: list[int] = []
    for r, c in np.argwhere(g == MAGENTA):
        rid = int(room_ids[int(r), int(c)])
        assert rid >= 0, "seed must lie inside a room"
        if rid not in seed_rooms:
            seed_rooms.append(rid)
    assert seed_rooms, "expected at least one magenta seed"

    fill_colors: dict[int, int] = {rid: MAGENTA for rid in seed_rooms}
    queue = deque(seed_rooms)
    while queue:
        room = queue.popleft()
        for nbr, wall_color in graph[room].items():
            if nbr in fill_colors:
                continue
            fill_colors[nbr] = int(wall_color)
            queue.append(nbr)

    assert len(fill_colors) == len(rooms), "every room should be reachable from a seed room"

    out = g.copy()
    for rid, coords in enumerate(rooms):
        color = int(fill_colors[rid])
        out[coords[:, 0], coords[:, 1]] = color
    return out


# --------------- generator ---------------

ROOM_COUNT_CHOICES = (3, 4, 5)
ROOM_SIZE_MIN = 2
ROOM_SIZE_MAX = 5
GRID_DIM_MIN = 8
GRID_DIM_MAX = 26
GEN_ATTEMPTS = 300
LAYOUT_ATTEMPTS = 100


def _room_spans(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + int(size) - 1
        spans.append((start, end))
        start = end + 2
    return spans


def _sample_layout() -> tuple[list[int], list[int]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        room_rows = random.choice(ROOM_COUNT_CHOICES)
        room_cols = random.choice(ROOM_COUNT_CHOICES)
        heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_rows)]
        widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_cols)]
        grid_h = sum(heights) + room_rows - 1
        grid_w = sum(widths) + room_cols - 1
        if GRID_DIM_MIN <= grid_h <= GRID_DIM_MAX and GRID_DIM_MIN <= grid_w <= GRID_DIM_MAX:
            return heights, widths
    return None


def _depths(rows: int, cols: int) -> np.ndarray:
    arr = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = min(i, j, rows - 1 - i, cols - 1 - j)
    return arr


def _is_boundary_room(i: int, j: int, rows: int, cols: int) -> bool:
    return i == 0 or j == 0 or i == rows - 1 or j == cols - 1


def _choose_room_colors(depths: np.ndarray) -> np.ndarray:
    rows, cols = depths.shape
    colors = np.full((rows, cols), MAGENTA, dtype=int)
    inner_rooms: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if depths[i, j] > 0:
                colors[i, j] = random.choice(WALL_COLORS)
                inner_rooms.append((i, j))

    if len(inner_rooms) >= 2:
        used = {int(colors[i, j]) for i, j in inner_rooms}
        if len(used) == 1:
            i, j = random.choice(inner_rooms)
            choices = [c for c in WALL_COLORS if c != int(colors[i, j])]
            colors[i, j] = random.choice(choices)

    return colors


def _wall_color_for_pair(
    depths: np.ndarray,
    room_colors: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> int:
    ai, aj = a
    bi, bj = b
    da = int(depths[ai, aj])
    db = int(depths[bi, bj])
    if da < db:
        return int(room_colors[bi, bj])
    if db < da:
        return int(room_colors[ai, aj])
    return int(random.choice(WALL_COLORS))


def _expected_output(
    shape: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    room_colors: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    rows, cols = room_colors.shape
    for i in range(rows):
        r0, r1 = row_spans[i]
        for j in range(cols):
            c0, c1 = col_spans[j]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(room_colors[i, j])
    return out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    heights, widths = layout
    rows = len(heights)
    cols = len(widths)
    depths = _depths(rows, cols)
    if int(depths.max()) < 1:
        return None

    row_spans = _room_spans(heights)
    col_spans = _room_spans(widths)
    grid_h = row_spans[-1][1] + 1
    grid_w = col_spans[-1][1] + 1
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    room_colors = _choose_room_colors(depths)
    expected = _expected_output(grid.shape, row_spans, col_spans, room_colors)

    vertical_colors = [[0 for _ in range(cols - 1)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols - 1):
            vertical_colors[i][j] = _wall_color_for_pair(depths, room_colors, (i, j), (i, j + 1))

    horizontal_colors = [[0 for _ in range(cols)] for _ in range(rows - 1)]
    for i in range(rows - 1):
        for j in range(cols):
            horizontal_colors[i][j] = _wall_color_for_pair(depths, room_colors, (i, j), (i + 1, j))

    for i in range(rows):
        r0, r1 = row_spans[i]
        for j in range(cols - 1):
            wall_c = col_spans[j][1] + 1
            color = int(vertical_colors[i][j])
            grid[r0 : r1 + 1, wall_c] = color
            expected[r0 : r1 + 1, wall_c] = color

    for i in range(rows - 1):
        wall_r = row_spans[i][1] + 1
        for j in range(cols):
            c0, c1 = col_spans[j]
            color = int(horizontal_colors[i][j])
            grid[wall_r, c0 : c1 + 1] = color
            expected[wall_r, c0 : c1 + 1] = color

    for i in range(rows - 1):
        wall_r = row_spans[i][1] + 1
        for j in range(cols - 1):
            wall_c = col_spans[j][1] + 1
            corner_choices = (
                int(vertical_colors[i][j]),
                int(vertical_colors[i + 1][j]),
                int(horizontal_colors[i][j]),
                int(horizontal_colors[i][j + 1]),
            )
            color = int(random.choice(corner_choices))
            grid[wall_r, wall_c] = color
            expected[wall_r, wall_c] = color

    for i in range(rows):
        r0, r1 = row_spans[i]
        for j in range(cols):
            if not _is_boundary_room(i, j, rows, cols):
                continue
            c0, c1 = col_spans[j]
            seed_r = random.randint(r0, r1)
            seed_c = random.randint(c0, c1)
            grid[seed_r, seed_c] = MAGENTA

    inner_count = int(np.count_nonzero(depths > 0))
    if inner_count == 0:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-propagation input")
