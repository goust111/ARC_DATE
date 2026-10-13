from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DOOR_TO_COLOR = {
    0: RED,
    1: BLUE,
    2: GREEN,
    3: YELLOW,
}

# --------------- solver ---------------


def _color_for_doors(count: int) -> int:
    return int(DOOR_TO_COLOR.get(int(count), ORANGE))


def _wall_positions(room_sizes: list[int]) -> list[int]:
    pos = 0
    walls = [0]
    for size in room_sizes:
        pos += int(size) + 1
        walls.append(pos)
    return walls


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    grey = grid == GREY
    row_counts = grey.sum(axis=1)
    col_counts = grey.sum(axis=0)

    wall_rows = [int(r) for r, count in enumerate(row_counts.tolist()) if int(count) > grid.shape[1] // 2]
    wall_cols = [int(c) for c, count in enumerate(col_counts.tolist()) if int(count) > grid.shape[0] // 2]

    assert len(wall_rows) >= 2 and wall_rows[0] == 0 and wall_rows[-1] == grid.shape[0] - 1
    assert len(wall_cols) >= 2 and wall_cols[0] == 0 and wall_cols[-1] == grid.shape[1] - 1
    assert all(b > a for a, b in zip(wall_rows, wall_rows[1:], strict=False))
    assert all(b > a for a, b in zip(wall_cols, wall_cols[1:], strict=False))
    return wall_rows, wall_cols


def _room_door_totals(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> np.ndarray:
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    totals = np.zeros((room_rows, room_cols), dtype=int)

    for i in range(room_rows):
        r0, r1 = int(wall_rows[i]), int(wall_rows[i + 1])
        assert r1 - r0 >= 2
        for j in range(room_cols):
            c0, c1 = int(wall_cols[j]), int(wall_cols[j + 1])
            assert c1 - c0 >= 2

            doors = 0
            if i > 0:
                doors += int(np.count_nonzero(grid[r0, c0 + 1 : c1] == BLACK))
            if i + 1 < room_rows:
                doors += int(np.count_nonzero(grid[r1, c0 + 1 : c1] == BLACK))
            if j > 0:
                doors += int(np.count_nonzero(grid[r0 + 1 : r1, c0] == BLACK))
            if j + 1 < room_cols:
                doors += int(np.count_nonzero(grid[r0 + 1 : r1, c1] == BLACK))

            totals[i, j] = doors

    return totals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room according to how many door gaps appear on its walls.

    The input is a black canvas split by a one-pixel grey lattice into rectangular
    rooms. Some internal wall pixels are missing, creating one-pixel black door
    gaps between neighboring rooms. Every room interior is black.

    The output keeps the walls and door gaps unchanged, and recolors each room
    interior using:
      0 -> red, 1 -> blue, 2 -> green, 3 -> yellow, 4+ -> orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_rows, wall_cols = _find_wall_lines(g)
    totals = _room_door_totals(g, wall_rows, wall_cols)

    out = g.copy()
    for i in range(totals.shape[0]):
        r0, r1 = int(wall_rows[i]), int(wall_rows[i + 1])
        for j in range(totals.shape[1]):
            c0, c1 = int(wall_cols[j]), int(wall_cols[j + 1])
            out[r0 + 1 : r1, c0 + 1 : c1] = _color_for_doors(int(totals[i, j]))
    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 30
ROOM_ROWS_RANGE = (2, 5)
ROOM_COLS_RANGE = (2, 5)
ROOM_SIZE_RANGE = (2, 7)
HUB_ROOM_SIZE_RANGE = (3, 7)
GENERATION_ATTEMPTS = 400

PROFILE_WEIGHTS = {
    "sparse": (8.0, 3.5, 1.0, 0.2),
    "mixed": (4.0, 4.0, 2.0, 0.8),
    "busy": (2.0, 3.0, 3.0, 1.5),
}


def _sample_room_sizes(count: int, *, hub_mode: bool) -> list[int] | None:
    size_range = HUB_ROOM_SIZE_RANGE if hub_mode else ROOM_SIZE_RANGE
    sizes = [random.randint(*size_range) for _ in range(int(count))]
    return sizes


def _sample_layout(mode: str) -> tuple[list[int], list[int]] | None:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)
    hub_mode = mode == "hub"

    heights = _sample_room_sizes(room_rows, hub_mode=hub_mode)
    widths = _sample_room_sizes(room_cols, hub_mode=hub_mode)
    if heights is None or widths is None:
        return None

    grid_h = sum(heights) + room_rows + 1
    grid_w = sum(widths) + room_cols + 1
    if not (GRID_MIN <= grid_h <= GRID_MAX and GRID_MIN <= grid_w <= GRID_MAX):
        return None
    return heights, widths


def _pick_door_count(length: int, mode: str) -> int:
    max_doors = min(3, (int(length) + 1) // 2)
    counts = list(range(max_doors + 1))
    weights = PROFILE_WEIGHTS["busy" if mode == "hub" else mode]
    return int(random.choices(counts, weights=weights[: max_doors + 1], k=1)[0])


def _available_positions(existing: set[int], length: int) -> list[int]:
    return [
        pos
        for pos in range(int(length))
        if pos not in existing and all(abs(pos - other) > 1 for other in existing)
    ]


def _grow_doors(existing: set[int], length: int, target_count: int) -> None:
    while len(existing) < int(target_count):
        choices = _available_positions(existing, length)
        if not choices:
            return
        existing.add(int(random.choice(choices)))


def _sample_segment_doors(length: int, mode: str) -> set[int]:
    doors: set[int] = set()
    _grow_doors(doors, length, _pick_door_count(length, mode))
    return doors


def _choose_hub_room(room_rows: int, room_cols: int) -> tuple[int, int]:
    interior = [
        (i, j)
        for i in range(room_rows)
        for j in range(room_cols)
        if 0 < i < room_rows - 1 and 0 < j < room_cols - 1
    ]
    if interior:
        return random.choice(interior)

    non_corners = [
        (i, j)
        for i in range(room_rows)
        for j in range(room_cols)
        if (i, j) not in {(0, 0), (0, room_cols - 1), (room_rows - 1, 0), (room_rows - 1, room_cols - 1)}
    ]
    if non_corners:
        return random.choice(non_corners)
    return random.randrange(room_rows), random.randrange(room_cols)


def _totals_from_doors(
    h_doors: list[list[set[int]]],
    v_doors: list[list[set[int]]],
    room_rows: int,
    room_cols: int,
) -> np.ndarray:
    totals = np.zeros((room_rows, room_cols), dtype=int)

    for i in range(room_rows - 1):
        for j in range(room_cols):
            count = len(h_doors[i][j])
            totals[i, j] += count
            totals[i + 1, j] += count

    for i in range(room_rows):
        for j in range(room_cols - 1):
            count = len(v_doors[i][j])
            totals[i, j] += count
            totals[i, j + 1] += count

    return totals


def _boost_hub_room(
    h_doors: list[list[set[int]]],
    v_doors: list[list[set[int]]],
    heights: list[int],
    widths: list[int],
) -> None:
    room_rows = len(heights)
    room_cols = len(widths)
    hub_i, hub_j = _choose_hub_room(room_rows, room_cols)

    segments: list[tuple[str, int, int, int]] = []
    if hub_i > 0:
        segments.append(("h", hub_i - 1, hub_j, widths[hub_j]))
    if hub_i + 1 < room_rows:
        segments.append(("h", hub_i, hub_j, widths[hub_j]))
    if hub_j > 0:
        segments.append(("v", hub_i, hub_j - 1, heights[hub_i]))
    if hub_j + 1 < room_cols:
        segments.append(("v", hub_i, hub_j, heights[hub_i]))

    random.shuffle(segments)
    for kind, i, j, length in segments:
        if kind == "h":
            _grow_doors(h_doors[i][j], length, max(1, len(h_doors[i][j])))
        else:
            _grow_doors(v_doors[i][j], length, max(1, len(v_doors[i][j])))

    while True:
        totals = _totals_from_doors(h_doors, v_doors, room_rows, room_cols)
        if int(totals[hub_i, hub_j]) >= 4:
            return

        candidates: list[tuple[int, str, int, int, int]] = []
        for idx, (kind, i, j, length) in enumerate(segments):
            existing = h_doors[i][j] if kind == "h" else v_doors[i][j]
            if _available_positions(existing, length):
                candidates.append((length, kind, i, j, idx))
        if not candidates:
            return

        longest = max(candidates)[0]
        chosen = random.choice([cand for cand in candidates if cand[0] == longest])
        _length, kind, i, j, _idx = chosen
        if kind == "h":
            _grow_doors(h_doors[i][j], lengths := widths[j], len(h_doors[i][j]) + 1)
        else:
            _grow_doors(v_doors[i][j], lengths := heights[i], len(v_doors[i][j]) + 1)


def _sample_doors(heights: list[int], widths: list[int], mode: str) -> tuple[list[list[set[int]]], list[list[set[int]]], np.ndarray]:
    room_rows = len(heights)
    room_cols = len(widths)

    h_doors = [
        [_sample_segment_doors(widths[j], mode) for j in range(room_cols)]
        for _ in range(room_rows - 1)
    ]
    v_doors = [
        [_sample_segment_doors(heights[i], mode) for _ in range(room_cols - 1)]
        for i in range(room_rows)
    ]

    if mode == "hub":
        _boost_hub_room(h_doors, v_doors, heights, widths)

    totals = _totals_from_doors(h_doors, v_doors, room_rows, room_cols)
    return h_doors, v_doors, totals


def _build_input(
    heights: list[int],
    widths: list[int],
    h_doors: list[list[set[int]]],
    v_doors: list[list[set[int]]],
) -> tuple[np.ndarray, list[int], list[int]]:
    wall_rows = _wall_positions(heights)
    wall_cols = _wall_positions(widths)
    grid = np.zeros((wall_rows[-1] + 1, wall_cols[-1] + 1), dtype=int)

    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY

    for i in range(len(heights) - 1):
        wall_r = wall_rows[i + 1]
        for j in range(len(widths)):
            start_c = wall_cols[j] + 1
            for pos in h_doors[i][j]:
                grid[wall_r, start_c + int(pos)] = BLACK

    for i in range(len(heights)):
        start_r = wall_rows[i] + 1
        for j in range(len(widths) - 1):
            wall_c = wall_cols[j + 1]
            for pos in v_doors[i][j]:
                grid[start_r + int(pos), wall_c] = BLACK

    return grid, wall_rows, wall_cols


def _expected_output(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int], totals: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for i in range(totals.shape[0]):
        r0, r1 = wall_rows[i], wall_rows[i + 1]
        for j in range(totals.shape[1]):
            c0, c1 = wall_cols[j], wall_cols[j + 1]
            out[r0 + 1 : r1, c0 + 1 : c1] = _color_for_doors(int(totals[i, j]))
    return out


def _good_variation(totals: np.ndarray, mode: str) -> bool:
    colors = {_color_for_doors(int(v)) for v in totals.ravel().tolist()}
    if int(totals.max()) == 0:
        return False
    if len(colors) < 2:
        return False
    if mode == "hub" and int(totals.max()) < 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(["sparse", "mixed", "hub"], weights=[3, 4, 2], k=1)[0]
    layout = _sample_layout(mode)
    if layout is None:
        return None
    heights, widths = layout

    h_doors, v_doors, totals = _sample_doors(heights, widths, mode)
    if not _good_variation(totals, mode):
        return None

    grid, wall_rows, wall_cols = _build_input(heights, widths, h_doors, v_doors)
    if _find_wall_lines(grid) != (wall_rows, wall_cols):
        return None

    expected = _expected_output(grid, wall_rows, wall_cols, totals)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-door grid")
