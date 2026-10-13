from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4)
DIRECTIONS = ("up", "down", "left", "right")

MIN_ROOM_BANDS = 2
MAX_ROOM_BANDS = 3
MIN_ROOM_SIZE = 2
MAX_ROOM_SIZE = 7
MAX_MARGIN = 2
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(coords)
    return components


def _enclosed_room_bbox(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = grid.shape

    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None

    room = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(room == WALL):
        return None

    ring = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
    if not (
        np.all(ring[0, :] == WALL)
        and np.all(ring[-1, :] == WALL)
        and np.all(ring[:, 0] == WALL)
        and np.all(ring[:, -1] == WALL)
    ):
        return None

    return int(r0), int(c0), int(r1), int(c1)


def _nearest_side_or_none(local_r: int, local_c: int, height: int, width: int) -> str | None:
    distances = {
        "up": int(local_r),
        "down": int(height - 1 - local_r),
        "left": int(local_c),
        "right": int(width - 1 - local_c),
    }
    best = min(distances.values())
    winners = [direction for direction, dist in distances.items() if dist == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _unique_nearest_side(local_r: int, local_c: int, height: int, width: int) -> str:
    direction = _nearest_side_or_none(local_r, local_c, height, width)
    assert direction is not None, "marker must have a unique nearest wall"
    return direction


def _erase_centered_door(out: np.ndarray, room_bbox: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = room_bbox
    if direction in ("up", "down"):
        wall_row = r0 - 1 if direction == "up" else r1 + 1
        span = c1 - c0 + 1
        door = 1 if span < 5 else 3
        start = c0 + (span - door) // 2
        out[wall_row, start : start + door] = BACKGROUND
    else:
        wall_col = c0 - 1 if direction == "left" else c1 + 1
        span = r1 - r0 + 1
        door = 1 if span < 5 else 3
        start = r0 + (span - door) // 2
        out[start : start + door, wall_col] = BACKGROUND


def _direction_candidates(height: int, width: int) -> dict[str, list[tuple[int, int]]]:
    candidates = {direction: [] for direction in DIRECTIONS}
    for r in range(height):
        for c in range(width):
            direction = _nearest_side_or_none(r, c, height, width)
            if direction is not None:
                candidates[direction].append((int(r), int(c)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open a centered door on the wall nearest to each room's off-center marker, then erase the marker.

    Input structure:
    - The background is black.
    - Gray pixels form a one-cell-thick wall network that partitions the grid into
      closed rectangular rooms.
    - Some rooms are empty; some rooms contain exactly one non-black, non-gray
      marker pixel somewhere inside the room.

    Rule:
    - Detect each enclosed rectangular room bounded by gray walls.
    - For a room with one marker, find which wall is uniquely nearest to that
      marker (up, down, left, or right).
    - Erase a centered door on that wall: width 3 when the wall span is at least
      5 cells, otherwise width 1. Erase the marker itself to black.
    - Empty rooms and all other wall pixels remain unchanged.

    Key insight:
    - The marker's color is irrelevant; only its position inside the room decides
      which wall gets the doorway.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    room_mask = g != WALL
    for coords in _components(room_mask):
        room_bbox = _enclosed_room_bbox(g, coords)
        if room_bbox is None:
            continue

        r0, c0, r1, c1 = room_bbox
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        markers = np.argwhere((room != BACKGROUND) & (room != WALL))
        assert len(markers) <= 1, "each room may contain at most one marker"
        if len(markers) == 0:
            continue

        local_r, local_c = map(int, markers[0])
        direction = _unique_nearest_side(local_r, local_c, room.shape[0], room.shape[1])
        out[r0 + local_r, c0 + local_c] = BACKGROUND
        _erase_centered_door(out, room_bbox, direction)

    return out


def _sample_room_sizes(count: int) -> list[int]:
    return [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(count)]


def _room_wall_positions(offset: int, sizes: list[int]) -> list[int]:
    positions = [int(offset)]
    cursor = int(offset)
    for size in sizes:
        cursor += int(size) + 1
        positions.append(int(cursor))
    return positions


def _build_empty_layout(
    row_sizes: list[int],
    col_sizes: list[int],
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[dict]]:
    top, bottom, left, right = margins
    patch_h = sum(row_sizes) + len(row_sizes) + 1
    patch_w = sum(col_sizes) + len(col_sizes) + 1
    grid_h = patch_h + top + bottom
    grid_w = patch_w + left + right
    assert grid_h <= MAX_GRID_SIDE and grid_w <= MAX_GRID_SIDE

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    wall_rows = _room_wall_positions(top, row_sizes)
    wall_cols = _room_wall_positions(left, col_sizes)
    patch_r1 = top + patch_h - 1
    patch_c1 = left + patch_w - 1

    for row in wall_rows:
        grid[row, left : patch_c1 + 1] = WALL
    for col in wall_cols:
        grid[top : patch_r1 + 1, col] = WALL

    rooms: list[dict] = []
    for i in range(len(row_sizes)):
        for j in range(len(col_sizes)):
            r0 = wall_rows[i] + 1
            r1 = wall_rows[i + 1] - 1
            c0 = wall_cols[j] + 1
            c1 = wall_cols[j + 1] - 1
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            rooms.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(height),
                    "width": int(width),
                    "candidates": _direction_candidates(height, width),
                }
            )
    return grid, rooms


def _choose_marked_rooms(rooms: list[dict]) -> list[dict]:
    feasible = [room for room in rooms if any(room["candidates"][direction] for direction in DIRECTIONS)]
    if not feasible:
        return []

    max_marks = min(len(feasible), max(1, len(rooms) - 1))
    mark_count = random.randint(1, max_marks)
    return random.sample(feasible, mark_count)


def _place_markers(grid: np.ndarray, rooms: list[dict]) -> None:
    usage = {direction: 0 for direction in DIRECTIONS}
    shuffled_rooms = random.sample(rooms, len(rooms))
    for room in shuffled_rooms:
        feasible_dirs = [direction for direction in DIRECTIONS if room["candidates"][direction]]
        min_used = min(usage[direction] for direction in feasible_dirs)
        candidate_dirs = [direction for direction in feasible_dirs if usage[direction] == min_used]
        direction = random.choice(candidate_dirs)
        usage[direction] += 1

        local_r, local_c = random.choice(room["candidates"][direction])
        r0, c0, _r1, _c1 = room["bbox"]
        grid[r0 + local_r, c0 + local_c] = random.choice(MARKER_COLORS)


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(MIN_ROOM_BANDS, MAX_ROOM_BANDS)
    room_cols = random.randint(MIN_ROOM_BANDS, MAX_ROOM_BANDS)
    row_sizes = _sample_room_sizes(room_rows)
    col_sizes = _sample_room_sizes(room_cols)

    patch_h = sum(row_sizes) + room_rows + 1
    patch_w = sum(col_sizes) + room_cols + 1
    if patch_h + 2 * MAX_MARGIN > MAX_GRID_SIDE or patch_w + 2 * MAX_MARGIN > MAX_GRID_SIDE:
        return None

    margins = tuple(random.randint(0, MAX_MARGIN) for _ in range(4))
    grid, rooms = _build_empty_layout(row_sizes, col_sizes, margins)

    marked_rooms = _choose_marked_rooms(rooms)
    if not marked_rooms:
        return None
    _place_markers(grid, marked_rooms)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle input")
