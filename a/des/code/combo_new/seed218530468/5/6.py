from __future__ import annotations

import random
from collections import deque

import numpy as np


WALL_COLOR = 5
MARKER_COLOR = 9
BACKGROUND_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8]

ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (2, 4)
ROOM_SIZE_RANGE = (3, 6)
MARGIN_RANGE = (0, 2)
MAX_MARKERS_PER_ROOM = 3
MAX_DOORS = 6

LAYOUT_RETRIES = 200
COUNT_RETRIES = 200
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    if WALL_COLOR in colors:
        return WALL_COLOR
    assert colors, "expected at least one non-background color"

    def score(color: int) -> tuple[int, int]:
        mask = grid == color
        row_hits = sum(_longest_true_run(mask[r]) >= 2 for r in range(mask.shape[0]))
        col_hits = sum(_longest_true_run(mask[:, c]) >= 2 for c in range(mask.shape[1]))
        return (int(row_hits + col_hits), int(mask.sum()))

    return max(colors, key=score)


def _detect_marker_color(grid: np.ndarray, bg: int, wall: int) -> int | None:
    colors = [int(color) for color in np.unique(grid) if int(color) not in {bg, wall}]
    if MARKER_COLOR in colors:
        return MARKER_COLOR
    if not colors:
        return None
    return min(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _detect_doors(grid: np.ndarray, bg: int, wall: int) -> list[tuple[str, int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    doors: list[tuple[str, int, int]] = []

    for row in range(1, height - 2):
        for col in range(1, width - 1):
            if (
                int(g[row, col]) == bg
                and int(g[row + 1, col]) == bg
                and int(g[row - 1, col]) == wall
                and int(g[row + 2, col]) == wall
            ):
                side_cells = (
                    (row, col - 1),
                    (row + 1, col - 1),
                    (row, col + 1),
                    (row + 1, col + 1),
                )
                if all(int(g[r, c]) != wall for r, c in side_cells):
                    doors.append(("v", int(row), int(col)))

    for row in range(1, height - 1):
        for col in range(1, width - 2):
            if (
                int(g[row, col]) == bg
                and int(g[row, col + 1]) == bg
                and int(g[row, col - 1]) == wall
                and int(g[row, col + 2]) == wall
            ):
                side_cells = (
                    (row - 1, col),
                    (row - 1, col + 1),
                    (row + 1, col),
                    (row + 1, col + 1),
                )
                if all(int(g[r, c]) != wall for r, c in side_cells):
                    doors.append(("h", int(row), int(col)))

    return doors


def _close_doors(grid: np.ndarray, doors: list[tuple[str, int, int]], wall: int) -> np.ndarray:
    closed = np.asarray(grid, dtype=int).copy()
    for orient, row, col in doors:
        if orient == "v":
            closed[row : row + 2, col] = wall
        else:
            closed[row, col : col + 2] = wall
    return closed


def _label_rooms(grid: np.ndarray, wall: int, marker: int | None) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    labels = np.full((height, width), -1, dtype=int)
    seen = np.zeros((height, width), dtype=bool)
    marker_counts: list[int] = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(g[row, col]) == wall:
                continue

            queue: deque[tuple[int, int]] = deque([(row, col)])
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            touches_boundary = False
            marker_count = 0

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                if r in (0, height - 1) or c in (0, width - 1):
                    touches_boundary = True
                if marker is not None and int(g[r, c]) == marker:
                    marker_count += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(g[nr, nc]) != wall
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_boundary:
                continue

            room_id = len(marker_counts)
            marker_counts.append(int(marker_count))
            for r, c in cells:
                labels[r, c] = room_id

    return labels, marker_counts


def _door_room_ids(door: tuple[str, int, int], labels: np.ndarray) -> tuple[int, int] | None:
    orient, row, col = door
    if orient == "v":
        left_ids = {int(labels[row, col - 1]), int(labels[row + 1, col - 1])} - {-1}
        right_ids = {int(labels[row, col + 1]), int(labels[row + 1, col + 1])} - {-1}
    else:
        left_ids = {int(labels[row - 1, col]), int(labels[row - 1, col + 1])} - {-1}
        right_ids = {int(labels[row + 1, col]), int(labels[row + 1, col + 1])} - {-1}

    if len(left_ids) != 1 or len(right_ids) != 1:
        return None

    room_a = next(iter(left_ids))
    room_b = next(iter(right_ids))
    if room_a == room_b:
        return None
    return (int(room_a), int(room_b))


def _analyze_layout(
    grid: np.ndarray,
    bg: int | None = None,
    wall: int | None = None,
    marker: int | None = None,
) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g) if bg is None else int(bg)
    wall = _detect_wall_color(g, bg) if wall is None else int(wall)
    marker = _detect_marker_color(g, bg, wall) if marker is None else int(marker)

    doors = _detect_doors(g, bg, wall)
    closed = _close_doors(g, doors, wall)
    labels, marker_counts = _label_rooms(closed, wall, marker)

    door_infos = []
    for door in doors:
        room_ids = _door_room_ids(door, labels)
        if room_ids is None:
            continue
        room_a, room_b = room_ids
        door_infos.append(
            {
                "door": door,
                "rooms": (room_a, room_b),
                "equal": bool(marker_counts[room_a] == marker_counts[room_b]),
            }
        )

    return {
        "bg": bg,
        "wall": wall,
        "marker": marker,
        "labels": labels,
        "marker_counts": marker_counts,
        "door_infos": door_infos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Close doors between equivalent rooms and erase the temporary room markers.

    Input structure:
    - The background is the dominant color.
    - Grey wall lines form closed rectangular rooms, and some shared walls contain exactly two background cells as open doorways.
    - Maroon marker pixels may appear anywhere inside the rooms.

    Rule:
    - Detect each two-cell doorway and identify the room on each side by temporarily sealing all doors.
    - Count the maroon markers inside the two adjacent rooms.
    - If the counts are equal, fill the doorway with grey wall pixels; otherwise leave it open.
    - Erase every maroon marker to the background color while leaving all other wall structure unchanged.

    Key insight:
    - The markers are only room labels: each door depends on whether its two rooms have matching marker counts, and the markers disappear after that comparison.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_layout(g)
    out = g.copy()

    for door_info in info["door_infos"]:
        if not door_info["equal"]:
            continue
        orient, row, col = door_info["door"]
        if orient == "v":
            out[row : row + 2, col] = info["wall"]
        else:
            out[row, col : col + 2] = info["wall"]

    marker = info["marker"]
    if marker is not None:
        out[out == marker] = info["bg"]

    return out


# --------------- generator ---------------


def _room_boxes(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for room_row in range(len(wall_rows) - 1):
        for room_col in range(len(wall_cols) - 1):
            rooms[(room_row, room_col)] = (
                int(wall_rows[room_row] + 1),
                int(wall_rows[room_row + 1]),
                int(wall_cols[room_col] + 1),
                int(wall_cols[room_col + 1]),
            )
    return rooms


def _adjacent_edges(room_rows: int, room_cols: int) -> list[tuple[str, int, int]]:
    edges: list[tuple[str, int, int]] = []
    for room_row in range(room_rows):
        for room_col in range(room_cols - 1):
            edges.append(("v", int(room_row), int(room_col)))
    for room_row in range(room_rows - 1):
        for room_col in range(room_cols):
            edges.append(("h", int(room_row), int(room_col)))
    return edges


def _edge_rooms(edge: tuple[str, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    orient, room_row, room_col = edge
    if orient == "v":
        return (room_row, room_col), (room_row, room_col + 1)
    return (room_row, room_col), (room_row + 1, room_col)


def _pick_counts(
    rng: random.Random,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    edges: list[tuple[str, int, int]],
) -> tuple[dict[tuple[int, int], int], list[tuple[str, int, int]], list[tuple[str, int, int]]] | None:
    for _ in range(COUNT_RETRIES):
        counts: dict[tuple[int, int], int] = {}
        has_nonzero = False

        for room_id, (r0, r1, c0, c1) in rooms.items():
            area = (r1 - r0) * (c1 - c0)
            upper = min(MAX_MARKERS_PER_ROOM, max(1, area // 4))
            count = rng.randint(0, upper)
            counts[room_id] = int(count)
            has_nonzero |= count > 0

        if not has_nonzero:
            continue

        equal_edges: list[tuple[str, int, int]] = []
        different_edges: list[tuple[str, int, int]] = []
        for edge in edges:
            room_a, room_b = _edge_rooms(edge)
            if counts[room_a] == counts[room_b]:
                equal_edges.append(edge)
            else:
                different_edges.append(edge)

        if equal_edges and different_edges:
            return counts, equal_edges, different_edges

    return None


def _pick_doors(
    rng: random.Random,
    equal_edges: list[tuple[str, int, int]],
    different_edges: list[tuple[str, int, int]],
    total_edges: int,
) -> list[tuple[str, int, int]]:
    target = rng.randint(2, min(MAX_DOORS, total_edges))

    chosen: list[tuple[str, int, int]] = []
    chosen_set: set[tuple[str, int, int]] = set()

    first_equal = rng.choice(equal_edges)
    first_diff = rng.choice(different_edges)
    chosen.extend([first_equal, first_diff])
    chosen_set.update(chosen)

    remaining_equal = [edge for edge in equal_edges if edge not in chosen_set]
    remaining_diff = [edge for edge in different_edges if edge not in chosen_set]
    take_equal = rng.choice([True, False])

    while len(chosen) < target and (remaining_equal or remaining_diff):
        pool = remaining_equal if take_equal else remaining_diff
        if not pool:
            pool = remaining_diff if take_equal else remaining_equal
        edge = rng.choice(pool)
        pool.remove(edge)
        chosen.append(edge)
        chosen_set.add(edge)
        take_equal = not take_equal

    return chosen


def _carve_door(
    grid: np.ndarray,
    bg: int,
    wall_rows: list[int],
    wall_cols: list[int],
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    edge: tuple[str, int, int],
    rng: random.Random,
) -> None:
    orient, room_row, room_col = edge
    if orient == "v":
        wall_col = wall_cols[room_col + 1]
        r0, r1, _, _ = rooms[(room_row, room_col)]
        start = rng.randint(r0, r1 - 2)
        grid[start : start + 2, wall_col] = bg
    else:
        wall_row = wall_rows[room_row + 1]
        _, _, c0, c1 = rooms[(room_row, room_col)]
        start = rng.randint(c0, c1 - 2)
        grid[wall_row, start : start + 2] = bg


def _place_markers(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    counts: dict[tuple[int, int], int],
    rng: random.Random,
) -> None:
    for room_id, count in counts.items():
        if count == 0:
            continue
        r0, r1, c0, c1 = rooms[room_id]
        cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
        for row, col in rng.sample(cells, count):
            grid[row, col] = MARKER_COLOR


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    for _ in range(LAYOUT_RETRIES):
        room_rows = rng.randint(*ROOM_ROW_RANGE)
        room_cols = rng.randint(*ROOM_COL_RANGE)
        heights = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
        widths = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
        top_margin = rng.randint(*MARGIN_RANGE)
        bottom_margin = rng.randint(*MARGIN_RANGE)
        left_margin = rng.randint(*MARGIN_RANGE)
        right_margin = rng.randint(*MARGIN_RANGE)

        grid_height = top_margin + bottom_margin + sum(heights) + room_rows + 1
        grid_width = left_margin + right_margin + sum(widths) + room_cols + 1
        if max(grid_height, grid_width) > 30:
            continue

        bg = rng.choice(BACKGROUND_CHOICES)
        grid = np.full((grid_height, grid_width), bg, dtype=int)

        wall_rows = [int(top_margin)]
        for height in heights:
            wall_rows.append(int(wall_rows[-1] + height + 1))
        wall_cols = [int(left_margin)]
        for width in widths:
            wall_cols.append(int(wall_cols[-1] + width + 1))

        top_wall, bottom_wall = wall_rows[0], wall_rows[-1]
        left_wall, right_wall = wall_cols[0], wall_cols[-1]

        for row in wall_rows:
            grid[row, left_wall : right_wall + 1] = WALL_COLOR
        for col in wall_cols:
            grid[top_wall : bottom_wall + 1, col] = WALL_COLOR

        rooms = _room_boxes(wall_rows, wall_cols)
        edges = _adjacent_edges(room_rows, room_cols)
        picked = _pick_counts(rng, rooms, edges)
        if picked is None:
            continue

        counts, equal_edges, different_edges = picked
        chosen_doors = _pick_doors(rng, equal_edges, different_edges, len(edges))
        for edge in chosen_doors:
            _carve_door(grid, bg, wall_rows, wall_cols, rooms, edge, rng)

        _place_markers(grid, rooms, counts, rng)

        color_counts = np.bincount(grid.ravel(), minlength=10)
        if int(color_counts[bg]) <= max(int(color_counts[color]) for color in range(10) if color != bg):
            continue

        analysis = _analyze_layout(grid, bg=bg, wall=WALL_COLOR, marker=MARKER_COLOR)
        door_infos = analysis["door_infos"]
        if len(door_infos) != len(chosen_doors):
            continue
        equal_count = sum(int(info["equal"]) for info in door_infos)
        different_count = len(door_infos) - equal_count
        if equal_count == 0 or different_count == 0:
            continue
        if int(np.count_nonzero(grid == MARKER_COLOR)) == 0:
            continue

        out = main(grid)
        if out.shape != grid.shape or np.any(out == MARKER_COLOR):
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway puzzle")
