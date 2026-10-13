from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue

            label = len(components)
            stack = [(r, c)]
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return labels, components


def _wall_rows(grid: np.ndarray) -> list[int]:
    if grid.shape[1] < 2:
        return []
    horizontal_pairs = (grid[:, :-1] == GREEN) & (grid[:, 1:] == GREEN)
    return [int(r) for r in np.flatnonzero(horizontal_pairs.any(axis=1))]


def _wall_cols(grid: np.ndarray) -> list[int]:
    if grid.shape[0] < 2:
        return []
    vertical_pairs = (grid[:-1, :] == GREEN) & (grid[1:, :] == GREEN)
    return [int(c) for c in np.flatnonzero(vertical_pairs.any(axis=0))]


def _is_horizontal_door(grid: np.ndarray, row: int, c0: int, c1: int) -> bool:
    h, w = grid.shape
    length = c1 - c0 + 1
    if not (1 <= length <= 2):
        return False
    if not (0 < row < h - 1):
        return False
    if c0 < 2 or c1 + 2 >= w:
        return False
    if not np.all(grid[row, c0 : c1 + 1] == BLACK):
        return False
    if int(grid[row, c0 - 1]) != GREEN or int(grid[row, c1 + 1]) != GREEN:
        return False
    if int(grid[row, c0 - 2]) != GREEN or int(grid[row, c1 + 2]) != GREEN:
        return False
    if np.any(grid[row - 1, c0 : c1 + 1] == GREEN) or np.any(grid[row + 1, c0 : c1 + 1] == GREEN):
        return False
    return True


def _is_vertical_door(grid: np.ndarray, col: int, r0: int, r1: int) -> bool:
    h, w = grid.shape
    length = r1 - r0 + 1
    if not (1 <= length <= 2):
        return False
    if not (0 < col < w - 1):
        return False
    if r0 < 2 or r1 + 2 >= h:
        return False
    if not np.all(grid[r0 : r1 + 1, col] == BLACK):
        return False
    if int(grid[r0 - 1, col]) != GREEN or int(grid[r1 + 1, col]) != GREEN:
        return False
    if int(grid[r0 - 2, col]) != GREEN or int(grid[r1 + 2, col]) != GREEN:
        return False
    if np.any(grid[r0 : r1 + 1, col - 1] == GREEN) or np.any(grid[r0 : r1 + 1, col + 1] == GREEN):
        return False
    return True


def _find_doors(grid: np.ndarray) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    doors: list[tuple[str, int, int, int]] = []

    for row in _wall_rows(grid):
        if row in (0, h - 1):
            continue
        col = 1
        while col < w - 1:
            if int(grid[row, col]) != BLACK:
                col += 1
                continue
            c0 = col
            while col < w - 1 and int(grid[row, col]) == BLACK:
                col += 1
            c1 = col - 1
            if _is_horizontal_door(grid, row, c0, c1):
                doors.append(("h", row, c0, c1))

    for col in _wall_cols(grid):
        if col in (0, w - 1):
            continue
        row = 1
        while row < h - 1:
            if int(grid[row, col]) != BLACK:
                row += 1
                continue
            r0 = row
            while row < h - 1 and int(grid[row, col]) == BLACK:
                row += 1
            r1 = row - 1
            if _is_vertical_door(grid, col, r0, r1):
                doors.append(("v", col, r0, r1))

    return doors


def _seal_grid(grid: np.ndarray, doors: list[tuple[str, int, int, int]]) -> np.ndarray:
    sealed = grid.copy()
    for kind, fixed, start, end in doors:
        if kind == "h":
            sealed[fixed, start : end + 1] = GREEN
        else:
            sealed[start : end + 1, fixed] = GREEN
    return sealed


def _room_info(sealed: np.ndarray, original: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels, components = _label_components(sealed != GREEN)
    rooms: list[dict[str, object]] = []

    for coords in components:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(coords) == area, "room interiors must be rectangles"
        assert np.all(sealed[r0 : r1 + 1, c0 : c1 + 1] != GREEN), "sealed room cannot contain walls"

        values = {
            int(original[r, c])
            for r, c in coords
            if int(original[r, c]) not in {BLACK}
        }
        assert GREEN not in values, "wall color cannot appear inside a room"
        assert len(values) <= 1, "each room may have at most one floor-marker color"
        marker = None if not values else next(iter(values))
        rooms.append({"bbox": (r0, c0, r1, c1), "marker": marker})

    assert rooms, "expected at least one room"
    return labels, rooms


def _window_color(marker: object) -> int:
    return GREY if marker is None else int(marker)


def _paint_horizontal(out: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    existing = out[row, c0 : c1 + 1]
    assert np.all(existing != GREEN), "window pixels must stay inside the room"
    assert np.all((existing == BLACK) | (existing == color)), "window cannot overwrite a floor marker"
    out[row, c0 : c1 + 1] = color


def _paint_vertical(out: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    existing = out[r0 : r1 + 1, col]
    assert np.all(existing != GREEN), "window pixels must stay inside the room"
    assert np.all((existing == BLACK) | (existing == color)), "window cannot overwrite a floor marker"
    out[r0 : r1 + 1, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect wall-separated rectangular rooms and mirror door information to windows.

    Input structure
    - black is floor.
    - green forms a one-pixel wall network.
    - black gaps of length 1-2 inside a wall are doors.
    - each room has at most one non-black floor-marker color.

    Rule
    - For every door, identify the two rooms it connects.
    - For each room, draw a same-sized window on the opposite wall interior.
    - The window color comes from the neighbor room's floor-marker color, or grey if absent.
    - Keep walls, doors, and original floor markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    doors = _find_doors(g)
    sealed = _seal_grid(g, doors)
    labels, rooms = _room_info(sealed, g)

    out = g.copy()
    for kind, fixed, start, end in doors:
        if kind == "h":
            above = {int(labels[fixed - 1, c]) for c in range(start, end + 1)}
            below = {int(labels[fixed + 1, c]) for c in range(start, end + 1)}
            above.discard(-1)
            below.discard(-1)
            assert len(above) == 1 and len(below) == 1, "door must border exactly two rooms"
            top_room = above.pop()
            bottom_room = below.pop()
            assert top_room != bottom_room, "a door must connect different rooms"

            top_bbox = rooms[top_room]["bbox"]
            bottom_bbox = rooms[bottom_room]["bbox"]
            assert isinstance(top_bbox, tuple) and isinstance(bottom_bbox, tuple)
            _paint_horizontal(out, int(top_bbox[0]), start, end, _window_color(rooms[bottom_room]["marker"]))
            _paint_horizontal(out, int(bottom_bbox[2]), start, end, _window_color(rooms[top_room]["marker"]))
        else:
            left = {int(labels[r, fixed - 1]) for r in range(start, end + 1)}
            right = {int(labels[r, fixed + 1]) for r in range(start, end + 1)}
            left.discard(-1)
            right.discard(-1)
            assert len(left) == 1 and len(right) == 1, "door must border exactly two rooms"
            left_room = left.pop()
            right_room = right.pop()
            assert left_room != right_room, "a door must connect different rooms"

            left_bbox = rooms[left_room]["bbox"]
            right_bbox = rooms[right_room]["bbox"]
            assert isinstance(left_bbox, tuple) and isinstance(right_bbox, tuple)
            _paint_vertical(out, int(left_bbox[1]), start, end, _window_color(rooms[right_room]["marker"]))
            _paint_vertical(out, int(right_bbox[3]), start, end, _window_color(rooms[left_room]["marker"]))

    return out


# --------------- generator ---------------

GEN_ROOM_AXIS_CHOICES = (2, 3)
GEN_ROOM_SPAN_RANGE = (5, 8)
GEN_MARKER_COLORS = (1, 2, 4, 6, 7, 8, 9)
GEN_MARKER_COUNT_RANGE = (1, 3)
GEN_RETRY_LIMIT = 400


def _sample_room_spans(count: int) -> list[int]:
    return [random.randint(*GEN_ROOM_SPAN_RANGE) for _ in range(count)]


def _room_boxes(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            boxes[(i, j)] = (wall_rows[i] + 1, wall_cols[j] + 1, wall_rows[i + 1] - 1, wall_cols[j + 1] - 1)
    return boxes


def _relative_door_intervals(span: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for width in (1, 2):
        for start in range(2, span - width - 1):
            intervals.append((start, start + width - 1))
    return intervals


def _sample_segment_doors(span: int) -> list[tuple[int, int]]:
    intervals = _relative_door_intervals(span)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, first in enumerate(intervals):
        for second in intervals[i + 1 :]:
            if second[0] - first[1] - 1 >= 2:
                pairs.append((first, second))

    options = [0]
    if intervals:
        options.append(1)
    if pairs:
        options.append(2)

    count = random.choice(options)
    if count == 0:
        return []
    if count == 1:
        return [random.choice(intervals)]
    pair = random.choice(pairs)
    return [pair[0], pair[1]]


def _build_layout() -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], list[dict[str, object]]] | None:
    room_rows = random.choice(GEN_ROOM_AXIS_CHOICES)
    room_cols = random.choice(GEN_ROOM_AXIS_CHOICES)
    heights = _sample_room_spans(room_rows)
    widths = _sample_room_spans(room_cols)

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    if h > 30 or w > 30:
        return None

    wall_rows = [0]
    for span in heights:
        wall_rows.append(wall_rows[-1] + span + 1)
    wall_cols = [0]
    for span in widths:
        wall_cols.append(wall_cols[-1] + span + 1)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[wall_rows, :] = GREEN
    grid[:, wall_cols] = GREEN
    boxes = _room_boxes(wall_rows, wall_cols)

    doors: list[dict[str, object]] = []
    for i in range(room_rows - 1):
        wall_row = wall_rows[i + 1]
        for j in range(room_cols):
            segment_start = boxes[(i, j)][1]
            span = widths[j]
            for rel0, rel1 in _sample_segment_doors(span):
                c0 = segment_start + rel0
                c1 = segment_start + rel1
                grid[wall_row, c0 : c1 + 1] = BLACK
                doors.append(
                    {
                        "kind": "h",
                        "fixed": wall_row,
                        "start": c0,
                        "end": c1,
                        "room_a": (i, j),
                        "room_b": (i + 1, j),
                    }
                )

    for i in range(room_rows):
        for j in range(room_cols - 1):
            wall_col = wall_cols[j + 1]
            segment_start = boxes[(i, j)][0]
            span = heights[i]
            for rel0, rel1 in _sample_segment_doors(span):
                r0 = segment_start + rel0
                r1 = segment_start + rel1
                grid[r0 : r1 + 1, wall_col] = BLACK
                doors.append(
                    {
                        "kind": "v",
                        "fixed": wall_col,
                        "start": r0,
                        "end": r1,
                        "room_a": (i, j),
                        "room_b": (i, j + 1),
                    }
                )

    if not doors:
        return None
    return grid, boxes, doors


def _window_targets(
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    doors: list[dict[str, object]],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    targets = {room: set() for room in boxes}
    for door in doors:
        start = int(door["start"])
        end = int(door["end"])
        room_a = door["room_a"]
        room_b = door["room_b"]
        assert isinstance(room_a, tuple) and isinstance(room_b, tuple)
        if door["kind"] == "h":
            top_bbox = boxes[room_a]
            bottom_bbox = boxes[room_b]
            targets[room_a].update((top_bbox[0], c) for c in range(start, end + 1))
            targets[room_b].update((bottom_bbox[2], c) for c in range(start, end + 1))
        else:
            left_bbox = boxes[room_a]
            right_bbox = boxes[room_b]
            targets[room_a].update((r, left_bbox[1]) for r in range(start, end + 1))
            targets[room_b].update((r, right_bbox[3]) for r in range(start, end + 1))
    return targets


def _marker_candidates(
    bbox: tuple[int, int, int, int], blocked: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    inner = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in blocked
    ]
    if inner:
        return inner
    return [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in blocked
    ]


def _assign_markers(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    doors: list[dict[str, object]],
) -> dict[tuple[int, int], int | None] | None:
    targets = _window_targets(boxes, doors)
    active_rooms = {
        room
        for door in doors
        for room in (door["room_a"], door["room_b"])
        if isinstance(room, tuple)
    }
    room_ids = list(boxes)

    has_marker = {room: random.random() < 0.5 for room in room_ids}
    active_list = list(active_rooms)
    if active_list and not any(has_marker[room] for room in active_list):
        has_marker[random.choice(active_list)] = True
    if active_list and all(has_marker[room] for room in active_list):
        has_marker[random.choice(active_list)] = False

    markers: dict[tuple[int, int], int | None] = {room: None for room in room_ids}
    for room in room_ids:
        if not has_marker[room]:
            continue

        candidates = _marker_candidates(boxes[room], targets[room])
        if not candidates:
            return None

        color = random.choice(GEN_MARKER_COLORS)
        count = random.randint(GEN_MARKER_COUNT_RANGE[0], min(GEN_MARKER_COUNT_RANGE[1], len(candidates)))
        for rr, cc in random.sample(candidates, count):
            grid[rr, cc] = color
        markers[room] = color

    return markers


def _expected_output(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    doors: list[dict[str, object]],
    markers: dict[tuple[int, int], int | None],
) -> np.ndarray:
    out = grid.copy()
    for door in doors:
        start = int(door["start"])
        end = int(door["end"])
        room_a = door["room_a"]
        room_b = door["room_b"]
        assert isinstance(room_a, tuple) and isinstance(room_b, tuple)
        if door["kind"] == "h":
            top_bbox = boxes[room_a]
            bottom_bbox = boxes[room_b]
            out[top_bbox[0], start : end + 1] = GREY if markers[room_b] is None else int(markers[room_b])
            out[bottom_bbox[2], start : end + 1] = GREY if markers[room_a] is None else int(markers[room_a])
        else:
            left_bbox = boxes[room_a]
            right_bbox = boxes[room_b]
            out[start : end + 1, left_bbox[1]] = GREY if markers[room_b] is None else int(markers[room_b])
            out[start : end + 1, right_bbox[3]] = GREY if markers[room_a] is None else int(markers[room_a])
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _build_layout()
    if built is None:
        return None
    grid, boxes, doors = built
    markers = _assign_markers(grid, boxes, doors)
    if markers is None:
        return None

    expected = _expected_output(grid, boxes, doors, markers)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-door input")
