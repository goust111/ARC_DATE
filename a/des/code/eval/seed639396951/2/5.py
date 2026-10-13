from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
WINDOW_COLORS = (1, 2, 3, 6, 7, 8, 9)
ROOM_KINDS = ("single", "cross", "parallel")
GEN_ATTEMPTS = 300


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(map(bool, mask)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _wall_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    non_black = [(color, count) for color, count in counts.items() if color != BLACK]
    assert non_black, "expected at least one non-black color"
    return max(non_black, key=lambda item: item[1])[0]


def _window_signature(window: dict) -> tuple:
    return (window["side"], int(window["start"]), int(window["end"]), int(window["color"]))


def _validate_room(g: np.ndarray, wall_color: int, r0: int, c0: int, r1: int, c1: int) -> dict | None:
    h, w = g.shape
    if not (0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1):
        return None

    if any(int(g[r, c]) != int(wall_color) for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return None

    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
        return None

    windows: list[dict] = []
    total_gaps = 0
    side_slices = [
        ("top", g[r0, c0 + 1 : c1]),
        ("bottom", g[r1, c0 + 1 : c1]),
        ("left", g[r0 + 1 : r1, c0]),
        ("right", g[r0 + 1 : r1, c1]),
    ]

    for side, values in side_slices:
        if np.any((values != BLACK) & (values != int(wall_color))):
            return None

        for start, end in _runs(values == BLACK):
            gap_len = end - start
            if not 1 <= gap_len <= 3:
                return None
            total_gaps += 1
            if total_gaps > 2:
                return None

            markers: list[int] = []
            if side == "top":
                outside_r = r0 - 1
                for offset in range(start, end):
                    val = int(g[outside_r, c0 + 1 + offset])
                    if val == BLACK:
                        continue
                    if val == int(wall_color):
                        return None
                    markers.append(val)
                span_start = c0 + 1 + start
                span_end = c0 + 1 + end
            elif side == "bottom":
                outside_r = r1 + 1
                for offset in range(start, end):
                    val = int(g[outside_r, c0 + 1 + offset])
                    if val == BLACK:
                        continue
                    if val == int(wall_color):
                        return None
                    markers.append(val)
                span_start = c0 + 1 + start
                span_end = c0 + 1 + end
            elif side == "left":
                outside_c = c0 - 1
                for offset in range(start, end):
                    val = int(g[r0 + 1 + offset, outside_c])
                    if val == BLACK:
                        continue
                    if val == int(wall_color):
                        return None
                    markers.append(val)
                span_start = r0 + 1 + start
                span_end = r0 + 1 + end
            else:
                outside_c = c1 + 1
                for offset in range(start, end):
                    val = int(g[r0 + 1 + offset, outside_c])
                    if val == BLACK:
                        continue
                    if val == int(wall_color):
                        return None
                    markers.append(val)
                span_start = r0 + 1 + start
                span_end = r0 + 1 + end

            if len(markers) != 1:
                return None
            windows.append(
                {
                    "side": side,
                    "start": int(span_start),
                    "end": int(span_end),
                    "color": int(markers[0]),
                }
            )

    if total_gaps not in (1, 2):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "windows": windows,
    }


def _find_rooms(g: np.ndarray, wall_color: int) -> list[dict]:
    h, w = g.shape
    wall_mask = g == int(wall_color)
    wall_cols_by_row = [set(map(int, np.flatnonzero(wall_mask[r]))) for r in range(h)]

    rooms: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in range(h - 2):
        if len(wall_cols_by_row[r0]) < 2:
            continue
        for r1 in range(r0 + 2, h):
            common_cols = sorted(wall_cols_by_row[r0] & wall_cols_by_row[r1])
            if len(common_cols) < 2:
                continue
            for idx, c0 in enumerate(common_cols[:-1]):
                for c1 in common_cols[idx + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    bbox = (int(r0), int(c0), int(r1), int(c1))
                    if bbox in seen:
                        continue
                    room = _validate_room(g, wall_color, r0, c0, r1, c1)
                    if room is not None:
                        seen.add(bbox)
                        rooms.append(room)

    assert rooms, "expected at least one room"
    return sorted(rooms, key=lambda room: room["bbox"])


def _paint_rooms(g: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = np.array(g, copy=True, dtype=int)

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        hit_count: dict[tuple[int, int], int] = {}
        first_color: dict[tuple[int, int], int] = {}

        for window in room["windows"]:
            color = int(window["color"])
            if window["side"] in ("top", "bottom"):
                for c in range(int(window["start"]), int(window["end"])):
                    for r in range(r0 + 1, r1):
                        pos = (int(r), int(c))
                        hit_count[pos] = hit_count.get(pos, 0) + 1
                        first_color.setdefault(pos, color)
            else:
                for r in range(int(window["start"]), int(window["end"])):
                    for c in range(c0 + 1, c1):
                        pos = (int(r), int(c))
                        hit_count[pos] = hit_count.get(pos, 0) + 1
                        first_color.setdefault(pos, color)

        for pos, count in hit_count.items():
            out[pos] = YELLOW if count > 1 else first_color[pos]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular rooms with wall gaps as windows, then project colored
    light inward from each window. Cells lit by two windows become yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_color = _wall_color(g)
    rooms = _find_rooms(g, wall_color)
    return _paint_rooms(g, rooms)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _side_capacity(side: str, height: int, width: int) -> int:
    return width - 2 if side in ("top", "bottom") else height - 2


def _sample_gap(side: str, height: int, width: int, blocked: tuple[int, int] | None = None) -> tuple[int, int] | None:
    capacity = _side_capacity(side, height, width)
    candidates: list[tuple[int, int]] = []
    for length in (1, 2, 3):
        if length > capacity:
            continue
        for start in range(capacity - length + 1):
            end = start + length
            if blocked is not None and _intervals_overlap(start, end, blocked[0], blocked[1]):
                continue
            candidates.append((start, end))
    if not candidates:
        return None
    return random.choice(candidates)


def _window_sides(kind: str) -> list[str]:
    if kind == "single":
        return [random.choice(("top", "bottom", "left", "right"))]
    if kind == "cross":
        return [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    return list(random.choice((("top", "bottom"), ("left", "right"))))


def _make_room_spec(r0: int, c0: int, height: int, width: int, kind: str) -> dict | None:
    sides = _window_sides(kind)
    colors = random.sample(WINDOW_COLORS, len(sides))
    windows: list[dict] = []
    used_interval: tuple[int, int] | None = None

    for idx, side in enumerate(sides):
        blocked = None
        if kind == "parallel" and idx == 1:
            blocked = used_interval
        gap = _sample_gap(side, height, width, blocked=blocked)
        if gap is None:
            return None
        if kind == "parallel" and idx == 0:
            used_interval = gap

        start, end = gap
        if side == "top":
            span_start = c0 + 1 + start
            span_end = c0 + 1 + end
            marker_row = r0 - 1
            marker_col = random.randrange(span_start, span_end)
        elif side == "bottom":
            span_start = c0 + 1 + start
            span_end = c0 + 1 + end
            marker_row = r0 + height
            marker_col = random.randrange(span_start, span_end)
        elif side == "left":
            span_start = r0 + 1 + start
            span_end = r0 + 1 + end
            marker_row = random.randrange(span_start, span_end)
            marker_col = c0 - 1
        else:
            span_start = r0 + 1 + start
            span_end = r0 + 1 + end
            marker_row = random.randrange(span_start, span_end)
            marker_col = c0 + width

        windows.append(
            {
                "side": side,
                "start": int(span_start),
                "end": int(span_end),
                "color": int(colors[idx]),
                "marker": (int(marker_row), int(marker_col)),
            }
        )

    return {
        "bbox": (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)),
        "windows": windows,
        "kind": kind,
    }


def _draw_room(grid: np.ndarray, room: dict, wall_color: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = int(wall_color)
    grid[r1, c0 : c1 + 1] = int(wall_color)
    grid[r0 : r1 + 1, c0] = int(wall_color)
    grid[r0 : r1 + 1, c1] = int(wall_color)

    for window in room["windows"]:
        if window["side"] == "top":
            grid[r0, int(window["start"]) : int(window["end"])] = BLACK
        elif window["side"] == "bottom":
            grid[r1, int(window["start"]) : int(window["end"])] = BLACK
        elif window["side"] == "left":
            grid[int(window["start"]) : int(window["end"]), c0] = BLACK
        else:
            grid[int(window["start"]) : int(window["end"]), c1] = BLACK

        marker_r, marker_c = window["marker"]
        grid[int(marker_r), int(marker_c)] = int(window["color"])


def _room_signature(room: dict) -> tuple:
    return (
        tuple(map(int, room["bbox"])),
        tuple(sorted(_window_signature(window) for window in room["windows"])),
    )


def _sample_grid_size() -> tuple[int, int]:
    return random.randint(18, 30), random.randint(18, 30)


def _place_rooms() -> tuple[np.ndarray, list[dict], int] | None:
    height, width = _sample_grid_size()
    grid = np.zeros((height, width), dtype=int)
    wall_color = 5
    target_rooms = random.randint(1, 4)
    reserved = np.zeros((height, width), dtype=bool)
    rooms: list[dict] = []

    for _ in range(target_rooms):
        placed = False
        for _try in range(80):
            kind = random.choice(ROOM_KINDS)
            room_h = random.randint(5, 10)
            room_w = random.randint(5, 10)
            if room_h >= height - 2 or room_w >= width - 2:
                continue

            r0 = random.randint(1, height - room_h - 1)
            c0 = random.randint(1, width - room_w - 1)
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1

            rr0 = max(0, r0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width - 1, c1 + 1)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            room = _make_room_spec(r0, c0, room_h, room_w, kind)
            if room is None:
                continue

            _draw_room(grid, room, wall_color)
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            rooms.append(room)
            placed = True
            break

        if not placed:
            break

    if not rooms:
        return None

    return grid, rooms, wall_color


def _attempt_input() -> np.ndarray | None:
    placed = _place_rooms()
    if placed is None:
        return None
    grid, rooms, wall_color = placed

    expected = _paint_rooms(grid, rooms)
    if np.array_equal(expected, grid):
        return None

    detected = _find_rooms(grid, wall_color)
    if sorted(_room_signature(room) for room in detected) != sorted(_room_signature(room) for room in rooms):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-light puzzle input")
