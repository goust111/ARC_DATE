from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUTS = ("left", "right", "top", "bottom")

GRID_MIN = 18
GRID_MAX = 30
ROOM_COUNT_MIN = 2
ROOM_COUNT_MAX = 5
OUTER_H_MIN = 4   # interior height >= 2
OUTER_H_MAX = 8
OUTER_W_MIN = 5   # interior width >= 3
OUTER_W_MAX = 10
PACK_MARGIN = 1
PACK_GAP = 1
GEN_ATTEMPTS = 400


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _room_boundary_positions(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        boundary.add((int(r0 - 1), int(c)))
        boundary.add((int(r1 + 1), int(c)))
    for r in range(r0, r1 + 1):
        boundary.add((int(r), int(c0 - 1)))
        boundary.add((int(r), int(c1 + 1)))
    return boundary


def _extract_rooms(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _most_common_color(g)

    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected non-background wall pixels"
    wall = _most_common_color(non_bg)

    rooms: list[dict[str, object]] = []
    for comp in _components(g == bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue

        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue

        boundary = _room_boundary_positions(r0, c0, r1, c1)
        if any(not (0 <= br < h and 0 <= bc < w) for br, bc in boundary):
            continue
        if any(int(g[br, bc]) == bg for br, bc in boundary):
            continue

        marker_positions = [(br, bc) for br, bc in boundary if int(g[br, bc]) != wall]
        if not marker_positions:
            continue

        marker_colors = {int(g[br, bc]) for br, bc in marker_positions}
        if len(marker_colors) != 1:
            continue

        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "marker_color": int(next(iter(marker_colors))),
                "marker_count": int(len(marker_positions)),
            }
        )

    rooms.sort(key=lambda room: tuple(room["bbox"]))  # deterministic order
    return int(bg), int(wall), rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room from the bottom using the count and color of its wall markers.

    Input structure:
    - Black background surrounds a connected network of gray walls.
    - Some closed rectangular black regions inside that wall network are rooms.
    - Each room's wall contains several adjacent colored marker pixels, all with
      one shared color for that room and a different color from other rooms.

    Rule:
    - Find every enclosed rectangular black room.
    - Count how many colored marker pixels lie on that room's wall and use their
      shared color as the fill color.
    - Fill the room interior with that color from the bottom upward for exactly
      that many rows, capped by the room's interior height.
    - Leave the gray walls and the colored marker pixels unchanged.

    Key insight:
    - The same wall markers encode two independent properties at once: their
      color says what to paint, and their count says how many rows to stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _bg, _wall, rooms = _extract_rooms(g)
    assert rooms, "expected at least one enclosed room"

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        marker_color = int(room["marker_color"])
        room_height = int(r1 - r0 + 1)
        fill_rows = min(int(room["marker_count"]), room_height)
        start_r = int(r1 - fill_rows + 1)
        out[start_r : r1 + 1, c0 : c1 + 1] = marker_color
    return out


def _random_sizes(count: int, min_size: int, max_size: int, capacity: int) -> list[int] | None:
    if count * min_size > capacity:
        return None
    sizes = [min_size] * count
    max_extra = min(capacity - count * min_size, count * (max_size - min_size))
    extra = random.randint(0, max_extra)
    while extra > 0:
        idx = random.randrange(count)
        if sizes[idx] < max_size:
            sizes[idx] += 1
            extra -= 1
    random.shuffle(sizes)
    return sizes


def _pack_positions(length: int, sizes: list[int], margin: int, gap: int) -> list[int] | None:
    required = 2 * margin + sum(sizes) + gap * (len(sizes) - 1)
    if required > length:
        return None

    extras = [0] * (len(sizes) + 1)
    leftover = length - required
    for _ in range(leftover):
        extras[random.randrange(len(extras))] += 1

    starts: list[int] = []
    pos = margin + extras[0]
    for idx, size in enumerate(sizes):
        starts.append(int(pos))
        pos += int(size)
        if idx + 1 < len(sizes):
            pos += int(gap + extras[idx + 1])
    return starts


def _feasible_counts(length: int, min_outer: int) -> list[int]:
    counts = []
    for count in range(ROOM_COUNT_MIN, ROOM_COUNT_MAX + 1):
        required = 2 * PACK_MARGIN + count * min_outer + (count - 1) * PACK_GAP
        if required <= length:
            counts.append(count)
    return counts


def _marker_candidates(room: dict[str, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["outer_bbox"]
    cells: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        cells.append((int(r0), int(c)))
        cells.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.append((int(r), int(c0)))
        cells.append((int(r), int(c1)))
    return cells


def _draw_room_border(grid: np.ndarray, room: dict[str, int]) -> None:
    r0, c0, r1, c1 = room["outer_bbox"]
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL


def _connect_room(grid: np.ndarray, room: dict[str, int], layout: str, trunk: int) -> None:
    r0, c0, r1, c1 = room["outer_bbox"]
    if layout == "left":
        rr = int(room["attach_row"])
        grid[rr, trunk : c0 + 1] = WALL
    elif layout == "right":
        rr = int(room["attach_row"])
        grid[rr, c1 : trunk + 1] = WALL
    elif layout == "top":
        cc = int(room["attach_col"])
        grid[trunk : r0 + 1, cc] = WALL
    else:
        cc = int(room["attach_col"])
        grid[r1 : trunk + 1, cc] = WALL


def _choose_marker_count(room_height: int, boundary_slots: int, full_fill: bool) -> int:
    if full_fill:
        upper = min(boundary_slots, room_height + 3)
        return random.randint(room_height, upper)
    return random.randint(1, room_height - 1)


def _build_layout() -> tuple[np.ndarray, np.ndarray] | None:
    layout = random.choice(LAYOUTS)
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    inp = np.full((h, w), BG, dtype=int)
    expected = inp.copy()

    if layout in ("left", "right"):
        counts = _feasible_counts(h, OUTER_H_MIN)
        if not counts or w < OUTER_W_MIN + 4:
            return None
        room_count = random.choice(counts)
        outer_heights = _random_sizes(
            room_count,
            OUTER_H_MIN,
            OUTER_H_MAX,
            h - 2 * PACK_MARGIN - (room_count - 1) * PACK_GAP,
        )
        if outer_heights is None:
            return None
        row_starts = _pack_positions(h, outer_heights, PACK_MARGIN, PACK_GAP)
        if row_starts is None:
            return None

        outer_widths = [random.randint(OUTER_W_MIN, min(OUTER_W_MAX, w - 4)) for _ in range(room_count)]
        trunk = 1 if layout == "left" else w - 2
        rooms = []
        for idx in range(room_count):
            oh = int(outer_heights[idx])
            ow = int(outer_widths[idx])
            r0 = int(row_starts[idx])
            r1 = int(r0 + oh - 1)
            if layout == "left":
                lo, hi = trunk + 2, w - ow - 1
            else:
                lo, hi = 1, trunk - ow - 1
            if lo > hi:
                return None
            c0 = random.randint(lo, hi)
            c1 = int(c0 + ow - 1)
            rooms.append(
                {
                    "outer_bbox": (r0, c0, r1, c1),
                    "attach_row": random.randint(r0 + 1, r1 - 1),
                }
            )

        attach_rows = [int(room["attach_row"]) for room in rooms]
        inp[min(attach_rows) : max(attach_rows) + 1, trunk] = WALL
    else:
        counts = _feasible_counts(w, OUTER_W_MIN)
        if not counts or h < OUTER_H_MIN + 4:
            return None
        room_count = random.choice(counts)
        outer_widths = _random_sizes(
            room_count,
            OUTER_W_MIN,
            OUTER_W_MAX,
            w - 2 * PACK_MARGIN - (room_count - 1) * PACK_GAP,
        )
        if outer_widths is None:
            return None
        col_starts = _pack_positions(w, outer_widths, PACK_MARGIN, PACK_GAP)
        if col_starts is None:
            return None

        outer_heights = [random.randint(OUTER_H_MIN, min(OUTER_H_MAX, h - 4)) for _ in range(room_count)]
        trunk = 1 if layout == "top" else h - 2
        rooms = []
        for idx in range(room_count):
            oh = int(outer_heights[idx])
            ow = int(outer_widths[idx])
            c0 = int(col_starts[idx])
            c1 = int(c0 + ow - 1)
            if layout == "top":
                lo, hi = trunk + 2, h - oh - 1
            else:
                lo, hi = 1, trunk - oh - 1
            if lo > hi:
                return None
            r0 = random.randint(lo, hi)
            r1 = int(r0 + oh - 1)
            rooms.append(
                {
                    "outer_bbox": (r0, c0, r1, c1),
                    "attach_col": random.randint(c0 + 1, c1 - 1),
                }
            )

        attach_cols = [int(room["attach_col"]) for room in rooms]
        inp[trunk, min(attach_cols) : max(attach_cols) + 1] = WALL

    colors = random.sample(MARKER_COLORS, len(rooms))
    fill_modes = [False] * (len(rooms) // 2) + [True] * (len(rooms) - len(rooms) // 2)
    random.shuffle(fill_modes)

    for room, color, full_fill in zip(rooms, colors, fill_modes):
        _draw_room_border(inp, room)
        _connect_room(inp, room, layout, trunk)

        candidates = _marker_candidates(room)
        room_height = int(room["outer_bbox"][2] - room["outer_bbox"][0] - 1)
        marker_count = _choose_marker_count(room_height, len(candidates), bool(full_fill))
        markers = random.sample(candidates, marker_count)
        for r, c in markers:
            inp[r, c] = int(color)

        expected_room = room["outer_bbox"]
        r0, c0, r1, c1 = map(int, expected_room)
        expected[r0 : r1 + 1, c0] = inp[r0 : r1 + 1, c0]
        expected[r0 : r1 + 1, c1] = inp[r0 : r1 + 1, c1]
        expected[r0, c0 : c1 + 1] = inp[r0, c0 : c1 + 1]
        expected[r1, c0 : c1 + 1] = inp[r1, c0 : c1 + 1]

        fill_rows = min(marker_count, room_height)
        start_r = int(r1 - fill_rows)
        expected[start_r:r1, c0 + 1 : c1] = int(color)

    expected[inp == WALL] = WALL
    expected[(inp != BG) & (inp != WALL)] = inp[(inp != BG) & (inp != WALL)]
    return inp, expected


def _valid_generated_grid(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        _bg, wall, rooms = _extract_rooms(inp)
    except Exception:
        return False
    if wall != WALL or len(rooms) < ROOM_COUNT_MIN:
        return False

    room_counts = [int(room["marker_count"]) for room in rooms]
    room_heights = [int(room["bbox"][2] - room["bbox"][0] + 1) for room in rooms]
    if not any(count < height for count, height in zip(room_counts, room_heights)):
        return False
    if not any(count >= height for count, height in zip(room_counts, room_heights)):
        return False

    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_layout()
        if built is None:
            continue
        inp, expected = built
        if _valid_generated_grid(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid room-marker ARC input")
