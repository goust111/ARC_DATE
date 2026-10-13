from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


Coord = tuple[int, int]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[Coord] = deque([(r, c)])
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _room_from_component(coords: list[Coord], color: int) -> dict[str, Any] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    coord_set = set(coords)
    missing: list[Coord] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_boundary = r in (r0, r1) or c in (c0, c1)
            if on_boundary:
                if (r, c) not in coord_set:
                    missing.append((r, c))
            elif (r, c) in coord_set:
                return None

    if len(coords) != 2 * (h + w) - 5 or len(missing) != 1:
        return None

    opening = missing[0]
    if opening[0] in (r0, r1) and opening[1] in (c0, c1):
        return None

    if opening[0] == r0:
        step = (-1, 0)
    elif opening[0] == r1:
        step = (1, 0)
    elif opening[1] == c0:
        step = (0, -1)
    else:
        step = (0, 1)

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "step": step,
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components4(grid == color):
            room = _room_from_component(coords, color)
            if room is not None:
                rooms.append(room)
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Empty each open rectangular room by pouring its interior dots outward through the opening.

    Input structure:
    - The background is black.
    - Several foreground colors form one-pixel-thick rectangular frames, and each
      frame is missing exactly one boundary pixel on one side.
    - The interior of each frame contains scattered pixels of one shared dot color.
    - Frames do not overlap, and the space outside each opening is clear.

    Rule:
    - Detect each colored frame and locate its unique opening on the top, bottom,
      left, or right wall.
    - Count the interior non-background dot pixels inside that frame.
    - Erase those interior dots to the background.
    - Starting from the first cell immediately outside the opening, draw a straight
      line of the same dot color in the opening direction, with length equal to the count.
    - Leave every frame pixel unchanged.

    Key insight:
    - The room color only identifies the container; the opening side determines the
      direction of motion, while the number of interior dots is compressed into the
      length of one external line.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for room in _find_rooms(g, bg):
        r0, c0, r1, c1 = room["bbox"]
        inner_in = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        dot_mask = (inner_in != bg) & (inner_in != int(room["color"]))
        dot_values = inner_in[dot_mask]
        if dot_values.size == 0:
            continue

        dot_color = int(dot_values[0])
        assert np.all(dot_values == dot_color), "expected one shared dot color inside each room"

        inner_out[dot_mask] = bg
        count = int(dot_mask.sum())
        start_r = int(room["opening"][0] + room["step"][0])
        start_c = int(room["opening"][1] + room["step"][1])
        step_r, step_c = room["step"]
        for k in range(count):
            rr = start_r + step_r * k
            cc = start_c + step_c * k
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = dot_color

    return out


BG = 0
DOT = 7
FRAME_COLORS = np.array([1, 2, 3, 4, 5, 6, 8, 9], dtype=int)

GRID_MIN = 16
GRID_MAX = 30
ROOM_H_RANGE = (5, 9)
ROOM_W_RANGE = (5, 9)
MAX_ROOMS = 4
MAX_DOTS = 7
ROOM_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _max_rooms_for_area(area: int) -> int:
    return int(min(MAX_ROOMS, max(1, area // 130 + 1)))


def _isolated_interior_cells(inner_h: int, inner_w: int, rng: np.random.Generator) -> list[Coord]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    rng.shuffle(cells)
    chosen: list[Coord] = []
    blocked: set[Coord] = set()
    for r, c in cells:
        if (r, c) in blocked:
            continue
        chosen.append((r, c))
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < inner_h and 0 <= cc < inner_w:
                    blocked.add((rr, cc))
    return chosen


def _mark_blocked(blocked: np.ndarray, cells: list[Coord]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _room_cells(
    r0: int,
    c0: int,
    h: int,
    w: int,
    opening: Coord,
    step: Coord,
    count: int,
) -> list[Coord]:
    cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    for k in range(1, count + 1):
        cells.append((opening[0] + step[0] * k, opening[1] + step[1] * k))
    return cells


def _draw_room(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, opening: Coord) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[opening] = BG


def _paint_output_line(grid: np.ndarray, opening: Coord, step: Coord, count: int) -> None:
    for k in range(1, count + 1):
        grid[opening[0] + step[0] * k, opening[1] + step[1] * k] = DOT


def _sample_room(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    h_grid, w_grid = grid.shape
    for _ in range(ROOM_ATTEMPTS):
        h = int(rng.integers(*ROOM_H_RANGE))
        w = int(rng.integers(*ROOM_W_RANGE))
        direction = int(rng.integers(4))

        if direction == 0:
            step = (-1, 0)
            min_r0, max_r0 = 1, h_grid - h
            min_c0, max_c0 = 0, w_grid - w
        elif direction == 1:
            step = (1, 0)
            min_r0, max_r0 = 0, h_grid - h - 1
            min_c0, max_c0 = 0, w_grid - w
        elif direction == 2:
            step = (0, -1)
            min_r0, max_r0 = 0, h_grid - h
            min_c0, max_c0 = 1, w_grid - w
        else:
            step = (0, 1)
            min_r0, max_r0 = 0, h_grid - h
            min_c0, max_c0 = 0, w_grid - w - 1

        if min_r0 > max_r0 or min_c0 > max_c0:
            continue

        r0 = int(rng.integers(min_r0, max_r0 + 1))
        c0 = int(rng.integers(min_c0, max_c0 + 1))

        inner_choices = _isolated_interior_cells(h - 2, w - 2, rng)
        if not inner_choices:
            continue

        if step == (-1, 0):
            opening = (r0, c0 + int(rng.integers(1, w - 1)))
            space = r0
        elif step == (1, 0):
            opening = (r0 + h - 1, c0 + int(rng.integers(1, w - 1)))
            space = h_grid - (r0 + h)
        elif step == (0, -1):
            opening = (r0 + int(rng.integers(1, h - 1)), c0)
            space = c0
        else:
            opening = (r0 + int(rng.integers(1, h - 1)), c0 + w - 1)
            space = w_grid - (c0 + w)

        max_count = min(MAX_DOTS, len(inner_choices), int(space))
        if max_count < 1:
            continue
        count = int(rng.integers(1, max_count + 1))

        used = _room_cells(r0, c0, h, w, opening, step, count)
        if any(blocked[r, c] for r, c in used):
            continue

        dots = [(r0 + 1 + dr, c0 + 1 + dc) for dr, dc in inner_choices[:count]]

        _draw_room(grid, r0, c0, h, w, color, opening)
        for r, c in dots:
            grid[r, c] = DOT
        _mark_blocked(blocked, used)

        return {
            "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
            "opening": opening,
            "step": step,
            "dots": dots,
            "count": count,
        }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    room_count = int(rng.integers(1, _max_rooms_for_area(h * w) + 1))
    colors = [int(v) for v in rng.choice(FRAME_COLORS, size=room_count, replace=False)]

    specs: list[dict[str, Any]] = []
    for color in colors:
        room = _sample_room(grid, blocked, color, rng)
        if room is None:
            return None
        specs.append(room)

    if _background_color(grid) != BG:
        return None
    if len(_find_rooms(grid, BG)) != room_count:
        return None

    expected = grid.copy()
    for spec in specs:
        for r, c in spec["dots"]:
            expected[r, c] = BG
        _paint_output_line(expected, spec["opening"], spec["step"], int(spec["count"]))

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
