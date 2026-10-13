from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 1
PALETTE = tuple(range(2, 10))

MIN_GRID = 16
MAX_GRID = 30
MIN_MARGIN = 2
MAX_MARGIN = 5
MIN_ROOM_SIZE = 3
MIN_ROOMS = 2
MAX_ROOMS = 5
MIN_SEG_LEN = 2
MAX_SEG_LEN = 4
GEN_RETRIES = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _corner_bg(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return Counter(corners).most_common(1)[0][0]


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    options = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != int(bg)]
    assert options, "expected at least one non-background color"
    return max(options, key=lambda item: item[1])[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _room_specs(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    h, w = grid.shape
    rooms: list[dict] = []
    for comp in _components(grid != wall):
        if _touches_border(comp, h, w):
            continue
        colored = [
            (int(r), int(c), int(grid[r, c]))
            for r, c in comp
            if int(grid[r, c]) not in (bg, wall)
        ]
        if len(colored) != 1:
            continue
        seed_r, seed_c, seed_color = colored[0]
        r0, c0, r1, c1 = _bbox(comp)
        rooms.append(
            {
                "coords": comp,
                "seed": (seed_r, seed_c),
                "color": seed_color,
                "bbox": (r0, c0, r1, c1),
            }
        )
    return rooms


def _segment_orientation(grid: np.ndarray, color: int) -> str:
    comps = _components(grid == color)
    assert len(comps) >= 2, "expected seed component and outer segment"
    segment = max(comps, key=len)
    assert len(segment) >= 2, "outer segment must be longer than the seed singleton"
    rows = set(map(int, segment[:, 0]))
    cols = set(map(int, segment[:, 1]))
    if len(rows) == 1:
        return "horizontal"
    if len(cols) == 1:
        return "vertical"
    raise AssertionError("outer segment must be axis-aligned")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with seed-colored stripes whose direction is given by the matching outer segment.

    Input structure:
    - The background is black, and a connected blue wall network forms a closed outer loop with internal partitions.
    - The blue walls split the interior into several disjoint enclosed regions, each containing exactly one single-pixel seed of a unique color.
    - Outside the blue outer wall, each seed color appears again as one short horizontal or vertical line segment.

    Rule:
    - Identify each enclosed non-wall region by flood-filling cells that are not blue; rooms are the components that do not touch the grid border.
    - For each room, use its unique seed color to find the matching outer segment of the same color, and read that segment's orientation.
    - A horizontal segment means fill the room with horizontal stripes; a vertical segment means fill it with vertical stripes.
    - Stripes alternate every row or column starting from the room bbox's minimum row or column, and the original seed pixel itself is cleared back to black.

    Key insight:
    - The deciding information is split across two separate places: the room identity comes from the interior seed color, while the fill direction comes only from the same-colored outer marker segment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_bg(g)
    wall = _largest_non_bg_color(g, bg)

    rooms = _room_specs(g, bg, wall)
    out = g.copy()

    for room in rooms:
        seed_color = int(room["color"])
        seed_r, seed_c = room["seed"]
        r0, c0, _r1, _c1 = room["bbox"]
        orientation = _segment_orientation(g, seed_color)

        for r, c in room["coords"]:
            rr = int(r)
            cc = int(c)
            if orientation == "horizontal":
                out[rr, cc] = seed_color if (rr - r0) % 2 == 0 else bg
            else:
                out[rr, cc] = seed_color if (cc - c0) % 2 == 0 else bg

        out[seed_r, seed_c] = bg

    return out


def _render_rooms(
    shape: tuple[int, int],
    outer_box: tuple[int, int, int, int],
    walls: list[tuple[str, int, int, int]],
    rooms: list[tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = shape
    top, left, bottom, right = outer_box
    grid = np.full((h, w), BG, dtype=int)
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    for kind, fixed, start, stop in walls:
        if kind == "v":
            grid[start : stop + 1, fixed] = WALL
        else:
            grid[fixed, start : stop + 1] = WALL
    return grid


def _split_layout(
    h: int,
    w: int,
    target_rooms: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], list[tuple[str, int, int, int]], list[tuple[int, int, int, int]]] | None:
    for _ in range(80):
        top = rng.randint(MIN_MARGIN, MAX_MARGIN)
        left = rng.randint(MIN_MARGIN, MAX_MARGIN)
        bottom = h - 1 - rng.randint(MIN_MARGIN, MAX_MARGIN)
        right = w - 1 - rng.randint(MIN_MARGIN, MAX_MARGIN)
        if bottom - top - 1 < 2 * MIN_ROOM_SIZE + 1 and right - left - 1 < 2 * MIN_ROOM_SIZE + 1:
            continue
        if bottom - top - 1 < MIN_ROOM_SIZE or right - left - 1 < MIN_ROOM_SIZE:
            continue

        rooms = [(top + 1, bottom - 1, left + 1, right - 1)]
        walls: list[tuple[str, int, int, int]] = []
        while len(rooms) < target_rooms:
            candidates = []
            for idx, (r0, r1, c0, c1) in enumerate(rooms):
                dirs = []
                if c1 - c0 + 1 >= 2 * MIN_ROOM_SIZE + 1:
                    dirs.append("v")
                if r1 - r0 + 1 >= 2 * MIN_ROOM_SIZE + 1:
                    dirs.append("h")
                if dirs:
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    candidates.append((idx, dirs, area))
            if not candidates:
                break
            areas = [item[2] for item in candidates]
            pick = candidates[rng.choices(range(len(candidates)), weights=areas, k=1)[0]]
            idx, dirs, _area = pick
            r0, r1, c0, c1 = rooms.pop(idx)
            kind = rng.choice(dirs)
            if kind == "v":
                wall_col = rng.randint(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE)
                walls.append(("v", wall_col, r0 - 1, r1 + 1))
                rooms.extend([(r0, r1, c0, wall_col - 1), (r0, r1, wall_col + 1, c1)])
            else:
                wall_row = rng.randint(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE)
                walls.append(("h", wall_row, c0 - 1, c1 + 1))
                rooms.extend([(r0, wall_row - 1, c0, c1), (wall_row + 1, r1, c0, c1)])
        if len(rooms) == target_rooms:
            return (top, left, bottom, right), walls, rooms
    return None


def _expected_output(
    grid: np.ndarray,
    room_infos: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for info in room_infos:
        r0, r1, c0, c1 = info["rect"]
        color = int(info["color"])
        orientation = info["orientation"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if orientation == "horizontal":
                    out[r, c] = color if (r - r0) % 2 == 0 else BG
                else:
                    out[r, c] = color if (c - c0) % 2 == 0 else BG
        seed_r, seed_c = info["seed"]
        out[seed_r, seed_c] = BG
    return out


def _candidate_segments(
    shape: tuple[int, int],
    outer_box: tuple[int, int, int, int],
    length: int,
    orientation: str,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    top, left, bottom, right = outer_box
    candidates: list[list[tuple[int, int]]] = []

    if orientation == "horizontal":
        if top - 1 >= 0:
            for c0 in range(left, right - length + 2):
                candidates.append([(top - 1, c) for c in range(c0, c0 + length)])
        if bottom + 1 < h:
            for c0 in range(left, right - length + 2):
                candidates.append([(bottom + 1, c) for c in range(c0, c0 + length)])
        if left - length >= 0:
            for r in range(top, bottom + 1):
                candidates.append([(r, c) for c in range(left - length, left)])
        if right + length < w:
            for r in range(top, bottom + 1):
                candidates.append([(r, c) for c in range(right + 1, right + length + 1)])
    else:
        if left - 1 >= 0:
            for r0 in range(top, bottom - length + 2):
                candidates.append([(r, left - 1) for r in range(r0, r0 + length)])
        if right + 1 < w:
            for r0 in range(top, bottom - length + 2):
                candidates.append([(r, right + 1) for r in range(r0, r0 + length)])
        if top - length >= 0:
            for c in range(left, right + 1):
                candidates.append([(r, c) for r in range(top - length, top)])
        if bottom + length < h:
            for c in range(left, right + 1):
                candidates.append([(r, c) for r in range(bottom + 1, bottom + length + 1)])

    return candidates


def _segment_fits(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    touches_wall = False
    for r, c in cells:
        if grid[r, c] != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not _in_bounds(h, w, rr, cc) or (rr, cc) in cell_set:
                    continue
                val = int(grid[rr, cc])
                if val not in (BG, WALL):
                    return False
        for nr, nc in neighbors4(r, c):
            if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == WALL:
                touches_wall = True
    return touches_wall


def _place_segment(
    grid: np.ndarray,
    outer_box: tuple[int, int, int, int],
    color: int,
    orientation: str,
    rng: random.Random,
) -> bool:
    length = rng.randint(MIN_SEG_LEN, MAX_SEG_LEN)
    candidates = _candidate_segments(grid.shape, outer_box, length, orientation)
    rng.shuffle(candidates)
    for cells in candidates:
        if not _segment_fits(grid, cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    room_count = rng.randint(MIN_ROOMS, MAX_ROOMS)

    layout = _split_layout(h, w, room_count, rng)
    if layout is None:
        return None
    outer_box, walls, rooms = layout
    grid = _render_rooms((h, w), outer_box, walls, rooms)

    colors = rng.sample(list(PALETTE), room_count)
    orientations = [rng.choice(["horizontal", "vertical"]) for _ in range(room_count)]
    room_infos: list[dict] = []

    for rect, color, orientation in zip(rooms, colors, orientations):
        r0, r1, c0, c1 = rect
        seed_r = rng.randint(r0, r1)
        seed_c = rng.randint(c0, c1)
        grid[seed_r, seed_c] = color
        room_infos.append(
            {
                "rect": rect,
                "color": color,
                "orientation": orientation,
                "seed": (seed_r, seed_c),
            }
        )

    order = list(range(room_count))
    rng.shuffle(order)
    for idx in order:
        if not _place_segment(grid, outer_box, room_infos[idx]["color"], room_infos[idx]["orientation"], rng):
            return None

    expected = _expected_output(grid, room_infos)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    if _largest_non_bg_color(grid, BG) != WALL:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
