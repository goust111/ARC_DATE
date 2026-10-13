import random

import numpy as np


GEN_MAX_ATTEMPTS = 300
GEN_PLACE_ATTEMPTS = 80
GEN_ROOMS_RANGE = (1, 5)  # upper bound exclusive
GEN_GRID_MIN = 14
GEN_GRID_MAX = 31  # upper bound exclusive
GEN_INTERIOR_MIN = 3
GEN_INTERIOR_MAX = 9  # upper bound exclusive
GEN_COLORS = list(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dominant_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    if len(values) == 0:
        raise ValueError("expected at least one non-background color")
    return int(values[np.argmax(counts)])


def _depth_from_side(side: str, coord: int, interior_span: int) -> int:
    if side in ("top", "left"):
        return coord
    return interior_span - coord + 1


def _window_direction(side: str) -> tuple[int, int]:
    if side == "top":
        return (1, 0)
    if side == "bottom":
        return (-1, 0)
    if side == "left":
        return (0, 1)
    if side == "right":
        return (0, -1)
    raise ValueError(side)


def _valid_room_candidate(
    grid: np.ndarray, bg: int, wall: int, r0: int, c0: int, r1: int, c1: int
) -> dict | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if int(grid[r0, c0]) != wall or int(grid[r0, c1]) != wall:
        return None
    if int(grid[r1, c0]) != wall or int(grid[r1, c1]) != wall:
        return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None

    windows = []
    per_side_positions: dict[str, list[int]] = {side: [] for side in ("top", "bottom", "left", "right")}

    def add_window(side: str, r: int, c: int, marker_r: int, marker_c: int) -> bool:
        if not (0 <= marker_r < grid.shape[0] and 0 <= marker_c < grid.shape[1]):
            return False
        marker = int(grid[marker_r, marker_c])
        if marker in (bg, wall):
            return False
        local_pos = c - c0 if side in ("top", "bottom") else r - r0
        per_side_positions[side].append(local_pos)
        windows.append(
            {
                "gap": (r, c),
                "dir": _window_direction(side),
                "color": marker,
                "side": side,
            }
        )
        return True

    for c in range(c0 + 1, c1):
        value = int(grid[r0, c])
        if value not in (bg, wall):
            return None
        if value == bg and not add_window("top", r0, c, r0 - 1, c):
            return None

    for c in range(c0 + 1, c1):
        value = int(grid[r1, c])
        if value not in (bg, wall):
            return None
        if value == bg and not add_window("bottom", r1, c, r1 + 1, c):
            return None

    for r in range(r0 + 1, r1):
        value = int(grid[r, c0])
        if value not in (bg, wall):
            return None
        if value == bg and not add_window("left", r, c0, r, c0 - 1):
            return None

    for r in range(r0 + 1, r1):
        value = int(grid[r, c1])
        if value not in (bg, wall):
            return None
        if value == bg and not add_window("right", r, c1, r, c1 + 1):
            return None

    if not (1 <= len(windows) <= 4):
        return None

    for positions in per_side_positions.values():
        positions.sort()
        if any(b - a == 1 for a, b in zip(positions, positions[1:])):
            return None

    return {"bbox": (r0, c0, r1, c1), "windows": windows}


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    wall_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == wall)]
    wall_set = set(wall_cells)
    rooms: dict[tuple[int, int, int, int], dict] = {}

    for r0, c0 in wall_cells:
        for r1, c1 in wall_cells:
            if r1 <= r0 + 1 or c1 <= c0 + 1:
                continue
            if (r0, c1) not in wall_set or (r1, c0) not in wall_set:
                continue
            bbox = (r0, c0, r1, c1)
            if bbox in rooms:
                continue
            room = _valid_room_candidate(grid, bg, wall, r0, c0, r1, c1)
            if room is not None:
                rooms[bbox] = room

    return [rooms[bbox] for bbox in sorted(rooms)]


def _grow_room_lines(out: np.ndarray, room: dict, bg: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    windows = room["windows"]
    length = len(windows)
    active = list(range(length))

    for distance in range(1, length + 1):
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx in active:
            gap_r, gap_c = windows[idx]["gap"]
            dr, dc = windows[idx]["dir"]
            tr = gap_r + dr * distance
            tc = gap_c + dc * distance
            if not (r0 < tr < r1 and c0 < tc < c1):
                continue
            if int(out[tr, tc]) != bg:
                continue
            proposals.setdefault((tr, tc), []).append(idx)

        next_active = []
        # Lines advance in lockstep; simultaneous claims cancel at the would-be crossing.
        for (tr, tc), claimants in proposals.items():
            if len(claimants) != 1:
                continue
            idx = claimants[0]
            out[tr, tc] = windows[idx]["color"]
            next_active.append(idx)
        active = next_active
        if not active:
            break


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _most_common_color(g)
    wall = _dominant_non_bg_color(g, bg)
    rooms = _find_rooms(g, bg, wall)

    out = g.copy()
    for room in rooms:
        _grow_room_lines(out, room, bg)
    return out, rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw inward decorative lines in each hollow room based on its border windows.

    Input structure:
    - bg is the most common color.
    - Several one-cell-thick rectangular rooms are outlined with a shared wall color.
    - Each room border has 1 to 4 single-cell gaps, and each gap has one colored marker
      immediately outside the room on that side.

    Rule:
    - Detect every rectangular room and collect all of its border gaps.
    - For each gap, use the outside marker color and draw a straight line from the gap
      inward, perpendicular to the touched border.
    - Every line in the same room has length equal to that room's total number of gaps.
    - If a line would enter a cell already claimed by another drawn line, it stops before
      that crossing; the gap cells and outside markers remain unchanged.

    Key insight:
    - A gap's side determines the drawing direction, its outside marker determines the
      line color, and the line length is not local to one gap but comes from counting all
      windows belonging to the same room.
    """
    out, _ = _solve(grid)
    return out


def _can_use_window(
    chosen: list[tuple[str, int]], side: str, pos: int, used_vertical: set[int], used_horizontal: set[int]
) -> bool:
    if side in ("top", "bottom") and pos in used_vertical:
        return False
    if side in ("left", "right") and pos in used_horizontal:
        return False
    for other_side, other_pos in chosen:
        if other_side == side and abs(other_pos - pos) <= 1:
            return False
    return True


def _sample_windows(
    interior_h: int, interior_w: int, count: int, force_cross: bool, rng: random.Random
) -> list[tuple[str, int]] | None:
    chosen: list[tuple[str, int]] = []
    used_vertical: set[int] = set()
    used_horizontal: set[int] = set()

    def add(side: str, pos: int) -> None:
        chosen.append((side, pos))
        if side in ("top", "bottom"):
            used_vertical.add(pos)
        else:
            used_horizontal.add(pos)

    if force_cross:
        vertical_side = rng.choice(["top", "bottom"])
        horizontal_side = rng.choice(["left", "right"])

        row_choices = (
            list(range(1, min(count, interior_h) + 1))
            if vertical_side == "top"
            else list(range(max(1, interior_h - count + 1), interior_h + 1))
        )
        col_choices = (
            list(range(1, min(count, interior_w) + 1))
            if horizontal_side == "left"
            else list(range(max(1, interior_w - count + 1), interior_w + 1))
        )

        crossing_pairs = []
        for row in row_choices:
            for col in col_choices:
                vertical_depth = _depth_from_side(vertical_side, row, interior_h)
                horizontal_depth = _depth_from_side(horizontal_side, col, interior_w)
                if vertical_depth != horizontal_depth:
                    crossing_pairs.append((row, col))
        if not crossing_pairs:
            return None
        row, col = rng.choice(crossing_pairs)
        add(vertical_side, col)
        add(horizontal_side, row)

    candidates = []
    for side in ("top", "bottom"):
        for pos in range(1, interior_w + 1):
            candidates.append((side, pos))
    for side in ("left", "right"):
        for pos in range(1, interior_h + 1):
            candidates.append((side, pos))
    rng.shuffle(candidates)

    for side, pos in candidates:
        if len(chosen) >= count:
            break
        if _can_use_window(chosen, side, pos, used_vertical, used_horizontal):
            add(side, pos)

    if len(chosen) != count:
        return None
    rng.shuffle(chosen)
    return chosen


def _render_room(
    grid: np.ndarray,
    top: int,
    left: int,
    interior_h: int,
    interior_w: int,
    windows: list[tuple[str, int]],
    bg: int,
    wall: int,
    colors: list[int],
) -> None:
    bottom = top + interior_h + 1
    right = left + interior_w + 1

    grid[top, left : right + 1] = wall
    grid[bottom, left : right + 1] = wall
    grid[top : bottom + 1, left] = wall
    grid[top : bottom + 1, right] = wall

    for (side, pos), color in zip(windows, colors):
        if side == "top":
            gap_r, gap_c = top, left + pos
            marker_r, marker_c = top - 1, gap_c
        elif side == "bottom":
            gap_r, gap_c = bottom, left + pos
            marker_r, marker_c = bottom + 1, gap_c
        elif side == "left":
            gap_r, gap_c = top + pos, left
            marker_r, marker_c = gap_r, left - 1
        else:
            gap_r, gap_c = top + pos, right
            marker_r, marker_c = gap_r, right + 1
        grid[gap_r, gap_c] = bg
        grid[marker_r, marker_c] = color


def _room_signature(top: int, left: int, interior_h: int, interior_w: int, windows: list[tuple[str, int]]) -> dict:
    bottom = top + interior_h + 1
    right = left + interior_w + 1
    rendered_windows = []
    for side, pos in windows:
        if side == "top":
            rendered_windows.append(("top", (top, left + pos)))
        elif side == "bottom":
            rendered_windows.append(("bottom", (bottom, left + pos)))
        elif side == "left":
            rendered_windows.append(("left", (top + pos, left)))
        else:
            rendered_windows.append(("right", (top + pos, right)))
    return {"bbox": (top, left, bottom, right), "windows": sorted(rendered_windows)}


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX - 1)
    width = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX - 1)

    bg = rng.choice(GEN_COLORS)
    wall_choices = [color for color in GEN_COLORS if color != bg]
    wall = rng.choice(wall_choices)
    marker_palette = [color for color in GEN_COLORS if color not in (bg, wall)]

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    room_count = rng.randrange(*GEN_ROOMS_RANGE)
    placed_signatures = []

    for _ in range(room_count):
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            window_count = rng.randint(1, 4)
            force_cross = window_count >= 2 and rng.random() < 0.5

            min_h = max(GEN_INTERIOR_MIN, window_count)
            min_w = max(GEN_INTERIOR_MIN, window_count)
            interior_h = rng.randint(min_h, GEN_INTERIOR_MAX - 1)
            interior_w = rng.randint(min_w, GEN_INTERIOR_MAX - 1)

            windows = _sample_windows(interior_h, interior_w, window_count, force_cross, rng)
            if windows is None:
                continue

            outer_h = interior_h + 2
            outer_w = interior_w + 2
            if outer_h + 2 > height or outer_w + 2 > width:
                continue

            valid_positions = []
            for top in range(1, height - outer_h):
                for left in range(1, width - outer_w):
                    if occupied[top - 1 : top + outer_h + 1, left - 1 : left + outer_w + 1].any():
                        continue
                    valid_positions.append((top, left))
            if not valid_positions:
                continue

            top, left = rng.choice(valid_positions)
            colors = [rng.choice(marker_palette) for _ in range(window_count)]
            _render_room(grid, top, left, interior_h, interior_w, windows, bg, wall, colors)
            occupied[top - 1 : top + outer_h + 1, left - 1 : left + outer_w + 1] = True
            placed_signatures.append(_room_signature(top, left, interior_h, interior_w, windows))
            placed = True
            break
        if not placed:
            return None

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    _, detected_rooms = _solve(grid)
    if len(detected_rooms) != len(placed_signatures):
        return None

    expected_signatures = sorted(
        (sig["bbox"], tuple(sig["windows"])) for sig in placed_signatures
    )
    actual_signatures = sorted(
        (
            room["bbox"],
            tuple(sorted((window["side"], window["gap"]) for window in room["windows"])),
        )
        for room in detected_rooms
    )
    if expected_signatures != actual_signatures:
        return None

    output = main(grid)
    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
