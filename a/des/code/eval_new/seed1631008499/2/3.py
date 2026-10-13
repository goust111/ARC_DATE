import random
from collections import Counter
from functools import lru_cache

import numpy as np


BG = 0
WALL = 8
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
SIDE_ORDER = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
SIDE_OFFSET = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GEN_RETRIES = 300


def _nearest_side(rel_r: int, rel_c: int, room_h: int, room_w: int) -> tuple[str, bool]:
    distances = (
        ("up", int(rel_r)),
        ("down", int(room_h - 1 - rel_r)),
        ("left", int(rel_c)),
        ("right", int(room_w - 1 - rel_c)),
    )
    min_dist = min(dist for _, dist in distances)
    side = next(name for name, dist in distances if dist == min_dist)
    is_tie = sum(dist == min_dist for _, dist in distances) > 1
    return side, is_tie


def _find_wall_layout(grid: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    h, w = map(int, grid.shape)
    best = None
    for color in map(int, np.unique(grid)):
        wall_rows = [int(r) for r in np.where(np.all(grid == color, axis=1))[0]]
        wall_cols = [int(c) for c in np.where(np.all(grid == color, axis=0))[0]]
        if len(wall_rows) < 2 or len(wall_cols) < 2:
            continue
        if wall_rows[0] != 0 or wall_rows[-1] != h - 1:
            continue
        if wall_cols[0] != 0 or wall_cols[-1] != w - 1:
            continue

        row_spans = [b - a - 1 for a, b in zip(wall_rows[:-1], wall_rows[1:])]
        col_spans = [b - a - 1 for a, b in zip(wall_cols[:-1], wall_cols[1:])]
        if not row_spans or not col_spans:
            continue
        if any(span <= 0 for span in row_spans + col_spans):
            continue
        if len(set(row_spans)) != 1 or len(set(col_spans)) != 1:
            continue

        score = (len(wall_rows) * len(wall_cols), row_spans[0] * col_spans[0])
        if best is None or score > best[0]:
            best = (score, int(color), wall_rows, wall_cols, int(row_spans[0]), int(col_spans[0]))

    assert best is not None, "could not identify the separator wall grid"
    _, wall_color, wall_rows, wall_cols, room_h, room_w = best
    return wall_color, wall_rows, wall_cols, room_h, room_w


def _parse_rooms(grid: np.ndarray) -> tuple[int, int, int, int, list[int], list[int], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    wall_color, wall_rows, wall_cols, room_h, room_w = _find_wall_layout(g)
    counts = Counter(map(int, g.reshape(-1)))
    bg = max((color for color in counts if color != wall_color), key=lambda color: counts[color])

    room_info: dict[tuple[int, int], dict] = {}
    for room_r in range(len(wall_rows) - 1):
        r0 = wall_rows[room_r] + 1
        r1 = wall_rows[room_r + 1] - 1
        for room_c in range(len(wall_cols) - 1):
            c0 = wall_cols[room_c] + 1
            c1 = wall_cols[room_c + 1] - 1
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            marker_pts = np.argwhere((room != bg) & (room != wall_color))
            assert len(marker_pts) == 1, "each room must contain exactly one marker"
            rel_r, rel_c = map(int, marker_pts[0])
            side, is_tie = _nearest_side(rel_r, rel_c, room_h, room_w)
            room_info[(room_r, room_c)] = {
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "marker_pos": (int(r0 + rel_r), int(c0 + rel_c)),
                "marker_color": int(room[rel_r, rel_c]),
                "nearest_side": side,
                "is_tie": bool(is_tie),
            }

    return int(bg), int(wall_color), int(room_h), int(room_w), wall_rows, wall_cols, room_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a one-cell shadow inside each room using the marker's opposite direction and the adjacent room's color.

    Input structure:
    - The background is black and a 1-pixel-thick azure wall lattice partitions the canvas into equal rectangular rooms.
    - Each room contains black interior pixels plus exactly one non-black, non-wall marker pixel.
    - The outer border is part of the same wall lattice, so edge rooms can face outward with no neighboring room beyond that wall.

    Rule:
    - For each room, measure the marker's distance to the top, bottom, left, and right room walls and choose the nearest wall; ties are resolved in the priority order up, down, left, right.
    - The shadow travels from the marker toward the opposite wall, but its color is borrowed from the marker in the neighboring room on the nearest-wall side.
    - If the nearest wall lies on the outer border and therefore has no neighboring room, that room produces no shadow; otherwise fill the black cells along the marker's row or column until the opposite wall, leaving the marker itself unchanged.

    Key insight:
    - The direction comes from the marker's bias inside its own room, but the shadow color comes from the opposite room across the nearest wall rather than from the marker itself.
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall_color, room_h, room_w, wall_rows, wall_cols, rooms = _parse_rooms(g)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    out = g.copy()

    for (room_r, room_c), info in rooms.items():
        nearest_side = info["nearest_side"]
        dr, dc = SIDE_OFFSET[nearest_side]
        nbr = room_r + dr
        nbc = room_c + dc
        if not (0 <= nbr < room_rows and 0 <= nbc < room_cols):
            continue

        shadow_color = rooms[(nbr, nbc)]["marker_color"]
        marker_r, marker_c = info["marker_pos"]
        r0, c0, r1, c1 = info["box"]
        shadow_side = OPPOSITE[nearest_side]

        if shadow_side == "up":
            segment = out[r0:marker_r, marker_c]
        elif shadow_side == "down":
            segment = out[marker_r + 1 : r1 + 1, marker_c]
        elif shadow_side == "left":
            segment = out[marker_r, c0:marker_c]
        else:
            segment = out[marker_r, marker_c + 1 : c1 + 1]
        segment[segment == bg] = shadow_color

    return out


@lru_cache(maxsize=None)
def _position_groups(room_h: int, room_w: int) -> dict[tuple[str, bool], tuple[tuple[int, int], ...]]:
    groups: dict[tuple[str, bool], list[tuple[int, int]]] = {
        (side, tie): [] for side in SIDE_ORDER for tie in (False, True)
    }
    for rel_r in range(room_h):
        for rel_c in range(room_w):
            side, is_tie = _nearest_side(rel_r, rel_c, room_h, room_w)
            groups[(side, is_tie)].append((int(rel_r), int(rel_c)))
    return {key: tuple(vals) for key, vals in groups.items()}


def _border_sides(room: tuple[int, int], room_rows: int, room_cols: int) -> list[str]:
    rr, cc = room
    sides = []
    if rr == 0:
        sides.append("up")
    if rr == room_rows - 1:
        sides.append("down")
    if cc == 0:
        sides.append("left")
    if cc == room_cols - 1:
        sides.append("right")
    return sides


def _color_rooms(room_rows: int, room_cols: int, rng: random.Random) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    for room_r in range(room_rows):
        for room_c in range(room_cols):
            forbidden = set()
            if room_r > 0:
                forbidden.add(colors[(room_r - 1, room_c)])
            if room_c > 0:
                forbidden.add(colors[(room_r, room_c - 1)])
            options = [color for color in MARKER_COLORS if color not in forbidden]
            colors[(room_r, room_c)] = int(rng.choice(options))
    return colors


def _coverage_ok(grid: np.ndarray) -> bool:
    _bg, _wall, _room_h, _room_w, wall_rows, wall_cols, rooms = _parse_rooms(grid)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    sides = {info["nearest_side"] for info in rooms.values()}
    if sides != set(SIDE_ORDER):
        return False
    if not any(info["is_tie"] for info in rooms.values()):
        return False
    has_no_shadow = False
    has_shadow = False
    for (room_r, room_c), info in rooms.items():
        dr, dc = SIDE_OFFSET[info["nearest_side"]]
        nbr = room_r + dr
        nbc = room_c + dc
        if 0 <= nbr < room_rows and 0 <= nbc < room_cols:
            has_shadow = True
        else:
            has_no_shadow = True
    return has_shadow and has_no_shadow


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    room_h = rng.randint(3, 6)
    room_w = rng.randint(3, 6)

    grid_h = room_rows * room_h + room_rows + 1
    grid_w = room_cols * room_w + room_cols + 1
    if max(grid_h, grid_w) > 30:
        return None

    groups = _position_groups(room_h, room_w)
    tie_sides = [side for side in SIDE_ORDER if groups[(side, True)]]
    if not tie_sides:
        return None

    rooms = [(room_r, room_c) for room_r in range(room_rows) for room_c in range(room_cols)]
    rng.shuffle(rooms)

    no_shadow_side = rng.choice(SIDE_ORDER)
    no_shadow_candidates = [room for room in rooms if no_shadow_side in _border_sides(room, room_rows, room_cols)]
    if not no_shadow_candidates:
        return None
    no_shadow_room = rng.choice(no_shadow_candidates)

    tie_side_choices = [side for side in tie_sides if side != no_shadow_side] or tie_sides
    tie_side = rng.choice(tie_side_choices)
    tie_room_choices = [room for room in rooms if room != no_shadow_room]
    if not tie_room_choices:
        tie_room_choices = [no_shadow_room]
    tie_room = rng.choice(tie_room_choices)

    side_by_room: dict[tuple[int, int], str] = {}
    remaining_rooms = [room for room in rooms if room not in (no_shadow_room, tie_room)]

    side_by_room[no_shadow_room] = no_shadow_side
    side_by_room[tie_room] = tie_side

    remaining_sides = [side for side in SIDE_ORDER if side not in {no_shadow_side, tie_side}]
    for side in remaining_sides:
        if not remaining_rooms:
            break
        room = remaining_rooms.pop()
        side_by_room[room] = side

    for room in remaining_rooms:
        side_by_room[room] = rng.choice(SIDE_ORDER)

    tie_rooms = {tie_room}
    for room, side in side_by_room.items():
        if room in tie_rooms or room == no_shadow_room:
            continue
        if groups[(side, True)] and rng.random() < 0.2:
            tie_rooms.add(room)

    marker_pos: dict[tuple[int, int], tuple[int, int]] = {}
    for room in rooms:
        side = side_by_room[room]
        want_tie = room in tie_rooms
        if want_tie and groups[(side, True)]:
            pool = groups[(side, True)]
        else:
            pool = groups[(side, False)] or groups[(side, True)]
        if not pool:
            return None
        marker_pos[room] = rng.choice(pool)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    wall_rows = [room_r * (room_h + 1) for room_r in range(room_rows + 1)]
    wall_cols = [room_c * (room_w + 1) for room_c in range(room_cols + 1)]
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    room_colors = _color_rooms(room_rows, room_cols, rng)
    for (room_r, room_c), color in room_colors.items():
        rel_r, rel_c = marker_pos[(room_r, room_c)]
        abs_r = wall_rows[room_r] + 1 + rel_r
        abs_c = wall_cols[room_c] + 1 + rel_c
        grid[abs_r, abs_c] = color

    if not _coverage_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
