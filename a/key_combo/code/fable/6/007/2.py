from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import components_mask, get_bg


WALL_COLOR = 5
MIN_ROOM_SIDE = 7
MAX_GEN_ATTEMPTS = 400
ROOM_GAP_SIDE = {
    "TL": "top",
    "TR": "right",
    "BL": "left",
    "BR": "bottom",
}


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    if int(bg) != WALL_COLOR and np.any(grid == WALL_COLOR):
        return WALL_COLOR
    counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    assert counts, "expected at least one non-background color"
    return counts.most_common(1)[0][0]


def _paint_centered_square(out: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    radius = int(side) // 2
    cr, cc = map(int, center)
    out[cr - radius : cr + radius + 1, cc - radius : cc + radius + 1] = int(color)


def _extract_squares(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    squares: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(wall)}):
        for coords in components_mask(grid == color, connectivity=4):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            side_r = int(r1 - r0 + 1)
            side_c = int(c1 - c0 + 1)
            assert side_r == side_c, "foreground object must have a square bbox"
            assert side_r % 2 == 1, "square side must be odd"
            assert int(coords.shape[0]) == side_r * side_r, "foreground square must be solid"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "foreground square must fill its bbox"
            squares.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "side": int(side_r),
                    "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
                }
            )
    return squares


def _boundary_exit_count(room_mask: np.ndarray) -> int:
    boundary = np.zeros_like(room_mask, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return len(components_mask(room_mask & boundary, connectivity=4))


def _max_centered_odd_side(room_mask: np.ndarray, center: tuple[int, int]) -> int:
    cr, cc = map(int, center)
    assert room_mask[cr, cc], "square center must lie inside its room"
    radius = 0
    while True:
        next_radius = radius + 1
        r0 = cr - next_radius
        c0 = cc - next_radius
        r1 = cr + next_radius
        c1 = cc + next_radius
        if r0 < 0 or c0 < 0 or r1 >= room_mask.shape[0] or c1 >= room_mask.shape[1]:
            return 2 * radius + 1
        if not room_mask[r0 : r1 + 1, c0 : c1 + 1].all():
            return 2 * radius + 1
        radius = next_radius


def _analyze_rooms(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = int(_infer_wall_color(g, bg))

    room_coords = components_mask(g != wall, connectivity=4)
    assert room_coords, "expected at least one room"
    labels = np.full(g.shape, -1, dtype=int)
    for room_id, coords in enumerate(room_coords):
        labels[coords[:, 0], coords[:, 1]] = int(room_id)

    squares = _extract_squares(g, bg, wall)
    assert len(squares) == len(room_coords), "expected exactly one square per room"

    by_room: dict[int, list[dict]] = {room_id: [] for room_id in range(len(room_coords))}
    for sq in squares:
        rr, cc = sq["center"]
        room_id = int(labels[rr, cc])
        assert room_id >= 0, "square center must belong to a room"
        assert np.all(labels[sq["coords"][:, 0], sq["coords"][:, 1]] == room_id), "square must stay inside one room"
        by_room[room_id].append(sq)

    info: list[dict] = []
    for room_id, coords in enumerate(room_coords):
        sqs = by_room[room_id]
        assert len(sqs) == 1, "each room must contain exactly one square"
        sq = sqs[0]
        room_mask = labels == room_id
        info.append(
            {
                "room_id": int(room_id),
                "room_mask": room_mask,
                "color": int(sq["color"]),
                "side": int(sq["side"]),
                "center": sq["center"],
                "exit_count": int(_boundary_exit_count(room_mask)),
                "max_side": int(_max_centered_odd_side(room_mask, sq["center"])),
            }
        )
    return bg, wall, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand squares in sealed rooms to the largest centered odd square, and deflate squares in leaking rooms by one layer per leak.

    Input structure:
    - The dominant color is the floor/background, and gray wall lines partition the grid into 4-connected rooms.
    - Every room contains exactly one monochrome solid odd-sided square, and square colors are unrelated to room type.
    - Some rooms are fully sealed by walls; others have one or more wall openings that let the room reach the grid boundary.

    Rule:
    - Keep the wall layout fixed and clear all original squares back to the floor color.
    - For a sealed room, keep the square center fixed and replace it with the largest odd-sided solid square that still fits entirely inside that room.
    - For a leaking room, count its distinct boundary-reaching leaks as g and replace side n by n - 2g; if the result is below 1, the square disappears.

    Key insight:
    - The room's connectivity decides whether its square inflates or deflates, and the amount of deflation is determined by the number of separate leaks rather than by a constant one-layer shrink.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall, rooms = _analyze_rooms(g)

    out = np.full_like(g, bg)
    out[g == wall] = wall

    for room in rooms:
        if room["exit_count"] == 0:
            new_side = room["max_side"]
        else:
            new_side = int(room["side"]) - 2 * int(room["exit_count"])
        if new_side >= 1:
            _paint_centered_square(out, room["center"], new_side, room["color"])

    return out


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [v for v in range(int(lo), int(hi) + 1) if v % 2 == 1]


def _choose_odd_split_sizes(total: int) -> tuple[int, int]:
    first_choices = [v for v in range(MIN_ROOM_SIDE, total - MIN_ROOM_SIDE + 1, 2)]
    first = random.choice(first_choices)
    return int(first), int(total - first)


def _choose_non_adjacent(values: list[int], count: int) -> list[int]:
    for _ in range(200):
        sample = sorted(random.sample(values, count))
        if all(b - a >= 2 for a, b in zip(sample, sample[1:])):
            return sample
    raise RuntimeError("failed to choose separated gap positions")


def _room_center(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _room_max_side(room: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = room
    return min(int(r1 - r0 + 1), int(c1 - c0 + 1))


def _open_gaps(grid: np.ndarray, room: tuple[int, int, int, int], side: str, gap_count: int, bg: int) -> None:
    if gap_count <= 0:
        return
    r0, c0, r1, c1 = map(int, room)
    if side == "top":
        cols = _choose_non_adjacent(list(range(c0 + 1, c1)), gap_count)
        for c in cols:
            grid[0, c] = int(bg)
    elif side == "bottom":
        cols = _choose_non_adjacent(list(range(c0 + 1, c1)), gap_count)
        for c in cols:
            grid[-1, c] = int(bg)
    elif side == "left":
        rows = _choose_non_adjacent(list(range(r0 + 1, r1)), gap_count)
        for r in rows:
            grid[r, 0] = int(bg)
    elif side == "right":
        rows = _choose_non_adjacent(list(range(r0 + 1, r1)), gap_count)
        for r in rows:
            grid[r, -1] = int(bg)
    else:
        raise AssertionError(side)


def _choose_square_side(max_side: int, gap_count: int, must_expand: bool, must_survive: bool) -> int:
    max_side = int(max_side)
    if gap_count == 0:
        choices = _odd_choices(1, max_side - 2)
        assert choices, "closed room must allow expansion"
        if must_expand:
            strong = [v for v in choices if v <= max_side - 4]
            if strong:
                choices = strong
        return int(random.choice(choices))

    if must_survive:
        choices = [v for v in _odd_choices(1, max_side) if v - 2 * gap_count >= 1]
        strong = [v for v in choices if v > 1]
        if strong:
            choices = strong
        return int(random.choice(choices))

    disappear = random.random() < 0.45
    if disappear:
        vanish = [v for v in _odd_choices(1, max_side) if v - 2 * gap_count < 1]
        if vanish:
            return int(random.choice(vanish))
    return int(random.choice(_odd_choices(max(1, 2 * gap_count + 1), max_side)))


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([21, 23, 25, 27, 29])
    w = random.choice([21, 23, 25, 27, 29])
    top_h, bottom_h = _choose_odd_split_sizes(h - 3)
    left_w, right_w = _choose_odd_split_sizes(w - 3)

    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    square_colors = random.sample([c for c in range(10) if c not in {bg, WALL_COLOR}], 4)

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = WALL_COLOR
    grid[-1, :] = WALL_COLOR
    grid[:, 0] = WALL_COLOR
    grid[:, -1] = WALL_COLOR

    split_r = top_h + 1
    split_c = left_w + 1
    grid[split_r, :] = WALL_COLOR
    grid[:, split_c] = WALL_COLOR

    rooms = {
        "TL": (1, 1, top_h, left_w),
        "TR": (1, split_c + 1, top_h, w - 2),
        "BL": (split_r + 1, 1, h - 2, left_w),
        "BR": (split_r + 1, split_c + 1, h - 2, w - 2),
    }

    role_counts = [0, 1, random.choice([2, 3]), random.choice([0, 1, 2, 3])]
    random.shuffle(role_counts)
    if 0 not in role_counts or 1 not in role_counts or max(role_counts) < 2:
        return None

    room_order = list(rooms)
    random.shuffle(room_order)
    gap_by_room = dict(zip(room_order, role_counts))

    for name, gap_count in gap_by_room.items():
        _open_gaps(grid, rooms[name], ROOM_GAP_SIDE[name], gap_count, bg)

    room_names = list(rooms)
    random.shuffle(room_names)
    guaranteed_expand_room = next(name for name in room_names if gap_by_room[name] == 0)
    guaranteed_survive_room = next(name for name in room_names if gap_by_room[name] == 1)

    total_square_area = 0
    for name, color in zip(rooms, square_colors):
        gap_count = gap_by_room[name]
        max_side = _room_max_side(rooms[name])
        side = _choose_square_side(
            max_side=max_side,
            gap_count=gap_count,
            must_expand=(name == guaranteed_expand_room),
            must_survive=(name == guaranteed_survive_room),
        )
        total_square_area += side * side
        _paint_centered_square(grid, _room_center(rooms[name]), side, color)

    wall_count = int(np.count_nonzero(grid == WALL_COLOR))
    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= wall_count:
        return None
    color_counts = Counter(int(v) for v in grid.ravel())
    if bg_count <= max(count for color, count in color_counts.items() if color != int(bg)):
        return None
    if int(get_bg(grid)) != int(bg):
        return None

    if np.count_nonzero(grid != bg) <= total_square_area:
        return None

    try:
        analyzed_bg, analyzed_wall, info = _analyze_rooms(grid)
    except AssertionError:
        return None
    if analyzed_bg != bg or analyzed_wall != WALL_COLOR or len(info) != 4:
        return None

    recovered_gaps = sorted(int(room["exit_count"]) for room in info)
    intended_gaps = sorted(int(v) for v in gap_by_room.values())
    if recovered_gaps != intended_gaps:
        return None
    if not any(int(room["exit_count"]) == 0 and int(room["max_side"]) > int(room["side"]) for room in info):
        return None
    if not any(int(room["exit_count"]) >= 2 for room in info):
        return None
    if not any(int(room["exit_count"]) == 1 and int(room["side"]) > 1 for room in info):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    non_bg_out = set(map(int, np.unique(out))) - {int(bg)}
    if not non_bg_out:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
