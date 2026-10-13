from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _row_sum(prefix: np.ndarray, row: int, c0: int, c1: int) -> int:
    total = int(prefix[row, c1])
    if c0:
        total -= int(prefix[row, c0 - 1])
    return total


def _col_sum(prefix: np.ndarray, col: int, r0: int, r1: int) -> int:
    if r0 > r1:
        return 0
    total = int(prefix[r1, col])
    if r0:
        total -= int(prefix[r0 - 1, col])
    return total


def _border_values(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 + 1 : r1, c0]
    right = grid[r0 + 1 : r1, c1]
    return np.concatenate([top, bottom, left, right])


def _run_positions(line: np.ndarray, value: int) -> list[int]:
    return [int(i) for i in np.flatnonzero(line == value)]


def _is_len2_run(indices: list[int]) -> bool:
    return len(indices) == 2 and indices[1] == indices[0] + 1


def _stripe_orientation(interior: np.ndarray, bg: int) -> str | None:
    colors = [int(color) for color in np.unique(interior) if int(color) != bg]
    if len(colors) != 1:
        return None

    stripe = colors[0]
    fg = interior == stripe
    h, w = fg.shape

    row_counts = np.sum(fg, axis=1)
    col_counts = np.sum(fg, axis=0)

    stripe_rows = np.flatnonzero(row_counts == w)
    stripe_cols = np.flatnonzero(col_counts == h)

    horizontal = (
        stripe_rows.size > 0
        and stripe_rows.size < h
        and np.all((row_counts == 0) | (row_counts == w))
        and np.all(col_counts > 0)
        and (stripe_rows.size == 1 or np.all(np.diff(stripe_rows) == 2))
    )
    vertical = (
        stripe_cols.size > 0
        and stripe_cols.size < w
        and np.all((col_counts == 0) | (col_counts == h))
        and np.all(row_counts > 0)
        and (stripe_cols.size == 1 or np.all(np.diff(stripe_cols) == 2))
    )

    if horizontal == vertical:
        return None
    return "horizontal" if horizontal else "vertical"


def _door_segments(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, door_color: int
) -> list[tuple[str, list[tuple[int, int]]]] | None:
    if any(
        int(grid[r, c]) == door_color
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    ):
        return None

    segments: list[tuple[str, list[tuple[int, int]]]] = []

    top = _run_positions(grid[r0, c0 + 1 : c1], door_color)
    bottom = _run_positions(grid[r1, c0 + 1 : c1], door_color)
    left = _run_positions(grid[r0 + 1 : r1, c0], door_color)
    right = _run_positions(grid[r0 + 1 : r1, c1], door_color)

    for side, indices in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        if not indices:
            continue
        if not _is_len2_run(indices):
            return None
        if side in ("top", "bottom"):
            row = r0 if side == "top" else r1
            coords = [(row, c0 + 1 + idx) for idx in indices]
        else:
            col = c0 if side == "left" else c1
            coords = [(r0 + 1 + idx, col) for idx in indices]
        segments.append((side, coords))

    if len(segments) != 2:
        return None
    axes = {0 if side in ("top", "bottom") else 1 for side, _coords in segments}
    if len(axes) != 2:
        return None
    return segments


def _room_candidate(
    grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int
) -> dict[str, object] | None:
    border = _border_values(grid, r0, c0, r1, c1)
    if np.any(border == bg):
        return None

    colors, counts = np.unique(border, return_counts=True)
    if len(colors) != 2:
        return None

    order = np.argsort(counts)
    door_color = int(colors[int(order[0])])
    door_count = int(counts[int(order[0])])
    wall_color = int(colors[int(order[1])])
    if door_count != 4:
        return None

    segments = _door_segments(grid, r0, c0, r1, c1, door_color)
    if segments is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    orientation = _stripe_orientation(interior, bg)
    if orientation is None:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "wall_color": wall_color,
        "door_color": door_color,
        "segments": segments,
        "orientation": orientation,
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    non_bg = grid != bg
    row_prefix = np.cumsum(non_bg, axis=1, dtype=int)
    col_prefix = np.cumsum(non_bg, axis=0, dtype=int)

    candidates: list[dict[str, object]] = []
    for r0 in range(h - 4):
        for r1 in range(r0 + 4, h):
            inner_h = r1 - r0 - 1
            for c0 in range(w - 4):
                for c1 in range(c0 + 4, w):
                    inner_w = c1 - c0 - 1
                    border_len = 2 * (inner_h + inner_w + 2)
                    border_non_bg = (
                        _row_sum(row_prefix, r0, c0, c1)
                        + _row_sum(row_prefix, r1, c0, c1)
                        + _col_sum(col_prefix, c0, r0 + 1, r1 - 1)
                        + _col_sum(col_prefix, c1, r0 + 1, r1 - 1)
                    )
                    if border_non_bg != border_len:
                        continue
                    room = _room_candidate(grid, bg, r0, c0, r1, c1)
                    if room is not None:
                        candidates.append(room)

    candidates.sort(
        key=lambda room: (
            (room["bbox"][2] - room["bbox"][0] + 1)
            * (room["bbox"][3] - room["bbox"][1] + 1)
        ),
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    used_interior = np.zeros(grid.shape, dtype=bool)
    for room in candidates:
        r0, c0, r1, c1 = room["bbox"]
        if np.any(used_interior[r0 + 1 : r1, c0 + 1 : c1]):
            continue
        selected.append(room)
        used_interior[r0 + 1 : r1, c0 + 1 : c1] = True

    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the door whose wall axis matches each room's stripe direction.

    Input structure:
    - The background is black.
    - Several rectangular rooms are enclosed by one-pixel walls of one color.
    - Each room border contains exactly two two-cell door segments in a second color.
    - Inside each room, one foreground color forms either horizontal stripes or vertical stripes with one-cell background gaps.

    Rule:
    - Detect every rectangular room from its non-background border.
    - Read the stripe orientation inside that room.
    - If the stripes are horizontal, erase the door on the left or right wall.
    - If the stripes are vertical, erase the door on the top or bottom wall.
    - Leave the other door, the walls, and the floor pattern unchanged.

    Key insight:
    - The stripes do not move; they only choose which door axis is activated, so the opened door is the one perpendicular to the stripe direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rooms = _find_rooms(g, bg)
    assert rooms, "expected at least one room"

    out = g.copy()
    for room in rooms:
        want_vertical_wall = room["orientation"] == "horizontal"
        for side, coords in room["segments"]:
            is_vertical_wall = side in ("left", "right")
            if is_vertical_wall == want_vertical_wall:
                for r, c in coords:
                    out[int(r), int(c)] = bg
    return out


_BG = 0
_PALETTE = list(range(10))
_ROOM_COUNT_RANGE = (2, 4)
_GRID_SIDE_RANGE = (18, 30)
_ROOM_OUTER_H_RANGE = (5, 9)
_ROOM_OUTER_W_RANGE = (5, 10)
_GEN_ATTEMPTS = 400
_PLACE_ATTEMPTS = 120


def _door_start(inner_span: int) -> int:
    return 1 + (inner_span - 2) // 2


def _build_room_patch(
    outer_h: int,
    outer_w: int,
    wall_color: int,
    door_color: int,
    stripe_color: int,
    orientation: str,
    stripe_phase: int,
    horizontal_side: str,
    vertical_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((outer_h, outer_w), _BG, dtype=int)
    patch[0, :] = wall_color
    patch[-1, :] = wall_color
    patch[:, 0] = wall_color
    patch[:, -1] = wall_color

    inner_h = outer_h - 2
    inner_w = outer_w - 2
    door_r = _door_start(inner_h)
    door_c = _door_start(inner_w)

    if horizontal_side == "top":
        patch[0, door_c : door_c + 2] = door_color
    else:
        patch[-1, door_c : door_c + 2] = door_color

    if vertical_side == "left":
        patch[door_r : door_r + 2, 0] = door_color
    else:
        patch[door_r : door_r + 2, -1] = door_color

    if orientation == "horizontal":
        for r in range(stripe_phase, inner_h, 2):
            patch[1 + r, 1:-1] = stripe_color
    else:
        for c in range(stripe_phase, inner_w, 2):
            patch[1:-1, 1 + c] = stripe_color

    opened = patch.copy()
    if orientation == "horizontal":
        if vertical_side == "left":
            opened[door_r : door_r + 2, 0] = _BG
        else:
            opened[door_r : door_r + 2, -1] = _BG
    else:
        if horizontal_side == "top":
            opened[0, door_c : door_c + 2] = _BG
        else:
            opened[-1, door_c : door_c + 2] = _BG

    return patch, opened


def _valid_positions(occupied: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer_h):
        for c0 in range(1, w - outer_w):
            r1 = r0 + outer_h
            c1 = c0 + outer_w
            if np.any(occupied[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1]):
                continue
            positions.append((r0, c0))
    return positions


def _orientation_mix(room_count: int) -> list[str]:
    extra_horizontal = random.randint(0, room_count % 2)
    horizontal_count = room_count // 2 + extra_horizontal
    vertical_count = room_count - horizontal_count
    orientations = ["horizontal"] * horizontal_count + ["vertical"] * vertical_count
    random.shuffle(orientations)
    return orientations


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIDE_RANGE)
    w = random.randint(*_GRID_SIDE_RANGE)
    room_count = random.randint(*_ROOM_COUNT_RANGE)

    usable_colors = random.sample(_PALETTE[1:], room_count + 2)
    wall_color = usable_colors[0]
    door_color = usable_colors[1]
    stripe_colors = usable_colors[2:]

    grid = np.full((h, w), _BG, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((h, w), dtype=bool)
    orientations = _orientation_mix(room_count)

    for room_index in range(room_count):
        placed = False
        for _ in range(_PLACE_ATTEMPTS):
            outer_h = random.randint(*_ROOM_OUTER_H_RANGE)
            outer_w = random.randint(*_ROOM_OUTER_W_RANGE)
            if outer_h >= h - 1 or outer_w >= w - 1:
                continue

            positions = _valid_positions(occupied, outer_h, outer_w)
            if not positions:
                return None

            r0, c0 = random.choice(positions)
            patch, opened = _build_room_patch(
                outer_h=outer_h,
                outer_w=outer_w,
                wall_color=wall_color,
                door_color=door_color,
                stripe_color=stripe_colors[room_index],
                orientation=orientations[room_index],
                stripe_phase=random.randint(0, 1),
                horizontal_side=random.choice(["top", "bottom"]),
                vertical_side=random.choice(["left", "right"]),
            )

            r1 = r0 + outer_h
            c1 = c0 + outer_w
            grid[r0:r1, c0:c1] = patch
            expected[r0:r1, c0:c1] = opened
            occupied[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    if _background_color(grid) != _BG:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle input")
