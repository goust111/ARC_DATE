import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _wall_lines(grid):
    wall_rows = [i for i in range(grid.shape[0]) if np.all(grid[i, :] != BLACK)]
    wall_cols = [j for j in range(grid.shape[1]) if np.all(grid[:, j] != BLACK)]
    return wall_rows, wall_cols


def _room_of_cell(r, c, wall_rows, wall_cols):
    ri = np.searchsorted(wall_rows, r) - 1
    ci = np.searchsorted(wall_cols, c) - 1
    if ri < 0 or ci < 0 or ri >= len(wall_rows) - 1 or ci >= len(wall_cols) - 1:
        return None
    if r in wall_rows or c in wall_cols:
        return None
    return int(ri), int(ci)


def _room_bounds(room, wall_rows, wall_cols):
    ri, ci = room
    top = wall_rows[ri] + 1
    bottom = wall_rows[ri + 1]
    left = wall_cols[ci] + 1
    right = wall_cols[ci + 1]
    return top, bottom, left, right


def _adjacent_rooms_for_window(r, c, wall_rows, wall_cols):
    on_row = r in wall_rows
    on_col = c in wall_cols
    if on_row == on_col:
        return []
    if on_row:
        k = wall_rows.index(r)
        ci = np.searchsorted(wall_cols, c) - 1
        if 0 < k < len(wall_rows) - 1 and 0 <= ci < len(wall_cols) - 1:
            return [(k - 1, int(ci)), (k, int(ci))]
        return []
    k = wall_cols.index(c)
    ri = np.searchsorted(wall_rows, r) - 1
    if 0 < k < len(wall_cols) - 1 and 0 <= ri < len(wall_rows) - 1:
        return [(int(ri), k - 1), (int(ri), k)]
    return []


def _room_has_color(grid, color, room, wall_rows, wall_cols):
    top, bottom, left, right = _room_bounds(room, wall_rows, wall_cols)
    return bool(np.any(grid[top:bottom, left:right] == color))


def main(grid: np.ndarray) -> np.ndarray:
    """Copy colored room sprites through matching colored wall windows.

    Input structure:
    The grid has black room interiors and one-pixel grey walls forming a
    rectangular array of rooms. Some rooms contain a small uniquely colored
    sprite. Some shared wall segments contain one non-black, non-grey window
    pixel whose color matches one of the sprites.

    Rule:
    Each window is restored to grey. For that window color, find the matching
    source sprite inside the rooms, keep all of its within-room offsets, and
    copy those colored pixels into each room adjacent to the window that does
    not already contain that source color. Existing pixels in the target room
    are overwritten only where the copied sprite has pixels.

    Key insight:
    A colored wall pixel is an instruction, not decoration: its color selects
    the sprite to teleport, while the wall location selects the destination
    room or rooms and the sprite keeps its relative room position.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    wall_rows, wall_cols = _wall_lines(grid)
    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)

    windows = []
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        if r in wall_row_set or c in wall_col_set:
            windows.append((int(r), int(c), int(grid[r, c])))
            out[r, c] = GREY

    source_by_color = {}
    for color in sorted({color for _, _, color in windows}):
        coords = []
        for r, c in np.argwhere(grid == color):
            if r not in wall_row_set and c not in wall_col_set:
                room = _room_of_cell(int(r), int(c), wall_rows, wall_cols)
                if room is not None:
                    coords.append((int(r), int(c), room))
        if not coords:
            continue
        source_room = coords[0][2]
        top, _, left, _ = _room_bounds(source_room, wall_rows, wall_cols)
        offsets = [(r - top, c - left) for r, c, room in coords if room == source_room]
        source_by_color[color] = (source_room, offsets)

    for r, c, color in windows:
        if color not in source_by_color:
            continue
        _, offsets = source_by_color[color]
        for room in _adjacent_rooms_for_window(r, c, wall_rows, wall_cols):
            if _room_has_color(grid, color, room, wall_rows, wall_cols):
                continue
            top, bottom, left, right = _room_bounds(room, wall_rows, wall_cols)
            for dr, dc in offsets:
                rr = top + dr
                cc = left + dc
                if top <= rr < bottom and left <= cc < right:
                    out[rr, cc] = color
    return out


def _shape_that_fits(room_h, room_w):
    templates = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ]
    fits = []
    for shape in templates:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        if max_r < room_h and max_c < room_w:
            fits.append(shape)
    return random.choice(fits)


def _neighbors(room, n_room_rows, n_room_cols):
    ri, ci = room
    result = []
    if ri > 0:
        result.append((ri - 1, ci))
    if ri + 1 < n_room_rows:
        result.append((ri + 1, ci))
    if ci > 0:
        result.append((ri, ci - 1))
    if ci + 1 < n_room_cols:
        result.append((ri, ci + 1))
    return result


def _all_adjacent_pairs(n_room_rows, n_room_cols):
    pairs = []
    for ri in range(n_room_rows):
        for ci in range(n_room_cols):
            if ri + 1 < n_room_rows:
                pairs.append(((ri, ci), (ri + 1, ci)))
            if ci + 1 < n_room_cols:
                pairs.append(((ri, ci), (ri, ci + 1)))
    return pairs


def _pair_key(a, b):
    return tuple(sorted((a, b)))


def _put_window(grid, color, pair, wall_rows, wall_cols):
    a, b = pair
    if a[0] == b[0]:
        row_room = a[0]
        wall_col = wall_cols[max(a[1], b[1])]
        r = random.randint(wall_rows[row_room] + 1, wall_rows[row_room + 1] - 1)
        grid[r, wall_col] = color
    else:
        col_room = a[1]
        wall_row = wall_rows[max(a[0], b[0])]
        c = random.randint(wall_cols[col_room] + 1, wall_cols[col_room + 1] - 1)
        grid[wall_row, c] = color


def generate_input() -> np.ndarray:
    n_room_rows = random.choice([2, 3, 4])
    n_room_cols = random.choice([2, 3, 4])
    max_h = (30 - (n_room_rows + 1)) // n_room_rows
    max_w = (30 - (n_room_cols + 1)) // n_room_cols
    room_h = random.randint(4, min(6, max_h))
    room_w = random.randint(4, min(6, max_w))

    height = n_room_rows * room_h + n_room_rows + 1
    width = n_room_cols * room_w + n_room_cols + 1
    grid = np.zeros((height, width), dtype=np.int8)

    wall_rows = [i * (room_h + 1) for i in range(n_room_rows + 1)]
    wall_cols = [j * (room_w + 1) for j in range(n_room_cols + 1)]
    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY

    rooms = [(ri, ci) for ri in range(n_room_rows) for ci in range(n_room_cols)]
    random.shuffle(rooms)
    sprite_count = random.randint(2, min(5, len(rooms), len(COLORS)))
    sprite_rooms = rooms[:sprite_count]
    colors = random.sample(COLORS, sprite_count)
    color_room = dict(zip(colors, sprite_rooms))

    for color, room in color_room.items():
        top, bottom, left, right = _room_bounds(room, wall_rows, wall_cols)
        shape = _shape_that_fits(room_h, room_w)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        anchor_r = random.randint(top, bottom - max_r - 1)
        anchor_c = random.randint(left, right - max_c - 1)
        for dr, dc in shape:
            grid[anchor_r + dr, anchor_c + dc] = color

    occupied_rooms = set(sprite_rooms)
    adjacent_pairs = _all_adjacent_pairs(n_room_rows, n_room_cols)
    used_pairs = set()
    window_goal = random.randint(1, min(sprite_count, 4))
    window_colors = colors[:]
    random.shuffle(window_colors)

    for color in window_colors:
        if len(used_pairs) >= len(adjacent_pairs) or window_goal <= 0:
            break
        source = color_room[color]
        prefer_occupied = random.random() < 0.5
        remote_mode = random.random() < 0.5
        pair = None

        if not remote_mode:
            candidates = [
                _pair_key(source, other)
                for other in _neighbors(source, n_room_rows, n_room_cols)
                if _pair_key(source, other) not in used_pairs
            ]
            preferred = [
                p for p in candidates
                if ((p[0] in occupied_rooms or p[1] in occupied_rooms) and prefer_occupied)
                or ((p[0] not in occupied_rooms and p[1] not in occupied_rooms) and not prefer_occupied)
            ]
            if preferred:
                candidates = preferred
            if candidates:
                pair = random.choice(candidates)

        if pair is None:
            candidates = [
                _pair_key(a, b)
                for a, b in adjacent_pairs
                if source not in (a, b) and _pair_key(a, b) not in used_pairs
            ]
            preferred = [
                p for p in candidates
                if ((p[0] in occupied_rooms or p[1] in occupied_rooms) and prefer_occupied)
                or ((p[0] not in occupied_rooms and p[1] not in occupied_rooms) and not prefer_occupied)
            ]
            if preferred:
                candidates = preferred
            if candidates:
                pair = random.choice(candidates)

        if pair is None:
            candidates = [
                _pair_key(source, other)
                for other in _neighbors(source, n_room_rows, n_room_cols)
                if _pair_key(source, other) not in used_pairs
            ]
            if candidates:
                pair = random.choice(candidates)

        if pair is not None:
            _put_window(grid, color, pair, wall_rows, wall_cols)
            used_pairs.add(pair)
            window_goal -= 1

    if not used_pairs:
        color = random.choice(colors)
        source = color_room[color]
        other = random.choice(_neighbors(source, n_room_rows, n_room_cols))
        pair = _pair_key(source, other)
        _put_window(grid, color, pair, wall_rows, wall_cols)

    return grid
