import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

LAMP_COLORS = (RED, BLUE, GREEN, ORANGE, YELLOW, MAGENTA)


def _segments(size, separator_indices):
    separators = sorted(int(x) for x in separator_indices)
    out = []
    start = 0
    for idx in separators:
        if start < idx:
            out.append((start, idx))
        start = idx + 1
    if start < size:
        out.append((start, size))
    return out


def _light_room(room):
    lit = room.copy()
    lamp_mask = (room != BLACK) & (room != GREY) & (room != AZURE)
    lamps = np.argwhere(lamp_mask)
    if len(lamps) == 0:
        return lit, 0

    r, c = lamps[0]
    color = int(room[r, c])
    seen = np.zeros(room.shape, dtype=bool)
    seen[r, c] = True

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = int(r) + dr, int(c) + dc
        while 0 <= rr < room.shape[0] and 0 <= cc < room.shape[1]:
            if room[rr, cc] in (BLACK, AZURE):
                break
            seen[rr, cc] = True
            if lit[rr, cc] == GREY:
                lit[rr, cc] = color
            rr += dr
            cc += dc

    return lit, int(seen.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Illuminate each separated room, then reorder rooms by illuminated area.

    Input structure: The grid is divided by single-pixel azure separator rows and
    columns into equal square grey rooms. Each room contains exactly one
    non-grey, non-black, non-azure lamp pixel and several black wall pixels.
    Rule: In every room, the lamp lights grey cells in the four orthogonal
    directions until a black wall or the room boundary is reached; the lamp cell
    itself counts as lit. The illuminated room states are then sorted by lit-cell
    count from largest to smallest and placed back into the original room slots
    in row-major order while all azure separators stay fixed.
    Key insight: The visible cross-shaped light spread is only the first layer;
    the room order is determined by counting the lit cells after wall blocking.
    """
    arr = np.asarray(grid)
    sep_rows = np.where(np.all(arr == AZURE, axis=1))[0]
    sep_cols = np.where(np.all(arr == AZURE, axis=0))[0]
    row_segments = _segments(arr.shape[0], sep_rows)
    col_segments = _segments(arr.shape[1], sep_cols)

    rooms = []
    original_index = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            room, count = _light_room(arr[r0:r1, c0:c1])
            rooms.append((count, original_index, room))
            original_index += 1

    rooms.sort(key=lambda item: (-item[0], item[1]))

    out = arr.copy()
    k = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            out[r0:r1, c0:c1] = rooms[k][2]
            k += 1
    return out


def _blank_grid(rows, cols, size, bordered):
    h = rows * size + (rows - 1) + (2 if bordered else 0)
    w = cols * size + (cols - 1) + (2 if bordered else 0)
    grid = np.full((h, w), GREY, dtype=int)

    if bordered:
        grid[0, :] = AZURE
        grid[-1, :] = AZURE
        grid[:, 0] = AZURE
        grid[:, -1] = AZURE

    offset = 1 if bordered else 0
    for rr in range(rows - 1):
        line = offset + size + rr * (size + 1)
        grid[line, :] = AZURE
    for cc in range(cols - 1):
        line = offset + size + cc * (size + 1)
        grid[:, line] = AZURE

    return grid


def _room_origin(row, col, size, bordered):
    offset = 1 if bordered else 0
    return offset + row * (size + 1), offset + col * (size + 1)


def _random_room(size, color, rng):
    room = np.full((size, size), GREY, dtype=int)
    lamp_r = int(rng.integers(0, size))
    lamp_c = int(rng.integers(0, size))
    room[lamp_r, lamp_c] = color

    walls = set()
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for dr, dc in directions:
        path = []
        rr, cc = lamp_r + dr, lamp_c + dc
        while 0 <= rr < size and 0 <= cc < size:
            path.append((rr, cc))
            rr += dr
            cc += dc
        if path and rng.random() < 0.5:
            walls.add(path[int(rng.integers(0, len(path)))])

    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) != (lamp_r, lamp_c) and (r, c) not in walls
    ]
    rng.shuffle(candidates)
    extra_max = min(len(candidates), max(1, size - 2))
    extra_count = int(rng.integers(0, extra_max + 1))
    for pos in candidates[:extra_count]:
        if rng.random() < 0.65:
            walls.add(pos)

    if not walls:
        candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) != (lamp_r, lamp_c)
        ]
        walls.add(candidates[int(rng.integers(0, len(candidates)))])

    for r, c in walls:
        room[r, c] = BLACK
    return room


def _layout_candidates(size, bordered):
    candidates = []
    for rows in range(1, 7):
        for cols in range(1, 7):
            total = rows * cols
            if not (2 <= total <= len(LAMP_COLORS)):
                continue
            h = rows * size + (rows - 1) + (2 if bordered else 0)
            w = cols * size + (cols - 1) + (2 if bordered else 0)
            if h <= 30 and w <= 30:
                candidates.append((rows, cols))
    return candidates


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        size = int(rng.integers(4, 9))
        bordered = bool(rng.integers(0, 2))
        layouts = _layout_candidates(size, bordered)
        rows, cols = layouts[int(rng.integers(0, len(layouts)))]
        room_count = rows * cols

        colors = list(LAMP_COLORS)
        rng.shuffle(colors)
        colors = colors[:room_count]

        rooms = []
        counts = []
        ok = True
        for color in colors:
            for _attempt in range(300):
                room = _random_room(size, color, rng)
                _, count = _light_room(room)
                if count not in counts:
                    rooms.append(room)
                    counts.append(count)
                    break
            else:
                ok = False
                break
        if ok:
            break
    else:
        size = 6
        bordered = True
        rows, cols = 2, 3
        room_count = rows * cols
        colors = list(LAMP_COLORS)
        rooms = []
        counts = []
        presets = ((0, 0), (0, 2), (1, 1), (2, 3), (3, 2), (5, 5))
        for i in range(room_count):
            room = np.full((size, size), GREY, dtype=int)
            r, c = presets[i]
            room[r, c] = colors[i]
            room[(r + 2) % size, c] = BLACK
            room[r, (c + 3) % size] = BLACK
            rooms.append(room)
            counts.append(_light_room(room)[1])

    grid = _blank_grid(rows, cols, size, bordered)
    for idx, room in enumerate(rooms):
        rr = idx // cols
        cc = idx % cols
        r0, c0 = _room_origin(rr, cc, size, bordered)
        grid[r0:r0 + size, c0:c0 + size] = room

    return grid
