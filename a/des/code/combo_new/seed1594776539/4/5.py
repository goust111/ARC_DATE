import random

import numpy as np


BG = 0
SHAPE = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _find_separator_columns(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    separators: list[tuple[int, int, int]] = []
    for c in range(w):
        column = grid[:, c]
        door_rows = np.flatnonzero(column == BG)
        if door_rows.size != 1:
            continue
        non_bg = column[column != BG]
        if non_bg.size != h - 1:
            continue
        colors = {int(value) for value in non_bg}
        if len(colors) != 1 or SHAPE in colors:
            continue
        separators.append((c, colors.pop(), int(door_rows[0])))
    return separators


def _find_separator_rows(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    separators: list[tuple[int, int, int]] = []
    for r in range(h):
        row = grid[r, :]
        door_cols = np.flatnonzero(row == BG)
        if door_cols.size != 1:
            continue
        non_bg = row[row != BG]
        if non_bg.size != w - 1:
            continue
        colors = {int(value) for value in non_bg}
        if len(colors) != 1 or SHAPE in colors:
            continue
        separators.append((r, colors.pop(), int(door_cols[0])))
    return separators


def _room_spans(boundaries: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        spans.append((start, boundary - 1))
        start = boundary + 1
    spans.append((start, limit - 1))
    return spans


def _solve_vertical(grid: np.ndarray, separators: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = grid.shape
    sep_cols = [c for c, _color, _door in separators]
    rooms = _room_spans(sep_cols, w)
    room_has_shape = []
    room_coords: list[np.ndarray] = []
    for c0, c1 in rooms:
        coords = np.argwhere(grid[:, c0 : c1 + 1] == SHAPE)
        if coords.size == 0:
            room_has_shape.append(False)
            room_coords.append(np.empty((0, 2), dtype=int))
            continue
        coords = coords.copy()
        coords[:, 1] += c0
        room_has_shape.append(True)
        room_coords.append(coords)

    moves: list[tuple[np.ndarray, np.ndarray, int]] = []
    for room_index, has_shape in enumerate(room_has_shape):
        if not has_shape:
            continue
        options: list[tuple[str, int, int]] = []
        if room_index > 0 and not room_has_shape[room_index - 1]:
            sep_index = room_index - 1
            sep_col, color, _door_row = separators[sep_index]
            options.append(("left", sep_col, color))
        if room_index + 1 < len(rooms) and not room_has_shape[room_index + 1]:
            sep_index = room_index
            sep_col, color, _door_row = separators[sep_index]
            options.append(("right", sep_col, color))
        assert len(options) == 1, "each occupied room must have one unique empty neighbor"

        direction, sep_col, color = options[0]
        src = room_coords[room_index]
        dst = src.copy()
        if direction == "left":
            shift = (sep_col - 1) - int(src[:, 1].max())
            dst[:, 1] += shift
            target_room = rooms[room_index - 1]
        else:
            shift = (sep_col + 1) - int(src[:, 1].min())
            dst[:, 1] += shift
            target_room = rooms[room_index + 1]
        c0, c1 = target_room
        assert int(dst[:, 1].min()) >= c0 and int(dst[:, 1].max()) <= c1
        moves.append((src, dst, color))

    out = grid.copy()
    out[out == SHAPE] = BG
    for _src, dst, color in moves:
        out[dst[:, 0], dst[:, 1]] = int(color)
    return out


def _solve_horizontal(grid: np.ndarray, separators: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = grid.shape
    sep_rows = [r for r, _color, _door in separators]
    rooms = _room_spans(sep_rows, h)
    room_has_shape = []
    room_coords: list[np.ndarray] = []
    for r0, r1 in rooms:
        coords = np.argwhere(grid[r0 : r1 + 1, :] == SHAPE)
        if coords.size == 0:
            room_has_shape.append(False)
            room_coords.append(np.empty((0, 2), dtype=int))
            continue
        coords = coords.copy()
        coords[:, 0] += r0
        room_has_shape.append(True)
        room_coords.append(coords)

    moves: list[tuple[np.ndarray, np.ndarray, int]] = []
    for room_index, has_shape in enumerate(room_has_shape):
        if not has_shape:
            continue
        options: list[tuple[str, int, int]] = []
        if room_index > 0 and not room_has_shape[room_index - 1]:
            sep_index = room_index - 1
            sep_row, color, _door_col = separators[sep_index]
            options.append(("up", sep_row, color))
        if room_index + 1 < len(rooms) and not room_has_shape[room_index + 1]:
            sep_index = room_index
            sep_row, color, _door_col = separators[sep_index]
            options.append(("down", sep_row, color))
        assert len(options) == 1, "each occupied room must have one unique empty neighbor"

        direction, sep_row, color = options[0]
        src = room_coords[room_index]
        dst = src.copy()
        if direction == "up":
            shift = (sep_row - 1) - int(src[:, 0].max())
            dst[:, 0] += shift
            target_room = rooms[room_index - 1]
        else:
            shift = (sep_row + 1) - int(src[:, 0].min())
            dst[:, 0] += shift
            target_room = rooms[room_index + 1]
        r0, r1 = target_room
        assert int(dst[:, 0].min()) >= r0 and int(dst[:, 0].max()) <= r1
        moves.append((src, dst, color))

    out = grid.copy()
    out[out == SHAPE] = BG
    for _src, dst, color in moves:
        out[dst[:, 0], dst[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each grey room shape through its doorway into the neighboring empty room and recolor it with the wall color.

    Input structure:
    - The background is black.
    - One or more full colored separator rows or columns split the canvas into rectangular rooms.
    - Each separator line is one pixel thick, uses a single non-grey color, and has exactly one black doorway pixel.
    - Some rooms contain one connected grey shape that stays away from the separator lines, while other rooms are empty.

    Rule:
    - Detect the separator rows or columns and treat the spaces between them as rooms.
    - For each room containing grey pixels, find the unique adjacent room that is empty; the shared separator doorway determines the push direction.
    - Remove the grey shape, keep its coordinates along the doorway-parallel axis unchanged, and shift it across the doorway until the side nearest the doorway is flush with the far side.
    - Repaint the moved shape with the color of that separator line, leaving all separator pixels and doorway gaps unchanged.

    Key insight:
    - The lone black gap on a colored separator is the door, and the separator that borders the unique empty neighboring room determines both the motion direction and the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_cols = _find_separator_columns(g)
    sep_rows = _find_separator_rows(g)
    assert sep_cols or sep_rows, "expected at least one separator line"

    if sep_cols and not sep_rows:
        return _solve_vertical(g, sep_cols)
    if sep_rows and not sep_cols:
        return _solve_horizontal(g, sep_rows)
    raise AssertionError("expected separator lines along exactly one axis")


def _shape_targets(occupied: tuple[bool, ...]) -> dict[int, int] | None:
    targets: dict[int, int] = {}
    for i, has_shape in enumerate(occupied):
        if not has_shape:
            continue
        neighbors = []
        if i > 0 and not occupied[i - 1]:
            neighbors.append(i - 1)
        if i + 1 < len(occupied) and not occupied[i + 1]:
            neighbors.append(i + 1)
        if len(neighbors) != 1:
            return None
        target = neighbors[0]
        if target in targets.values():
            return None
        targets[i] = target
    return targets if targets else None


def _valid_patterns(room_count: int) -> list[tuple[bool, ...]]:
    patterns: list[tuple[bool, ...]] = []
    for mask in range(1, 1 << room_count):
        occupied = tuple(bool((mask >> i) & 1) for i in range(room_count))
        if _shape_targets(occupied) is not None:
            patterns.append(occupied)
    return patterns


def _sample_connected_mask(max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    max_area = h * w
    min_area = 2 if max_area > 1 else 1
    area = random.randint(min_area, max_area)

    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return mask


def _room_column_spans(room_count: int, room_w: int) -> list[tuple[int, int]]:
    return [(i * (room_w + 1), i * (room_w + 1) + room_w - 1) for i in range(room_count)]


def _room_row_spans(room_count: int, room_h: int) -> list[tuple[int, int]]:
    return [(i * (room_h + 1), i * (room_h + 1) + room_h - 1) for i in range(room_count)]


def _build_vertical_attempt() -> np.ndarray:
    room_count = random.randint(4, 6)
    max_room_w = min(6, (30 - (room_count - 1)) // room_count)
    room_w = random.randint(4, max_room_w)
    height = random.randint(7, 18)
    width = room_count * room_w + (room_count - 1)

    grid = np.full((height, width), BG, dtype=int)
    room_spans = _room_column_spans(room_count, room_w)
    separators = [span[1] + 1 for span in room_spans[:-1]]
    line_colors = random.sample(LINE_COLORS, len(separators))

    for sep_col, color in zip(separators, line_colors):
        grid[:, sep_col] = int(color)

    occupied = random.choice(_valid_patterns(room_count))
    if random.random() < 0.5:
        occupied = tuple(reversed(occupied))
    targets = _shape_targets(occupied)
    assert targets is not None

    used_doors: dict[int, int] = {}
    for room_index, has_shape in enumerate(occupied):
        if not has_shape:
            continue
        c0, c1 = room_spans[room_index]
        mask = _sample_connected_mask(min(height - 2, 6), min(room_w - 2, 4))
        sh, sw = mask.shape
        top = random.randint(1, height - sh - 1)
        left = random.randint(c0 + 1, c1 - sw)
        coords = np.argwhere(mask == 1)
        for r, c in coords:
            grid[top + int(r), left + int(c)] = SHAPE

        target_room = targets[room_index]
        sep_index = min(room_index, target_room)
        used_doors[sep_index] = random.choice([top + int(r) for r, _c in coords])

    for sep_index, sep_col in enumerate(separators):
        door_row = used_doors.get(sep_index, random.randint(1, height - 2))
        grid[door_row, sep_col] = BG

    return grid


def _build_horizontal_attempt() -> np.ndarray:
    room_count = random.randint(4, 6)
    max_room_h = min(6, (30 - (room_count - 1)) // room_count)
    room_h = random.randint(4, max_room_h)
    width = random.randint(7, 18)
    height = room_count * room_h + (room_count - 1)

    grid = np.full((height, width), BG, dtype=int)
    room_spans = _room_row_spans(room_count, room_h)
    separators = [span[1] + 1 for span in room_spans[:-1]]
    line_colors = random.sample(LINE_COLORS, len(separators))

    for sep_row, color in zip(separators, line_colors):
        grid[sep_row, :] = int(color)

    occupied = random.choice(_valid_patterns(room_count))
    if random.random() < 0.5:
        occupied = tuple(reversed(occupied))
    targets = _shape_targets(occupied)
    assert targets is not None

    used_doors: dict[int, int] = {}
    for room_index, has_shape in enumerate(occupied):
        if not has_shape:
            continue
        r0, r1 = room_spans[room_index]
        mask = _sample_connected_mask(min(room_h - 2, 4), min(width - 2, 6))
        sh, sw = mask.shape
        top = random.randint(r0 + 1, r1 - sh)
        left = random.randint(1, width - sw - 1)
        coords = np.argwhere(mask == 1)
        for r, c in coords:
            grid[top + int(r), left + int(c)] = SHAPE

        target_room = targets[room_index]
        sep_index = min(room_index, target_room)
        used_doors[sep_index] = random.choice([left + int(c) for _r, c in coords])

    for sep_index, sep_row in enumerate(separators):
        door_col = used_doors.get(sep_index, random.randint(1, width - 2))
        grid[sep_row, door_col] = BG

    return grid


def _gen_attempt() -> np.ndarray:
    if random.random() < 0.5:
        return _build_vertical_attempt()
    return _build_horizontal_attempt()


def _is_valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if np.array_equal(out, grid):
        return False
    if np.any(out == SHAPE):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if _is_valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate valid doorway-room input")
