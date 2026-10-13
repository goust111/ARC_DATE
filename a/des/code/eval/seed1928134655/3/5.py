import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 9
MIRROR_COLOR = 1
COLOR_CYCLE = (2, 3, 4, 7)  # red -> green -> yellow -> orange
GEN_ATTEMPTS = 400
ROOM_SIZE_RANGE = (6, 9)
MIRROR_COUNT_OPTIONS = (1, 2, 3, 4)
SHAPE_CELL_RANGE = (3, 6)
SHAPE_SPAN_MAX = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bands_between_lines(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bands = []
    prev = -1
    for line in sorted(int(x) for x in lines):
        if prev + 1 <= line - 1:
            bands.append((prev + 1, line - 1))
        prev = line
    if prev + 1 <= limit - 1:
        bands.append((prev + 1, limit - 1))
    return bands


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        full_rows = np.count_nonzero(np.all(grid == color, axis=1))
        full_cols = np.count_nonzero(np.all(grid == color, axis=0))
        if full_rows >= 2 and full_cols >= 2:
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one wall color"
    return candidates[0]


def _extract_rooms(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if np.all(grid[r, :] == wall)]
    wall_cols = [c for c in range(w) if np.all(grid[:, c] == wall)]
    row_bands = _bands_between_lines(wall_rows, h)
    col_bands = _bands_between_lines(wall_cols, w)
    assert len(row_bands) == 1, "expected one horizontal corridor band"
    assert len(col_bands) >= 3, "expected source, mirror corridor(s), and target"
    r0, r1 = row_bands[0]
    return [(r0, c0, r1, c1) for c0, c1 in col_bands]


def _subgrid(grid: np.ndarray, room: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = room
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    assert coords.size > 0, "shape must contain at least one cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = True
    return out


def _extract_source_shape(grid: np.ndarray, room: tuple[int, int, int, int], bg: int) -> tuple[np.ndarray, int]:
    sub = _subgrid(grid, room)
    coords = np.argwhere(sub != bg)
    assert coords.size > 0, "source room must contain a shape"
    colors = np.unique(sub[sub != bg])
    assert colors.size == 1, "source room must contain one monochrome shape"
    return _coords_to_mask(coords), int(colors[0])


def _mirror_orientation(grid: np.ndarray, room: tuple[int, int, int, int], bg: int) -> str:
    sub = _subgrid(grid, room)
    coords = np.argwhere(sub != bg)
    assert coords.shape[0] >= 2, "mirror room must contain a diagonal marker"
    colors = np.unique(sub[sub != bg])
    assert colors.size == 1, "mirror room must contain a single diagonal marker"
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    score = float(np.sum((rows - rows.mean()) * (cols - cols.mean())))
    assert score != 0.0, "diagonal marker must have a clear orientation"
    return "\\" if score > 0 else "/"


def _reflect_main_diagonal(mask: np.ndarray) -> np.ndarray:
    return mask.T.copy()


def _reflect_anti_diagonal(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask.T, 2).copy()


def _advance_color(color: int, steps: int) -> int:
    assert color in COLOR_CYCLE, "source color must belong to the visible cycle"
    idx = COLOR_CYCLE.index(int(color))
    return int(COLOR_CYCLE[(idx + steps) % len(COLOR_CYCLE)])


def _paste_centered(
    out: np.ndarray,
    room: tuple[int, int, int, int],
    mask: np.ndarray,
    color: int,
) -> None:
    r0, c0, r1, c1 = room
    room_h = int(r1 - r0 + 1)
    room_w = int(c1 - c0 + 1)
    shape_h, shape_w = mask.shape
    assert shape_h <= room_h and shape_w <= room_w, "target room too small for reflected shape"
    top = r0 + (room_h - shape_h) // 2
    left = c0 + (room_w - shape_w) // 2
    coords = np.argwhere(mask)
    for rr, cc in coords:
        out[top + int(rr), left + int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pass a source shape through a linear sequence of mirror rooms.

    Input structure:
    - A single horizontal corridor is split into rectangular rooms by a wall color.
    - The first room contains a small monochrome shape.
    - Each middle room contains one diagonal mirror marker: "\" or "/".
    - The last room is empty.

    Rule:
    - Start from the source shape's normalized bitmap.
    - For each middle room, reflect the bitmap across the room's diagonal.
    - Advance the shape color one step along red -> green -> yellow -> orange -> red
      for each mirror room, starting from the source shape color.
    - Draw the final transformed shape centered in the last room.
    - Keep the source room and mirror markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    wall = _detect_wall_color(g, bg)
    rooms = _extract_rooms(g, wall)

    shape, source_color = _extract_source_shape(g, rooms[0], bg)

    transformed = shape
    for room in rooms[1:-1]:
        if _mirror_orientation(g, room, bg) == "\\":
            transformed = _reflect_main_diagonal(transformed)
        else:
            transformed = _reflect_anti_diagonal(transformed)

    out = g.copy()
    target = _subgrid(g, rooms[-1])
    assert np.all(target == bg), "target room must be empty in the input"
    final_color = _advance_color(source_color, len(rooms) - 2)
    _paste_centered(out, rooms[-1], transformed, final_color)
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _random_connected_shape(room_size: int) -> np.ndarray:
    max_span = min(SHAPE_SPAN_MAX, room_size - 2)
    for _ in range(200):
        n_cells = random.randint(*SHAPE_CELL_RANGE)
        box_h = random.randint(2, max_span)
        box_w = random.randint(2, max_span)
        if box_h * box_w < n_cells:
            continue
        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < n_cells:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != n_cells:
            continue
        mask = _cells_to_mask(cells)
        if mask.shape[0] > room_size - 2 or mask.shape[1] > room_size - 2:
            continue
        refl_main = _reflect_main_diagonal(mask)
        refl_anti = _reflect_anti_diagonal(mask)
        if np.array_equal(mask, refl_main) and np.array_equal(mask, refl_anti):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate connected shape")


def _room_left(room_index: int, room_size: int) -> int:
    return room_index * (room_size + 1)


def _draw_layout(room_size: int, room_count: int) -> np.ndarray:
    height = room_size + 2
    width = room_count * room_size + room_count + 1
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[0, :] = WALL_COLOR
    grid[-1, :] = WALL_COLOR
    for room_index in range(room_count + 1):
        grid[:, _room_left(room_index, room_size)] = WALL_COLOR
    return grid


def _place_source_shape(
    grid: np.ndarray,
    room_size: int,
    mask: np.ndarray,
    color: int,
) -> None:
    shape_h, shape_w = mask.shape
    centered_top = (room_size - shape_h) // 2
    centered_left = (room_size - shape_w) // 2
    top_choices = [t for t in range(1, room_size - shape_h) if t != centered_top]
    left_choices = [l for l in range(1, room_size - shape_w) if l != centered_left]
    if not top_choices:
        top_choices = list(range(1, room_size - shape_h))
    if not left_choices:
        left_choices = list(range(1, room_size - shape_w))
    top = random.choice(top_choices)
    left = random.choice(left_choices)
    base_r = 1 + top
    base_c = 1 + left
    for rr, cc in np.argwhere(mask):
        grid[base_r + int(rr), base_c + int(cc)] = int(color)


def _draw_mirror(grid: np.ndarray, room_index: int, room_size: int, orient: str) -> None:
    left_wall = _room_left(room_index, room_size)
    if orient == "\\":
        for d in range(room_size):
            grid[1 + d, left_wall + 1 + d] = MIRROR_COLOR
    else:
        for d in range(room_size):
            grid[1 + d, left_wall + room_size - d] = MIRROR_COLOR


def _generate_candidate() -> np.ndarray:
    room_size = random.randint(*ROOM_SIZE_RANGE)
    mirror_count = random.choice(MIRROR_COUNT_OPTIONS)
    room_count = mirror_count + 2
    mirrors = [random.choice(("\\", "/")) for _ in range(mirror_count)]
    source_color = random.choice(COLOR_CYCLE)
    shape = _random_connected_shape(room_size)

    transformed = shape
    for orient in mirrors:
        if orient == "\\":
            transformed = _reflect_main_diagonal(transformed)
        else:
            transformed = _reflect_anti_diagonal(transformed)
    if transformed.shape[0] > room_size or transformed.shape[1] > room_size:
        return None

    grid = _draw_layout(room_size, room_count)
    _place_source_shape(grid, room_size, shape, source_color)
    for index, orient in enumerate(mirrors, start=1):
        _draw_mirror(grid, index, room_size, orient)

    if _most_common_color(grid) != BG_COLOR:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    target_room = _extract_rooms(grid, WALL_COLOR)[-1]
    r0, c0, r1, c1 = target_room
    target = out[r0 : r1 + 1, c0 : c1 + 1]
    final_color = _advance_color(source_color, mirror_count)
    if np.count_nonzero(target == final_color) != np.count_nonzero(transformed):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-corridor input")
