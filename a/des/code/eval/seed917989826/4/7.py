import numpy as np


UP, RIGHT, DOWN, LEFT = range(4)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

WALL_COLOR = 5
MAX_GRID_SIDE = 30
MAX_PATTERN_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 300
RNG = np.random.default_rng()

COLOR_POOL = [c for c in range(10) if c != WALL_COLOR]
LAYOUTS = [
    (room_size, n_rows, n_cols)
    for room_size in (5, 7)
    for n_rows in range(2, 5)
    for n_cols in range(2, 5)
    if n_rows * room_size + n_rows + 1 <= MAX_GRID_SIDE
    if n_cols * room_size + n_cols + 1 <= MAX_GRID_SIDE
]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = prev = idx
    runs.append((start, prev + 1))
    return runs


def _find_wall_structure(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    row_thresh = max(2, int(np.ceil(0.6 * w)))
    col_thresh = max(2, int(np.ceil(0.6 * h)))

    best = None
    for color in map(int, np.unique(g)):
        if color == bg:
            continue

        row_hits = [r for r in range(h) if int(np.count_nonzero(g[r] == color)) >= row_thresh]
        col_hits = [c for c in range(w) if int(np.count_nonzero(g[:, c] == color)) >= col_thresh]
        row_bands = _group_runs(row_hits)
        col_bands = _group_runs(col_hits)
        if len(row_bands) < 2 or len(col_bands) < 2:
            continue

        room_heights = [int(row_bands[i + 1][0] - row_bands[i][1]) for i in range(len(row_bands) - 1)]
        room_widths = [int(col_bands[i + 1][0] - col_bands[i][1]) for i in range(len(col_bands) - 1)]
        if not room_heights or not room_widths:
            continue
        if min(room_heights) <= 0 or min(room_widths) <= 0:
            continue
        if len(set(room_heights)) != 1 or len(set(room_widths)) != 1:
            continue

        score = (
            len(row_bands) * len(col_bands),
            int(np.count_nonzero(g == color)),
        )
        if best is None or score > best[0]:
            best = (score, color, row_bands, col_bands)

    assert best is not None, "failed to infer wall color and partition lines"
    return best[1], best[2], best[3]


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall, row_bands, col_bands = _find_wall_structure(g, bg)

    room_heights = [int(row_bands[i + 1][0] - row_bands[i][1]) for i in range(len(row_bands) - 1)]
    room_widths = [int(col_bands[i + 1][0] - col_bands[i][1]) for i in range(len(col_bands) - 1)]
    assert room_heights and room_widths
    assert len(set(room_heights)) == 1 and len(set(room_widths)) == 1
    room_size = room_heights[0]
    assert room_size == room_widths[0], "room interiors must be square to rotate in place"

    rooms = []
    for i in range(len(row_bands) - 1):
        for j in range(len(col_bands) - 1):
            r0 = row_bands[i][1]
            r1 = row_bands[i + 1][0]
            c0 = col_bands[j][1]
            c1 = col_bands[j + 1][0]
            rooms.append(
                {
                    "row": i,
                    "col": j,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "top_wall": row_bands[i][1] - 1,
                    "bottom_wall": row_bands[i + 1][0],
                    "left_wall": col_bands[j][1] - 1,
                    "right_wall": col_bands[j + 1][0],
                }
            )

    return {
        "bg": bg,
        "wall": wall,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "room_size": room_size,
        "rooms": rooms,
    }


def _edge_counts(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    tight = mask[r0 : r1 + 1, c0 : c1 + 1]
    return np.array(
        [
            int(np.count_nonzero(tight[0, :])),
            int(np.count_nonzero(tight[:, -1])),
            int(np.count_nonzero(tight[-1, :])),
            int(np.count_nonzero(tight[:, 0])),
        ],
        dtype=int,
    )


def _feature_direction(room: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(room) if c != bg]
    assert len(colors) == 1, "each room must contain exactly one monochrome pattern"
    counts = _edge_counts(room == colors[0])
    best = int(np.max(counts))
    assert int(np.count_nonzero(counts == best)) == 1, "feature edge must be unique"
    return int(np.argmax(counts))


def _door_direction(grid: np.ndarray, room: dict, wall: int) -> int:
    g = np.asarray(grid, dtype=int)
    segments = {
        UP: g[room["top_wall"], room["c0"] : room["c1"]],
        RIGHT: g[room["r0"] : room["r1"], room["right_wall"]],
        DOWN: g[room["bottom_wall"], room["c0"] : room["c1"]],
        LEFT: g[room["r0"] : room["r1"], room["left_wall"]],
    }
    hits = []
    for direction, segment in segments.items():
        count = int(np.count_nonzero(segment != wall))
        if count:
            hits.append(direction)
            assert count == 1, "a room side can carry at most one door pixel"
    assert len(hits) == 1, "each room must have exactly one door"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room's monochrome pattern so its feature edge faces the room's colored door.

    Input structure:
    - The dominant background is split by full grey separator rows and columns into equal square rooms.
    - Each room interior contains one monochrome asymmetric pattern on the background.
    - Exactly one colored door pixel lies on the room's top/right/bottom/left wall segment.

    Rule:
    - For each room, locate the door side.
    - Find the pattern's feature edge: the bbox side with the highest pixel count.
    - Rotate the entire room interior by 0/90/180/270 degrees so that feature edge points at the door.
    - Keep walls and door pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_structure(g)
    bg = info["bg"]
    wall = info["wall"]

    out = g.copy()
    for room in info["rooms"]:
        room_slice = np.s_[room["r0"] : room["r1"], room["c0"] : room["c1"]]
        interior = g[room_slice]
        current_dir = _feature_direction(interior, bg)
        target_dir = _door_direction(g, room, wall)
        rot_k = (current_dir - target_dir) % 4
        out[room_slice] = np.rot90(interior, rot_k)
    return out


def _pick_color(forbidden: set[int]) -> int:
    options = [c for c in COLOR_POOL if c not in forbidden]
    assert options
    return int(RNG.choice(options))


def _make_down_mask(room_size: int) -> np.ndarray:
    max_dim = room_size - 2
    for _ in range(MAX_PATTERN_ATTEMPTS):
        h = int(RNG.integers(3, max_dim + 1))
        w = int(RNG.integers(3, max_dim + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[-1, :] = True

        stem_col = int(RNG.integers(1, w - 1))
        mask[:, stem_col] = True

        cap_dir = -1 if RNG.random() < 0.5 else 1
        max_cap = stem_col + 1 if cap_dir == -1 else w - stem_col
        cap_len = int(RNG.integers(1, min(3, max_cap) + 1))
        if cap_dir == -1:
            mask[0, stem_col - cap_len + 1 : stem_col + 1] = True
        else:
            mask[0, stem_col : stem_col + cap_len] = True

        nub_count = int(RNG.integers(1, 4))
        for _ in range(nub_count):
            if h < 4:
                break
            row = int(RNG.integers(1, h - 1))
            nub_dir = cap_dir if RNG.random() < 0.35 else -cap_dir
            max_nub = stem_col if nub_dir == -1 else w - stem_col - 1
            if max_nub <= 0:
                continue
            nub_len = int(RNG.integers(1, min(2, max_nub) + 1))
            if nub_dir == -1:
                mask[row, stem_col - nub_len : stem_col] = True
            else:
                mask[row, stem_col + 1 : stem_col + 1 + nub_len] = True

        counts = _edge_counts(mask)
        if int(np.argmax(counts)) != DOWN:
            continue
        if int(np.count_nonzero(counts == counts.max())) != 1:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to build a valid asymmetric pattern")


def _make_room_pattern(room_size: int, bg: int, color: int, input_rot: int) -> np.ndarray:
    room = np.full((room_size, room_size), bg, dtype=int)
    mask = _make_down_mask(room_size)
    h, w = map(int, mask.shape)
    top = int(RNG.integers(1, room_size - h))
    left = int(RNG.integers(1, room_size - w))
    patch = room[top : top + h, left : left + w]
    patch[mask] = color
    return np.rot90(room, input_rot)


def _available_door_options(door_dirs: np.ndarray, i: int, j: int) -> list[tuple[str, int]]:
    nr, nc = map(int, door_dirs.shape)
    options: list[tuple[str, int]] = []
    if i == 0:
        options.append(("boundary", UP))
    if j == nc - 1:
        options.append(("boundary", RIGHT))
    if i == nr - 1:
        options.append(("boundary", DOWN))
    if j == 0:
        options.append(("boundary", LEFT))
    for direction, (di, dj) in enumerate(DIRS4):
        ni, nj = i + di, j + dj
        if 0 <= ni < nr and 0 <= nj < nc and int(door_dirs[ni, nj]) == -1:
            options.append(("pair", direction))
    return options


def _assign_door_dirs(n_rows: int, n_cols: int) -> np.ndarray:
    door_dirs = np.full((n_rows, n_cols), -1, dtype=int)

    def search() -> bool:
        unassigned = [(i, j) for i in range(n_rows) for j in range(n_cols) if int(door_dirs[i, j]) == -1]
        if not unassigned:
            return True

        cell = min(unassigned, key=lambda rc: len(_available_door_options(door_dirs, rc[0], rc[1])))
        i, j = cell
        options = _available_door_options(door_dirs, i, j)
        RNG.shuffle(options)

        for kind, direction in options:
            if kind == "boundary":
                door_dirs[i, j] = direction
                if search():
                    return True
                door_dirs[i, j] = -1
                continue

            di, dj = DIRS4[direction]
            ni, nj = i + di, j + dj
            if int(door_dirs[ni, nj]) != -1:
                continue
            door_dirs[i, j] = direction
            door_dirs[ni, nj] = (direction + 2) % 4
            if search():
                return True
            door_dirs[i, j] = -1
            door_dirs[ni, nj] = -1

        return False

    assert search(), "failed to assign one door side per room"
    return door_dirs


def _room_origin(room_size: int, i: int, j: int) -> tuple[int, int]:
    return i * (room_size + 1) + 1, j * (room_size + 1) + 1


def _paint_boundary_door(grid: np.ndarray, room_size: int, i: int, j: int, direction: int, color: int) -> None:
    r0, c0 = _room_origin(room_size, i, j)
    pos = int(RNG.integers(0, room_size))
    if direction == UP:
        grid[r0 - 1, c0 + pos] = color
    elif direction == RIGHT:
        grid[r0 + pos, c0 + room_size] = color
    elif direction == DOWN:
        grid[r0 + room_size, c0 + pos] = color
    else:
        grid[r0 + pos, c0 - 1] = color


def _paint_internal_horizontal_door(grid: np.ndarray, room_size: int, i: int, j: int, color: int) -> None:
    r0, c0 = _room_origin(room_size, i, j)
    pos = int(RNG.integers(0, room_size))
    grid[r0 + pos, c0 + room_size] = color


def _paint_internal_vertical_door(grid: np.ndarray, room_size: int, i: int, j: int, color: int) -> None:
    r0, c0 = _room_origin(room_size, i, j)
    pos = int(RNG.integers(0, room_size))
    grid[r0 + room_size, c0 + pos] = color


def _validate_generated_instance(grid: np.ndarray) -> bool:
    try:
        info = _extract_structure(grid)
        rooms = info["rooms"]
        door_dirs = []
        current_dirs = []
        rotations = []
        for room in rooms:
            interior = grid[room["r0"] : room["r1"], room["c0"] : room["c1"]]
            current_dir = _feature_direction(interior, info["bg"])
            door_dir = _door_direction(grid, room, info["wall"])
            current_dirs.append(current_dir)
            door_dirs.append(door_dir)
            rotations.append((current_dir - door_dir) % 4)

        out = main(grid)
        out_info = _extract_structure(out)
        assert info["row_bands"] == out_info["row_bands"]
        assert info["col_bands"] == out_info["col_bands"]

        fixed_mask = np.ones(grid.shape, dtype=bool)
        for room in rooms:
            fixed_mask[room["r0"] : room["r1"], room["c0"] : room["c1"]] = False
        assert np.array_equal(grid[fixed_mask], out[fixed_mask])

        for out_room in out_info["rooms"]:
            out_interior = out[out_room["r0"] : out_room["r1"], out_room["c0"] : out_room["c1"]]
            assert _feature_direction(out_interior, out_info["bg"]) == _door_direction(out, out_room, out_info["wall"])

        bg = get_bg(grid)
        counts = dict(zip(*np.unique(grid, return_counts=True)))
        bg_count = int(counts[bg])
        others = [int(v) for color, v in counts.items() if int(color) != bg]
        assert bg_count > max(others)
        assert len(set(door_dirs)) >= min(3, len(door_dirs))
        assert len(set(current_dirs)) >= min(3, len(current_dirs))
        assert any(rot != 0 for rot in rotations)
        assert len(set(rotations)) >= 2
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    room_size, n_rows, n_cols = LAYOUTS[int(RNG.integers(0, len(LAYOUTS)))]
    bg = _pick_color({WALL_COLOR})
    room_color_pool = [c for c in COLOR_POOL if c != bg]

    door_dirs = _assign_door_dirs(n_rows, n_cols)

    n_rooms = n_rows * n_cols
    rotations = np.resize(np.arange(4), n_rooms).copy()
    RNG.shuffle(rotations)
    rotations = rotations.reshape(n_rows, n_cols)

    room_colors = RNG.choice(room_color_pool, size=(n_rows, n_cols), replace=True)

    height = n_rows * room_size + n_rows + 1
    width = n_cols * room_size + n_cols + 1
    grid = np.full((height, width), WALL_COLOR, dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            r0, c0 = _room_origin(room_size, i, j)
            grid[r0 : r0 + room_size, c0 : c0 + room_size] = _make_room_pattern(
                room_size,
                bg,
                int(room_colors[i, j]),
                int(rotations[i, j]),
            )

    for i in range(n_rows):
        for j in range(n_cols):
            direction = int(door_dirs[i, j])
            if direction == UP and i == 0:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j])})
                _paint_boundary_door(grid, room_size, i, j, direction, color)
            elif direction == RIGHT and j == n_cols - 1:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j])})
                _paint_boundary_door(grid, room_size, i, j, direction, color)
            elif direction == DOWN and i == n_rows - 1:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j])})
                _paint_boundary_door(grid, room_size, i, j, direction, color)
            elif direction == LEFT and j == 0:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j])})
                _paint_boundary_door(grid, room_size, i, j, direction, color)

    for i in range(n_rows):
        for j in range(n_cols - 1):
            if int(door_dirs[i, j]) == RIGHT and int(door_dirs[i, j + 1]) == LEFT:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j]), int(room_colors[i, j + 1])})
                _paint_internal_horizontal_door(grid, room_size, i, j, color)

    for i in range(n_rows - 1):
        for j in range(n_cols):
            if int(door_dirs[i, j]) == DOWN and int(door_dirs[i + 1, j]) == UP:
                color = _pick_color({bg, WALL_COLOR, int(room_colors[i, j]), int(room_colors[i + 1, j])})
                _paint_internal_vertical_door(grid, room_size, i, j, color)

    if not _validate_generated_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-rotation ARC input")
