import random

import numpy as np


ROOM_ROWS_RANGE = (2, 3)
ROOM_COLS_RANGE = (2, 3)
ROOM_SIZE_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 400
GAP_PROBABILITY = 0.5

SHAPE_TEMPLATES = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1, 0], [1, 0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=int),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _split_line_indices(counts: np.ndarray) -> list[int]:
    values = sorted({int(v) for v in counts if int(v) > 0})
    assert len(values) >= 2
    split = max(range(len(values) - 1), key=lambda i: values[i + 1] - values[i])
    threshold = (values[split] + values[split + 1]) / 2.0
    indices = [i for i, count in enumerate(counts) if float(count) > threshold]
    assert len(indices) >= 2
    return indices


def _infer_wall_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if corners[0] != 0 and len(set(corners)) == 1:
        return corners[0]

    best: tuple[int, int] | None = None
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != 0):
        mask = grid == color
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        try:
            row_lines = _split_line_indices(row_counts)
            col_lines = _split_line_indices(col_counts)
        except AssertionError:
            continue
        row_set = set(row_lines)
        col_set = set(col_lines)
        coords = np.argwhere(mask)
        if all(int(r) in row_set or int(c) in col_set for r, c in coords):
            score = int(mask.sum())
            if best is None or score > best[0]:
                best = (score, color)

    assert best is not None
    return best[1]


def _room_slices(row_lines: list[int], col_lines: list[int]) -> list[tuple[int, int, int, int]]:
    rooms = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1] - 1
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1] - 1
            assert r0 <= r1 and c0 <= c1
            rooms.append((r0, r1, c0, c1))
    return rooms


def _rotate_clockwise(sprite: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(sprite, -int(turns % 4))


def _apply_room_rotation(
    out: np.ndarray,
    grid: np.ndarray,
    wall_color: int,
    room: tuple[int, int, int, int],
    exits: int,
) -> None:
    r0, r1, c0, c1 = room
    interior = grid[r0 : r1 + 1, c0 : c1 + 1]
    obj_mask = (interior != 0) & (interior != wall_color)
    assert obj_mask.any()

    room_view = out[r0 : r1 + 1, c0 : c1 + 1]
    room_view[(room_view != 0) & (room_view != wall_color)] = 0

    br0, bc0, br1, bc1 = _bbox(obj_mask)
    abs_r0 = r0 + br0
    abs_r1 = r0 + br1
    abs_c0 = c0 + bc0
    abs_c1 = c0 + bc1
    sprite = grid[abs_r0 : abs_r1 + 1, abs_c0 : abs_c1 + 1]
    rotated = _rotate_clockwise(sprite, exits)

    new_h, new_w = map(int, rotated.shape)
    new_r0_num = abs_r0 + abs_r1 - (new_h - 1)
    new_c0_num = abs_c0 + abs_c1 - (new_w - 1)
    assert new_r0_num % 2 == 0 and new_c0_num % 2 == 0
    new_r0 = new_r0_num // 2
    new_c0 = new_c0_num // 2

    assert r0 <= new_r0 and new_r0 + new_h - 1 <= r1
    assert c0 <= new_c0 and new_c0 + new_w - 1 <= c1
    patch = out[new_r0 : new_r0 + new_h, new_c0 : new_c0 + new_w]
    patch[rotated != 0] = rotated[rotated != 0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_color = _infer_wall_color(g)
    wall_mask = g == wall_color
    row_lines = _split_line_indices(wall_mask.sum(axis=1))
    col_lines = _split_line_indices(wall_mask.sum(axis=0))
    assert row_lines[0] == 0 and row_lines[-1] == g.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == g.shape[1] - 1

    out = g.copy()
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            room = (row_lines[i] + 1, row_lines[i + 1] - 1, col_lines[j] + 1, col_lines[j + 1] - 1)
            exits = 0
            exits += int(np.sum(g[row_lines[i], col_lines[j] + 1 : col_lines[j + 1]] == 0))
            exits += int(np.sum(g[row_lines[i + 1], col_lines[j] + 1 : col_lines[j + 1]] == 0))
            exits += int(np.sum(g[row_lines[i] + 1 : row_lines[i + 1], col_lines[j]] == 0))
            exits += int(np.sum(g[row_lines[i] + 1 : row_lines[i + 1], col_lines[j + 1]] == 0))
            _apply_room_rotation(out, g, wall_color, room, exits)
    return out


def _validate_templates() -> None:
    for template in SHAPE_TEMPLATES:
        rotations = {_rotate_clockwise(template, k).tobytes() for k in range(4)}
        assert len(rotations) == 4
        h, w = map(int, template.shape)
        assert h % 2 == w % 2


def _sample_room_sizes(count: int) -> list[int]:
    return [random.randint(*ROOM_SIZE_RANGE) for _ in range(count)]


def _build_wall_grid(
    room_heights: list[int], room_widths: list[int], wall_color: int
) -> tuple[np.ndarray, list[int], list[int]]:
    row_lines = [0]
    for height in room_heights:
        row_lines.append(row_lines[-1] + int(height) + 1)
    col_lines = [0]
    for width in room_widths:
        col_lines.append(col_lines[-1] + int(width) + 1)

    grid = np.zeros((row_lines[-1] + 1, col_lines[-1] + 1), dtype=int)
    for r in row_lines:
        grid[r, :] = wall_color
    for c in col_lines:
        grid[:, c] = wall_color
    return grid, row_lines, col_lines


def _place_gap(
    grid: np.ndarray,
    *,
    horizontal: bool,
    line: int,
    start: int,
    end: int,
) -> bool:
    if start > end or random.random() >= GAP_PROBABILITY:
        return False
    pos = random.randint(start, end)
    if horizontal:
        grid[line, pos] = 0
    else:
        grid[pos, line] = 0
    return True


def _sample_gaps(
    grid: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    room_rows = len(row_lines) - 1
    room_cols = len(col_lines) - 1
    horizontal_gaps = np.zeros((room_rows + 1, room_cols), dtype=int)
    vertical_gaps = np.zeros((room_rows, room_cols + 1), dtype=int)

    for i in range(room_rows + 1):
        for j in range(room_cols):
            if _place_gap(
                grid,
                horizontal=True,
                line=row_lines[i],
                start=col_lines[j] + 1,
                end=col_lines[j + 1] - 1,
            ):
                horizontal_gaps[i, j] = 1

    for i in range(room_rows):
        for j in range(room_cols + 1):
            if _place_gap(
                grid,
                horizontal=False,
                line=col_lines[j],
                start=row_lines[i] + 1,
                end=row_lines[i + 1] - 1,
            ):
                vertical_gaps[i, j] = 1

    return horizontal_gaps, vertical_gaps


def _room_exit_counts(horizontal_gaps: np.ndarray, vertical_gaps: np.ndarray) -> np.ndarray:
    return (
        horizontal_gaps[:-1, :]
        + horizontal_gaps[1:, :]
        + vertical_gaps[:, :-1]
        + vertical_gaps[:, 1:]
    )


def _choose_shape(room_h: int, room_w: int) -> np.ndarray:
    candidates = [shape for shape in SHAPE_TEMPLATES if max(shape.shape) <= min(room_h, room_w)]
    assert candidates
    shape = random.choice(candidates).copy()
    shape = _rotate_clockwise(shape, random.randrange(4))
    return shape


def _paint_object(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    shape: np.ndarray,
    color: int,
) -> dict[str, int | np.ndarray]:
    r0, r1, c0, c1 = room
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    h, w = map(int, shape.shape)
    support = max(h, w)
    support_r0 = random.randint(0, room_h - support)
    support_c0 = random.randint(0, room_w - support)
    center_r_num = 2 * (r0 + support_r0) + support - 1
    center_c_num = 2 * (c0 + support_c0) + support - 1
    obj_r0 = (center_r_num - (h - 1)) // 2
    obj_c0 = (center_c_num - (w - 1)) // 2

    patch = grid[obj_r0 : obj_r0 + h, obj_c0 : obj_c0 + w]
    patch[shape != 0] = color
    return {
        "room_r0": r0,
        "room_r1": r1,
        "room_c0": c0,
        "room_c1": c1,
        "center_r_num": center_r_num,
        "center_c_num": center_c_num,
        "sprite": (shape * color).astype(int),
    }


def _build_expected(
    grid: np.ndarray,
    wall_color: int,
    room_infos: list[dict[str, int | np.ndarray]],
    counts: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    room_cols = counts.shape[1]
    for idx, info in enumerate(room_infos):
        r0 = int(info["room_r0"])
        r1 = int(info["room_r1"])
        c0 = int(info["room_c0"])
        c1 = int(info["room_c1"])
        room_view = out[r0 : r1 + 1, c0 : c1 + 1]
        room_view[(room_view != 0) & (room_view != wall_color)] = 0

        turns = int(counts[idx // room_cols, idx % room_cols])
        sprite = np.asarray(info["sprite"], dtype=int)
        rotated = _rotate_clockwise(sprite, turns)
        new_h, new_w = map(int, rotated.shape)
        center_r_num = int(info["center_r_num"])
        center_c_num = int(info["center_c_num"])
        new_r0 = (center_r_num - (new_h - 1)) // 2
        new_c0 = (center_c_num - (new_w - 1)) // 2
        patch = out[new_r0 : new_r0 + new_h, new_c0 : new_c0 + new_w]
        patch[rotated != 0] = rotated[rotated != 0]
    return out


def _coverage_ok(counts: np.ndarray) -> bool:
    values = set(map(int, counts.ravel()))
    return 1 in values and 2 in values and 3 in values and (0 in values or 4 in values)


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)
    room_heights = _sample_room_sizes(room_rows)
    room_widths = _sample_room_sizes(room_cols)

    wall_color = random.randint(1, 9)
    object_palette = [color for color in range(1, 10) if color != wall_color]
    grid, row_lines, col_lines = _build_wall_grid(room_heights, room_widths, wall_color)
    horizontal_gaps, vertical_gaps = _sample_gaps(grid, row_lines, col_lines)
    counts = _room_exit_counts(horizontal_gaps, vertical_gaps)
    if not _coverage_ok(counts):
        return None

    rooms = _room_slices(row_lines, col_lines)
    room_infos = []
    for room in rooms:
        r0, r1, c0, c1 = room
        shape = _choose_shape(r1 - r0 + 1, c1 - c0 + 1)
        color = random.choice(object_palette)
        room_infos.append(_paint_object(grid, room, shape, color))

    expected = _build_expected(grid, wall_color, room_infos, counts)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    _validate_templates()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
