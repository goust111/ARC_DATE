from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_NAME = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}
NAME_TO_DIR = {name: direction for direction, name in DIR_TO_NAME.items()}
ARROW_PATTERNS = {
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "right": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}


def _longest_true_run(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in values.astype(bool).tolist():
        if value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _separator_indices(structural: np.ndarray, axis: int) -> list[int]:
    lines = structural if axis == 0 else structural.T
    indices: list[int] = []
    for idx in range(lines.shape[0]):
        if _longest_true_run(lines[idx]) >= 6:
            indices.append(idx)
    return indices


def _room_boxes(row_bounds: list[int], col_bounds: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ri in range(len(row_bounds) - 1):
        for ci in range(len(col_bounds) - 1):
            boxes[(ri, ci)] = (row_bounds[ri], row_bounds[ri + 1], col_bounds[ci], col_bounds[ci + 1])
    return boxes


def _arrow_direction(room: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(room == GREY)
    assert coords.shape[0] == 5, "each room must contain exactly one 5-pixel grey arrow"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "arrow must occupy a 3x3 box"
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for name, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            return NAME_TO_DIR[name]
    raise AssertionError("grey pixels do not match an arrow pattern")


def _gap_exists(grid: np.ndarray, box: tuple[int, int, int, int], direction: tuple[int, int]) -> bool:
    top, bottom, left, right = box
    dr, dc = direction
    if dr == -1:
        return bool(np.any(grid[top, left + 1 : right] == BG))
    if dr == 1:
        return bool(np.any(grid[bottom, left + 1 : right] == BG))
    if dc == -1:
        return bool(np.any(grid[top + 1 : bottom, left] == BG))
    return bool(np.any(grid[top + 1 : bottom, right] == BG))


def _neighbor_index(
    room_idx: tuple[int, int], direction: tuple[int, int], room_shape: tuple[int, int]
) -> tuple[int, int] | None:
    nr = room_idx[0] + direction[0]
    nc = room_idx[1] + direction[1]
    if 0 <= nr < room_shape[0] and 0 <= nc < room_shape[1]:
        return nr, nc
    return None


def _translate_payload(coords: np.ndarray, box: tuple[int, int, int, int], direction: tuple[int, int]) -> np.ndarray:
    out = coords.copy()
    top, bottom, left, right = box
    dr, dc = direction
    if dr == -1:
        delta = (top - 1) - int(coords[:, 0].max())
        out[:, 0] += delta
    elif dr == 1:
        delta = (bottom + 1) - int(coords[:, 0].min())
        out[:, 0] += delta
    elif dc == -1:
        delta = (left - 1) - int(coords[:, 1].max())
        out[:, 1] += delta
    else:
        delta = (right + 1) - int(coords[:, 1].min())
        out[:, 1] += delta
    return out


def _trace_payload(
    start_room: tuple[int, int],
    coords: np.ndarray,
    arrows: dict[tuple[int, int], tuple[int, int]],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    room_shape: tuple[int, int],
    grid: np.ndarray,
) -> np.ndarray:
    room = start_room
    out = coords.copy()
    seen = {room}
    while True:
        direction = arrows[room]
        box = boxes[room]
        nxt = _neighbor_index(room, direction, room_shape)
        if nxt is None or not _gap_exists(grid, box, direction):
            break
        if nxt in seen:
            break
        out = _translate_payload(out, box, direction)
        room = nxt
        seen.add(room)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In wall-separated rectangular rooms, move each payload through the gap that matches the
    room arrow direction, chaining across adjacent rooms until no matching gap exists.
    All arrows are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    structural = (g != BG) & (g != GREY)
    row_bounds = _separator_indices(structural, axis=0)
    col_bounds = _separator_indices(structural, axis=1)
    assert row_bounds and col_bounds
    assert row_bounds[0] == 0 and row_bounds[-1] == h - 1
    assert col_bounds[0] == 0 and col_bounds[-1] == w - 1
    assert len(row_bounds) >= 3 and len(col_bounds) >= 3

    boxes = _room_boxes(row_bounds, col_bounds)
    room_shape = (len(row_bounds) - 1, len(col_bounds) - 1)

    arrows: dict[tuple[int, int], tuple[int, int]] = {}
    payloads: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
    out = g.copy()
    out[g == GREY] = BG

    for room_idx, box in boxes.items():
        top, bottom, left, right = box
        assert bottom - top >= 5 and right - left >= 5, "room interior must fit arrow and payload"
        room = g[top + 1 : bottom, left + 1 : right]
        arrows[room_idx] = _arrow_direction(room)

        local_coords = np.argwhere((room != BG) & (room != GREY))
        if local_coords.size == 0:
            continue
        global_coords = local_coords + np.array([top + 1, left + 1], dtype=int)
        colors = g[global_coords[:, 0], global_coords[:, 1]].copy()
        payloads.append((room_idx, global_coords, colors))
        out[global_coords[:, 0], global_coords[:, 1]] = BG

    for room_idx, coords, colors in payloads:
        final_coords = _trace_payload(room_idx, coords, arrows, boxes, room_shape, g)
        for (rr, cc), color in zip(final_coords.tolist(), colors.tolist()):
            assert out[rr, cc] == BG, "final payloads must not overlap"
            out[rr, cc] = int(color)

    return out


PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (6, 8)
CHAIN_LENGTH_RANGE = (2, 4)
GEN_RETRIES = 400
SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _cumulative_bounds(sizes: list[int]) -> list[int]:
    bounds = [0]
    cur = 0
    for size in sizes:
        cur += size + 1
        bounds.append(cur)
    return bounds


def _paint_walls(grid: np.ndarray, row_bounds: list[int], col_bounds: list[int], wall_color: int) -> None:
    for row in row_bounds:
        grid[row, :] = wall_color
    for col in col_bounds:
        grid[:, col] = wall_color


def _boundary_info(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, int]:
    if a[0] == b[0]:
        return "v", min(a[1], b[1]) + 1
    return "h", min(a[0], b[0]) + 1


def _sample_path(
    room_rows: int,
    room_cols: int,
    length: int,
    reserved_rooms: set[tuple[int, int]],
    reserved_h: set[int],
    reserved_v: set[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols) if (ri, ci) not in reserved_rooms]
    rng.shuffle(starts)

    def dfs(
        path: list[tuple[int, int]],
        used_h: set[int],
        used_v: set[int],
    ) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return path.copy()

        ri, ci = path[-1]
        moves = []
        for dr, dc in DIRS:
            nr = ri + dr
            nc = ci + dc
            nxt = (nr, nc)
            if not (0 <= nr < room_rows and 0 <= nc < room_cols):
                continue
            if nxt in path or nxt in reserved_rooms:
                continue
            kind, idx = _boundary_info((ri, ci), nxt)
            if kind == "h":
                if idx in used_h or idx in reserved_h:
                    continue
            else:
                if idx in used_v or idx in reserved_v:
                    continue
            moves.append((nxt, kind, idx))
        rng.shuffle(moves)

        for nxt, kind, idx in moves:
            path.append(nxt)
            if kind == "h":
                used_h.add(idx)
            else:
                used_v.add(idx)
            result = dfs(path, used_h, used_v)
            if result is not None:
                return result
            if kind == "h":
                used_h.remove(idx)
            else:
                used_v.remove(idx)
            path.pop()
        return None

    for start in starts:
        result = dfs([start], set(), set())
        if result is not None:
            return result
    return None


def _all_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask.copy()
        for k in range(4):
            rotated = np.rot90(cur, k)
            key = rotated.tobytes() + str(rotated.shape).encode()
            if not any(v.tobytes() + str(v.shape).encode() == key for v in variants):
                variants.append(rotated.copy())
    return variants


def _sample_shape(max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    choices = []
    for base in SHAPES:
        for variant in _all_rotations(base):
            vh, vw = variant.shape
            if vh <= max_h and vw <= max_w:
                choices.append(variant)
    if not choices:
        return None
    return rng.choice(choices).copy()


def _arrow_cells(box: tuple[int, int, int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    top, bottom, left, right = box
    room_h = bottom - top - 1
    room_w = right - left - 1
    anchor_r = top + 1 + (room_h - 3) // 2
    anchor_c = left + 1 + (room_w - 3) // 2
    pattern = ARROW_PATTERNS[DIR_TO_NAME[direction]]
    return {(anchor_r + dr, anchor_c + dc) for dr, dc in pattern}


def _paint_arrow(grid: np.ndarray, box: tuple[int, int, int, int], direction: tuple[int, int]) -> None:
    for rr, cc in _arrow_cells(box, direction):
        grid[rr, cc] = GREY


def _place_gap_for_step(
    grid: np.ndarray,
    row_bounds: list[int],
    col_bounds: list[int],
    a: tuple[int, int],
    b: tuple[int, int],
    rng: random.Random,
) -> None:
    gap_width = rng.choice([1, 2])
    if a[0] == b[0]:
        room_row = a[0]
        sep_col = col_bounds[min(a[1], b[1]) + 1]
        top = row_bounds[room_row]
        bottom = row_bounds[room_row + 1]
        start_min = top + 2
        start_max = bottom - gap_width - 1
        assert start_min <= start_max
        start = rng.randint(start_min, start_max)
        grid[start : start + gap_width, sep_col] = BG
    else:
        room_col = a[1]
        sep_row = row_bounds[min(a[0], b[0]) + 1]
        left = col_bounds[room_col]
        right = col_bounds[room_col + 1]
        start_min = left + 2
        start_max = right - gap_width - 1
        assert start_min <= start_max
        start = rng.randint(start_min, start_max)
        grid[sep_row, start : start + gap_width] = BG


def _candidate_payload_positions(
    box: tuple[int, int, int, int],
    shape: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[np.ndarray]:
    top, bottom, left, right = box
    sh, sw = shape.shape
    shape_offsets = np.argwhere(shape)
    candidates: list[np.ndarray] = []
    for r0 in range(top + 1, bottom - sh + 1):
        for c0 in range(left + 1, right - sw + 1):
            coords = shape_offsets + np.array([r0, c0], dtype=int)
            cells = {tuple(map(int, coord)) for coord in coords.tolist()}
            if cells & blocked:
                continue
            candidates.append(coords.astype(int, copy=True))
    return candidates


def _choose_payload_coords(
    box: tuple[int, int, int, int],
    shape: np.ndarray,
    blocked: set[tuple[int, int]],
    direction: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    candidates = _candidate_payload_positions(box, shape, blocked)
    if not candidates:
        return None

    top, bottom, left, right = box

    def score(coords: np.ndarray) -> tuple[int, int, int]:
        if direction == (0, 1):
            primary = right - int(coords[:, 1].max())
        elif direction == (0, -1):
            primary = int(coords[:, 1].min()) - left
        elif direction == (1, 0):
            primary = bottom - int(coords[:, 0].max())
        else:
            primary = int(coords[:, 0].min()) - top
        secondary = abs(int(coords[:, 0].mean() * 10) - (top + bottom) * 5)
        tertiary = abs(int(coords[:, 1].mean() * 10) - (left + right) * 5)
        return primary, -secondary, -tertiary

    candidates.sort(key=score, reverse=True)
    best_band = candidates[: min(4, len(candidates))]
    return rng.choice(best_band).copy()


def _manual_expected(
    grid: np.ndarray,
    payload_specs: list[dict],
    arrows: dict[tuple[int, int], tuple[int, int]],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    room_shape: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[grid == GREY] = BG
    for spec in payload_specs:
        coords = spec["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for spec in payload_specs:
        final_coords = spec["coords"].copy()
        path = spec["path"]
        for src, dst in zip(path, path[1:]):
            direction = (dst[0] - src[0], dst[1] - src[1])
            final_coords = _translate_payload(final_coords, boxes[src], direction)
        for (rr, cc) in final_coords.tolist():
            assert out[rr, cc] == BG
        out[final_coords[:, 0], final_coords[:, 1]] = spec["color"]
    return out


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(*ROOM_COUNT_RANGE)
    room_cols = rng.randint(*ROOM_COUNT_RANGE)
    row_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    row_bounds = _cumulative_bounds(row_sizes)
    col_bounds = _cumulative_bounds(col_sizes)
    height = row_bounds[-1] + 1
    width = col_bounds[-1] + 1

    wall_color = rng.choice(PALETTE)
    payload_palette = [color for color in PALETTE if color != wall_color]
    grid = np.full((height, width), BG, dtype=int)
    _paint_walls(grid, row_bounds, col_bounds, wall_color)

    boxes = _room_boxes(row_bounds, col_bounds)
    room_shape = (room_rows, room_cols)

    target_chains = 2 if room_rows * room_cols >= 6 and rng.random() < 0.5 else 1
    reserved_rooms: set[tuple[int, int]] = set()
    reserved_h: set[int] = set()
    reserved_v: set[int] = set()
    paths: list[list[tuple[int, int]]] = []

    for chain_idx in range(target_chains):
        remaining_rooms = room_rows * room_cols - len(reserved_rooms)
        if remaining_rooms < 2:
            break
        max_len = min(CHAIN_LENGTH_RANGE[1], remaining_rooms)
        min_len = min(CHAIN_LENGTH_RANGE[0], max_len)
        lengths = list(range(min_len, max_len + 1))
        rng.shuffle(lengths)
        found = None
        for length in lengths:
            found = _sample_path(room_rows, room_cols, length, reserved_rooms, reserved_h, reserved_v, rng)
            if found is not None:
                break
        if found is None:
            if chain_idx == 0:
                return None
            break
        paths.append(found)
        reserved_rooms.update(found)
        for a, b in zip(found, found[1:]):
            kind, idx = _boundary_info(a, b)
            if kind == "h":
                reserved_h.add(idx)
            else:
                reserved_v.add(idx)

    if not paths:
        return None

    arrows: dict[tuple[int, int], tuple[int, int]] = {}
    for room_idx in boxes:
        arrows[room_idx] = rng.choice(DIRS)

    for path in paths:
        for src, dst in zip(path, path[1:]):
            arrows[src] = (dst[0] - src[0], dst[1] - src[1])
        final_room = path[-1]
        back = (path[-2][0] - final_room[0], path[-2][1] - final_room[1])
        stop_choices = [direction for direction in DIRS if direction != back]
        rng.shuffle(stop_choices)
        arrows[final_room] = stop_choices[0]

    for path in paths:
        for a, b in zip(path, path[1:]):
            _place_gap_for_step(grid, row_bounds, col_bounds, a, b, rng)

    for room_idx, box in boxes.items():
        _paint_arrow(grid, box, arrows[room_idx])

    payload_specs: list[dict] = []
    rng.shuffle(payload_palette)
    if len(payload_palette) < len(paths):
        return None

    for payload_color, path in zip(payload_palette, paths):
        min_room_h = min(row_sizes[ri] for ri, _ in path)
        min_room_w = min(col_sizes[ci] for _, ci in path)
        shape = _sample_shape(min(3, min_room_h), min(3, min_room_w), rng)
        if shape is None:
            return None

        start_room = path[0]
        start_box = boxes[start_room]
        blocked = _arrow_cells(start_box, arrows[start_room])
        coords = _choose_payload_coords(start_box, shape, blocked, arrows[start_room], rng)
        if coords is None:
            return None

        if np.any(grid[coords[:, 0], coords[:, 1]] != BG):
            return None
        grid[coords[:, 0], coords[:, 1]] = payload_color
        payload_specs.append(
            {
                "color": payload_color,
                "coords": coords.copy(),
                "path": path,
            }
        )

    try:
        expected = _manual_expected(grid, payload_specs, arrows, boxes, room_shape)
    except AssertionError:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None

    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    if sum(int(np.count_nonzero(grid == spec["color"])) for spec in payload_specs) < len(payload_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
