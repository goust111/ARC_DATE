import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
WALL_SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300
MAX_ROOM_DIM = 7

BASE_KEY_MASKS = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the matching bottom key pattern, anchored opposite the room's wall marker.

    Input structure:
    - The background is black, and blue full-width/full-height wall lines partition the upper part of the grid into closed rectangular rooms.
    - Every room interior contains exactly one non-black marker pixel placed on one inner wall line, indicating both a key color and a wall side.
    - Below the wall structure, a black key strip contains one disconnected monochrome key sprite for each marker color.

    Rule:
    - Crop away the bottom key strip, keeping only the room area up to the last blue wall row.
    - For each room, match its marker color to the key sprite of the same color.
    - Tile that sprite across the room interior at its original size, anchoring the tiling to the wall opposite the marker side; keep the walls and the marker pixel unchanged.

    Key insight:
    - The marker encodes two independent facts: its color selects which bottom key to use, and its wall position selects which opposite wall the repeated tiling must align to.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    blue_rows = np.where(np.any(g == BLUE, axis=1))[0]
    assert len(blue_rows) > 0, "expected blue wall rows"
    crop_h = int(blue_rows[-1] + 1)
    room_area = g[:crop_h].copy()
    key_area = g[crop_h:]
    assert key_area.size > 0, "expected a bottom key strip"

    sep_rows = [r for r in range(room_area.shape[0]) if np.all(room_area[r, :] == BLUE)]
    sep_cols = [c for c in range(room_area.shape[1]) if np.all(room_area[:, c] == BLUE)]
    assert sep_rows and sep_cols, "expected full blue separator rows and columns"

    row_spans = _bands(room_area.shape[0], sep_rows)
    col_spans = _bands(room_area.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one room"

    keys_by_color = _extract_keys(key_area)

    out = room_area.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            room = room_area[r0 : r1 + 1, c0 : c1 + 1]
            markers = np.argwhere((room != BLACK) & (room != BLUE))
            assert len(markers) == 1, "each room must contain exactly one marker"
            mr, mc = map(int, markers[0])
            marker_color = int(room[mr, mc])
            assert marker_color in keys_by_color, "marker color must match a bottom key"
            wall_side = _marker_wall_side(mr, mc, room.shape)
            tiled = _tile_key(keys_by_color[marker_color], room.shape, wall_side)
            tiled[mr, mc] = marker_color
            out[r0 : r1 + 1, c0 : c1 + 1] = tiled
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-key puzzle")


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(int(s) for s in separators)
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if sep > start:
            bands.append((start, sep - 1))
        start = sep + 1
    if start < length:
        bands.append((start, length - 1))
    return bands


def _marker_wall_side(row: int, col: int, shape: tuple[int, int]) -> str:
    h, w = map(int, shape)
    sides = []
    if row == 0:
        sides.append("top")
    if row == h - 1:
        sides.append("bottom")
    if col == 0:
        sides.append("left")
    if col == w - 1:
        sides.append("right")
    assert len(sides) == 1, "marker must lie on exactly one wall side"
    return sides[0]


def _tile_key(key: np.ndarray, room_shape: tuple[int, int], wall_side: str) -> np.ndarray:
    room_h, room_w = map(int, room_shape)
    key_h, key_w = map(int, key.shape)
    reps_r = room_h // key_h + 2
    reps_c = room_w // key_w + 2
    tiled = np.tile(key, (reps_r, reps_c))

    if wall_side == "top":
        row_start = tiled.shape[0] - room_h
        col_start = 0
    elif wall_side == "bottom":
        row_start = 0
        col_start = 0
    elif wall_side == "left":
        row_start = 0
        col_start = tiled.shape[1] - room_w
    elif wall_side == "right":
        row_start = 0
        col_start = 0
    else:
        raise AssertionError(f"unknown wall side: {wall_side}")

    return tiled[row_start : row_start + room_h, col_start : col_start + room_w].copy()


def _extract_keys(key_area: np.ndarray) -> dict[int, np.ndarray]:
    h, w = map(int, key_area.shape)
    seen = np.zeros((h, w), dtype=bool)
    keys: dict[int, np.ndarray] = {}
    for r in range(h):
        for c in range(w):
            if seen[r, c] or key_area[r, c] == BLACK:
                continue
            color = int(key_area[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and key_area[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = key_area[r0 : r1 + 1, c0 : c1 + 1].copy()
            assert set(map(int, np.unique(patch))).issubset({BLACK, color}), "keys must be monochrome"
            assert color not in keys, "each key color must appear exactly once"
            keys[color] = patch
    assert keys, "expected at least one key"
    return keys


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    key_count = int(rng.integers(2, 5))
    color_pool = rng.choice(COLORS, size=key_count, replace=False)
    key_masks = _sample_key_masks(rng, key_count)
    if key_masks is None:
        return None

    keys = {int(color): np.where(mask, int(color), BLACK).astype(int) for color, mask in zip(color_pool, key_masks)}
    max_key_h = max(int(key.shape[0]) for key in keys.values())
    max_key_w = max(int(key.shape[1]) for key in keys.values())

    gaps = [int(rng.integers(1, 3)) for _ in range(max(0, key_count - 1))]
    total_key_w = sum(int(key.shape[1]) for key in keys.values()) + sum(gaps)

    grid_choices = [
        (rows, cols)
        for rows, cols in ((2, 2), (2, 3), (3, 2), (3, 3))
        if total_key_w <= cols * MAX_ROOM_DIM + cols + 1
    ]
    if not grid_choices:
        return None
    n_rows, n_cols = grid_choices[int(rng.integers(len(grid_choices)))]

    min_room_h = max(max_key_h + 1, 4)
    min_room_w = max(max_key_w + 1, 4)
    if min_room_h > MAX_ROOM_DIM or min_room_w > MAX_ROOM_DIM:
        return None

    row_heights = _sample_dims(rng, n_rows, min_room_h, MAX_ROOM_DIM, n_rows * min_room_h)
    col_widths = _sample_dims(
        rng,
        n_cols,
        min_room_w,
        MAX_ROOM_DIM,
        max(n_cols * min_room_w, total_key_w - (n_cols + 1)),
    )
    if row_heights is None or col_widths is None:
        return None

    top_h = sum(row_heights) + n_rows + 1
    top_w = sum(col_widths) + n_cols + 1
    total_h = top_h + 1 + max_key_h
    if total_h > 30 or top_w > 30:
        return None

    room_colors = list(color_pool)
    while len(room_colors) < n_rows * n_cols:
        room_colors.append(int(rng.choice(color_pool)))
    rng.shuffle(room_colors)

    wall_sides = list(WALL_SIDES)
    while len(wall_sides) < n_rows * n_cols:
        wall_sides.append(str(rng.choice(WALL_SIDES)))
    rng.shuffle(wall_sides)

    room_spans_r = []
    start = 1
    for h in row_heights:
        room_spans_r.append((start, start + h - 1))
        start += h + 1
    room_spans_c = []
    start = 1
    for w in col_widths:
        room_spans_c.append((start, start + w - 1))
        start += w + 1

    grid = np.full((total_h, top_w), BLACK, dtype=int)
    room_area = np.full((top_h, top_w), BLUE, dtype=int)
    room_specs = []
    idx = 0
    for r_span in room_spans_r:
        for c_span in room_spans_c:
            r0, r1 = r_span
            c0, c1 = c_span
            room_area[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            wall_side = wall_sides[idx]
            color = int(room_colors[idx])
            key = keys[color]
            room_h = r1 - r0 + 1
            room_w = c1 - c0 + 1
            if not _direction_is_visible(key, room_h, room_w, wall_side):
                return None
            marker_r, marker_c = _random_marker(rng, room_h, room_w, wall_side)
            room_area[r0 + marker_r, c0 + marker_c] = color
            room_specs.append((wall_side, color))
            idx += 1

    grid[:top_h, :] = room_area

    key_left = int(rng.integers(0, top_w - total_key_w + 1))
    key_row = top_h + 1
    cursor = key_left
    for i, color in enumerate(color_pool):
        key = keys[int(color)]
        kh, kw = map(int, key.shape)
        row_offset = max_key_h - kh if rng.random() < 0.5 else 0
        grid[key_row + row_offset : key_row + row_offset + kh, cursor : cursor + kw] = key
        cursor += kw
        if i < len(gaps):
            cursor += gaps[i]

    if not set(room_colors) == set(map(int, color_pool)):
        return None
    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (top_h, top_w):
        return None
    return grid


def _sample_key_masks(rng: np.random.Generator, count: int) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    signatures = set()
    for _ in range(200):
        base = BASE_KEY_MASKS[int(rng.integers(len(BASE_KEY_MASKS)))]
        mask = np.rot90(base, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        sig = (mask.shape, mask.astype(int).tobytes())
        if sig in signatures:
            continue
        signatures.add(sig)
        chosen.append(mask.copy())
        if len(chosen) == count:
            return chosen
    return None


def _sample_dims(
    rng: np.random.Generator,
    count: int,
    min_dim: int,
    max_dim: int,
    needed_sum: int,
) -> list[int] | None:
    dims = [min_dim] * count
    if needed_sum > count * max_dim:
        return None

    required_extra = max(0, needed_sum - sum(dims))
    for _ in range(required_extra):
        choices = [i for i, dim in enumerate(dims) if dim < max_dim]
        if not choices:
            return None
        idx = int(rng.choice(choices))
        dims[idx] += 1

    random_extra = int(rng.integers(0, count * max_dim - sum(dims) + 1))
    for _ in range(random_extra):
        choices = [i for i, dim in enumerate(dims) if dim < max_dim]
        if not choices:
            break
        idx = int(rng.choice(choices))
        dims[idx] += 1

    rng.shuffle(dims)
    return dims


def _direction_is_visible(key: np.ndarray, room_h: int, room_w: int, wall_side: str) -> bool:
    if wall_side in ("top", "bottom"):
        top_fill = _tile_key(key, (room_h, room_w), "bottom")
        bottom_fill = _tile_key(key, (room_h, room_w), "top")
        return not np.array_equal(top_fill, bottom_fill)
    left_fill = _tile_key(key, (room_h, room_w), "right")
    right_fill = _tile_key(key, (room_h, room_w), "left")
    return not np.array_equal(left_fill, right_fill)


def _random_marker(
    rng: np.random.Generator,
    room_h: int,
    room_w: int,
    wall_side: str,
) -> tuple[int, int]:
    if wall_side == "top":
        return 0, int(rng.integers(1, room_w - 1))
    if wall_side == "bottom":
        return room_h - 1, int(rng.integers(1, room_w - 1))
    if wall_side == "left":
        return int(rng.integers(1, room_h - 1)), 0
    if wall_side == "right":
        return int(rng.integers(1, room_h - 1)), room_w - 1
    raise AssertionError(f"unknown wall side: {wall_side}")
