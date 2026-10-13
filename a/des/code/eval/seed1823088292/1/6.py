import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
FIGURE_COLORS = (1, 4, 5, 6, 7, 8, 9)

LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3))
MIN_ROOM_SIZE = 6
MAX_CANVAS_SIZE = 30
GENERATION_RETRIES = 400


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


DIGIT_MASKS = {
    1: _mask(("1", "1", "1")),
    2: _mask(("111", "011", "110")),
    3: _mask(("111", "011", "111")),
    4: _mask(("101", "111", "001")),
    5: _mask(("111", "110", "011")),
    6: _mask(("111", "110", "111")),
    7: _mask(("111", "001", "001")),
    8: _mask(("111", "111", "111")),
    9: _mask(("111", "111", "001")),
}
DIGIT_LOOKUP = {
    (tuple(mask.shape), tuple(mask.reshape(-1).tolist())): digit
    for digit, mask in DIGIT_MASKS.items()
}


def _unique_rotations(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (tuple(rotated.shape), tuple(rotated.reshape(-1).tolist()))
        if key not in seen:
            seen.add(key)
            variants.append(rotated.copy())
    return tuple(variants)


BASE_SHAPES = (
    _mask(("11",)),
    _mask(("111",)),
    _mask(("1111",)),
    _mask(("11", "11")),
    _mask(("111", "111")),
    _mask(("100", "100", "111")),
    _mask(("111", "010", "010")),
    _mask(("010", "111", "010")),
    _mask(("110", "011")),
    _mask(("111", "101")),
)

SHAPE_MASKS = tuple(
    variant
    for base in BASE_SHAPES
    for variant in _unique_rotations(base)
)


def _full_green_lines(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREEN)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREEN)]


def _non_separator_slices(lines: list[int], size: int) -> list[slice]:
    anchors = [-1] + sorted(set(int(x) for x in lines)) + [int(size)]
    spans = []
    for start, stop in zip(anchors, anchors[1:]):
        if stop - start > 1:
            spans.append(slice(start + 1, stop))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_digit(room: np.ndarray) -> int:
    red_mask = room == RED
    assert red_mask.any(), "each room must contain a red digit"
    r0, c0, r1, c1 = _bbox(red_mask)
    digit_mask = red_mask[r0 : r1 + 1, c0 : c1 + 1]
    key = (tuple(digit_mask.shape), tuple(digit_mask.reshape(-1).tolist()))
    assert key in DIGIT_LOOKUP, "unrecognized room digit"
    return int(DIGIT_LOOKUP[key])


def _extract_layer(room: np.ndarray) -> np.ndarray:
    layer = room.copy()
    layer[(layer == BLACK) | (layer == RED) | (layer == GREEN)] = BLACK
    colors = np.unique(layer[layer != BLACK])
    assert colors.size <= 1, "each room figure must be monochrome"
    return layer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite room-local figure layers by the red execution number in each room.

    The input is split by one-cell green lines into equal rooms. Each room has a
    red digit in its upper-left corner and one monochrome figure elsewhere.
    Figures are stamped onto a room-sized black canvas in ascending digit order,
    with later rooms overwriting earlier colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_lines = _full_green_lines(g, axis=0)
    col_lines = _full_green_lines(g, axis=1)
    assert row_lines and col_lines, "expected green room separators"

    row_slices = _non_separator_slices(row_lines, g.shape[0])
    col_slices = _non_separator_slices(col_lines, g.shape[1])
    assert row_slices and col_slices

    room_h = row_slices[0].stop - row_slices[0].start
    room_w = col_slices[0].stop - col_slices[0].start
    assert all(sl.stop - sl.start == room_h for sl in row_slices), "room heights must match"
    assert all(sl.stop - sl.start == room_w for sl in col_slices), "room widths must match"

    layers: list[tuple[int, np.ndarray]] = []
    for row_sl in row_slices:
        for col_sl in col_slices:
            room = g[row_sl, col_sl]
            layers.append((_decode_digit(room), _extract_layer(room)))

    expected_digits = list(range(1, len(layers) + 1))
    digits = sorted(digit for digit, _layer in layers)
    assert digits == expected_digits, "room digits must form 1..K"

    out = np.full((room_h, room_w), BLACK, dtype=int)
    for _digit, layer in sorted(layers, key=lambda item: item[0]):
        mask = layer != BLACK
        out[mask] = layer[mask]
    return out


def _sample_room_size(n_rows: int, n_cols: int) -> tuple[int, int]:
    max_h = min(11, (MAX_CANVAS_SIZE - (n_rows + 1)) // n_rows)
    max_w = min(11, (MAX_CANVAS_SIZE - (n_cols + 1)) // n_cols)
    assert max_h >= MIN_ROOM_SIZE and max_w >= MIN_ROOM_SIZE
    return random.randint(MIN_ROOM_SIZE, max_h), random.randint(MIN_ROOM_SIZE, max_w)


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _candidate_positions(room_h: int, room_w: int, shape: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = shape.shape
    if sh > room_h or sw > room_w:
        return []
    positions = []
    for r0 in range(room_h - sh + 1):
        for c0 in range(room_w - sw + 1):
            if np.any(blocked[r0 : r0 + sh, c0 : c0 + sw] & shape):
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _shape_center(r0: int, c0: int, shape: np.ndarray) -> tuple[float, float]:
    sh, sw = shape.shape
    return r0 + (sh - 1) / 2.0, c0 + (sw - 1) / 2.0


def _choose_shape_and_position(
    room_h: int,
    room_w: int,
    blocked: np.ndarray,
    occupied: np.ndarray,
    mode: str,
    anchor: tuple[int, int],
) -> tuple[np.ndarray, int, int] | None:
    shapes = list(SHAPE_MASKS)
    random.shuffle(shapes)
    padded_blocked = _dilate(blocked)
    for shape in shapes:
        positions = _candidate_positions(room_h, room_w, shape, padded_blocked)
        if not positions:
            positions = _candidate_positions(room_h, room_w, shape, blocked)
        if not positions:
            continue

        scored = []
        for r0, c0 in positions:
            sh, sw = shape.shape
            overlap = int(occupied[r0 : r0 + sh, c0 : c0 + sw][shape].sum())
            center_r, center_c = _shape_center(r0, c0, shape)
            dist = abs(center_r - anchor[0]) + abs(center_c - anchor[1])
            jitter = random.random()
            if mode == "clustered":
                scored.append((overlap, -dist, jitter, r0, c0))
            else:
                scored.append((-overlap, dist, jitter, r0, c0))

        scored.sort(reverse=True)
        top = scored[: min(5, len(scored))]
        _score0, _score1, _jitter, r0, c0 = random.choice(top)
        return shape, int(r0), int(c0)
    return None


def _compose_layers(
    labels: list[int], layers: list[np.ndarray]
) -> tuple[np.ndarray, int, set[int]]:
    room_h, room_w = layers[0].shape
    out = np.full((room_h, room_w), BLACK, dtype=int)
    writes = np.zeros((room_h, room_w), dtype=int)
    owner = np.full((room_h, room_w), -1, dtype=int)
    for label, layer in sorted(zip(labels, layers), key=lambda item: item[0]):
        mask = layer != BLACK
        writes[mask] += 1
        owner[mask] = int(label)
        out[mask] = layer[mask]
    visible_labels = {int(x) for x in np.unique(owner[owner > 0])}
    overlap_cells = int(np.count_nonzero(writes > 1))
    return out, overlap_cells, visible_labels


def _build_room(room_h: int, room_w: int, digit: int, layer: np.ndarray) -> np.ndarray:
    room = np.full((room_h, room_w), BLACK, dtype=int)
    digit_mask = DIGIT_MASKS[digit]
    dh, dw = digit_mask.shape
    room[:dh, :dw][digit_mask] = RED
    room[layer != BLACK] = layer[layer != BLACK]
    return room


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUT_OPTIONS)
    room_count = n_rows * n_cols
    room_h, room_w = _sample_room_size(n_rows, n_cols)

    grid_h = n_rows * room_h + n_rows + 1
    grid_w = n_cols * room_w + n_cols + 1
    if grid_h > MAX_CANVAS_SIZE or grid_w > MAX_CANVAS_SIZE:
        return None

    labels = list(range(1, room_count + 1))
    random.shuffle(labels)
    if labels == list(range(1, room_count + 1)):
        return None

    palette_size = random.randint(2, min(4, len(FIGURE_COLORS)))
    palette = random.sample(FIGURE_COLORS, k=palette_size)
    mode = random.choice(("clustered", "spread"))
    anchor = (
        random.randint(room_h // 2, room_h - 1),
        random.randint(room_w // 2, room_w - 1),
    )

    layers = []
    occupied = np.zeros((room_h, room_w), dtype=int)
    for digit in labels:
        digit_mask = DIGIT_MASKS[digit]
        blocked = np.zeros((room_h, room_w), dtype=bool)
        dh, dw = digit_mask.shape
        blocked[:dh, :dw][digit_mask] = True

        choice = _choose_shape_and_position(room_h, room_w, blocked, occupied, mode, anchor)
        if choice is None:
            return None
        shape, r0, c0 = choice

        layer = np.full((room_h, room_w), BLACK, dtype=int)
        sh, sw = shape.shape
        color = random.choice(palette)
        layer[r0 : r0 + sh, c0 : c0 + sw][shape] = int(color)
        layers.append(layer)
        occupied[r0 : r0 + sh, c0 : c0 + sw][shape] += 1

    expected, overlap_cells, visible_labels = _compose_layers(labels, layers)
    visible_colors = np.unique(expected[expected != BLACK])
    if visible_colors.size < 2:
        return None
    if len(visible_labels) < 2:
        return None
    if np.count_nonzero(expected) < 6:
        return None
    if mode == "clustered" and overlap_cells == 0:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for rr in range(n_rows + 1):
        grid[rr * (room_h + 1), :] = GREEN
    for cc in range(n_cols + 1):
        grid[:, cc * (room_w + 1)] = GREEN

    room_index = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            room = _build_room(room_h, room_w, labels[room_index], layers[room_index])
            r0 = 1 + rr * (room_h + 1)
            c0 = 1 + cc * (room_w + 1)
            grid[r0 : r0 + room_h, c0 : c0 + room_w] = room
            room_index += 1

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
