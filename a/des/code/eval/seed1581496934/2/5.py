import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIDE_TO_ROTATION = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_tower_patterns(grid: np.ndarray) -> list[np.ndarray]:
    maroon = np.argwhere(grid == MAROON)
    assert maroon.size > 0, "tower border is missing"
    r0, c0, r1, c1 = _bbox(maroon)
    tower = grid[r0 : r1 + 1, c0 : c1 + 1]

    assert tower.shape[0] >= 5 and tower.shape[1] >= 3
    assert np.all(tower[:, 0] == MAROON)
    assert np.all(tower[:, -1] == MAROON)

    sep_rows = [r for r in range(tower.shape[0]) if np.all(tower[r] == MAROON)]
    assert len(sep_rows) >= 2
    assert sep_rows[0] == 0 and sep_rows[-1] == tower.shape[0] - 1

    gaps = [sep_rows[i + 1] - sep_rows[i] - 1 for i in range(len(sep_rows) - 1)]
    assert min(gaps) > 0 and len(set(gaps)) == 1

    inner_w = tower.shape[1] - 2
    patterns = []
    for upper, lower in zip(sep_rows, sep_rows[1:]):
        layer = tower[upper + 1 : lower, 1:-1].copy()
        assert layer.shape[1] == inner_w
        patterns.append(layer)
    return patterns


def _parse_room_patch(patch: np.ndarray) -> tuple[str, int] | None:
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return None
    if not (
        patch[0, 0] == GREY
        and patch[0, -1] == GREY
        and patch[-1, 0] == GREY
        and patch[-1, -1] == GREY
    ):
        return None
    if not np.all(patch[1:-1, 1:-1] == BLACK):
        return None

    edges = {
        "top": patch[0, 1:-1],
        "right": patch[1:-1, -1],
        "bottom": patch[-1, 1:-1],
        "left": patch[1:-1, 0],
    }
    counts = {side: int(np.sum(edge == BLACK)) for side, edge in edges.items()}
    nonzero = [side for side, count in counts.items() if count > 0]
    if len(nonzero) != 1:
        return None

    side = nonzero[0]
    for name, edge in edges.items():
        if name == side:
            if not np.all((edge == GREY) | (edge == BLACK)):
                return None
        elif not np.all(edge == GREY):
            return None
    return side, counts[side]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_rooms(grid: np.ndarray, inner_shape: tuple[int, int]) -> list[dict[str, object]]:
    inner_h, inner_w = inner_shape
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    h, w = grid.shape
    rooms = []
    seen = set()

    for r0 in range(h - outer_h + 1):
        for c0 in range(w - outer_w + 1):
            patch = grid[r0 : r0 + outer_h, c0 : c0 + outer_w]
            parsed = _parse_room_patch(patch)
            if parsed is None:
                continue
            side, count = parsed
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if bbox in seen:
                continue
            seen.add(bbox)
            rooms.append({"bbox": bbox, "side": side, "window_count": count})

    rooms.sort(key=lambda room: tuple(room["bbox"]))  # type: ignore[arg-type]
    assert rooms, "no rooms found"
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            assert not _boxes_overlap(
                rooms[i]["bbox"],  # type: ignore[arg-type]
                rooms[j]["bbox"],  # type: ignore[arg-type]
            )
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey room with the tower layer selected by its window count,
    rotated by the side where those windows appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    patterns = _extract_tower_patterns(g)
    room_shape = tuple(int(x) for x in patterns[0].shape)
    out = g.copy()

    for room in _find_rooms(g, room_shape):
        count = int(room["window_count"])
        assert 1 <= count <= len(patterns)
        pattern = patterns[count - 1]
        rotated = np.rot90(pattern, SIDE_TO_ROTATION[str(room["side"])])
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[misc]
        assert rotated.shape == (r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


def _random_pattern(size: int, existing: list[np.ndarray], rng: random.Random) -> np.ndarray:
    for _ in range(300):
        n_colors = rng.randint(2, 3)
        colors = rng.sample(PATTERN_COLORS, n_colors)
        weights = [0.52, 0.48] if n_colors == 2 else [0.46, 0.34, 0.20]
        flat = rng.choices(colors, weights=weights, k=size * size)
        pattern = np.array(flat, dtype=int).reshape(size, size)

        if any(np.count_nonzero(pattern == color) == 0 for color in colors):
            continue
        if len(np.unique(pattern, axis=0)) < 2 or len(np.unique(pattern, axis=1)) < 2:
            continue
        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue

        duplicate = False
        for prev in existing:
            if any(np.array_equal(pattern, np.rot90(prev, k)) for k in range(4)):
                duplicate = True
                break
        if duplicate:
            continue
        return pattern
    raise RuntimeError("failed to sample a non-degenerate pattern")


def _window_positions(length: int, count: int, rng: random.Random) -> list[int]:
    if count >= length:
        return list(range(length))

    spread_options = []
    for start in (0, 1):
        slots = list(range(start, length, 2))
        if len(slots) >= count:
            spread_options.append(slots)
    if spread_options and rng.random() < 0.7:
        return sorted(rng.sample(rng.choice(spread_options), count))
    return sorted(rng.sample(range(length), count))


def _draw_tower(grid: np.ndarray, top: int, left: int, patterns: list[np.ndarray]) -> None:
    size = int(patterns[0].shape[0])
    width = size + 2
    grid[top : top + len(patterns) * (size + 1) + 1, left] = MAROON
    grid[top : top + len(patterns) * (size + 1) + 1, left + width - 1] = MAROON
    for i, pattern in enumerate(patterns):
        row = top + i * (size + 1)
        grid[row, left : left + width] = MAROON
        grid[row + 1 : row + 1 + size, left + 1 : left + 1 + size] = pattern
    grid[top + len(patterns) * (size + 1), left : left + width] = MAROON


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    size: int,
    side: str,
    windows: list[int],
) -> tuple[int, int, int, int]:
    bottom = top + size + 1
    right = left + size + 1
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY

    if side == "top":
        for pos in windows:
            grid[top, left + 1 + pos] = BLACK
    elif side == "right":
        for pos in windows:
            grid[top + 1 + pos, right] = BLACK
    elif side == "bottom":
        for pos in windows:
            grid[bottom, left + 1 + pos] = BLACK
    else:
        for pos in windows:
            grid[top + 1 + pos, left] = BLACK
    return top, left, bottom, right


def _expected_output(grid: np.ndarray, patterns: list[np.ndarray], rooms: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        pattern = patterns[int(room["window_count"]) - 1]
        rotated = np.rot90(pattern, SIDE_TO_ROTATION[str(room["side"])])
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[misc]
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_layers = rng.randint(2, 4)
    size = rng.randint(max(3, n_layers + 1), 5)
    room_count = rng.randint(max(4, n_layers), 6)

    patterns: list[np.ndarray] = []
    for _ in range(n_layers):
        patterns.append(_random_pattern(size, patterns, rng))

    count_seq = [(i % n_layers) + 1 for i in range(room_count)]
    side_seq = ["top", "right", "bottom", "left"] * ((room_count + 3) // 4)
    count_seq = count_seq[:room_count]
    side_seq = side_seq[:room_count]
    rng.shuffle(count_seq)
    rng.shuffle(side_seq)

    outer = size + 2
    n_cols = 2
    n_rows = (room_count + n_cols - 1) // n_cols
    tower_h = n_layers * size + n_layers + 1
    tower_w = outer
    room_gap_x = rng.randint(3, 5)
    room_gap_y = rng.randint(2, 4)
    section_gap = rng.randint(4, 6)
    tower_top = rng.randint(1, 3)
    room_top = rng.randint(1, 3)
    tower_left = rng.randint(1, 3)
    room_left = tower_left + tower_w + section_gap

    grid_h = max(
        tower_top + tower_h,
        room_top + n_rows * outer + (n_rows - 1) * room_gap_y,
    ) + rng.randint(1, 3)
    grid_w = room_left + n_cols * outer + (n_cols - 1) * room_gap_x + rng.randint(1, 3)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    _draw_tower(grid, tower_top, tower_left, patterns)

    positions = []
    for idx in range(room_count):
        rr = idx // n_cols
        cc = idx % n_cols
        top = room_top + rr * (outer + room_gap_y)
        left = room_left + cc * (outer + room_gap_x)
        positions.append((top, left))
    rng.shuffle(positions)

    rooms = []
    for (top, left), count, side in zip(positions, count_seq, side_seq):
        windows = _window_positions(size, count, rng)
        bbox = _draw_room(grid, top, left, size, side, windows)
        rooms.append({"bbox": bbox, "side": side, "window_count": count})

    expected = _expected_output(grid, patterns, rooms)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
