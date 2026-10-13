import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 1
MARKER = 2
PATTERN_COLORS = (3, 4, 5, 6, 7, 8, 9)

GEN_RETRIES = 400
DOOR_LAYOUT_RETRIES = 200


def _compress_indices(indices: Iterable[int]) -> list[int]:
    values = sorted(int(v) for v in indices)
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [group[len(group) // 2] for group in groups]


def _dense_line_indices(counts: np.ndarray) -> list[int]:
    positive = sorted({int(v) for v in counts if int(v) > 0})
    if not positive:
        return []
    if len(positive) == 1:
        threshold = positive[0]
    else:
        gaps = [positive[i + 1] - positive[i] for i in range(len(positive) - 1)]
        split = max(range(len(gaps)), key=lambda i: gaps[i])
        threshold = positive[split] + 1
    return [int(i) for i, v in enumerate(counts) if int(v) >= threshold]


def _marker_direction(grid: np.ndarray) -> str:
    h, w = grid.shape
    corners = {
        (0, 0): "tl",
        (0, w - 1): "tr",
        (h - 1, 0): "bl",
        (h - 1, w - 1): "br",
    }
    for (r, c), label in corners.items():
        if int(grid[r, c]) == MARKER:
            return label
    return "tl"


def _position_key(room: dict, direction: str) -> tuple[int, int]:
    r0, c0, _, _ = room["interior"]
    if direction == "tr":
        return (r0, -c0)
    if direction == "bl":
        return (-r0, c0)
    if direction == "br":
        return (-r0, -c0)
    return (r0, c0)


def _pattern_from_room(grid: np.ndarray, room: dict) -> dict:
    r0, c0, r1, c1 = room["interior"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (patch != BG) & (patch != WALL) & (patch != MARKER)
    coords = np.argwhere(mask)
    assert coords.size > 0
    colors = {int(patch[r, c]) for r, c in coords}
    assert len(colors) == 1
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    rel = tuple(sorted((int(r - pr0), int(c - pc0)) for r, c in coords))
    return {
        "color": int(next(iter(colors))),
        "shape": (int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)),
        "coords": rel,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign room decorations after sorting rooms by how many door gaps they have.

    The red corner marker chooses the positional scan direction used to break ties:
    top-left, top-right, bottom-left, or bottom-right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue = g == WALL
    h_lines = _compress_indices(_dense_line_indices(blue.sum(axis=1)))
    v_lines = _compress_indices(_dense_line_indices(blue.sum(axis=0)))
    assert len(h_lines) >= 2 and len(v_lines) >= 2

    rooms: list[dict] = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            r0, r1 = h_lines[i] + 1, h_lines[i + 1] - 1
            c0, c1 = v_lines[j] + 1, v_lines[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            rooms.append(
                {
                    "row_band": i,
                    "col_band": j,
                    "interior": (r0, c0, r1, c1),
                }
            )

    assert rooms
    row_count = len(h_lines) - 1
    col_count = len(v_lines) - 1

    for room in rooms:
        i = int(room["row_band"])
        j = int(room["col_band"])
        r0, c0, r1, c1 = room["interior"]
        door_count = 0
        if i > 0:
            door_count += int(np.count_nonzero(g[h_lines[i], c0 : c1 + 1] != WALL))
        if i < row_count - 1:
            door_count += int(np.count_nonzero(g[h_lines[i + 1], c0 : c1 + 1] != WALL))
        if j > 0:
            door_count += int(np.count_nonzero(g[r0 : r1 + 1, v_lines[j]] != WALL))
        if j < col_count - 1:
            door_count += int(np.count_nonzero(g[r0 : r1 + 1, v_lines[j + 1]] != WALL))
        room["door_count"] = int(door_count)
        room["pattern"] = _pattern_from_room(g, room)

    direction = _marker_direction(g)
    ordered = sorted(rooms, key=lambda room: (int(room["door_count"]), _position_key(room, direction)))
    reversed_patterns = [room["pattern"] for room in ordered][::-1]

    out = np.where((g == WALL) | (g == MARKER), g, BG)
    for room, pattern in zip(ordered, reversed_patterns):
        r0, c0, r1, c1 = room["interior"]
        room_h = int(r1 - r0 + 1)
        room_w = int(c1 - c0 + 1)
        pat_h, pat_w = pattern["shape"]
        assert pat_h <= room_h and pat_w <= room_w
        start_r = int(r0 + (room_h - pat_h) // 2)
        start_c = int(c0 + (room_w - pat_w) // 2)
        for dr, dc in pattern["coords"]:
            out[start_r + dr, start_c + dc] = int(pattern["color"])
    return out


def _line_positions(lengths: list[int], start: int) -> list[int]:
    lines = [int(start)]
    cursor = int(start)
    for length in lengths:
        cursor += int(length) + 1
        lines.append(int(cursor))
    return lines


def _door_slots(start: int, end: int, rng: random.Random) -> list[int]:
    if start > end:
        return []
    parity = rng.randrange(2)
    slots = list(range(start + parity, end + 1, 2))
    if not slots:
        slots = list(range(start, end + 1, 2))
    if not slots:
        slots = [start]
    return slots


def _sample_door_layout(
    row_sizes: list[int], col_sizes: list[int], h_lines: list[int], v_lines: list[int], rng: random.Random
) -> tuple[dict[tuple[int, int], tuple[int, ...]], dict[tuple[int, int], tuple[int, ...]], np.ndarray]:
    nr = len(row_sizes)
    nc = len(col_sizes)
    counts = np.zeros((nr, nc), dtype=int)
    horizontal: dict[tuple[int, int], tuple[int, ...]] = {}
    vertical: dict[tuple[int, int], tuple[int, ...]] = {}

    for i in range(nr - 1):
        row = h_lines[i + 1]
        for j in range(nc):
            slots = _door_slots(v_lines[j] + 1, v_lines[j + 1] - 1, rng)
            max_doors = min(3, len(slots))
            k = rng.choices(range(max_doors + 1), weights=(4, 4, 3, 2)[: max_doors + 1], k=1)[0]
            chosen = tuple(sorted(rng.sample(slots, k)))
            horizontal[(i, j)] = chosen
            counts[i, j] += k
            counts[i + 1, j] += k
            assert row >= 0

    for i in range(nr):
        for j in range(nc - 1):
            slots = _door_slots(h_lines[i] + 1, h_lines[i + 1] - 1, rng)
            max_doors = min(3, len(slots))
            k = rng.choices(range(max_doors + 1), weights=(4, 4, 3, 2)[: max_doors + 1], k=1)[0]
            chosen = tuple(sorted(rng.sample(slots, k)))
            vertical[(i, j)] = chosen
            counts[i, j] += k
            counts[i, j + 1] += k

    return horizontal, vertical, counts


def _rotate_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    current = shape
    for _ in range(4):
        if not any(np.array_equal(current, seen) for seen in variants):
            variants.append(current.copy())
        current = np.rot90(current)
    return variants


def _shape_catalog(max_h: int, max_w: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []

    def add(shape: np.ndarray) -> None:
        if shape.shape[0] <= max_h and shape.shape[1] <= max_w:
            if not any(np.array_equal(shape, seen) for seen in shapes):
                shapes.append(shape.copy())

    add(np.array([[1]], dtype=bool))
    add(np.ones((1, 3), dtype=bool))
    add(np.ones((3, 1), dtype=bool))
    add(np.ones((1, 5), dtype=bool))
    add(np.ones((5, 1), dtype=bool))

    plus3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    ring3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    tee3 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool)
    elbow3 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    zig3 = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
    cross35 = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool)
    cross53 = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool)
    plus5 = np.zeros((5, 5), dtype=bool)
    plus5[2, :] = True
    plus5[:, 2] = True

    for base in (plus3, ring3, np.ones((3, 3), dtype=bool), tee3, elbow3, zig3, cross35, cross53, plus5):
        for variant in _rotate_variants(base):
            add(variant)

    return shapes


def _sample_offset(room_size: int, shape_size: int, rng: random.Random) -> int:
    slack = int(room_size - shape_size)
    if slack <= 0:
        return 0
    if slack >= 2:
        return rng.randint(1, slack - 1)
    return rng.randint(0, slack)


def _build_rooms(h_lines: list[int], v_lines: list[int]) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            rooms.append((h_lines[i] + 1, v_lines[j] + 1, h_lines[i + 1] - 1, v_lines[j + 1] - 1))
    return rooms


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_tie = rng.random() < 0.5
    layouts = [(2, 2), (2, 3), (3, 2)]
    nr, nc = (2, 2) if not want_tie else rng.choice(layouts)

    if want_tie:
        row_sizes = [rng.choice((5, 7)) for _ in range(nr)]
        col_sizes = [rng.choice((5, 7)) for _ in range(nc)]
    else:
        row_sizes = [7] * nr
        col_sizes = [7] * nc

    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    left_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)

    h_lines = _line_positions(row_sizes, top_margin)
    v_lines = _line_positions(col_sizes, left_margin)
    height = h_lines[-1] + bottom_margin + 1
    width = v_lines[-1] + right_margin + 1
    if height > 30 or width > 30:
        return None

    horizontal_doors = {}
    vertical_doors = {}
    counts = None
    for _ in range(DOOR_LAYOUT_RETRIES):
        horizontal_doors, vertical_doors, counts = _sample_door_layout(row_sizes, col_sizes, h_lines, v_lines, rng)
        flat = [int(v) for v in counts.flatten()]
        unique_count = len(set(flat))
        if max(flat) == 0 or unique_count < 2:
            continue
        if want_tie and unique_count == len(flat):
            continue
        if not want_tie and unique_count != len(flat):
            continue
        break
    else:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r in h_lines:
        grid[r, v_lines[0] : v_lines[-1] + 1] = WALL
    for c in v_lines:
        grid[h_lines[0] : h_lines[-1] + 1, c] = WALL

    for (i, j), cols in horizontal_doors.items():
        row = h_lines[i + 1]
        for col in cols:
            grid[row, col] = BG
    for (i, j), rows in vertical_doors.items():
        col = v_lines[j + 1]
        for row in rows:
            grid[row, col] = BG

    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        grid[0, 0] = MARKER
    elif corner == "tr":
        grid[0, width - 1] = MARKER
    elif corner == "bl":
        grid[height - 1, 0] = MARKER
    else:
        grid[height - 1, width - 1] = MARKER

    min_room_h = min(row_sizes)
    min_room_w = min(col_sizes)
    shapes = _shape_catalog(min_room_h - 2, min_room_w - 2)
    if not shapes:
        return None

    rooms = _build_rooms(h_lines, v_lines)
    colors = rng.sample(PATTERN_COLORS, len(rooms))
    for (r0, c0, r1, c1), color in zip(rooms, colors):
        shape = rng.choice(shapes)
        sh, sw = shape.shape
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        if sh > room_h or sw > room_w:
            return None
        off_r = _sample_offset(room_h, sh, rng)
        off_c = _sample_offset(room_w, sw, rng)
        target = grid[r0 + off_r : r0 + off_r + sh, c0 + off_c : c0 + off_c + sw]
        target[shape] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
