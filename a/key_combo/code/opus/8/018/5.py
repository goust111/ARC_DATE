from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _label_rooms(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    room_mask = grid != WALL
    labels = np.full(grid.shape, -1, dtype=int)
    rooms: list[list[tuple[int, int]]] = []
    next_id = 0
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not room_mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and room_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            rooms.append(coords)
            next_id += 1
    return labels, rooms


def _room_seed_color(grid: np.ndarray, coords: list[tuple[int, int]]) -> int | None:
    colors = {
        int(grid[r, c])
        for r, c in coords
        if int(grid[r, c]) not in (BG, WALL)
    }
    assert len(colors) <= 1, "each room must contain at most one seed color"
    return None if not colors else next(iter(colors))


def _analyze_rooms(grid: np.ndarray):
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1], "expected a square 2D grid"
    labels, rooms = _label_rooms(grid)
    info = []
    for room_id, coords in enumerate(rooms):
        seed_color = _room_seed_color(grid, coords)
        mirror_ids = {int(labels[c, r]) for r, c in coords}
        assert len(mirror_ids) == 1, "room transpose must map into a unique mirror room"
        mirror_id = next(iter(mirror_ids))
        assert mirror_id >= 0, "mirror room id must exist"
        info.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "seed_color": seed_color,
                "mirror_id": mirror_id,
            }
        )
    return labels, info


def _fill_rooms(grid: np.ndarray, room_info: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for info in room_info:
        color = info["seed_color"]
        if color is None:
            color = room_info[info["mirror_id"]]["seed_color"]
        if color is None:
            continue
        for r, c in info["coords"]:
            if int(out[r, c]) == BG:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-bounded rooms, copying colors into seedless rooms from their main-diagonal mirror rooms.

    Input structure:
    - The grid is square with black background cells and grey one-cell-thick orthogonal walls.
    - The grey walls partition the non-wall area into 4-connected rectangular rooms, and the wall layout is invariant under transposition along the main diagonal.
    - Some rooms contain a single non-black non-grey seed pixel, while other rooms have no seed.

    Rule:
    - Treat grey cells as barriers and identify every 4-connected non-wall room.
    - In a room that contains a seed, fill every black cell of that room with the seed color while leaving the seed pixel unchanged.
    - In a room with no seed, transpose any room coordinate `(r, c)` to `(c, r)` to locate the mirror room, then fill the seedless room's black cells with that mirror room's seed color.
    - Keep all grey walls and all original seed pixels unchanged.

    Key insight:
    - Flood-filling the rooms is necessary to know each room's extent, and diagonal symmetry is necessary because an unseeded room's color is determined only by its transposed partner rather than by local adjacency.
    """
    g = np.asarray(grid, dtype=int)
    _labels, room_info = _analyze_rooms(g)
    return _fill_rooms(g, room_info)


def _partition_lengths() -> list[int]:
    k = random.choice((3, 4, 5))
    for _ in range(80):
        lengths = [random.randint(2, 4) for _ in range(k)]
        size = sum(lengths) + k + 1
        if size <= 26 and len(set(lengths)) >= 2:
            return lengths
    return [2, 3, 4]


def _intervals_from_lengths(lengths: list[int]) -> list[tuple[int, int]]:
    intervals = []
    pos = 1
    for length in lengths:
        intervals.append((pos, pos + length - 1))
        pos += length + 1
    return intervals


def _build_base_grid(lengths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    size = sum(lengths) + len(lengths) + 1
    grid = np.full((size, size), BG, dtype=int)
    wall_lines = [0]
    pos = 0
    for length in lengths:
        pos += length + 1
        wall_lines.append(pos)
    for idx in wall_lines:
        grid[idx, :] = WALL
        grid[:, idx] = WALL
    return grid, _intervals_from_lengths(lengths)


def _pair_seed_mode(i: int, j: int, k: int) -> int:
    return (i + 2 * j + k) % 3


def _room_color_map(k: int) -> dict[tuple[int, int], int]:
    pair_keys = [(i, j) for i in range(k) for j in range(i, k)]
    n_used = random.randint(3, min(6, len(PALETTE)))
    colors = random.sample(PALETTE, n_used)
    mapping = {}
    for key in pair_keys:
        mapping[key] = random.choice(colors)
    return mapping


def _room_anchor(
    bbox: tuple[int, int, int, int],
    room_i: int,
    room_j: int,
    seed_role: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if seed_role == "diag":
        return r0, c0
    if seed_role == "upper":
        return r0, c1
    if seed_role == "lower":
        return r1, c0
    if seed_role == "both_a":
        return r0, c0 if (room_i + room_j) % 2 == 0 else c1
    return r1, c1 if (room_i + room_j) % 2 == 0 else c0


def _room_bbox(intervals: list[tuple[int, int]], i: int, j: int) -> tuple[int, int, int, int]:
    r0, r1 = intervals[i]
    c0, c1 = intervals[j]
    return r0, c0, r1, c1


def _place_seeds(
    grid: np.ndarray,
    intervals: list[tuple[int, int]],
    pair_colors: dict[tuple[int, int], int],
):
    k = len(intervals)
    seeded_rooms: dict[tuple[int, int], int] = {}
    room_targets: dict[tuple[int, int], int] = {}

    for i in range(k):
        for j in range(k):
            room_targets[(i, j)] = pair_colors[(min(i, j), max(i, j))]

    for i in range(k):
        bbox = _room_bbox(intervals, i, i)
        color = room_targets[(i, i)]
        sr, sc = _room_anchor(bbox, i, i, "diag")
        grid[sr, sc] = color
        seeded_rooms[(i, i)] = color

    for i in range(k):
        for j in range(i + 1, k):
            color = room_targets[(i, j)]
            mode = _pair_seed_mode(i, j, k)
            upper_bbox = _room_bbox(intervals, i, j)
            lower_bbox = _room_bbox(intervals, j, i)
            if mode in (0, 2):
                sr, sc = _room_anchor(upper_bbox, i, j, "upper" if mode == 0 else "both_a")
                grid[sr, sc] = color
                seeded_rooms[(i, j)] = color
            if mode in (1, 2):
                sr, sc = _room_anchor(lower_bbox, j, i, "lower" if mode == 1 else "both_b")
                grid[sr, sc] = color
                seeded_rooms[(j, i)] = color

    return seeded_rooms, room_targets


def _seedless_rooms(k: int) -> list[tuple[int, int]]:
    out = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            mode = _pair_seed_mode(min(i, j), max(i, j), k)
            if i < j and mode == 1:
                out.append((i, j))
            elif i > j and mode == 0:
                out.append((i, j))
    return out


def _adjacent_rooms(k: int, room: tuple[int, int]) -> list[tuple[int, int]]:
    i, j = room
    out = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni = i + di
        nj = j + dj
        if 0 <= ni < k and 0 <= nj < k:
            out.append((ni, nj))
    return out


def _room_center(intervals: list[tuple[int, int]], room: tuple[int, int]) -> tuple[float, float]:
    bbox = _room_bbox(intervals, room[0], room[1])
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _nearest_seed_color(
    intervals: list[tuple[int, int]],
    seeded_rooms: dict[tuple[int, int], int],
    room: tuple[int, int],
) -> int:
    cr, cc = _room_center(intervals, room)
    best = None
    best_color = None
    for seeded_room, color in seeded_rooms.items():
        sr, sc = _room_center(intervals, seeded_room)
        dist = abs(cr - sr) + abs(cc - sc)
        key = (dist, seeded_room[0], seeded_room[1])
        if best is None or key < best:
            best = key
            best_color = color
    assert best_color is not None
    return int(best_color)


def _seeded_only_output(grid: np.ndarray) -> np.ndarray:
    _labels, room_info = _analyze_rooms(grid)
    out = np.array(grid, copy=True)
    for info in room_info:
        color = info["seed_color"]
        if color is None:
            continue
        for r, c in info["coords"]:
            if int(out[r, c]) == BG:
                out[r, c] = int(color)
    return out


def _nearest_seed_output(
    grid: np.ndarray,
    intervals: list[tuple[int, int]],
    seeded_rooms: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    k = len(intervals)
    for i in range(k):
        for j in range(k):
            bbox = _room_bbox(intervals, i, j)
            color = seeded_rooms.get((i, j))
            if color is None:
                color = _nearest_seed_color(intervals, seeded_rooms, (i, j))
            r0, c0, r1, c1 = bbox
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if int(out[r, c]) == BG:
                        out[r, c] = int(color)
    return out


def _expected_from_targets(
    grid: np.ndarray,
    intervals: list[tuple[int, int]],
    room_targets: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    k = len(intervals)
    for i in range(k):
        for j in range(k):
            color = room_targets[(i, j)]
            r0, c0, r1, c1 = _room_bbox(intervals, i, j)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if int(out[r, c]) == BG:
                        out[r, c] = int(color)
    return out


def _nondegenerate_layout(
    grid: np.ndarray,
    intervals: list[tuple[int, int]],
    seeded_rooms: dict[tuple[int, int], int],
    room_targets: dict[tuple[int, int], int],
) -> bool:
    k = len(intervals)
    seedless = _seedless_rooms(k)
    if len(seedless) < 2:
        return False

    target_colors = {room_targets[room] for room in seedless}
    if len(target_colors) < 2:
        return False

    has_adjacent_counterexample = False
    has_nearest_counterexample = False
    for room in seedless:
        target = room_targets[room]
        adj_seed_colors = {seeded_rooms[nbr] for nbr in _adjacent_rooms(k, room) if nbr in seeded_rooms}
        if adj_seed_colors and target not in adj_seed_colors:
            has_adjacent_counterexample = True
        if _nearest_seed_color(intervals, seeded_rooms, room) != target:
            has_nearest_counterexample = True

    if not has_adjacent_counterexample:
        return False
    if not has_nearest_counterexample:
        return False

    both_seeded_pairs = 0
    one_seeded_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            a = (i, j) in seeded_rooms
            b = (j, i) in seeded_rooms
            if a and b:
                both_seeded_pairs += 1
            elif a or b:
                one_seeded_pairs += 1
    if both_seeded_pairs == 0 or one_seeded_pairs == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    lengths = _partition_lengths()
    grid, intervals = _build_base_grid(lengths)
    k = len(intervals)
    pair_colors = _room_color_map(k)
    seeded_rooms, room_targets = _place_seeds(grid, intervals, pair_colors)

    if not _nondegenerate_layout(grid, intervals, seeded_rooms, room_targets):
        return None

    expected = _expected_from_targets(grid, intervals, room_targets)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(_seeded_only_output(grid), expected):
        return None
    if np.array_equal(_nearest_seed_output(grid, intervals, seeded_rooms), expected):
        return None

    # Ensure the final grid really uses mirror propagation in at least two colors.
    if len({room_targets[room] for room in _seedless_rooms(k)}) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-room propagation input")
