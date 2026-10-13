import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


BG = 0
ARROW_COLOR = 4
GEN_ATTEMPTS = 300
PATTERN_COLORS = (1, 2, 3, 5, 6, 7, 8)

ROOM_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_SHAPES = {
    "up": ((0, 0), (-1, -1), (-1, 1)),
    "down": ((0, 0), (1, -1), (1, 1)),
    "left": ((0, 0), (-1, -1), (1, -1)),
    "right": ((0, 0), (-1, 1), (1, 1)),
}

# Incoming-to-target priority is: from above, from left, from below, from right.
# Those correspond to source arrow directions: down, right, up, left.
OVERWRITE_ORDER = ("left", "up", "right", "down")


@dataclass
class Room:
    room_r: int
    room_c: int
    r0: int
    c0: int
    height: int
    width: int
    direction: str | None
    arrow_mask: np.ndarray
    pattern: np.ndarray


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _is_regular_partition(indices: list[int], last_index: int) -> bool:
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != last_index:
        return False
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    return len(set(gaps)) == 1 and gaps[0] >= 2


def _detect_wall_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in map(int, np.unique(grid)):
        rows = _full_line_indices(grid, color, axis=0)
        cols = _full_line_indices(grid, color, axis=1)
        if not _is_regular_partition(rows, grid.shape[0] - 1):
            continue
        if not _is_regular_partition(cols, grid.shape[1] - 1):
            continue
        score = len(rows) * len(cols)
        candidate = (score, color, rows, cols)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise AssertionError("failed to detect regular room walls")
    return best[1], best[2], best[3]


def _detect_arrow(interior: np.ndarray) -> tuple[str | None, np.ndarray]:
    yellow = np.argwhere(interior == ARROW_COLOR)
    if yellow.size == 0:
        return None, np.zeros(interior.shape, dtype=bool)
    if yellow.shape[0] != 3:
        raise AssertionError("each room must contain either zero or exactly three yellow arrow pixels")
    pts = [tuple(map(int, rc)) for rc in yellow]
    pset = set(pts)
    for center_r, center_c in pts:
        for direction, offsets in ARROW_SHAPES.items():
            expected = {(center_r + dr, center_c + dc) for dr, dc in offsets}
            if expected == pset:
                mask = np.zeros(interior.shape, dtype=bool)
                for r, c in expected:
                    mask[r, c] = True
                return direction, mask
    raise AssertionError("yellow pixels do not form a valid arrow")


def _extract_rooms(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    wall_color, wall_rows, wall_cols = _detect_wall_color(g)
    room_h = wall_rows[1] - wall_rows[0] - 1
    room_w = wall_cols[1] - wall_cols[0] - 1
    if room_h <= 0 or room_w <= 0:
        raise AssertionError("rooms must have positive interior size")

    room_map: dict[tuple[int, int], Room] = {}
    for room_r in range(len(wall_rows) - 1):
        for room_c in range(len(wall_cols) - 1):
            r0 = wall_rows[room_r] + 1
            r1 = wall_rows[room_r + 1]
            c0 = wall_cols[room_c] + 1
            c1 = wall_cols[room_c + 1]
            interior = g[r0:r1, c0:c1]
            if interior.shape != (room_h, room_w):
                raise AssertionError("all rooms must share one interior size")
            direction, arrow_mask = _detect_arrow(interior)
            pattern = interior.copy()
            pattern[(pattern == BG) | (pattern == ARROW_COLOR)] = BG
            room_map[(room_r, room_c)] = Room(
                room_r=room_r,
                room_c=room_c,
                r0=r0,
                c0=c0,
                height=room_h,
                width=room_w,
                direction=direction,
                arrow_mask=arrow_mask,
                pattern=pattern,
            )

    return {
        "grid": g,
        "wall_color": wall_color,
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "room_h": room_h,
        "room_w": room_w,
        "room_rows": len(wall_rows) - 1,
        "room_cols": len(wall_cols) - 1,
        "rooms": room_map,
    }


def _incoming_projections(parsed: dict) -> dict[tuple[int, int], list[tuple[str, np.ndarray]]]:
    incoming: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    room_rows = parsed["room_rows"]
    room_cols = parsed["room_cols"]
    for (room_r, room_c), room in parsed["rooms"].items():
        if room.direction is None:
            continue
        dr, dc = ROOM_DELTAS[room.direction]
        trg = (room_r + dr, room_c + dc)
        if 0 <= trg[0] < room_rows and 0 <= trg[1] < room_cols:
            incoming[trg].append((room.direction, room.pattern.copy()))
    return incoming


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each arrow room's non-arrow pattern into the adjacent room indicated by
    the yellow arrow, then clear the source pattern while keeping walls and arrows.
    Incoming overlaps are resolved with fixed priority: from above, from left,
    from below, then from right.
    """
    parsed = _extract_rooms(grid)
    g = parsed["grid"]
    out = g.copy()

    for room in parsed["rooms"].values():
        if room.direction is None:
            continue
        interior = out[room.r0 : room.r0 + room.height, room.c0 : room.c0 + room.width]
        interior[room.pattern != BG] = BG

    incoming = _incoming_projections(parsed)
    for target_idx, items in incoming.items():
        target = parsed["rooms"][target_idx]
        interior = out[target.r0 : target.r0 + target.height, target.c0 : target.c0 + target.width]
        by_direction = {direction: pattern for direction, pattern in items}
        for direction in OVERWRITE_ORDER:
            pattern = by_direction.get(direction)
            if pattern is None:
                continue
            write_mask = pattern != BG
            if target.direction is not None:
                write_mask &= ~target.arrow_mask
            interior[write_mask] = pattern[write_mask]

    return out


def _room_origin(room_r: int, room_c: int, room_h: int, room_w: int) -> tuple[int, int]:
    return 1 + room_r * (room_h + 1), 1 + room_c * (room_w + 1)


def _build_empty_grid(room_rows: int, room_cols: int, room_h: int, room_w: int, wall_color: int) -> np.ndarray:
    height = room_rows * room_h + room_rows + 1
    width = room_cols * room_w + room_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    wall_rows = [r * (room_h + 1) for r in range(room_rows + 1)]
    wall_cols = [c * (room_w + 1) for c in range(room_cols + 1)]
    grid[wall_rows, :] = wall_color
    grid[:, wall_cols] = wall_color
    return grid


def _arrow_mask_for_direction(room_h: int, room_w: int, direction: str, rng: random.Random) -> np.ndarray:
    candidates = []
    for center_r in range(room_h):
        for center_c in range(room_w):
            pts = [(center_r + dr, center_c + dc) for dr, dc in ARROW_SHAPES[direction]]
            if all(0 <= r < room_h and 0 <= c < room_w for r, c in pts):
                candidates.append(pts)
    if not candidates:
        raise AssertionError("room too small for arrow placement")
    chosen = rng.choice(candidates)
    mask = np.zeros((room_h, room_w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _neighbors4(cell: tuple[int, int], room_h: int, room_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < room_h and 0 <= nc < room_w:
            out.append((nr, nc))
    return out


def _choose_connected_cells(
    available: set[tuple[int, int]],
    room_h: int,
    room_w: int,
    rng: random.Random,
    min_size: int,
    max_size: int,
) -> set[tuple[int, int]]:
    if len(available) < min_size:
        raise AssertionError("not enough free cells for a connected pattern")
    target_size = rng.randint(min_size, min(max_size, len(available)))
    available_list = list(available)
    for _ in range(80):
        start = rng.choice(available_list)
        shape = {start}
        while len(shape) < target_size:
            frontier = []
            for cell in shape:
                frontier.extend(_neighbors4(cell, room_h, room_w))
            candidates = [cell for cell in frontier if cell in available and cell not in shape]
            if not candidates:
                break
            shape.add(rng.choice(candidates))
        if len(shape) >= min_size:
            return shape
    raise AssertionError("failed to sample a connected pattern")


def _extend_shape(
    base: set[tuple[int, int]],
    available: set[tuple[int, int]],
    room_h: int,
    room_w: int,
    rng: random.Random,
    max_extra: int,
) -> set[tuple[int, int]]:
    shape = set(base)
    extra_target = rng.randint(0, min(max_extra, len(available - shape)))
    while len(shape) < len(base) + extra_target:
        frontier = []
        for cell in shape:
            frontier.extend(_neighbors4(cell, room_h, room_w))
        candidates = [cell for cell in frontier if cell in available and cell not in shape]
        if not candidates:
            candidates = [cell for cell in available if cell not in shape]
        if not candidates:
            break
        shape.add(rng.choice(candidates))
    return shape


def _paint_cells(pattern: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        pattern[r, c] = color


def _random_room_pattern(
    room_h: int,
    room_w: int,
    arrow_mask: np.ndarray,
    rng: random.Random,
    color: int,
    min_size: int = 1,
    max_size: int = 4,
) -> np.ndarray:
    available = {
        (r, c)
        for r in range(room_h)
        for c in range(room_w)
        if not arrow_mask[r, c]
    }
    cells = _choose_connected_cells(available, room_h, room_w, rng, min_size=min_size, max_size=max_size)
    pattern = np.full((room_h, room_w), BG, dtype=int)
    _paint_cells(pattern, cells, color)
    return pattern


def _valid_directions(room_r: int, room_c: int, room_rows: int, room_cols: int) -> list[str]:
    valid = []
    for direction, (dr, dc) in ROOM_DELTAS.items():
        nr = room_r + dr
        nc = room_c + dc
        if 0 <= nr < room_rows and 0 <= nc < room_cols:
            valid.append(direction)
    return valid


def _paint_room(
    grid: np.ndarray,
    room_r: int,
    room_c: int,
    room_h: int,
    room_w: int,
    pattern: np.ndarray,
    arrow_mask: np.ndarray,
) -> None:
    r0, c0 = _room_origin(room_r, room_c, room_h, room_w)
    interior = grid[r0 : r0 + room_h, c0 : c0 + room_w]
    write_mask = pattern != BG
    interior[write_mask] = pattern[write_mask]
    interior[arrow_mask] = ARROW_COLOR


def _validate_generated(grid: np.ndarray) -> None:
    parsed = _extract_rooms(grid)
    directions = {room.direction for room in parsed["rooms"].values() if room.direction is not None}
    if directions != set(ROOM_DELTAS):
        raise AssertionError("generator must expose all four arrow directions")
    if not any(room.direction is None for room in parsed["rooms"].values()):
        raise AssertionError("generator must keep at least one stationary room")

    incoming = _incoming_projections(parsed)
    if not any(len(items) >= 2 for items in incoming.values()):
        raise AssertionError("generator must create at least one multi-source target")

    contested = False
    overwrite_stationary = False
    for target_idx, items in incoming.items():
        target = parsed["rooms"][target_idx]
        occupancy = np.zeros((target.height, target.width), dtype=int)
        for _direction, pattern in items:
            occupancy += (pattern != BG).astype(int)
        contested |= bool(np.any(occupancy >= 2))
        overwrite_stationary |= bool(
            target.direction is None
            and np.any((target.pattern != BG) & (occupancy >= 1))
        )
    if not contested:
        raise AssertionError("generator must produce at least one true overlap conflict")
    if not overwrite_stationary:
        raise AssertionError("generator must let some projection overwrite a stationary pattern")

    out = main(grid)
    if np.array_equal(out, grid):
        raise AssertionError("generated task must change under main()")


def _gen_attempt(rng: random.Random) -> np.ndarray:
    room_rows = rng.randint(3, 5)
    room_cols = rng.randint(3, 5)
    room_h = rng.randint(5, 7)
    room_w = rng.randint(5, 7)
    wall_color = 9
    grid = _build_empty_grid(room_rows, room_cols, room_h, room_w, wall_color)

    room_dirs: dict[tuple[int, int], str | None] = {}
    arrow_masks: dict[tuple[int, int], np.ndarray] = {}
    patterns: dict[tuple[int, int], np.ndarray] = {}

    target = (rng.randrange(1, room_rows - 1), rng.randrange(1, room_cols - 1))
    required_sources = {
        (target[0] - 1, target[1]): "down",
        (target[0] + 1, target[1]): "up",
        (target[0], target[1] - 1): "right",
        (target[0], target[1] + 1): "left",
    }

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            room_dirs[(room_r, room_c)] = None
            arrow_masks[(room_r, room_c)] = np.zeros((room_h, room_w), dtype=bool)
            patterns[(room_r, room_c)] = np.full((room_h, room_w), BG, dtype=int)

    for idx, direction in required_sources.items():
        room_dirs[idx] = direction
        arrow_masks[idx] = _arrow_mask_for_direction(room_h, room_w, direction, rng)

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            idx = (room_r, room_c)
            if idx == target or idx in required_sources:
                continue
            if rng.random() < 0.4:
                direction = rng.choice(_valid_directions(room_r, room_c, room_rows, room_cols))
                room_dirs[idx] = direction
                arrow_masks[idx] = _arrow_mask_for_direction(room_h, room_w, direction, rng)

    free_for_conflict = None
    for idx in required_sources:
        free = {
            (r, c)
            for r in range(room_h)
            for c in range(room_w)
            if not arrow_masks[idx][r, c]
        }
        free_for_conflict = free if free_for_conflict is None else free_for_conflict & free
    if free_for_conflict is None:
        raise AssertionError("missing required source rooms")
    shared_cells = _choose_connected_cells(
        free_for_conflict,
        room_h,
        room_w,
        rng,
        min_size=2,
        max_size=3,
    )

    central_colors = list(rng.sample(PATTERN_COLORS, 5))
    source_items = list(required_sources.items())
    rng.shuffle(source_items)
    for idx, color in zip((item[0] for item in source_items), central_colors[:4]):
        available = {
            (r, c)
            for r in range(room_h)
            for c in range(room_w)
            if not arrow_masks[idx][r, c]
        }
        cells = _extend_shape(shared_cells, available, room_h, room_w, rng, max_extra=2)
        _paint_cells(patterns[idx], cells, color)

    target_base = set(rng.sample(sorted(shared_cells), rng.randint(1, len(shared_cells))))
    available_target = {(r, c) for r in range(room_h) for c in range(room_w)}
    target_cells = _extend_shape(target_base, available_target, room_h, room_w, rng, max_extra=2)
    _paint_cells(patterns[target], target_cells, central_colors[4])

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            idx = (room_r, room_c)
            if np.any(patterns[idx] != BG):
                continue
            color = rng.choice(PATTERN_COLORS)
            min_size = 1 if room_dirs[idx] is None else 2
            max_size = 5 if room_dirs[idx] is None else 4
            patterns[idx] = _random_room_pattern(
                room_h,
                room_w,
                arrow_masks[idx],
                rng,
                color=color,
                min_size=min_size,
                max_size=max_size,
            )

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            _paint_room(
                grid,
                room_r,
                room_c,
                room_h,
                room_w,
                patterns[(room_r, room_c)],
                arrow_masks[(room_r, room_c)],
            )

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid room-projection puzzle")
