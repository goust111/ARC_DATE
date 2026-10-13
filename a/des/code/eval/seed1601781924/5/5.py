from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
WALL = 5
ARROW = 8
TARGET_COLORS = (1, 2, 3, 4, 6, 7, 9)

DIRS: dict[str, tuple[int, int]] = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
LEFT_OF: dict[tuple[int, int], tuple[int, int]] = {
    (0, 1): (-1, 0),
    (1, 0): (0, 1),
    (0, -1): (1, 0),
    (-1, 0): (0, -1),
}
CW_PERP: dict[tuple[int, int], tuple[int, int]] = {
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
    (-1, 0): (0, 1),
}

ROOM_LAYOUTS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
)
ROOM_SIZE_MIN = 6
ROOM_SIZE_MAX = 9
MAX_GENERATION_ATTEMPTS = 300


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = tip
    dr, dc = direction
    lr, lc = LEFT_OF[direction]
    cells = {(tr - k * dr, tc - k * dc) for k in (1, 2, 3)}
    cells.add((tr + lr, tc + lc))
    cells.add((tr - lr, tc - lc))
    return cells


def _path_cells(shape: tuple[int, int], tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = tip
    path: list[tuple[int, int]] = []
    while _inside(shape, r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def _stripe_cells(
    stop: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = CW_PERP[direction]
    sr, sc = stop
    return [(sr + i * dr, sc + i * dc) for i in range(length)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _room_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != WALL):
        r0, c0 = map(int, comp.min(axis=0))
        r1, c1 = map(int, comp.max(axis=0))
        room = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(room != WALL), "rooms must be solid rectangles"
        rooms.append((r0, c0, r1, c1))
    rooms.sort()
    return rooms


def _detect_arrow(room: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int]]:
    arrow_pts = {tuple(map(int, rc)) for rc in np.argwhere(room == ARROW)}
    assert len(arrow_pts) == 5, "each room must contain exactly five azure arrow pixels"

    matches: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for name, direction in DIRS.items():
        for r in range(room.shape[0]):
            for c in range(room.shape[1]):
                expected = _arrow_cells((r, c), direction)
                if all(_inside(room.shape, rr, cc) for rr, cc in expected) and expected == arrow_pts:
                    matches.append((name, direction, (r, c)))

    assert len(matches) == 1, "arrow geometry must determine a unique direction"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each rectangular room, trace the azure arrow's direction, collect target colors
    along the ray from the tip, and redraw the collected sequence as a stripe at the
    ray's terminal cell using the room's left-side ordering rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r0, c0, r1, c1 in _room_bboxes(g):
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        _, direction, tip = _detect_arrow(room)
        path = _path_cells(room.shape, tip, direction)
        stop = path[-1]

        target_mask = (room != BG) & (room != ARROW)
        target_mask &= room != WALL
        collected = [int(room[r, c]) for r, c in path if target_mask[r, c]]

        out_room = out[r0 : r1 + 1, c0 : c1 + 1]
        out_room[room == ARROW] = BG
        out_room[target_mask] = BG

        for (rr, cc), color in zip(_stripe_cells(stop, direction, len(collected)), collected):
            assert _inside(room.shape, rr, cc), "stripe must remain inside the room"
            out_room[rr, cc] = color

    return out


def _tip_candidates(h: int, w: int, direction: tuple[int, int], stripe_len: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            tip = (r, c)
            arrow = _arrow_cells(tip, direction)
            if not all(_inside((h, w), rr, cc) for rr, cc in arrow):
                continue
            path = _path_cells((h, w), tip, direction)
            if len(path) < stripe_len:
                continue
            stripe = _stripe_cells(path[-1], direction, stripe_len)
            if not all(_inside((h, w), rr, cc) for rr, cc in stripe):
                continue
            candidates.append(tip)
    return candidates


def _pick_path_indices(length: int, count: int) -> list[int]:
    assert 0 < count <= length
    spaced = [combo for combo in combinations(range(length), count) if all(b - a >= 2 for a, b in zip(combo, combo[1:]))]
    if spaced:
        return list(random.choice(spaced))
    return sorted(random.sample(range(length), count))


def _pick_offpath_cells(
    free_cells: list[tuple[int, int]],
    count: int,
    anchors: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    random.shuffle(free_cells)

    picked: list[tuple[int, int]] = []
    for cell in free_cells:
        if any(max(abs(cell[0] - ar), abs(cell[1] - ac)) <= 1 for ar, ac in anchors):
            continue
        if any(max(abs(cell[0] - pr), abs(cell[1] - pc)) <= 1 for pr, pc in picked):
            continue
        picked.append(cell)
        if len(picked) == count:
            return picked

    if len(free_cells) < count:
        return None
    return random.sample(free_cells, count)


def _choose_direction(boundary_dirs: list[str]) -> str:
    wall_dirs = [name for name in DIRS if name not in boundary_dirs]
    if boundary_dirs and wall_dirs and random.random() < 0.5:
        return random.choice(boundary_dirs)
    pool = wall_dirs or boundary_dirs
    return random.choice(pool)


def _populate_room(
    room: np.ndarray,
    boundary_dirs: list[str],
) -> bool:
    h, w = room.shape
    direction_name = _choose_direction(boundary_dirs)
    direction = DIRS[direction_name]

    feasible: dict[int, list[tuple[int, int]]] = {}
    for stripe_len in range(1, 5):
        tips = _tip_candidates(h, w, direction, stripe_len)
        if tips:
            feasible[stripe_len] = tips
    if not feasible:
        return False

    stripe_len = random.choice(sorted(feasible))
    tip = random.choice(feasible[stripe_len])
    arrow = _arrow_cells(tip, direction)
    path = _path_cells((h, w), tip, direction)
    stop = path[-1]
    stripe = _stripe_cells(stop, direction, stripe_len)

    path_indices = _pick_path_indices(len(path), stripe_len)
    path_targets = [path[i] for i in path_indices]

    blocked = set(arrow) | set(path) | set(stripe)
    free_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    max_total_targets = min(4, stripe_len + len(free_cells))
    if max_total_targets < stripe_len:
        return False

    total_targets = random.randint(stripe_len, max_total_targets)
    extra_targets = total_targets - stripe_len
    offpath_targets = _pick_offpath_cells(free_cells[:], extra_targets, path_targets)
    if offpath_targets is None:
        return False

    colors = random.sample(TARGET_COLORS, total_targets)
    path_colors = colors[:stripe_len]
    extra_colors = colors[stripe_len:]

    room[:, :] = BG
    for rr, cc in arrow:
        room[rr, cc] = ARROW
    for (rr, cc), color in zip(path_targets, path_colors):
        room[rr, cc] = color
    for (rr, cc), color in zip(offpath_targets, extra_colors):
        room[rr, cc] = color
    return True


def _build_partition() -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]], int, int]:
    n_rows, n_cols = random.choice(ROOM_LAYOUTS)
    heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(n_rows)]
    widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(n_cols)]

    H = sum(heights) + (n_rows - 1)
    W = sum(widths) + (n_cols - 1)
    grid = np.full((H, W), BG, dtype=int)

    offset = heights[0]
    for h in heights[1:]:
        grid[offset, :] = WALL
        offset += 1 + h

    offset = widths[0]
    for w in widths[1:]:
        grid[:, offset] = WALL
        offset += 1 + w

    rooms: list[tuple[int, int, int, int, int, int]] = []
    row_start = 0
    for ri, h in enumerate(heights):
        col_start = 0
        for ci, w in enumerate(widths):
            rooms.append((row_start, col_start, row_start + h - 1, col_start + w - 1, ri, ci))
            col_start += w + 1
        row_start += h + 1

    return grid, rooms, n_rows, n_cols


def _gen_attempt() -> np.ndarray | None:
    grid, rooms, n_rows, n_cols = _build_partition()

    for r0, c0, r1, c1, ri, ci in rooms:
        room = grid[r0 : r1 + 1, c0 : c1 + 1]
        boundary_dirs: list[str] = []
        if ri == 0:
            boundary_dirs.append("up")
        if ri == n_rows - 1:
            boundary_dirs.append("down")
        if ci == 0:
            boundary_dirs.append("left")
        if ci == n_cols - 1:
            boundary_dirs.append("right")
        if not _populate_room(room, boundary_dirs):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == ARROW) != 5 * len(rooms):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
