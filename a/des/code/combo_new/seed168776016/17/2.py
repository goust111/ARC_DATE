from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_wall_color(grid: np.ndarray) -> int:
    if np.any(grid == GREY):
        return GREY
    vals, counts = np.unique(grid[grid != BLACK], return_counts=True)
    if len(vals) == 0:
        return GREY
    return int(vals[np.argmax(counts)])


def _touches_boundary(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _find_enclosed_rooms(grid: np.ndarray, wall: int):
    h, w = grid.shape
    rooms = []
    for comp in _components(grid != wall):
        if _touches_boundary(comp, h, w):
            continue
        seeds: list[tuple[int, int, int]] = []
        border: list[tuple[int, int]] = []
        for r, c in comp:
            value = int(grid[r, c])
            if value not in (BLACK, wall):
                seeds.append((r, c, value))
            elif any(int(grid[nr, nc]) == wall for nr, nc in _neighbors4(r, c, h, w)):
                border.append((r, c))
        rooms.append((comp, seeds, border))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """Color each seeded room's inner wall border and erase the seed marker.

    Input structure:
    - The background is black.
    - Grey pixels form a one-pixel-thick wall network that encloses irregular 4-connected black rooms.
    - Some enclosed rooms contain exactly one uniquely colored seed pixel, while other rooms are empty and the exterior black background may lie outside the wall network.

    Rule:
    - Find each 4-connected component of non-wall cells and ignore components that touch the grid boundary because they are outside the closed rooms.
    - For every enclosed component containing exactly one non-black, non-wall seed pixel, recolor every black cell in that component that is 4-neighbor-adjacent to a grey wall.
    - Restore the seed pixel itself to black, leaving empty rooms and all grey wall pixels unchanged.

    Key insight:
    - The seed is only an instruction that assigns a color to its enclosing room, and the target is the one-cell inner border just inside the grey walls rather than the whole room.
    """
    g = np.asarray(grid, dtype=int)
    wall = _detect_wall_color(g)
    out = g.copy()

    for _, seeds, border in _find_enclosed_rooms(g, wall):
        if len(seeds) != 1:
            continue
        sr, sc, color = seeds[0]
        for r, c in border:
            out[r, c] = color
        out[sr, sc] = BLACK

    return out


def _coarse_neighbors(r: int, c: int, rows: int, cols: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def _grow_partition(rows: int, cols: int, room_count: int) -> np.ndarray:
    seeds = random.sample([(r, c) for r in range(rows) for c in range(cols)], room_count)
    part = np.full((rows, cols), -1, dtype=int)
    frontiers = [set() for _ in range(room_count)]

    for room_id, (r, c) in enumerate(seeds):
        part[r, c] = room_id

    def add_frontier(room_id: int, r: int, c: int):
        for nr, nc in _coarse_neighbors(r, c, rows, cols):
            if part[nr, nc] == -1:
                frontiers[room_id].add((nr, nc))

    for room_id, (r, c) in enumerate(seeds):
        add_frontier(room_id, r, c)

    remaining = rows * cols - room_count
    while remaining:
        valid_rooms = [room_id for room_id, frontier in enumerate(frontiers) if frontier]
        room_id = random.choice(valid_rooms)
        cell = random.choice(tuple(frontiers[room_id]))
        frontiers[room_id].discard(cell)
        r, c = cell
        if part[r, c] != -1:
            continue
        part[r, c] = room_id
        remaining -= 1
        add_frontier(room_id, r, c)

    return part


def _cell_origin(index: int, cell_size: int) -> int:
    return 1 + index * (cell_size + 1)


def _draw_rooms(partition: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = partition.shape
    height = rows * (cell_h + 1) + 1
    width = cols * (cell_w + 1) + 1
    grid = np.full((height, width), GREY, dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = _cell_origin(r, cell_h)
            c0 = _cell_origin(c, cell_w)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = BLACK

    for r in range(rows - 1):
        for c in range(cols):
            if partition[r, c] != partition[r + 1, c]:
                continue
            wall_r = _cell_origin(r, cell_h) + cell_h
            c0 = _cell_origin(c, cell_w)
            grid[wall_r, c0 : c0 + cell_w] = BLACK

    for r in range(rows):
        for c in range(cols - 1):
            if partition[r, c] != partition[r, c + 1]:
                continue
            wall_c = _cell_origin(c, cell_w) + cell_w
            r0 = _cell_origin(r, cell_h)
            grid[r0 : r0 + cell_h, wall_c] = BLACK

    return grid


def _embed_in_canvas(room_grid: np.ndarray) -> np.ndarray:
    rh, rw = room_grid.shape
    extra_h = random.randint(0, min(4, 30 - rh))
    extra_w = random.randint(0, min(4, 30 - rw))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    canvas = np.zeros((rh + extra_h, rw + extra_w), dtype=int)
    canvas[top : top + rh, left : left + rw] = room_grid
    return canvas


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.randrange(2):
        out = np.flipud(out)
    if random.randrange(2):
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(3, 4)
    cell_w = random.randint(3, 4)
    max_rows = min(5, 29 // (cell_h + 1))
    max_cols = min(6, 29 // (cell_w + 1))
    room_rows = random.randint(2, max_rows)
    room_cols = random.randint(2, max_cols)
    total_cells = room_rows * room_cols
    room_count = random.randint(2, min(7, total_cells))

    partition = _grow_partition(room_rows, room_cols, room_count)
    room_grid = _draw_rooms(partition, cell_h, cell_w)
    grid = _embed_in_canvas(room_grid)

    rooms = _find_enclosed_rooms(grid, GREY)
    if len(rooms) != room_count:
        return None

    seeded_count = random.randint(1, min(len(SEED_COLORS), room_count - 1))
    seeded_ids = set(random.sample(range(room_count), seeded_count))
    seed_colors = random.sample(SEED_COLORS, seeded_count)

    for room_id, (_, _, border) in enumerate(rooms):
        if room_id not in seeded_ids and not border:
            return None

    color_idx = 0
    for room_id, (cells, _, border) in enumerate(rooms):
        if room_id not in seeded_ids:
            continue
        border_set = set(border)
        candidates = [(r, c) for r, c in cells if int(grid[r, c]) == BLACK and (r, c) not in border_set]
        if not candidates:
            return None
        sr, sc = random.choice(candidates)
        grid[sr, sc] = seed_colors[color_idx]
        color_idx += 1

    grid = _random_transform(grid)

    if not _is_valid_input(grid):
        return None

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    wall = _detect_wall_color(grid)
    if wall != GREY:
        return False

    rooms = _find_enclosed_rooms(grid, wall)
    if len(rooms) < 2:
        return False

    seeded_rooms = []
    empty_rooms = []
    seed_colors: list[int] = []
    for _, seeds, border in rooms:
        if len(seeds) == 1:
            seeded_rooms.append((seeds[0], border))
            seed_colors.append(seeds[0][2])
            if not border:
                return False
        elif len(seeds) == 0:
            empty_rooms.append(border)
        else:
            return False

    if not seeded_rooms or not empty_rooms:
        return False
    if len(seed_colors) != len(set(seed_colors)):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    for (sr, sc, color), border in seeded_rooms:
        if int(out[sr, sc]) != BLACK:
            return False
        if any(int(out[r, c]) != color for r, c in border):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seeded-room input")
