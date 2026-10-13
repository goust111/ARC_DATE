import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rooms: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == WALL:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != WALL:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise AssertionError("room interior must be a solid rectangle")
            rooms.append((r0, c0, r1, c1))

    return rooms


def _marker_direction(room: tuple[int, int, int, int], pos: tuple[int, int]) -> str:
    r0, c0, r1, c1 = room
    pr, pc = pos
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = pr - center_r
    dc = pc - center_c
    if abs(dr) > abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    rooms = _find_rooms(g)
    assert rooms, "expected at least one room"

    for room in rooms:
        r0, c0, r1, c1 = room
        assert r0 > 0 and c0 > 0 and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1]

        markers = np.argwhere((g[r0 : r1 + 1, c0 : c1 + 1] != BG) & (g[r0 : r1 + 1, c0 : c1 + 1] != WALL))
        assert 1 <= len(markers) <= 4, "each room must have 1-4 markers"

        used_dirs: set[str] = set()
        for mr, mc in markers:
            rr = r0 + int(mr)
            cc = c0 + int(mc)
            color = int(g[rr, cc])
            direction = _marker_direction(room, (rr, cc))
            assert direction not in used_dirs, "room directions must be unique"
            used_dirs.add(direction)

            if direction == "top":
                out[r0 - 1, c0 - 1 : c1 + 2] = color
            elif direction == "right":
                out[r0 - 1 : r1 + 2, c1 + 1] = color
            elif direction == "bottom":
                out[r1 + 1, c0 - 1 : c1 + 2] = color
            else:
                out[r0 - 1 : r1 + 2, c0 - 1] = color

            out[rr, cc] = BG

    return out


def _sample_partition(total: int, parts: int, min_size: int, max_size: int) -> list[int] | None:
    vals = [min_size] * parts
    remain = total - min_size * parts
    if remain < 0:
        return None
    for i in range(parts):
        room_left = parts - i - 1
        extra_cap = max_size - vals[i]
        min_future = room_left * min_size
        add = min(extra_cap, remain - min_future)
        if add < 0:
            return None
        delta = random.randint(0, add)
        vals[i] += delta
        remain -= delta
    vals[-1] += remain
    if all(min_size <= v <= max_size for v in vals):
        return vals
    return None


def _direction_slot(direction: str, h: int, w: int) -> tuple[int, int]:
    mid_r = h // 2
    mid_c = w // 2
    if direction == "top":
        return (0, mid_c)
    if direction == "right":
        return (mid_r, w - 1)
    if direction == "bottom":
        return (h - 1, mid_c)
    return (mid_r, 0)


def _room_instruction_sets(room_count: int) -> list[list[str]]:
    base = [["top"], ["right"], ["bottom"], ["left"], ["top", "right"], ["bottom", "left"], ["top", "right", "bottom"], ["top", "right", "bottom", "left"]]
    random.shuffle(base)
    result = []
    for i in range(room_count):
        result.append(base[i % len(base)].copy())
    random.shuffle(result)
    return result


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(1, 3)
    room_cols = random.randint(1, 3)
    room_count = room_rows * room_cols

    total_h = random.randint(max(3 * room_rows, 5), min(6 * room_rows, 15))
    total_w = random.randint(max(3 * room_cols, 5), min(6 * room_cols, 15))
    heights = _sample_partition(total_h, room_rows, 3, 6)
    widths = _sample_partition(total_w, room_cols, 3, 6)
    if heights is None or widths is None:
        return None

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    grid = np.full((h, w), BG, dtype=int)

    row_edges = [0]
    for rh in heights:
        row_edges.append(row_edges[-1] + rh + 1)
    col_edges = [0]
    for cw in widths:
        col_edges.append(col_edges[-1] + cw + 1)

    for rr in row_edges:
        grid[rr, :] = WALL
    for cc in col_edges:
        grid[:, cc] = WALL

    planned = _room_instruction_sets(room_count)
    idx = 0
    shared_wall_colors: dict[tuple[str, int, int], int] = {}

    for i in range(room_rows):
        for j in range(room_cols):
            r0 = row_edges[i] + 1
            r1 = row_edges[i + 1] - 1
            c0 = col_edges[j] + 1
            c1 = col_edges[j + 1] - 1
            directions = planned[idx]
            idx += 1

            if i > 0 and "top" in directions:
                key = ("h", i - 1, j)
                if key in shared_wall_colors:
                    return None
            if i + 1 < room_rows and "bottom" in directions:
                key = ("h", i, j)
                if key in shared_wall_colors:
                    return None
            if j > 0 and "left" in directions:
                key = ("v", i, j - 1)
                if key in shared_wall_colors:
                    return None
            if j + 1 < room_cols and "right" in directions:
                key = ("v", i, j)
                if key in shared_wall_colors:
                    return None

            random.shuffle(directions)
            colors = random.sample(MARKER_COLORS, len(directions))
            used_cells = set()

            for direction, color in zip(directions, colors):
                lr, lc = _direction_slot(direction, r1 - r0 + 1, c1 - c0 + 1)
                if (lr, lc) in used_cells:
                    return None
                used_cells.add((lr, lc))
                grid[r0 + lr, c0 + lc] = color

                if direction == "top" and i > 0:
                    shared_wall_colors[("h", i - 1, j)] = color
                elif direction == "bottom" and i + 1 < room_rows:
                    shared_wall_colors[("h", i, j)] = color
                elif direction == "left" and j > 0:
                    shared_wall_colors[("v", i, j - 1)] = color
                elif direction == "right" and j + 1 < room_cols:
                    shared_wall_colors[("v", i, j)] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
