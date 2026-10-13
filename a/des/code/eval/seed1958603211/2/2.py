import random
from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _legend_on_edge(grid: np.ndarray, bg: int, wall: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for c in (0, w - 1):
        rows = np.flatnonzero((grid[:, c] != bg) & (grid[:, c] != wall))
        if rows.size == 0:
            continue
        if rows.size > 1 and not np.all(np.diff(rows) == 1):
            continue
        candidates.append((int(rows.size), int(c), rows))
    assert candidates, "expected a vertical legend on the left or right edge"
    _, legend_col, legend_rows = max(candidates)
    legend_mask = np.zeros_like(grid, dtype=bool)
    colors: list[int] = []
    for r in legend_rows:
        legend_mask[int(r), legend_col] = True
        colors.append(int(grid[int(r), legend_col]))
    assert colors, "legend must contain at least one color"
    return colors, legend_mask


def _room_slices(boundaries: list[int]) -> list[tuple[int, int]]:
    rooms = []
    for a, b in zip(boundaries, boundaries[1:]):
        assert b - a >= 2, "room interior must have positive area"
        rooms.append((int(a) + 1, int(b) - 1))
    return rooms


def _find_room_index(value: int, spans: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(spans):
        if lo <= value <= hi:
            return int(idx)
    raise AssertionError("coordinate does not fall inside any room")


def _analyze_layout(
    grid: np.ndarray,
) -> tuple[list[int], np.ndarray, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    colors, counts = np.unique(g[g != bg], return_counts=True)
    assert colors.size >= 2, "expected wall color plus at least one non-background color"
    wall = int(colors[np.argmax(counts)])

    legend_colors, legend_mask = _legend_on_edge(g, bg, wall)

    wall_pts = np.argwhere(g == wall)
    assert wall_pts.size, "expected wall pixels"
    r0, c0 = np.min(wall_pts, axis=0)
    r1, c1 = np.max(wall_pts, axis=0)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]

    row_counts = np.count_nonzero(sub == wall, axis=1)
    col_counts = np.count_nonzero(sub == wall, axis=0)
    wall_rows = [int(r0 + i) for i, n in enumerate(row_counts) if int(n) * 2 > sub.shape[1]]
    wall_cols = [int(c0 + i) for i, n in enumerate(col_counts) if int(n) * 2 > sub.shape[0]]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "failed to recover room boundaries"

    room_rows = _room_slices(wall_rows)
    room_cols = _room_slices(wall_cols)

    seed_mask = (g != bg) & (g != wall) & (~legend_mask)
    seed_pts = np.argwhere(seed_mask)
    assert seed_pts.size, "expected exactly one seed room"
    seed_color = int(g[tuple(seed_pts[0])])
    assert np.all(g[seed_mask] == seed_color), "seed room must use a single color"

    seed_r = _find_room_index(int(seed_pts[0, 0]), room_rows)
    seed_c = _find_room_index(int(seed_pts[0, 1]), room_cols)
    sr0, sr1 = room_rows[seed_r]
    sc0, sc1 = room_cols[seed_c]
    assert np.all(g[sr0 : sr1 + 1, sc0 : sc1 + 1] == seed_color), "seed room must be fully filled"

    return (
        legend_colors,
        legend_mask,
        wall,
        wall_rows,
        wall_cols,
        room_rows,
        room_cols,
        (seed_r, seed_c),
        seed_color,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread room colors by BFS distance through single-pixel doors, using the edge legend as the layer palette.
    """
    g = np.asarray(grid, dtype=int)
    (
        legend_colors,
        _legend_mask,
        bg_wall,
        wall_rows,
        wall_cols,
        room_rows,
        room_cols,
        seed_room,
        seed_color,
    ) = _analyze_layout(g)
    _ = bg_wall

    n_room_rows = len(room_rows)
    n_room_cols = len(room_cols)
    dist = -np.ones((n_room_rows, n_room_cols), dtype=int)
    queue = deque([seed_room])
    dist[seed_room] = 0

    while queue:
        rr, cc = queue.popleft()

        if cc + 1 < n_room_cols:
            wr = room_rows[rr]
            door_segment = g[wr[0] : wr[1] + 1, wall_cols[cc + 1]]
            if np.any(door_segment == _most_common_color(g)) and dist[rr, cc + 1] < 0:
                dist[rr, cc + 1] = dist[rr, cc] + 1
                queue.append((rr, cc + 1))
        if cc - 1 >= 0:
            wr = room_rows[rr]
            door_segment = g[wr[0] : wr[1] + 1, wall_cols[cc]]
            if np.any(door_segment == _most_common_color(g)) and dist[rr, cc - 1] < 0:
                dist[rr, cc - 1] = dist[rr, cc] + 1
                queue.append((rr, cc - 1))
        if rr + 1 < n_room_rows:
            wc = room_cols[cc]
            door_segment = g[wall_rows[rr + 1], wc[0] : wc[1] + 1]
            if np.any(door_segment == _most_common_color(g)) and dist[rr + 1, cc] < 0:
                dist[rr + 1, cc] = dist[rr, cc] + 1
                queue.append((rr + 1, cc))
        if rr - 1 >= 0:
            wc = room_cols[cc]
            door_segment = g[wall_rows[rr], wc[0] : wc[1] + 1]
            if np.any(door_segment == _most_common_color(g)) and dist[rr - 1, cc] < 0:
                dist[rr - 1, cc] = dist[rr, cc] + 1
                queue.append((rr - 1, cc))

    out = g.copy()
    for rr, (r0, r1) in enumerate(room_rows):
        for cc, (c0, c1) in enumerate(room_cols):
            layer = int(dist[rr, cc])
            if layer < 0:
                continue
            if layer == 0:
                fill = seed_color
            elif layer <= len(legend_colors):
                fill = int(legend_colors[layer - 1])
            else:
                continue
            out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


BACKGROUND = 0
WALL = 5
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 5)
GRID_EXTRA_HEIGHT = (2, 6)
LEGEND_GAP_RANGE = (1, 3)
FAR_MARGIN_RANGE = (1, 3)
GENERATION_RETRIES = 400
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _room_boundaries(start: int, sizes: list[int]) -> list[int]:
    bounds = [int(start)]
    cur = int(start)
    for size in sizes:
        cur += int(size) + 1
        bounds.append(cur)
    return bounds


def _expected_output_from_layout(
    grid: np.ndarray,
    room_rows: list[tuple[int, int]],
    room_cols: list[tuple[int, int]],
    seed_room: tuple[int, int],
    seed_color: int,
    legend_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rr, (r0, r1) in enumerate(room_rows):
        for cc, (c0, c1) in enumerate(room_cols):
            layer = abs(rr - seed_room[0]) + abs(cc - seed_room[1])
            if layer == 0:
                fill = seed_color
            elif layer <= len(legend_colors):
                fill = int(legend_colors[layer - 1])
            else:
                continue
            out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    n_room_rows = random.randint(*ROOM_ROWS_RANGE)
    n_room_cols = random.randint(*ROOM_COLS_RANGE)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_cols)]

    maze_h = sum(room_heights) + n_room_rows + 1
    maze_w = sum(room_widths) + n_room_cols + 1

    side = random.choice(["left", "right"])
    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    far_margin = random.randint(*FAR_MARGIN_RANGE)
    grid_w = maze_w + legend_gap + far_margin + 1
    grid_h = max(maze_h, 1) + random.randint(*GRID_EXTRA_HEIGHT)

    if grid_h < 3 or grid_w < 6:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    if side == "left":
        maze_left = 1 + legend_gap
    else:
        maze_left = random.randint(1, far_margin)
    if side == "right":
        maze_left = random.randint(1, far_margin)

    max_top = grid_h - maze_h - 1
    if max_top < 1:
        return None
    maze_top = random.randint(1, max_top)

    wall_rows = _room_boundaries(maze_top, room_heights)
    wall_cols = _room_boundaries(maze_left, room_widths)
    room_rows = _room_slices(wall_rows)
    room_cols = _room_slices(wall_cols)

    for wr in wall_rows:
        grid[wr, wall_cols[0] : wall_cols[-1] + 1] = WALL
    for wc in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, wc] = WALL

    for boundary_idx in range(1, len(wall_rows) - 1):
        wr = wall_rows[boundary_idx]
        for c0, c1 in room_cols:
            door_c = random.randint(c0, c1)
            grid[wr, door_c] = BACKGROUND

    for boundary_idx in range(1, len(wall_cols) - 1):
        wc = wall_cols[boundary_idx]
        for r0, r1 in room_rows:
            door_r = random.randint(r0, r1)
            grid[door_r, wc] = BACKGROUND

    seed_room = (random.randrange(n_room_rows), random.randrange(n_room_cols))
    max_depth = max(
        abs(rr - seed_room[0]) + abs(cc - seed_room[1])
        for rr in range(n_room_rows)
        for cc in range(n_room_cols)
    )
    short_legend = max_depth >= 2 and random.random() < 0.5
    if short_legend:
        legend_len = random.randint(1, max_depth - 1)
    else:
        legend_len = random.randint(max(1, max_depth), min(6, max_depth + 2))

    if legend_len + 1 > len(COLOR_POOL):
        return None
    chosen = random.sample(COLOR_POOL, legend_len + 1)
    seed_color = int(chosen[0])
    legend_colors = [int(x) for x in chosen[1:]]

    sr0, sr1 = room_rows[seed_room[0]]
    sc0, sc1 = room_cols[seed_room[1]]
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = seed_color

    legend_col = 0 if side == "left" else grid_w - 1
    if legend_len > grid_h:
        return None
    legend_top = random.randint(0, grid_h - legend_len)
    for i, color in enumerate(legend_colors):
        grid[legend_top + i, legend_col] = int(color)

    expected = _expected_output_from_layout(grid, room_rows, room_cols, seed_room, seed_color, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
