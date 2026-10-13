from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MAROON = 9
CONTROL_COLORS = (1, 3, 4, 7)

LEGEND_H = 11
LEGEND_W = 7

GLYPH_LIBRARY = (
    np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _unique_shortest_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    passable = grid != GREY
    dist = np.full((h, w), -1, dtype=int)
    count = np.zeros((h, w), dtype=int)
    parent = np.full((h, w, 2), -1, dtype=int)

    sr, sc = start
    er, ec = end
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    count[sr, sc] = 1

    while q:
        r, c = q.popleft()
        if (r, c) == (er, ec):
            continue
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or not passable[nr, nc]:
                continue
            nd = dist[r, c] + 1
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                count[nr, nc] = count[r, c]
                parent[nr, nc] = (r, c)
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                count[nr, nc] = min(2, count[nr, nc] + count[r, c])

    assert dist[er, ec] >= 0, "endpoints must be connected"
    assert count[er, ec] == 1, "expected a unique shortest path"

    path = []
    cur = (er, ec)
    while cur != (sr, sc):
        path.append(cur)
        pr, pc = parent[cur]
        assert pr >= 0
        cur = (int(pr), int(pc))
    path.append((sr, sc))
    path.reverse()
    return path


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    glyph_comps = _components_mask(grid == MAROON)
    assert len(glyph_comps) == 2, "expected two maroon glyphs in the legend"

    glyphs: dict[int, np.ndarray] = {}
    label_cells = []
    all_points = []
    for comp in glyph_comps:
        r0, c0, r1, c1 = _bbox_coords(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (4, 4), "legend glyphs must be 4x4"
        mask = (grid[r0 : r1 + 1, c0 : c1 + 1] == MAROON)
        label_col = c1 + 1
        assert label_col < grid.shape[1]
        candidates = [
            (r, label_col)
            for r in range(r0, r1 + 1)
            if int(grid[r, label_col]) not in (BLACK, RED, GREY, MAROON)
        ]
        assert len(candidates) == 1, "each legend glyph needs one color label"
        label_r, label_c = candidates[0]
        color = int(grid[label_r, label_c])
        assert color in CONTROL_COLORS
        assert color not in glyphs
        glyphs[color] = mask
        label_cells.append((label_r, label_c))
        all_points.extend((int(r), int(c)) for r, c in comp)
        all_points.append((label_r, label_c))

    coords = np.array(all_points, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    legend_bbox = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
    lr0, lc0, lr1, lc1 = legend_bbox
    assert 0 <= lr0 < lr1 < grid.shape[0]
    assert 0 <= lc0 < lc1 < grid.shape[1]
    assert np.all(grid[lr0, lc0 : lc1 + 1] == GREY)
    assert np.all(grid[lr1, lc0 : lc1 + 1] == GREY)
    assert np.all(grid[lr0 : lr1 + 1, lc0] == GREY)
    assert np.all(grid[lr0 : lr1 + 1, lc1] == GREY)
    return glyphs, legend_bbox


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    glyphs, legend_bbox = _parse_legend(g)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    endpoints = [tuple(map(int, rc)) for rc in np.argwhere((g == RED) & ~legend_mask)]
    assert len(endpoints) == 2, "expected two red endpoints"
    assert endpoints[0] != endpoints[1]

    path = _unique_shortest_path(g, endpoints[0], endpoints[1])
    path_set = set(path)

    controls = []
    for r, c in np.argwhere(~legend_mask):
        val = int(g[r, c])
        if val in CONTROL_COLORS:
            controls.append((int(r), int(c), val))
    assert 4 <= len(controls) <= 6, "expected 4-6 control pixels"

    active = [item for item in controls if (item[0], item[1]) in path_set]
    inactive = [item for item in controls if (item[0], item[1]) not in path_set]
    active_colors = {color for _r, _c, color in active}
    inactive_colors = {color for _r, _c, color in inactive}

    assert len(active) >= 2
    assert len(inactive) >= 2
    assert len(active_colors) == 2, "legend must encode exactly the path colors"
    assert set(glyphs) == active_colors
    assert active_colors <= inactive_colors, "same colors must appear both on and off the path"

    return {
        "grid": g,
        "glyphs": glyphs,
        "legend_bbox": legend_bbox,
        "legend_mask": legend_mask,
        "endpoints": endpoints,
        "path": path,
        "path_set": path_set,
        "controls": controls,
        "active": active,
        "inactive": inactive,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter control pixels by the unique shortest maze path, then stamp only the surviving glyphs and erase the legend.

    Input structure:
    - Grey pixels are one-cell-thick maze walls and black pixels are corridors.
    - Two red singleton endpoints lie in the corridors, and 4-6 singleton control pixels use two colors chosen from blue/green/yellow/orange.
    - The lower-right legend room is enclosed by grey walls and contains exactly two 4x4 maroon glyphs, each with one adjacent color label.

    Rule:
    - Treat every non-grey cell as passable and find the unique shortest 4-connected path between the two red endpoints.
    - A control pixel is active iff its coordinate lies on that shortest path.
    - For each active control pixel, look up the glyph with the same legend-label color and stamp that 4x4 maroon pattern with the control pixel as the glyph's top-left corner, skipping out-of-bounds cells and grey walls.
    - Erase every inactive control pixel to black and erase the entire legend room, including its enclosing wall.

    Key insight:
    - The shortest path is not the thing being painted; it acts as a spatial filter that decides which same-colored control pixels survive long enough to trigger a glyph stamp.
    """
    info = _analyze_input(grid)
    g = info["grid"]
    out = g.copy()

    lr0, lc0, lr1, lc1 = info["legend_bbox"]
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    for r, c, _color in info["controls"]:
        out[r, c] = BLACK

    for r, c, color in info["active"]:
        glyph = info["glyphs"][color]
        for dr in range(4):
            for dc in range(4):
                if not glyph[dr, dc]:
                    continue
                nr = r + dr
                nc = c + dc
                if not _in_bounds(out.shape, nr, nc):
                    continue
                if info["legend_mask"][nr, nc]:
                    continue
                if int(g[nr, nc]) == GREY:
                    continue
                if int(g[nr, nc]) == RED:
                    continue
                out[nr, nc] = MAROON

    return out


def _generate_perfect_maze(cell_rows: int, cell_cols: int) -> np.ndarray:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    maze = np.full((h, w), GREY, dtype=int)
    visited = np.zeros((cell_rows, cell_cols), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    maze[1, 1] = BLACK

    while stack:
        cr, cc = stack[-1]
        options = []
        for dr, dc in DIR4:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not visited[nr, nc]:
                options.append((nr, nc))
        if not options:
            stack.pop()
            continue
        nr, nc = random.choice(options)
        r0 = 2 * cr + 1
        c0 = 2 * cc + 1
        r1 = 2 * nr + 1
        c1 = 2 * nc + 1
        maze[(r0 + r1) // 2, (c0 + c1) // 2] = BLACK
        maze[r1, c1] = BLACK
        visited[nr, nc] = True
        stack.append((nr, nc))

    return maze


def _path_in_open_grid(
    patch: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    return _unique_shortest_path(patch, start, end)


def _pick_endpoint_pair(maze: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    corridors = [tuple(map(int, rc)) for rc in np.argwhere(maze == BLACK)]
    ranked = sorted(corridors, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    top = ranked[: max(8, len(ranked) // 6)]
    bottom = ranked[-max(8, len(ranked) // 6) :]

    best = None
    for start in random.sample(top, min(len(top), 8)):
        for end in random.sample(bottom, min(len(bottom), 8)):
            if start == end:
                continue
            try:
                path = _path_in_open_grid(maze, start, end)
            except AssertionError:
                continue
            dist = len(path) - 1
            score = (dist, end[0] + end[1], -(start[0] + start[1]))
            if best is None or score > best[0]:
                best = (score, start, end, path)
    if best is None:
        return None
    _score, start, end, path = best
    if len(path) < 14:
        return None
    return start, end, path


def _spaced_sample(
    coords: list[tuple[int, int]], count: int, min_dist: int
) -> list[tuple[int, int]] | None:
    order = coords[:]
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for cell in order:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= min_dist for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _draw_legend(
    grid: np.ndarray,
    room_top: int,
    room_left: int,
    glyphs: list[np.ndarray],
    colors: tuple[int, int],
) -> None:
    grid[room_top + 1 : room_top + LEGEND_H - 1, room_left + 1 : room_left + LEGEND_W - 1] = BLACK

    top_r = room_top + 1
    top_c = room_left + 1
    bottom_r = room_top + 6
    for idx, (glyph, color, base_r) in enumerate(zip(glyphs, colors, (top_r, bottom_r))):
        for dr in range(4):
            for dc in range(4):
                if glyph[dr, dc]:
                    grid[base_r + dr, top_c + dc] = MAROON
        label_r = base_r + 1 + (idx % 2)
        grid[label_r, top_c + 4] = int(color)


def _construct_candidate() -> np.ndarray | None:
    cell_rows = random.randint(6, 7)
    cell_cols = random.randint(6, 8)
    maze = _generate_perfect_maze(cell_rows, cell_cols)
    endpoints = _pick_endpoint_pair(maze)
    if endpoints is None:
        return None
    start, end, path_local = endpoints

    maze_h, maze_w = maze.shape
    gap_x = random.randint(2, 4)
    h = max(maze_h + 2, LEGEND_H + 2) + random.randint(0, 3)
    w = maze_w + gap_x + LEGEND_W + 2
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), GREY, dtype=int)
    maze_r0 = 1
    maze_c0 = 1
    grid[maze_r0 : maze_r0 + maze_h, maze_c0 : maze_c0 + maze_w] = maze

    room_top = h - LEGEND_H - 1
    room_left = w - LEGEND_W - 1
    if room_left <= maze_c0 + maze_w:
        return None

    legend_colors = tuple(random.sample(CONTROL_COLORS, 2))
    glyph_choices = random.sample(GLYPH_LIBRARY, 2)
    _draw_legend(grid, room_top, room_left, glyph_choices, legend_colors)

    path_global = [(maze_r0 + r, maze_c0 + c) for r, c in path_local]
    eligible_on = path_global[3:-3]
    if len(eligible_on) < 2:
        return None

    total_controls = random.randint(4, 6)
    on_count = random.randint(2, min(4, total_controls - 2))
    off_count = total_controls - on_count

    on_cells = _spaced_sample(eligible_on, on_count, 3)
    if on_cells is None:
        return None

    path_set = set(path_global)
    off_pool = [
        (maze_r0 + r, maze_c0 + c)
        for r, c in np.argwhere(maze == BLACK)
        if (maze_r0 + r, maze_c0 + c) not in path_set
        and (maze_r0 + r, maze_c0 + c) not in {path_global[0], path_global[-1]}
    ]
    off_cells = _spaced_sample(off_pool, off_count, 3)
    if off_cells is None:
        return None

    for rc in path_global[0], path_global[-1]:
        grid[rc] = RED

    on_colors = [legend_colors[0], legend_colors[1]]
    while len(on_colors) < on_count:
        on_colors.append(random.choice(legend_colors))
    random.shuffle(on_colors)
    for (r, c), color in zip(on_cells, on_colors):
        grid[r, c] = int(color)

    off_colors = [legend_colors[0], legend_colors[1]]
    while len(off_colors) < off_count:
        off_colors.append(random.choice(legend_colors))
    random.shuffle(off_colors)
    for (r, c), color in zip(off_cells, off_colors):
        grid[r, c] = int(color)

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False

    active_colors = {color for _r, _c, color in info["active"]}
    if active_colors != set(info["glyphs"]):
        return False

    lr0, lc0, lr1, lc1 = info["legend_bbox"]
    if not np.all(out[lr0 : lr1 + 1, lc0 : lc1 + 1] == BLACK):
        return False

    if any(int(out[r, c]) != BLACK for r, c, _color in info["inactive"]):
        return False

    if any(color not in {c for _r, _c, c in info["inactive"]} for color in active_colors):
        return False

    stamped_any = False
    for r, c, color in info["active"]:
        glyph = info["glyphs"][color]
        local_visible = 0
        for dr in range(4):
            for dc in range(4):
                if not glyph[dr, dc]:
                    continue
                nr = r + dr
                nc = c + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if info["legend_mask"][nr, nc]:
                    continue
                if int(grid[nr, nc]) in (GREY, RED):
                    continue
                if int(out[nr, nc]) == MAROON:
                    local_visible += 1
        if local_visible:
            stamped_any = True
    return stamped_any


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _construct_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
