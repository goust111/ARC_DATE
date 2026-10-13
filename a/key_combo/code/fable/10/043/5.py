from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

WALL_COLOR = BLUE
DOOR_COLOR = GREEN
MARKER_COLOR = YELLOW
CORRIDOR_COLORS = (BLACK, TEAL, PINK, ORANGE)
TOGGLE = {
    BLACK: TEAL,
    TEAL: BLACK,
    PINK: ORANGE,
    ORANGE: PINK,
}

GEN_RETRIES = 500
GRID_MIN = 18
GRID_MAX = 30
WALL_COUNT_RANGE = (3, 5)
WALL_MIN_GAP = 3
WALL_MARGIN = 4
DOOR_MARGIN = 3
MARKER_EDGE_RANGE = (1, 3)
NOISE_RATE_RANGE = (0.08, 0.2)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be orthogonal")


def _compress_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cell in path:
        if not out or out[-1] != cell:
            out.append(cell)
    return out


def _sample_spaced_positions(rng: random.Random, count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    need = count + (count - 1) * min_gap
    if span < need:
        return None
    compressed = span - (count - 1) * min_gap
    picks = sorted(rng.sample(range(compressed), count))
    return [lo + picks[i] + i * min_gap for i in range(count)]


def _find_vertical_layout(g: np.ndarray) -> tuple[list[int], list[int]] | None:
    h, w = g.shape
    wall_cols: list[int] = []
    door_rows: list[int] = []
    for c in range(w):
        col = g[:, c]
        if not np.all(np.isin(col, [WALL_COLOR, DOOR_COLOR])):
            continue
        door_idx = np.flatnonzero(col == DOOR_COLOR)
        if len(door_idx) != 3:
            continue
        if int(door_idx[-1] - door_idx[0]) != 2:
            continue
        if not np.all(col[door_idx[0] : door_idx[-1] + 1] == DOOR_COLOR):
            continue
        wall_mask = np.ones(h, dtype=bool)
        wall_mask[door_idx[0] : door_idx[-1] + 1] = False
        if not np.all(col[wall_mask] == WALL_COLOR):
            continue
        wall_cols.append(c)
        door_rows.append(int(door_idx[1]))
    if len(wall_cols) < 1:
        return None
    mask = np.zeros_like(g, dtype=bool)
    mask[:, wall_cols] = True
    structural = (g == WALL_COLOR) | (g == DOOR_COLOR)
    if not np.array_equal(mask, structural):
        return None
    return wall_cols, door_rows


def _find_horizontal_layout(g: np.ndarray) -> tuple[list[int], list[int]] | None:
    h, w = g.shape
    wall_rows: list[int] = []
    door_cols: list[int] = []
    for r in range(h):
        row = g[r, :]
        if not np.all(np.isin(row, [WALL_COLOR, DOOR_COLOR])):
            continue
        door_idx = np.flatnonzero(row == DOOR_COLOR)
        if len(door_idx) != 3:
            continue
        if int(door_idx[-1] - door_idx[0]) != 2:
            continue
        if not np.all(row[door_idx[0] : door_idx[-1] + 1] == DOOR_COLOR):
            continue
        wall_mask = np.ones(w, dtype=bool)
        wall_mask[door_idx[0] : door_idx[-1] + 1] = False
        if not np.all(row[wall_mask] == WALL_COLOR):
            continue
        wall_rows.append(r)
        door_cols.append(int(door_idx[1]))
    if len(wall_rows) < 1:
        return None
    mask = np.zeros_like(g, dtype=bool)
    mask[wall_rows, :] = True
    structural = (g == WALL_COLOR) | (g == DOOR_COLOR)
    if not np.array_equal(mask, structural):
        return None
    return wall_rows, door_cols


def _extract_structure(g: np.ndarray) -> tuple[str, list[int], list[int], tuple[int, int], tuple[int, int]]:
    assert g.ndim == 2, "expected a 2D grid"
    vertical = _find_vertical_layout(g)
    horizontal = _find_horizontal_layout(g)
    assert (vertical is None) != (horizontal is None), "expected exactly one wall orientation"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    assert len(markers) == 2, "expected exactly two yellow markers"

    if vertical is not None:
        walls, doors = vertical
        markers.sort(key=lambda rc: rc[1])
        start, end = markers
        assert start[1] < walls[0] and end[1] > walls[-1], "markers must be outside opposite wall sides"
        return "vertical", walls, doors, start, end

    walls, doors = horizontal  # type: ignore[misc]
    markers.sort(key=lambda rc: rc[0])
    start, end = markers
    assert start[0] < walls[0] and end[0] > walls[-1], "markers must be outside opposite wall sides"
    return "horizontal", walls, doors, start, end


def _route_cells(
    shape: tuple[int, int],
    orientation: str,
    walls: list[int],
    doors: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    path: list[tuple[int, int]] = [start]
    cur = start

    if orientation == "vertical":
        for wall_c, door_r in zip(walls, doors):
            before = (cur[0], wall_c - 1)
            align = (door_r, wall_c - 1)
            cross = (door_r, wall_c)
            path.extend(_line_cells(cur, before)[1:])
            path.extend(_line_cells(before, align)[1:])
            path.extend(_line_cells(align, cross)[1:])
            cur = cross
        align_end = (end[0], cur[1])
        path.extend(_line_cells(cur, align_end)[1:])
        path.extend(_line_cells(align_end, end)[1:])
    else:
        for wall_r, door_c in zip(walls, doors):
            before = (wall_r - 1, cur[1])
            align = (wall_r - 1, door_c)
            cross = (wall_r, door_c)
            path.extend(_line_cells(cur, before)[1:])
            path.extend(_line_cells(before, align)[1:])
            path.extend(_line_cells(align, cross)[1:])
            cur = cross
        align_end = (cur[0], end[1])
        path.extend(_line_cells(cur, align_end)[1:])
        path.extend(_line_cells(align_end, end)[1:])

    path = _compress_path(path)
    seen: set[tuple[int, int]] = set()
    for i, (r, c) in enumerate(path):
        assert 0 <= r < h and 0 <= c < w, "route must stay in bounds"
        if i and abs(r - path[i - 1][0]) + abs(c - path[i - 1][1]) != 1:
            raise AssertionError("route must move by 4-neighbor steps")
        if (r, c) in seen:
            raise AssertionError("route must be simple")
        seen.add((r, c))
    return path


def _toggle_on_route(g: np.ndarray, route: list[tuple[int, int]]) -> np.ndarray:
    out = g.copy()
    for r, c in route:
        color = int(g[r, c])
        if color in TOGGLE:
            out[r, c] = TOGGLE[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique door-threaded corridor route and invert the colors only on its corridor cells.

    Input structure:
    - The grid contains several parallel full-span blue walls, each with one vertical or horizontal three-cell green door segment.
    - Exactly two yellow marker pixels sit outside the wall stack on opposite sides.
    - Every non-wall corridor cell uses only black, teal, pink, or orange.

    Rule:
    - Start from the upper/left-side marker and process the walls in order toward the lower/right-side marker.
    - In each corridor, move orthogonally toward the next wall, slide along that wall to the door center, pass through that center, and continue corridor by corridor until the far marker is reached.
    - Apply black<->teal and pink<->orange to every route cell that belongs to the four corridor colors; keep the walls, all three door cells, and both markers unchanged.

    Key insight:
    - The output does not draw a new line color: the route is the unique maze thread defined by marker side order and door centers, and it is revealed only by which existing corridor cells get involutively recolored.
    """
    g = np.asarray(grid, dtype=int)
    orientation, walls, doors, start, end = _extract_structure(g)
    route = _route_cells(g.shape, orientation, walls, doors, start, end)
    return _toggle_on_route(g, route)


def _sample_layout(rng: random.Random) -> tuple[int, int, str, list[int], list[int], tuple[int, int], tuple[int, int]] | None:
    orientation = rng.choice(["vertical", "horizontal"])
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    if orientation == "vertical":
        wall_count = rng.randint(*WALL_COUNT_RANGE)
        walls = _sample_spaced_positions(rng, wall_count, WALL_MARGIN, w - WALL_MARGIN - 1, WALL_MIN_GAP)
        if walls is None or walls[0] <= 2 or walls[-1] >= w - 3:
            return None
        doors = [rng.randint(DOOR_MARGIN, h - DOOR_MARGIN - 1) for _ in walls]
        start = (rng.randint(1, h - 2), rng.randint(MARKER_EDGE_RANGE[0], min(MARKER_EDGE_RANGE[1], walls[0] - 2)))
        end = (rng.randint(1, h - 2), rng.randint(max(walls[-1] + 2, w - 1 - MARKER_EDGE_RANGE[1]), w - 1 - MARKER_EDGE_RANGE[0]))
        if start[0] == doors[0] or end[0] == doors[-1]:
            return None
        if any(a == b for a, b in zip(doors, doors[1:])):
            return None
    else:
        wall_count = rng.randint(*WALL_COUNT_RANGE)
        walls = _sample_spaced_positions(rng, wall_count, WALL_MARGIN, h - WALL_MARGIN - 1, WALL_MIN_GAP)
        if walls is None or walls[0] <= 2 or walls[-1] >= h - 3:
            return None
        doors = [rng.randint(DOOR_MARGIN, w - DOOR_MARGIN - 1) for _ in walls]
        start = (rng.randint(MARKER_EDGE_RANGE[0], min(MARKER_EDGE_RANGE[1], walls[0] - 2)), rng.randint(1, w - 2))
        end = (rng.randint(max(walls[-1] + 2, h - 1 - MARKER_EDGE_RANGE[1]), h - 1 - MARKER_EDGE_RANGE[0]), rng.randint(1, w - 2))
        if start[1] == doors[0] or end[1] == doors[-1]:
            return None
        if any(a == b for a, b in zip(doors, doors[1:])):
            return None

    return h, w, orientation, walls, doors, start, end


def _paint_structure(
    h: int,
    w: int,
    orientation: str,
    walls: list[int],
    doors: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    if orientation == "vertical":
        for wall_c, door_r in zip(walls, doors):
            grid[:, wall_c] = WALL_COLOR
            grid[door_r - 1 : door_r + 2, wall_c] = DOOR_COLOR
    else:
        for wall_r, door_c in zip(walls, doors):
            grid[wall_r, :] = WALL_COLOR
            grid[wall_r, door_c - 1 : door_c + 2] = DOOR_COLOR
    grid[start] = MARKER_COLOR
    grid[end] = MARKER_COLOR
    return grid


def _assign_corridor_colors(rng: random.Random, grid: np.ndarray, route: list[tuple[int, int]]) -> np.ndarray | None:
    g = grid.copy()
    route_set = set(route)
    corridor = np.isin(g, CORRIDOR_COLORS)
    corridor_cells = [tuple(map(int, rc)) for rc in np.argwhere(corridor)]
    route_toggle_cells = [(r, c) for r, c in route if g[r, c] in TOGGLE]
    off_route_cells = [cell for cell in corridor_cells if cell not in route_set]
    if len(route_toggle_cells) < 12 or len(off_route_cells) < 10:
        return None

    noise_rate = rng.uniform(*NOISE_RATE_RANGE)
    off_noise = max(6, int(round(len(off_route_cells) * noise_rate)))
    noisy = rng.sample(off_route_cells, min(len(off_route_cells), off_noise))
    for r, c in noisy:
        g[r, c] = rng.choice([TEAL, PINK, ORANGE])

    route_colors = [BLACK] * len(route_toggle_cells)
    anchors = rng.sample(range(len(route_toggle_cells)), 4)
    forced = [BLACK, TEAL, PINK, ORANGE]
    for idx, color in zip(anchors, forced):
        route_colors[idx] = color
    for i in range(len(route_toggle_cells)):
        if i in anchors:
            continue
        route_colors[i] = rng.choices(
            population=[BLACK, TEAL, PINK, ORANGE],
            weights=[10, 4, 3, 3],
            k=1,
        )[0]
    rng.shuffle(route_colors)
    for (r, c), color in zip(route_toggle_cells, route_colors):
        g[r, c] = color

    return g


def _semantic_checks(grid: np.ndarray) -> None:
    orientation, walls, doors, start, end = _extract_structure(grid)
    route = _route_cells(grid.shape, orientation, walls, doors, start, end)
    out = main(grid)

    changed = grid != out
    route_toggle_mask = np.zeros_like(grid, dtype=bool)
    for r, c in route:
        if int(grid[r, c]) in TOGGLE:
            route_toggle_mask[r, c] = True

    assert np.array_equal(changed, route_toggle_mask), "only toggle-eligible route cells may change"
    assert np.array_equal(main(out), grid), "solver must be involutive on valid instances"
    assert not np.any(changed & (grid == MARKER_COLOR)), "markers must stay fixed"
    assert not np.any(changed & (grid == DOOR_COLOR)), "door cells must stay fixed"
    assert not np.any(changed & (grid == WALL_COLOR)), "wall cells must stay fixed"

    changed_colors = {int(grid[r, c]) for r, c in np.argwhere(changed)}
    assert changed_colors == set(CORRIDOR_COLORS), "route must cover all four toggle colors"

    off_route = np.isin(grid, [TEAL, PINK, ORANGE]) & ~route_toggle_mask
    assert int(off_route.sum()) >= 6, "need enough non-route colored noise"

    if orientation == "vertical":
        assert start[0] != doors[0] and end[0] != doors[-1], "outer corridors must require turns"
    else:
        assert start[1] != doors[0] and end[1] != doors[-1], "outer corridors must require turns"
    assert all(a != b for a, b in zip(doors, doors[1:])), "each door alignment step must matter"


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    layout = _sample_layout(rng)
    if layout is None:
        return None
    h, w, orientation, walls, doors, start, end = layout
    base = _paint_structure(h, w, orientation, walls, doors, start, end)
    route = _route_cells(base.shape, orientation, walls, doors, start, end)
    grid = _assign_corridor_colors(rng, base, route)
    if grid is None:
        return None
    try:
        _semantic_checks(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
