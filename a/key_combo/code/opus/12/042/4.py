from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GRAY = 5
ENDPOINT_COLORS = (2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
GRID_SIZE_RANGE = (14, 21)
ROOM_COUNT_CHOICES = (2, 2, 3)
ROOM_HEIGHT_RANGE = (6, 10)
ROOM_WIDTH_RANGE = (6, 10)
ROOM_MARGIN = 3
GAP_MIN = 4
SPUR_PROB = 0.65


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(shape: tuple[int, int], r: int, c: int):
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _black_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(grid.shape, cr, cc):
                    if grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(grid.shape, r, c):
            if grid[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _endpoint_pairs(grid: np.ndarray) -> list[tuple[int, Coord, Coord]]:
    pairs: list[tuple[int, Coord, Coord]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (BLACK, BLUE, GRAY)):
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        p0 = tuple(int(v) for v in coords[0])
        p1 = tuple(int(v) for v in coords[1])
        pairs.append((color, p0, p1))
    assert 2 <= len(pairs) <= 3, "expected 2 or 3 endpoint pairs"
    return pairs


def _unique_shortest_path(grid: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    q: deque[Coord] = deque([start])
    dist[start] = 0
    ways[start] = 1

    def passable(r: int, c: int) -> bool:
        if (r, c) == start or (r, c) == goal:
            return True
        return int(grid[r, c]) == BLACK

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(grid.shape, r, c):
            if not passable(nr, nc):
                continue
            nd = dist[r, c] + 1
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                ways[nr, nc] = min(2, ways[nr, nc] + ways[r, c])

    assert dist[goal] >= 0, "endpoints must be connected"
    assert ways[goal] == 1, "shortest path must be unique"

    path_rev = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        preds = []
        for nr, nc in _neighbors(grid.shape, r, c):
            if dist[nr, nc] == dist[r, c] - 1 and ways[nr, nc] > 0 and passable(nr, nc):
                preds.append((nr, nc))
        assert len(preds) == 1, "unique shortest path reconstruction failed"
        cur = preds[0]
        path_rev.append(cur)
    path_rev.reverse()
    return path_rev


def _solve_paths(grid: np.ndarray) -> tuple[np.ndarray, dict[Coord, int]]:
    out = grid.copy()
    path_color_by_cell: dict[Coord, int] = {}
    for color, start, goal in _endpoint_pairs(grid):
        path = _unique_shortest_path(grid, start, goal)
        assert len(path) >= 4, "each path should have nontrivial interior"
        for r, c in path[1:-1]:
            assert out[r, c] == BLACK, "path interior must run through background"
            key = (int(r), int(c))
            assert key not in path_color_by_cell, "paths must not overlap"
            path_color_by_cell[key] = color
            out[r, c] = GRAY
    return out, path_color_by_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each endpoint pair's unique shortest path, turn those paths into gray barriers, and fill every newly enclosed black region with the path's endpoint color.

    Input structure:
    - The grid has black background cells, blue wall cells, and 2-3 endpoint color pairs.
    - Each endpoint pair uses a distinct non-blue color and each endpoint is a single pixel.
    - For every same-color pair there is one unique shortest 4-neighbor route through black cells only.

    Rule:
    - Preserve all original blue walls and endpoint pixels.
    - For each color pair, find its unique shortest path while treating blue cells and other endpoints as blocked, then recolor only the path's interior cells gray.
    - Using the resulting gray path pixels together with all non-black cells as barriers, flood from the grid border through black cells; every remaining black component is enclosed and must be filled with the color of the gray path touching its boundary.

    Key insight:
    - The paths are not the final answer by themselves: they act as missing wall segments that create enclosed regions, and the endpoint color of the path determines how that enclosed region is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    with_paths, path_color_by_cell = _solve_paths(g)
    exterior = _exterior_black_mask(with_paths)
    out = with_paths.copy()

    h, w = out.shape
    seen = exterior.copy()
    for r in range(h):
        for c in range(w):
            if out[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            boundary_colors: set[int] = set()
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(out.shape, cr, cc):
                    if out[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif out[nr, nc] == GRAY:
                        boundary_colors.add(path_color_by_cell[(nr, nc)])
            assert len(boundary_colors) == 1, "each enclosed region must touch exactly one path color"
            fill_color = next(iter(boundary_colors))
            for cr, cc in comp:
                out[cr, cc] = fill_color
    return out


def _draw_room(
    grid: np.ndarray,
    color: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
) -> dict[str, object]:
    grid[r0, c0 : c0 + w] = BLUE
    grid[r0 + h - 1, c0 : c0 + w] = BLUE
    grid[r0 : r0 + h, c0] = BLUE
    grid[r0 : r0 + h, c0 + w - 1] = BLUE

    if side in ("top", "bottom"):
        max_gap = min(6, w - 2)
        gap_len = int(np.random.randint(GAP_MIN, max_gap + 1))
        gap_start = int(np.random.randint(1, w - gap_len))
        cols = list(range(c0 + gap_start, c0 + gap_start + gap_len))
        row = r0 if side == "top" else r0 + h - 1
        grid[row, cols] = BLACK
        grid[row, cols[0]] = color
        grid[row, cols[-1]] = color
        endpoints = ((row, cols[0]), (row, cols[-1]))
    else:
        max_gap = min(6, h - 2)
        gap_len = int(np.random.randint(GAP_MIN, max_gap + 1))
        gap_start = int(np.random.randint(1, h - gap_len))
        rows = list(range(r0 + gap_start, r0 + gap_start + gap_len))
        col = c0 if side == "left" else c0 + w - 1
        grid[rows, col] = BLACK
        grid[rows[0], col] = color
        grid[rows[-1], col] = color
        endpoints = ((rows[0], col), (rows[-1], col))

    _add_spurs(grid, r0, c0, h, w, side)
    return {
        "color": color,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "endpoints": endpoints,
    }


def _add_spurs(grid: np.ndarray, r0: int, c0: int, h: int, w: int, gap_side: str) -> None:
    if np.random.random() >= SPUR_PROB:
        return
    side_options = [side for side in ("top", "bottom", "left", "right") if side != gap_side]
    spur_count = int(np.random.randint(1, 3))
    np.random.shuffle(side_options)
    for side in side_options[:spur_count]:
        length = int(np.random.randint(1, 4))
        if side == "top":
            col = int(np.random.randint(c0 + 1, c0 + w - 1))
            for k in range(1, length + 1):
                rr = r0 - k
                if rr < 0 or grid[rr, col] != BLACK:
                    break
                grid[rr, col] = BLUE
        elif side == "bottom":
            col = int(np.random.randint(c0 + 1, c0 + w - 1))
            for k in range(1, length + 1):
                rr = r0 + h - 1 + k
                if rr >= grid.shape[0] or grid[rr, col] != BLACK:
                    break
                grid[rr, col] = BLUE
        elif side == "left":
            row = int(np.random.randint(r0 + 1, r0 + h - 1))
            for k in range(1, length + 1):
                cc = c0 - k
                if cc < 0 or grid[row, cc] != BLACK:
                    break
                grid[row, cc] = BLUE
        else:
            row = int(np.random.randint(r0 + 1, r0 + h - 1))
            for k in range(1, length + 1):
                cc = c0 + w - 1 + k
                if cc >= grid.shape[1] or grid[row, cc] != BLACK:
                    break
                grid[row, cc] = BLUE


def _bbox_clear(grid: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(grid.shape[0], r0 + h + margin)
    cc1 = min(grid.shape[1], c0 + w + margin)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BLACK)


def _random_room_dims(side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        h = int(np.random.randint(ROOM_HEIGHT_RANGE[0], ROOM_HEIGHT_RANGE[1]))
        w = int(np.random.randint(max(ROOM_WIDTH_RANGE[0], GAP_MIN + 2), ROOM_WIDTH_RANGE[1]))
    else:
        h = int(np.random.randint(max(ROOM_HEIGHT_RANGE[0], GAP_MIN + 2), ROOM_HEIGHT_RANGE[1]))
        w = int(np.random.randint(ROOM_WIDTH_RANGE[0], ROOM_WIDTH_RANGE[1]))
    return h, w


def _generate_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*GRID_SIZE_RANGE))
    w = int(np.random.randint(*GRID_SIZE_RANGE))
    grid = np.full((h, w), BLACK, dtype=int)
    room_count = int(np.random.choice(ROOM_COUNT_CHOICES))
    colors = list(np.random.choice(ENDPOINT_COLORS, room_count, replace=False))
    np.random.shuffle(colors)

    sides = np.random.choice(["top", "bottom", "left", "right"], size=room_count, replace=True)
    rooms = []
    for color, side in zip(colors, sides):
        placed = False
        for _ in range(120):
            rh, rw = _random_room_dims(str(side))
            if rh >= h - 2 or rw >= w - 2:
                continue
            r0 = int(np.random.randint(1, h - rh))
            c0 = int(np.random.randint(1, w - rw))
            if not _bbox_clear(grid, r0, c0, rh, rw, ROOM_MARGIN):
                continue
            room = _draw_room(grid, int(color), r0, c0, rh, rw, str(side))
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None

    if not _input_semantics_ok(grid, room_count):
        return None
    return grid


def _input_semantics_ok(grid: np.ndarray, room_count: int) -> bool:
    if len(_black_components(grid)) != 1:
        return False

    try:
        with_paths, path_color_by_cell = _solve_paths(grid)
        out = main(grid)
    except AssertionError:
        return False

    if int(np.sum(with_paths == GRAY)) < room_count * 2:
        return False

    exterior = _exterior_black_mask(with_paths)
    enclosed_comps = []
    seen = exterior.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if with_paths[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            boundary_colors: set[int] = set()
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(grid.shape, cr, cc):
                    if with_paths[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif with_paths[nr, nc] == GRAY:
                        boundary_colors.add(path_color_by_cell[(nr, nc)])
            if len(boundary_colors) != 1:
                return False
            enclosed_comps.append((coords, next(iter(boundary_colors))))

    if len(enclosed_comps) != room_count:
        return False

    if not np.any(exterior):
        return False

    for coords, color in enclosed_comps:
        for r, c in coords:
            if out[r, c] != color:
                return False

    color_set = set(int(x) for x in np.unique(grid) if int(x) not in (BLACK, BLUE))
    return len(color_set) == room_count


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
