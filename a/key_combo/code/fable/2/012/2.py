from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
NON_SPECIAL_COLORS = (1, 3, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 300


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_monochromatic_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~mask) & (~exterior)))


def _wall_mask(grid: np.ndarray) -> np.ndarray:
    blockers = np.zeros_like(grid, dtype=bool)
    for component in _find_monochromatic_components(grid):
        if _has_enclosed_hole(_component_mask(component)):
            for r, c in component["coords"]:
                blockers[r, c] = True
    return blockers


def _unique_shortest_path(passable: np.ndarray, start: Coord, goal: Coord) -> list[Coord] | None:
    h, w = map(int, passable.shape)
    ds = np.full((h, w), -1, dtype=int)
    q: deque[Coord] = deque([start])
    ds[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and ds[nr, nc] < 0:
                ds[nr, nc] = ds[r, c] + 1
                q.append((nr, nc))
    if ds[goal] < 0:
        return None

    cells = [tuple(map(int, rc)) for rc in np.argwhere(ds >= 0)]
    cells.sort(key=lambda rc: int(ds[rc]))
    ways = np.zeros((h, w), dtype=int)
    ways[start] = 1
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and ds[nr, nc] == ds[r, c] + 1:
                ways[nr, nc] = min(2, ways[nr, nc] + ways[r, c])
    if int(ways[goal]) != 1:
        return None

    dg = np.full((h, w), -1, dtype=int)
    q = deque([goal])
    dg[goal] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dg[nr, nc] < 0:
                dg[nr, nc] = dg[r, c] + 1
                q.append((nr, nc))

    total_dist = int(ds[goal])
    path = [start]
    cur = start
    while cur != goal:
        r, c = cur
        candidates = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and ds[nr, nc] == ds[r, c] + 1 and dg[nr, nc] == total_dist - ds[nr, nc]
        ]
        if len(candidates) != 1:
            return None
        cur = candidates[0]
        path.append(cur)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the unique shortest path between the two red endpoints while only hole-containing objects act as walls.

    Input structure:
    - Black background with exactly two red endpoint pixels.
    - Several monochromatic 4-connected foreground objects; some enclose at least one background hole, others are open line-like shapes with no hole.

    Rule:
    - Classify each foreground object only by topology: an object with an enclosed hole is impassable, while an object without a hole is passable just like the background.
    - Find the unique shortest 4-neighbor path between the two red endpoints through all non-wall cells.
    - Recolor every non-endpoint cell on that path yellow, including any traversed open-object pixels; keep everything else unchanged.

    Key insight:
    - Color, size, and position are decoys: the only wall criterion is whether a connected object encloses a hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red_coords[0]))
    goal = tuple(map(int, red_coords[1]))

    blockers = _wall_mask(g)
    assert not blockers[start] and not blockers[goal], "endpoints must stay passable"
    path = _unique_shortest_path(~blockers, start, goal)
    assert path is not None, "expected a unique shortest path between the endpoints"

    out = g.copy()
    for r, c in path[1:-1]:
        out[r, c] = YELLOW
    return out


def _carve_perfect_maze(cell_rows: int, cell_cols: int) -> np.ndarray:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    maze = np.ones((h, w), dtype=int)

    seen = np.zeros((cell_rows, cell_cols), dtype=bool)
    start = (random.randrange(cell_rows), random.randrange(cell_cols))
    stack = [start]
    seen[start] = True
    maze[2 * start[0] + 1, 2 * start[1] + 1] = 0

    while stack:
        r, c = stack[-1]
        options: list[tuple[int, int, int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not seen[nr, nc]:
                options.append((nr, nc, dr, dc))
        if not options:
            stack.pop()
            continue

        nr, nc, dr, dc = random.choice(options)
        seen[nr, nc] = True
        maze[2 * nr + 1, 2 * nc + 1] = 0
        maze[2 * r + 1 + dr, 2 * c + 1 + dc] = 0
        stack.append((nr, nc))
    return maze


def _corridor_leaves(open_mask: np.ndarray) -> list[Coord]:
    h, w = map(int, open_mask.shape)
    leaves: list[Coord] = []
    for r, c in np.argwhere(open_mask):
        rr = int(r)
        cc = int(c)
        degree = sum(1 for nr, nc in _neighbors4(rr, cc) if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc])
        if degree == 1:
            leaves.append((rr, cc))
    return leaves


def _long_leaf_path(maze_open: np.ndarray) -> list[Coord] | None:
    leaves = _corridor_leaves(maze_open)
    best_path: list[Coord] | None = None
    for i, start in enumerate(leaves):
        for goal in leaves[i + 1 :]:
            path = _unique_shortest_path(maze_open, start, goal)
            if path is not None and (best_path is None or len(path) > len(best_path)):
                best_path = path
    return best_path


def _outside_region(shape: tuple[int, int], maze_bbox: tuple[int, int, int, int], gap: int = 1) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = maze_bbox
    allowed = np.ones((h, w), dtype=bool)
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h - 1, r1 + gap)
    cc1 = min(w - 1, c1 + gap)
    allowed[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
    return allowed


def _reserve_halo(mask: np.ndarray, coords: list[Coord]) -> None:
    h, w = map(int, mask.shape)
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _place_outside_ring(grid: np.ndarray, reserved: np.ndarray, allowed: np.ndarray, color: int) -> list[Coord] | None:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, int]] = []
    for height in (3, 4, 5):
        for width in (3, 4, 5):
            for r0 in range(h - height + 1):
                r1 = r0 + height - 1
                for c0 in range(w - width + 1):
                    c1 = c0 + width - 1
                    if not np.all(allowed[r0 : r1 + 1, c0 : c1 + 1]):
                        continue
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(h, r1 + 2)
                    cc1 = min(w, c1 + 2)
                    if np.any(reserved[rr0:rr1, cc0:cc1]):
                        continue
                    candidates.append((r0, c0, r1, c1))
    if not candidates:
        return None

    r0, c0, r1, c1 = random.choice(candidates)
    cells: list[Coord] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                grid[r, c] = color
                cells.append((r, c))
    _reserve_halo(reserved, cells)
    return cells


def _place_outside_open_walk(
    grid: np.ndarray,
    reserved: np.ndarray,
    allowed: np.ndarray,
    color: int,
    min_len: int = 2,
    max_len: int = 5,
) -> list[Coord] | None:
    h, w = map(int, grid.shape)
    starts = [tuple(map(int, rc)) for rc in np.argwhere(allowed & (~reserved))]
    random.shuffle(starts)
    for start in starts:
        target_len = random.randint(min_len, max_len)
        path = [start]
        used = {start}
        while len(path) < target_len:
            r, c = path[-1]
            choices: list[Coord] = []
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not allowed[nr, nc] or reserved[nr, nc] or (nr, nc) in used:
                    continue
                if any((ar, ac) in used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
                    continue
                choices.append((nr, nc))
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            used.add(nxt)
        if len(path) >= min_len:
            for r, c in path:
                grid[r, c] = color
            _reserve_halo(reserved, path)
            return path
    return None


def _choose_path_segment(path: list[Coord], min_len: int = 2, max_len: int = 5) -> list[Coord] | None:
    interior = path[2:-2]
    if len(interior) < min_len:
        return None

    candidates: list[list[Coord]] = []
    for seg_len in range(min_len, min(max_len, len(interior)) + 1):
        for start_idx in range(0, len(interior) - seg_len + 1):
            candidates.append(interior[start_idx : start_idx + seg_len])
    return random.choice(candidates) if candidates else None


def _build_expected_output(grid: np.ndarray, path: list[Coord]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in path[1:-1]:
        out[r, c] = YELLOW
    return out


def _validate_generated_case(
    grid: np.ndarray,
    intended_blockers: np.ndarray,
    intended_path: list[Coord],
    on_path_open: set[Coord],
    off_path_open: set[Coord],
) -> bool:
    if np.argwhere(grid == RED).shape != (2, 2):
        return False

    reds = np.argwhere(grid == RED)
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))
    derived_blockers = _wall_mask(grid)
    if not np.array_equal(derived_blockers, intended_blockers):
        return False

    path = _unique_shortest_path(~intended_blockers, start, goal)
    if path is None:
        return False
    if path != intended_path and path != intended_path[::-1]:
        return False
    if not on_path_open or not on_path_open.issubset(set(path)):
        return False
    if off_path_open & set(path):
        return False

    wrong_blockers = (grid != BLACK) & (grid != RED)
    wrong_path = _unique_shortest_path(~wrong_blockers, start, goal)
    if wrong_path is not None and (wrong_path == intended_path or wrong_path == intended_path[::-1]):
        return False

    expected = _build_expected_output(grid, intended_path)
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        cell_rows = random.randint(5, 9)
        cell_cols = random.randint(5, 9)
        maze_open = _carve_perfect_maze(cell_rows, cell_cols)
        maze_path = _long_leaf_path(maze_open == 0)
        if maze_path is None or len(maze_path) < 15:
            continue

        maze_h, maze_w = map(int, maze_open.shape)
        top = random.randint(2, 5)
        left = random.randint(2, 5)
        bottom = random.randint(2, 5)
        right = random.randint(2, 5)
        height = maze_h + top + bottom
        width = maze_w + left + right
        if height > 30 or width > 30:
            continue

        grid = np.zeros((height, width), dtype=int)
        maze_bbox = (top, left, top + maze_h - 1, left + maze_w - 1)
        maze_color = random.choice(NON_SPECIAL_COLORS)
        shared_color = random.choice([color for color in NON_SPECIAL_COLORS if color != maze_color])

        maze_patch = grid[top : top + maze_h, left : left + maze_w]
        maze_patch[maze_open == 1] = maze_color
        intended_blockers = grid != BLACK

        full_path = [(top + r, left + c) for r, c in maze_path]
        start = full_path[0]
        goal = full_path[-1]
        on_path_segment = _choose_path_segment(full_path)
        if on_path_segment is None:
            continue
        for r, c in on_path_segment:
            grid[r, c] = shared_color

        grid[start] = RED
        grid[goal] = RED

        reserved = np.zeros_like(grid, dtype=bool)
        maze_outside_allowed = _outside_region(grid.shape, maze_bbox, gap=1)
        _reserve_halo(reserved, [(maze_bbox[0], maze_bbox[1]), (maze_bbox[0], maze_bbox[3]), (maze_bbox[2], maze_bbox[1]), (maze_bbox[2], maze_bbox[3])])
        reserved[max(0, maze_bbox[0] - 1) : min(height, maze_bbox[2] + 2), max(0, maze_bbox[1] - 1) : min(width, maze_bbox[3] + 2)] = True

        ring_cells = _place_outside_ring(grid, reserved, maze_outside_allowed, shared_color)
        if ring_cells is None:
            continue
        for r, c in ring_cells:
            intended_blockers[r, c] = True

        outside_open = _place_outside_open_walk(
            grid,
            reserved,
            maze_outside_allowed,
            random.choice(NON_SPECIAL_COLORS),
            min_len=2,
            max_len=5,
        )
        if outside_open is None:
            continue

        if random.random() < 0.5:
            _place_outside_open_walk(
                grid,
                reserved,
                maze_outside_allowed,
                random.choice(NON_SPECIAL_COLORS),
                min_len=2,
                max_len=4,
            )

        if not _validate_generated_case(
            grid,
            intended_blockers,
            full_path,
            set(on_path_segment),
            set(outside_open),
        ):
            continue
        return grid

    raise RuntimeError("failed to generate a valid topological-maze input")
