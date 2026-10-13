from __future__ import annotations

from collections import Counter
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


Coord = Tuple[int, int]
Direction = Tuple[int, int]


BG = 0
PATH = 3
GREY = 5
PORTAL_COLORS = [1, 2, 4, 6, 7, 8, 9]
DIRS: List[Direction] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _add(cell: Coord, direction: Direction, steps: int = 1) -> Coord:
    return (cell[0] + direction[0] * steps, cell[1] + direction[1] * steps)


def _neg(direction: Direction) -> Direction:
    return (-direction[0], -direction[1])


def _inside(shape: Tuple[int, int], cell: Coord) -> bool:
    h, w = shape
    return 0 <= cell[0] < h and 0 <= cell[1] < w


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _infer_path_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: List[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) < 2:
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        same_row = bool(np.all(rows == rows[0]))
        same_col = bool(np.all(cols == cols[0]))
        if not (same_row or same_col):
            continue

        touches_edge = bool(
            np.any(
                (rows == 0)
                | (rows == h - 1)
                | (cols == 0)
                | (cols == w - 1)
            )
        )
        if not touches_edge:
            continue

        if same_row:
            r = int(rows[0])
            c0, c1 = int(cols.min()), int(cols.max())
            if np.all(grid[r, c0 : c1 + 1] == color):
                candidates.append(color)
        else:
            c = int(cols[0])
            r0, r1 = int(rows.min()), int(rows.max())
            if np.all(grid[r0 : r1 + 1, c] == color):
                candidates.append(color)

    assert len(candidates) == 1, "expected exactly one edge-touching straight path"
    return candidates[0]


def _infer_grey_color(grid: np.ndarray, bg: int, path_color: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    remaining = [int(c) for c in np.unique(grid) if int(c) not in (bg, path_color)]
    assert remaining, "expected grey markers and portals"
    grey = int(max(remaining, key=lambda color: (counts[color], -color)))
    assert counts[grey] > 2, "grey markers should outnumber any single portal pair"
    for color in remaining:
        if color == grey:
            continue
        assert counts[color] == 2, "each portal color must appear exactly twice"
    return grey


def _portal_data(
    grid: np.ndarray, bg: int, path_color: int, grey_color: int
) -> tuple[Dict[Coord, Coord], Dict[Coord, Direction]]:
    pair: Dict[Coord, Coord] = {}
    directions: Dict[Coord, Direction] = {}
    portal_cells: set[Coord] = set()

    for color in np.unique(grid):
        color = int(color)
        if color in (bg, path_color, grey_color):
            continue
        coords = [tuple(map(int, cell)) for cell in np.argwhere(grid == color)]
        assert len(coords) == 2, "every portal color must define exactly one pair"
        a, b = coords
        pair[a] = b
        pair[b] = a
        portal_cells.update(coords)

        for cell in coords:
            dirs = []
            for direction in DIRS:
                marker = _add(cell, direction)
                if _inside(grid.shape, marker) and int(grid[marker]) == grey_color:
                    dirs.append(direction)
            assert len(dirs) == 1, "every portal endpoint needs exactly one grey marker"
            directions[cell] = dirs[0]

    for marker in np.argwhere(grid == grey_color):
        marker_cell = tuple(map(int, marker))
        touches = 0
        for direction in DIRS:
            portal = _add(marker_cell, direction)
            if portal in portal_cells:
                touches += 1
        assert touches == 1, "grey markers must each belong to exactly one endpoint"

    return pair, directions


def _initial_path(grid: np.ndarray, path_color: int) -> tuple[Coord, Coord, Direction]:
    h, w = grid.shape
    coords = np.argwhere(grid == path_color)
    rows = coords[:, 0]
    cols = coords[:, 1]
    same_row = bool(np.all(rows == rows[0]))
    same_col = bool(np.all(cols == cols[0]))
    assert same_row or same_col

    if same_row:
        r = int(rows[0])
        c0, c1 = int(cols.min()), int(cols.max())
        ends = [(r, c0), (r, c1)]
    else:
        c = int(cols[0])
        r0, r1 = int(rows.min()), int(rows.max())
        ends = [(r0, c), (r1, c)]

    edge_ends = [
        cell
        for cell in ends
        if cell[0] in (0, h - 1) or cell[1] in (0, w - 1)
    ]
    assert len(edge_ends) == 1, "the input path must start from exactly one border"

    start = edge_ends[0]
    end = ends[1] if start == ends[0] else ends[0]
    dr = 0 if end[0] == start[0] else (1 if end[0] > start[0] else -1)
    dc = 0 if end[1] == start[1] else (1 if end[1] > start[1] else -1)
    return start, end, (dr, dc)


def _extend_until_event(
    out: np.ndarray,
    pair: Dict[Coord, Coord],
    bg: int,
    path_color: int,
    cursor: Coord,
    direction: Direction,
) -> tuple[str, Coord]:
    r, c = cursor
    dr, dc = direction
    while True:
        nr, nc = r + dr, c + dc
        if not _inside(out.shape, (nr, nc)):
            return ("stop", (r, c))

        if (nr, nc) in pair:
            return ("portal", (nr, nc))

        if int(out[nr, nc]) != bg:
            return ("stop", (r, c))

        out[nr, nc] = int(path_color)
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the teleport path.

    The visible path always enters a portal by directly touching its colored
    endpoint. Each endpoint's grey marker specifies the direction of the
    outgoing ray at the paired endpoint; the first new green cell appears just
    beyond that grey marker so the marker stays visible in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    path_color = _infer_path_color(g, bg)
    grey_color = _infer_grey_color(g, bg, path_color)
    pair, directions = _portal_data(g, bg, path_color, grey_color)
    _, frontier, travel_dir = _initial_path(g, path_color)

    out = g.copy()
    seen: set[tuple[Coord, Direction]] = set()
    cursor = frontier
    direction = travel_dir

    while True:
        event, target = _extend_until_event(out, pair, bg, path_color, cursor, direction)
        if event != "portal":
            break

        state = (target, direction)
        assert state not in seen, "portal chain should not loop forever"
        seen.add(state)

        exit_portal = pair[target]
        direction = directions[exit_portal]
        cursor = _add(exit_portal, direction)
        if not _inside(out.shape, cursor):
            break

    return out


def _free(shape: Tuple[int, int], reserved: set[Coord], cells: Iterable[Coord]) -> bool:
    for cell in cells:
        if not _inside(shape, cell) or cell in reserved:
            return False
    return True


def _interior_cells(h: int, w: int) -> List[Coord]:
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _available_grey_dirs(
    shape: Tuple[int, int],
    reserved: set[Coord],
    portal: Coord,
    blocked_side: Direction,
) -> List[Direction]:
    dirs = []
    for direction in DIRS:
        if direction == blocked_side:
            continue
        marker = _add(portal, direction)
        if _free(shape, reserved, [marker]):
            dirs.append(direction)
    return dirs


def _pick_initial_entry(
    shape: Tuple[int, int], reserved: set[Coord]
) -> tuple[Coord, Direction, List[Coord]]:
    h, w = shape
    for _ in range(300):
        direction = random.choice(DIRS)
        if direction == (0, 1):
            portal = (random.randrange(1, h - 1), random.randrange(2, w - 2))
            path_cells = [(portal[0], c) for c in range(0, portal[1])]
        elif direction == (0, -1):
            portal = (random.randrange(1, h - 1), random.randrange(1, w - 3))
            path_cells = [(portal[0], c) for c in range(portal[1] + 1, w)]
        elif direction == (1, 0):
            portal = (random.randrange(2, h - 2), random.randrange(1, w - 1))
            path_cells = [(r, portal[1]) for r in range(0, portal[0])]
        else:
            portal = (random.randrange(1, h - 3), random.randrange(1, w - 1))
            path_cells = [(r, portal[1]) for r in range(portal[0] + 1, h)]

        blocked_side = _neg(direction)
        temp_reserved = reserved | {portal} | set(path_cells)
        if not _free(shape, reserved, [portal, *path_cells]):
            continue
        if not _available_grey_dirs(shape, temp_reserved, portal, blocked_side):
            continue
        return portal, direction, path_cells
    raise RuntimeError("failed to place the initial path")


def _distance_to_inner_edge(shape: Tuple[int, int], cell: Coord, direction: Direction) -> int:
    h, w = shape
    if direction == (-1, 0):
        return cell[0] - 1
    if direction == (1, 0):
        return h - 2 - cell[0]
    if direction == (0, -1):
        return cell[1] - 1
    return w - 2 - cell[1]


def _pick_exit_with_next_entry(
    shape: Tuple[int, int], reserved: set[Coord]
) -> tuple[Coord, Direction, List[Coord], Coord]:
    cells = _interior_cells(*shape)
    for _ in range(600):
        exit_portal = random.choice(cells)
        if exit_portal in reserved:
            continue

        directions = DIRS[:]
        random.shuffle(directions)
        for direction in directions:
            exit_marker = _add(exit_portal, direction)
            max_steps = _distance_to_inner_edge(shape, exit_portal, direction)
            if max_steps < 3:
                continue

            lengths = list(range(3, max_steps + 1))
            random.shuffle(lengths)
            for length in lengths:
                next_entry = _add(exit_portal, direction, length)
                path_cells = [_add(exit_portal, direction, step) for step in range(2, length)]
                temp_reserved = reserved | {exit_portal, exit_marker, next_entry} | set(path_cells)
                if not _free(shape, reserved, [exit_portal, exit_marker, next_entry, *path_cells]):
                    continue
                if not _available_grey_dirs(shape, temp_reserved, next_entry, _neg(direction)):
                    continue
                return exit_portal, direction, path_cells, next_entry
    raise RuntimeError("failed to place a portal-to-portal segment")


def _pick_exit_to_boundary(
    shape: Tuple[int, int], reserved: set[Coord]
) -> tuple[Coord, Direction, List[Coord]]:
    cells = _interior_cells(*shape)
    for _ in range(400):
        exit_portal = random.choice(cells)
        if exit_portal in reserved:
            continue

        directions = DIRS[:]
        random.shuffle(directions)
        for direction in directions:
            exit_marker = _add(exit_portal, direction)
            if not _free(shape, reserved, [exit_portal, exit_marker]):
                continue

            ray_cells: List[Coord] = []
            cur = _add(exit_portal, direction, 2)
            while _inside(shape, cur):
                ray_cells.append(cur)
                cur = _add(cur, direction)

            if ray_cells and _free(shape, reserved, ray_cells):
                return exit_portal, direction, ray_cells
    raise RuntimeError("failed to place a boundary-ending segment")


def _pick_exit_to_obstacle(
    shape: Tuple[int, int], reserved: set[Coord]
) -> tuple[Coord, Direction, List[Coord], Coord, Coord]:
    cells = _interior_cells(*shape)
    for _ in range(600):
        exit_portal = random.choice(cells)
        if exit_portal in reserved:
            continue

        directions = DIRS[:]
        random.shuffle(directions)
        for direction in directions:
            exit_marker = _add(exit_portal, direction)
            max_steps = _distance_to_inner_edge(shape, exit_portal, direction)
            if max_steps < 3 or not _free(shape, reserved, [exit_portal, exit_marker]):
                continue

            lengths = list(range(3, max_steps + 1))
            random.shuffle(lengths)
            for length in lengths:
                obstacle_marker = _add(exit_portal, direction, length)
                path_cells = [_add(exit_portal, direction, step) for step in range(2, length)]
                if not _free(shape, reserved, [obstacle_marker, *path_cells]):
                    continue

                perps = [(-direction[1], direction[0]), (direction[1], -direction[0])]
                random.shuffle(perps)
                for perp in perps:
                    obstacle_portal = _add(obstacle_marker, perp)
                    if _free(shape, reserved, [obstacle_portal]):
                        return exit_portal, direction, path_cells, obstacle_marker, obstacle_portal
    raise RuntimeError("failed to place an obstacle-ending segment")


def _pick_free_endpoint(shape: Tuple[int, int], reserved: set[Coord]) -> tuple[Coord, Direction]:
    cells = _interior_cells(*shape)
    for _ in range(400):
        portal = random.choice(cells)
        if portal in reserved:
            continue
        directions = DIRS[:]
        random.shuffle(directions)
        for direction in directions:
            marker = _add(portal, direction)
            if _free(shape, reserved, [marker]):
                return portal, direction
    raise RuntimeError("failed to place a distractor endpoint")


def _render(
    shape: Tuple[int, int],
    portal_colors: Dict[Coord, int],
    portal_dirs: Dict[Coord, Direction],
    input_path: set[Coord],
    extra_path: set[Coord],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)

    for cell, color in portal_colors.items():
        inp[cell] = int(color)
        out[cell] = int(color)

    for cell, direction in portal_dirs.items():
        marker = _add(cell, direction)
        inp[marker] = GREY
        out[marker] = GREY

    for cell in input_path:
        inp[cell] = PATH
        out[cell] = PATH
    for cell in extra_path:
        out[cell] = PATH

    return inp, out


def _unambiguous_portals(portal_dirs: Dict[Coord, Direction]) -> bool:
    portals = set(portal_dirs)
    greys = {_add(cell, direction) for cell, direction in portal_dirs.items()}

    for portal in portals:
        grey_count = sum(_add(portal, direction) in greys for direction in DIRS)
        if grey_count != 1:
            return False

    for marker in greys:
        portal_count = sum(_add(marker, direction) in portals for direction in DIRS)
        if portal_count != 1:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    shape = (h, w)

    chain_len = random.randint(1, 3)
    final_branch = random.choice(["boundary", "obstacle"])
    min_extra = 1 if chain_len == 1 and final_branch == "boundary" else 0
    extra_pairs = random.randint(min_extra, 2)
    obstacle_pairs = 1 if final_branch == "obstacle" else 0
    total_pairs = chain_len + extra_pairs + obstacle_pairs
    if total_pairs > len(PORTAL_COLORS):
        return None

    colors = random.sample(PORTAL_COLORS, total_pairs)
    used_colors = colors[:chain_len]
    next_color_idx = chain_len

    reserved: set[Coord] = set()
    input_path: set[Coord] = set()
    extra_path: set[Coord] = set()
    portal_colors: Dict[Coord, int] = {}
    portal_dirs: Dict[Coord, Direction] = {}
    endpoints_by_color: Dict[int, List[Coord]] = {color: [] for color in colors}

    try:
        current_entry, incoming, start_path = _pick_initial_entry(shape, reserved)
    except RuntimeError:
        return None

    input_path.update(start_path)
    reserved.update(start_path)
    reserved.add(current_entry)

    for color in used_colors:
        entry_options = _available_grey_dirs(shape, reserved, current_entry, _neg(incoming))
        if not entry_options:
            return None
        entry_dir = random.choice(entry_options)
        portal_dirs[current_entry] = entry_dir
        portal_colors[current_entry] = color
        endpoints_by_color[color].append(current_entry)
        reserved.add(_add(current_entry, entry_dir))

        is_last = color == used_colors[-1]
        try:
            if not is_last:
                exit_portal, exit_dir, segment_cells, next_entry = _pick_exit_with_next_entry(shape, reserved)
                extra_path.update(segment_cells)
                reserved.update(segment_cells)
                reserved.add(next_entry)
                current_entry = next_entry
                incoming = exit_dir
            elif final_branch == "boundary":
                exit_portal, exit_dir, segment_cells = _pick_exit_to_boundary(shape, reserved)
                extra_path.update(segment_cells)
                reserved.update(segment_cells)
            else:
                (
                    exit_portal,
                    exit_dir,
                    segment_cells,
                    obstacle_marker,
                    obstacle_portal,
                ) = _pick_exit_to_obstacle(shape, reserved)
                extra_path.update(segment_cells)
                reserved.update(segment_cells)

                obstacle_color = colors[next_color_idx]
                next_color_idx += 1
                obstacle_dir = (obstacle_marker[0] - obstacle_portal[0], obstacle_marker[1] - obstacle_portal[1])
                portal_colors[obstacle_portal] = obstacle_color
                portal_dirs[obstacle_portal] = obstacle_dir
                endpoints_by_color[obstacle_color].append(obstacle_portal)
                reserved.add(obstacle_portal)
                reserved.add(obstacle_marker)
        except RuntimeError:
            return None

        if exit_portal in reserved:
            pass
        portal_colors[exit_portal] = color
        portal_dirs[exit_portal] = exit_dir
        endpoints_by_color[color].append(exit_portal)
        reserved.add(exit_portal)
        reserved.add(_add(exit_portal, exit_dir))

    remaining_colors = colors[next_color_idx:]
    for color in remaining_colors:
        for _ in range(2 - len(endpoints_by_color[color])):
            try:
                portal, direction = _pick_free_endpoint(shape, reserved)
            except RuntimeError:
                return None
            portal_colors[portal] = color
            portal_dirs[portal] = direction
            endpoints_by_color[color].append(portal)
            reserved.add(portal)
            reserved.add(_add(portal, direction))

    for color, endpoints in endpoints_by_color.items():
        if len(endpoints) != 2:
            return None

    inp, expected = _render(shape, portal_colors, portal_dirs, input_path, extra_path)

    if not _unambiguous_portals(portal_dirs):
        return None
    if _dominant_color(inp) != BG:
        return None
    if len(extra_path) < 2:
        return None
    if np.count_nonzero(inp == PATH) < 2:
        return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport puzzle")
