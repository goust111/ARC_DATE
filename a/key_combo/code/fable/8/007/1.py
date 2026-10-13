from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GRAY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)

GEN_MAX_ATTEMPTS = 500
GEN_CELL_SIDE_RANGE = (5, 13)
GEN_OBJECT_SIZES = (1, 2, 2, 3, 3, 4, 5)
GEN_EXTRA_LOOP_DIVISOR = 7
GEN_COLOR_COUNT_RANGE = (3, 6)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bfs_distances(passable: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _gray_component(grid: np.ndarray) -> np.ndarray:
    comps = _components4(grid == GRAY)
    assert len(comps) == 1, "expected exactly one gray 4-connected object"
    return comps[0]


def _marker_groups(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, BLUE, GRAY):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert coords, "marker color should have at least one pixel"
        groups[color] = coords
    assert groups, "expected at least one marker color"
    return groups


def _solve_details(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    gray_coords = _gray_component(g)
    markers = _marker_groups(g)
    passable = g != BLUE
    gray_list = [tuple(map(int, rc)) for rc in gray_coords]
    dist = _bfs_distances(passable, gray_list)

    color_dists: dict[int, int] = {}
    for color, coords in markers.items():
        steps = [int(dist[r, c]) for r, c in coords if dist[r, c] >= 0]
        color_dists[color] = min(steps) if steps else -1

    reachable = {color: d for color, d in color_dists.items() if d >= 0}
    assert reachable, "expected at least one reachable marker color"
    best = min(reachable.values())
    winners = [color for color, d in reachable.items() if d == best]
    assert len(winners) == 1, "expected a unique geodesic nearest color"
    return winners[0], gray_coords, dist, color_dists


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray object with the uniquely geodesic-nearest marker color, then erase all markers.

    Input structure:
    - The grid has a black background with blue wall pixels that form a maze of 4-neighbor corridors.
    - There is exactly one gray 4-connected foreground object.
    - Several other colors appear only as scattered marker pixels, with one or more pixels per marker color.

    Rule:
    - For each marker color, measure the shortest 4-neighbor path length from any pixel of that color to any gray-object pixel.
    - Paths may use any non-blue cells; colors whose markers cannot reach the gray object through non-blue cells are ignored.
    - Recolor every gray-object pixel with the unique reachable color of minimum geodesic distance, keep blue walls, erase every marker pixel, and leave all other cells black.

    Key insight:
    - The deciding cue is maze reachability distance through non-blue corridors, not straight-line proximity, so a visually nearer marker can lose or be fully cut off by blue walls.
    """
    winner, gray_coords, _dist, _color_dists = _solve_details(grid)
    out = np.zeros_like(np.asarray(grid, dtype=int))
    out[np.asarray(grid) == BLUE] = BLUE
    for r, c in gray_coords:
        out[int(r), int(c)] = winner
    return out


def _make_maze(cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray:
    h = 2 * cell_h + 1
    w = 2 * cell_w + 1
    grid = np.full((h, w), BLUE, dtype=int)
    visited = np.zeros((cell_h, cell_w), dtype=bool)
    start = (int(rng.integers(cell_h)), int(rng.integers(cell_w)))
    stack = [start]
    visited[start] = True
    grid[2 * start[0] + 1, 2 * start[1] + 1] = BLACK

    while stack:
        cr, cc = stack[-1]
        options = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and not visited[nr, nc]:
                options.append((nr, nc, dr, dc))
        if not options:
            stack.pop()
            continue
        nr, nc, dr, dc = options[int(rng.integers(len(options)))]
        visited[nr, nc] = True
        grid[2 * cr + 1 + dr, 2 * cc + 1 + dc] = BLACK
        grid[2 * nr + 1, 2 * nc + 1] = BLACK
        stack.append((nr, nc))

    max_loops = max(1, (cell_h * cell_w) // GEN_EXTRA_LOOP_DIVISOR)
    for _ in range(int(rng.integers(max_loops + 1))):
        if rng.random() < 0.5:
            rr = int(rng.integers(cell_h))
            cc = int(rng.integers(max(1, cell_w - 1)))
            wall = (2 * rr + 1, 2 * cc + 2)
        else:
            rr = int(rng.integers(max(1, cell_h - 1)))
            cc = int(rng.integers(cell_w))
            wall = (2 * rr + 2, 2 * cc + 1)
        grid[wall] = BLACK
    return grid


def _grow_object(passable: np.ndarray, size: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(passable)]
    rng.shuffle(cells)
    for start in cells:
        chosen = {start}
        frontier = [start]
        while frontier and len(chosen) < size:
            base = frontier[int(rng.integers(len(frontier)))]
            nbrs = []
            for nr, nc in _neighbors4(*base):
                if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and (nr, nc) not in chosen:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = nbrs[int(rng.integers(len(nbrs)))]
            chosen.add(nxt)
            frontier.append(nxt)
        if len(chosen) == size:
            return list(chosen)
    return None


def _manhattan_to_object(gray_coords: np.ndarray, pos: tuple[int, int]) -> int:
    r, c = pos
    return min(abs(r - int(gr)) + abs(c - int(gc)) for gr, gc in gray_coords)


def _cell_info(
    grid: np.ndarray, gray_coords: np.ndarray, dist: np.ndarray
) -> list[dict[str, int | tuple[int, int]]]:
    info: list[dict[str, int | tuple[int, int]]] = []
    gray_set = {tuple(map(int, rc)) for rc in gray_coords}
    for r, c in np.argwhere(grid != BLUE):
        pos = (int(r), int(c))
        if pos in gray_set:
            continue
        d = int(dist[pos])
        if d < 0:
            continue
        info.append({"pos": pos, "dist": d, "manh": _manhattan_to_object(gray_coords, pos)})
    return info


def _reachable_pair(
    info: list[dict[str, int | tuple[int, int]]], rng: np.random.Generator
) -> tuple[dict[str, int | tuple[int, int]], dict[str, int | tuple[int, int]]] | None:
    order = list(range(len(info)))
    rng.shuffle(order)
    for idx in order:
        winner = info[idx]
        if int(winner["dist"]) < 4 or int(winner["manh"]) < 3:
            continue
        decoys = [
            cand
            for cand in info
            if cand["pos"] != winner["pos"]
            and int(cand["manh"]) < int(winner["manh"])
            and int(cand["dist"]) > int(winner["dist"]) + 1
        ]
        if decoys:
            return winner, decoys[int(rng.integers(len(decoys)))]
    return None


def _isolated_marker_sites(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sites = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLUE:
                continue
            if np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == BLUE):
                sites.append((r, c))
    return sites


def _choose_sealed_decoy(
    grid: np.ndarray,
    gray_coords: np.ndarray,
    info: list[dict[str, int | tuple[int, int]]],
    rng: np.random.Generator,
) -> tuple[dict[str, int | tuple[int, int]], tuple[int, int]] | None:
    sites = _isolated_marker_sites(grid)
    if not sites:
        return None
    order = list(range(len(info)))
    rng.shuffle(order)
    for idx in order:
        winner = info[idx]
        if int(winner["dist"]) < 4 or int(winner["manh"]) < 4:
            continue
        close_sites = [pos for pos in sites if _manhattan_to_object(gray_coords, pos) < int(winner["manh"])]
        if close_sites:
            return winner, close_sites[int(rng.integers(len(close_sites)))]
    return None


def _spaced(pos: tuple[int, int], used: set[tuple[int, int]], min_manh: int = 2) -> bool:
    return all(abs(pos[0] - r) + abs(pos[1] - c) >= min_manh for r, c in used)


def _pick_extra_open_cell(
    info: list[dict[str, int | tuple[int, int]]],
    used: set[tuple[int, int]],
    min_dist: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    choices = [
        tuple(cand["pos"])
        for cand in info
        if int(cand["dist"]) >= min_dist and tuple(cand["pos"]) not in used and _spaced(tuple(cand["pos"]), used)
    ]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _pick_extra_isolated_cell(
    grid: np.ndarray, used: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[int, int] | None:
    choices = [pos for pos in _isolated_marker_sites(grid) if pos not in used and _spaced(pos, used)]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _expected_output(grid: np.ndarray) -> np.ndarray:
    winner, gray_coords, _dist, _color_dists = _solve_details(grid)
    out = np.zeros_like(grid)
    out[grid == BLUE] = BLUE
    for r, c in gray_coords:
        out[int(r), int(c)] = winner
    return out


def _validate_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        winner, gray_coords, dist, color_dists = _solve_details(grid)
    except AssertionError:
        return False

    if len(color_dists) < 3:
        return False
    reachable = {color: d for color, d in color_dists.items() if d >= 0}
    if len(reachable) < 2:
        return False
    win_dist = reachable[winner]
    win_manh = min(
        _manhattan_to_object(gray_coords, pos)
        for pos in [tuple(map(int, rc)) for rc in np.argwhere(grid == winner)]
    )

    decoy_found = False
    for color, d in color_dists.items():
        if color == winner:
            continue
        for pos in [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]:
            manh = _manhattan_to_object(gray_coords, pos)
            if manh < win_manh and (d < 0 or d > win_dist):
                decoy_found = True
                break
        if decoy_found:
            break
    if not decoy_found:
        return False

    if not np.array_equal(main(grid), _expected_output(grid)):
        return False

    out = main(grid)
    if np.any(out == GRAY):
        return False
    if np.any((out != BLACK) & (out != BLUE) & (out != winner)):
        return False
    return True


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    cell_h = int(rng.integers(*GEN_CELL_SIDE_RANGE))
    cell_w = int(rng.integers(*GEN_CELL_SIDE_RANGE))
    grid = _make_maze(cell_h, cell_w, rng)

    passable = grid != BLUE
    obj_size = int(rng.choice(np.array(GEN_OBJECT_SIZES)))
    gray_cells = _grow_object(passable, obj_size, rng)
    if gray_cells is None:
        return None
    for r, c in gray_cells:
        grid[r, c] = GRAY

    gray_coords = _gray_component(grid)
    dist = _bfs_distances(grid != BLUE, [tuple(map(int, rc)) for rc in gray_coords])
    info = _cell_info(grid, gray_coords, dist)
    if len(info) < 8:
        return None

    use_sealed_decoy = bool(rng.integers(2))
    winner_info: dict[str, int | tuple[int, int]] | None = None
    decoy_open: dict[str, int | tuple[int, int]] | None = None
    decoy_sealed: tuple[int, int] | None = None

    if not use_sealed_decoy:
        pair = _reachable_pair(info, rng)
        if pair is not None:
            winner_info, decoy_open = pair
        else:
            sealed = _choose_sealed_decoy(grid, gray_coords, info, rng)
            if sealed is None:
                return None
            winner_info, decoy_sealed = sealed
    else:
        sealed = _choose_sealed_decoy(grid, gray_coords, info, rng)
        if sealed is not None:
            winner_info, decoy_sealed = sealed
        else:
            pair = _reachable_pair(info, rng)
            if pair is None:
                return None
            winner_info, decoy_open = pair

    assert winner_info is not None
    winner_pos = tuple(winner_info["pos"])
    win_dist = int(winner_info["dist"])

    n_colors = int(rng.integers(*GEN_COLOR_COUNT_RANGE))
    colors = list(map(int, rng.choice(np.array(MARKER_COLORS), size=n_colors, replace=False)))
    winner_color = colors[0]
    used: set[tuple[int, int]] = set(gray_cells)

    grid[winner_pos] = winner_color
    used.add(winner_pos)

    if rng.random() < 0.7:
        extra_count = 1 + int(rng.random() < 0.25)
        for _ in range(extra_count):
            extra = _pick_extra_open_cell(info, used, win_dist + 2, rng)
            if extra is None:
                break
            grid[extra] = winner_color
            used.add(extra)

    decoy_color = colors[1]
    if decoy_open is not None:
        pos = tuple(decoy_open["pos"])
        grid[pos] = decoy_color
        used.add(pos)
    else:
        assert decoy_sealed is not None
        grid[decoy_sealed] = decoy_color
        used.add(decoy_sealed)

    for color in colors[2:]:
        if rng.random() < 0.4:
            pos = _pick_extra_isolated_cell(grid, used, rng)
        else:
            pos = _pick_extra_open_cell(info, used, win_dist + 2, rng)
            if pos is None:
                pos = _pick_extra_isolated_cell(grid, used, rng)
        if pos is None:
            return None
        grid[pos] = color
        used.add(pos)
        if rng.random() < 0.45:
            extra = _pick_extra_open_cell(info, used, win_dist + 3, rng)
            if extra is not None:
                grid[extra] = color
                used.add(extra)

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-geodesic recolor input")
