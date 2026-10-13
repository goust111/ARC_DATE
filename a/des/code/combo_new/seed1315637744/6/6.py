from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
WHITE = 8
COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _components_for_colors(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((h, w), rr, cc):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _white_roles(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    white_components = _components_for_colors(grid, {WHITE})
    start: tuple[int, int] | None = None
    goal_cells: set[tuple[int, int]] | None = None

    for comp in white_components:
        coords = comp["coords"]
        if len(coords) == 1:
            start = coords[0]
            continue
        if len(coords) == 4:
            rows = sorted({r for r, _c in coords})
            cols = sorted({c for _r, c in coords})
            if rows == list(range(rows[0], rows[0] + 2)) and cols == list(range(cols[0], cols[0] + 2)):
                goal_cells = set(coords)

    assert start is not None, "expected one white start pixel"
    assert goal_cells is not None, "expected one white 2x2 goal block"
    return start, goal_cells


def _is_line_component(coords: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1:
        ordered = sorted(c for _r, c in coords)
        return ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    if len(cols) == 1:
        ordered = sorted(r for r, _c in coords)
        return ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    return False


def _line_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1:
        ordered = sorted(c for _r, c in coords)
        return rows[0], ordered[len(ordered) // 2]
    ordered = sorted(r for r, _c in coords)
    return ordered[len(ordered) // 2], cols[0]


def _path_to_goal(grid: np.ndarray, start: tuple[int, int], goal_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    end: tuple[int, int] | None = None

    while q:
        cell = q.popleft()
        if cell in goal_cells:
            end = cell
            break
        r, c = cell
        for nr, nc in _neighbors4((h, w), r, c):
            if (nr, nc) in parent or int(grid[nr, nc]) == WALL:
                continue
            parent[(nr, nc)] = cell
            q.append((nr, nc))

    assert end is not None, "start must reach the goal"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _component_path_index(
    coords: list[tuple[int, int]], path_index: dict[tuple[int, int], int]
) -> int | None:
    hits = [path_index[coord] for coord in coords if coord in path_index]
    if not hits:
        return None
    return min(hits)


def _key_anchor(coords: list[tuple[int, int]]) -> tuple[int, int]:
    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degrees[(r, c)] = sum((r + dr, c + dc) in coord_set for dr, dc in DIRS4)

    best = max(degrees.values())
    candidates = [coord for coord, degree in degrees.items() if degree == best]
    if len(candidates) == 1:
        return candidates[0]

    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    target_r = (min(rs) + max(rs)) / 2.0
    target_c = (min(cs) + max(cs)) / 2.0
    candidates.sort(key=lambda rc: (abs(rc[0] - target_r) + abs(rc[1] - target_c), rc))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each door on the unique start-to-goal maze route and stamp the matching key shape at that door.

    Input structure:
    - A black-background maze is bounded by grey wall pixels, leaving one-cell-wide corridors.
    - The maze contains one white start pixel and one white 2x2 goal block.
    - Colored line segments placed on corridor cells act as doors, and colored 3-5-cell connected objects act as keys.
    - The only traversable route from the start to the goal is unique when doors are treated as passable.

    Rule:
    - Follow the unique corridor path from the white start pixel to the white goal block.
    - Read the keys encountered on that path in order, and pair them with the doors encountered later on the path in the same order; matching colors are guaranteed.
    - For each paired door, clear the door back to background and then redraw the key's shape centered on the door's center pixel using the grey wall color.
    - Keep the start marker, goal block, original key locations, and any unused keys unchanged.

    Key insight:
    - The colored objects near the start are not only color tokens for opening doors; their geometry is the information that gets transferred to each later door location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start, goal_cells = _white_roles(g)
    path = _path_to_goal(g, start, goal_cells)
    path_index = {cell: i for i, cell in enumerate(path)}

    colored = _components_for_colors(g, set(COLOR_POOL))
    keys: list[dict] = []
    doors: list[dict] = []
    for comp in colored:
        idx = _component_path_index(comp["coords"], path_index)
        if _is_line_component(comp["coords"]):
            if idx is not None:
                doors.append(
                    {
                        "color": int(comp["color"]),
                        "coords": list(comp["coords"]),
                        "index": idx,
                        "center": _line_center(comp["coords"]),
                    }
                )
        elif idx is not None:
            keys.append(
                {
                    "color": int(comp["color"]),
                    "coords": list(comp["coords"]),
                    "index": idx,
                    "anchor": _key_anchor(comp["coords"]),
                }
            )

    keys.sort(key=lambda item: item["index"])
    doors.sort(key=lambda item: item["index"])
    assert doors, "expected at least one door on the route"
    assert len(keys) >= len(doors), "expected at least as many encountered keys as doors"

    pairs: list[tuple[dict, dict]] = []
    for i, door in enumerate(doors):
        key = keys[i]
        assert key["index"] < door["index"], "keys must appear before their matching doors"
        assert key["color"] == door["color"], "door/key colors must match in path order"
        pairs.append((key, door))

    out = g.copy()
    for key, door in pairs:
        for r, c in door["coords"]:
            out[r, c] = BG

        ar, ac = key["anchor"]
        dr, dc = door["center"]
        for r, c in key["coords"]:
            rr = dr + (r - ar)
            cc = dc + (c - ac)
            if not _in_bounds(out.shape, rr, cc):
                continue
            if int(g[rr, cc]) == WHITE:
                continue
            out[rr, cc] = WALL

    return out


def _lattice_neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _make_tree(cell_h: int, cell_w: int, rng: random.Random) -> dict[tuple[int, int], set[tuple[int, int]]]:
    adj = {(r, c): set() for r in range(cell_h) for c in range(cell_w)}
    start = (rng.randrange(cell_h), rng.randrange(cell_w))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack[-1]
        options = [nb for nb in _lattice_neighbors(cell, cell_h, cell_w) if nb not in seen]
        if not options:
            stack.pop()
            continue
        nb = rng.choice(options)
        adj[cell].add(nb)
        adj[nb].add(cell)
        seen.add(nb)
        stack.append(nb)
    return adj


def _tree_parent_map(
    adj: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]
) -> dict[tuple[int, int], tuple[int, int] | None]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nb in adj[cell]:
            if nb in parent:
                continue
            parent[nb] = cell
            q.append(nb)
    return parent


def _tree_distances(
    adj: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]
) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nb in adj[cell]:
            if nb in dist:
                continue
            dist[nb] = dist[cell] + 1
            q.append(nb)
    return dist


def _tree_path(
    adj: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    parent = _tree_parent_map(adj, start)
    assert end in parent
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _grid_coord(cell: tuple[int, int]) -> tuple[int, int]:
    return 2 * cell[0] + 1, 2 * cell[1] + 1


def _render_tree_grid(
    adj: dict[tuple[int, int], set[tuple[int, int]]], cell_h: int, cell_w: int
) -> np.ndarray:
    grid = np.full((2 * cell_h + 1, 2 * cell_w + 1), WALL, dtype=int)
    for cell, nbs in adj.items():
        r, c = _grid_coord(cell)
        grid[r, c] = BG
        for nb in nbs:
            nr, nc = _grid_coord(nb)
            grid[(r + nr) // 2, (c + nc) // 2] = BG
    return grid


def _actual_path(lattice_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i, cell in enumerate(lattice_path):
        cur = _grid_coord(cell)
        if i == 0:
            out.append(cur)
            continue
        prev = _grid_coord(lattice_path[i - 1])
        out.append(((prev[0] + cur[0]) // 2, (prev[1] + cur[1]) // 2))
        out.append(cur)
    return out


def _passable_degree(grid: np.ndarray, coord: tuple[int, int]) -> int:
    r, c = coord
    return sum(int(grid[nr, nc]) != WALL for nr, nc in _neighbors4(grid.shape, r, c))


def _goal_square_candidates(
    grid: np.ndarray, goal_entry: tuple[int, int], parent: tuple[int, int]
) -> list[set[tuple[int, int]]]:
    r, c = goal_entry
    candidates: list[set[tuple[int, int]]] = []
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            if not _in_bounds(grid.shape, r0, c0) or not _in_bounds(grid.shape, r0 + 1, c0 + 1):
                continue
            cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
            if goal_entry not in cells:
                continue
            fresh = cells - {goal_entry}
            if any(int(grid[rr, cc]) != WALL for rr, cc in fresh):
                continue
            outside_neighbors: set[tuple[int, int]] = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(grid.shape, rr, cc):
                    if (nr, nc) in cells:
                        continue
                    if int(grid[nr, nc]) != WALL:
                        outside_neighbors.add((nr, nc))
            if outside_neighbors == {parent}:
                candidates.append(cells)
    return candidates


def _pick_spaced_indices(
    candidates: list[int], count: int, min_gap: int, rng: random.Random
) -> list[int] | None:
    available = list(candidates)
    chosen: list[int] = []
    while available and len(chosen) < count:
        pick = rng.choice(available)
        chosen.append(pick)
        available = [idx for idx in available if abs(idx - pick) >= min_gap]
    if len(chosen) != count:
        return None
    chosen.sort()
    return chosen


def _safe_leaf_targets(grid: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    targets: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if not (1 <= rr < grid.shape[0] - 1 and 1 <= cc < grid.shape[1] - 1):
            continue
        if int(grid[rr, cc]) != WALL:
            continue
        passable_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(grid.shape, rr, cc)
            if int(grid[nr, nc]) != WALL
        ]
        if passable_neighbors == [center]:
            targets.append((rr, cc))
    return targets


def _place_key_component(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    index: int,
    color: int,
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    center = path[index]
    prev = path[index - 1]
    nxt = path[index + 1]
    core = {prev, center, nxt}
    if any(coord in reserved for coord in core):
        return None

    leaves = [coord for coord in _safe_leaf_targets(grid, center) if coord not in reserved]
    if not leaves:
        return None
    rng.shuffle(leaves)
    leaf_count = 2 if len(leaves) >= 2 and rng.random() < 0.5 else 1
    coords = list(core) + leaves[:leaf_count]

    for r, c in coords:
        grid[r, c] = color
    reserved.update(coords)
    return {"color": color, "coords": coords, "anchor": center}


def _door_expected_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return _line_center(coords)


def _expected_output(grid: np.ndarray, used_keys: list[dict], doors: list[dict]) -> np.ndarray:
    out = grid.copy()
    for key, door in zip(used_keys, doors):
        for r, c in door["coords"]:
            out[r, c] = BG
        ar, ac = key["anchor"]
        dr, dc = _door_expected_center(door["coords"])
        for r, c in key["coords"]:
            rr = dr + (r - ar)
            cc = dc + (c - ac)
            if not _in_bounds(out.shape, rr, cc):
                continue
            if int(grid[rr, cc]) == WHITE:
                continue
            out[rr, cc] = WALL
    return out


def _path_cell_coord(cell: tuple[int, int]) -> tuple[int, int]:
    return 2 * cell[0] + 2, 2 * cell[1] + 2


def _random_lattice_path(cell_h: int, cell_w: int, target_len: int, rng: random.Random) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(1, cell_h - 1) for c in range(1, cell_w - 1)]
    if not starts:
        starts = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    rng.shuffle(starts)

    for start in starts[: min(len(starts), 12)]:
        path = [start]
        used = {start}

        def dfs() -> bool:
            if len(path) == target_len:
                return True
            cur = path[-1]
            options = [nb for nb in _lattice_neighbors(cur, cell_h, cell_w) if nb not in used]
            rng.shuffle(options)
            options.sort(
                key=lambda nb: sum(nnb not in used for nnb in _lattice_neighbors(nb, cell_h, cell_w)),
                reverse=True,
            )
            for nb in options:
                used.add(nb)
                path.append(nb)
                if dfs():
                    return True
                path.pop()
                used.remove(nb)
            return False

        if dfs():
            return list(path)
    return None


def _render_lattice_path(
    lattice_path: list[tuple[int, int]], cell_h: int, cell_w: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    grid = np.full((2 * cell_h + 3, 2 * cell_w + 3), WALL, dtype=int)
    actual: list[tuple[int, int]] = []
    for i, cell in enumerate(lattice_path):
        cur = _path_cell_coord(cell)
        if i == 0:
            grid[cur] = BG
            actual.append(cur)
            continue
        prev = _path_cell_coord(lattice_path[i - 1])
        mid = ((prev[0] + cur[0]) // 2, (prev[1] + cur[1]) // 2)
        grid[mid] = BG
        grid[cur] = BG
        actual.extend([mid, cur])
    return grid, actual


def _oriented_path_with_goal(
    grid: np.ndarray, actual_path: list[tuple[int, int]], rng: random.Random
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    options: list[tuple[list[tuple[int, int]], list[set[tuple[int, int]]]]] = []
    for oriented in (actual_path, list(reversed(actual_path))):
        goal_options = _goal_square_candidates(grid, oriented[-1], oriented[-2])
        if goal_options:
            options.append((oriented, goal_options))
    if not options:
        return None
    path, goal_options = rng.choice(options)
    return path, rng.choice(goal_options)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(7, 10)
    cell_w = rng.randint(7, 10)
    target_len = rng.randint(10, min(15, cell_h * cell_w))
    lattice_path = _random_lattice_path(cell_h, cell_w, target_len, rng)
    if lattice_path is None:
        return None

    grid, actual_path = _render_lattice_path(lattice_path, cell_h, cell_w)
    oriented = _oriented_path_with_goal(grid, actual_path, rng)
    if oriented is None:
        return None
    path, goal_cells = oriented
    start_rc = path[0]
    for r, c in goal_cells:
        grid[r, c] = WHITE
    grid[start_rc] = WHITE

    degrees = [_passable_degree(grid, coord) for coord in path]

    key_candidates = [i for i in range(3, max(4, len(path) // 2 - 1)) if degrees[i] == 2]
    door_candidates = [i for i in range(max(len(path) // 2 + 1, 7), len(path) - 3) if degrees[i] == 2]
    max_pairs = min(4, len(key_candidates) // 3, len(door_candidates) // 2)
    if max_pairs < 1:
        return None
    pair_count = rng.randint(1, max_pairs)

    key_indices = _pick_spaced_indices(key_candidates, pair_count, 4, rng)
    door_indices = _pick_spaced_indices(door_candidates, pair_count, 3, rng)
    if key_indices is None or door_indices is None:
        return None

    add_unused = rng.random() < 0.5
    color_count = pair_count + (1 if add_unused else 0)
    colors = rng.sample(COLOR_POOL, color_count)

    reserved: set[tuple[int, int]] = set(goal_cells)
    reserved.add(start_rc)
    keys: list[dict] = []
    for index, color in zip(key_indices, colors[:pair_count]):
        key = _place_key_component(grid, path, index, color, reserved, rng)
        if key is None:
            return None
        keys.append(key)

    doors: list[dict] = []
    for index, color in zip(door_indices, colors[:pair_count]):
        coord = path[index]
        if coord in reserved:
            return None
        grid[coord] = color
        reserved.add(coord)
        doors.append({"color": color, "coords": [coord], "center": coord})

    if add_unused:
        later_candidates = [
            i
            for i in range(door_indices[-1] + 3, len(path) - 2)
            if degrees[i] == 2 and path[i] not in reserved
        ]
        if later_candidates:
            rng.shuffle(later_candidates)
            for index in later_candidates:
                extra = _place_key_component(grid, path, index, colors[-1], reserved, rng)
                if extra is not None:
                    break

    expected = _expected_output(grid, keys, doors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-key-door input")
