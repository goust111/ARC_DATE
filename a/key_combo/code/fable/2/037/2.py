from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg, in_bounds, neighbors4


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RIGHT_OF = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
PALETTE = list(range(10))


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert abs(dr) + abs(dc) == 1
    return dr, dc


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _left_neighbor(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    lr, lc = _left_of(_direction(a, b))
    return int(a[0] + lr), int(a[1] + lc)


def _route_directions(route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_direction(a, b) for a, b in zip(route, route[1:])]


def _passable_component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                stack.append((int(nr), int(nc)))
    return seen


def _trace_simple_route(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    component = _passable_component(mask, start)
    assert end in component, "exit must lie on the entry corridor"
    assert len(component) == int(np.count_nonzero(mask)), "all passable cells must belong to the same corridor"

    for cell in component:
        deg = sum(
            1
            for nr, nc in neighbors4(*cell)
            if in_bounds(mask.shape, nr, nc) and (nr, nc) in component
        )
        if cell in {start, end}:
            assert deg == 1, "entry and exit must be the two corridor endpoints"
        else:
            assert deg == 2, "the corridor must be a single one-cell-wide route"

    route = [start]
    prev = None
    cur = start
    while cur != end:
        nxt = [
            (int(nr), int(nc))
            for nr, nc in neighbors4(*cur)
            if in_bounds(mask.shape, nr, nc) and (nr, nc) in component and (nr, nc) != prev
        ]
        assert len(nxt) == 1, "the corridor route must be unique"
        prev, cur = cur, nxt[0]
        route.append(cur)
    return route


def _special_profiles(
    grid: np.ndarray,
    corridor_bg: int,
    special_set: set[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, list[tuple[int, int]]]]:
    profiles: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = {}
    for cell in special_set:
        bg_neighbors = []
        special_neighbors = []
        for nr, nc in neighbors4(*cell):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) == int(corridor_bg):
                bg_neighbors.append((int(nr), int(nc)))
            if (nr, nc) in special_set:
                special_neighbors.append((int(nr), int(nc)))
        profiles[cell] = {
            "bg_neighbors": bg_neighbors,
            "special_neighbors": special_neighbors,
        }
    return profiles


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = int(get_bg(g))
    counts = Counter(map(int, g.flat))
    non_wall = colors_present(g, exclude={wall})
    assert len(non_wall) >= 4, "expected corridor, entry, exit, and decoration colors"
    corridor_bg = int(max(non_wall, key=lambda color: counts[int(color)]))
    assert counts[corridor_bg] >= 3, "corridor background must be visibly repeated"

    special_colors = [int(color) for color in non_wall if int(color) != corridor_bg]
    special_components = find_components(
        g,
        bg=wall,
        connectivity=4,
        monochromatic=True,
        colors=special_colors,
        return_format="dict",
    )
    assert special_components, "expected entry/exit markers and side pixels"
    assert all(int(component["area"]) == 1 for component in special_components), "special items must be singleton pixels"

    coords_to_color = {
        tuple(map(int, component["coords"][0])): int(component["color"])
        for component in special_components
    }
    assert len(coords_to_color) == len(special_components)
    special_set = set(coords_to_color)
    profiles = _special_profiles(g, corridor_bg, special_set)

    entry_candidates = [
        cell
        for cell, profile in profiles.items()
        if len(profile["bg_neighbors"]) == 1 and len(profile["special_neighbors"]) == 1
    ]
    assert len(entry_candidates) == 1, "expected a unique entry marker adjacent to the demo side pixel"
    entry = entry_candidates[0]
    decor = profiles[entry]["special_neighbors"][0]
    assert len(profiles[decor]["bg_neighbors"]) == 0
    assert set(profiles[decor]["special_neighbors"]) == {entry}

    exit_candidates = [
        cell
        for cell, profile in profiles.items()
        if cell not in {entry, decor}
        and len(profile["bg_neighbors"]) == 1
        and len(profile["special_neighbors"]) == 0
    ]
    assert len(exit_candidates) == 1, "expected a unique exit marker"
    exit_cell = exit_candidates[0]

    distractors = [cell for cell in special_set if cell not in {entry, decor, exit_cell}]
    assert all(
        len(profiles[cell]["bg_neighbors"]) == 0 and len(profiles[cell]["special_neighbors"]) == 0
        for cell in distractors
    ), "distractors must be isolated from the corridor semantics"

    passable = g == corridor_bg
    passable[entry] = True
    passable[exit_cell] = True
    route = _trace_simple_route(passable, entry, exit_cell)
    assert route[1] == profiles[entry]["bg_neighbors"][0]
    assert route[-2] == profiles[exit_cell]["bg_neighbors"][0]
    assert _left_neighbor(route[0], route[1]) == decor

    entry_color = coords_to_color[entry]
    exit_color = coords_to_color[exit_cell]
    decor_color = coords_to_color[decor]
    assert len({entry_color, exit_color, decor_color}) == 3

    return {
        "grid": g,
        "wall": wall,
        "bg": corridor_bg,
        "entry": entry,
        "entry_color": entry_color,
        "exit": exit_cell,
        "exit_color": exit_color,
        "decor": decor,
        "decor_color": decor_color,
        "distractors": distractors,
        "passable": passable,
        "route": route,
    }


def _render_output(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(spec["bg"])
    entry_color = int(spec["entry_color"])
    decor_color = int(spec["decor_color"])
    passable = spec["passable"]
    route = spec["route"]

    for r, c in route[1:-1]:
        if int(g[r, c]) == bg:
            out[r, c] = entry_color

    for a, b in zip(route, route[1:]):
        lr, lc = _left_neighbor(a, b)
        assert in_bounds(out.shape, lr, lc)
        assert not bool(passable[lr, lc]), "left-side stamp targets must stay on wall cells"
        out[lr, lc] = decor_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the unique corridor from the entry marker to the exit and stamp the wall cell on the traveler's left at every step.

    Input structure:
    - One dominant color forms the maze walls, while one repeated secondary color forms a single-cell-wide open corridor.
    - A unique entry marker and a unique exit marker sit on the two corridor endpoints.
    - A demo decoration pixel touches the entry on the wall cell immediately to the left of the first travel step, and any other colored pixels are isolated distractors in the walls.

    Rule:
    - Recover the unique 4-connected route from the entry marker to the exit marker through the corridor cells.
    - Recolor every corridor background cell on that route with the entry color.
    - For each directed step along the route, recolor the wall cell immediately to the traveler's left with the demo decoration color; after a turn, "left" rotates with the new direction.
    - Leave the exit marker, all unstamped wall cells, and all unrelated distractor pixels unchanged.

    Key insight:
    - The demo pixel only identifies the relative side at the entry; the real task is to keep updating that side in the moving frame of reference while tracing the unique corridor.
    """
    spec = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), spec)


# --------------- generator ---------------

GRID_ROW_RANGE = (11, 24)
GRID_COL_RANGE = (11, 24)
PATH_MIN_LEN = 7
PATH_MAX_LEN = 18
DISTRACTOR_RANGE = (1, 3)
ROUTE_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _valid_extension(
    path_set: set[tuple[int, int]],
    current: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    nr = int(current[0] + direction[0])
    nc = int(current[1] + direction[1])
    if not (1 <= nr < shape[0] - 1 and 1 <= nc < shape[1] - 1):
        return None
    nxt = (nr, nc)
    if nxt in path_set:
        return None
    touching = {
        (int(rr), int(cc))
        for rr, cc in neighbors4(nr, nc)
        if (int(rr), int(cc)) in path_set
    }
    if touching != {current}:
        return None
    return nxt


def _build_route(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    max_len = min(PATH_MAX_LEN, max(PATH_MIN_LEN, ((shape[0] - 2) * (shape[1] - 2)) // 4))
    for _ in range(ROUTE_ATTEMPTS):
        target_len = random.randint(PATH_MIN_LEN, max_len)
        start = (
            random.randint(1, shape[0] - 2),
            random.randint(1, shape[1] - 2),
        )
        direction = random.choice(DIRS)
        route = [start]
        path_set = {start}
        right_turns = 0

        while len(route) < target_len:
            moves = []
            for next_dir, turned in ((direction, False), (RIGHT_OF[direction], True)):
                nxt = _valid_extension(path_set, route[-1], next_dir, shape)
                if nxt is not None:
                    moves.append((nxt, next_dir, turned))
            if not moves:
                break

            if right_turns == 0:
                weights = [1 if not turned else 4 for _, _, turned in moves]
            else:
                weights = [4 if not turned else 2 for _, _, turned in moves]
            nxt, direction, turned = random.choices(moves, weights=weights, k=1)[0]
            route.append(nxt)
            path_set.add(nxt)
            right_turns += int(turned)

            if len(route) >= PATH_MIN_LEN and right_turns >= 1 and random.random() < 0.18:
                break

        directions = _route_directions(route)
        if len(route) < PATH_MIN_LEN or right_turns < 1 or len(set(directions)) < 2:
            continue

        stamp_cells = [_left_neighbor(a, b) for a, b in zip(route, route[1:])]
        if any(not in_bounds(shape, r, c) for r, c in stamp_cells):
            continue
        if any(cell in path_set for cell in stamp_cells):
            continue
        if len(set(stamp_cells)) != len(stamp_cells):
            continue
        return route
    return None


def _expected_from_route(
    grid: np.ndarray,
    route: list[tuple[int, int]],
    corridor_bg: int,
    entry_color: int,
    decor_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in route[1:-1]:
        if int(out[r, c]) == corridor_bg:
            out[r, c] = entry_color
    for a, b in zip(route, route[1:]):
        lr, lc = _left_neighbor(a, b)
        out[lr, lc] = decor_color
    return out


def _fixed_absolute_left_output(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    route = spec["route"]
    bg = int(spec["bg"])
    entry_color = int(spec["entry_color"])
    decor_color = int(spec["decor_color"])
    demo_delta = (
        int(spec["decor"][0] - spec["entry"][0]),
        int(spec["decor"][1] - spec["entry"][1]),
    )

    for r, c in route[1:-1]:
        if int(out[r, c]) == bg:
            out[r, c] = entry_color
    for r, c in route[:-1]:
        rr = int(r + demo_delta[0])
        cc = int(c + demo_delta[1])
        if in_bounds(out.shape, rr, cc):
            out[rr, cc] = decor_color
    return out


def _validate_generated(
    grid: np.ndarray,
    route: list[tuple[int, int]],
    corridor_bg: int,
    entry_color: int,
    exit_color: int,
    decor_color: int,
    distractors: list[tuple[int, int]],
) -> bool:
    try:
        spec = _analyze_grid(grid)
        assert spec["route"] == route
        assert int(spec["bg"]) == int(corridor_bg)
        assert int(spec["entry_color"]) == int(entry_color)
        assert int(spec["exit_color"]) == int(exit_color)
        assert int(spec["decor_color"]) == int(decor_color)
        assert len(spec["distractors"]) == len(distractors)

        expected = _expected_from_route(grid, route, corridor_bg, entry_color, decor_color)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        directions = _route_directions(route)
        assert len(set(directions)) >= 2
        assert any(directions[i] != directions[i - 1] for i in range(1, len(directions)))
        assert all(
            directions[i] == directions[i - 1] or directions[i] == RIGHT_OF[directions[i - 1]]
            for i in range(1, len(directions))
        )

        stamp_cells = {_left_neighbor(a, b) for a, b in zip(route, route[1:])}
        assert stamp_cells.isdisjoint(distractors)
        assert not np.array_equal(actual, _fixed_absolute_left_output(grid, spec))
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROW_RANGE)
    w = random.randint(*GRID_COL_RANGE)
    route = _build_route((h, w))
    if route is None:
        return None

    colors = random.sample(PALETTE, len(PALETTE))
    wall, corridor_bg, entry_color, exit_color, decor_color = colors[:5]
    remaining = colors[5:]

    grid = np.full((h, w), wall, dtype=int)
    for r, c in route[1:-1]:
        grid[r, c] = corridor_bg
    grid[route[0]] = entry_color
    grid[route[-1]] = exit_color
    decor = _left_neighbor(route[0], route[1])
    grid[decor] = decor_color

    stamp_cells = {_left_neighbor(a, b) for a, b in zip(route, route[1:])}
    forbidden = set(route) | stamp_cells
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in forbidden
        and int(grid[r, c]) == wall
        and all(int(grid[nr, nc]) == wall for nr, nc in neighbors4(r, c))
    ]
    random.shuffle(candidates)
    distractor_count = min(random.randint(*DISTRACTOR_RANGE), len(candidates), len(remaining))
    if distractor_count < 1:
        return None

    distractors = candidates[:distractor_count]
    for color, cell in zip(remaining[:distractor_count], distractors):
        grid[cell] = color

    if not _validate_generated(
        grid,
        route,
        corridor_bg,
        entry_color,
        exit_color,
        decor_color,
        distractors,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-side-stamp maze")
