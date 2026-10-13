import random

import numpy as np


BG = 0
RED = 2
PATH_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERP_DIRS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

GEN_RETRIES = 400
OBJECT_RETRIES = 160
LOCAL_PATH_RETRIES = 120
GRID_MAX_SIDE = 28
GRID_MIN_SIDE_BY_COUNT = {1: 14, 2: 18, 3: 21, 4: 24}
SEG_LEN_BY_TURNS = {
    0: (2, 6),
    1: (1, 5),
    2: (1, 4),
    3: (1, 4),
    4: (1, 3),
}


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _in_bounds(shape: tuple[int, int], rc: tuple[int, int]) -> bool:
    return 0 <= rc[0] < shape[0] and 0 <= rc[1] < shape[1]


def _neighbors4(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(cells: set[tuple[int, int]], delta: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _grow_tree_cells(endpoint: tuple[int, int], direction: tuple[int, int], turns: int) -> set[tuple[int, int]]:
    trunk = _add(endpoint, direction)
    cells = {trunk}
    if turns == 0:
        return cells

    tips = {trunk}
    for layer in range(1, turns + 1):
        next_tips = set()
        for tip in tips:
            base = tip if layer == 1 else _add(tip, direction)
            if layer > 1:
                cells.add(base)
            for side in PERP_DIRS[direction]:
                child = _add(base, side)
                cells.add(child)
                next_tips.add(child)
        tips = next_tips
    return cells


def _trace_path(grid: np.ndarray, start: tuple[int, int]) -> dict:
    shape = tuple(map(int, grid.shape))
    start_neighbors = [
        nbr for nbr in _neighbors4(start) if _in_bounds(shape, nbr) and int(grid[nbr]) != BG
    ]
    assert len(start_neighbors) == 1, "each red start must have exactly one path neighbor"
    current = start_neighbors[0]
    path_color = int(grid[current])
    assert path_color not in (BG, RED), "path body color must differ from bg and red"

    previous = start
    direction = _sub(current, previous)
    turns = 0

    while True:
        next_neighbors = [
            nbr
            for nbr in _neighbors4(current)
            if _in_bounds(shape, nbr) and int(grid[nbr]) != BG and nbr != previous
        ]
        assert len(next_neighbors) <= 1, "path must be a simple polyline"
        if not next_neighbors:
            return {
                "color": path_color,
                "turns": turns,
                "endpoint": current,
                "final_dir": direction,
            }

        nxt = next_neighbors[0]
        assert int(grid[nxt]) == path_color, "path body must keep a constant non-red color"
        new_direction = _sub(nxt, current)
        if new_direction != direction:
            turns += 1
        previous, current, direction = current, nxt, new_direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a direction-aware branching tree from the end of each colored path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    starts = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert starts, "expected at least one red start pixel"

    out = g.copy()
    for start in starts:
        info = _trace_path(g, start)
        for rc in _grow_tree_cells(info["endpoint"], info["final_dir"], info["turns"]):
            if _in_bounds(g.shape, rc):
                out[rc] = info["color"]
    return out


def _tree_conflicts_with_path(
    path_cells: set[tuple[int, int]],
    tree_cells: set[tuple[int, int]],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    trunk = _add(endpoint, direction)
    allowed_touch = {(endpoint, trunk), (trunk, endpoint)}

    for cell in tree_cells:
        if cell in path_cells:
            return True
        for nbr in _neighbors4(cell):
            if nbr in path_cells and (cell, nbr) not in allowed_touch:
                return True
    return False


def _random_local_spec(color: int, min_turns: int = 0) -> dict | None:
    for _ in range(LOCAL_PATH_RETRIES):
        turns = random.randint(min_turns, 4)
        directions = [random.choice(DIRS)]
        for _ in range(turns):
            directions.append(random.choice(PERP_DIRS[directions[-1]]))

        lo, hi = SEG_LEN_BY_TURNS[turns]
        lengths = [random.randint(lo, hi) for _ in directions]

        path_list = [(0, 0)]
        path_set = {(0, 0)}
        valid = True
        for direction, seg_len in zip(directions, lengths):
            for _ in range(seg_len):
                nxt = _add(path_list[-1], direction)
                if nxt in path_set:
                    valid = False
                    break
                if any(nbr in path_set and nbr != path_list[-1] for nbr in _neighbors4(nxt)):
                    valid = False
                    break
                path_list.append(nxt)
                path_set.add(nxt)
            if not valid:
                break
        if not valid:
            continue

        endpoint = path_list[-1]
        final_dir = directions[-1]
        tree_cells = _grow_tree_cells(endpoint, final_dir, turns)
        if _tree_conflicts_with_path(path_set, tree_cells, endpoint, final_dir):
            continue

        return {
            "color": color,
            "turns": turns,
            "path_list": path_list,
            "path_set": path_set,
            "endpoint": endpoint,
            "final_dir": final_dir,
            "tree_set": tree_cells,
            "trunk": _add(endpoint, final_dir),
            "first_branch": {_add(_add(endpoint, final_dir), side) for side in PERP_DIRS[final_dir]},
        }
    return None


def _conflicts_with_occupied(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    *,
    diagonal: bool,
) -> bool:
    if not occupied:
        return False
    neighbor_fn = _neighbors8 if diagonal else _neighbors4
    for cell in cells:
        if cell in occupied:
            return True
        if any(nbr in occupied for nbr in neighbor_fn(cell)):
            return True
    return False


def _place_spec(
    spec: dict,
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    *,
    require_full_tree: bool,
    require_clipped_tree: bool,
) -> dict | None:
    path_set = spec["path_set"]
    tree_set = spec["tree_set"]

    if require_full_tree:
        fit_cells = path_set | tree_set
    else:
        fit_cells = path_set

    min_r, min_c, max_r, max_c = _bbox(fit_cells)
    row_range = range(-min_r, shape[0] - max_r)
    col_range = range(-min_c, shape[1] - max_c)
    placements = [(r, c) for r in row_range for c in col_range]
    random.shuffle(placements)

    for delta in placements:
        placed_path = _translate(path_set, delta)
        if _conflicts_with_occupied(placed_path, occupied_input, diagonal=True):
            continue

        trunk = _add(spec["trunk"], delta)
        if not _in_bounds(shape, trunk):
            continue

        first_branch_visible = any(_in_bounds(shape, _add(cell, delta)) for cell in spec["first_branch"])
        if spec["turns"] > 0 and not first_branch_visible:
            continue

        placed_tree_all = _translate(tree_set, delta)
        placed_tree_visible = {cell for cell in placed_tree_all if _in_bounds(shape, cell)}
        is_full_tree = len(placed_tree_visible) == len(tree_set)
        if require_full_tree and not is_full_tree:
            continue
        if require_clipped_tree and is_full_tree:
            continue

        output_footprint = placed_path | placed_tree_visible
        if _conflicts_with_occupied(output_footprint, occupied_output, diagonal=False):
            continue

        return {
            "color": spec["color"],
            "turns": spec["turns"],
            "path_list": [_add(cell, delta) for cell in spec["path_list"]],
            "path_set": placed_path,
            "tree_visible": placed_tree_visible,
            "clipped": not is_full_tree,
        }
    return None


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        start = obj["path_list"][0]
        grid[start] = RED
        for cell in obj["path_list"][1:]:
            grid[cell] = obj["color"]
    return grid


def _render_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        for cell in obj["tree_visible"]:
            out[cell] = obj["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    min_side = GRID_MIN_SIDE_BY_COUNT[n_objects]
    shape = (
        random.randint(min_side, GRID_MAX_SIDE),
        random.randint(min_side, GRID_MAX_SIDE),
    )

    want_clipped_tree = random.random() < 0.3
    clipped_index = random.randrange(n_objects) if want_clipped_tree else -1

    colors = random.sample(PATH_COLORS, n_objects)
    roles = [
        {
            "color": color,
            "min_turns": 1 if i == clipped_index else 0,
            "require_full_tree": i != clipped_index,
            "require_clipped_tree": i == clipped_index,
        }
        for i, color in enumerate(colors)
    ]
    roles.sort(key=lambda role: role["require_clipped_tree"], reverse=True)

    objects = []
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    branching_present = False

    for role in roles:
        placed = None
        for _ in range(OBJECT_RETRIES):
            spec = _random_local_spec(role["color"], min_turns=role["min_turns"])
            if spec is None:
                continue
            placed = _place_spec(
                spec,
                shape,
                occupied_input,
                occupied_output,
                require_full_tree=role["require_full_tree"],
                require_clipped_tree=role["require_clipped_tree"],
            )
            if placed is not None:
                break
        if placed is None:
            return None
        branching_present |= placed["turns"] > 0
        objects.append(placed)
        occupied_input |= placed["path_set"]
        occupied_output |= placed["path_set"] | placed["tree_visible"]

    if n_objects > 1 and not branching_present:
        return None
    if want_clipped_tree and not any(obj["clipped"] for obj in objects):
        return None

    inp = _render_input(shape, objects)
    expected = _render_output(inp, objects)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
