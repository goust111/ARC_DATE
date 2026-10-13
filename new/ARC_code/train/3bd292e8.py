from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from arc_utils import Color, get_bg, label_mask, neighbors4
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor wall-separated background regions alternately, starting with grey in the lower-left region.

    Input structure
    - The grid has one dominant background color and one foreground wall color.
    - The wall pixels form one or more 4-connected one-cell-thick segments.
    - Each wall segment separates exactly two 4-connected background regions.
    Rule
    - Keep all wall pixels unchanged.
    - Label the 4-connected background regions and connect two labels when they
      touch opposite sides of the same wall segment.
    - Color the region containing the lower-left corner grey, then flip between
      grey and green across every wall adjacency.
    Key insight
    - The red walls define a bipartite map of regions; the lower-left background
      region fixes which side of the alternating coloring is grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    colors = set(map(int, g.flat))
    assert len(colors) == 2, "expected exactly one background color and one wall color"
    bg = get_bg(g)
    wall_colors = colors - {int(bg)}
    assert len(wall_colors) == 1
    wall = int(next(iter(wall_colors)))

    bg_lab, n_regions = label_mask(g == int(bg), connectivity=4)
    assert n_regions >= 2, "walls must split the background into multiple regions"
    start_region = int(bg_lab[H - 1, 0])
    assert start_region > 0, "lower-left corner must belong to a background region"

    wall_lab, n_walls = label_mask(g == wall, connectivity=4)
    assert n_walls >= 1

    graph: dict[int, set[int]] = defaultdict(set)
    for wall_id in range(1, n_walls + 1):
        adjacent_regions: set[int] = set()
        for r, c in np.argwhere(wall_lab == wall_id):
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W:
                    region_id = int(bg_lab[rr, cc])
                    if region_id > 0:
                        adjacent_regions.add(region_id)

        assert len(adjacent_regions) == 2, "each wall segment must separate exactly two regions"
        a, b = tuple(adjacent_regions)
        graph[a].add(b)
        graph[b].add(a)

    parity = {start_region: 0}
    queue = deque([start_region])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            neighbor_parity = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == neighbor_parity, "region graph must be bipartite"
            else:
                parity[neighbor] = neighbor_parity
                queue.append(neighbor)

    assert set(parity) == set(range(1, n_regions + 1)), "all background regions must be wall-connected"

    out = g.copy()
    for region_id, region_parity in parity.items():
        out[bg_lab == region_id] = Color.GREY if region_parity == 0 else Color.GREEN
    return out

# --------------- generator (att2, o5x) ---------------

# Grid, colors, and retry policy.
GRID_SIZES = (7, 8, 9)
GRID_SIZES_FOR_THREE_WALLS = (8, 9)
WALL_COUNT_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 300
BACKGROUND_COLOR = Color.ORANGE
WALL_COLOR = Color.RED

# Wall geometry constraints.
WALL_LENGTH_RANGE = (4, 13)
WALL_TURN_RANGE = (1, 5)
ENDPOINT_SPACING = 2
MAX_STAIR_PARTS = 3
MIN_FIRST_STAIR_COL = 1

# Left-right corridor layout.
LR_LAYOUT_MAX_WALLS = 2
LR_LAYOUT_PROBABILITY = 0.55
LR_TURN_OPTIONS = (2, 4)
LR_CORRIDOR_MAX_SPAN = 3
LR_CORRIDOR_GAP = 2

# Canonical grids are randomly mapped through one of eight dihedral transforms.
TRANSFORM_OPS = 8


def _contains(bounds: tuple[int, int], value: int) -> bool:
    lo, hi = bounds
    return lo <= int(value) <= hi


def _paint_wall_path(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if grid[r, c] != BACKGROUND_COLOR:
            return False
        grid[r, c] = WALL_COLOR
    return True


def _split_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, int(total)), int(parts) - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [int(total)])]


def _spaced_combos(values: list[int], n: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(values, n)
        if all(combo[i + 1] - combo[i] >= ENDPOINT_SPACING for i in range(len(combo) - 1))
    ]


def _choose_endpoints(H: int, W: int, n_walls: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    rows = _spaced_combos(list(range(1, H - 1)), n_walls)
    cols = _spaced_combos(list(range(1, W - 1)), n_walls)
    valid: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for start_rows in rows:
        for cols_asc in cols:
            end_cols = tuple(reversed(cols_asc))
            lengths = [c + (H - 1 - r) + 1 for r, c in zip(start_rows, end_cols)]
            if all(_contains(WALL_LENGTH_RANGE, length) for length in lengths):
                valid.append((start_rows, end_cols))
    return random.choice(valid) if valid else None


def _stair_path_lb(H: int, W: int, start_r: int, end_c: int, min_first_col: int) -> list[tuple[int, int]] | None:
    down = H - 1 - int(start_r)
    right = int(end_c)
    max_parts = min(MAX_STAIR_PARTS, down, right)
    feasible = [k for k in range(1, max_parts + 1) if right >= int(min_first_col) + k - 1]
    if not feasible:
        return None
    weights = [k * k for k in feasible]
    k = random.choices(feasible, weights=weights, k=1)[0]

    if k == 1:
        horiz = [right]
    else:
        first = random.randint(int(min_first_col), right - (k - 1))
        horiz = [first] + _split_positive(right - first, k - 1)
    vert = _split_positive(down, k)

    r, c = int(start_r), 0
    cells = [(r, c)]
    for h_run, v_run in zip(horiz, vert):
        for _ in range(h_run):
            c += 1
            cells.append((r, c))
        for _ in range(v_run):
            r += 1
            cells.append((r, c))

    if (r, c) != (H - 1, int(end_c)):
        return None
    return cells


def _build_lower_left_grid(H: int, W: int, n_walls: int) -> np.ndarray | None:
    endpoints = _choose_endpoints(H, W, n_walls)
    if endpoints is None:
        return None
    start_rows, end_cols = endpoints

    grid = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    for i, (start_r, end_c) in enumerate(zip(start_rows, end_cols)):
        min_first_col = end_cols[i + 1] + ENDPOINT_SPACING if i + 1 < n_walls else MIN_FIRST_STAIR_COL
        cells = _stair_path_lb(H, W, start_r, end_c, min_first_col)
        if cells is None or not _paint_wall_path(grid, cells):
            return None
    return grid


def _choose_lr_corridors(H: int, n_walls: int) -> tuple[tuple[int, int], ...] | None:
    intervals = []
    for lo in range(1, H - 2):
        for hi in range(lo + 1, min(H - 2, lo + LR_CORRIDOR_MAX_SPAN) + 1):
            intervals.append((lo, hi))
    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(intervals, n_walls):
        if all(combo[i][1] + LR_CORRIDOR_GAP <= combo[i + 1][0] for i in range(n_walls - 1)):
            valid.append(combo)
    return random.choice(valid) if valid else None


def _lr_path(W: int, lo: int, hi: int) -> list[tuple[int, int]] | None:
    short_turn, long_turn = LR_TURN_OPTIONS
    turn_goal = random.choice(LR_TURN_OPTIONS)
    if turn_goal == long_turn and W - 1 < long_turn:
        turn_goal = short_turn

    if turn_goal == short_turn:
        h1 = random.randint(1, W - 2)
        horiz = [h1, W - 1 - h1]
        r0, r1 = random.sample(range(lo, hi + 1), 2)
        rows = [r0, r1]
    else:
        if hi - lo >= 2:
            if random.random() < 0.5:
                rows = [lo, random.randint(lo + 1, hi - 1), hi]
            else:
                rows = [hi, random.randint(lo + 1, hi - 1), lo]
            middle_min = 1
        else:
            rows = [lo, hi, lo] if random.random() < 0.5 else [hi, lo, hi]
            middle_min = 2
        if W - 1 < middle_min + 2:
            return None
        h2 = random.randint(middle_min, W - 3)
        h1 = random.randint(1, W - 1 - h2 - 1)
        horiz = [h1, h2, W - 1 - h1 - h2]

    r, c = int(rows[0]), 0
    cells = [(r, c)]
    for i, h_run in enumerate(horiz):
        for _ in range(h_run):
            c += 1
            cells.append((r, c))
        if i < len(rows) - 1:
            target = int(rows[i + 1])
            step = 1 if target > r else -1
            while r != target:
                r += step
                cells.append((r, c))

    if c != W - 1:
        return None
    if not _contains(WALL_LENGTH_RANGE, len(cells)):
        return None
    return cells


def _build_lr_grid(H: int, W: int, n_walls: int) -> np.ndarray | None:
    corridors = _choose_lr_corridors(H, n_walls)
    if corridors is None:
        return None
    grid = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    for lo, hi in corridors:
        cells = _lr_path(W, lo, hi)
        if cells is None or not _paint_wall_path(grid, cells):
            return None
    return grid


def _transform(grid: np.ndarray) -> np.ndarray:
    op = random.randrange(TRANSFORM_OPS)
    if op == 0:
        out = grid
    elif op == 1:
        out = np.rot90(grid, 1)
    elif op == 2:
        out = np.rot90(grid, 2)
    elif op == 3:
        out = np.rot90(grid, 3)
    elif op == 4:
        out = np.fliplr(grid)
    elif op == 5:
        out = np.flipud(grid)
    elif op == 6:
        out = grid.T
    else:
        out = np.fliplr(grid).T
    return np.ascontiguousarray(out)


def _ordered_path(coords: np.ndarray) -> list[tuple[int, int]] | None:
    cells = {tuple(map(int, rc)) for rc in coords}
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        here = [(rr, cc) for rr, cc in neighbors4(r, c) if (rr, cc) in cells]
        if len(here) > 2:
            return None
        nbrs[(r, c)] = here

    endpoints = [cell for cell, here in nbrs.items() if len(here) == 1]
    if len(cells) == 1 or len(endpoints) != 2:
        return None

    path = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        nxts = [cell for cell in nbrs[cur] if cell != prev]
        if not nxts:
            return None
        prev, cur = cur, nxts[0]
        path.append(cur)
        if len(path) > len(cells):
            return None
    return path if len(path) == len(cells) else None


def _turn_count(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((int(np.sign(r1 - r0)), int(np.sign(c1 - c0))))
    return sum(a != b for a, b in zip(dirs, dirs[1:]))


def _validate_wall_paths(wall_lab: np.ndarray, n_walls: int, H: int, W: int) -> bool:
    for wall_id in range(1, n_walls + 1):
        coords = np.argwhere(wall_lab == wall_id)
        if not _contains(WALL_LENGTH_RANGE, len(coords)):
            return False
        path = _ordered_path(coords)
        if path is None:
            return False
        endpoints = [path[0], path[-1]]
        if not all(r in (0, H - 1) or c in (0, W - 1) for r, c in endpoints):
            return False
        turns = _turn_count(path)
        if not _contains(WALL_TURN_RANGE, turns):
            return False
    return True


def _build_region_graph(wall_lab, bg_lab, n_walls: int, n_regions: int, H: int, W: int):
    """Connect background-region labels separated by each wall component."""
    graph: dict[int, set[int]] = defaultdict(set)
    edges: set[tuple[int, int]] = set()
    for wall_id in range(1, n_walls + 1):
        adjacent: set[int] = set()
        for r, c in np.argwhere(wall_lab == wall_id):
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W:
                    region_id = int(bg_lab[rr, cc])
                    if region_id > 0:
                        adjacent.add(region_id)
        if len(adjacent) != 2:
            return None
        a, b = sorted(adjacent)
        graph[a].add(b)
        graph[b].add(a)
        edges.add((a, b))

    if len(edges) != n_walls:
        return None
    if set(graph) != set(range(1, n_regions + 1)):
        return None
    if any(len(graph[k]) > 2 for k in graph):
        return None
    return graph


def _region_parity(graph: dict[int, set[int]], start_region: int, n_regions: int) -> dict[int, int] | None:
    parity = {start_region: 0}
    queue = deque([start_region])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            want = 1 - parity[region]
            if neighbor in parity:
                if parity[neighbor] != want:
                    return None
            else:
                parity[neighbor] = want
                queue.append(neighbor)
    if set(parity) != set(range(1, n_regions + 1)):
        return None
    return parity


def _check_and_color(grid: np.ndarray, expected_walls: int) -> np.ndarray | None:
    """Validate task constraints and return the solver-expected output grid."""
    H, W = grid.shape
    if int(grid[H - 1, 0]) != BACKGROUND_COLOR:
        return None

    wall_mask = grid == WALL_COLOR
    wall_lab, n_walls = label_mask(wall_mask, connectivity=4)
    if n_walls != int(expected_walls):
        return None
    _, n_walls8 = label_mask(wall_mask, connectivity=8)
    if n_walls8 != n_walls:
        return None

    if not _validate_wall_paths(wall_lab, n_walls, H, W):
        return None

    bg_lab, n_regions = label_mask(grid == BACKGROUND_COLOR, connectivity=4)
    if n_regions != n_walls + 1:
        return None
    start_region = int(bg_lab[H - 1, 0])
    if start_region <= 0:
        return None

    graph = _build_region_graph(wall_lab, bg_lab, n_walls, n_regions, H, W)
    if graph is None:
        return None
    parity = _region_parity(graph, start_region, n_regions)
    if parity is None:
        return None

    out = grid.copy()
    for region_id, p in parity.items():
        out[bg_lab == region_id] = Color.GREY if p == 0 else Color.GREEN
    return out


def _build_canonical_grid(H: int, W: int, n_walls: int) -> np.ndarray | None:
    if n_walls <= LR_LAYOUT_MAX_WALLS and random.random() < LR_LAYOUT_PROBABILITY:
        return _build_lr_grid(H, W, n_walls)
    return _build_lower_left_grid(H, W, n_walls)


def _gen_attempt() -> np.ndarray | None:
    n_walls = random.randint(*WALL_COUNT_RANGE)
    sizes = GRID_SIZES if n_walls < WALL_COUNT_RANGE[1] else GRID_SIZES_FOR_THREE_WALLS
    H = random.choice(sizes)
    W = H
    grid = _build_canonical_grid(H, W, n_walls)
    if grid is None:
        return None
    grid = _transform(grid)
    expected = _check_and_color(grid, n_walls)
    if expected is None:
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 3bd292e8 input")
