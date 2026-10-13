import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
WALL = 9
MERGE_FILL = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 300
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BASE_SHAPES = [
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
]


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns)


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).ravel())


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    rotations = [_rotate_cw(mask, k) for k in range(4)]
    best = min(range(4), key=lambda k: _mask_key(rotations[k]))
    return rotations[best]


def _orientation_from_mask(mask: np.ndarray, canonical: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(mask, _rotate_cw(canonical, k))]
    assert len(matches) == 1
    return matches[0]


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _component_adjacency_rows(seed_grid: np.ndarray) -> tuple[int, int]:
    same_edges = 0
    diff_edges = 0
    rows, cols = seed_grid.shape
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                if int(seed_grid[r, c]) == int(seed_grid[r + 1, c]):
                    same_edges += 1
                else:
                    diff_edges += 1
            if c + 1 < cols:
                if int(seed_grid[r, c]) == int(seed_grid[r, c + 1]):
                    same_edges += 1
                else:
                    diff_edges += 1
    return same_edges, diff_edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge neighboring 5x5 regions when their centered key shapes advance by a 90-degree clockwise rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    wall = int(g[0, 0])
    wall_rows = [r for r in range(h) if np.all(g[r, :] == wall)]
    wall_cols = [c for c in range(w) if np.all(g[:, c] == wall)]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2
    assert all(wall_rows[i + 1] - wall_rows[i] == 6 for i in range(len(wall_rows) - 1))
    assert all(wall_cols[i + 1] - wall_cols[i] == 6 for i in range(len(wall_cols) - 1))

    non_wall = [int(x) for x in g.ravel() if int(x) != wall]
    counts = Counter(non_wall)
    bg = counts.most_common(1)[0][0]

    region_rows = len(wall_rows) - 1
    region_cols = len(wall_cols) - 1
    masks: list[list[np.ndarray]] = []
    for rr in range(region_rows):
        row_masks = []
        for cc in range(region_cols):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1]
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1]
            region = g[r0:r1, c0:c1]
            assert region.shape == (5, 5)
            core = region[1:4, 1:4]
            row_masks.append(core != bg)
        masks.append(row_masks)

    canonical = _canonical_mask(masks[0][0])
    orientations = np.zeros((region_rows, region_cols), dtype=int)
    for rr in range(region_rows):
        for cc in range(region_cols):
            orientations[rr, cc] = _orientation_from_mask(masks[rr][cc], canonical)

    parent = list(range(region_rows * region_cols))
    h_edges: dict[tuple[int, int], bool] = {}
    v_edges: dict[tuple[int, int], bool] = {}
    out = g.copy()

    def region_index(rr: int, cc: int) -> int:
        return rr * region_cols + cc

    for rr in range(region_rows):
        for cc in range(region_cols):
            if cc + 1 < region_cols:
                ok = int(orientations[rr, cc + 1]) == (int(orientations[rr, cc]) + 1) % 4
                h_edges[(rr, cc)] = ok
                if ok:
                    _union(parent, region_index(rr, cc), region_index(rr, cc + 1))
                    mid_col = wall_cols[cc + 1]
                    out[wall_rows[rr] + 1 : wall_rows[rr + 1], mid_col] = bg
            if rr + 1 < region_rows:
                ok = int(orientations[rr + 1, cc]) == (int(orientations[rr, cc]) + 1) % 4
                v_edges[(rr, cc)] = ok
                if ok:
                    _union(parent, region_index(rr, cc), region_index(rr + 1, cc))
                    mid_row = wall_rows[rr + 1]
                    out[mid_row, wall_cols[cc] + 1 : wall_cols[cc + 1]] = bg

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr in range(region_rows):
        for cc in range(region_cols):
            groups[_find(parent, region_index(rr, cc))].append((rr, cc))

    for members in groups.values():
        if len(members) < 2:
            continue
        fill_mask = np.zeros_like(out, dtype=bool)
        member_set = set(members)
        for rr, cc in members:
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1]
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1]
            fill_mask[r0:r1, c0:c1] = True
            if cc + 1 < region_cols and (rr, cc + 1) in member_set and h_edges.get((rr, cc), False):
                fill_mask[r0:r1, wall_cols[cc + 1]] = True
            if rr + 1 < region_rows and (rr + 1, cc) in member_set and v_edges.get((rr, cc), False):
                fill_mask[wall_rows[rr + 1], c0:c1] = True
        out[fill_mask & (out == bg)] = MERGE_FILL

    return out


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _grow_blob(
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
    rng.shuffle(starts)
    for start in starts:
        blob = {start}
        frontier = {nbr for nbr in _neighbors(start, rows, cols) if nbr not in blocked}
        while frontier and len(blob) < size:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in blob or cell in blocked:
                continue
            blob.add(cell)
            for nbr in _neighbors(cell, rows, cols):
                if nbr not in blocked and nbr not in blob:
                    frontier.add(nbr)
        if len(blob) == size:
            return blob
    return None


def _build_partition(rows: int, cols: int, mode: str, rng: random.Random) -> list[set[tuple[int, int]]] | None:
    area = rows * cols
    components: list[set[tuple[int, int]]] = []
    blocked: set[tuple[int, int]] = set()

    if mode == "mix":
        blob_size = rng.randint(2, min(4, area - 1))
        blob = _grow_blob(rows, cols, blocked, blob_size, rng)
        if blob is None:
            return None
        components.append(blob)
        blocked |= blob
    elif mode == "big":
        blob_size = rng.randint(3, min(6, area - 1))
        blob = _grow_blob(rows, cols, blocked, blob_size, rng)
        if blob is None:
            return None
        components.append(blob)
        blocked |= blob
    else:
        max_first = min(4, area - 3)
        if max_first < 2:
            return None
        first_size = rng.randint(2, max_first)
        first = _grow_blob(rows, cols, blocked, first_size, rng)
        if first is None:
            return None
        components.append(first)
        blocked |= first

        max_second = min(4, area - len(blocked) - 1)
        if max_second < 2:
            return None
        second_size = rng.randint(2, max_second)
        second = _grow_blob(rows, cols, blocked, second_size, rng)
        if second is None:
            return None
        components.append(second)
        blocked |= second

    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell not in blocked:
                components.append({cell})

    sizes = sorted(len(comp) for comp in components)
    if not any(size > 1 for size in sizes):
        return None
    if not any(size == 1 for size in sizes):
        return None
    if mode == "big" and max(sizes) < 3:
        return None
    if mode == "double" and sum(size > 1 for size in sizes) < 2:
        return None
    return components


def _component_graph(
    rows: int, cols: int, components: list[set[tuple[int, int]]]
) -> tuple[list[set[int]], dict[tuple[int, int], int]]:
    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        for cell in comp:
            cell_to_comp[cell] = idx

    adjacency = [set() for _ in components]
    for r in range(rows):
        for c in range(cols):
            here = cell_to_comp[(r, c)]
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= rows or nc >= cols:
                    continue
                there = cell_to_comp[(nr, nc)]
                if here != there:
                    adjacency[here].add(there)
                    adjacency[there].add(here)
    return adjacency, cell_to_comp


def _color_components(adjacency: list[set[int]], rng: random.Random) -> list[int] | None:
    n = len(adjacency)
    colors = [-1] * n
    palette = [0, 1, 2, 3]

    def choose_node() -> int | None:
        best = None
        best_key = None
        for node in range(n):
            if colors[node] != -1:
                continue
            used = {colors[other] for other in adjacency[node] if colors[other] != -1}
            key = (len(used), len(adjacency[node]))
            if best is None or key > best_key:
                best = node
                best_key = key
        return best

    def backtrack() -> bool:
        node = choose_node()
        if node is None:
            return True
        used = {colors[other] for other in adjacency[node] if colors[other] != -1}
        options = [color for color in rng.sample(palette, len(palette)) if color not in used]
        for color in options:
            colors[node] = color
            if backtrack():
                return True
        colors[node] = -1
        return False

    if backtrack():
        return colors
    return None


def _seed_grid_from_partition(
    rows: int, cols: int, components: list[set[tuple[int, int]]], rng: random.Random
) -> np.ndarray | None:
    adjacency, cell_to_comp = _component_graph(rows, cols, components)
    comp_colors = _color_components(adjacency, rng)
    if comp_colors is None:
        return None
    seed_grid = np.zeros((rows, cols), dtype=int)
    for cell, comp_idx in cell_to_comp.items():
        seed_grid[cell] = comp_colors[comp_idx]
    return seed_grid


def _choose_dims(mode: str, rng: random.Random) -> tuple[int, int]:
    while True:
        rows = rng.randint(2, 5)
        cols = rng.randint(2, 5)
        area = rows * cols
        if mode == "double" and area < 6:
            continue
        if area < 4:
            continue
        return rows, cols


def _generate_seed_grid(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["mix", "big", "double"])
    rows, cols = _choose_dims(mode, rng)
    components = _build_partition(rows, cols, mode, rng)
    if components is None:
        return None
    seed_grid = _seed_grid_from_partition(rows, cols, components, rng)
    if seed_grid is None:
        return None
    sizes = sorted(len(comp) for comp in components)
    same_edges, diff_edges = _component_adjacency_rows(seed_grid)
    if same_edges == 0 or diff_edges == 0:
        return None
    if max(sizes) < 2 or not any(size == 1 for size in sizes):
        return None
    return seed_grid


def _shape_ok(shape: np.ndarray) -> bool:
    keys = {_mask_key(_rotate_cw(shape, k)) for k in range(4)}
    return len(keys) == 4


assert all(_shape_ok(shape) for shape in BASE_SHAPES)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    seed_grid = _generate_seed_grid(rng)
    if seed_grid is None:
        return None

    region_rows, region_cols = seed_grid.shape
    orientations = (seed_grid + np.arange(region_rows)[:, None] + np.arange(region_cols)[None, :]) % 4
    shape = rng.choice(BASE_SHAPES)
    pattern_color = rng.choice(PATTERN_COLORS)

    h = region_rows * 6 + 1
    w = region_cols * 6 + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[::6, :] = WALL
    grid[:, ::6] = WALL

    for rr in range(region_rows):
        for cc in range(region_cols):
            base_r = rr * 6 + 1
            base_c = cc * 6 + 1
            rotated = _rotate_cw(shape, int(orientations[rr, cc]))
            core = grid[base_r + 1 : base_r + 4, base_c + 1 : base_c + 4]
            core[rotated] = pattern_color

    out = main(grid)
    if not np.any(out == MERGE_FILL):
        return None
    if not np.any(out == BG):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
