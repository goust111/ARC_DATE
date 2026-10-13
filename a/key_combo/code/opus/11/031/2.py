import random
from collections import defaultdict

import numpy as np


BG = 0
OBJ_COLORS = (1, 2, 3)
AXIS = 5
MARKER_COLORS = (4, 6, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_left_shapes(grid: np.ndarray, axis_col: int) -> list[dict]:
    shapes: list[dict] = []
    for color in OBJ_COLORS:
        mask = (grid == color)
        if axis_col > 0:
            mask[:, axis_col:] = False
        for coords in _components_of_color(mask):
            rows = coords[:, 0]
            cols = coords[:, 1]
            shapes.append(
                {
                    "color": color,
                    "coords": coords,
                    "top": int(rows.min()),
                    "right": int(cols.max()),
                }
            )
    assert 2 <= len(shapes) <= 4, "expected 2-4 left-side objects"
    return shapes


def _reflected_cells(shape: dict, axis_col: int) -> list[tuple[int, int]]:
    right = int(shape["right"])
    cells = []
    for r, c in shape["coords"]:
        offset = right - int(c)
        cells.append((int(r), axis_col + 1 + offset))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-side object to the right of a grey axis and resolve reflected overlaps by the source object's axis distance.

    Input structure:
    - The background is black.
    - A single grey vertical line is the reflection axis.
    - The axis has 2-4 disconnected monochrome objects on its left in blue, red, or green.
    - The right side contains one isolated marker pixel inside the exclusive reflected area of each object.

    Rule:
    - Keep the grey axis and all original left-side objects unchanged.
    - For each left object, mirror its shape to the right side so the mirrored column nearest the axis lands immediately to the axis's right.
    - Paint every reflected cell with that object's marker color.
    - If multiple reflected objects want the same right-side cell, the object whose source pixels are horizontally closer to the axis wins; if that distance ties, the object with the smaller top row wins.
    - Remove the marker pixels themselves after using them as color instructions.

    Key insight:
    - The right side does not show the reflected objects directly: each lone marker names a reflected copy's color, while overlap cells must be decided from the pre-reflection geometry on the left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    axis_cols = np.unique(np.argwhere(g == AXIS)[:, 1])
    assert len(axis_cols) == 1, "expected one grey axis column"
    axis_col = int(axis_cols[0])

    shapes = _extract_left_shapes(g, axis_col)

    cover: dict[tuple[int, int], list[int]] = defaultdict(list)
    reflected: list[list[tuple[int, int]]] = []
    for idx, shape in enumerate(shapes):
        cells = _reflected_cells(shape, axis_col)
        reflected.append(cells)
        for cell in cells:
            cover[cell].append(idx)

    marker_color_by_shape: dict[int, int] = {}
    for idx, shape in enumerate(shapes):
        unique_cells = [cell for cell in reflected[idx] if len(cover[cell]) == 1]
        markers = [
            int(g[r, c])
            for r, c in unique_cells
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) in MARKER_COLORS
        ]
        assert len(markers) == 1, "each object needs exactly one marker in its exclusive reflected area"
        marker_color_by_shape[idx] = markers[0]

    out = g.copy()
    out[np.isin(out, MARKER_COLORS)] = BG

    for cell, owners in cover.items():
        winner = min(
            owners,
            key=lambda idx: (axis_col - int(shapes[idx]["right"]), int(shapes[idx]["top"])),
        )
        r, c = cell
        out[r, c] = marker_color_by_shape[winner]

    return out


def _random_connected_shape(rng: random.Random, height: int, width: int, size: int) -> set[tuple[int, int]] | None:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    return cells


def _grow_branch(
    rng: random.Random,
    core: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    branch_size: int,
    row_lo: int,
    row_hi: int,
    off_hi: int,
) -> set[tuple[int, int]] | None:
    frontier = []
    for r, c in core:
        for nr, nc in _neighbors4(r, c):
            if row_lo <= nr <= row_hi and 0 <= nc <= off_hi and (nr, nc) not in occupied:
                frontier.append((nr, nc))
    frontier = list(dict.fromkeys(frontier))
    if not frontier:
        return None
    branch = {rng.choice(frontier)}
    while len(branch) < branch_size:
        step_frontier = []
        for r, c in branch | core:
            for nr, nc in _neighbors4(r, c):
                if row_lo <= nr <= row_hi and 0 <= nc <= off_hi:
                    if (nr, nc) not in occupied and (nr, nc) not in branch:
                        step_frontier.append((nr, nc))
        step_frontier = list(dict.fromkeys(step_frontier))
        if not step_frontier:
            return None
        branch.add(rng.choice(step_frontier))
    return branch


def _choose_marker_colors(
    rng: random.Random, overlap_graph: dict[int, set[int]], n: int
) -> list[int] | None:
    colors = list(MARKER_COLORS)
    order = sorted(range(n), key=lambda i: (-len(overlap_graph[i]), i))
    chosen = [None] * n

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        rng.shuffle(colors)
        for color in colors:
            if any(chosen[nb] == color for nb in overlap_graph[idx]):
                continue
            chosen[idx] = color
            if dfs(pos + 1):
                return True
            chosen[idx] = None
        return False

    if not dfs(0):
        return None
    return [int(x) for x in chosen]


def _cells_touch(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _normalize_shape_offsets(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_c = min(c for _r, c in cells)
    return {(r, c - min_c) for r, c in cells}


def _build_instance(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 20)
    n_shapes = rng.randint(2, 4)
    pair_templates = [
        (
            {(0, 0), (0, 1), (1, 1), (1, 0), (2, 1)},
            {(0, 0), (0, 1), (1, 1), (0, 2), (1, 2)},
        ),
        (
            {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
            {(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 1)},
        ),
        (
            {(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)},
            {(0, 1), (1, 0), (1, 1), (2, 1), (0, 2), (1, 2)},
        ),
    ]
    independent_templates = [
        {(0, 0), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 1), (1, 0), (1, 1), (2, 1)},
    ]

    def shift_shape(shape: set[tuple[int, int]], row_shift: int) -> set[tuple[int, int]]:
        return {(r + row_shift, c) for r, c in shape}

    pair_a, pair_b = rng.choice(pair_templates)
    pair_height = 1 + max(max(r for r, _c in pair_a), max(r for r, _c in pair_b))
    pair_row0 = rng.randint(1, h - pair_height - 1)
    target_shapes: list[set[tuple[int, int]]] = [
        shift_shape(_normalize_shape_offsets(pair_a), pair_row0),
        shift_shape(_normalize_shape_offsets(pair_b), pair_row0),
    ]

    occupied_rows = set(range(pair_row0 - 1, pair_row0 + pair_height + 1))
    for _ in range(n_shapes - 2):
        placed = False
        for _ in range(60):
            template = _normalize_shape_offsets(rng.choice(independent_templates))
            height = 1 + max(r for r, _c in template)
            candidates = [
                row0
                for row0 in range(1, h - height)
                if not occupied_rows.intersection(range(row0 - 1, row0 + height + 1))
            ]
            if not candidates:
                continue
            row0 = rng.choice(candidates)
            shape = shift_shape(template, row0)
            target_shapes.append(shape)
            occupied_rows.update(range(row0 - 1, row0 + height + 1))
            placed = True
            break
        if not placed:
            return None

    overlap_graph = {i: set() for i in range(n_shapes)}
    cover_count: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, shape in enumerate(target_shapes):
        for cell in shape:
            cover_count[cell].append(idx)
    for owners in cover_count.values():
        if len(owners) > 1:
            for i in owners:
                for j in owners:
                    if i != j:
                        overlap_graph[i].add(j)
    if not any(overlap_graph[i] for i in range(n_shapes)):
        return None
    overlap_cells = {cell for cell, owners in cover_count.items() if len(owners) > 1}
    if len(overlap_cells) < 3:
        return None

    marker_colors = _choose_marker_colors(rng, overlap_graph, n_shapes)
    if marker_colors is None:
        return None

    distances_pool = [1, 6, 11, 16]
    if n_shapes > len(distances_pool):
        return None
    distances = rng.sample(distances_pool, n_shapes)
    if distances[0] > distances[1]:
        distances[0], distances[1] = distances[1], distances[0]

    max_offset = max(max(c for _r, c in shape) for shape in target_shapes)
    axis_col = max(distances) + max_offset + rng.randint(2, 4)
    w = axis_col + max_offset + rng.randint(4, 7)
    if w > 30:
        return None

    source_shapes: list[set[tuple[int, int]]] = []
    for dist, target_shape in zip(distances, target_shapes):
        source = {(r, axis_col - dist - off) for r, off in target_shape}
        if min(c for _r, c in source) < 0:
            return None
        if max(c for _r, c in source) >= axis_col:
            return None
        source_shapes.append(source)

    for i in range(n_shapes):
        for j in range(i + 1, n_shapes):
            if source_shapes[i] & source_shapes[j]:
                return None
            if _cells_touch(source_shapes[i], source_shapes[j]):
                return None

    unique_cells_by_shape: list[list[tuple[int, int]]] = []
    for idx, shape in enumerate(target_shapes):
        unique = [cell for cell in shape if len(cover_count[cell]) == 1]
        if not unique:
            return None
        unique_cells_by_shape.append(unique)

    marker_positions: list[tuple[int, int]] = []
    used_markers: set[tuple[int, int]] = set()
    for idx, unique in enumerate(unique_cells_by_shape):
        candidates = [cell for cell in unique if cell[1] >= 1]
        rng.shuffle(candidates)
        chosen = None
        for cell in candidates:
            if cell in used_markers:
                continue
            if any(abs(cell[0] - r) + abs(cell[1] - c) <= 1 for r, c in used_markers):
                continue
            chosen = cell
            break
        if chosen is None:
            return None
        marker_positions.append(chosen)
        used_markers.add(chosen)

    grid = np.full((h, w), BG, dtype=int)
    grid[:, axis_col] = AXIS

    obj_colors = [rng.choice(OBJ_COLORS) for _ in range(n_shapes)]
    for color, cells in zip(obj_colors, source_shapes):
        for r, c in cells:
            grid[r, c] = color

    for color, (r, off) in zip(marker_colors, marker_positions):
        c = axis_col + 1 + off
        if c >= w:
            return None
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    out = main(grid)

    if len(set(marker_colors)) < 2:
        return None

    winning_overlap_colors = {
        int(out[r, axis_col + 1 + off])
        for r, off in overlap_cells
    }
    losing_overlap_colors = {
        marker_colors[idx]
        for owners in cover_count.values()
        if len(owners) > 1
        for idx in owners
    }
    if len(overlap_cells) < 3:
        return None
    if len(winning_overlap_colors) == 1 and len(losing_overlap_colors) == 1:
        return None

    for idx, unique in enumerate(unique_cells_by_shape):
        marker_hits = 0
        for r, off in unique:
            c = axis_col + 1 + off
            if int(grid[r, c]) in MARKER_COLORS:
                marker_hits += 1
        if marker_hits != 1:
            return None

    for r, off in overlap_cells:
        c = axis_col + 1 + off
        if int(grid[r, c]) in MARKER_COLORS:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflection-overlap instance")
