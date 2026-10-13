import random

import numpy as np


WALL = 5
DOOR_BOTH_COLOR = 2
SYMMETRY_TYPES = ("horizontal", "vertical", "both", "none")

INTERIOR_SIZE_RANGE = (6, 8)
LAYOUT_CHOICES = ((2, 3), (3, 2), (3, 3))
MAX_GENERATION_ATTEMPTS = 300


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid[grid != WALL], return_counts=True)
    assert colors.size > 0, "expected non-wall cells"
    return int(colors[np.argmax(counts)])


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, mask[::-1, :])
    vertical = np.array_equal(mask, mask[:, ::-1])
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return "none"


def _wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_counts = np.count_nonzero(grid == WALL, axis=1)
    col_counts = np.count_nonzero(grid == WALL, axis=0)

    row_threshold = 0.5 * (int(row_counts.min()) + int(row_counts.max()))
    col_threshold = 0.5 * (int(col_counts.min()) + int(col_counts.max()))
    wall_rows = [int(r) for r, cnt in enumerate(row_counts) if float(cnt) > row_threshold]
    wall_cols = [int(c) for c, cnt in enumerate(col_counts) if float(cnt) > col_threshold]

    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected enclosing grey walls"
    assert wall_rows == sorted(wall_rows) and wall_cols == sorted(wall_cols)
    return wall_rows, wall_cols


def _room_furniture(region: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    mask = (region != bg) & (region != WALL)
    coords = np.argwhere(mask)
    assert coords.size > 0, "each room needs furniture"
    colors = np.unique(region[mask])
    assert colors.size == 1, "each room must contain exactly one furniture color"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    return int(colors[0]), patch


def _fill_room(out: np.ndarray, region: np.ndarray, color: int, kind: str, bg: int) -> None:
    if kind == "none":
        return
    if kind == "both":
        region[region == bg] = color
        return
    if kind == "horizontal":
        for r in range(region.shape[0]):
            if r % 2 == 0:
                row = region[r]
                row[row == bg] = color
        return
    assert kind == "vertical"
    for c in range(region.shape[1]):
        if c % 2 == 0:
            col_mask = region[:, c] == bg
            region[col_mask, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated rooms according to each room furniture's symmetry, then blend door colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    wall_rows, wall_cols = _wall_lines(g)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1

    out = g.copy()
    room_info: dict[tuple[int, int], dict[str, object]] = {}

    for rr in range(room_rows):
        r0, r1 = wall_rows[rr], wall_rows[rr + 1]
        assert r1 - r0 >= 3, "room interior must be at least one cell tall"
        for cc in range(room_cols):
            c0, c1 = wall_cols[cc], wall_cols[cc + 1]
            assert c1 - c0 >= 3, "room interior must be at least one cell wide"
            region = out[r0 + 1 : r1, c0 + 1 : c1]
            color, patch = _room_furniture(g[r0 + 1 : r1, c0 + 1 : c1], bg)
            kind = _symmetry_kind(patch)
            _fill_room(out, region, color, kind, bg)
            room_info[(rr, cc)] = {"kind": kind, "filled": kind != "none", "color": color}

    for rr in range(room_rows - 1):
        wall_r = wall_rows[rr + 1]
        for cc in range(room_cols):
            c0, c1 = wall_cols[cc], wall_cols[cc + 1]
            segment = g[wall_r, c0 + 1 : c1]
            door_offsets = np.flatnonzero(segment == bg)
            assert door_offsets.size == 1, "each shared horizontal wall must contain one door"
            door_c = c0 + 1 + int(door_offsets[0])
            top = room_info[(rr, cc)]
            bottom = room_info[(rr + 1, cc)]
            top_filled = bool(top["filled"])
            bottom_filled = bool(bottom["filled"])
            if top_filled and bottom_filled:
                out[wall_r, door_c] = DOOR_BOTH_COLOR
            elif top_filled:
                out[wall_r, door_c] = int(top["color"])
            elif bottom_filled:
                out[wall_r, door_c] = int(bottom["color"])
            else:
                out[wall_r, door_c] = bg

    for rr in range(room_rows):
        r0, r1 = wall_rows[rr], wall_rows[rr + 1]
        for cc in range(room_cols - 1):
            wall_c = wall_cols[cc + 1]
            segment = g[r0 + 1 : r1, wall_c]
            door_offsets = np.flatnonzero(segment == bg)
            assert door_offsets.size == 1, "each shared vertical wall must contain one door"
            door_r = r0 + 1 + int(door_offsets[0])
            left = room_info[(rr, cc)]
            right = room_info[(rr, cc + 1)]
            left_filled = bool(left["filled"])
            right_filled = bool(right["filled"])
            if left_filled and right_filled:
                out[door_r, wall_c] = DOOR_BOTH_COLOR
            elif left_filled:
                out[door_r, wall_c] = int(left["color"])
            elif right_filled:
                out[door_r, wall_c] = int(right["color"])
            else:
                out[door_r, wall_c] = bg

    return out


_BOTH_TEMPLATES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
)

_HORIZONTAL_ONLY_TEMPLATES = (
    np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=bool),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=bool,
    ),
)

_VERTICAL_ONLY_TEMPLATES = tuple(mask.T.copy() for mask in _HORIZONTAL_ONLY_TEMPLATES)

_NONE_TEMPLATES = (
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _dihedral(mask: np.ndarray) -> list[np.ndarray]:
    rotations = [np.rot90(mask, k).copy() for k in range(4)]
    flips = [np.fliplr(rot).copy() for rot in rotations]
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for arr in rotations + flips:
        key = arr.tobytes() + str(arr.shape).encode()
        if key not in seen:
            seen.add(key)
            variants.append(arr)
    return variants


def _sample_template(kind: str, max_h: int, max_w: int) -> np.ndarray:
    if kind == "both":
        pool = _BOTH_TEMPLATES
        allowed = _dihedral
    elif kind == "horizontal":
        pool = _HORIZONTAL_ONLY_TEMPLATES
        allowed = lambda x: [x.copy(), np.fliplr(x).copy(), np.rot90(x, 2).copy()]
    elif kind == "vertical":
        pool = _VERTICAL_ONLY_TEMPLATES
        allowed = lambda x: [x.copy(), np.flipud(x).copy(), np.rot90(x, 2).copy()]
    else:
        pool = _NONE_TEMPLATES
        allowed = _dihedral

    candidates: list[np.ndarray] = []
    for base in pool:
        for variant in allowed(base):
            if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
                if _connected(variant) and _symmetry_kind(variant) == kind:
                    candidates.append(variant)
    assert candidates, f"no {kind} furniture fits the room"
    return random.choice(candidates).copy()


def _room_graph(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            neighbors: list[tuple[int, int]] = []
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols:
                    neighbors.append((rr, cc))
            graph[pos] = neighbors
    return graph


def _adjacency_case(a: str, b: str) -> str:
    a_filled = a != "none"
    b_filled = b != "none"
    if a_filled and b_filled:
        return "both"
    if a_filled or b_filled:
        return "one"
    return "none"


def _sample_room_types(rows: int, cols: int) -> dict[tuple[int, int], str]:
    total = rows * cols
    types = ["horizontal", "vertical", "both", "none", "none"]
    while len(types) < total:
        counts = {kind: types.count(kind) for kind in SYMMETRY_TYPES}
        target = min(counts.values())
        choices = [kind for kind, count in counts.items() if count == target]
        types.append(random.choice(choices))

    graph = _room_graph(rows, cols)
    positions = list(graph)
    for _ in range(200):
        shuffled = types[:]
        random.shuffle(shuffled)
        assignment = dict(zip(positions, shuffled, strict=False))
        cases = {
            _adjacency_case(assignment[pos], assignment[nb])
            for pos in positions
            for nb in graph[pos]
            if pos < nb
        }
        if cases == {"both", "one", "none"}:
            return assignment
    raise RuntimeError("failed to arrange room types with all door branches")


def _sample_room_colors(
    rows: int, cols: int, bg: int
) -> dict[tuple[int, int], int]:
    graph = _room_graph(rows, cols)
    palette = [color for color in range(10) if color not in {bg, WALL, DOOR_BOTH_COLOR}]
    order = list(graph)
    random.shuffle(order)
    colors: dict[tuple[int, int], int] = {}
    for pos in order:
        blocked = {colors[nb] for nb in graph[pos] if nb in colors}
        choices = [color for color in palette if color not in blocked]
        assert choices, "palette should be large enough for planar room adjacency"
        colors[pos] = random.choice(choices)
    return colors


def _place_walls_and_doors(
    bg: int, row_sizes: list[int], col_sizes: list[int]
) -> tuple[np.ndarray, list[int], list[int]]:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), bg, dtype=int)

    wall_rows = [0]
    acc = 0
    for size in row_sizes:
        acc += size + 1
        wall_rows.append(acc)
    wall_cols = [0]
    acc = 0
    for size in col_sizes:
        acc += size + 1
        wall_cols.append(acc)

    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    for rr in range(len(row_sizes) - 1):
        wall_r = wall_rows[rr + 1]
        for cc in range(len(col_sizes)):
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1] - 1
            grid[wall_r, random.randint(left, right)] = bg

    for rr in range(len(row_sizes)):
        top = wall_rows[rr] + 1
        bottom = wall_rows[rr + 1] - 1
        for cc in range(len(col_sizes) - 1):
            wall_c = wall_cols[cc + 1]
            grid[random.randint(top, bottom), wall_c] = bg

    return grid, wall_rows, wall_cols


def _build_candidate() -> np.ndarray:
    room_rows, room_cols = random.choice(LAYOUT_CHOICES)
    row_sizes = [random.randint(*INTERIOR_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [random.randint(*INTERIOR_SIZE_RANGE) for _ in range(room_cols)]
    bg = random.choice([color for color in range(10) if color not in {WALL, DOOR_BOTH_COLOR}])

    grid, wall_rows, wall_cols = _place_walls_and_doors(bg, row_sizes, col_sizes)
    room_types = _sample_room_types(room_rows, room_cols)
    room_colors = _sample_room_colors(room_rows, room_cols, bg)

    for rr in range(room_rows):
        for cc in range(room_cols):
            kind = room_types[(rr, cc)]
            color = room_colors[(rr, cc)]
            inner_h = wall_rows[rr + 1] - wall_rows[rr] - 1
            inner_w = wall_cols[cc + 1] - wall_cols[cc] - 1
            furniture = _sample_template(kind, inner_h - 2, inner_w - 2)
            top = random.randint(1, inner_h - furniture.shape[0] - 1)
            left = random.randint(1, inner_w - furniture.shape[1] - 1)
            region = grid[wall_rows[rr] + 1 : wall_rows[rr + 1], wall_cols[cc] + 1 : wall_cols[cc + 1]]
            region[top : top + furniture.shape[0], left : left + furniture.shape[1]][furniture] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        try:
            out = main(grid)
        except AssertionError:
            continue

        kinds = set()
        wall_rows, wall_cols = _wall_lines(grid)
        bg = _bg_color(grid)
        for rr in range(len(wall_rows) - 1):
            for cc in range(len(wall_cols) - 1):
                region = grid[wall_rows[rr] + 1 : wall_rows[rr + 1], wall_cols[cc] + 1 : wall_cols[cc + 1]]
                _, patch = _room_furniture(region, bg)
                kinds.add(_symmetry_kind(patch))
        if kinds != set(SYMMETRY_TYPES):
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
