import numpy as np


BLACK = 0
RED = 2
GREY = 5
PATTERN_COLORS = tuple(color for color in range(10) if color not in {BLACK, RED, GREY})
DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
EDGE_TO_DIR = {
    "top": "down",
    "bottom": "up",
    "left": "right",
    "right": "left",
}
BASE_SPRITES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
]
HORIZONTAL_ARROW_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
)
VERTICAL_ARROW_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
)
_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each red region-to-region arrow chain until a terminal region with no arrow,
    then replace every region on that chain by the terminal region's sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_lines, col_lines, cell_h, cell_w = _parse_lattice(g)
    region_rows = len(row_lines) - 1
    region_cols = len(col_lines) - 1

    sprites: dict[tuple[int, int], np.ndarray] = {}
    arrows: dict[tuple[int, int], str] = {}

    for rr in range(region_rows):
        for cc in range(region_cols):
            inner = _region_view(g, row_lines, col_lines, rr, cc)
            assert inner.shape == (cell_h, cell_w)
            assert not np.any(inner == GREY), "grey must only appear on lattice lines"

            red_mask = inner == RED
            sprites[(rr, cc)] = np.where(red_mask, BLACK, inner)
            direction = _decode_arrow(red_mask)
            if direction is not None:
                arrows[(rr, cc)] = direction

    terminal_cache: dict[tuple[int, int], tuple[int, int]] = {}
    visiting: set[tuple[int, int]] = set()

    def resolve(cell: tuple[int, int]) -> tuple[int, int]:
        if cell in terminal_cache:
            return terminal_cache[cell]
        if cell not in arrows:
            terminal_cache[cell] = cell
            return cell
        assert cell not in visiting, "arrow graph must be acyclic"
        visiting.add(cell)
        dr, dc = DIR_TO_STEP[arrows[cell]]
        nr = cell[0] + dr
        nc = cell[1] + dc
        assert 0 <= nr < region_rows and 0 <= nc < region_cols, "arrow must stay inside the region grid"
        terminal = resolve((nr, nc))
        visiting.remove(cell)
        terminal_cache[cell] = terminal
        return terminal

    out = g.copy()
    for rr in range(region_rows):
        for cc in range(region_cols):
            inner = _region_view(out, row_lines, col_lines, rr, cc)
            terminal = resolve((rr, cc))
            inner[:] = sprites[terminal]
    return out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _parse_lattice(grid: np.ndarray) -> tuple[list[int], list[int], int, int]:
    row_lines = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_lines = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected full grey lattice lines"
    assert row_lines[0] == 0 and row_lines[-1] == grid.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == grid.shape[1] - 1

    row_steps = np.diff(row_lines)
    col_steps = np.diff(col_lines)
    assert len(set(map(int, row_steps))) == 1, "row lattice spacing must be uniform"
    assert len(set(map(int, col_steps))) == 1, "column lattice spacing must be uniform"

    cell_h = int(row_steps[0] - 1)
    cell_w = int(col_steps[0] - 1)
    assert cell_h >= 2 and cell_w >= 2
    return row_lines, col_lines, cell_h, cell_w


def _region_view(grid: np.ndarray, row_lines: list[int], col_lines: list[int], rr: int, cc: int) -> np.ndarray:
    r0 = row_lines[rr] + 1
    r1 = row_lines[rr + 1]
    c0 = col_lines[cc] + 1
    c1 = col_lines[cc + 1]
    return grid[r0:r1, c0:c1]


def _decode_arrow(red_mask: np.ndarray) -> str | None:
    coords = np.argwhere(red_mask)
    if coords.size == 0:
        return None
    assert coords.shape[0] == 3, "each arrow must use exactly three red cells"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2, "arrow must be a 2x2 L triomino"

    box = red_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert int(np.count_nonzero(box)) == 3, "arrow bbox must contain a 3-cell L"

    touches = []
    if np.any(red_mask[0, :]):
        touches.append("top")
    if np.any(red_mask[-1, :]):
        touches.append("bottom")
    if np.any(red_mask[:, 0]):
        touches.append("left")
    if np.any(red_mask[:, -1]):
        touches.append("right")
    assert len(touches) == 1, "generator-style arrows touch exactly one region edge"
    return EDGE_TO_DIR[touches[0]]


def _gen_attempt() -> np.ndarray | None:
    region_rows = int(_RNG.integers(3, 6))
    region_cols = int(_RNG.integers(3, 6))
    cell_h = int(_RNG.integers(6, 9))
    cell_w = int(_RNG.integers(6, 9))
    total_regions = region_rows * region_cols
    root_cap = min(4, total_regions - 2)
    root_count = int(_RNG.integers(1, root_cap + 1))

    roots, parents, depths = _grow_forest(region_rows, region_cols, root_count)
    if max(depths.values()) < 2:
        return None

    grid = _blank_grid(region_rows, region_cols, cell_h, cell_w)
    row_lines = list(range(0, region_rows * (cell_h + 1) + 1, cell_h + 1))
    col_lines = list(range(0, region_cols * (cell_w + 1) + 1, cell_w + 1))
    sprites: dict[tuple[int, int], np.ndarray] = {}
    directions: dict[tuple[int, int], str] = {}

    for rr in range(region_rows):
        for cc in range(region_cols):
            cell = (rr, cc)
            direction = None
            if cell in parents:
                pr, pc = parents[cell]
                dr = pr - rr
                dc = pc - cc
                direction = next(name for name, step in DIR_TO_STEP.items() if step == (dr, dc))
                directions[cell] = direction
                arrow_mask = _make_arrow_mask(direction, cell_h, cell_w)
            else:
                arrow_mask = np.zeros((cell_h, cell_w), dtype=bool)

            sprite = _sample_sprite(cell_h, cell_w, arrow_mask)
            if sprite is None:
                return None

            inner = _region_view(
                grid,
                row_lines,
                col_lines,
                rr,
                cc,
            )
            inner[:] = sprite
            inner[arrow_mask] = RED
            sprites[cell] = sprite

    resolve_cache: dict[tuple[int, int], tuple[int, int]] = {}

    def resolve(cell: tuple[int, int]) -> tuple[int, int]:
        if cell in resolve_cache:
            return resolve_cache[cell]
        if cell not in parents:
            resolve_cache[cell] = cell
            return cell
        terminal = resolve(parents[cell])
        resolve_cache[cell] = terminal
        return terminal

    changed = sum(not np.array_equal(sprites[cell], sprites[resolve(cell)]) for cell in parents)
    if changed == 0:
        return None
    if len(parents) >= 3 and len(set(directions.values())) < 2:
        return None
    if len(roots) > 1 and len({sprites[root].tobytes() for root in roots}) < 2:
        return None

    expected = _blank_grid(region_rows, region_cols, cell_h, cell_w)
    for rr in range(region_rows):
        for cc in range(region_cols):
            _region_view(expected, row_lines, col_lines, rr, cc)[:] = sprites[resolve((rr, cc))]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _blank_grid(region_rows: int, region_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    height = region_rows * cell_h + region_rows + 1
    width = region_cols * cell_w + region_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)
    grid[:: cell_h + 1, :] = GREY
    grid[:, :: cell_w + 1] = GREY
    return grid


def _grow_forest(region_rows: int, region_cols: int, root_count: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]]:
    cells = [(rr, cc) for rr in range(region_rows) for cc in range(region_cols)]
    root_indices = _RNG.choice(len(cells), size=root_count, replace=False)
    roots = [cells[int(idx)] for idx in root_indices]

    assigned = set(roots)
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    depths = {cell: 0 for cell in roots}

    while len(assigned) < len(cells):
        frontier: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for cell in cells:
            if cell in assigned:
                continue
            neighbors = [nbr for nbr in _neighbors(cell, region_rows, region_cols) if nbr in assigned]
            if neighbors:
                frontier.append((cell, neighbors))
        if not frontier:
            raise AssertionError("frontier unexpectedly empty")
        cell, neighbors = frontier[int(_RNG.integers(len(frontier)))]
        parent = neighbors[int(_RNG.integers(len(neighbors)))]
        parents[cell] = parent
        depths[cell] = depths[parent] + 1
        assigned.add(cell)

    return roots, parents, depths


def _neighbors(cell: tuple[int, int], region_rows: int, region_cols: int) -> list[tuple[int, int]]:
    rr, cc = cell
    result = []
    if rr > 0:
        result.append((rr - 1, cc))
    if rr + 1 < region_rows:
        result.append((rr + 1, cc))
    if cc > 0:
        result.append((rr, cc - 1))
    if cc + 1 < region_cols:
        result.append((rr, cc + 1))
    return result


def _make_arrow_mask(direction: str, cell_h: int, cell_w: int) -> np.ndarray:
    mask = np.zeros((cell_h, cell_w), dtype=bool)
    if direction == "right":
        top = int(_RNG.integers(1, cell_h - 2))
        left = 0
        shape = HORIZONTAL_ARROW_SHAPES[int(_RNG.integers(len(HORIZONTAL_ARROW_SHAPES)))]
    elif direction == "left":
        top = int(_RNG.integers(1, cell_h - 2))
        left = cell_w - 2
        shape = np.fliplr(HORIZONTAL_ARROW_SHAPES[int(_RNG.integers(len(HORIZONTAL_ARROW_SHAPES)))])
    elif direction == "down":
        top = 0
        left = int(_RNG.integers(1, cell_w - 2))
        shape = VERTICAL_ARROW_SHAPES[int(_RNG.integers(len(VERTICAL_ARROW_SHAPES)))]
    elif direction == "up":
        top = cell_h - 2
        left = int(_RNG.integers(1, cell_w - 2))
        shape = np.flipud(VERTICAL_ARROW_SHAPES[int(_RNG.integers(len(VERTICAL_ARROW_SHAPES)))])
    else:
        raise ValueError(f"unknown direction: {direction}")
    mask[top : top + 2, left : left + 2] = shape
    return mask


def _sample_sprite(cell_h: int, cell_w: int, arrow_mask: np.ndarray) -> np.ndarray | None:
    forbidden = _expand_mask(arrow_mask)
    for _ in range(80):
        mask = _random_sprite_mask()
        patch_h, patch_w = mask.shape
        if patch_h > cell_h or patch_w > cell_w:
            continue
        anchors = []
        for top in range(cell_h - patch_h + 1):
            for left in range(cell_w - patch_w + 1):
                if np.any(forbidden[top : top + patch_h, left : left + patch_w] & mask):
                    continue
                anchors.append((top, left))
        if not anchors:
            continue
        top, left = anchors[int(_RNG.integers(len(anchors)))]
        sprite = np.full((cell_h, cell_w), BLACK, dtype=int)
        colored = _colorize_mask(mask)
        region = sprite[top : top + patch_h, left : left + patch_w]
        region[mask] = colored[mask]
        if np.any((sprite != BLACK) & arrow_mask):
            continue
        return sprite
    return None


def _random_sprite_mask() -> np.ndarray:
    mask = BASE_SPRITES[int(_RNG.integers(len(BASE_SPRITES)))]
    transformed = np.array(mask, copy=True)
    if int(_RNG.integers(2)):
        transformed = np.fliplr(transformed)
    transformed = np.rot90(transformed, int(_RNG.integers(4)))
    coords = np.argwhere(transformed)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return transformed[r0 : r1 + 1, c0 : c1 + 1]


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    coords = np.argwhere(mask)
    if coords.shape[0] == 1:
        patch[mask] = int(_RNG.choice(PATTERN_COLORS))
        return patch

    color_count = 1 if int(_RNG.integers(100)) < 60 else 2
    colors = _RNG.choice(PATTERN_COLORS, size=color_count, replace=False)
    if color_count == 1:
        patch[mask] = int(colors[0])
        return patch

    assignments = np.full(coords.shape[0], int(colors[0]), dtype=int)
    second_count = int(_RNG.integers(1, coords.shape[0]))
    second_idx = _RNG.choice(coords.shape[0], size=second_count, replace=False)
    assignments[second_idx] = int(colors[1])
    if np.all(assignments == int(colors[1])):
        assignments[0] = int(colors[0])
    for idx, (rr, cc) in enumerate(coords):
        patch[int(rr), int(cc)] = int(assignments[idx])
    return patch


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    coords = np.argwhere(mask)
    for rr, cc in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    expanded[nr, nc] = True
    return expanded
