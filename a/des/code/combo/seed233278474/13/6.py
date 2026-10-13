import random

import numpy as np


BG = 0
POINTER_COLOR = 5
TARGET_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_HEIGHT_RANGE = (10, 20)
GRID_WIDTH_RANGE = (10, 20)
MIN_GAP = 2
MAX_PAIRS = 5
MAX_EXTRA_TARGETS = 3
PAIR_PLACEMENT_ATTEMPTS = 120
EXTRA_PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict[str, np.ndarray | tuple[int, int, int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) != color:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            components.append({"coords": arr, "bbox": bbox})

    return components


def _pointer_info(component: dict[str, np.ndarray | tuple[int, int, int, int]]) -> dict[str, np.ndarray | tuple[int, int]]:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    assert coords.shape == (3, 2), "gray objects must have exactly three cells"
    assert r1 == r0 + 1 and c1 == c0 + 1, "gray objects must fit a 2x2 box"

    coord_set = {(int(r), int(c)) for r, c in coords}
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = [cell for cell in corners if cell not in coord_set]
    assert len(missing) == 1, "gray object must be an L shape with one missing corner"

    mr, mc = missing[0]
    dr = -1 if mr == r0 else 1
    dc = -1 if mc == c0 else 1
    return {"coords": coords, "missing": (mr, mc), "direction": (dr, dc)}


def _scan_target_color(grid: np.ndarray, missing: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = missing
    dr, dc = direction
    while True:
        r += dr
        c += dc
        assert _in_bounds(grid.shape, r, c), "each pointer ray must hit a target block"
        value = int(grid[r, c])
        if value == BG:
            continue
        assert value != POINTER_COLOR, "pointer ray hit another pointer before a target"
        return value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray L-shaped pointer with the color of the first target block on its diagonal ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pointers = _find_components_of_color(g, POINTER_COLOR)
    assert pointers, "expected at least one gray pointer"

    out = g.copy()
    for component in pointers:
        info = _pointer_info(component)
        target_color = _scan_target_color(g, info["missing"], info["direction"])
        coords = np.asarray(info["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = target_color
    return out


def _target_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [
        (top, left),
        (top, left + 1),
        (top + 1, left),
        (top + 1, left + 1),
    ]


def _first_hit_from_target(top: int, left: int, dr: int, dc: int) -> tuple[int, int]:
    return top + (1 if dr == -1 else 0), left + (1 if dc == -1 else 0)


def _pointer_cells_from_missing(mr: int, mc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    top = mr - (1 if dr == 1 else 0)
    left = mc - (1 if dc == 1 else 0)
    missing_row = 1 if dr == 1 else 0
    missing_col = 1 if dc == 1 else 0
    return [
        (top + rr, left + cc)
        for rr in range(2)
        for cc in range(2)
        if not (rr == missing_row and cc == missing_col)
    ]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _cells_are_placeable(grid: np.ndarray, cells: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    h, w = grid.shape
    cell_set = set(cells)

    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BG or bool(forbidden[r, c]):
            return False

    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _cell_sets_are_separated(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _ray_cells_after(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _target_anchor_bounds(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    dr, dc = direction
    r_min = MIN_GAP + 1 if dr == 1 else 0
    r_max = h - 2 if dr == 1 else h - MIN_GAP - 3
    c_min = MIN_GAP + 1 if dc == 1 else 0
    c_max = w - 2 if dc == 1 else w - MIN_GAP - 3
    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _place_pair(
    grid: np.ndarray,
    forbidden: np.ndarray,
    direction: tuple[int, int],
    color: int,
) -> list[tuple[int, int]] | None:
    bounds = _target_anchor_bounds(grid.shape, direction)
    if bounds is None:
        return None

    h, w = grid.shape
    r_min, r_max, c_min, c_max = bounds
    dr, dc = direction

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        top = random.randint(r_min, r_max)
        left = random.randint(c_min, c_max)
        target_cells = _target_cells(top, left)
        if not _cells_are_placeable(grid, target_cells, forbidden):
            continue

        hit_r, hit_c = _first_hit_from_target(top, left, dr, dc)
        row_room = hit_r - 1 if dr == 1 else h - 2 - hit_r
        col_room = hit_c - 1 if dc == 1 else w - 2 - hit_c
        max_gap = min(row_room, col_room)
        if max_gap < MIN_GAP:
            continue

        gap = random.randint(MIN_GAP, max_gap)
        missing_r = hit_r - gap * dr
        missing_c = hit_c - gap * dc
        pointer_cells = _pointer_cells_from_missing(missing_r, missing_c, dr, dc)
        if not _cells_are_placeable(grid, pointer_cells, forbidden):
            continue
        if not _cell_sets_are_separated(pointer_cells, target_cells):
            continue

        path_cells = [
            (missing_r + step * dr, missing_c + step * dc)
            for step in range(1, gap)
        ]
        if any(int(grid[r, c]) != BG for r, c in path_cells):
            continue

        _paint_cells(grid, pointer_cells, POINTER_COLOR)
        _paint_cells(grid, target_cells, color)
        for r, c in _ray_cells_after(grid.shape, (missing_r, missing_c), direction):
            forbidden[r, c] = True
        return pointer_cells

    return None


def _place_extra_targets(grid: np.ndarray, forbidden: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    placed = 0
    for _ in range(count * EXTRA_PLACEMENT_ATTEMPTS):
        if placed >= count:
            return True
        top = random.randint(0, h - 2)
        left = random.randint(0, w - 2)
        cells = _target_cells(top, left)
        if not _cells_are_placeable(grid, cells, forbidden):
            continue
        _paint_cells(grid, cells, random.choice(TARGET_COLORS))
        placed += 1
    return placed == count


def _sample_directions(count: int) -> list[tuple[int, int]]:
    choices = DIRECTIONS.copy()
    random.shuffle(choices)
    result = choices[: min(count, len(choices))]
    while len(result) < count:
        result.append(random.choice(DIRECTIONS))
    random.shuffle(result)
    return result


def _max_pair_count(shape: tuple[int, int]) -> int:
    h, w = shape
    return max(1, min(MAX_PAIRS, (h * w) // 40, min(h, w) // 4))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    pair_count = random.randint(1, _max_pair_count(grid.shape))
    pointer_specs: list[tuple[list[tuple[int, int]], int]] = []

    for direction in _sample_directions(pair_count):
        color = random.choice(TARGET_COLORS)
        pointer_cells = _place_pair(grid, forbidden, direction, color)
        if pointer_cells is None:
            return None
        pointer_specs.append((pointer_cells, color))

    extra_count = random.randint(0, min(MAX_EXTRA_TARGETS, pair_count + 1))
    if not _place_extra_targets(grid, forbidden, extra_count):
        return None

    expected = grid.copy()
    for pointer_cells, color in pointer_specs:
        _paint_cells(expected, pointer_cells, color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
