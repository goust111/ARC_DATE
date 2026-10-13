import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_ATTEMPTS = 500

ROW_RANGE = (6, 9)
COL_RANGE = (8, 12)
MARKER_COUNT_CHOICES = (2, 3, 4)
OBSTACLE_COUNT_CHOICES = (1, 2, 3)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mirror_concat(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.concatenate((g, np.fliplr(g)), axis=1)


def _toggle_ray_color(color: int) -> int:
    return YELLOW if color == GREEN else GREEN


def _transform(
    grid: np.ndarray, *, collect_stats: bool = False
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 15, "input must fit ARC bounds after mirroring"

    out = _mirror_concat(g)
    split_col = g.shape[1]
    markers = sorted((int(r), int(c)) for r, c in np.argwhere(g == RED))

    stats = {
        "rays": 0,
        "axis_crossings": 0,
        "obstacle_stops": 0,
        "painted_stops": 0,
        "boundary_stops": 0,
    }

    for r, c in markers:
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            stats["rays"] += 1
            color = GREEN
            rr, cc = r + dr, c + dc
            if not _in_bounds(out.shape, rr, cc):
                stats["boundary_stops"] += 1
                continue

            while True:
                cell = int(out[rr, cc])
                if cell == BLUE:
                    stats["obstacle_stops"] += 1
                    break
                if cell == GREEN or cell == YELLOW:
                    stats["painted_stops"] += 1
                    break
                if cell == BLACK:
                    out[rr, cc] = color

                nr, nc = rr + dr, cc + dc
                if not _in_bounds(out.shape, nr, nc):
                    stats["boundary_stops"] += 1
                    break

                if dc != 0 and ((cc < split_col) != (nc < split_col)):
                    color = _toggle_ray_color(color)
                    stats["axis_crossings"] += 1

                rr, cc = nr, nc

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the grid to the right, then emit orthogonal rays from the original red markers with axis-triggered recoloring.

    Input structure:
    - The input is a black grid with 2-4 red marker pixels and 1-3 blue obstacle regions.
    - Markers are interpreted only from the original left half; the mirrored right-half markers are ordinary red pixels.
    - Obstacles and markers remain fixed after the horizontal mirror concatenation doubles the canvas width.

    Rule:
    - Build the output by appending a left-right mirror copy of the input to the right side.
    - Process the original red markers from top to bottom, then left to right; for each marker cast rays in the order up, right, down, left.
    - Each ray starts on the neighboring cell, paints only black cells, stops at the boundary, at a blue obstacle, or at a cell already painted by an earlier ray, and flips between green and yellow whenever it crosses the central mirror axis.

    Key insight:
    - The mirror step creates a meaningful vertical symmetry axis, so a ray's color is determined not by distance but by whether it has crossed from the original half into the mirrored half.
    """
    return _transform(grid)


def _component_sizes(mask: np.ndarray) -> list[int]:
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        size = 0
        while q:
            rr, cc = q.popleft()
            size += 1
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        sizes.append(size)
    return sizes


def _marker_adjacent_to_obstacle(grid: np.ndarray) -> bool:
    markers = {(int(r), int(c)) for r, c in np.argwhere(grid == RED)}
    for r, c in np.argwhere(grid == BLUE):
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in markers:
                return True
    return False


def _random_connected_shape(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(DIRS4)
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _build_special_obstacle(
    rows: int, cols: int, marker: tuple[int, int]
) -> set[tuple[int, int]] | None:
    r, c = marker
    max_len = min(3, c - 1)
    if max_len < 2:
        return None

    size = random.randint(2, 5)
    bar_len = random.randint(2, max_len)
    right_end = random.randint(bar_len - 1, c - 2)
    cells = {(r, cc) for cc in range(right_end - bar_len + 1, right_end + 1)}

    while len(cells) < size:
        candidates = []
        for rr, cc in cells:
            for nr, nc in _neighbors4(rr, cc):
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in cells:
                    continue
                if abs(nr - r) + abs(nc - c) == 1:
                    continue
                candidates.append((nr, nc))
        if not candidates:
            return None
        cells.add(random.choice(candidates))

    return cells


def _place_component(
    grid: np.ndarray,
    shape: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    marker_set: set[tuple[int, int]],
    special_row: int,
    special_col: int,
) -> bool:
    rows, cols = grid.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)

    for _ in range(120):
        r0 = random.randint(0, rows - max_r - 1)
        c0 = random.randint(0, cols - max_c - 1)
        placed = {(r0 + r, c0 + c) for r, c in shape}
        if placed & forbidden:
            continue
        if placed & marker_set:
            continue
        if any(abs(r - mr) + abs(c - mc) == 1 for r, c in placed for mr, mc in marker_set):
            continue
        if any(r == special_row and c > special_col for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = BLUE
        return True
    return False


def _sample_markers(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        marker_rows = sorted(random.sample(range(1, rows - 1), count))
        marker_cols = random.sample(range(1, cols - 1), count)
        first_col_choices = [c for c in range(3, cols - 2)]
        if not first_col_choices:
            return None
        first_col = random.choice(first_col_choices)
        if first_col in marker_cols:
            marker_cols.remove(first_col)
        else:
            marker_cols = marker_cols[:-1]
        marker_cols = [first_col] + marker_cols
        markers = list(zip(marker_rows, marker_cols))
        if len({c for _, c in markers}) != count:
            continue
        if min(abs(r1 - r2) + abs(c1 - c2) for i, (r1, c1) in enumerate(markers) for r2, c2 in markers[i + 1 :]) < 3:
            continue
        return markers
    return None


def _aligned_with_obstacle(grid: np.ndarray, markers: list[tuple[int, int]]) -> bool:
    obstacle_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)}
    for r, c in markers:
        if any(orow == r or ocol == c for orow, ocol in obstacle_cells):
            return True
    return False


def _valid_input(grid: np.ndarray) -> bool:
    colors = set(map(int, np.unique(grid)))
    if not colors <= {BLACK, BLUE, RED}:
        return False

    markers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED))
    if len(markers) not in MARKER_COUNT_CHOICES:
        return False
    if len({r for r, _ in markers}) != len(markers):
        return False
    if len({c for _, c in markers}) != len(markers):
        return False
    if _marker_adjacent_to_obstacle(grid):
        return False

    sizes = _component_sizes(grid == BLUE)
    if len(sizes) not in OBSTACLE_COUNT_CHOICES:
        return False
    if any(size < 2 or size > 5 for size in sizes):
        return False
    if not _aligned_with_obstacle(grid, markers):
        return False

    out, stats = _transform(grid, collect_stats=True)
    if stats["axis_crossings"] < 1:
        return False
    if stats["obstacle_stops"] < 1:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_RANGE)
    cols = random.randint(*COL_RANGE)
    count = random.choice(MARKER_COUNT_CHOICES)
    markers = _sample_markers(rows, cols, count)
    if markers is None:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for r, c in markers:
        grid[r, c] = RED
    marker_set = set(markers)

    special_marker = markers[0]
    special_component = _build_special_obstacle(rows, cols, special_marker)
    if special_component is None:
        return None
    if any(cell in marker_set for cell in special_component):
        return None
    if any(abs(r - special_marker[0]) + abs(c - special_marker[1]) == 1 for r, c in special_component):
        return None
    for r, c in special_component:
        grid[r, c] = BLUE

    forbidden = set(special_component)
    for r, c in list(special_component):
        forbidden.update(_neighbors4(r, c))

    target_components = random.choice(OBSTACLE_COUNT_CHOICES)
    need_painted_stop = random.random() < 0.65
    for _ in range(target_components - 1):
        size = random.randint(2, 5)
        shape = _random_connected_shape(size)
        if not _place_component(grid, shape, forbidden, marker_set, special_marker[0], special_marker[1]):
            return None
        for r, c in np.argwhere(grid == BLUE):
            forbidden.add((int(r), int(c)))
            forbidden.update(_neighbors4(int(r), int(c)))

    if not _valid_input(grid):
        return None

    _, stats = _transform(grid, collect_stats=True)
    if need_painted_stop and stats["painted_stops"] < 1:
        return None
    if random.random() < 0.35 and stats["boundary_stops"] < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and np.array_equal(main(grid), _transform(grid)):
            return grid
    raise RuntimeError("failed to generate a valid mirrored ray-network input")
