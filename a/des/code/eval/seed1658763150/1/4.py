from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

QUADRANTS = ("tl", "tr", "bl", "br")
LABEL_COLOR_BY_QUADRANT = {
    "tl": BLUE,
    "tr": RED,
    "bl": GREEN,
    "br": ORANGE,
}
ENDPOINT_COLORS = [YELLOW, MAGENTA, AZURE, MAROON]
DIRS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _quadrant_of(cell: tuple[int, int], cross_row: int, cross_col: int) -> str:
    r, c = cell
    assert r != cross_row and c != cross_col, "markers and endpoints cannot lie on the cross axis"
    if r < cross_row:
        return "tl" if c < cross_col else "tr"
    return "bl" if c < cross_col else "br"


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    grey_comps = _components(grid == GREY)
    candidates: list[tuple[int, int]] = []
    for comp in grey_comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        row_counts = np.bincount(np.array(rows) - r0, minlength=height)
        col_counts = np.bincount(np.array(cols) - c0, minlength=width)
        full_rows = np.flatnonzero(row_counts == width)
        full_cols = np.flatnonzero(col_counts == height)
        if len(comp) == height + width - 1 and len(full_rows) == 1 and len(full_cols) == 1:
            candidates.append((r0 + int(full_rows[0]), c0 + int(full_cols[0])))
    assert len(candidates) == 1, "expected exactly one grey cross component"
    return candidates[0]


def _find_labels(
    grid: np.ndarray, cross_row: int, cross_col: int
) -> tuple[dict[str, int], np.ndarray]:
    counts = Counter(map(int, grid.reshape(-1)))
    label_colors = [color for color, count in counts.items() if color not in {BLACK, GREY} and count == 1]
    assert len(label_colors) == 4, "expected four singleton quadrant labels"

    quadrant_colors: dict[str, int] = {}
    label_mask = np.zeros_like(grid, dtype=bool)
    for color in label_colors:
        locations = np.argwhere(grid == color)
        assert len(locations) == 1
        cell = tuple(map(int, locations[0]))
        quadrant = _quadrant_of(cell, cross_row, cross_col)
        assert quadrant not in quadrant_colors, "each quadrant must have exactly one label"
        quadrant_colors[quadrant] = int(color)
        label_mask[cell] = True
    assert set(quadrant_colors) == set(QUADRANTS)
    return quadrant_colors, label_mask


def _find_pairs(
    grid: np.ndarray, quadrant_colors: dict[str, int], cross_row: int, cross_col: int
) -> list[dict[str, object]]:
    counts = Counter(map(int, grid.reshape(-1)))
    label_color_set = set(quadrant_colors.values())
    pairs: list[dict[str, object]] = []
    for color, count in counts.items():
        if color in {BLACK, GREY} or color in label_color_set or count != 2:
            continue
        points = [tuple(map(int, cell)) for cell in np.argwhere(grid == color)]
        points.sort()
        quadrants = tuple(_quadrant_of(point, cross_row, cross_col) for point in points)
        pairs.append(
            {
                "endpoint_color": int(color),
                "points": tuple(points),
                "quadrants": quadrants,
            }
        )
    assert pairs, "expected at least one endpoint pair"
    pairs.sort(key=lambda item: tuple(item["points"]))  # deterministic visible order
    return pairs


def _bfs_path(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    height, width = blocked.shape
    queue = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        r, c = cell
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if nxt in prev:
                continue
            if blocked[nr, nc] and nxt != goal:
                continue
            prev[nxt] = cell
            queue.append(nxt)
    if goal not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _solve(
    grid: np.ndarray, return_paths: bool = False
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_row, cross_col = _find_cross(g)
    quadrant_colors, label_mask = _find_labels(g, cross_row, cross_col)
    pairs = _find_pairs(g, quadrant_colors, cross_row, cross_col)

    out = g.copy()
    static_blocked = (g == GREY) | label_mask
    endpoint_mask = np.zeros_like(g, dtype=bool)
    for pair in pairs:
        for point in pair["points"]:
            endpoint_mask[point] = True
            out[point] = BLACK

    line_mask = np.zeros_like(g, dtype=bool)
    line_pixel_counts = {color: 0 for color in quadrant_colors.values()}
    path_records: list[dict[str, object]] = []

    for pair in pairs:
        start, goal = pair["points"]
        q_start, q_goal = pair["quadrants"]

        blocked = static_blocked | endpoint_mask | line_mask
        blocked[start] = False
        blocked[goal] = False
        path = _bfs_path(blocked, start, goal)
        assert path is not None, "every endpoint pair must be connectable"
        assert len(path) >= 3, "the visible line must contain at least one pixel"

        if q_start == q_goal:
            line_color = quadrant_colors[q_start]
        else:
            color_a = quadrant_colors[q_start]
            color_b = quadrant_colors[q_goal]
            count_a = line_pixel_counts[color_a]
            count_b = line_pixel_counts[color_b]
            if count_a < count_b:
                line_color = color_a
            elif count_b < count_a:
                line_color = color_b
            else:
                line_color = color_a

        for cell in path[1:-1]:
            assert not line_mask[cell], "lines cannot cross or overlap"
            out[cell] = line_color
            line_mask[cell] = True
        line_pixel_counts[line_color] += len(path) - 2

        path_records.append(
            {
                "path": path,
                "line_color": int(line_color),
                "same_quadrant": q_start == q_goal,
            }
        )

    if return_paths:
        return out, path_records
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


MIN_SIZE = 20
MAX_SIZE = 25
CROSS_MARGIN_MIN = 1
CROSS_MARGIN_MAX = 3
EDGE_BUFFER = 2
OBSTACLE_COUNT_MIN = 3
OBSTACLE_COUNT_MAX = 6
OBSTACLE_SIZE_MIN = 2
OBSTACLE_SIZE_MAX = 4
PLACEMENT_TRIES = 300
MAX_GENERATION_ATTEMPTS = 300
MIN_ENDPOINT_SPAN = 6


def _randint_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _place_cross(rng: np.random.Generator, grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    cross_row = _randint_inclusive(rng, height // 2 - 1, height // 2 + 1)
    cross_col = _randint_inclusive(rng, width // 2 - 1, width // 2 + 1)

    top = _randint_inclusive(rng, CROSS_MARGIN_MIN, min(CROSS_MARGIN_MAX, cross_row - 2))
    bottom_margin = _randint_inclusive(rng, CROSS_MARGIN_MIN, min(CROSS_MARGIN_MAX, height - cross_row - 3))
    left = _randint_inclusive(rng, CROSS_MARGIN_MIN, min(CROSS_MARGIN_MAX, cross_col - 2))
    right_margin = _randint_inclusive(rng, CROSS_MARGIN_MIN, min(CROSS_MARGIN_MAX, width - cross_col - 3))

    bottom = height - 1 - bottom_margin
    right = width - 1 - right_margin
    grid[cross_row, left : right + 1] = GREY
    grid[top : bottom + 1, cross_col] = GREY
    return cross_row, cross_col


def _place_labels(rng: np.random.Generator, grid: np.ndarray) -> dict[str, tuple[int, int]]:
    height, width = grid.shape
    positions = {
        "tl": (_randint_inclusive(rng, 0, 1), _randint_inclusive(rng, 0, 1)),
        "tr": (_randint_inclusive(rng, 0, 1), width - 1 - _randint_inclusive(rng, 0, 1)),
        "bl": (height - 1 - _randint_inclusive(rng, 0, 1), _randint_inclusive(rng, 0, 1)),
        "br": (
            height - 1 - _randint_inclusive(rng, 0, 1),
            width - 1 - _randint_inclusive(rng, 0, 1),
        ),
    }
    for quadrant, color in LABEL_COLOR_BY_QUADRANT.items():
        grid[positions[quadrant]] = color
    return positions


def _place_obstacles(rng: np.random.Generator, grid: np.ndarray, count: int) -> bool:
    height, width = grid.shape
    placed = 0
    tries = 0
    while placed < count and tries < PLACEMENT_TRIES:
        tries += 1
        rect_h = _randint_inclusive(rng, OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX)
        rect_w = _randint_inclusive(rng, OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX)
        max_r0 = height - EDGE_BUFFER - rect_h
        max_c0 = width - EDGE_BUFFER - rect_w
        if max_r0 < EDGE_BUFFER or max_c0 < EDGE_BUFFER:
            return False
        r0 = _randint_inclusive(rng, EDGE_BUFFER, max_r0)
        c0 = _randint_inclusive(rng, EDGE_BUFFER, max_c0)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height, r1 + 2)
        cc1 = min(width, c1 + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        placed += 1
    return placed == count


def _candidate_cells(
    grid: np.ndarray, cross_row: int, cross_col: int, quadrant: str
) -> list[tuple[int, int]]:
    height, width = grid.shape
    if quadrant == "tl":
        row_range = range(EDGE_BUFFER, cross_row - 1)
        col_range = range(EDGE_BUFFER, cross_col - 1)
    elif quadrant == "tr":
        row_range = range(EDGE_BUFFER, cross_row - 1)
        col_range = range(cross_col + 2, width - EDGE_BUFFER)
    elif quadrant == "bl":
        row_range = range(cross_row + 2, height - EDGE_BUFFER)
        col_range = range(EDGE_BUFFER, cross_col - 1)
    else:
        row_range = range(cross_row + 2, height - EDGE_BUFFER)
        col_range = range(cross_col + 2, width - EDGE_BUFFER)

    cells: list[tuple[int, int]] = []
    for r in row_range:
        for c in col_range:
            if grid[r, c] == BLACK:
                cells.append((r, c))
    return cells


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _endpoint_cell_ok(cell: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return all(_chebyshev(cell, other) > 1 for other in taken)


def _place_pair_endpoints(
    rng: np.random.Generator,
    grid: np.ndarray,
    cross_row: int,
    cross_col: int,
    quadrants: tuple[str, str],
    taken: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    qa, qb = quadrants
    cand_a = _candidate_cells(grid, cross_row, cross_col, qa)
    cand_b = _candidate_cells(grid, cross_row, cross_col, qb)
    if qa == qb and len(cand_a) < 2:
        return None
    if not cand_a or not cand_b:
        return None

    for _ in range(PLACEMENT_TRIES):
        a = _choice(rng, cand_a)
        b = _choice(rng, cand_b)
        if a == b:
            continue
        if not _endpoint_cell_ok(a, taken) or not _endpoint_cell_ok(b, taken):
            continue
        if _chebyshev(a, b) <= 1 or _manhattan(a, b) < MIN_ENDPOINT_SPAN:
            continue
        return a, b
    return None


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir = None
    for idx in range(1, len(path)):
        cur_dir = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = _randint_inclusive(rng, MIN_SIZE, MAX_SIZE)
    width = _randint_inclusive(rng, MIN_SIZE, MAX_SIZE)
    grid = np.full((height, width), BLACK, dtype=int)

    cross_row, cross_col = _place_cross(rng, grid)
    _place_labels(rng, grid)

    obstacle_count = _randint_inclusive(rng, OBSTACLE_COUNT_MIN, OBSTACLE_COUNT_MAX)
    if not _place_obstacles(rng, grid, obstacle_count):
        return None

    same_quadrants = list(QUADRANTS)
    first_same = _choice(rng, same_quadrants)
    same_quadrants.remove(first_same)
    second_same = _choice(rng, same_quadrants)
    adjacent_options = [("tl", "tr"), ("bl", "br"), ("tl", "bl"), ("tr", "br")]
    diagonal_options = [("tl", "br"), ("tr", "bl")]
    pair_quadrants = [
        (first_same, first_same),
        (second_same, second_same),
        _choice(rng, adjacent_options),
        _choice(rng, diagonal_options),
    ]
    pair_colors = ENDPOINT_COLORS.copy()
    rng.shuffle(pair_colors)

    taken: list[tuple[int, int]] = []
    for color, quadrants in zip(pair_colors, pair_quadrants):
        cells = _place_pair_endpoints(rng, grid, cross_row, cross_col, quadrants, taken)
        if cells is None:
            return None
        a, b = cells
        grid[a] = color
        grid[b] = color
        taken.extend([a, b])

    try:
        solved, records = _solve(grid, return_paths=True)
    except AssertionError:
        return None

    if len(records) != 4:
        return None
    if sum(int(record["same_quadrant"]) for record in records) != 2:
        return None
    if min(len(record["path"]) - 2 for record in records) < 2:
        return None
    if sum(_turn_count(record["path"]) > 0 for record in records) < 1:
        return None
    if len({int(record["line_color"]) for record in records}) < 2:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
