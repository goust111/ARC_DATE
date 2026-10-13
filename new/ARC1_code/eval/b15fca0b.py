from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import Color, find_components, in_bounds, neighbors4
import random


def _unique_shortest_path(g: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = map(int, g.shape)
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((H, W), nr, nc):
                continue
            if int(g[nr, nc]) == Color.BLUE:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))

    assert goal in dist, "expected the red endpoints to be connected through background cells"

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        r, c = cur
        predecessors = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if (nr, nc) in dist and dist[(nr, nc)] == dist[cur] - 1
        ]
        assert len(predecessors) == 1, "expected a unique shortest path between the red endpoints"
        cur = predecessors[0]
        if cur != start:
            path.append(cur)

    assert len(path) == dist[goal] - 1, "path reconstruction length mismatch"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the unique shortest background path between the two red endpoints yellow.

    Input structure
    - The grid contains black background cells, blue blocker cells, and exactly two red endpoint pixels.
    - The endpoints are connected by one shortest 4-neighbor route through non-blue cells.

    Rule
    - Keep every original cell unchanged.
    - Find the unique shortest 4-neighbor path from one red endpoint to the other while treating blue cells as blocked.
    - Recolor only the black interior cells of that path to yellow.

    Key insight
    - The blue cells form maze walls, and the output is the single shortest corridor between the two red endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED}, "unexpected input colors"

    red_coords = np.argwhere(g == Color.RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red_coords[0]))
    goal = tuple(map(int, red_coords[1]))

    out = g.copy()
    for r, c in _unique_shortest_path(g, start, goal):
        assert int(out[r, c]) == Color.BLACK, "path interior should contain only background cells"
        out[r, c] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------

# Variant mix and global retry budget.
GEN_MAX_ATTEMPTS = 300
GEN_VARIANTS = ("small", "large", "turn6", "turn7", "turn8", "five", "dense", "density")
GEN_VARIANT_WEIGHTS = (2, 3, 2, 2, 1, 1, 1, 3)

# Validation bounds from the task spec.
WALL_COUNT_RANGE = (1, 5)
WALL_LENGTH_FRAC_RANGE = (0.4, 0.85)
BLUE_DENSITY_RANGE = (0.19, 0.36)
PATH_LENGTH_RANGE = (13, 28)
TURN_COUNT_RANGE = (4, 10)
PATH_MANHATTAN_RATIO_RANGE = (1.5, 3.3)

# Symmetry transforms applied after constructing a canonical slalom.
ROTATION_COUNT = 4
FLIP_PROB = 0.5

# Slalom layout choices.
SMALL_SIZES = (5, 6)
SMALL_MID_OFFSETS = (0, 1)
LARGE_SIZES = (7, 8, 9, 10)
LARGE_WALL_COLS = (1, 3, 5)
DENSE_WALL_COLS = (1, 3, 5, 7)
FOUR_WALL_COLS = (1, 3, 5, 7)

# Hand-tuned fixed layouts that fill out density and turn-count coverage.
TURN_VARIANT_TARGETS = {"turn6": 6, "turn7": 7, "turn8": 8}
TURN6_TOP_CHOICES = (4, 5)
TURN8_GOAL_ROWS = (2, 3, 4, 5)
TURN7_SIZES = (9, 10)
TURN7_N9_GOAL_ROWS = (0, 1, 2, 3, 4, 5, 6)
TURN7_N10_BOTTOM2_CHOICES = (4, 5, 6, 7, 8)
TURN7_N10_TOP2_CHOICES = (5, 6, 7, 8)
TURN7_N10_GOAL_ROWS = (4, 5, 6, 7, 8)
DENSE_GAP_LAYOUTS = ((8, 0, (6, 7), 0), (0, 8, (1, 2), 8))
DENSITY_FILLER_CHOICES = ("d20", "d30", "d32")
DENSITY_FILLERS = {
    "d20": (8, ((1, 0, 3), (3, 3, 7), (6, 3, 6)), (4, 7), (3, 0)),
    "d30": (9, ((7, 0, 5), (1, 2, 8), (3, 0, 3), (5, 1, 7)), (0, 6), (8, 0)),
    "d32": (5, ((3, 1, 4), (1, 0, 3)), (0, 0), (3, 4)),
}
FIVE_WALL_SEGMENTS = ((1, 2, 9), (4, 1, 4), (5, 5, 8), (6, 0, 3), (8, 3, 6))
FIVE_WALL_ENDPOINTS = (((0, 4), (4, 9)), ((0, 1), (2, 9)))


def _gen_turn_count(path: list[tuple[int, int]]) -> int:
    return sum(
        (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
        for a, b, c in zip(path, path[1:], path[2:])
    )


def _gen_between(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _gen_wall_len_bounds(n: int) -> tuple[int, int]:
    return (
        int(np.ceil(WALL_LENGTH_FRAC_RANGE[0] * n)),
        int(np.floor(WALL_LENGTH_FRAC_RANGE[1] * n)),
    )


def _gen_transform(g: np.ndarray) -> np.ndarray:
    out = np.rot90(g, random.randrange(ROTATION_COUNT))
    if random.random() < FLIP_PROB:
        out = np.fliplr(out)
    if random.random() < FLIP_PROB:
        out = np.flipud(out)
    return out.copy()


def _gen_add_segment(g: np.ndarray, col: int, r0: int, r1: int) -> None:
    g[r0 : r1 + 1, col] = Color.BLUE


def _gen_add_segments(g: np.ndarray, segments: tuple[tuple[int, int, int], ...]) -> None:
    for col, r0, r1 in segments:
        _gen_add_segment(g, col, r0, r1)


def _gen_place_endpoints(g: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    g[start] = Color.RED
    g[goal] = Color.RED
    return g


def _gen_add_gap_walls(g: np.ndarray, cols: tuple[int, ...], gaps: list[int], max_len: int) -> None:
    """Render vertical walls that leave one side open at each requested gap."""
    n = int(g.shape[0])
    for col, gap in zip(cols, gaps):
        if gap >= n // 2:
            length = min(max_len, gap)
            _gen_add_segment(g, col, gap - length, gap - 1)
        else:
            length = min(max_len, n - gap - 1)
            _gen_add_segment(g, col, gap + 1, gap + length)


def _gen_small_slalom(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=int)
    mid_choices = [n // 2 + offset for offset in SMALL_MID_OFFSETS]
    e0 = random.choice(mid_choices)
    s1 = random.choice(mid_choices)
    if n == 5 and e0 == 2 and s1 == 3:
        s1 = 2
    _gen_add_segment(g, 1, 0, e0)
    _gen_add_segment(g, 3, s1, n - 1)
    return _gen_place_endpoints(g, (0, 0), (n - 1, 4))


def _gen_large_slalom(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=int)
    min_len, max_len = _gen_wall_len_bounds(n)
    low_choices = [r for r in range((2 * n) // 3, n - 1) if r >= min_len]
    high_choices = [r for r in range(1, max(2, n // 4 + 1)) if n - r - 1 >= min_len]

    if random.random() < FLIP_PROB:
        gaps = [random.choice(low_choices), random.choice(high_choices), n - 1]
    else:
        gaps = [random.choice(high_choices), random.choice(low_choices), 0]

    _gen_add_gap_walls(g, LARGE_WALL_COLS, gaps, max_len)
    start_row = 0 if gaps[0] >= n // 2 else n - 1
    goal_row = n - 1 if gaps[-1] >= n // 2 else 0
    return _gen_place_endpoints(g, (start_row, 0), (goal_row, 6))


def _gen_four_wall_layout(turn_target: int) -> tuple[int, tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    """Choose size, wall lengths, and endpoints for one four-wall slalom."""
    if turn_target == 6:
        n = 9
        lengths = (4, random.choice(TURN6_TOP_CHOICES), 5, 4)
        start = (0, 2)
        goal = (0, 8)
    elif turn_target == 8:
        n = 9
        lengths = (4, 5, 4, 6)
        start = (8, 0)
        goal = (random.choice(TURN8_GOAL_ROWS), 8)
    else:
        n = random.choice(TURN7_SIZES)
        if n == 9:
            lengths = (4, 5, 4, 6)
            start = (4, 0)
            goal = (random.choice(TURN7_N9_GOAL_ROWS), 8)
        else:
            lengths = (
                4,
                6,
                random.choice(TURN7_N10_BOTTOM2_CHOICES),
                random.choice(TURN7_N10_TOP2_CHOICES),
            )
            start = (9, 0)
            goal = (random.choice(TURN7_N10_GOAL_ROWS), 9)
    return n, lengths, start, goal


def _gen_four_wall_slalom(turn_target: int) -> np.ndarray:
    n, lengths, start, goal = _gen_four_wall_layout(turn_target)
    g = np.zeros((n, n), dtype=int)
    bottom_1, top_1, bottom_2, top_2 = lengths
    _gen_add_segment(g, FOUR_WALL_COLS[0], n - bottom_1, n - 1)
    _gen_add_segment(g, FOUR_WALL_COLS[1], 0, top_1 - 1)
    _gen_add_segment(g, FOUR_WALL_COLS[2], n - bottom_2, n - 1)
    _gen_add_segment(g, FOUR_WALL_COLS[3], 0, top_2 - 1)
    return _gen_place_endpoints(g, start, goal)


def _gen_dense_slalom() -> np.ndarray:
    n = 9
    g = np.zeros((n, n), dtype=int)
    gap_layout = DENSE_GAP_LAYOUTS[0] if random.random() < FLIP_PROB else DENSE_GAP_LAYOUTS[1]
    gaps = [gap_layout[0], gap_layout[1], random.choice(gap_layout[2]), gap_layout[3]]
    _, max_len = _gen_wall_len_bounds(n)
    _gen_add_gap_walls(g, DENSE_WALL_COLS, gaps, max_len)
    start_row = 0 if gaps[0] >= n // 2 else n - 1
    goal_row = n - 1 if gaps[-1] >= n // 2 else 0
    return _gen_place_endpoints(g, (start_row, 0), (goal_row, 8))


def _gen_density_filler() -> np.ndarray:
    n, segments, start, goal = DENSITY_FILLERS[random.choice(DENSITY_FILLER_CHOICES)]
    g = np.zeros((n, n), dtype=int)
    _gen_add_segments(g, segments)
    return _gen_place_endpoints(g, start, goal)


def _gen_five_wall_slalom() -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    _gen_add_segments(g, FIVE_WALL_SEGMENTS)
    return _gen_place_endpoints(g, *random.choice(FIVE_WALL_ENDPOINTS))


def _gen_wall_components_ok(g: np.ndarray) -> bool:
    n = int(g.shape[0])
    min_len, max_len = _gen_wall_len_bounds(n)
    comps = find_components(g == Color.BLUE, bg=False, connectivity=4, monochromatic=False, return_format="dict")
    if not _gen_between(len(comps), WALL_COUNT_RANGE):
        return False
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != r1 and c0 != c1:
            return False
        if not min_len <= int(comp["area"]) <= max_len:
            return False
    return True


def _gen_sample_canonical() -> tuple[np.ndarray, int | None]:
    """Sample an untransformed slalom and its optional exact turn target."""
    variant = random.choices(
        GEN_VARIANTS,
        weights=GEN_VARIANT_WEIGHTS,
        k=1,
    )[0]
    if variant == "small":
        return _gen_small_slalom(random.choice(SMALL_SIZES)), None
    elif variant == "large":
        return _gen_large_slalom(random.choice(LARGE_SIZES)), None
    elif variant in TURN_VARIANT_TARGETS:
        target_turn = TURN_VARIANT_TARGETS[variant]
        return _gen_four_wall_slalom(target_turn), target_turn
    elif variant == "dense":
        return _gen_dense_slalom(), None
    elif variant == "density":
        return _gen_density_filler(), None
    return _gen_five_wall_slalom(), None


def _gen_path_ok(g: np.ndarray, reds: np.ndarray, target_turn: int | None) -> bool:
    """Validate the solver-visible path constraints for a candidate grid."""
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))
    try:
        rev_path = _unique_shortest_path(g, start, goal)
        out = main(g)
    except AssertionError:
        return False

    path = [start] + list(reversed(rev_path)) + [goal]
    path_len = len(path)
    if not _gen_between(path_len, PATH_LENGTH_RANGE):
        return False

    turns = _gen_turn_count(path)
    if not _gen_between(turns, TURN_COUNT_RANGE):
        return False
    if target_turn is not None and turns != target_turn:
        return False

    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if manhattan == 0 or not _gen_between(path_len / manhattan, PATH_MANHATTAN_RATIO_RANGE):
        return False
    return int(np.sum(out == Color.YELLOW)) == path_len - 2


def _gen_candidate_ok(g: np.ndarray, target_turn: int | None) -> bool:
    """Run structural and oracle checks shared by all layout variants."""
    reds = np.argwhere(g == Color.RED)
    if reds.shape != (2, 2):
        return False

    n = int(g.shape[0])
    if not all(r in (0, n - 1) or c in (0, n - 1) for r, c in reds):
        return False

    density = float(np.mean(g == Color.BLUE))
    if not _gen_between(density, BLUE_DENSITY_RANGE):
        return False
    if not _gen_wall_components_ok(g):
        return False
    return _gen_path_ok(g, reds, target_turn)


def _gen_attempt() -> np.ndarray | None:
    g, target_turn = _gen_sample_canonical()
    g = _gen_transform(g)
    if not _gen_candidate_ok(g, target_turn):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid b15fca0b input")
