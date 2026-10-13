import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each red control pixel upward through black space, sidestepping right under gray blockers.

    Input structure
    - The background is black.
    - Red pixels are control pixels on the bottom row.
    - The only other foreground color is the obstacle color.

    Rule
    - From each red control pixel, repeatedly try to move one cell upward.
    - If the cell above is background, paint it red and continue from there.
    - If the cell above is an obstacle, step one cell right on the current row
      and paint that background cell red; if that right cell is unavailable or
      not background, stop that path.

    Key insight
    - The red output is not a set of fixed columns: each bottom control emits an
      upward path whose turns are caused locally by gray cells directly above it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    obstacle_colors = colors_present(g, exclude={bg, Color.RED})
    assert len(obstacle_colors) == 1, "expected one obstacle color besides red controls"
    obstacle = obstacle_colors[0]

    starts = np.argwhere(g == Color.RED)
    assert starts.size > 0, "expected at least one red control pixel"
    assert np.all(starts[:, 0] == height - 1), "red controls must be on the bottom row"

    out = g.copy()
    for sr, sc in starts:
        r = int(sr)
        c = int(sc)
        while r > 0:
            if g[r - 1, c] == bg:
                r -= 1
                out[r, c] = Color.RED
            elif g[r - 1, c] == obstacle:
                if c + 1 >= width or g[r, c + 1] != bg:
                    break
                c += 1
                out[r, c] = Color.RED
            else:
                raise AssertionError("unexpected color above a path")

    return out

# --------------- generator (o5x) ---------------

# Colors/RNG.
_GEN_BG, _GEN_RED, _GEN_OBSTACLE = 0, 2, 5
_GEN_RNG = np.random.default_rng()

# Canvas/controls; high rng bounds are exclusive.
_GEN_HEIGHT_RANGE, _GEN_WIDTH_RANGE = (12, 14), (13, 16)
_GEN_CONTROL_COUNT, _GEN_CONTROL_GAP_RANGE = 3, (2, 6)

# Path shape and turn weighting.
_GEN_PATH_MIN_CELLS, _GEN_PATH_MAX_CELLS = 4, 15
_GEN_BLOCKED_UP_MIN, _GEN_MAX_TURNS, _GEN_MAX_RIGHT_PER_UP_STEP = 3, 5, 2
_GEN_TURN_WEIGHT_BASE, _GEN_TURN_WEIGHT_SLOPE = 1.0, 0.2
_GEN_BLOCKED_PATH_COUNT_CHOICES = (0, 1, 1, 2)

# Obstacle density and row fill.
_GEN_OBSTACLE_COUNT_MIN, _GEN_OBSTACLE_COUNT_MAX = 25, 46
_GEN_DENSITY_PERCENT_RANGE, _GEN_PERCENT_DENOMINATOR = (16, 27), 100
_GEN_ROW_OBSTACLE_MAX = 6

# Retry budget.
_GEN_ATTEMPT_LIMIT = 300


def _gen_choose_columns(width, rng):
    choices = []
    for c0 in range(width):
        for d1 in range(*_GEN_CONTROL_GAP_RANGE):
            for d2 in range(*_GEN_CONTROL_GAP_RANGE):
                c1 = c0 + d1
                c2 = c1 + d2
                if c2 < width:
                    choices.append((c0, c1, c2))
    return choices[int(rng.integers(len(choices)))]


def _gen_split_runs(total, max_events, rng):
    if total == 0:
        return []
    min_events = (total + 1) // 2
    max_events = min(max_events, total)
    if min_events > max_events:
        return None
    n_events = int(rng.integers(min_events, max_events + 1))
    runs = [1] * n_events
    extra = total - n_events
    if extra:
        for idx in rng.choice(n_events, size=extra, replace=False):
            runs[int(idx)] += 1
    rng.shuffle(runs)
    return runs


def _gen_weighted_turn_count(low, high, rng):
    values = np.arange(low, high + 1)
    weights = _GEN_TURN_WEIGHT_BASE + _GEN_TURN_WEIGHT_SLOPE * values
    weights = weights / weights.sum()
    return int(rng.choice(values, p=weights))


def _gen_put_obstacle(grid, row_counts, blocked_cells, r, c):
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if grid[r, c] != _GEN_BG or (r, c) in blocked_cells or row_counts[r] >= _GEN_ROW_OBSTACLE_MAX:
        return False
    grid[r, c] = _GEN_OBSTACLE
    row_counts[r] += 1
    return True

def _gen_build_path(grid, row_counts, used_path, start_col, right_limit, side_limit, blocked, rng):
    h, w = grid.shape
    start = (h - 1, start_col)
    if start in used_path or grid[start] != _GEN_BG:
        return None

    possible_up = range(_GEN_BLOCKED_UP_MIN, h - 1) if blocked else (h - 1,)

    max_shift = right_limit - start_col
    feasible = []
    for up_steps in possible_up:
        min_turns = max(0, _GEN_PATH_MIN_CELLS - up_steps) if blocked else 0
        max_turns = min(
            _GEN_MAX_TURNS,
            max_shift,
            _GEN_PATH_MAX_CELLS - up_steps,
            _GEN_MAX_RIGHT_PER_UP_STEP * up_steps,
        )
        if min_turns <= max_turns:
            feasible.append((up_steps, min_turns, max_turns))
    if not feasible:
        return None

    up_steps, min_turns, max_turns = feasible[int(rng.integers(len(feasible)))]
    turn_count = _gen_weighted_turn_count(min_turns, max_turns, rng)
    runs = _gen_split_runs(turn_count, up_steps, rng)
    if runs is None:
        return None

    event_rows = {}
    if runs:
        for idx, run in zip(rng.choice(up_steps, size=len(runs), replace=False), runs):
            event_rows[int(idx)] = int(run)

    r = h - 1
    c = start_col
    path = [start]
    current_path = {start}

    def add_path_cell(rr, cc):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if grid[rr, cc] != _GEN_BG or (rr, cc) in used_path or (rr, cc) in current_path:
            return False
        current_path.add((rr, cc))
        path.append((rr, cc))
        return True

    for step_i in range(up_steps):
        run = event_rows.get(step_i, 0)
        if run:
            if c + run > right_limit:
                return None
            blockers = current_path | used_path
            for cc in range(c, c + run):
                if not _gen_put_obstacle(grid, row_counts, blockers, r - 1, cc):
                    return None
            for cc in range(c + 1, c + run + 1):
                if not add_path_cell(r, cc):
                    return None
            c += run

        if grid[r - 1, c] != _GEN_BG or (r - 1, c) in used_path or (r - 1, c) in current_path:
            return None
        r -= 1
        if not add_path_cell(r, c):
            return None

    if blocked:
        blockers = current_path | used_path
        if not _gen_put_obstacle(grid, row_counts, blockers, r - 1, c):
            return None
        if c + 1 < w:
            if c + 1 > side_limit:
                return None
            if not _gen_put_obstacle(grid, row_counts, blockers, r, c + 1):
                return None

    return path

def _gen_scatter_obstacles(grid, row_counts, used_path, target_count, rng):
    h, w = grid.shape
    obstacle_count = int(np.count_nonzero(grid == _GEN_OBSTACLE))
    if obstacle_count > target_count:
        return False

    for r in rng.permutation(h - 1):
        r = int(r)
        if row_counts[r] != 0:
            continue
        if obstacle_count >= target_count:
            return False
        choices = [
            c for c in range(w)
            if row_counts[r] < _GEN_ROW_OBSTACLE_MAX and grid[r, c] == _GEN_BG and (r, c) not in used_path
        ]
        if not choices:
            return False
        c = int(choices[int(rng.integers(len(choices)))])
        grid[r, c] = _GEN_OBSTACLE
        row_counts[r] += 1
        obstacle_count += 1

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if row_counts[r] < _GEN_ROW_OBSTACLE_MAX and grid[r, c] == _GEN_BG and (r, c) not in used_path
    ]
    rng.shuffle(candidates)
    for r, c in candidates:
        if obstacle_count >= target_count:
            break
        if row_counts[r] >= _GEN_ROW_OBSTACLE_MAX or grid[r, c] != _GEN_BG or (r, c) in used_path:
            continue
        grid[r, c] = _GEN_OBSTACLE
        row_counts[r] += 1
        obstacle_count += 1

    return obstacle_count == target_count

def _gen_choose_blocked_indices(rng):
    blocked_n = int(rng.choice(_GEN_BLOCKED_PATH_COUNT_CHOICES))
    return set(int(i) for i in rng.choice(_GEN_CONTROL_COUNT, size=blocked_n, replace=False))

def _gen_build_paths(grid, row_counts, starts, blocked_indices, rng):
    """Return non-overlapping paths plus their occupied cells, or None on failure."""
    w = grid.shape[1]
    used_path = set()
    paths = []
    for i, col in enumerate(starts):
        is_blocked = i in blocked_indices
        next_col = starts[i + 1] if i < _GEN_CONTROL_COUNT - 1 else w
        if is_blocked and i < _GEN_CONTROL_COUNT - 1:
            right_limit = next_col - 2
            side_limit = next_col - 1
        else:
            right_limit = next_col - 1
            side_limit = w - 1
        if right_limit < col:
            return None, None
        path = _gen_build_path(grid, row_counts, used_path, col, right_limit, side_limit, is_blocked, rng)
        if path is None:
            return None, None
        paths.append(path)
        used_path.update(path)
    return paths, used_path

def _gen_mark_starts(grid, starts):
    bottom = grid.shape[0] - 1
    if any(grid[bottom, col] != _GEN_BG for col in starts):
        return False
    grid[bottom, list(starts)] = _GEN_RED
    return True

def _gen_obstacle_target(grid, row_counts, rng):
    """Choose a legal total obstacle count after reserving one obstacle for empty rows."""
    h, w = grid.shape
    cell_count = h * w
    min_density, max_density = _GEN_DENSITY_PERCENT_RANGE
    min_count = max(
        _GEN_OBSTACLE_COUNT_MIN,
        (min_density * cell_count + _GEN_PERCENT_DENOMINATOR - 1) // _GEN_PERCENT_DENOMINATOR,
    )
    max_count = min(_GEN_OBSTACLE_COUNT_MAX, (max_density * cell_count) // _GEN_PERCENT_DENOMINATOR)
    row_fill_needed = sum(1 for r in range(h - 1) if row_counts[r] == 0)
    min_count = max(min_count, int(np.count_nonzero(grid == _GEN_OBSTACLE)) + row_fill_needed)
    if min_count > max_count:
        return None
    return int(rng.integers(min_count, max_count + 1))

def _gen_obstacles_are_valid(grid, row_counts):
    h, w = grid.shape
    obstacle_count = int(np.count_nonzero(grid == _GEN_OBSTACLE))
    density = obstacle_count / (h * w)
    min_density, max_density = _GEN_DENSITY_PERCENT_RANGE
    if not (
        _GEN_OBSTACLE_COUNT_MIN <= obstacle_count <= _GEN_OBSTACLE_COUNT_MAX
        and min_density / _GEN_PERCENT_DENOMINATOR <= density <= max_density / _GEN_PERCENT_DENOMINATOR
    ):
        return False
    if any(row_counts[r] < 1 or row_counts[r] > _GEN_ROW_OBSTACLE_MAX for r in range(h - 1)):
        return False
    return row_counts[h - 1] <= _GEN_ROW_OBSTACLE_MAX

def _gen_matches_solver(grid, paths):
    """Render expected red paths and compare them with the public solver."""
    expected = grid.copy()
    for path in paths:
        expected[tuple(zip(*path))] = _GEN_RED
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)

def _gen_attempt():
    rng = _GEN_RNG
    h = int(rng.integers(*_GEN_HEIGHT_RANGE))
    w = int(rng.integers(*_GEN_WIDTH_RANGE))
    starts = _gen_choose_columns(w, rng)
    grid = np.zeros((h, w), dtype=int)
    row_counts = [0] * h

    blocked_indices = _gen_choose_blocked_indices(rng)
    paths, used_path = _gen_build_paths(grid, row_counts, starts, blocked_indices, rng)
    if paths is None:
        return None

    if not _gen_mark_starts(grid, starts):
        return None

    target_count = _gen_obstacle_target(grid, row_counts, rng)
    if target_count is None:
        return None
    if not _gen_scatter_obstacles(grid, row_counts, used_path, target_count, rng):
        return None
    if not _gen_obstacles_are_valid(grid, row_counts):
        return None

    if not _gen_matches_solver(grid, paths):
        return None
    return grid

def generate_input():
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
