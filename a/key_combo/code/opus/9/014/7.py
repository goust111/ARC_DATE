import random
import sys
from pathlib import Path

import numpy as np


try:
    ROOT = Path(__file__).resolve().parents[6]
    ARC_UTILS_DIR = ROOT / "new" / "solvers"
    if str(ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(ARC_UTILS_DIR))
    from arc_utils import Color, colors_present, in_bounds
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7
        TEAL = 8

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        excluded = set() if exclude is None else set(int(c) for c in exclude)
        return sorted(int(v) for v in np.unique(grid) if int(v) not in excluded)

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


COARSE_SIZE = 5
BLOCK_SIZE = 3
SEPARATOR = 1
BIG_SIZE = COARSE_SIZE * BLOCK_SIZE + (COARSE_SIZE - 1) * SEPARATOR

GEN_RETRIES = 3000
MIN_PATH_CELLS = 4
MIN_TURNS = 1
MAX_TURNS = 5
MIN_OBSTACLES = 2


def _partition_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(int(v) for v in separators) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _compress_partition_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == Color.RED)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == Color.RED)]
    row_spans = _partition_spans(h, sep_rows)
    col_spans = _partition_spans(w, sep_cols)

    assert len(row_spans) == COARSE_SIZE and len(col_spans) == COARSE_SIZE

    out = np.full((len(row_spans), len(col_spans)), Color.BLACK, dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            block = g[r0:r1, c0:c1]
            block_colors = colors_present(block, exclude={Color.BLACK})
            assert len(block_colors) <= 1
            if not block_colors:
                continue
            color = int(block_colors[0])
            assert color in {Color.GREEN, Color.YELLOW, Color.TEAL}
            out[rr, cc] = color
    return out


def _find_single(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(grid, dtype=int) == int(color))
    assert coords.shape[0] == 1
    return int(coords[0, 0]), int(coords[0, 1])


def _start_state(grid: np.ndarray) -> tuple[int, int, int, int]:
    r, c = _find_single(grid, Color.GREEN)
    h, w = grid.shape
    if c == 0:
        dr, dc = 0, 1
    elif c == w - 1:
        dr, dc = 0, -1
    elif r == 0:
        dr, dc = 1, 0
    elif r == h - 1:
        dr, dc = -1, 0
    else:
        raise AssertionError("start marker must touch the coarse-grid border")
    return int(r), int(c), int(dr), int(dc)


def _is_traversable(cell: int) -> bool:
    return int(cell) in {Color.BLACK, Color.YELLOW}


def _is_goal_state(grid: np.ndarray, state: tuple[int, int, int, int]) -> bool:
    r, c, dr, dc = state
    nr, nc = int(r + dr), int(c + dc)
    return in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.YELLOW


def _successors(grid: np.ndarray, state: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    if _is_goal_state(grid, state):
        return []

    r, c, dr, dc = state
    nr, nc = int(r + dr), int(c + dc)
    if not in_bounds(grid.shape, nr, nc):
        return []

    nxt = int(grid[nr, nc])
    if nxt == Color.BLACK:
        return [(int(nr), int(nc), int(dr), int(dc))]
    if nxt != Color.TEAL:
        return []

    options: list[tuple[int, int, int, int]] = []
    for tdr, tdc in ((-dc, dr), (dc, -dr)):
        tr, tc = int(r + tdr), int(c + tdc)
        if in_bounds(grid.shape, tr, tc) and _is_traversable(int(grid[tr, tc])):
            options.append((int(r), int(c), int(tdr), int(tdc)))
    return options


def _all_solution_traces_from_start(
    grid: np.ndarray,
    start: tuple[int, int, int, int],
    limit: int = 2,
) -> list[tuple[tuple[int, int, int, int], ...]]:
    solutions: list[tuple[tuple[int, int, int, int], ...]] = []

    def dfs(
        state: tuple[int, int, int, int],
        seen: set[tuple[int, int, int, int]],
        trace: list[tuple[int, int, int, int]],
    ) -> None:
        if len(solutions) >= limit:
            return
        if _is_goal_state(grid, state):
            solutions.append(tuple(trace))
            return
        for succ in _successors(grid, state):
            if succ in seen:
                continue
            seen.add(succ)
            trace.append(succ)
            dfs(succ, seen, trace)
            trace.pop()
            seen.remove(succ)

    dfs(start, {start}, [start])
    return solutions


def _all_solution_traces(
    grid: np.ndarray,
    limit: int = 2,
) -> list[tuple[tuple[int, int, int, int], ...]]:
    return _all_solution_traces_from_start(grid, _start_state(grid), limit=limit)


def _paint_trace(
    grid: np.ndarray,
    trace: tuple[tuple[int, int, int, int], ...],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.asarray(grid, dtype=int).copy()
    painted: list[tuple[int, int]] = []
    for prev, curr in zip(trace, trace[1:]):
        if (prev[0], prev[1]) == (curr[0], curr[1]):
            continue
        rr, cc = int(curr[0]), int(curr[1])
        assert int(out[rr, cc]) == Color.BLACK
        out[rr, cc] = Color.ORANGE
        painted.append((rr, cc))
    assert len(set(painted)) == len(painted)
    return out, painted


def _solve_unique(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...], list[tuple[int, int]]]:
    state = _start_state(grid)
    trace = [state]
    seen = {state}
    while not _is_goal_state(grid, state):
        succs = _successors(grid, state)
        assert len(succs) == 1
        succ = succs[0]
        assert succ not in seen
        seen.add(succ)
        trace.append(succ)
        state = succ
    out, painted = _paint_trace(grid, tuple(trace))
    return out, tuple(trace), painted


def _trace_turn_count(trace: tuple[tuple[int, int, int, int], ...]) -> int:
    return sum(1 for prev, curr in zip(trace, trace[1:]) if (prev[0], prev[1]) == (curr[0], curr[1]))


def _prune_obstacles(grid: np.ndarray) -> np.ndarray | None:
    current = np.asarray(grid, dtype=int).copy()
    solutions = _all_solution_traces(current, limit=2)
    if len(solutions) != 1:
        return None
    current_out, _ = _paint_trace(current, solutions[0])

    changed = True
    while changed:
        changed = False
        for r_raw, c_raw in np.argwhere(current == Color.TEAL):
            r, c = int(r_raw), int(c_raw)
            trial = current.copy()
            trial[r, c] = Color.BLACK
            trial_solutions = _all_solution_traces(trial, limit=2)
            if len(trial_solutions) != 1:
                continue
            trial_out, _ = _paint_trace(trial, trial_solutions[0])
            if np.array_equal(trial_out, current_out):
                current = trial
                current_out = trial_out
                changed = True
                break
    return current


def _mirror_needed(mirrored: np.ndarray, mirrored_out: np.ndarray) -> bool:
    base = np.fliplr(mirrored)
    start_r, start_c = _find_single(base, Color.GREEN)
    forced_left = (start_r, start_c, 0, -1)
    solutions = _all_solution_traces_from_start(base, forced_left, limit=2)
    if len(solutions) != 1:
        return True
    shortcut_out, _ = _paint_trace(base, solutions[0])
    return not np.array_equal(np.fliplr(shortcut_out), mirrored_out)


def _marker_position() -> tuple[int, int]:
    return random.randrange(BLOCK_SIZE), random.randrange(BLOCK_SIZE)


def _expand_to_input(base_grid: np.ndarray) -> np.ndarray:
    big = np.zeros((BIG_SIZE, BIG_SIZE), dtype=int)
    separators = [BLOCK_SIZE + i * (BLOCK_SIZE + SEPARATOR) for i in range(COARSE_SIZE - 1)]
    for idx in separators:
        big[idx, :] = Color.RED
        big[:, idx] = Color.RED

    for rr in range(COARSE_SIZE):
        for cc in range(COARSE_SIZE):
            color = int(base_grid[rr, cc])
            if color == Color.BLACK:
                continue
            top = rr * (BLOCK_SIZE + SEPARATOR)
            left = cc * (BLOCK_SIZE + SEPARATOR)
            dr, dc = _marker_position()
            big[top + dr, left + dc] = color
    return big


def _random_mirrored_grid() -> np.ndarray:
    for _ in range(100):
        turn_count = random.randint(1, 3)
        rows = random.sample(range(COARSE_SIZE), turn_count + 1)
        cols = sorted(random.sample([2, 3, 4], turn_count), reverse=True)

        path_cells: set[tuple[int, int]] = {(rows[0], COARSE_SIZE - 1)}
        obstacle_cells: set[tuple[int, int]] = set()
        prev_col = COARSE_SIZE - 1
        valid = True

        for idx, turn_col in enumerate(cols):
            r0 = rows[idx]
            r1 = rows[idx + 1]
            step = 1 if r1 > r0 else -1

            if not in_bounds((COARSE_SIZE, COARSE_SIZE), r1 + step, turn_col):
                valid = False
                break

            for cc in range(prev_col, turn_col - 1, -1):
                path_cells.add((r0, cc))

            obstacle_cells.add((r0, turn_col - 1))
            opp_r = r0 - step
            if in_bounds((COARSE_SIZE, COARSE_SIZE), opp_r, turn_col):
                obstacle_cells.add((opp_r, turn_col))

            for rr in range(r0 + step, r1 + step, step):
                path_cells.add((rr, turn_col))

            obstacle_cells.add((r1 + step, turn_col))
            if in_bounds((COARSE_SIZE, COARSE_SIZE), r1, turn_col + 1):
                obstacle_cells.add((r1, turn_col + 1))

            prev_col = turn_col

        if not valid:
            continue

        end_row = rows[-1]
        for cc in range(prev_col, 0, -1):
            path_cells.add((end_row, cc))

        end_cell = (end_row, 0)
        if path_cells & obstacle_cells:
            continue
        if end_cell in obstacle_cells:
            continue

        grid = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=int)
        for r, c in obstacle_cells:
            grid[r, c] = Color.TEAL
        start_r = rows[0]
        grid[start_r, COARSE_SIZE - 1] = Color.GREEN
        grid[end_row, 0] = Color.YELLOW
        return grid

    return np.full((0, 0), Color.BLACK, dtype=int)


def _validate_candidate(mirrored: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if mirrored.shape != (COARSE_SIZE, COARSE_SIZE):
        return None
    if len(_all_solution_traces(mirrored, limit=2)) != 1:
        return None
    pruned = _prune_obstacles(mirrored)
    if pruned is None:
        return None
    solutions = _all_solution_traces(pruned, limit=2)
    if len(solutions) != 1:
        return None

    try:
        mirrored_out, trace, painted = _solve_unique(pruned)
    except AssertionError:
        return None
    turn_count = _trace_turn_count(trace)
    if len(painted) < MIN_PATH_CELLS or not (MIN_TURNS <= turn_count <= MAX_TURNS):
        return None

    if int(np.count_nonzero(pruned == Color.TEAL)) < MIN_OBSTACLES:
        return None

    if not _mirror_needed(pruned, mirrored_out):
        return None

    for r_raw, c_raw in np.argwhere(pruned == Color.TEAL):
        r, c = int(r_raw), int(c_raw)
        trial = pruned.copy()
        trial[r, c] = Color.BLACK
        trial_solutions = _all_solution_traces(trial, limit=2)
        if len(trial_solutions) == 1:
            trial_out, _ = _paint_trace(trial, trial_solutions[0])
            if np.array_equal(trial_out, mirrored_out):
                return None

    base = np.fliplr(pruned)
    big = _expand_to_input(base)
    compressed = _compress_partition_grid(big)
    if not np.array_equal(compressed, base):
        return None
    if not np.array_equal(main(big), mirrored_out):
        return None
    return big, mirrored_out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-partitioned grid into a small maze, mirror it horizontally, then trace the unique obstacle-triggered path.

    Input structure:
    - A large black grid is split by full red separator rows and columns into equal square blocks.
    - Each non-separator block is either empty, contains one teal obstacle marker, one green start marker, or one yellow end marker.
    - The green marker lies on one outer block edge of the coarse maze, and the yellow marker lies on the opposite outer block edge.

    Rule:
    - Drop the red separators and replace each block by one pixel: empty -> black, teal-marked -> teal, green-marked -> green, yellow-marked -> yellow.
    - Mirror the compressed maze left-to-right.
    - Starting from the green cell and heading inward from the border, move straight through black cells; when a teal cell blocks the next step, turn to the unique perpendicular traversable direction; stop when the next step would enter yellow.
    - Recolor every traversed black cell on that mirrored coarse maze to orange and return the mirrored 5x5 result.

    Key insight:
    - The true maze exists only after compressing separator blocks, and the mirror step changes the obstacle layout before the path-following rule is applied.
    """
    compressed = _compress_partition_grid(grid)
    mirrored = np.fliplr(compressed)
    solved, _trace, _painted = _solve_unique(mirrored)
    return solved


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mirrored = _random_mirrored_grid()
        validated = _validate_candidate(mirrored)
        if validated is not None:
            return validated[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
