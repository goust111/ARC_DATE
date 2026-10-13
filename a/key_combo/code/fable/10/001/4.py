from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREY = 5

_RNG = np.random.default_rng()

_GEN_RETRY_LIMIT = 400
_PATH_RETRY_LIMIT = 120
_HEIGHT_RANGE = (12, 19)
_PATH_COUNT_CHOICES = (2, 3, 4)
_PATH_COUNT_WEIGHTS = np.array((1, 4, 3), dtype=float)
_LANE_WIDTH_RANGE = (3, 7)
_LANE_GAP_RANGE = (1, 4)
_LEFT_MARGIN_RANGE = (0, 3)
_RIGHT_MARGIN_RANGE = (1, 4)
_START_OFFSET_CHOICES = (0, 1)
_MAX_RUN_PER_ROW = 3
_TURN_PROBABILITY = 0.34
_EXTRA_BLOCKER_COUNT_RANGE = (1, 7)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _render_grid(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    grey_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in grey_cells:
        grid[r, c] = GREY
    for r, c in red_cells:
        grid[r, c] = RED
    return grid


def _bottom_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    starts = []
    for c in range(w):
        if grid[h - 1, c] != RED:
            continue
        if c > 0 and grid[h - 1, c - 1] == RED:
            continue
        starts.append((h - 1, c))
    assert starts, "expected at least one bottom-row red path start"
    return starts


def _trace_paths(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    assert red_cells, "expected at least one red path cell"

    visited: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    for sr, sc in _bottom_starts(grid):
        start = (int(sr), int(sc))
        assert start not in visited, "bottom starts must be distinct"
        path = [start]
        seen_local = {start}
        visited.add(start)
        r, c = start

        while True:
            up = r > 0 and grid[r - 1, c] == RED
            right = c + 1 < w and grid[r, c + 1] == RED
            assert not (up and right), "a path cell cannot branch upward and rightward"
            if up:
                nxt = (r - 1, c)
            elif right:
                nxt = (r, c + 1)
            else:
                break
            assert nxt not in seen_local, "path must be acyclic"
            assert nxt not in visited, "different paths must not share red cells"
            seen_local.add(nxt)
            visited.add(nxt)
            path.append(nxt)
            r, c = nxt

        paths.append(path)

    assert visited == red_cells, "every red cell must belong to exactly one traced path"
    return paths


def _required_blockers(
    paths: list[list[tuple[int, int]]],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = shape
    required: set[tuple[int, int]] = set()
    turn_required: set[tuple[int, int]] = set()
    halt_required: set[tuple[int, int]] = set()

    for path in paths:
        for (r, c), (nr, nc) in zip(path, path[1:]):
            if nr == r and nc == c + 1:
                blocker = (r - 1, c)
                assert _in_bounds(shape, blocker[0], blocker[1]), "horizontal move must have an in-bounds blocker above"
                required.add(blocker)
                turn_required.add(blocker)
            else:
                assert nr == r - 1 and nc == c, "paths may only move upward or rightward"

        end_r, end_c = path[-1]
        if end_r > 0:
            above = (end_r - 1, end_c)
            required.add(above)
            halt_required.add(above)
            if end_c + 1 < w:
                side = (end_r, end_c + 1)
                required.add(side)
                halt_required.add(side)

    return required, turn_required, halt_required


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing grey blockers that must have caused the visible red climbing paths.

    Input structure:
    - The background is black.
    - One or more red path traces rise from the bottom row.
    - Some grey obstacle cells are already visible, but others that caused turns
      or halts may have been erased.

    Rule:
    - Trace each red path from its leftmost bottom-row start, following red cells
      upward when present and otherwise rightward.
    - For every rightward step of a path, the cell directly above the step's
      source red cell must be a blocker; repaint that cell grey if it is black.
    - For every path that ends before reaching the top row, the cell above its
      terminal red cell and the cell immediately to its right must both be
      occupied in the original growth process; repaint either of those cells grey
      whenever it is in bounds and currently black.
    - Leave every existing red or grey cell unchanged and keep the grid size.

    Key insight:
    - The red residue is not the result to extend; each jog and each premature
      stop is evidence about an invisible blocker position that must be restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREY}, "expected only black, red, and grey cells"

    paths = _trace_paths(g)
    required, _turn_required, _halt_required = _required_blockers(paths, g.shape)

    out = g.copy()
    for r, c in required:
        if out[r, c] == BLACK:
            out[r, c] = GREY
    return out


def _simulate_growth(
    shape: tuple[int, int],
    start_cols: list[int],
    obstacles: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    red_cells = {(h - 1, int(c)) for c in start_cols}
    for start_c in sorted(start_cols):
        r = h - 1
        c = int(start_c)
        while r > 0:
            if (r - 1, c) not in obstacles and (r - 1, c) not in red_cells:
                r -= 1
                red_cells.add((r, c))
                continue
            if (r - 1, c) in obstacles:
                if c + 1 >= w or (r, c + 1) in obstacles or (r, c + 1) in red_cells:
                    break
                c += 1
                red_cells.add((r, c))
                continue
            break
    return red_cells


def _sample_lane_layout(path_count: int) -> tuple[int, list[int], list[int]] | None:
    widths = [int(_RNG.integers(*_LANE_WIDTH_RANGE)) for _ in range(path_count)]
    gaps = [int(_RNG.integers(*_LANE_GAP_RANGE)) for _ in range(path_count - 1)]
    left_margin = int(_RNG.integers(*_LEFT_MARGIN_RANGE))
    right_margin = int(_RNG.integers(*_RIGHT_MARGIN_RANGE))

    starts: list[int] = []
    right_limits: list[int] = []
    cursor = left_margin
    for i, width in enumerate(widths):
        start_offset = int(_RNG.choice(_START_OFFSET_CHOICES))
        start_offset = min(start_offset, width - 1)
        starts.append(cursor + start_offset)
        right_limits.append(cursor + width - 1)
        cursor += width
        if i < len(gaps):
            cursor += gaps[i]

    width = cursor + right_margin
    if width > 30:
        return None
    return width, starts, right_limits


def _place_turn_run(
    r: int,
    c: int,
    run_len: int,
    occupied_red: set[tuple[int, int]],
    occupied_obstacles: set[tuple[int, int]],
    local_red: set[tuple[int, int]],
    local_obstacles: set[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]] | None:
    current_c = c
    new_red = []
    for _ in range(run_len):
        blocker = (r - 1, current_c)
        step_to = (r, current_c + 1)
        if blocker in occupied_red or blocker in local_red or blocker in occupied_obstacles or blocker in local_obstacles:
            return None
        if step_to in occupied_red or step_to in local_red or step_to in occupied_obstacles or step_to in local_obstacles:
            return None
        local_obstacles.add(blocker)
        local_red.add(step_to)
        new_red.append(step_to)
        current_c += 1
    return current_c, new_red


def _build_single_path(
    shape: tuple[int, int],
    start_col: int,
    right_limit: int,
    occupied_red: set[tuple[int, int]],
    occupied_obstacles: set[tuple[int, int]],
    must_halt: bool,
    force_turn: bool,
    force_long_run: bool,
) -> dict | None:
    h, w = shape
    for _ in range(_PATH_RETRY_LIMIT):
        start = (h - 1, start_col)
        if start in occupied_red or start in occupied_obstacles:
            return None

        local_red = {start}
        local_obstacles: set[tuple[int, int]] = set()
        path = [start]
        run_lengths: list[int] = []
        c = start_col
        r = h - 1
        halted = False
        has_turn = False
        has_long_run = False
        halt_row = int(_RNG.integers(1, h - 1)) if must_halt else None

        while True:
            if r == 0:
                break

            max_run = min(_MAX_RUN_PER_ROW, right_limit - c)
            remaining_up_moves = r if halt_row is None else r - halt_row
            must_finish_here = halt_row is not None and r == halt_row
            must_turn_now = force_turn and not has_turn and remaining_up_moves <= 2 and max_run >= 1
            must_long_now = force_long_run and not has_long_run and remaining_up_moves <= 3 and max_run >= 2

            if must_finish_here:
                min_run = 2 if must_long_now else 1 if must_turn_now else 0
                choices = [run_len for run_len in range(min_run, max_run + 1) if c + run_len + 1 < w]
                if not choices:
                    break
                run_len = int(_RNG.choice(choices))
            else:
                if max_run == 0:
                    run_len = 0
                else:
                    if must_long_now:
                        run_len = int(_RNG.integers(2, max_run + 1))
                    elif must_turn_now:
                        run_len = int(_RNG.integers(1, max_run + 1))
                    elif _RNG.random() < _TURN_PROBABILITY:
                        run_len = int(_RNG.integers(1, max_run + 1))
                    else:
                        run_len = 0

            if run_len:
                placed = _place_turn_run(r, c, run_len, occupied_red, occupied_obstacles, local_red, local_obstacles)
                if placed is None:
                    break
                c, new_red = placed
                path.extend(new_red)
                run_lengths.append(run_len)
                has_turn = True
                has_long_run |= run_len >= 2

            if must_finish_here:
                above = (r - 1, c)
                side = (r, c + 1)
                if above in occupied_red or above in local_red or above in occupied_obstacles or above in local_obstacles:
                    break
                if side in occupied_red or side in local_red or side in occupied_obstacles or side in local_obstacles:
                    break
                local_obstacles.add(above)
                local_obstacles.add(side)
                halted = True
                break

            next_cell = (r - 1, c)
            if next_cell in occupied_red or next_cell in local_red or next_cell in occupied_obstacles or next_cell in local_obstacles:
                break
            local_red.add(next_cell)
            path.append(next_cell)
            r -= 1

        if not path:
            continue
        if must_halt != halted:
            continue
        if force_turn and not has_turn:
            continue
        if force_long_run and not has_long_run:
            continue
        if len(path) < 5:
            continue
        if not must_halt and path[-1][0] != 0:
            continue

        return {
            "path": path,
            "red": local_red,
            "obstacles": local_obstacles,
            "halted": halted,
            "has_turn": has_turn,
            "has_long_run": has_long_run,
            "start_col": start_col,
            "run_lengths": run_lengths,
        }
    return None


def _pick_extra_blockers(
    shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
    required: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    candidates = [
        (r, c)
        for r in range(h - 1)
        for c in range(w)
        if (r, c) not in red_cells and (r, c) not in required
    ]
    if not candidates:
        return set()
    _RNG.shuffle(candidates)
    extra_count = min(len(candidates), int(_RNG.integers(*_EXTRA_BLOCKER_COUNT_RANGE)))
    return set(candidates[:extra_count])


def _choose_erased_required(
    turn_required: set[tuple[int, int]],
    halt_required: set[tuple[int, int]],
    required: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    erased: set[tuple[int, int]] = set()
    if turn_required:
        erased.add(tuple(turn_required.pop()))
    if halt_required:
        erased.add(tuple(halt_required.pop()))
    remaining = list(required - erased)
    _RNG.shuffle(remaining)
    extra_cap = max(0, len(required) - len(erased) - 1)
    extra_n = 0 if extra_cap == 0 else int(_RNG.integers(0, extra_cap + 1))
    erased.update(remaining[:extra_n])
    if not erased:
        erased.add(next(iter(required)))
    if erased == required:
        erased.remove(next(iter(erased)))
    return erased


def _generate_once() -> np.ndarray | None:
    path_count = int(_RNG.choice(_PATH_COUNT_CHOICES, p=_PATH_COUNT_WEIGHTS / _PATH_COUNT_WEIGHTS.sum()))
    h = int(_RNG.integers(*_HEIGHT_RANGE))
    layout = _sample_lane_layout(path_count)
    if layout is None:
        return None
    w, starts, right_limits = layout
    shape = (h, w)

    must_halt = np.zeros(path_count, dtype=bool)
    halt_indices = _RNG.choice(path_count, size=max(1, path_count // 2), replace=False)
    must_halt[halt_indices] = True
    if np.all(must_halt):
        must_halt[int(halt_indices[0])] = False

    capacities = [right_limits[i] - starts[i] for i in range(path_count)]
    long_candidates = [i for i, cap in enumerate(capacities) if cap >= 2]
    if not long_candidates:
        return None
    long_run_idx = int(_RNG.choice(long_candidates))

    turn_candidates = [i for i in range(path_count) if capacities[i] >= 1]
    if not turn_candidates:
        return None
    turn_idx = int(_RNG.choice(turn_candidates))

    occupied_red: set[tuple[int, int]] = set()
    occupied_obstacles: set[tuple[int, int]] = set()
    red_cells: set[tuple[int, int]] = set()
    required_only: set[tuple[int, int]] = set()
    path_infos: list[dict] = []

    for i, (start_col, right_limit) in enumerate(zip(starts, right_limits)):
        info = _build_single_path(
            shape=shape,
            start_col=int(start_col),
            right_limit=int(right_limit),
            occupied_red=occupied_red,
            occupied_obstacles=occupied_obstacles,
            must_halt=bool(must_halt[i]),
            force_turn=(i == turn_idx),
            force_long_run=(i == long_run_idx),
        )
        if info is None:
            return None
        path_infos.append(info)
        occupied_red |= info["red"]
        occupied_obstacles |= info["obstacles"]
        red_cells |= info["red"]
        required_only |= info["obstacles"]

    if not any(info["halted"] for info in path_infos):
        return None
    if not any(not info["halted"] for info in path_infos):
        return None
    if not any(info["has_long_run"] for info in path_infos):
        return None

    required_grid = _render_grid(shape, red_cells, required_only)
    traced_paths = _trace_paths(required_grid)
    required, turn_required, halt_required = _required_blockers(traced_paths, shape)
    if required != required_only:
        return None

    extra_blockers = _pick_extra_blockers(shape, red_cells, required)
    full_obstacles = required | extra_blockers
    simulated_red = _simulate_growth(shape, starts, full_obstacles)
    if simulated_red != red_cells:
        return None

    if len(required) < 2 or not turn_required or not halt_required:
        return None

    erased = _choose_erased_required(set(turn_required), set(halt_required), set(required))
    if not erased:
        return None
    visible_obstacles = full_obstacles - erased
    inp = _render_grid(shape, red_cells, visible_obstacles)
    expected = _render_grid(shape, red_cells, full_obstacles)
    actual = main(inp)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(main(expected), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _generate_once()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid puzzle-4 input")
