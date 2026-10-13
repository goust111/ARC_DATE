import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

HEIGHT_RANGE = (7, 12)
WIDTH_RANGE = (13, 18)
RUN_LEN_RANGE = (2, 4)
MOVING_ROW_RANGE = (3, 5)
MAX_ATTEMPTS = 500


def _moving_run_bounds(row: np.ndarray, bg: int = BG):
    """Return (start_col, run_len, color) for the right-edge run, or None."""
    width = len(row)
    if width == 0 or int(row[-1]) == int(bg):
        return None

    color = int(row[-1])
    c = width - 1
    while c >= 0 and int(row[c]) == color:
        c -= 1
    start = c + 1
    return int(start), int(width - start), color


def _transform_row(row: np.ndarray, bg: int = BG) -> np.ndarray:
    info = _moving_run_bounds(row, bg)
    if info is None:
        return row.copy()

    start, run_len, current = info
    out = row.copy()
    out[start:] = bg

    stop_col = None
    consumed_cols: list[int] = []
    for c in range(start - 1, -1, -1):
        color = int(row[c])
        if color == int(bg):
            continue
        if color == current:
            stop_col = c
            break
        consumed_cols.append(c)
        current = color

    if consumed_cols:
        out[np.array(consumed_cols, dtype=int)] = bg

    target_start = 0 if stop_col is None else stop_col + 1
    out[target_start : target_start + run_len] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge colored run leftward, recoloring it at mismatched obstacles and stopping only at the first obstacle matching its current color.

    Input structure:
    - The grid has a black background.
    - Some rows end with one contiguous colored horizontal run at the right edge; these runs are the movable strips.
    - The cells to their left may contain isolated single-pixel colored obstacles, while other rows may contain only obstacles or be empty.

    Rule:
    - Process each row independently.
    - A movable strip slides left from the right edge and encounters obstacles from right to left.
    - When the next obstacle has a different color, erase that obstacle to black, recolor the whole strip to that obstacle color, and continue sliding.
    - When the next obstacle has the same color as the strip's current color, keep that obstacle and stop with the strip packed immediately to its right; if no obstacle ever matches, the strip finishes against the left edge.

    Key insight:
    - Obstacles are dynamic recolor checkpoints rather than fixed blockers, so both the stop position and the final strip color depend on the full right-to-left color chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r in range(g.shape[0]):
        out[r] = _transform_row(g[r], BG)
    return out


def _random_distinct_color(exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    return int(random.choice(choices))


def _make_chain(initial: int, consumed_count: int, stop: bool) -> list[int]:
    current = int(initial)
    chain: list[int] = []
    for _ in range(consumed_count):
        nxt = _random_distinct_color({current})
        chain.append(nxt)
        current = nxt
    if stop:
        chain.append(current if chain else int(initial))
    return chain


def _spaced_positions(last_allowed: int, count: int, nearest_gap: int) -> list[int] | None:
    if count == 0:
        return []
    pos = int(last_allowed - nearest_gap)
    if pos < 0:
        return None
    positions = [pos]
    for _ in range(count - 1):
        pos -= random.randint(2, 4)
        if pos < 0:
            return None
        positions.append(pos)
    return positions


def _sample_extra_colors(count: int) -> list[int]:
    return [int(random.choice(PALETTE)) for _ in range(count)]


def _build_row_plan(width: int, branch: str, force_left_extra: bool = False) -> dict | None:
    for _ in range(120):
        run_len = random.randint(*RUN_LEN_RANGE)
        initial = int(random.choice(PALETTE))

        if branch == "immediate_stop":
            encountered = [initial]
            extra_count = random.randint(1 if force_left_extra else 0, 3)
            nearest_gap = random.randint(0, 3)
        elif branch == "multi_stop":
            encountered = _make_chain(initial, consumed_count=random.randint(2, 3), stop=True)
            extra_count = random.randint(1 if force_left_extra else 0, 2)
            nearest_gap = random.randint(0, 2)
        elif branch == "short_stop":
            encountered = _make_chain(initial, consumed_count=1, stop=True)
            extra_count = random.randint(1 if force_left_extra else 0, 2)
            nearest_gap = random.randint(0, 3)
        elif branch == "to_edge":
            encountered = _make_chain(initial, consumed_count=random.randint(1, 4), stop=False)
            extra_count = 0
            nearest_gap = random.randint(0, 2)
        elif branch == "no_obstacles":
            encountered = []
            extra_count = 0
            nearest_gap = 0
        else:
            raise ValueError(f"unknown branch: {branch}")

        total_obstacles = len(encountered) + extra_count
        last_allowed = width - run_len - 1
        positions = _spaced_positions(last_allowed, total_obstacles, nearest_gap)
        if positions is None:
            continue

        colors = encountered + _sample_extra_colors(extra_count)
        row = np.zeros(width, dtype=int)
        for pos, color in zip(positions, colors):
            row[int(pos)] = int(color)
        row[width - run_len :] = initial

        stop_col = None
        current = initial
        consumed = 0
        for idx, color in enumerate(encountered):
            if int(color) == current:
                stop_col = int(positions[idx])
                break
            current = int(color)
            consumed += 1

        output = _transform_row(row, BG)
        target_start = 0 if stop_col is None else stop_col + 1
        slide_distance = (width - run_len) - target_start
        return {
            "branch": branch,
            "row": row,
            "output": output,
            "run_len": run_len,
            "initial": initial,
            "positions": positions,
            "encountered_count": len(encountered),
            "extra_count": extra_count,
            "stop_col": stop_col,
            "consumed_count": consumed,
            "slide_distance": int(slide_distance),
            "final_color": int(current),
        }
    return None


def _sample_spaced_columns(limit_exclusive: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(limit_exclusive))
    random.shuffle(candidates)
    chosen: list[int] = []
    for col in candidates:
        if all(abs(col - other) >= 2 for other in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return sorted(chosen, reverse=True)
    return None


def _build_static_row(width: int, force_obstacles: bool) -> np.ndarray | None:
    for _ in range(60):
        count = random.randint(1 if force_obstacles else 0, 4)
        cols = _sample_spaced_columns(width - 1, count)
        if cols is None:
            continue
        row = np.zeros(width, dtype=int)
        for col in cols:
            row[int(col)] = int(random.choice(PALETTE))
        if row[-1] != BG:
            continue
        return row
    return None


def _shortcut_static_blocker(row: np.ndarray) -> np.ndarray:
    info = _moving_run_bounds(row, BG)
    if info is None:
        return row.copy()
    start, run_len, color = info
    out = row.copy()
    out[start:] = BG
    blockers = np.flatnonzero(row[:start] != BG)
    target_start = 0 if blockers.size == 0 else int(blockers[-1]) + 1
    out[target_start : target_start + run_len] = color
    return out


def _shortcut_consume_all(row: np.ndarray) -> np.ndarray:
    info = _moving_run_bounds(row, BG)
    if info is None:
        return row.copy()
    start, run_len, color = info
    out = row.copy()
    out[start:] = BG
    obstacles = np.flatnonzero(row[:start] != BG)
    if obstacles.size:
        color = int(row[int(obstacles[0])])
        out[obstacles] = BG
    out[:run_len] = color
    return out


def _shortcut_stop_on_initial(row: np.ndarray) -> np.ndarray:
    info = _moving_run_bounds(row, BG)
    if info is None:
        return row.copy()
    start, run_len, initial = info
    out = row.copy()
    out[start:] = BG

    current = initial
    consumed_cols: list[int] = []
    stop_col = None
    for c in range(start - 1, -1, -1):
        color = int(row[c])
        if color == BG:
            continue
        if color == initial:
            stop_col = c
            break
        consumed_cols.append(c)
        current = color

    if consumed_cols:
        out[np.array(consumed_cols, dtype=int)] = BG
    target_start = 0 if stop_col is None else stop_col + 1
    out[target_start : target_start + run_len] = current
    return out


def _apply_rowwise(grid: np.ndarray, row_fn) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r in range(out.shape[0]):
        out[r] = row_fn(out[r])
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, plans: list[dict], moving_rows: list[int]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    nonmoving_rows = [r for r in range(inp.shape[0]) if r not in set(moving_rows)]
    if any(int(inp[r, -1]) != BG for r in nonmoving_rows):
        return False
    if any(int(inp[r, -1]) == BG for r in moving_rows):
        return False

    if not any(plan["stop_col"] is None for plan in plans):
        return False
    if not any(plan["stop_col"] is not None for plan in plans):
        return False
    if not any(plan["consumed_count"] >= 2 and plan["stop_col"] is not None for plan in plans):
        return False
    if not any(plan["extra_count"] > 0 and plan["stop_col"] is not None for plan in plans):
        return False
    if not any(plan["final_color"] != plan["initial"] for plan in plans):
        return False
    if not any(plan["slide_distance"] > 0 for plan in plans):
        return False

    static_guess = _apply_rowwise(inp, _shortcut_static_blocker)
    consume_all_guess = _apply_rowwise(inp, _shortcut_consume_all)
    initial_match_guess = _apply_rowwise(inp, _shortcut_stop_on_initial)
    if np.array_equal(static_guess, expected):
        return False
    if np.array_equal(consume_all_guess, expected):
        return False
    if np.array_equal(initial_match_guess, expected):
        return False

    moving_outputs = {tuple(expected[r].tolist()) for r in moving_rows}
    if len(moving_outputs) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    n_moving = random.randint(*MOVING_ROW_RANGE)
    height = random.randint(max(HEIGHT_RANGE[0], n_moving + 2), HEIGHT_RANGE[1])
    moving_rows = sorted(random.sample(range(height), n_moving))

    required = ["multi_stop", "to_edge", random.choice(["immediate_stop", "no_obstacles"])]
    optional_pool = ["short_stop", "immediate_stop", "to_edge", "no_obstacles"]
    branches = required + [random.choice(optional_pool) for _ in range(n_moving - len(required))]
    random.shuffle(branches)

    plans: list[dict] = []
    forced_extra_used = False
    for branch in branches:
        force_left_extra = False
        if branch in {"multi_stop", "short_stop", "immediate_stop"} and not forced_extra_used:
            force_left_extra = True
        plan = _build_row_plan(width, branch, force_left_extra=force_left_extra)
        if plan is None:
            return None
        forced_extra_used = forced_extra_used or bool(plan["extra_count"] > 0 and plan["stop_col"] is not None)
        plans.append(plan)

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    for r, plan in zip(moving_rows, plans):
        inp[r] = plan["row"]
        expected[r] = plan["output"]

    static_rows = [r for r in range(height) if r not in set(moving_rows)]
    if static_rows:
        obstacle_row = random.choice(static_rows)
        row = _build_static_row(width, force_obstacles=True)
        if row is None:
            return None
        inp[obstacle_row] = row
        expected[obstacle_row] = row
        for r in static_rows:
            if r == obstacle_row:
                continue
            row = _build_static_row(width, force_obstacles=False)
            if row is None:
                return None
            inp[r] = row
            expected[r] = row

    if not _validate_case(inp, expected, plans, moving_rows):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
