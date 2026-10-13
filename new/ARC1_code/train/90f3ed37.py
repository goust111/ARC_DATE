import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


State = tuple[int, ...]
Run = tuple[State, int, int]


def _row_groups(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    row_has_foreground = np.any(g != bg, axis=1)
    groups: list[tuple[int, int]] = []
    r = 0
    while r < g.shape[0]:
        if not row_has_foreground[r]:
            r += 1
            continue
        r0 = r
        while r + 1 < g.shape[0] and row_has_foreground[r + 1]:
            r += 1
        groups.append((r0, r))
        r += 1
    return groups


def _column_states(g: np.ndarray, bg: int, r0: int, r1: int) -> list[State]:
    band = g[r0 : r1 + 1, :]
    states: list[State] = []
    for c in range(g.shape[1]):
        states.append(tuple(int(r) for r in np.flatnonzero(band[:, c] != bg)))
    return states


def _compress(states: list[State]) -> list[Run]:
    assert states and all(states)
    runs: list[Run] = []
    cur = states[0]
    start = 0
    for i, state in enumerate(states[1:], start=1):
        if state != cur:
            runs.append((cur, start, i - 1))
            cur = state
            start = i
    runs.append((cur, start, len(states) - 1))
    return runs


def _suffix_states(top_runs: list[Run], prefix_run_count: int, needed: int) -> list[State]:
    assert needed > 0
    assert 0 < prefix_run_count <= len(top_runs)
    if prefix_run_count == len(top_runs):
        return [top_runs[-1][0]] * needed

    suffix: list[State] = []
    for state, c0, c1 in top_runs[prefix_run_count:]:
        suffix.extend([state] * (c1 - c0 + 1))
    assert len(suffix) >= needed
    return suffix[:needed]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each lower foreground prefix by continuing the top complete column pattern in blue.

    Input structure
    A black-background grid has one foreground color arranged in separated row-groups. The top row-group spans every column and is the complete reference pattern; each lower row-group starts at the left edge and contains a shorter prefix of the same column-state pattern.

    Rule
    Keep all existing foreground pixels. For every lower row-group, compare its compressed sequence of nonempty row-offset states by column to the top pattern, then paint the missing suffix columns with blue using the remaining top states; if the lower group already reaches the top pattern's final state, continue that final state to the right edge.

    Key insight
    The exact run lengths before a state change can vary, but the compressed order of column states is shared: horizontal bars, zigzags, and multi-row stripes are all completed by extending that state sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg != Color.BLUE

    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1
    assert Color.BLUE not in foreground_colors

    groups = _row_groups(g, bg)
    assert len(groups) >= 2

    spanning_groups = [
        i
        for i, (r0, r1) in enumerate(groups)
        if all(_column_states(g, bg, r0, r1))
    ]
    assert spanning_groups == [0]

    top_r0, top_r1 = groups[0]
    top_runs = _compress(_column_states(g, bg, top_r0, top_r1))
    top_run_states = [state for state, _, _ in top_runs]

    out = g.copy()
    _, W = g.shape
    for r0, r1 in groups[1:]:
        states = _column_states(g, bg, r0, r1)
        filled_cols = [c for c, state in enumerate(states) if state]
        assert filled_cols
        last_col = filled_cols[-1]
        assert last_col < W - 1
        assert filled_cols == list(range(last_col + 1))

        lower_runs = _compress(states[: last_col + 1])
        lower_run_states = [state for state, _, _ in lower_runs]
        assert lower_run_states == top_run_states[: len(lower_run_states)]

        suffix = _suffix_states(top_runs, len(lower_runs), W - last_col - 1)
        for c, state in zip(range(last_col + 1, W), suffix):
            for rr in state:
                r = r0 + rr
                assert 0 <= r < g.shape[0]
                assert out[r, c] == bg
                out[r, c] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# Grid, colors, and acceptance bounds.
GRID_HEIGHT, GRID_WIDTH = 15, 10
BACKGROUND_COLOR, COMPLETION_COLOR = 0, Color.BLUE
FOREGROUND_PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
OBSERVED_FOREGROUND_COLOR, OBSERVED_FOREGROUND_PROB = 8, 0.6
MAX_INPUT_FOREGROUND_CELLS = (GRID_HEIGHT * GRID_WIDTH) // 2

# Top pattern shape bounds.
MIN_RUN_WIDTH = 1
SINGLE_RUN_MIN_HEIGHT, SINGLE_RUN_MAX_HEIGHT = 1, 3
MULTI_RUN_HEIGHT_OPTIONS = (2, 3)

# Group placement bounds.
MIN_LOWER_GROUPS, MAX_LOWER_GROUPS = 1, 3
MIN_TOP_GAP, MIN_GROUP_GAP, MIN_BOTTOM_GAP = 1, 1, 0

# Lower prefix and completion sampling.
MIN_SUFFIX_COLS = 1
OBSERVED_MIN_SUFFIX_COLS, OBSERVED_MAX_SUFFIX_COLS = 5, 8
OBSERVED_PREFIX_PROB, OBSERVED_SUFFIX_PROB = 0.6, 0.75

# Retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(MIN_RUN_WIDTH, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _gen_states(height: int, run_count: int) -> list[State]:
    all_states = [
        tuple(r for r in range(height) if mask & (1 << r))
        for mask in range(1, 1 << height)
    ]
    if run_count == 1:
        return [tuple(range(height))]

    first_options = [tuple(range(k)) for k in range(1, height + 1)]
    states = [random.choice(first_options)]
    covered = set(states[0])
    for idx in range(1, run_count):
        remaining_slots = run_count - idx
        missing = set(range(height)) - covered
        required: set[int] = set()
        if missing and idx == run_count - 1:
            required = set(missing)
        elif missing and len(missing) >= remaining_slots:
            required = {min(missing)}
        choices = [
            state
            for state in all_states
            if state != states[-1] and required <= set(state)
        ]
        state = random.choice(choices)
        states.append(state)
        covered.update(state)
    return states


def _gen_valid_prefix_counts(top_states: list[State], top_widths: list[int]) -> list[int]:
    valid: list[int] = []
    seen: set[int] = set()
    max_prefix_runs = min(len(top_states), GRID_WIDTH - MIN_SUFFIX_COLS)
    for prefix_runs in range(1, max_prefix_runs + 1):
        seen.update(top_states[prefix_runs - 1])
        max_seen = max(seen)
        rows_are_contiguous = seen == set(range(max_seen + 1))
        if prefix_runs < len(top_states):
            min_prefix_cols = sum(top_widths[:prefix_runs])
        else:
            min_prefix_cols = prefix_runs
        if rows_are_contiguous and min_prefix_cols <= GRID_WIDTH - MIN_SUFFIX_COLS:
            valid.append(prefix_runs)
    return valid


def _gen_min_prefix_cols(top_widths: list[int], prefix_runs: int) -> int:
    if prefix_runs < len(top_widths):
        return sum(top_widths[:prefix_runs])
    return prefix_runs


def _gen_suffix_states(
    top_states: list[State], top_widths: list[int], prefix_run_count: int, needed: int
) -> list[State]:
    if prefix_run_count == len(top_states):
        return [top_states[-1]] * needed

    suffix: list[State] = []
    for state, width in zip(top_states[prefix_run_count:], top_widths[prefix_run_count:]):
        suffix.extend([state] * width)
    assert len(suffix) >= needed
    return suffix[:needed]


def _gen_group_starts(height: int, lower_count: int) -> list[int]:
    group_count = lower_count + 1
    gaps = [MIN_TOP_GAP] + [MIN_GROUP_GAP] * lower_count + [MIN_BOTTOM_GAP]
    slack = GRID_HEIGHT - (group_count * height + sum(gaps))
    assert slack >= 0
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    starts = [gaps[0]]
    r = gaps[0] + height
    for i in range(lower_count):
        r += gaps[i + 1]
        starts.append(r)
        r += height
    return starts


def _gen_paint_run_columns(
    grid: np.ndarray, row0: int, col0: int, widths: list[int], states: list[State], color: int
) -> int:
    c = col0
    for width, state in zip(widths, states):
        rows = row0 + np.asarray(state)
        cols = np.arange(c, c + width)
        grid[np.ix_(rows, cols)] = color
        c += width
    return c


def _gen_pattern_plan() -> tuple[list[State], list[int], list[int], list[int]] | None:
    run_count = random.randint(1, GRID_WIDTH)
    if run_count == 1:
        top_height = random.randint(SINGLE_RUN_MIN_HEIGHT, SINGLE_RUN_MAX_HEIGHT)
    else:
        top_height = random.choice(MULTI_RUN_HEIGHT_OPTIONS)
    max_fit = (GRID_HEIGHT - MIN_TOP_GAP - top_height) // (top_height + MIN_GROUP_GAP)
    lower_count = random.randint(MIN_LOWER_GROUPS, min(MAX_LOWER_GROUPS, max_fit))
    top_states = _gen_states(top_height, run_count)
    top_widths = _gen_partition(GRID_WIDTH, run_count)
    valid_prefixes = _gen_valid_prefix_counts(top_states, top_widths)
    if not valid_prefixes:
        return None
    starts = _gen_group_starts(top_height, lower_count)
    return top_states, top_widths, valid_prefixes, starts


def _gen_prefix_run_count(valid_prefixes: list[int], top_widths: list[int]) -> int:
    observed_prefixes = [
        prefix
        for prefix in valid_prefixes
        if _gen_min_prefix_cols(top_widths, prefix) <= GRID_WIDTH - OBSERVED_MIN_SUFFIX_COLS
    ]
    if observed_prefixes and random.random() < OBSERVED_PREFIX_PROB:
        return random.choice(observed_prefixes)
    return random.choice(valid_prefixes)


def _gen_prefix_cols(min_prefix_cols: int) -> int:
    observed_like = list(
        range(
            max(min_prefix_cols, GRID_WIDTH - OBSERVED_MAX_SUFFIX_COLS),
            min(GRID_WIDTH - OBSERVED_MIN_SUFFIX_COLS, GRID_WIDTH - MIN_SUFFIX_COLS) + 1,
        )
    )
    if observed_like and random.random() < OBSERVED_SUFFIX_PROB:
        return random.choice(observed_like)
    return random.randint(min_prefix_cols, GRID_WIDTH - MIN_SUFFIX_COLS)


def _gen_paint_lower_group(
    out: np.ndarray, row0: int, top_states: list[State], top_widths: list[int],
    valid_prefixes: list[int], fg: int,
) -> bool:
    prefix_runs = _gen_prefix_run_count(valid_prefixes, top_widths)
    min_prefix_cols = _gen_min_prefix_cols(top_widths, prefix_runs)
    min_prefix_cols = max(min_prefix_cols, prefix_runs)
    if min_prefix_cols > GRID_WIDTH - MIN_SUFFIX_COLS:
        return False

    prefix_cols = _gen_prefix_cols(min_prefix_cols)
    prefix_widths = _gen_partition(prefix_cols, prefix_runs)
    _gen_paint_run_columns(out, row0, 0, prefix_widths, top_states[:prefix_runs], fg)

    suffix = _gen_suffix_states(top_states, top_widths, prefix_runs, GRID_WIDTH - prefix_cols)
    for c, state in zip(range(prefix_cols, GRID_WIDTH), suffix):
        rows = row0 + np.asarray(state)
        out[rows, c] = COMPLETION_COLOR
    return True


def _gen_render_output(
    fg: int, top_states: list[State], top_widths: list[int],
    valid_prefixes: list[int], starts: list[int],
) -> np.ndarray | None:
    out = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    _gen_paint_run_columns(out, starts[0], 0, top_widths, top_states, fg)
    for row0 in starts[1:]:
        if not _gen_paint_lower_group(out, row0, top_states, top_widths, valid_prefixes, fg):
            return None
    return out


def _gen_attempt() -> np.ndarray | None:
    if random.random() < OBSERVED_FOREGROUND_PROB:
        fg = OBSERVED_FOREGROUND_COLOR
    else:
        fg = random.choice(FOREGROUND_PALETTE)
    plan = _gen_pattern_plan()
    if plan is None:
        return None
    out = _gen_render_output(fg, *plan)
    if out is None:
        return None
    inp = out.copy()
    inp[inp == COMPLETION_COLOR] = BACKGROUND_COLOR
    if np.count_nonzero(inp != BACKGROUND_COLOR) >= MAX_INPUT_FOREGROUND_CELLS:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
