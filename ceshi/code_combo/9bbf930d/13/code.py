# Combo variant of 9bbf930d (index 13)
# combo_spec: D14/1+D2/2
# Name: 角落调色板优先级+双向歧义取较长路径

from __future__ import annotations

import random

import numpy as np


BG = 7
PARTICLE = 6
_DELTAS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


# ---------------------------------------------------------------------------
# Design analysis for combo D14/1 + D2/2
#
# a) Concrete behavior and trace
# - The base canvas stays close to the source task: even rows are colored bar
#   rows, odd rows are particle channels, col 0 is all magenta particles, and
#   col 1 is bg.
# - Two new visual cues are fused into one task:
#   1. A top-right / bottom-right corner palette lists corridor colors in
#      priority order.
#   2. Some blockers open both perpendicular turns at once, so a particle sees a
#      genuine fork instead of the source task's unique turn.
# - There are two fork families in the generator:
#   1. "priority fork": upper and lower branches have different colors. One
#      branch is visibly longer, but the shorter branch has the higher-ranked
#      palette color, so the particle takes the shorter branch.
#   2. "length fork": both branches share the same color, so palette rank ties
#      and the particle falls back to the longer branch.
# - Small trace:
#   source task: move right; if blocked, there must be at most one valid 90°
#   turn, otherwise the solver asserts.
#   combo task: move right; if blocked and both turns are legal, first compare
#   the corridor colors against the corner palette. If one color ranks earlier,
#   choose it even if it leads to a shorter continuation. Only if ranks tie do
#   we compare remaining path lengths and choose the longer continuation.
#
# b) Complexity calibration
# - The rule is still small enough for ARC: one short palette and two fork
#   behaviors. The generator always includes at least one priority-sensitive fork
#   and one length-only fork, so the fusion is not hidden in a rare edge case.
# - The palette length is fixed at 3 colors. That creates repeated evidence
#   without exploding the number of possible branch categories.
# - Ambiguity is local and sparse: turn columns are spaced apart, and active
#   channels are non-adjacent, so each fork is readable.
#
# c) Main() assumption audit
# - Original `_trace()` assumed `len(options) <= 1`. That is invalid now.
# - Original main-loop entrance detection can stay, because the start of each
#   real channel is still an aligned same-color pair beginning at col 2 or 3.
# - New hidden assumptions introduced here:
#   1. The top row ends with a contiguous non-BG suffix encoding the palette.
#   2. Reachable ambiguous branches either have distinct palette ranks or, if
#      ranks tie, a deterministic longer branch exists.
#   3. Recursive path comparison terminates. The generator guarantees acyclic,
#      edge-reaching or dead-ending shafts.
# - The generator explicitly validates those assumptions by constructing only
#   straight rows, edge-reaching L turns, and dead-end-vs-edge forks, then
#   measuring reachable fork behavior afterward.
#
# d) Placement legality
# - Special channels are chosen non-adjacent so a short fork branch always points
#   into a fake neighboring channel, which blocks accidental secondary turns.
# - Turn columns reserve a 4-column window, preventing overlapping blockers or
#   contradictory carved shafts.
# - A right-side legend strip is reserved; no corridors are built into it.
#
# e) Visual observability
# - Priority is encoded only by visible color order in the corner palette.
# - Length comparison is also visible: one branch clearly reaches the edge while
#   the other dies at the first intact bar.
# - Because generated samples contain both a priority-overrides-shorter case and
#   a rank-tie-uses-longer case, a human can infer the combined rule from a few
#   examples instead of guessing invisible color ids.
#
# f) Visual cleanliness
# - The palette sits in an isolated right strip, duplicated on the top and bottom
#   borders.
# - Forks are sparse and column-separated. No dense overpainting or stacked
#   legends are used.
#
# g) Color inferability
# - All branch colors come from the same small extras palette already used by the
#   bars. No arithmetic on raw color ids is used.
# - The only color-specific logic is ordinal lookup in the corner palette. When
#   ordinal lookup does not distinguish the branches, geometry alone decides via
#   longer path.
# ---------------------------------------------------------------------------


def _read_priority(grid: np.ndarray) -> list[int]:
    """Read the contiguous non-BG suffix on the top row as the priority palette."""
    top = np.array(grid[0], dtype=int)
    colors: list[int] = []
    c = len(top) - 1
    while c >= 0 and int(top[c]) not in {BG, PARTICLE}:
        colors.append(int(top[c]))
        c -= 1
    colors.reverse()
    assert len(colors) >= 2, "missing corner priority palette"
    return colors


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Return the wall color for a legal corridor cell, else None."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in {PARTICLE, BG}:
        return int(a)
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    return _corridor_color(grid, r, c, direction) is not None


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid real-corridor entrance column for row r."""
    H, W = grid.shape
    if not (0 < r < H - 1):
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            bar_above = next((k for k in range(W) if int(grid[r - 1, k]) == above), W)
            bar_below = next((k for k in range(W) if int(grid[r + 1, k]) == below), W)
            if bar_above == bar_below and bar_above <= 3:
                return c
    return None


def _next_turn_options(
    grid: np.ndarray, r: int, c: int, direction: str
) -> list[tuple[int, int, str, int]]:
    """Return valid perpendicular turn states and their corridor colors."""
    H, W = grid.shape
    options: list[tuple[int, int, str, int]] = []
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        color = _corridor_color(grid, tr, tc, turn_dir)
        if color is not None:
            options.append((tr, tc, turn_dir, int(color)))
    return options


def _best_trace_state(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    priority_rank: dict[int, int],
    memo: dict[tuple[int, int, str, tuple[tuple[int, int], ...]], tuple[int, int, int]],
    visiting: set[tuple[int, int, str, tuple[tuple[int, int], ...]]],
) -> tuple[int, int, int]:
    """Return (remaining_steps, final_r, final_c) from a movement state."""
    rank_key = tuple(sorted((int(color), int(rank)) for color, rank in priority_rank.items()))
    key = (r, c, direction, rank_key)
    if key in memo:
        return memo[key]
    if key in visiting:
        return 0, r, c
    visiting.add(key)

    H, W = grid.shape
    dr, dc = _DELTAS[direction]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= H or nc < 0 or nc >= W:
        result = (0, r, c)
    elif int(grid[nr, nc]) in {PARTICLE, BG}:
        extra, end_r, end_c = _best_trace_state(
            grid, nr, nc, direction, priority_rank, memo, visiting
        )
        result = (1 + extra, end_r, end_c)
    else:
        options = _next_turn_options(grid, r, c, direction)
        if not options:
            result = (0, r, c)
        else:
            default_rank = len(priority_rank) + 10
            best_key: tuple[int, int, int] | None = None
            best_result: tuple[int, int, int] | None = None
            for idx, (tr, tc, turn_dir, color) in enumerate(options):
                extra, end_r, end_c = _best_trace_state(
                    grid, tr, tc, turn_dir, priority_rank, memo, visiting
                )
                steps = 1 + extra
                cand_key = (priority_rank.get(int(color), default_rank), -steps, idx)
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_result = (steps, end_r, end_c)
            assert best_result is not None
            result = best_result

    visiting.remove(key)
    memo[key] = result
    return result


def _trace_with_priority(
    grid: np.ndarray, r: int, c: int, priority_rank: dict[int, int]
) -> tuple[int, int]:
    memo: dict[tuple[int, int, str, tuple[tuple[int, int], ...]], tuple[int, int, int]] = {}
    _, end_r, end_c = _best_trace_state(grid, r, c, "right", priority_rank, memo, set())
    return end_r, end_c


def _main_with_priority(grid: np.ndarray, priority_rank: dict[int, int]) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace_with_priority(g, r, 0, priority_rank)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through striped corridors with palette-first ambiguous turns.

    Input structure:
    - bg=7, particle=6, col 0 full of particles, col 1 bg.
    - Even rows are colored bars, odd rows are travel channels.
    - The top-right and bottom-right corners contain the same short color palette.

    Rule:
    - A particle moves straight through bg cells.
    - When blocked by a colored cell, it checks the two perpendicular turn cells.
    - If only one turn is valid, take it.
    - If both turns are valid, compare the two corridor colors against the corner
      palette: earlier color wins. If palette rank ties, take the branch whose
      continuation is longer. Stop at the grid edge or at a full blockage.

    Key insight:
    - Geometry alone is no longer enough at a fork. The solver must fuse two
      signals: visible corner-palette priority first, then visible path length as
      the fallback tie-break.
    """
    g = np.array(grid, dtype=int)
    priority = _read_priority(g)
    priority_rank = {int(color): i for i, color in enumerate(priority)}
    return _main_with_priority(g, priority_rank)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_LEGEND_LEN = 3
_MIN_MOVED = 3
_MAX_MOVED = 5
_MAX_RETRIES = 700


def _pick_active_channels(n_channels: int, count: int) -> list[int]:
    order = list(range(n_channels))
    random.shuffle(order)
    chosen: list[int] = []
    for idx in order:
        if all(abs(idx - other) > 1 for other in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _allocate_turn_col(bar_start: int, path_right: int, used_cols: set[int]) -> int | None:
    candidates = []
    for c_v in range(bar_start + 5, path_right - 1):
        window = {c_v - 2, c_v - 1, c_v, c_v + 1}
        if window & used_cols:
            continue
        candidates.append(c_v)
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_cols.update({c_v - 2, c_v - 1, c_v, c_v + 1})
    return c_v


def _build_row_colors(n_channels: int, active_set: set[int], extras: list[int]) -> list[int]:
    row_colors = [random.choice(extras)]
    for i in range(n_channels):
        prev = row_colors[-1]
        if i in active_set:
            row_colors.append(prev)
        else:
            row_colors.append(random.choice([c for c in extras if c != prev]))
    return row_colors


def _paint_base_bars(
    g: np.ndarray, bar_rows: list[int], row_colors: list[int], bar_start: int, sep: int
) -> None:
    for idx, br in enumerate(bar_rows):
        g[br, bar_start:sep] = int(row_colors[idx])


def _apply_real_entrance(
    g: np.ndarray, top: int, bot: int, bar_start: int, stop: int, color: int
) -> None:
    g[top, bar_start:stop] = int(color)
    g[bot, bar_start:stop] = int(color)


def _build_edge_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    color: int,
) -> None:
    """Build a one-sided L-turn that reaches the grid edge."""
    g[row, c_v] = int(color)
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = int(color)
    g[turn_bar, c_v] = int(color)
    g[opp_bar, c_v - 1] = int(color)

    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _build_length_fork(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    color: int,
    long_dir: str,
) -> None:
    """Build an ambiguous fork where both branches have the same color."""
    g[row, c_v] = int(color)
    for turn_bar in (top, bot):
        g[turn_bar, c_v - 1] = BG
        g[turn_bar, c_v - 2] = int(color)
        g[turn_bar, c_v] = int(color)
    if long_dir == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _build_priority_fork(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    bar_start: int,
    sep: int,
    entry_color: int,
    c_v: int,
    switch_c: int,
    up_color: int,
    down_color: int,
    long_dir: str,
) -> None:
    """Build an ambiguous fork where palette rank can override length."""
    _apply_real_entrance(g, top, bot, bar_start, switch_c, entry_color)
    g[top, switch_c:sep] = int(up_color)
    g[bot, switch_c:sep] = int(down_color)
    g[row, c_v] = int(entry_color)
    g[top, c_v - 1] = BG
    g[bot, c_v - 1] = BG
    g[top, c_v - 2] = int(up_color)
    g[top, c_v] = int(up_color)
    g[bot, c_v - 2] = int(down_color)
    g[bot, c_v] = int(down_color)

    if long_dir == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _add_fake_decor(
    g: np.ndarray,
    inactive: list[int],
    bar_rows: list[int],
    row_colors: list[int],
    bar_start: int,
    sep: int,
    extras: list[int],
) -> None:
    fake_rows = [bar_rows[idx] + 1 for idx in inactive]
    for idx in inactive:
        r = bar_rows[idx] + 1
        if random.random() < 0.4:
            candidates = [c for c in range(bar_start + 1, sep) if int(g[r, c]) == BG]
            if candidates:
                c = random.choice(candidates)
                g[r, c] = random.choice([row_colors[idx], row_colors[idx + 1]])

    if not fake_rows:
        return

    used = set(map(int, g.flatten()))
    missing = [color for color in extras if color not in used]
    for color in missing:
        r = random.choice(fake_rows)
        candidates = [c for c in range(bar_start + 1, sep) if int(g[r, c]) == BG]
        if candidates:
            g[r, random.choice(candidates)] = int(color)


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original trace rule: ambiguous turns are invalid."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = _next_turn_options(grid, r, c, direction)
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction, _ = options[0]
    return r, c


def _main_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace_original(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def _collect_ambiguous_states(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    seen: set[tuple[int, int, str]],
    states: set[tuple[int, int, str]],
) -> None:
    key = (r, c, direction)
    if key in seen:
        return
    seen.add(key)

    H, W = grid.shape
    dr, dc = _DELTAS[direction]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= H or nc < 0 or nc >= W:
        return
    if int(grid[nr, nc]) in {PARTICLE, BG}:
        _collect_ambiguous_states(grid, nr, nc, direction, seen, states)
        return

    options = _next_turn_options(grid, r, c, direction)
    if len(options) == 2:
        states.add(key)
    for tr, tc, turn_dir, _ in options:
        _collect_ambiguous_states(grid, tr, tc, turn_dir, seen, states)


def _option_summaries(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    priority_rank: dict[int, int],
    memo: dict[tuple[int, int, str, tuple[tuple[int, int], ...]], tuple[int, int, int]],
) -> list[dict[str, int]]:
    options = _next_turn_options(grid, r, c, direction)
    default_rank = len(priority_rank) + 10
    summaries: list[dict[str, int]] = []
    for idx, (tr, tc, turn_dir, color) in enumerate(options):
        extra, end_r, end_c = _best_trace_state(
            grid, tr, tc, turn_dir, priority_rank, memo, set()
        )
        steps = 1 + extra
        summaries.append(
            {
                "idx": idx,
                "rank": int(priority_rank.get(int(color), default_rank)),
                "steps": int(steps),
                "end_r": int(end_r),
                "end_c": int(end_c),
                "color": int(color),
            }
        )
    return summaries


def _choose_summary(summaries: list[dict[str, int]]) -> dict[str, int]:
    return min(
        summaries,
        key=lambda item: (item["rank"], -item["steps"], item["idx"]),
    )


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    priority = _read_priority(grid)
    priority_rank = {int(color): i for i, color in enumerate(priority)}
    combo_memo: dict[
        tuple[int, int, str, tuple[tuple[int, int], ...]], tuple[int, int, int]
    ] = {}
    length_memo: dict[
        tuple[int, int, str, tuple[tuple[int, int], ...]], tuple[int, int, int]
    ] = {}
    states: set[tuple[int, int, str]] = set()

    H, _ = grid.shape
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(grid, r) is None:
            continue
        _collect_ambiguous_states(grid, r, 0, "right", set(), states)

    priority_overrides = 0
    length_resolutions = 0
    for r, c, direction in states:
        combo = _option_summaries(grid, r, c, direction, priority_rank, combo_memo)
        length_only = _option_summaries(grid, r, c, direction, {}, length_memo)
        combo_best = _choose_summary(combo)
        length_best = _choose_summary(length_only)
        ranks = {item["rank"] for item in combo}
        lengths = {item["steps"] for item in combo}

        if combo_best["idx"] != length_best["idx"]:
            longest = max(item["steps"] for item in combo)
            if combo_best["steps"] < longest:
                priority_overrides += 1
        elif len(ranks) == 1 and len(lengths) > 1:
            length_resolutions += 1

    combo_out = main(grid)
    length_out = _main_with_priority(grid, {})
    old_solver_failed = 0
    try:
        _main_original(grid)
    except AssertionError:
        old_solver_failed = 1

    return {
        "ambiguous_states": len(states),
        "priority_overrides": priority_overrides,
        "length_resolutions": length_resolutions,
        "combo_differs_from_length_only": int(not np.array_equal(combo_out, length_out)),
        "old_solver_failed": old_solver_failed,
    }


def _choose_priority_colors(
    priority: list[int],
    priority_rank: dict[int, int],
    short_dir: str,
    outer_short_color: int,
    entry_color: int,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for up_color in priority:
        for down_color in priority:
            if up_color == down_color:
                continue
            if up_color == entry_color or down_color == entry_color:
                continue
            short_color = up_color if short_dir == "up" else down_color
            long_color = down_color if short_dir == "up" else up_color
            if short_color == outer_short_color:
                continue
            if priority_rank[int(short_color)] < priority_rank[int(long_color)]:
                candidates.append((int(up_color), int(down_color)))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    sep = W - _LEGEND_LEN - 1
    path_right = sep - 1
    if path_right - bar_start < 8:
        return None

    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], _N_EXTRA_COLORS)
    priority = random.sample(extras, _LEGEND_LEN)
    priority_rank = {int(color): i for i, color in enumerate(priority)}

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    max_non_adjacent = (n_channels + 1) // 2
    n_active = random.randint(_MIN_MOVED, min(_MAX_MOVED, max_non_adjacent))
    active = _pick_active_channels(n_channels, n_active)
    if len(active) < _MIN_MOVED:
        return None

    eligible_special = [idx for idx in active if 1 <= idx <= n_channels - 2]
    if len(eligible_special) < 2:
        return None
    priority_idx, length_idx = random.sample(eligible_special, 2)

    active_set = set(active)
    inactive = [idx for idx in range(n_channels) if idx not in active_set]
    row_colors = _build_row_colors(n_channels, active_set, extras)
    if len(set(row_colors)) < 3:
        return None

    _paint_base_bars(g, bar_rows, row_colors, bar_start, sep)

    used_cols: set[int] = set()
    for idx in active:
        top, bot = bar_rows[idx], bar_rows[idx + 1]
        row = top + 1
        entry_color = int(row_colors[idx])

        if idx == priority_idx:
            c_v = _allocate_turn_col(bar_start, path_right, used_cols)
            if c_v is None:
                return None
            switch_low = bar_start + 2
            switch_high = c_v - 2
            if switch_low > switch_high:
                return None
            long_dir = random.choice(["up", "down"])
            short_dir = "down" if long_dir == "up" else "up"
            outer_short_color = int(
                row_colors[idx - 1] if short_dir == "up" else row_colors[idx + 2]
            )
            branch_colors = _choose_priority_colors(
                priority, priority_rank, short_dir, outer_short_color, entry_color
            )
            if branch_colors is None:
                return None
            up_color, down_color = branch_colors
            switch_c = random.randint(switch_low, switch_high)
            _build_priority_fork(
                g,
                H,
                top,
                bot,
                row,
                bar_start,
                sep,
                entry_color,
                c_v,
                switch_c,
                up_color,
                down_color,
                long_dir,
            )
            continue

        if idx == length_idx:
            c_v = _allocate_turn_col(bar_start, path_right, used_cols)
            if c_v is None:
                return None
            long_dir = random.choice(["up", "down"])
            _build_length_fork(g, H, top, bot, row, c_v, entry_color, long_dir)
            continue

        _apply_real_entrance(g, top, bot, bar_start, sep, entry_color)
        if random.random() < 0.45:
            continue
        c_v = _allocate_turn_col(bar_start, path_right, used_cols)
        if c_v is None:
            continue
        direction = random.choice(["up", "down"])
        _build_edge_l_turn(g, H, top, bot, row, c_v, direction, entry_color)

    _add_fake_decor(g, inactive, bar_rows, row_colors, bar_start, sep, extras)

    g[0, sep] = BG
    g[H - 1, sep] = BG
    g[0, W - _LEGEND_LEN : W] = np.array(priority, dtype=int)
    g[H - 1, W - _LEGEND_LEN : W] = np.array(priority, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    moved_rows = [
        r for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    ]
    expected_rows = [bar_rows[idx] + 1 for idx in active]
    if len(moved_rows) != len(active):
        return None
    if sorted(moved_rows) != sorted(expected_rows):
        return None
    if len(set(map(int, g.flatten()))) != 6:
        return None

    metrics = _combo_metrics(g)
    if metrics["ambiguous_states"] < 2:
        return None
    if metrics["priority_overrides"] < 1:
        return None
    if metrics["length_resolutions"] < 1:
        return None
    if metrics["combo_differs_from_length_only"] != 1:
        return None
    if metrics["old_solver_failed"] != 1:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
