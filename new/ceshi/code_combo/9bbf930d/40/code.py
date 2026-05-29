# Combo variant of 9bbf930d (index 40)
# combo_spec: D2/2+D16/0
# Name: 双向歧义取较长路径+间隙宽度限制

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
MAX_GAP_WIDTH = 2

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
# Design analysis for the combo variant D2/2 + D16/0
#
# a) Pixel concretization and behavior trace
# - The grid keeps the original striped world: even rows are bar rows, odd rows
#   are channel rows, col 0 is particle, col 1 is bg, and real horizontal
#   corridors are channels flanked by same-color bars.
# - The decisive object is an ambiguous fork on one real channel. The particle
#   moves right until a colored blocker at (row, c_v). Both turn cells
#   (top, c_v-1) and (bot, c_v-1) are carved to bg, so both perpendicular
#   directions are valid.
# - One branch is visually longer: its vertical shaft continues farther toward
#   the distant edge. Under the original D2-only change, the solver would pick
#   this branch because its remaining path length is larger.
# - The fusion adds a width bound N=2 on unsupported bg traversal. On the long
#   branch, the generator erases both side flanks on one bar row inside the
#   shaft. Vertical movement already coasts through odd rows unsupported; after
#   one erased support row, the unsupported streak reaches 3 and the particle
#   stops early. The shorter branch stays intact, so its bounded reachable
#   length becomes larger.
# - Small trace:
#   1. Old original task: both perpendicular turns valid -> assert/fail.
#   2. D2-only task: compare full branch lengths without gap budget -> choose
#      the visually longer shaft.
#   3. Fused task here: compare branch lengths after applying the width cap on
#      every candidate continuation -> choose the other branch if the long shaft
#      is truncated by an over-wide gap.
# - A separate straight real corridor contains a short horizontal gap of width
#   1 or 2. That gap is still traversed successfully, so examples show both
#   "small gaps pass" and "large gaps stop", rather than collapsing to a vague
#   "any hole blocks" rule.
#
# b) Complexity calibration
# - The generator always includes exactly one reachable decisive ambiguous fork,
#   plus at least one independent short traversable gap. This keeps the new
#   behavior visible in every sample.
# - Free choices remain limited: the fork row, fork column, long-branch
#   direction, long-gap depth, and 1-2 auxiliary real channels. The rule space
#   is still small enough to infer from a few ARC examples.
#
# c) Implicit assumptions audited against main()
# - Original `_trace()` assumed "all bg ahead is traversable forever". This is
#   broken by D16/0, so the fused solver must carry an `unsupported_run` state.
# - D2/2 means branch comparison can no longer depend only on geometry at the
#   fork. It must recursively evaluate candidate continuations with the same
#   gap budget rule, because one candidate may be truncated later.
# - Entrance detection still assumes the first real corridor starts at col 2 or
#   3 with aligned full-width bars. The generator preserves that.
# - Turn targets are still required to be supported real corridor cells. The
#   generator guarantees both fork turn cells satisfy `_has_corridor(...)`.
# - Ties are possible in theory, but weak visually. The generator rejects
#   samples unless the bounded and unbounded best directions differ strictly.
#
# d) Placement legality
# - Every vertical feature reserves a 3-column window around its wall column.
#   This prevents overlapping turn carvings and contradictory flanks.
# - The fork channel is chosen away from the top/bottom edges so both branches
#   exist and the visually longer side has room for a truncating gap.
# - Auxiliary straight-gap rows are kept separate from the fork row. Gap carving
#   never touches the entrance area, so particles can still enter cleanly.
#
# e) Visual observability
# - Humans can see both ingredients directly: the ambiguous fork is explicit,
#   and the long branch visibly contains a wider orange interruption that stops
#   it earlier than its apparent geometric extent suggests.
# - The short traversable gap supplies repeated evidence that the stopping rule
#   depends on gap width, not merely on the presence of any missing supports.
#
# f) Visual cleanliness
# - The layout remains sparse: mostly full-width bars, one fork column, and at
#   most a couple of extra L-turn columns.
# - Gap markings are localized erasures on otherwise regular corridors, so they
#   read as holes rather than as dense noise.
#
# g) Change inferability
# - Colors still only indicate corridor matching vs mismatch. No ordering or
#   branching decision depends on the numeric color ids.
# - The decisive branch flip is caused by visible path length after truncation,
#   not by hidden metadata. The generator rejects samples where the fused solver
#   behaves the same as the unbounded D2-only solver.
# ---------------------------------------------------------------------------


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by matching same-color bars."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


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
) -> list[tuple[int, int, str]]:
    """Return valid perpendicular turn states if the forward cell is blocked."""
    H, W = grid.shape
    options = []
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))
    return options


def _best_trace_state(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    unsupported_run: int,
    max_gap_width: int | None,
    memo: dict[tuple[int, int, str, int, int], tuple[int, int, int]],
    visiting: set[tuple[int, int, str, int, int]],
) -> tuple[int, int, int]:
    """Return (remaining_steps, final_r, final_c) from a movement state."""
    limit_key = -1 if max_gap_width is None else max_gap_width
    run_key = 0 if max_gap_width is None else unsupported_run
    key = (r, c, direction, run_key, limit_key)
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
        if max_gap_width is None:
            next_run = 0
        else:
            next_run = 0 if _has_corridor(grid, nr, nc, direction) else unsupported_run + 1
        if max_gap_width is not None and next_run > max_gap_width:
            result = (0, r, c)
        else:
            extra, end_r, end_c = _best_trace_state(
                grid,
                nr,
                nc,
                direction,
                next_run,
                max_gap_width,
                memo,
                visiting,
            )
            result = (1 + extra, end_r, end_c)
    else:
        options = _next_turn_options(grid, r, c, direction)
        if not options:
            result = (0, r, c)
        else:
            best = None
            for tr, tc, turn_dir in options:
                extra, end_r, end_c = _best_trace_state(
                    grid,
                    tr,
                    tc,
                    turn_dir,
                    0,
                    max_gap_width,
                    memo,
                    visiting,
                )
                cand = (1 + extra, end_r, end_c)
                if best is None or cand[0] > best[0]:
                    best = cand
            result = best

    visiting.remove(key)
    memo[key] = result
    return result


def _trace_with_limit(
    grid: np.ndarray, r: int, c: int, max_gap_width: int | None
) -> tuple[int, int]:
    memo: dict[tuple[int, int, str, int, int], tuple[int, int, int]] = {}
    _, end_r, end_c = _best_trace_state(
        grid,
        r,
        c,
        "right",
        0,
        max_gap_width,
        memo,
        set(),
    )
    return end_r, end_c


def _main_from_limit(grid: np.ndarray, max_gap_width: int | None) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace_with_limit(g, r, 0, max_gap_width)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through striped corridors with length-aware ambiguous turns.

    Input structure:
    - bg=7 and particles=6 fill col 0; col 1 is bg.
    - Even rows are colored bar rows and odd rows are channel rows.
    - Real corridors are bg cells flanked by same-color bars; fake corridors use
      mismatched bar colors and only act as distractors.

    Rule:
    - The particle moves straight through bg cells.
    - Traversal through unsupported bg is limited to width 2; the third
      consecutive unsupported cell is not entered.
    - When a colored blocker is ahead, the particle checks both perpendicular
      directions. If both are valid real corridors, it chooses the branch with
      the longer remaining reachable path under the same width-2 rule.
    - The particle stops at the grid edge, at a blocker with no valid turn, or
      immediately before exceeding the gap-width budget.

    Key insight:
    - Ambiguous forks are resolved by comparing reachable future distance after
      applying the gap-width limit, so a visually longer branch can lose if it
      contains a too-wide interruption later on.
    """
    return _main_from_limit(grid, MAX_GAP_WIDTH)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5]
_MIN_PALETTE_DIVERSITY = 3

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_STRAIGHT_PROB = 0.65
_FAKE_DECOR_RANGE = (0, 2)
_SHORT_GAP_OPTIONS = [1, 2]

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 600


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical feature and return the wall col."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
) -> None:
    """Build a one-sided L turn that reaches the chosen edge."""
    g[row, c_v] = bar_color[top]
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _build_fork(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    bar_color: dict[int, int],
) -> None:
    """Build an ambiguous fork with both up/down turn cells open."""
    color = bar_color[top]
    g[row, c_v] = color
    for turn_bar in (top, bot):
        g[turn_bar, c_v - 1] = BG
        g[turn_bar, c_v - 2] = bar_color[turn_bar]
        g[turn_bar, c_v] = bar_color[turn_bar]

    for r_br in range(0, top, 2):
        g[r_br, c_v - 1] = BG
    for r_br in range(bot + 2, H, 2):
        g[r_br, c_v - 1] = BG


def _branch_supported_rows(top: int, bot: int, H: int, direction: str) -> list[int]:
    """Supported even rows on a fork branch after the initial turn cell."""
    if direction == "up":
        return list(range(top - 2, -1, -2))
    return list(range(bot + 2, H, 2))


def _carve_vertical_long_gap(g: np.ndarray, gap_row: int, c_v: int) -> None:
    """Erase both side flanks at one bar row to create a fatal width>2 gap."""
    g[gap_row, c_v - 2] = BG
    g[gap_row, c_v] = BG


def _carve_horizontal_gap(g: np.ndarray, top: int, bot: int, start: int, length: int) -> None:
    g[top, start : start + length] = BG
    g[bot, start : start + length] = BG


def _pick_fork_index(n_channels: int) -> int:
    low = 2 if n_channels >= 7 else 1
    high = n_channels - 3 if n_channels >= 7 else n_channels - 2
    return random.randint(low, high)


def _build_bar_colors(
    bar_rows: list[int], channel_spec: list[str], extras: list[int]
) -> dict[int, int] | None:
    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None
    return bar_color


def _paint_fake_decor(
    g: np.ndarray,
    channel_spec: list[str],
    bar_rows: list[int],
    bar_start: int,
    bar_color: dict[int, int],
    extras: list[int],
) -> None:
    W = g.shape[1]
    fake_rows = []
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        fake_rows.append(row)
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        choices = [bar_color[top], bar_color[bot]]
        for c in cols:
            if int(g[row, c]) == BG:
                g[row, c] = random.choice(choices)

    if not fake_rows:
        return
    used = set(map(int, g.flatten()))
    missing = [color for color in extras if color not in used]
    for color in missing:
        row = random.choice(fake_rows)
        candidates = [c for c in range(bar_start + 1, W) if int(g[row, c]) == BG]
        if candidates:
            g[row, random.choice(candidates)] = color


def _best_dir_for_fork(
    grid: np.ndarray, row: int, c_v: int, max_gap_width: int | None
) -> tuple[str, dict[str, int]] | None:
    state_r, state_c = row, c_v - 1
    options = _next_turn_options(grid, state_r, state_c, "right")
    if len(options) != 2:
        return None

    memo: dict[tuple[int, int, str, int, int], tuple[int, int, int]] = {}
    lengths: dict[str, int] = {}
    for tr, tc, turn_dir in options:
        extra, _, _ = _best_trace_state(
            grid,
            tr,
            tc,
            turn_dir,
            0,
            max_gap_width,
            memo,
            set(),
        )
        lengths[turn_dir] = 1 + extra
    best_dir = max(lengths, key=lengths.get)
    return best_dir, lengths


def _variant_metrics(
    grid: np.ndarray, meta: dict[str, int | str]
) -> dict[str, int | str | tuple[int, int]]:
    H, _ = grid.shape
    moved_rows = 0
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(grid, r) is None:
            continue
        if _trace_with_limit(grid, r, 0, MAX_GAP_WIDTH) != (r, 0):
            moved_rows += 1

    fork_row = int(meta["fork_row"])
    fork_c_v = int(meta["fork_c_v"])
    bounded = _best_dir_for_fork(grid, fork_row, fork_c_v, MAX_GAP_WIDTH)
    unbounded = _best_dir_for_fork(grid, fork_row, fork_c_v, None)
    if bounded is None or unbounded is None:
        return {
            "moved_rows": moved_rows,
            "fork_two_options": 0,
            "choice_flips": 0,
            "bounded_best": "",
            "unbounded_best": "",
            "bounded_lengths": (0, 0),
            "unbounded_lengths": (0, 0),
            "short_gap_ok": 0,
        }

    bounded_best, bounded_lengths = bounded
    unbounded_best, unbounded_lengths = unbounded
    short_row = int(meta["short_gap_row"])
    short_start = int(meta["short_gap_start"])
    short_len = int(meta["short_gap_len"])
    bounded_end = _trace_with_limit(grid, short_row, 0, MAX_GAP_WIDTH)
    unbounded_end = _trace_with_limit(grid, short_row, 0, None)
    short_gap_ok = int(
        bounded_end == unbounded_end and bounded_end[1] > short_start + short_len - 1
    )

    return {
        "moved_rows": moved_rows,
        "fork_two_options": 1,
        "choice_flips": int(bounded_best != unbounded_best),
        "bounded_best": bounded_best,
        "unbounded_best": unbounded_best,
        "bounded_lengths": (
            bounded_lengths.get("up", 0),
            bounded_lengths.get("down", 0),
        ),
        "unbounded_lengths": (
            unbounded_lengths.get("up", 0),
            unbounded_lengths.get("down", 0),
        ),
        "short_gap_ok": short_gap_ok,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    if n_channels < 5:
        return None

    fork_idx = _pick_fork_index(n_channels)
    all_indices = list(range(n_channels))
    candidates = [i for i in all_indices if i != fork_idx]
    random.shuffle(candidates)

    n_real = min(random.choice(_N_REAL_OPTIONS), n_channels)
    if n_real < 4 or len(candidates) < n_real - 1:
        return None
    real_set = {fork_idx, *candidates[: n_real - 1]}

    channel_spec = ["real" if i in real_set else "fake" for i in all_indices]
    bar_color = _build_bar_colors(bar_rows, channel_spec, extras)
    if bar_color is None:
        return None

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    used_vcols: set[int] = set()

    top, bot = bar_rows[fork_idx], bar_rows[fork_idx + 1]
    fork_row = top + 1
    fork_c_v = _allocate_vcol(bar_start, W, used_vcols)
    if fork_c_v is None:
        return None
    _build_fork(g, H, top, bot, fork_row, fork_c_v, bar_color)

    up_rows = _branch_supported_rows(top, bot, H, "up")
    down_rows = _branch_supported_rows(top, bot, H, "down")
    if len(up_rows) < 1 or len(down_rows) < 1:
        return None

    raw_lengths = {
        "up": top + 1,
        "down": H - bot,
    }
    long_dir = "up" if raw_lengths["up"] > raw_lengths["down"] else "down"
    short_dir = "down" if long_dir == "up" else "up"
    long_rows = up_rows if long_dir == "up" else down_rows
    short_rows = down_rows if long_dir == "up" else up_rows
    if raw_lengths[long_dir] <= raw_lengths[short_dir]:
        return None
    if not long_rows or not short_rows:
        return None

    gap_candidates = long_rows[: min(3, len(long_rows))]
    gap_row = random.choice(gap_candidates)
    _carve_vertical_long_gap(g, gap_row, fork_c_v)

    short_gap_idx = next((i for i in candidates if i in real_set), None)
    if short_gap_idx is None or short_gap_idx == fork_idx:
        return None

    remaining_reals = [i for i in sorted(real_set) if i not in {fork_idx}]
    for i in remaining_reals:
        if i == short_gap_idx:
            continue
        if random.random() < _STRAIGHT_PROB:
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            continue
        top_i, bot_i = bar_rows[i], bar_rows[i + 1]
        if top_i <= 2:
            direction = "down"
        elif bot_i >= H - 3:
            direction = "up"
        else:
            direction = random.choice(["up", "down"])
        _build_l_turn(g, H, top_i, bot_i, top_i + 1, c_v, direction, bar_color)

    short_top, short_bot = bar_rows[short_gap_idx], bar_rows[short_gap_idx + 1]
    short_gap_len = random.choice(_SHORT_GAP_OPTIONS)
    short_start_min = bar_start + 2
    short_start_max = max(short_start_min, W - short_gap_len - 3)
    short_start = random.randint(short_start_min, short_start_max)
    _carve_horizontal_gap(g, short_top, short_bot, short_start, short_gap_len)

    _paint_fake_decor(g, channel_spec, bar_rows, bar_start, bar_color, extras)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    meta = {
        "fork_row": fork_row,
        "fork_c_v": fork_c_v,
        "fork_long_dir": long_dir,
        "short_gap_row": short_top + 1,
        "short_gap_start": short_start,
        "short_gap_len": short_gap_len,
    }

    try:
        out = main(g)
        out_unbounded = _main_from_limit(g, None)
    except Exception:
        return None

    if np.array_equal(out, out_unbounded):
        return None

    n_moved = sum(
        1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None

    metrics = _variant_metrics(g, meta)
    if metrics["fork_two_options"] != 1:
        return None
    if metrics["choice_flips"] != 1:
        return None
    if metrics["short_gap_ok"] != 1:
        return None
    if metrics["bounded_best"] != short_dir:
        return None
    if metrics["unbounded_best"] != long_dir:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
