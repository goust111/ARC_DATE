# Variant of 9bbf930d
# Dimension: D2 — 转弯规则 (path)
# Change: 被彩色格挡住时检查两个垂直方向，选择有同色条对走廊的一侧转90度；必须恰好一个方向有效（无歧义断裂）。... → 允许双向歧义并取较长路径

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
# Design analysis for the D2 variant
#
# a) Pixel concretization and behavior trace
# - The input keeps the original striped geometry: even rows are colored bars,
#   odd rows are channels, col 0 is filled with particles, col 1 is bg.
# - The new visible object is an ambiguous fork on a real channel row:
#   the particle moves right until a colored blocker at (row, c_v); both the
#   top bar cell (top, c_v-1) and the bottom bar cell (bot, c_v-1) are carved
#   to bg, so both perpendicular turn cells are valid.
# - One branch is intentionally short: after entering it, the next blocking bar
#   is left sealed and the intervening channel is fake (mismatched colors), so
#   the particle cannot turn again and stops quickly.
# - The other branch is intentionally long: the generator carves the same
#   vertical column all the way to the grid edge, so the particle travels much
#   farther before exiting.
# - Concrete trace at the fork:
#   old logic: current cell sees two valid perpendicular corridors -> assert.
#   new logic: evaluate both continuations recursively, compare remaining path
#   lengths, then choose the longer branch and place the particle at that final
#   position.
#
# b) Complexity calibration
# - The variant is not reduced to an invisible tie-break. Each generated grid
#   must contain at least one reachable decisive ambiguity with two visible
#   branch lengths.
# - The rule space stays small: there are only three trajectory families in the
#   generator (straight, one-sided L turn, ambiguous fork), so examples remain
#   learnable from a few samples.
#
# c) Main() old assumptions that break under the variant
# - Original _trace() assumes len(options) <= 1. This is the direct old D2
#   invariant and fails immediately on a fork.
# - Original main() entrance detection does not depend on uniqueness of later
#   turns, so that part can stay unchanged.
# - Any new comparison must be structural, not color-id based. The solver now
#   compares total reachable step counts from the candidate turn states.
# - Tie handling is deterministic but secondary: the generator guarantees at
#   least one strict inequality, so visible behavior never depends on hidden
#   option ordering.
#
# d) Placement legality
# - Each vertical feature reserves a 3-column window around its wall column to
#   avoid overlapping carved turns and contradictory flanks.
# - The fork source is chosen away from the top/bottommost channels so both
#   perpendicular turn cells exist inside the grid.
# - The fork's short side is adjacent to a fake channel; this guarantees the
#   short branch stops cleanly instead of re-entering another real corridor.
#
# e) Visual observability
# - Humans can see the new rule because the decisive fork is drawn explicitly:
#   both up and down openings are present at the same blocker, but one shaft is
#   visibly much longer.
# - The generator rejects samples without a reachable decisive ambiguity, so the
#   training distribution does not collapse back to the original unambiguous
#   task.
#
# f) Visual cleanliness
# - Full-width horizontal bars preserve the original striped look.
# - Vertical structures are sparse and column-separated, so each fork/L-turn is
#   readable without dense crossings.
#
# g) Color inferability
# - Colors still only indicate corridor walls vs fake mismatches; they are not
#   used for ordering or arithmetic.
# - The new behavior depends on visible path length, not on hidden palette ids.
# ---------------------------------------------------------------------------


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color bars."""
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
    """Return valid turn states if the forward cell is blocked."""
    options = []
    H, W = grid.shape
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
    memo: dict[tuple[int, int, str], tuple[int, int, int]],
    visiting: set[tuple[int, int, str]],
) -> tuple[int, int, int]:
    """Return (remaining_steps, final_r, final_c) from a movement state."""
    key = (r, c, direction)
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
        extra, end_r, end_c = _best_trace_state(grid, nr, nc, direction, memo, visiting)
        result = (1 + extra, end_r, end_c)
    else:
        options = _next_turn_options(grid, r, c, direction)
        if not options:
            result = (0, r, c)
        else:
            best = None
            for tr, tc, turn_dir in options:
                extra, end_r, end_c = _best_trace_state(
                    grid, tr, tc, turn_dir, memo, visiting
                )
                cand = (1 + extra, end_r, end_c)
                if best is None or cand[0] > best[0]:
                    best = cand
            result = best

    visiting.remove(key)
    memo[key] = result
    return result


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace from (r,c), choosing the longer continuation when a fork appears."""
    memo: dict[tuple[int, int, str], tuple[int, int, int]] = {}
    _, end_r, end_c = _best_trace_state(grid, r, c, "right", memo, set())
    return end_r, end_c


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through striped corridors, resolving ambiguous turns by length."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------
#
# Generator-side guarantees for the new solver assumptions:
# - _find_entrance_col assumes real channels begin as aligned full-width bars
#   from col 2 or 3; the generator preserves that layout.
# - _best_trace_state compares full remaining path lengths and uses a finite
#   state cache. The generator avoids cycles by only emitting straight runs,
#   edge-reaching L turns, and a single dead-end-vs-edge fork.
# - Tie cases are legal but visually weak, so the generator rejects samples
#   without a reachable strict inequality at some ambiguous turn.

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_STRAIGHT_PROB = 0.25
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
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
    """Build a one-sided L turn that runs to the grid edge."""
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
    long_dir: str,
) -> None:
    """Build an ambiguous turn: both perpendicular options work, one goes farther."""
    color = bar_color[top]
    g[row, c_v] = color
    for turn_bar in (top, bot):
        g[turn_bar, c_v - 1] = BG
        g[turn_bar, c_v - 2] = bar_color[turn_bar]
        g[turn_bar, c_v] = bar_color[turn_bar]

    if long_dir == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original trace rule for validation: ambiguous turns are invalid."""
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
        r, c, direction = options[0]
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


def _collect_decisive_ambiguities(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    seen: set[tuple[int, int, str]],
    decisive: set[tuple[int, int, str]],
    memo: dict[tuple[int, int, str], tuple[int, int, int]],
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
        _collect_decisive_ambiguities(grid, nr, nc, direction, seen, decisive, memo)
        return

    options = _next_turn_options(grid, r, c, direction)
    if len(options) == 2:
        lengths = [
            _best_trace_state(grid, tr, tc, turn_dir, memo, set())[0]
            for tr, tc, turn_dir in options
        ]
        if lengths[0] != lengths[1]:
            decisive.add(key)
    for tr, tc, turn_dir in options:
        _collect_decisive_ambiguities(grid, tr, tc, turn_dir, seen, decisive, memo)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Measure whether the generated sample actually exercises the new D2 rule."""
    H, _ = grid.shape
    decisive: set[tuple[int, int, str]] = set()
    memo: dict[tuple[int, int, str], tuple[int, int, int]] = {}
    moved_rows = 0
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(grid, r) is None:
            continue
        seen: set[tuple[int, int, str]] = set()
        _collect_decisive_ambiguities(grid, r, 0, "right", seen, decisive, memo)
        end_r, end_c = _trace(grid, r, 0)
        if (end_r, end_c) != (r, 0):
            moved_rows += 1

    old_solver_failed = 0
    try:
        old_out = _main_original(grid)
    except AssertionError:
        old_solver_failed = 1
        old_out = None
    new_out = main(grid)
    differs_from_old = 0 if old_out is None else int(not np.array_equal(new_out, old_out))

    return {
        "decisive_ambiguities": len(decisive),
        "moved_rows": moved_rows,
        "old_solver_failed": old_solver_failed,
        "differs_from_old": differs_from_old,
    }


def _paint_fake_decor(
    g: np.ndarray,
    channel_spec: list[str],
    bar_rows: list[int],
    bar_start: int,
    extras: list[int],
    bar_color: dict[int, int],
) -> None:
    H, W = g.shape
    fake_rows = []
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        r = bar_rows[i] + 1
        fake_rows.append(r)
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        choices = [bar_color[bar_rows[i]], bar_color[bar_rows[i + 1]]]
        for c in cols:
            if g[r, c] == BG:
                g[r, c] = random.choice(choices)

    if not fake_rows:
        return
    used = set(map(int, g.flatten()))
    missing = [color for color in extras if color not in used]
    for color in missing:
        r = random.choice(fake_rows)
        candidates = [c for c in range(bar_start + 1, W) if int(g[r, c]) == BG]
        if candidates:
            g[r, random.choice(candidates)] = color


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    bar_start = random.choice(_BAR_START_OPTIONS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    if n_channels < 4:
        return None

    fork_low = 2 if n_channels >= 6 else 1
    fork_high = n_channels - 3 if n_channels >= 6 else n_channels - 2
    fork_idx = random.randint(fork_low, fork_high)
    fork_long_dir = random.choice(["up", "down"])
    short_adjacent = fork_idx + 1 if fork_long_dir == "up" else fork_idx - 1

    n_real = min(random.choice(_N_REAL_OPTIONS), n_channels - 1)
    candidates = [i for i in range(n_channels) if i != short_adjacent and i != fork_idx]
    if len(candidates) < n_real - 1:
        return None
    other_reals = set(random.sample(candidates, n_real - 1))
    real_set = {fork_idx, *other_reals}
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)
    if len(set(bar_color.values())) < 3:
        return None

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    used_vcols: set[int] = set()
    fork_c_v = _allocate_vcol(bar_start, W, used_vcols)
    if fork_c_v is None:
        return None

    for i in sorted(real_set):
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        if i == fork_idx:
            _build_fork(g, H, top, bot, row, fork_c_v, bar_color, fork_long_dir)
            continue
        if random.random() < _STRAIGHT_PROB:
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            continue
        direction = random.choice(["up", "down"])
        _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color)

    _paint_fake_decor(g, channel_spec, bar_rows, bar_start, extras, bar_color)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    out = main(g)
    n_moved = sum(
        1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None

    metrics = _variant_metrics(g)
    if metrics["decisive_ambiguities"] < 1:
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
