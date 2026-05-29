# Variant of 9bbf930d
# Dimension: D9 — 交叉口行为 (obstacle)
# Change: 垂直走廊与水平走廊交叉时，要么两条条完全断开让粒子惯性通过，要么两条条不断让粒子完全阻挡——不存在半通过状态。... → 交叉口信号灯机制

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
TURN_SIGNAL = 8
STRAIGHT_SIGNAL = 9

_TRAVERSABLE = {BG, PARTICLE, TURN_SIGNAL, STRAIGHT_SIGNAL}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


# Design analysis for the D9 variant
# ----------------------------------
# reference_task:
# - The prompt names e87109e9 as the source of the "symbol_instructions" idea.
# - In this workspace only the JSON for e87109e9 is available; no solver file was found
#   after searching the provided tree. The implementation below therefore uses the source
#   idea conservatively: local visible symbol pixels embedded in the traversed structure.
#
# a) concretization / behavior trace
# - The original task uses striped geometry:
#   even rows are colored bars, odd rows are channels, col 0 is filled with magenta
#   particles, and col 1 is blank orange background.
# - In the original task a particle only changes direction when the next cell is a solid
#   colored blocker. The blocker sits just to the right of the horizontal path, so the
#   turn decision is made from the cell before the blocker.
# - In this variant the decision point is moved onto the crossing cell itself. The cell
#   is a special visible symbol color:
#   TURN_SIGNAL   = 8
#   STRAIGHT_SIGNAL = 9
# - Both signal colors are traversable. A particle enters the signal cell, then:
#   TURN_SIGNAL: turn into the unique perpendicular branch available at that junction.
#   STRAIGHT_SIGNAL: keep current direction and pass through.
# - Small trace:
#   original: ... BG BG [wall-color] ...
#             particle stops seeing the wall, then tries a perpendicular corridor.
#   variant:  ... BG BG [signal-color] BG ...
#             particle steps onto the signal pixel. If it is TURN_SIGNAL it leaves along
#             the side branch; if it is STRAIGHT_SIGNAL it continues forward.
# - This creates the intended half-pass state that the original task forbids:
#   the junction can be geometrically open while a local symbol decides whether the
#   branch is taken.
#
# b) complexity calibration
# - Only two instruction colors are introduced. This keeps the new degree of freedom
#   small and repeated across examples.
# - The geometry of the relevant junctions is intentionally reused:
#   same striped bars, same channel entrances, same one-cell-wide vertical carve-outs.
# - Several real channels share the same two instruction colors in each sample so the
#   mapping "8 means turn / 9 means straight" can be inferred visually.
#
# c) implicit assumptions audit for main()
# - _has_corridor in the original code assumed the center cell must be BG/PARTICLE.
#   That no longer holds at signal cells, so traversability now includes both signal
#   colors. This matters when a later step checks a branch adjacent to a signal.
# - _trace previously treated any non-BG color ahead as a blocker. That would make the
#   new signal pixels behave like walls and destroy the variant. The signal case must be
#   checked before the old blocker logic.
# - main() entrance detection can stay based on a plain BG cell because generator keeps
#   the corridor entrance unchanged near the left edge. This prevents signal colors from
#   interfering with which rows are active.
#
# d) placement legality
# - Every signal column is spaced by a 3-column reservation window so vertical branches
#   do not overlap or create ambiguous multi-branch junctions.
# - A TURN_SIGNAL branch opens only on one side of the channel, preserving a unique
#   perpendicular option. STRAIGHT_SIGNAL uses the same T-junction geometry, so the two
#   cases are visually comparable.
# - Vertical carve-outs are only made inside painted bar rows and stay within bounds.
#
# e) visual observability
# - The rule is visible: identical-looking T-junctions appear with two different symbol
#   colors, and the particles end either at the right edge or at the top/bottom edge.
# - The logic does not depend on hidden color ordering or numeric color ids beyond these
#   two stable, repeated instruction colors.
#
# f) visual cleanliness
# - Signals are single pixels placed directly on channel cells. No overlaid multi-pixel
#   markers are used.
# - Branch columns are separated, so crossings do not pile up into unreadable clusters.
#
# g) color-change inferability
# - Colors are not remapped. The only semantic colors added are the two signal colors,
#   and they appear repeatedly with consistent meaning.
#
# new assumptions introduced by the solver and guaranteed by generate_input()
# - Every TURN_SIGNAL encountered by a moved particle has exactly one viable perpendicular
#   branch. The generator enforces this by carving only one vertical side at that column.
# - STRAIGHT_SIGNAL cells are followed by at least one forward cell inside the grid.
# - No signal column is reused, so a particle moving vertically never meets another signal
#   in the generated instances. The solver still supports that case generically.


def _is_bar_color(value: int) -> bool:
    return value not in _TRAVERSABLE


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars perpendicular to direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in _TRAVERSABLE:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and _is_bar_color(a)


def _turn_options_from(grid: np.ndarray, r: int, c: int, direction: str) -> list[str]:
    options: list[str] = []
    H, W = grid.shape
    for turn_dir in _PERPENDICULAR[direction]:
        dr, dc = _DELTAS[turn_dir]
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and _has_corridor(grid, nr, nc, turn_dir):
            options.append(turn_dir)
    return options


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace a particle starting at (r,c), initially moving right."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break

        nxt = int(grid[nr, nc])
        if nxt == TURN_SIGNAL:
            options = _turn_options_from(grid, nr, nc, direction)
            assert len(options) <= 1, "ambiguous turn-signal junction"
            if not options:
                break
            r, c = nr, nc
            direction = options[0]
            continue
        if nxt in {BG, PARTICLE, STRAIGHT_SIGNAL}:
            r, c = nr, nc
            continue

        options = _turn_options_from(grid, r, c, direction)
        assert len(options) <= 1, "ambiguous blocker turn"
        if not options:
            break
        direction = options[0]
        dr, dc = _DELTAS[direction]
        r, c = r + dr, c + dc
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through striped corridors with signal-controlled junctions."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue

        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and _is_bar_color(above):
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next(
            (c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])),
            W,
        )
        bar_below = next(
            (c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])),
            W,
        )
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_BAR_COLOR_POOL = [0, 1, 2, 3, 4, 5]

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_BAR_DIVERSITY = 3

_SIGNAL_WINDOW = (-1, 0, 1)
_STRAIGHT_SIGNAL_BRANCH_PROB = 0.5
_MIN_MOVED = 3
_MAX_MOVED = 6
_MAX_RETRIES = 400


def _allocate_signal_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    c_min = bar_start + 3
    c_max = W - 4
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_cols for d in _SIGNAL_WINDOW)
    ]
    if not candidates:
        return None
    x = random.choice(candidates)
    used_cols.update({x + d for d in _SIGNAL_WINDOW})
    return x


def _max_signal_slots(bar_start: int, W: int) -> int:
    c_min = bar_start + 3
    c_max = W - 4
    if c_min > c_max:
        return 0
    return 1 + (c_max - c_min) // 3


def _carve_vertical_branch(g: np.ndarray, top: int, bot: int, x: int, direction: str) -> None:
    H, _ = g.shape
    if direction == "up":
        rows = range(top, -1, -2)
    else:
        rows = range(bot, H, 2)
    for r_br in rows:
        g[r_br, x] = BG


def _plan_signal_kinds(signal_count: int) -> list[str]:
    kinds = ["turn", "straight_signal"]
    for _ in range(signal_count - 2):
        kinds.append(
            "straight_signal" if random.random() < _STRAIGHT_SIGNAL_BRANCH_PROB else "turn"
        )
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)

    max_slots = _max_signal_slots(bar_start, W)
    if max_slots < 2:
        return None

    extras = random.sample(_BAR_COLOR_POOL, 4)
    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)

    if len(set(bar_color.values())) < _MIN_BAR_DIVERSITY:
        return None

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    signal_count = random.randint(2, min(len(real_indices), max_slots))
    signal_indices = set(random.sample(real_indices, signal_count))
    signal_kinds = _plan_signal_kinds(signal_count)

    used_cols: set[int] = set()
    signal_rows = list(signal_indices)
    random.shuffle(signal_rows)
    for chan_idx, kind in zip(signal_rows, signal_kinds):
        x = _allocate_signal_col(bar_start, W, used_cols)
        if x is None:
            return None
        top, bot = bar_rows[chan_idx], bar_rows[chan_idx + 1]
        row = top + 1
        direction = random.choice(["up", "down"])
        g[row, x] = TURN_SIGNAL if kind == "turn" else STRAIGHT_SIGNAL
        _carve_vertical_branch(g, top, bot, x, direction)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        r = bar_rows[i] + 1
        n_decor = random.randint(0, 2)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        top, bot = bar_rows[i], bar_rows[i + 1]
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    moved = sum(
        1
        for r in range(H)
        if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None

    signal_values = set(int(v) for v in g.flatten() if int(v) in {TURN_SIGNAL, STRAIGHT_SIGNAL})
    if signal_values != {TURN_SIGNAL, STRAIGHT_SIGNAL}:
        return None

    turn_count = int(np.sum(g == TURN_SIGNAL))
    straight_signal_count = int(np.sum(g == STRAIGHT_SIGNAL))
    if turn_count < 1 or straight_signal_count < 1:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
