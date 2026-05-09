# Variant of 9bbf930d
# Dimension: D10 — 真走廊与假走廊的区分 (obstacle)
# Change: 真走廊两侧同色，假走廊两侧异色；粒子只沿真走廊转弯。假走廊作为干扰项，结构上与真走廊几乎相同。... → 假走廊为陷阱

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


# Design analysis for the D10 trap variant
# a) Concrete behavior:
#    The original task only lets a blocked particle turn into a perpendicular
#    corridor mouth when the two flanking bar cells have the same color.
#    This variant keeps that rule for true corridors, but if the unique
#    perpendicular mouth is corridor-shaped with two different flank colors,
#    the particle is "fooled": it steps into that mouth cell and stops there.
#    Small example:
#    - Original: particle moves right, sees a blocker, sees only a fake L-mouth,
#      and therefore stops before the blocker.
#    - Variant: the same particle enters the fake mouth cell and gets trapped,
#      so the output particle is shifted by one perpendicular step.
#    The visible difference is that some particles end on even-row turn-mouth
#    cells rather than staying on their source channel.
# b) Complexity calibration:
#    The variant is kept local to turn resolution. Real turns, U-turns, row
#    starts, palette size, and striped layout stay recognizable. Generator
#    emits both real turns and trap mouths so the contrast is visible.
# c) Old implicit assumptions that no longer hold:
#    - Old _has_corridor assumed only same-color mouths matter.
#    - Old _trace treated every non-real mouth as absence of a turn.
#    - Old generator only needed fake channels as inert distractors; now it must
#      construct fake L-mouths that are reachable from moving particles.
# d) Placement legality:
#    Trap mouths reuse the original 3-column reservation for turn columns, so
#    they do not overlap other vertical structures. Each trap seals the opposite
#    side exactly like a real L-turn, preventing ambiguous up/down choices.
# e) Visual observability:
#    Outputs differ from the original task on generated trap instances because
#    the trapped particle ends inside a mismatched-color mouth. The generator
#    rejects grids whose output matches the old solver exactly.
# f) Visual cleanliness:
#    Traps are short vertical branches attached to otherwise standard channels.
#    They occupy reserved columns, so the grid remains striped and readable.
# g) Color inference:
#    No hidden color ordering is used. The only semantic cue is visible equality
#    vs inequality of the two flank colors around a corridor mouth.
# 9) New solver assumptions and generator guarantees:
#    - At any blocker, there is at most one corridor-like perpendicular mouth
#      (real or fake). The generator seals the opposite side and reserves turn
#      columns so main() can assert against ambiguity.
#    - Start rows are still real horizontal channels. The generator keeps fake
#      horizontal rows mismatched so main() only launches particles from real
#      rows.
#    - A trap must be attached to a moving real channel. The generator validates
#      this by requiring the new solver output to differ from the old solver.


def _corridor_kind(grid: np.ndarray, r: int, c: int, direction: str) -> str | None:
    """Classify a corridor mouth as 'real', 'fake', or None."""
    h, w = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a in {PARTICLE, BG} or b in {PARTICLE, BG}:
        return None
    return "real" if a == b else "fake"


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle under the trap variant."""
    h, w = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue

        real_options = []
        fake_options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            kind = _corridor_kind(grid, tr, tc, turn_dir)
            if kind == "real":
                real_options.append((tr, tc, turn_dir))
            elif kind == "fake":
                fake_options.append((tr, tc, turn_dir))

        total = len(real_options) + len(fake_options)
        assert total <= 1, "ambiguous turn: multiple perpendicular mouths"
        if real_options:
            r, c, direction = real_options[0]
            continue
        if fake_options:
            return fake_options[0][0], fake_options[0][1]
        break
    return r, c


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original tracing rule used only to reject degenerate generators."""
    h, w = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _corridor_kind(grid, tr, tc, turn_dir) == "real":
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _solve_with_trace(grid: np.ndarray, trace_fn) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            if _corridor_kind(g, r, c, "right") == "real":
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), w)
        bar_below = next((c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), w)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = trace_fn(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through real corridors; fake mouths trap at the entrance."""
    return _solve_with_trace(grid, _trace)


def _main_original(grid: np.ndarray) -> np.ndarray:
    return _solve_with_trace(grid, _trace_original)


# --------------- generator (sa_max_effort) ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80
_TRAP_COUNT_RANGE = (1, 2)
_TRAP_DECOR_DEPTH_RANGE = (0, 2)

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = w - _VCOL_RIGHT_PAD
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


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Pair same-color channels, but leave spare channels for trap plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 3:
            continue
        random.shuffle(chans)
        max_pairs = (len(chans) - 1) // 2
        n_pairs = random.randint(0, max_pairs)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, w, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if random.random() < 0.5:
                plans[j] = {
                    "kind": "u",
                    "target": i,
                    "direction": "up",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[j] = {"kind": "u_target"}


def _plan_traps(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    available = [i for i in real_indices if i not in plans]
    if not available:
        return
    random.shuffle(available)
    lo, hi = _TRAP_COUNT_RANGE
    n_traps = min(len(available), random.randint(lo, min(hi, len(available))))
    for i in available[:n_traps]:
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            break
        plans[i] = {"kind": "trap", "direction": random.choice(["up", "down"]), "c_v": c_v}


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": random.choice(["up", "down"]), "c_v": c_v}


def _build_l_turn(
    g: np.ndarray,
    h: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    g[row, c_v] = color
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
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, h, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    w: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    g[row, c_v] = color
    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = bar_color[top]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, w - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _build_fake_trap(
    g: np.ndarray,
    h: int,
    top: int,
    bot: int,
    row: int,
    source_color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    trap_color: int,
) -> None:
    """Build a fake L-mouth. The entrance is fake; extra carved cells are decorative."""
    g[row, c_v] = source_color
    if direction == "up":
        turn_bar, opp_bar = top, bot
        extra_rows = list(range(top - 2, -1, -2))
    else:
        turn_bar, opp_bar = bot, top
        extra_rows = list(range(bot + 2, h, 2))

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = trap_color
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    if extra_rows:
        depth = random.randint(*_TRAP_DECOR_DEPTH_RANGE)
        for r_br in extra_rows[:depth]:
            g[r_br, c_v - 1] = BG
            g[r_br, c_v] = trap_color


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        kind = plan["kind"]
        if kind in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, h - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, h, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif kind == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                w,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )
        elif kind == "trap":
            trap_color_choices = [c for c in extras if c != color]
            if not trap_color_choices:
                trap_color_choices = [c for c in range(10) if c not in (BG, PARTICLE, color)]
            _build_fake_trap(
                g,
                h,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                bar_color,
                random.choice(trap_color_choices),
            )


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
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
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_traps(real_indices, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, w, used_vcols, plans)
    if not any(plan["kind"] == "trap" for plan in plans.values()):
        return None
    _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w), min(n_decor, w - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        old_out = _main_original(g)
    except Exception:
        return None
    if np.array_equal(out, old_out):
        return None

    n_moved = sum(1 for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
