# Variant of 9bbf930d
# Dimension: D4 — trajectory
# Change: no trajectory marks in the output -> mark only the turn cells

from __future__ import annotations

import numpy as np
import random

BG = 7
PARTICLE = 6
TURN_MARK = PARTICLE
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}

# Design analysis for the D4 variant
# a) Concretization and behavior trace
#    The input stays identical to the base task: particles in column 0, striped
#    horizontal bars on even rows, real corridors where opposite bars share a
#    color, fake corridors where they do not. The path logic is also identical:
#    a particle moves right through background cells and turns only when a
#    colored blocker sits ahead and exactly one perpendicular same-color
#    corridor is available.
#    The variant changes only the output rendering. Instead of erasing the
#    start cell and placing a single particle at the final stop, the output now
#    keeps the final particle and additionally paints every turn-entry cell
#    visited during the trace. Straight segments remain background. Example:
#    moving right to a blocker at (5, 10) and turning up into (4, 9) used to
#    produce only the final endpoint; now (4, 9) is also colored 6.
# b) Complexity calibration
#    Marking only turn cells is subtle if too few turns exist, so the generator
#    must reject samples where almost all moved particles are straight. The
#    geometric layout is otherwise unchanged; the puzzle still requires
#    understanding the same corridor semantics before the turn marks become
#    predictable.
# c) Hidden assumptions in main()
#    - _has_corridor previously allowed BG or PARTICLE in the traversed cell.
#      That remains valid because the solver traces on the original input grid,
#      not on the marked output.
#    - The old main() assumed a single write per moved particle: clear start and
#      place one endpoint. In the variant this is no longer sufficient; we must
#      preserve every turn location returned by tracing.
#    - Writing marks is safe because a valid turn target is always a corridor
#      cell, hence BG/PARTICLE in the input. The generator never places colored
#      walls on those cells.
# d) Placement legality
#    The original corridor builder already guarantees non-overlapping turn
#    structures via reserved vertical-column windows and sealed opposite bars.
#    Since the variant only changes output paint, geometry legality is inherited
#    from the base generator. Additional validation checks ensure there are
#    enough turning trajectories to make the new behavior visible.
# e) Visual observability
#    The mark color is the particle color itself. This avoids introducing a new
#    unexplained palette mapping and makes the rule visible from a few examples:
#    magenta appears at the final resting place and at every corner, but not
#    along straight runs.
# f) Visual cleanliness
#    Only sparse corner cells are marked, so outputs stay readable. The input
#    still contains uncluttered striped corridors; the output adds a small set
#    of isolated magenta corner cells instead of painting whole paths.
# g) Color inference
#    No new color transformation is introduced. The only visible change is where
#    the existing particle color appears, which is structurally determined by
#    turns rather than by any hidden color-id ordering.


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars."""
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
    """Return the first valid corridor entrance for row r, if any."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            entrance_col = c
            break
    if entrance_col is None:
        return None
    bar_above = next(
        (c for c in range(W) if int(grid[r - 1, c]) == int(grid[r - 1, entrance_col])),
        W,
    )
    bar_below = next(
        (c for c in range(W) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])),
        W,
    )
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Trace one particle and return its final cell plus the visited turn cells."""
    H, W = grid.shape
    direction = "right"
    turns: list[tuple[int, int]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        turns.append((r, c))
    return r, c, tuple(turns)


def _collect_traces(grid: np.ndarray) -> list[dict[str, object]]:
    """Collect trace summaries for all valid particle rows in the input."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    traces: list[dict[str, object]] = []
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns = _trace(g, r, 0)
        traces.append(
            {
                "row": r,
                "end": (end_r, end_c),
                "turns": turns,
                "moved": (end_r, end_c) != (r, 0),
            }
        )
    return traces


def main(grid: np.ndarray) -> np.ndarray:
    """Move each valid particle; output keeps the endpoint and paints turn cells."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns = _trace(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
        out[r, 0] = BG
        for tr, tc in turns:
            out[tr, tc] = TURN_MARK
        out[end_r, end_c] = PARTICLE

    return out


# --------------- generator (sa_max_effort) ---------------

# === CFG: tunable generation parameters ===

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
_MIN_TURNING_PARTICLES = 2
_MIN_TURN_MARKS = 2
_MAX_RETRIES = 300


def _allocate_vcol(bar_start, W, used_vcols):
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans):
    """Pair same-color real channels into U-turn plans."""
    by_color = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
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


def _plan_remaining(real_indices, bar_start, W, used_vcols, plans):
    """Assign straight or L-turn plans to real channels not yet planned."""
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras):
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = plan["c_v"]

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = plan["direction"]
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at):
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
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
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g, W, top, bot, row, color, c_v, direction, j_top, j_bot, bar_color, bar_start, extras, internal_stop
):
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot)."""
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
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D4 variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

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
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    traces = [trace for trace in _collect_traces(g) if trace["moved"]]
    n_moved = len(traces)
    turning_particles = sum(1 for trace in traces if trace["turns"])
    turn_marks = sum(len(trace["turns"]) for trace in traces)

    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if turning_particles < _MIN_TURNING_PARTICLES or turn_marks < _MIN_TURN_MARKS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    # The variant must differ from the base task on these accepted inputs.
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
