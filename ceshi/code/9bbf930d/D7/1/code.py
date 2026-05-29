# Variant of 9bbf930d
# Dimension: D7 — 变换：输出处理 (transform)
# Change: 粒子从起点消失（变背景），在终点出现（变粒子色），其他像素不变。... → 粒子改色为走廊色
#
# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior and trace point
# - Input is unchanged: odd channel rows hold moving particles in col 0, colored bar pairs
#   define real corridors, mismatched bar pairs define fake corridors.
# - Original output logic was: erase start cell -> write magenta (6) at traced endpoint.
# - Variant output logic is: erase start cell -> write the color of the endpoint corridor.
# - Small trace example:
#   input channel color 4 runs right, then turns upward through a vertical color-4 corridor.
#   Original solver stops at the final BG cell and writes 6 there.
#   Variant solver stops at the same cell but writes 4, because the final cell is flanked
#   by color-4 bars in the movement-perpendicular axis.
# - Therefore the only semantic change is in transform/output, but the solver now must know
#   the final travel direction so it can read the correct flanking bar pair at the endpoint.
#
# b) Complexity calibration
# - The variant stays close to the source task: path tracing, ambiguity checks, fake corridors,
#   and placement strategy remain intact.
# - To keep the recolor visible to humans, the generator rejects cases where all moved particles
#   would recolor to the same corridor color. This is a minimal observability adjustment, not a
#   rule change.
#
# c) Hidden assumptions in the original main()
# - `_trace()` previously returned only coordinates; that was sufficient when the endpoint write
#   color was constant (6). In the variant, that is no longer enough because the endpoint corridor
#   orientation matters.
# - `out[end_r, end_c] = PARTICLE` relied on endpoint color being input-independent. This becomes
#   wrong under D7 and must be replaced by corridor-color lookup on the original grid.
# - The corridor lookup must read from `g`, not `out`, because earlier particles may already have
#   changed `out`.
#
# d) Placement legality
# - The generator still uses the original corridor construction, so no new overlaps are introduced.
# - Endpoint recolor is always legal because the traced endpoint is a BG cell inside a real corridor;
#   writing the corridor color there only changes one output pixel and cannot affect tracing.
#
# e) Visual observability
# - Humans can infer the new rule because the endpoint pixel now matches the bars surrounding the
#   terminal corridor rather than staying magenta. This cue is local and directly visible.
# - Requiring at least two distinct endpoint colors per generated instance prevents the dataset from
#   collapsing into "all moved particles always become one fixed non-magenta color".
#
# f) Visual cleanliness
# - No new markers or overlays are introduced. The only changed pixel at the destination is now
#   harmonized with its corridor, which is visually cleaner than the original magenta endpoint.
#
# g) Color-change inferability
# - The recolor is not based on hidden color IDs or ordering. It is determined by the visible local
#   structure: the matched same-color bar pair flanking the particle's final corridor cell.

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg-like cell flanked by same-color bars perpendicular to direction."""
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


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int:
    """Return the bar-pair color of the corridor containing the endpoint cell."""
    H, W = grid.shape

    if direction in {"left", "right"} and 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            return a
    if direction in {"up", "down"} and 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            return a

    candidates = []
    if 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    if 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    assert len(set(candidates)) == 1, "final position is not on a unique real corridor"
    return int(candidates[0])


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Find the first valid real-corridor entrance for row r, or None."""
    _H, W = grid.shape
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
    return int(entrance_col)


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, str]:
    """Trace particle from (r, c) going right. Return final position and final direction."""
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
    return r, c, direction


def _moved_endpoint_colors(grid: np.ndarray) -> list[int]:
    """Collect endpoint corridor colors for all particles that actually move."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    colors = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, end_dir = _trace(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
        colors.append(_corridor_color(g, end_r, end_c, end_dir))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles along corridors and recolor each endpoint to its terminal corridor color."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c, end_dir = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = _corridor_color(g, end_r, end_c, end_dir)

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
_MIN_ENDPOINT_COLOR_DIVERSITY = 2
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


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D7 recolor-at-endpoint variant."""
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
        _out = main(g)
    except Exception:
        return None

    endpoint_colors = _moved_endpoint_colors(g)
    n_moved = len(endpoint_colors)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if len(set(endpoint_colors)) < _MIN_ENDPOINT_COLOR_DIVERSITY:
        return None
    return g


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


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
