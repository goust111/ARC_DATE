from __future__ import annotations
import numpy as np
import random

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {"right": ["up", "down"], "left": ["up", "down"],
                  "up": ["left", "right"], "down": ["left", "right"]}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
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


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace particle from (r, c) going right. Return final position.

    Simple rule: go straight through bg; when blocked by a colored cell,
    check both perpendicular directions for a bg cell with same-color walls
    on both sides. Exactly one should work (turn); zero means stop;
    two means ambiguous (assert).
    """
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
        # Blocked by colored cell — try perpendicular turns
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
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along corridors formed by parallel colored bars.

    Input structure:
    - bg=7. Particles=6 in col 0. Colored bars on even rows form corridors on odd rows.

    Rule:
    - Particle goes straight through bg cells.
    - When blocked by a colored cell, turns 90° into the perpendicular direction
      that has a bg cell flanked by same-color bars. Stops at grid edge or full blockage.

    Key insight: Same-color parallel bars define the only valid corridors, so each particle follows a deterministic local trace.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        # Find corridor entrance: first bg cell with matching bars above/below starting at col ≤ 3
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out

# --------------- generator (sa_max_effort) ---------------

# === CFG: tunable generation parameters ===

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]          # odd heights (spec: 17-23)
_WIDTH_RANGE = (14, 21)                     # columns (spec: 14-21)
_BAR_START_OPTIONS = [2, 3]                 # leftmost column of horizontal bars
_N_EXTRA_COLORS = 4                         # corridor/decorator colors (+ BG + PARTICLE = 6)

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]              # real channel count choices (spec: 3-6)
_N_REAL_WEIGHTS = [4, 3, 2, 1]             # bias toward fewer reals to keep real_path_pct in 11-18%
_MIN_PALETTE_DIVERSITY = 3                  # reject if fewer distinct bar colors

# Turn probabilities
_STRAIGHT_PROB = 0.15                       # fraction of real channels with 0 turns
_L_INTERNAL_PROB = 0.30                     # fraction of L-turns that stop internally
_U_INTERNAL_STOP_PROB = 0.80                # fraction of U-turns that stop internally

# Vertical column allocation
_VCOL_LEFT_PAD = 4                          # min offset from bar_start for vertical corridors
_VCOL_RIGHT_PAD = 2                         # min distance from right edge
_VCOL_WINDOW = (-1, 0, 1)                   # columns reserved per allocated vcol

# Decoration
_FAKE_DECOR_RANGE = (0, 2)                  # decorative cells per fake channel row

# Validation
_MIN_MOVED = 3                              # min particles that must move
_MAX_MOVED = 6                              # max particles that must move
_MIN_DISTINCT_COLORS = 6                    # grid must have at least this many colors
_MAX_RETRIES = 300                          # outer retry loop limit


def _allocate_vcol(bar_start, W, used_vcols):
    """Reserve a 3-column window for a vertical corridor and return the wall column.

    Returns c_v such that c_v-1 (corridor), c_v (wall), c_v+1 (flank) are all free,
    or None if no space remains.
    """
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [c for c in range(c_min, c_max + 1)
                  if all((c + d) not in used_vcols for d in _VCOL_WINDOW)]
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
    for _, chans in by_color.items():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p:2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if random.random() < 0.5:
                plans[j] = {'kind': 'u', 'target': i, 'direction': 'up',
                            'c_v': c_v, 'internal_stop': internal_stop}
                plans[i] = {'kind': 'u_target'}
            else:
                plans[i] = {'kind': 'u', 'target': j, 'direction': 'down',
                            'c_v': c_v, 'internal_stop': internal_stop}
                plans[j] = {'kind': 'u_target'}


def _plan_remaining(real_indices, bar_start, W, used_vcols, plans):
    """Assign straight or L-turn plans to real channels not yet planned."""
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {'kind': 'straight'}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {'kind': 'straight'}
            continue
        direction = random.choice(['up', 'down'])
        kind = 'l_internal' if random.random() < _L_INTERNAL_PROB else 'l_edge'
        plans[i] = {'kind': kind, 'direction': direction, 'c_v': c_v}


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for task 9bbf930d.

    Striped layout: even rows = bar rows, odd rows = channel rows (BG).
    Real channels have same-color bar pairs; fake channels have mismatched colors.
    Trajectories: straight (0 turns), L-turn (1 turn), or U-turn (2 turns).
    """
    # --- Grid size and palette ---
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)],
                           _N_EXTRA_COLORS)

    # --- Canvas ---
    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    # --- Real/fake channel assignment ---
    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ['real' if i in real_set else 'fake' for i in range(n_channels)]

    # --- Bar colors: real = same top/bot; fake = different ---
    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == 'real':
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    # --- Paint full-width bars ---
    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    # --- Plan and execute turns ---
    real_indices = [i for i, s in enumerate(channel_spec) if s == 'real']
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    # --- Add decorative cells in fake channel rows ---
    for i, ctype in enumerate(channel_spec):
        if ctype != 'fake':
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

    # --- Re-enforce col 0 / col 1 ---
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    # --- Validate via solver ---
    try:
        out = main(g)
    except Exception:
        return None
    n_moved = sum(1 for r in range(H)
                  if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras):
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        if plan['kind'] in ('straight', 'u_target'):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = plan['c_v']

        if plan['kind'] in ('l_edge', 'l_internal'):
            direction = plan['direction']
            stop_at = None
            if plan['kind'] == 'l_internal':
                if direction == 'up':
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction,
                          bar_color, stop_at)
        elif plan['kind'] == 'u':
            j_top, j_bot = bar_rows[plan['target']], bar_rows[plan['target'] + 1]
            _build_u_turn(g, W, top, bot, row, color, c_v, plan['direction'],
                          j_top, j_bot, bar_color, bar_start, extras,
                          plan.get('internal_stop', False))


def _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at):
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v).

    direction 'up': turn cell at (top, c_v-1); vertical corridor extends up
                    to row 0 (or stop_at).
    direction 'down': turn cell at (bot, c_v-1); vertical corridor extends down
                      to row H-1 (or stop_at).
    """
    g[row, c_v] = color  # wall blocking horizontal travel

    # turn_bar = bar row where the turn cell is carved
    # opp_bar  = opposite bar row (kept continuous to prevent ambiguous turn)
    if direction == 'up':
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG                  # turn cell
    g[turn_bar, c_v - 2] = bar_color[turn_bar]  # left flank (re-enforce)
    g[turn_bar, c_v] = bar_color[turn_bar]       # right flank
    g[opp_bar, c_v - 1] = bar_color[opp_bar]    # seal opposite side

    # Carve vertical corridor gaps in bar rows beyond the turn
    if direction == 'up':
        for r_br in range(0, top, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(g, W, top, bot, row, color, c_v, direction, j_top, j_bot,
                  bar_color, bar_start, extras, internal_stop):
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot).

    Carves a vertical corridor at c_v-1, places a wall at (row, c_v), and
    breaks the target channel's entrance so the solver doesn't trace from it.
    If internal_stop, places a dead-end wall on the re-entry channel.
    """
    g[row, c_v] = color  # primary blocker

    if direction == 'up':
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = bar_color[bot]    # seal opposite side of source
        g[j_top, c_v - 2] = BG              # disambiguate re-entry turn
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = bar_color[top]
        g[j_bot, c_v - 2] = BG

    # Break target channel's entrance so its own particle doesn't trace straight
    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    # Optional dead-end wall on re-entry channel
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
