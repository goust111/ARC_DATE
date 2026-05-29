# Variant of 9bbf930d
# Dimension: D4 — 轨迹涂色 (trajectory)
# Change: 无轨迹——粒子移动后只在终点出现，路径上不留任何痕迹。... → 连续轨迹涂色

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

"""
Design analysis for the D4 variant
==================================

a) Concrete behavior and a traced example
-----------------------------------------
Original task behavior:
- Input is a striped corridor field. Particles (color 6) sit in column 0.
- A movable particle starts on a real odd-row corridor, travels through background
  cells, optionally turns into same-color perpendicular corridors, and stops at an
  edge or a dead end.
- Output only shows the final resting particle: the start cell becomes background,
  all intermediate visited background cells remain background.

Variant concretization:
- Keep the same input objects, same corridor legality, same turn rule, same stop rule.
- Only change the output rendering: after a particle leaves column 0, every visited
  background cell on its route is painted color 6, so the full route becomes a
  visible magenta trail. The final position is still magenta as part of that trail.
- Non-background structure cells are never recolored. The start cell at column 0 is
  still cleared to background once the particle actually moves, because it is not a
  visited background cell.

Small execution trace:
- Suppose row r has a real corridor, the particle moves
  (r,0) -> (r,1) -> (r,2) -> (r,3), hits a wall at (r,4), turns up to (r-1,3),
  then continues to (r-2,3) and stops.
- Original output: set (r,0)=7 and (r-2,3)=6; cells (r,1),(r,2),(r,3),(r-1,3)
  stay 7.
- Variant output: set (r,0)=7 and paint visited background cells
  (r,1),(r,2),(r,3),(r-1,3),(r-2,3) to 6.

The only semantic change is therefore at the "trace result consumption" stage:
the solver must return the full path instead of only the terminal coordinate.

b) Complexity calibration
-------------------------
- The variant is intentionally minimal: movement logic is unchanged, so the task is
  still about understanding corridor following and 90-degree turns.
- Coloring the whole path in a single fixed color avoids introducing an extra color
  mapping rule, which would be a different dimension from D4.
- The generator already creates long paths (straight/L/U), so the new behavior is
  visually obvious without changing geometry. To avoid accidental near-original
  samples, generation additionally requires a minimum number of trail cells beyond
  the final endpoints.

c) Review of original assumptions in main()
-------------------------------------------
- _has_corridor(): assumes a traversable corridor cell is currently BG or PARTICLE.
  This remains valid because path painting happens only in the output copy; tracing
  still inspects the original input grid.
- _trace() in the original code assumes only the end position matters. That is no
  longer sufficient: if we kept returning only (end_r,end_c), all intermediate cells
  would be lost and the variant would silently collapse back to the base task.
- main(): `out[r,0] = BG; out[end_r,end_c] = PARTICLE` relied on the old invariant
  that exactly one new particle-colored cell should appear per moving particle.
  In the variant this write pattern is wrong, because many visited background cells
  must be colored. The replacement logic therefore paints each visited BG cell.
- Entrance detection and the "start only from genuine real corridors" filter still
  hold unchanged; the variant does not alter which particles are active.

d) Placement legality
---------------------
- The generator keeps the original corridor construction. Vertical corridors still
  reserve 3-column windows, L-turns still seal the non-turn side, and U-turn pairs
  still connect same-color channels only.
- Because output painting never feeds back into tracing, no new path-overlap or
  ambiguity constraints are needed beyond the original generator's legality checks.
- The new metric only inspects solver output; it does not mutate the placement logic.

e) Visual observability
-----------------------
- From 2-3 examples, a human now sees a clear difference: instead of a lone terminal
  magenta pixel, there is a continuous magenta snake exactly following the legal
  corridor route, including turns.
- No hidden numeric ordering or color-id trick is used. The visible rule is simply
  "the moving particle leaves a visible trail behind".

f) Visual cleanliness
---------------------
- Trail cells occupy only the 1-cell-wide corridor background, so they do not cover
  corridor walls or decorative bars.
- A single trail color keeps the output readable even when several particles move.
- Original generator spacing already prevents dense overlapping turn structures.

g) Color-change inferability
----------------------------
- There is no conditional recoloring by object identity or palette order.
- Every visited background cell becomes the same visible particle color 6, which is
  directly inferable from the before/after pairs.

New solver-side assumptions introduced here
-------------------------------------------
- We assume tracing is done against the original input grid, not the partially painted
  output. The implementation preserves this by calling _trace_path(g, ...), not on
  `out`.
- We assume every path cell that should be painted is background in the input. The
  original generator already guarantees this for all traversed cells except the start
  at column 0, which is intentionally excluded from path painting by not inserting it
  into the traced path list.
- We assume a moving path is long enough to make the D4 change visible. The generator
  now enforces a lower bound on "extra trail cells beyond endpoints".
"""


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars."""
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


def _trace_path(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    """Trace from (r,c) going right and return every visited non-start cell."""
    H, W = grid.shape
    direction = "right"
    path: list[tuple[int, int]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            path.append((r, c))
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
        path.append((r, c))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through the same corridors, but paint the whole trail."""
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
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        path = _trace_path(g, r, 0)
        if not path:
            continue
        out[r, 0] = BG
        for pr, pc in path:
            if int(g[pr, pc]) == BG:
                out[pr, pc] = PARTICLE

    return out


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
_MIN_EXTRA_TRAIL_CELLS = 12
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


def _painted_trail_stats(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """Return (moved_particles, extra_trail_cells_beyond_endpoints)."""
    moved = sum(
        1
        for r in range(inp.shape[0])
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    painted_bg = int(np.sum((inp == BG) & (out == PARTICLE)))
    return moved, painted_bg - moved


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D4 trail-painting variant."""
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

    moved, extra_trail = _painted_trail_stats(g, out)
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None
    if extra_trail < _MIN_EXTRA_TRAIL_CELLS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
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
    g,
    W,
    top,
    bot,
    row,
    color,
    c_v,
    direction,
    j_top,
    j_bot,
    bar_color,
    bar_start,
    extras,
    internal_stop,
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
