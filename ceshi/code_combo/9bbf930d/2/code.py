# Combo variant of 9bbf930d (index 2)
# combo_spec: D4/0+D7/3
# Name: 连续轨迹+擦除走廊条对

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
Design analysis for combo D4/0 + D7/3
======================================

a) Concrete behavior and traced example
---------------------------------------
Base task:
- A magenta particle starts at (r, 0), moves right through BG cells, and turns 90
  degrees only when a perpendicular same-color corridor is available.
- The original output clears the start and keeps only one magenta endpoint.

Combo behavior:
- Keep the exact same motion rule and the same real/fake corridor distinction.
- Change only the write-back stage in two synchronized ways:
  1. Every visited BG path cell becomes magenta, so the full route is visible.
  2. For every occupied corridor cell, the two colored wall cells that define that
     local corridor are erased to BG.

Concrete L-turn trace:
- Suppose the particle moves
  (r,0) -> (r,1) -> (r,2) -> (r,3), finds a blocker at (r,4), turns up into
  (r-1,3), then continues to (r-2,3) and stops.
- Base output:
  clear (r,0), color only (r-2,3) with PARTICLE.
- D4-only output:
  clear (r,0), paint (r,1),(r,2),(r,3),(r-1,3),(r-2,3) magenta.
- D7-only output:
  clear (r,0), keep only endpoint magenta, and erase the wall pairs flanking those
  occupied corridor cells.
- Combo output:
  clear (r,0), paint the same visited BG cells magenta, and also erase the wall
  pairs flanking those same occupied corridor cells. The walked corridor therefore
  disappears and is replaced by a magenta trail line.

b) Complexity calibration
-------------------------
- The geometry is unchanged, so the core reasoning remains "follow same-color
  corridors and turn only at legal junctions".
- The combo is visually stronger than either single variant: magenta trail and wall
  disappearance reinforce each other along the same route.
- To avoid underpowered samples that look too close to the base task, generation
  requires both a minimum number of trail cells beyond endpoints and a minimum number
  of erased wall cells.
- No extra color mapping is introduced: all newly painted cells use the particle
  color, and erased walls simply become BG.

c) Review of implicit assumptions in main()
-------------------------------------------
- `_has_corridor()` still assumes traversable cells are BG or PARTICLE in the input.
  This remains valid because tracing always reads the untouched input grid.
- The base `_trace()` only returned an endpoint. That assumption breaks here because
  the combo needs the full occupied route plus direction-specific wall pairs.
- The base `out[r,0] = BG; out[end_r,end_c] = PARTICLE` assumed exactly one visible
  particle-colored write per moving particle. That is false now: every visited BG
  path cell must be colored, while wall cells along the same route are erased.
- The entrance filter assumes active particles start on genuine same-color horizontal
  corridors whose bars begin at column 2 or 3. The generator still guarantees this.
- New combo-specific assumption: erasures must be computed from the travel direction
  at the moment the particle enters a cell. The implementation therefore records
  walls during tracing rather than inferring them afterward from the painted output.

d) Placement legality
---------------------
- The generator keeps the original corridor grammar: striped rows, same-color real
  channels, mismatched fake channels, reserved vertical windows, sealed L-turns, and
  U-turn connections only between same-color channels.
- Because output mutations never feed back into tracing, erasing walls in one route
  cannot create new legal turns for another route.
- Erasure only targets already existing colored wall cells returned by
  `_corridor_walls()`, so it cannot go out of bounds or delete arbitrary structure.

e) Visual observability
-----------------------
- Across a few examples, a human can see that visited corridors are not merely
  endpoints: the route becomes a continuous magenta snake.
- At the same time, the colored rails that used to bound that route vanish exactly
  along the snake, which makes the combo distinguishable from both the base task and
  the D4-only trail-painting variant.
- The rule is structural and repeated: "where the particle goes, magenta appears in
  the channel and the corresponding corridor bars disappear".

f) Visual cleanliness
---------------------
- The two effects live on complementary cell sets: trail on corridor path cells,
  erasure on the adjacent wall cells. They therefore do not overwrite each other.
- Unvisited corridors remain fully intact, so the output preserves clear contrast
  between transformed and untouched regions.
- The existing generator spacing for vertical corridors avoids dense overlapping
  edits in the same small area.

g) Inferability of the changes
------------------------------
- Only two color changes occur and both are highly constrained:
  visited BG -> PARTICLE, and traversed wall color -> BG.
- No hidden ranking by color id or object order is used; the changed cells are fully
  determined by visible corridor occupancy.
- Because both effects follow the same route, 2-3 training pairs are enough to infer
  that the magenta path and the erased rails are coupled manifestations of motion.
"""


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color corridor bars."""
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


def _corridor_walls(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the two colored wall cells flanking (r,c) in the travel direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = (r - 1, c), (r + 1, c)
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = (r, c - 1), (r, c + 1)
    va, vb = int(grid[a]), int(grid[b])
    if va != vb or va in {PARTICLE, BG}:
        return None
    return a, b


def _find_real_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first genuine horizontal corridor entrance for row r, else None."""
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


def _trace_path_and_erasures(
    grid: np.ndarray, r: int, c: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Trace from (r,c) and return visited non-start cells plus corridor wall cells."""
    H, W = grid.shape
    direction = "right"
    path: list[tuple[int, int]] = []
    eaten: set[tuple[int, int]] = set()

    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            path.append((r, c))
            walls = _corridor_walls(grid, r, c, direction)
            if walls is not None:
                eaten.update(walls)
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
        walls = _corridor_walls(grid, r, c, direction)
        if walls is not None:
            eaten.update(walls)

    return path, eaten


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the combo rule and expose metrics used by the generator."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    moved = 0
    trail_cells: set[tuple[int, int]] = set()
    eaten_union: set[tuple[int, int]] = set()

    for r in range(1, H - 1):
        if _find_real_entrance_col(g, r) is None:
            continue

        path, eaten = _trace_path_and_erasures(g, r, 0)
        if not path:
            continue

        moved += 1
        out[r, 0] = BG
        eaten_union.update(eaten)
        for pr, pc in path:
            if int(g[pr, pc]) == BG:
                out[pr, pc] = PARTICLE
                trail_cells.add((pr, pc))

    for rr, cc in eaten_union:
        out[rr, cc] = BG

    return out, {
        "moved": moved,
        "trail_cells": len(trail_cells),
        "extra_trail_cells": len(trail_cells) - moved,
        "eaten_walls": len(eaten_union),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles and replace traversed corridors with a fused trail result.

    Input structure:
    - bg=7 and particle=6.
    - Column 0 is filled with particles; true horizontal corridors live on odd rows
      between same-color bars on the adjacent even rows.
    - Some real corridors include legal vertical turn corridors; fake corridors use
      mismatched bar colors and must be ignored.

    Rule:
    - Each active particle starts at column 0 and travels right through BG cells.
    - When blocked by a colored cell, it may turn 90 degrees only into a perpendicular
      BG cell flanked by same-color bars. If no such turn exists, it stops.
    - Every visited BG path cell becomes magenta, producing a continuous trail.
    - Every traversed corridor cell also erases its two flanking corridor-bar cells to BG.
    - Unvisited structure remains unchanged; the start cell clears if the particle moves.

    Key insight:
    - The motion logic is unchanged from the base task. The combo only changes how the
      traced route is rendered: the path itself becomes magenta while the corridor rails
      that justified each step disappear along that same route.
    """
    out, _ = _solve_with_stats(grid)
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
_MIN_EATEN_WALLS = 18
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


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the combo variant."""
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

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
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
        out, stats = _solve_with_stats(g)
    except Exception:
        return None

    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["extra_trail_cells"] < _MIN_EXTRA_TRAIL_CELLS:
        return None
    if stats["eaten_walls"] < _MIN_EATEN_WALLS:
        return None
    if np.array_equal(out, g):
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
