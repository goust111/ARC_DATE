# Combo variant of 9bbf930d (index 32)
# combo_spec: D13/1+D7/3+D20/1
# Name: 链式反应+擦除条对+Z形走廊

from __future__ import annotations

from collections import deque
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
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}

# ---------------------------------------------------------------------------
# Design analysis for combo D13/1 + D7/3 + D20/1
#
# a) Concrete behavior and trace
# - Base task: a seed in col 0 goes right through a same-color horizontal corridor,
#   turns only when a colored blocker forces a perpendicular move, and stops at an
#   edge or dead end. Output normally moves one particle once.
# - Fusion rule: each launch still follows the same local turn test, but now:
#   1. two-turn paths are Z-shaped (`right -> up/down -> right`) rather than U;
#   2. every traveled corridor cell erases its two flanking same-color wall cells;
#   3. when a launch ends, the endpoint keeps a particle and emits one more launch
#      in the opposite terminal direction.
# - Concrete trace on one band:
#   seed row goes right -> forced down through a Z shaft -> resumes right on a
#   lower same-color lane -> stops at the far edge. That first trip erases the
#   horizontal wall pair before the shaft, the shaft walls themselves, and the
#   post-shaft wall pair on the target lane. A new particle launches leftward
#   from the far-edge endpoint on the already-erased grid; because the old shaft
#   has become background, it may fail to re-enter that shaft and instead keep
#   going left or take a different intact turn. This is the key non-degenerate
#   interaction between chain reaction and erasure.
#
# b) Complexity calibration
# - A pure D13 queue on static geometry would often be invisible or would simply
#   ping-pong through mirrored gadgets. A pure D7 erase would only add holes.
# - Reusing the original random corridor grammar but replacing U with Z keeps the
#   visual language familiar, while the generator explicitly rejects samples where
#   dynamic erasure produces the same final output as a static chain reaction.
# - Only a few topology classes remain in play (straight, L, Z), repeated across
#   several rows per sample, so the rule stays inferable from 2-4 examples.
#
# c) Hidden assumptions audited from the original main()
# - Original assumption: each seed is independent. Invalid here; launches are now
#   FIFO-processed because endpoints spawn new launches on the updated grid.
# - Original assumption: the geometry can stay immutable while writing output.
#   Invalid here; erasure must hit the tracing grid before the next launch.
# - Original assumption: starts are transferred to endpoints. Invalid here; the
#   visible change is accumulation plus eaten walls, so col-0 seeds remain.
# - Original assumption: endpoint position is enough. Invalid here; the terminal
#   direction is also required to choose the emitted launch direction.
#
# d) Placement legality
# - The generator still reserves a 3-column window for each vertical shaft, which
#   keeps perpendicular turn choices unique before any erasure happens.
# - Z targets are same-color real lanes whose left entrance is deliberately broken
#   so they are destinations/continuations, not extra initial seeds.
# - Every produced candidate is validated by the fused solver itself; any sample
#   that creates an assert, no movement, or degenerate chain behavior is rejected.
#
# e) Visual observability
# - The output shows three repeated cues at once: extra particles accumulate, bar
#   pairs disappear exactly along traveled lanes, and at least one real path has
#   the distinctive offset-Z geometry rather than folding back leftward.
# - The generator additionally requires the dynamic fused output to differ from a
#   static no-erasure chain baseline, so the interaction is visible in examples.
#
# f) Visual cleanliness
# - The striped layout from the source task is preserved. Decorations stay sparse,
#   and all important structures are still one-cell-wide corridors bounded by long
#   bars, so the eaten-wall pattern remains readable instead of noisy.
#
# g) New solver assumptions and generator guarantees
# - `_trace` assumes at most one legal perpendicular turn at each blocker. The
#   shaft-spacing reservation and target-entrance breaking preserve that.
# - `_run_reaction` assumes launches eventually terminate. The launched-state set
#   prevents re-enqueuing identical `(cell, direction)` states, and already-filled
#   endpoints stop further spawning there.
# - The combo-specific assumption is that erasure matters. `generate_input()` now
#   enforces a positive interaction metric by comparing dynamic and static outputs.
# ---------------------------------------------------------------------------

# Generator configuration
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_Z_INTERNAL_STOP_PROB = 0.55

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_EATEN_WALLS = 24
_MIN_CHAIN_SPAWNED = 5
_MIN_CHAIN_DEPTH = 2
_MIN_INTERACTION_PIXELS = 6
_MAX_RETRIES = 500


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is passable and flanked by same-color bars for this direction."""
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
    """Return the two same-color wall cells flanking a traveled corridor cell."""
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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid same-color entrance cell for seed row r."""
    H, W = grid.shape
    if r <= 0 or r >= H - 1:
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
    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == color), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == color), W)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _seed_rows(grid: np.ndarray) -> list[int]:
    """Rows in col 0 that belong to genuine same-color entrances."""
    return [
        r
        for r in range(1, grid.shape[0] - 1)
        if int(grid[r, 0]) == PARTICLE and _find_entrance_col(grid, r) is not None
    ]


def _classify_turns(turns: list[tuple[str, str]]) -> str:
    """Classify an individual trace for generator-side behavior checks."""
    if not turns:
        return "straight"
    if turns in ([("right", "up")], [("right", "down")]):
        return "l"
    if turns in (
        [("right", "up"), ("up", "right")],
        [("right", "down"), ("down", "right")],
    ):
        return "z"
    if turns in (
        [("right", "up"), ("up", "left")],
        [("right", "down"), ("down", "left")],
    ):
        return "u"
    return f"{len(turns)}-turn"


def _trace(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[int, int, str, set[tuple[int, int]], list[tuple[str, str]]]:
    """Trace one launch on the current grid and collect erased wall cells."""
    H, W = grid.shape
    eaten: set[tuple[int, int]] = set()
    turns: list[tuple[str, str]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
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
        next_r, next_c, next_dir = options[0]
        turns.append((direction, next_dir))
        r, c, direction = next_r, next_c, next_dir
        walls = _corridor_walls(grid, r, c, direction)
        if walls is not None:
            eaten.update(walls)
    return r, c, direction, eaten, turns


def _run_reaction(grid: np.ndarray, dynamic_erase: bool) -> tuple[np.ndarray, dict[str, int]]:
    """Run the fused chain reaction; optionally let erasure mutate future tracing."""
    base = np.array(grid, dtype=int)
    geom = base.copy()
    out = base.copy()
    queue: deque[tuple[int, int, str, int]] = deque()
    launched: set[tuple[int, int, str]] = set()
    eaten_union: set[tuple[int, int]] = set()
    endpoints: list[tuple[int, int]] = []

    for r in _seed_rows(base):
        state = (r, 0, "right")
        queue.append((r, 0, "right", 0))
        launched.add(state)

    spawned = 0
    moved = 0
    max_depth = 0
    z_traces = 0

    while queue:
        r, c, direction, depth = queue.popleft()
        trace_grid = geom if dynamic_erase else base
        end_r, end_c, end_dir, eaten, turns = _trace(trace_grid, r, c, direction)
        if (end_r, end_c) == (r, c):
            continue

        moved += 1
        if _classify_turns(turns) == "z":
            z_traces += 1

        for rr, cc in eaten:
            out[rr, cc] = BG
            eaten_union.add((rr, cc))
            if dynamic_erase:
                geom[rr, cc] = BG

        if int(out[end_r, end_c]) == PARTICLE:
            continue

        out[end_r, end_c] = PARTICLE
        if dynamic_erase:
            geom[end_r, end_c] = PARTICLE
        endpoints.append((end_r, end_c))
        spawned += 1
        max_depth = max(max_depth, depth + 1)

        next_state = (end_r, end_c, _OPPOSITE[end_dir])
        if next_state not in launched:
            launched.add(next_state)
            queue.append((end_r, end_c, _OPPOSITE[end_dir], depth + 1))

    return out, {
        "seed_count": len(_seed_rows(base)),
        "moved": moved,
        "spawned": spawned,
        "max_depth": max_depth,
        "eaten_walls": len(eaten_union),
        "z_traces": z_traces,
        "final_particles": int(np.sum(out == PARTICLE)),
        "endpoints": len(endpoints),
    }


def _base_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Summarize the original visible seed traces before any dynamic erasure."""
    g = np.array(grid, dtype=int)
    metrics = {"moved": 0, "straight": 0, "l": 0, "z": 0, "u": 0, "other": 0}
    for r in _seed_rows(g):
        end_r, end_c, _, _, turns = _trace(g, r, 0, "right")
        if (end_r, end_c) == (r, 0):
            continue
        metrics["moved"] += 1
        kind = _classify_turns(turns)
        if kind in metrics:
            metrics[kind] += 1
        else:
            metrics["other"] += 1
    return metrics


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Run the fused solver and expose generator-side validation metrics."""
    dynamic_out, dynamic_stats = _run_reaction(grid, dynamic_erase=True)
    static_out, static_stats = _run_reaction(grid, dynamic_erase=False)
    dynamic_stats = dict(dynamic_stats)
    dynamic_stats["interaction_pixels"] = int(np.count_nonzero(dynamic_out != static_out))
    dynamic_stats["spawn_delta_vs_static"] = dynamic_stats["spawned"] - static_stats["spawned"]
    return dynamic_out, dynamic_stats


def main(grid: np.ndarray) -> np.ndarray:
    """Run a dynamic chain reaction through same-color corridors, erasing consumed walls.

    Input structure:
    - bg=7 and particle=6. Column 0 contains candidate seed particles.
    - Even rows carry colored bars; odd rows are horizontal channels.
    - Real two-turn corridors are Z-shaped: a right-moving lane shifts to another
      same-color lane and then continues right on the far side of the shaft.

    Rule:
    - Valid seeds are the col-0 particles whose row has a genuine same-color entrance.
    - Process launches in FIFO order. Each launch traces on the current grid using
      the original local movement rule: move straight through BG/PARTICLE cells and
      only turn 90 degrees when a colored blocker leaves exactly one legal same-color
      perpendicular corridor.
    - Every traveled corridor cell erases its flanking same-color wall pair to BG.
      After a launch stops, its endpoint keeps a particle and emits one new launch in
      the opposite terminal direction, unless that endpoint already contained a particle.

    Key insight:
    - The erased walls are not just an output trail. They mutate the corridor network
      before the next launch, so later chain links may no longer be able to reuse the
      same Z shaft and therefore reroute or terminate differently.
    """
    out, _ = _solve_with_stats(grid)
    return out


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for one vertical shaft."""
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


def _preferred_break_side(target: int, direction: str, channel_spec: list[str]) -> str:
    """Choose which target bar row to break at the left entrance."""
    n_channels = len(channel_spec)
    preferred = "top" if direction == "up" else "bottom"
    top_safe = target == 0 or channel_spec[target - 1] == "fake"
    bottom_safe = target == n_channels - 1 or channel_spec[target + 1] == "fake"
    if preferred == "top":
        if top_safe:
            return "top"
        if bottom_safe:
            return "bottom"
    else:
        if bottom_safe:
            return "bottom"
        if top_safe:
            return "top"
    return preferred


def _plan_z_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
    channel_spec: list[str],
) -> int:
    """Pair same-color real channels into right-moving Z turns."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)

    candidates: list[tuple[int, int, str, str]] = []
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        pool = chans[:]
        random.shuffle(pool)
        while len(pool) >= 2:
            a = pool.pop()
            b = pool.pop()
            upper, lower = sorted((a, b))
            up_break = _preferred_break_side(upper, "up", channel_spec)
            down_break = _preferred_break_side(lower, "down", channel_spec)
            if (up_break == "top") and (down_break != "bottom"):
                candidates.append((lower, upper, "up", up_break))
            elif (down_break == "bottom") and (up_break != "top"):
                candidates.append((upper, lower, "down", down_break))
            else:
                if random.random() < 0.5:
                    candidates.append((lower, upper, "up", up_break))
                else:
                    candidates.append((upper, lower, "down", down_break))

    if not candidates:
        return 0

    random.shuffle(candidates)
    max_pairs = min(len(candidates), max(1, len(real_indices) - 3))
    n_pairs = random.randint(1, max_pairs)
    built = 0
    for source, target, direction, break_side in candidates:
        if built >= n_pairs:
            break
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            break
        plans[source] = {
            "kind": "z",
            "target": target,
            "direction": direction,
            "c_v": c_v,
            "break_side": break_side,
            "internal_stop": random.random() < _Z_INTERNAL_STOP_PROB,
        }
        plans[target] = {"kind": "z_target"}
        built += 1
    return built


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Assign straight or L paths to real channels not used by a Z turn."""
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


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build the original-family L turn."""
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
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_z_turn(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_start: int,
    extras: list[int],
    break_side: str,
    internal_stop: bool,
    wall_cols: set[int],
) -> None:
    """Build a right-moving Z turn between same-color channels."""
    g[row, c_v] = color

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = color
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = color
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        if break_side == "top":
            g[j_top, bar_start] = random.choice(other_colors)
        else:
            g[j_bot, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        candidates = [c for c in range(c_v + 3, W) if c not in wall_cols]
        if candidates:
            g[j_row, random.choice(candidates)] = color


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned path gadgets onto the striped canvas."""
    wall_cols = {int(plan["c_v"]) for plan in plans.values() if "c_v" in plan}
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "z_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "z":
            target = int(plan["target"])
            j_top, j_bot = bar_rows[target], bar_rows[target + 1]
            _build_z_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_start,
                extras,
                str(plan["break_side"]),
                bool(plan.get("internal_stop", False)),
                wall_cols,
            )


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused combo variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
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
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    z_count = _plan_z_turns(
        real_indices,
        bar_rows,
        bar_color,
        bar_start,
        W,
        used_vcols,
        plans,
        channel_spec,
    )
    if z_count < 1:
        return None

    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if int(g[row, c_decor]) == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out, stats = _solve_with_stats(g)
        base = _base_behavior_metrics(g)
    except Exception:
        return None

    if not (_MIN_MOVED <= base["moved"] <= _MAX_MOVED):
        return None
    if base["z"] < 1 or base["u"] != 0 or base["other"] != 0:
        return None
    if stats["spawned"] < _MIN_CHAIN_SPAWNED:
        return None
    if stats["max_depth"] < _MIN_CHAIN_DEPTH:
        return None
    if stats["eaten_walls"] < _MIN_EATEN_WALLS:
        return None
    if stats["interaction_pixels"] < _MIN_INTERACTION_PIXELS:
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
