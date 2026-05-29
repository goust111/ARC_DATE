# Combo variant of 9bbf930d (index 10)
# combo_spec: D12/1+D13/0
# Name: 粒子互相阻挡+逐个顺序处理

from __future__ import annotations

import random
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D12/1 + D13/0
#
# a) concrete_behavior_and_trace:
#   - Visual language stays close to the original task: even rows are colored bars,
#     odd rows are corridors, and magenta particles still start from col 0.
#   - The fused change is not "simultaneous collisions" and not "independent tracing":
#     particles are processed one by one from top to bottom, and every settled particle
#     remains on the working grid as a real obstacle for later particles.
#   - The generator forces a visible three-source cascade:
#       source A (highest row) \
#       source B              ---> all bend downward into one shared target tail
#       source C (lowest)    /
#     A is geometrically the longest path, C is the shortest path, but A is processed
#     first and therefore claims the far-right endpoint anyway. B and C then stop one
#     cell earlier in a contiguous queue because they hit already-settled particles.
#   - Tiny trace:
#       original independent solver: A, B, C all compute on the pristine grid and all
#       land on the same terminal cell, so the output only shows one survivor there.
#       fused solver: process A first -> endpoint E becomes occupied; process B next ->
#       B reaches E-1 and stops; process C last -> C reaches E-2 and stops.
#     The visible output is a right-side magenta queue ordered by source row.
#
# b) complexity_calibration:
#   - The interaction is limited to one forced cascade cluster of exactly three sources.
#     This is strong enough to be obvious, but still small enough to read visually.
#   - Extra real channels may exist, but the forced cluster is always present and is the
#     main explanatory pattern across examples.
#   - The target tail is clean and horizontal, so the blocking effect is legible without
#     requiring humans to reason about hidden timing or simultaneous updates.
#
# c) implicit_assumption_audit_of_original_main:
#   - Original assumption: PARTICLE cells are traversable just like BG.
#     Broken now. If later particles could enter settled magenta cells, the sequential
#     dependency would disappear. Therefore the combo solver only allows BG as an open
#     cell on the working grid.
#   - Original assumption: each row can be traced independently on the static input grid.
#     Broken now. Later particles depend on earlier outcomes, so tracing must run on a
#     mutable work grid that is updated after every successful move.
#   - Original assumption: writing only to the final output image is enough.
#     Broken now. The end position must be written back immediately, otherwise later
#     particles would not see the obstacle.
#   - Original assumption: entrance detection may read the same grid that tracing uses.
#     Adjusted now. "Is this row a real corridor?" is evaluated on the immutable base
#     layout, not on the mutable work grid, so a blocked particle still gets a turn to
#     move and is merely stopped by an obstacle rather than silently skipped.
#   - New combo interaction between D12 and D13:
#     D12 contributes "particles can obstruct other particles";
#     D13 contributes "the obstruction order is top-to-bottom, not arrival-time-based".
#     The generator therefore verifies a case where the highest source has the longest
#     geometric path but still occupies the best endpoint, proving the order rule matters.
#
# d) placement_legality:
#   - The forced cluster uses three source channels, one fake spacer channel, and one
#     target channel below them. Distinct vertical shafts are reserved with a 3-column
#     exclusion window so turn gadgets never overlap.
#   - The existing L/U builders are reused because they already guarantee in-bounds
#     carving and unique turn availability when the local seals are respected.
#   - The target channel entrance is intentionally broken so its own starting particle
#     does not move and interfere with the forced cascade semantics.
#
# e) visual_observability:
#   - The processing order is fully visible from geometry: "higher row goes first".
#     No color-value ordering or invisible metadata is used.
#   - The queue appears on one obvious tail row near the right side, so 2-4 examples are
#     enough to infer that later particles are blocked by earlier settled ones.
#   - The strongest clue is deliberate: path length and processing order disagree.
#     The shortest-path source does not win; the topmost source wins.
#
# f) visual_tidiness:
#   - Only one shared-tail interaction cluster is forced, keeping the grid readable.
#   - A fake spacer channel separates the source band from the target channel so the
#     target entrance break does not visually or structurally corrupt neighboring sources.
#   - Decorative fake cells are sparse and never placed on occupied turn gadgets.
#
# g) inference_of_change_rule:
#   - There is no color remapping in this combo, only movement. The inference burden is
#     therefore entirely structural: identify real corridors, notice the top-to-bottom
#     processing order, and track how already-settled magenta cells block later paths.
#   - The generator rejects samples where the combo output matches the original parallel
#     baseline for the forced source rows, because such samples would visually degenerate.
#
# new_solver_assumptions_checked_by_generator:
#   - The three forced source rows are real movable corridors in the base grid.
#   - All three forced sources independently target the same terminal cell in the
#     original static-grid baseline.
#   - Under the combo rule, those three rows end on the target row as a consecutive
#     queue, ordered from top source to bottom source.
#   - The independent path lengths of those three sources are strictly decreasing from
#     top to bottom, so the observed queue cannot be explained by "shortest path wins".
# ---------------------------------------------------------------------------

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _is_open_value(value: int, particle_open: bool) -> bool:
    return value == BG or (particle_open and value == PARTICLE)


def _has_corridor(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    *,
    particle_open: bool,
) -> bool:
    """True if (r,c) is an enterable corridor cell flanked by same-color bars."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if not _is_open_value(int(grid[r, c]), particle_open):
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    particle_open: bool,
) -> tuple[int, int, int]:
    """Trace one particle on the given grid and return (end_r, end_c, steps)."""
    h, w = grid.shape
    direction = "right"
    steps = 0
    for _ in range(h * w * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if _is_open_value(int(grid[nr, nc]), particle_open):
            r, c = nr, nc
            steps += 1
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir, particle_open=particle_open):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        steps += 1
    return r, c, steps


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    h, w = grid.shape
    if r <= 0 or r >= h - 1:
        return None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _is_real_channel(grid: np.ndarray, r: int) -> bool:
    w = grid.shape[1]
    entrance_col = _find_entrance_col(grid, r)
    if entrance_col is None:
        return False
    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(w) if int(grid[r - 1, c]) == color), w)
    bar_below = next((c for c in range(w) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])), w)
    return bar_above == bar_below and bar_above <= 3


def _simulate(
    grid: np.ndarray,
    *,
    sequential: bool,
    particle_open: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Common engine for the combo solver and the original independent baseline."""
    base = np.array(grid, dtype=int)
    h, _ = base.shape
    out = base.copy()
    work = base.copy()
    moves: list[dict[str, Any]] = []

    for r in range(1, h - 1):
        if int(base[r, 0]) != PARTICLE:
            continue
        if not _is_real_channel(base, r):
            continue

        trace_grid = work if sequential else base
        end_r, end_c, steps = _trace(trace_grid, r, 0, particle_open=particle_open)
        moved = (end_r, end_c) != (r, 0)
        moves.append({"row": r, "end": (end_r, end_c), "moved": moved, "steps": steps})
        if not moved:
            continue

        if sequential:
            work[r, 0] = BG
            work[end_r, end_c] = PARTICLE
        else:
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return (work if sequential else out), moves


def _parallel_main(grid: np.ndarray) -> np.ndarray:
    """Original-style independent baseline used only for generator validation."""
    return _simulate(grid, sequential=False, particle_open=True)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through corridors with sequential blocking.

    Input structure:
    - bg=7 and particle=6. Even rows hold colored bars, odd rows hold corridor cells.
    - Visible particles begin in col 0; only rows with a real same-color corridor are movable.

    Rule:
    - Process movable particles from top to bottom.
    - Each particle follows the original rightward / 90-degree turn rule on the current
      working grid, but settled particle cells are now solid obstacles, not pass-through.
    - After a particle stops, its final position is written back immediately and can block
      all later particles.

    Key insight:
    - The answer is not computed independently per row: earlier final positions become new
      obstacles, so the visible row order determines how a shared tail fills from right to left.
    """
    return _simulate(np.array(grid, dtype=int), sequential=True, particle_open=False)[0]


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 500


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
    plans: dict[int, dict[str, object]],
) -> None:
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
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


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
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
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
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
                    cands = list(range(bot + 2, h - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, h, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
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


def _pick_special_cluster(n_channels: int) -> dict[str, object]:
    target = random.randint(4, n_channels - 1)
    return {
        "sources": [target - 4, target - 3, target - 2],
        "spacer": target - 1,
        "target": target,
    }


def _max_consecutive(cols: list[int]) -> int:
    if not cols:
        return 0
    best = 1
    run = 1
    for i in range(1, len(cols)):
        if cols[i] == cols[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _interaction_metrics(
    g: np.ndarray,
    bar_rows: list[int],
    special: dict[str, object],
) -> dict[str, object]:
    seq_out, seq_moves = _simulate(g, sequential=True, particle_open=False)
    par_out, par_moves = _simulate(g, sequential=False, particle_open=True)
    seq_by_row = {int(m["row"]): m for m in seq_moves}
    par_by_row = {int(m["row"]): m for m in par_moves}
    interaction_rows = sorted(r for r in seq_by_row if seq_by_row[r]["end"] != par_by_row[r]["end"])
    source_channels = list(special["sources"])
    source_rows = [bar_rows[i] + 1 for i in source_channels]
    target = int(special["target"])
    target_row = bar_rows[target] + 1
    tail_cols = [c for c in range(2, g.shape[1]) if int(seq_out[target_row, c]) == PARTICLE]
    path_lengths = [int(par_by_row[r]["steps"]) for r in source_rows if r in par_by_row]
    return {
        "seq_out": seq_out,
        "par_out": par_out,
        "seq_by_row": seq_by_row,
        "par_by_row": par_by_row,
        "interaction_rows": interaction_rows,
        "target_row": target_row,
        "source_rows": source_rows,
        "tail_cols": tail_cols,
        "path_lengths": path_lengths,
        "max_queue": _max_consecutive(sorted(tail_cols)),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1

    special = _pick_special_cluster(n_channels)
    sources = list(special["sources"])
    spacer = int(special["spacer"])
    target = int(special["target"])

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels - 1)
    forced_real = set(sources + [target])
    forced_fake = {spacer}
    pool = [i for i in range(n_channels) if i not in forced_real and i not in forced_fake]
    if n_real < len(forced_real):
        n_real = len(forced_real)
    extra_needed = n_real - len(forced_real)
    if extra_needed > len(pool):
        return None
    extra_reals = set(random.sample(pool, extra_needed))
    real_set = forced_real | extra_reals
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    channel_spec[spacer] = "fake"

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
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

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {target: {"kind": "u_target"}}

    special_cols = []
    for _ in sources:
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            return None
        special_cols.append(c_v)
    special_cols.sort()
    for src, c_v in zip(sources, special_cols):
        plans[src] = {
            "kind": "u",
            "target": target,
            "direction": "down",
            "c_v": c_v,
            "internal_stop": False,
        }

    remaining_reals = [i for i, s in enumerate(channel_spec) if s == "real" and i not in plans]
    _plan_u_turns(remaining_reals, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_remaining(remaining_reals, bar_start, w, used_vcols, plans)
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
        seq_out = main(g)
        par_out = _parallel_main(g)
    except Exception:
        return None

    metrics = _interaction_metrics(g, bar_rows, special)
    seq_by_row = metrics["seq_by_row"]
    par_by_row = metrics["par_by_row"]
    source_rows = list(metrics["source_rows"])
    target_row = int(metrics["target_row"])

    if any(r not in seq_by_row or r not in par_by_row for r in source_rows):
        return None

    special_seq = [seq_by_row[r] for r in source_rows]
    special_par = [par_by_row[r] for r in source_rows]
    if not all(bool(m["moved"]) for m in special_seq + special_par):
        return None

    par_ends = [tuple(m["end"]) for m in special_par]
    seq_ends = [tuple(m["end"]) for m in special_seq]
    if len(set(par_ends)) != 1:
        return None
    if any(end[0] != target_row for end in par_ends):
        return None
    if any(end[0] != target_row for end in seq_ends):
        return None

    seq_cols = [end[1] for end in seq_ends]
    par_col = par_ends[0][1]
    if seq_cols[0] != par_col:
        return None
    if seq_cols != list(range(par_col, par_col - len(source_rows), -1)):
        return None
    if seq_cols[1] == par_col:
        return None

    path_lengths = list(metrics["path_lengths"])
    if len(path_lengths) != len(source_rows):
        return None
    if not all(path_lengths[i] > path_lengths[i + 1] for i in range(len(path_lengths) - 1)):
        return None

    if metrics["max_queue"] < len(source_rows):
        return None
    if sum(1 for r in source_rows if seq_by_row[r]["end"] != par_by_row[r]["end"]) < 2:
        return None
    if np.array_equal(seq_out, par_out):
        return None
    if not metrics["interaction_rows"]:
        return None

    n_moved = sum(1 for m in metrics["seq_by_row"].values() if bool(m["moved"]))
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
