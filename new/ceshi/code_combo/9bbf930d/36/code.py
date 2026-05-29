# Combo variant of 9bbf930d (index 36)
# combo_spec: D4/0+D6/0+D7/0
# Name: 连续轨迹+目标色停止+终点扩散

from __future__ import annotations

import random
import numpy as np

BG = 7
PARTICLE = 6
TARGET = 8
_TRAVERSABLE = {PARTICLE, BG, TARGET}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_BURST_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

"""
Design analysis for combo D4/0 + D6/0 + D7/0
=============================================

a) Concrete behavior and traced example
---------------------------------------
Base task:
- A magenta particle starts at (r,0), enters a genuine same-color corridor, goes
  straight through background cells, turns only when a colored blocker exposes one
  valid perpendicular corridor, and stops at an edge or dead end.
- Output clears the start and leaves only one terminal magenta pixel.

Combo behavior:
- Keep the same corridor legality and turn mechanism.
- Add one fixed target color pixel inside each moving real path. TARGET is
  traversable, but entering it ends motion immediately.
- Every visited non-start traversable cell is painted magenta in the output, so
  the full route becomes visible.
- After stopping, paint a 4-neighbor cross centered on the stop cell. Because the
  stop is typically a target, the target dot is consumed and covered by the burst.

Example trace:
- Natural path without combo:
  (r,0)->(r,1)->(r,2)->(r,3)->turn up->(r-1,3)->(r-2,3)->edge stop.
- Put TARGET at (r-1,3).
- Base output would still show only one final pixel, and if TARGET were ignored the
  particle would continue to (r-2,3).
- Combo output clears (r,0), paints magenta on
  (r,1),(r,2),(r,3),(r-1,3), then adds a cross on
  {(r-1,3),(r-2,3),(r,3),(r-1,2),(r-1,4)}.

The visible effect is therefore not a loose sum of three edits: the trail now ends
exactly at the inserted target, and that stopping site becomes the center of the
terminal cross.

b) Complexity calibration
-------------------------
- The target rule uses exactly one reserved color, not per-row or per-color target
  meanings. That keeps the new stop condition inferable.
- The burst shape is always a cross, not a per-sample choice, so humans can infer it
  from 2-4 examples.
- Targets are placed on later interior cells of real paths, which guarantees a clear
  pre-target trail and avoids tiny or degenerate outputs that would look too close to
  the base task.
- Burst centers are kept apart, so multiple final marks remain visually separable.

c) Hidden-assumption audit of the original solver
-------------------------------------------------
- Original straight-step logic assumed only BG and PARTICLE are traversable. The
  combo breaks that, so TARGET must be added to the traversable set everywhere that
  movement legality is checked.
- Original tracing returned only a terminal coordinate. That is insufficient once the
  output must paint the whole route, so tracing must return the visited path.
- Original `main()` mutated output per particle with one endpoint write. That would
  lose intermediate trail cells and under-render the new terminal mark. The combo
  therefore separates three stages: clear starts, paint trails, then paint bursts.
- Original entrance detection scanned for early BG corridor cells. If a target were
  placed in the entrance segment, a real corridor could become invisible to the
  solver. The generator must therefore forbid targets near the entrance.
- Multiple-target interactions are a new risk: a target on a crossing shared by two
  paths could make both particles stop there. The generator avoids this by selecting
  target cells that belong to exactly one moving path.

d) Placement legality
---------------------
- Targets are chosen only from already validated natural paths of moving real
  particles.
- Candidate targets must be interior enough to support a full 5-cell burst and must
  not share burst cells with any previously chosen target.
- Targets never overlap bars, decorators, starts, or each other, because candidates
  are required to be BG cells before insertion.
- The underlying corridor generator is unchanged apart from reserving TARGET as a
  non-bar color, so turn legality and the no-ambiguous-turn assert remain valid.

e) Visual observability
-----------------------
- Inputs visibly contain a repeated special target color that appears only as isolated
  dots on real traversable routes.
- Outputs visibly replace each dot with the end of a magenta trail and a small cross.
- Because targets are placed on later path cells, examples show a clear "particle
  would have gone farther, but it stops at the dot" pattern.

f) Visual cleanliness
---------------------
- A 1-cell trail plus a 5-cell burst is compact enough to preserve corridor geometry.
- Keeping burst centers disjoint avoids unreadable endpoint clusters.
- Using the same particle color for trail and burst makes the result look like one
  coherent motion trace rather than several unrelated markings.

g) Change inferability
----------------------
- No rule depends on numeric color ordering or latent metadata.
- TARGET is globally fixed and never reused for corridor bars, so humans can infer its
  role from its repeated placement and disappearance.
- The output color change is structural: visited route cells and the target-centered
  cross both become the existing particle color.

New solver assumptions and generator guarantees
-----------------------------------------------
- Assumption: the combo solver traces against the original input, not the partially
  painted output. Guarantee: all rendering helpers read from `g` and write to `out`.
- Assumption: each moving path has room for a visible trail before the target and for
  a full cross around the stop. Guarantee: target selection enforces interior, late,
  non-overlapping candidate cells.
- Assumption: burst overlap is not needed to explain the rule. Guarantee: target
  selection rejects overlapping burst footprints.
"""


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal entrance cell for row r, if any."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in _TRAVERSABLE:
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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars."""
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
    return a == b and a not in _TRAVERSABLE


def _trace_path(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    stop_on_target: bool,
) -> list[tuple[int, int]]:
    """Trace from (r,c), returning the full visited path including the start."""
    H, W = grid.shape
    direction = "right"
    path = [(r, c)]
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        cell = int(grid[nr, nc])
        if cell in _TRAVERSABLE:
            r, c = nr, nc
            path.append((r, c))
            if stop_on_target and cell == TARGET:
                break
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
        if stop_on_target and int(grid[r, c]) == TARGET:
            break
    return path


def _collect_moves(grid: np.ndarray, *, stop_on_target: bool) -> list[tuple[int, list[tuple[int, int]]]]:
    """Collect `(start_row, path)` for particles that truly move."""
    H, _ = grid.shape
    moves: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(1, H - 1):
        if _find_entrance_col(grid, r) is None:
            continue
        path = _trace_path(grid, r, 0, stop_on_target=stop_on_target)
        if len(path) > 1:
            moves.append((r, path))
    return moves


def _burst_cells(H: int, W: int, r: int, c: int) -> list[tuple[int, int]]:
    """Return the in-bounds cross centered on (r, c)."""
    return [
        (rr, cc)
        for dr, dc in _BURST_DELTAS
        if 0 <= (rr := r + dr) < H and 0 <= (cc := c + dc) < W
    ]


def _render_trail_only(
    grid: np.ndarray,
    moves: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    """Render start clearing plus full trail, but without the terminal burst."""
    out = np.array(grid, dtype=int, copy=True)
    for start_r, path in moves:
        out[start_r, 0] = BG
        for rr, cc in path[1:]:
            if int(grid[rr, cc]) in {BG, TARGET}:
                out[rr, cc] = PARTICLE
    return out


def _render_combo_output(
    grid: np.ndarray,
    moves: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    """Render cleared starts, continuous trails, and terminal bursts."""
    out = _render_trail_only(grid, moves)
    H, W = out.shape
    for _, path in moves:
        end_r, end_c = path[-1]
        for rr, cc in _burst_cells(H, W, end_r, end_c):
            out[rr, cc] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through corridors, stop on targets, and mark the full stop site.

    Input structure:
    - bg=7, particle=6, target=8. Particles start in col 0, col 1 stays bg, and
      same-color bars on even rows define real horizontal and vertical corridors.

    Rule:
    - A valid particle enters only a genuine same-color corridor, moves through bg or
      target cells, and turns 90 degrees only into the unique perpendicular corridor
      that is flanked by matching same-color bars.
    - Entering a target cell stops the motion immediately; otherwise the particle stops
      only at the natural edge/dead-end defined by the original corridor logic.
    - Every visited non-start traversable cell is painted particle color as a visible
      trail, and then the final stop cell expands to a 4-neighbor cross.

    Key insight:
    - The navigation geometry is still the original corridor-following task; the combo
      only changes which cells count as traversable/stopping cells and how the traced
      path is rendered in the output.
    """
    g = np.array(grid, dtype=int)
    moves = _collect_moves(g, stop_on_target=True)
    return _render_combo_output(g, moves)


# --------------- generator (sa_max_effort) ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 3  # reserved TARGET keeps total palette at 6

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

# Variant validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_TARGET_STOPS = 3
_MIN_EXTRA_TRAIL_CELLS = 12
_MIN_BURST_GAIN = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
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


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Pair same-color real channels into U-turn plans."""
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


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
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


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
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
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
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


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in {"l_edge", "l_internal"}:
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        else:
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
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
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _choose_target_cells(g: np.ndarray) -> dict[int, tuple[int, int]] | None:
    """Pick one safe interior target cell on each moving real path."""
    H, W = g.shape
    natural_moves = _collect_moves(g, stop_on_target=False)
    if len(natural_moves) < _MIN_TARGET_STOPS:
        return None

    path_visit_count: dict[tuple[int, int], int] = {}
    natural_paths = {start_r: path for start_r, path in natural_moves}
    for _, path in natural_moves:
        for cell in path[1:]:
            path_visit_count[cell] = path_visit_count.get(cell, 0) + 1

    target_cells: dict[int, tuple[int, int]] = {}
    used_burst_cells: set[tuple[int, int]] = set()
    for start_r, path in natural_moves:
        entrance_col = _find_entrance_col(g, start_r)
        if entrance_col is None or len(path) < 8:
            return None
        min_idx = max(entrance_col + 2, len(path) // 2)
        max_idx = len(path) - 2
        if min_idx > max_idx:
            return None

        candidates = []
        for idx in range(min_idx, max_idx + 1):
            rr, cc = path[idx]
            if int(g[rr, cc]) != BG:
                continue
            if path_visit_count[(rr, cc)] != 1:
                continue
            if not (0 < rr < H - 1 and 2 < cc < W - 2):
                continue
            burst = set(_burst_cells(H, W, rr, cc))
            if len(burst) != 5 or burst & used_burst_cells:
                continue
            candidates.append((rr, cc, burst))
        if not candidates:
            return None

        rr, cc, burst = random.choice(candidates)
        target_cells[start_r] = (rr, cc)
        used_burst_cells.update(burst)

    if len(target_cells) != len(natural_paths):
        return None
    return target_cells


def _variant_stats(g: np.ndarray) -> dict[str, int]:
    """Quantify whether target-stop, trail, and burst effects are all active."""
    natural_moves = _collect_moves(g, stop_on_target=False)
    combo_moves = _collect_moves(g, stop_on_target=True)
    natural_end = {start_r: path[-1] for start_r, path in natural_moves}

    moved = len(combo_moves)
    target_stops = 0
    shortened = 0
    for start_r, path in combo_moves:
        end_cell = path[-1]
        if int(g[end_cell]) == TARGET:
            target_stops += 1
        if natural_end.get(start_r) != end_cell:
            shortened += 1

    trail_out = _render_trail_only(g, combo_moves)
    combo_out = _render_combo_output(g, combo_moves)
    new_particle_cells = int(np.sum((trail_out == PARTICLE) & (g != PARTICLE)))
    burst_gain = int(np.sum((combo_out == PARTICLE) & (trail_out != PARTICLE)))
    return {
        "moved": moved,
        "target_stops": target_stops,
        "shortened": shortened,
        "extra_trail_cells": new_particle_cells - moved,
        "burst_gain": burst_gain,
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused trail+target-stop+burst variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE, TARGET)], _N_EXTRA_COLORS)

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
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
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
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        target_cells = _choose_target_cells(g)
    except Exception:
        return None
    if target_cells is None:
        return None
    for rr, cc in target_cells.values():
        g[rr, cc] = TARGET

    try:
        out = main(g)
        stats = _variant_stats(g)
    except Exception:
        return None
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["target_stops"] < _MIN_TARGET_STOPS:
        return None
    if stats["shortened"] < _MIN_TARGET_STOPS:
        return None
    if stats["extra_trail_cells"] < _MIN_EXTRA_TRAIL_CELLS:
        return None
    if stats["burst_gain"] < _MIN_BURST_GAIN:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if int(np.sum(g == TARGET)) != stats["moved"]:
        return None
    if not np.any(out == PARTICLE):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
