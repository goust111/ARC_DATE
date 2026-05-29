# Combo variant of 9bbf930d (index 27)
# combo_spec: D12/1+D13/0+D4/0
# Name: 顺序处理+互相阻挡+连续轨迹

from __future__ import annotations

import random
from typing import Any

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


# Design analysis for combo D12/1 + D13/0 + D4/0
# ------------------------------------------------
# a) Concrete behavior and traced example
#    - Input remains the striped 9bbf930d field: even rows are colored bars, odd rows are
#      channel rows, col 0 holds magenta particles, col 1 is background.
#    - Original task: each movable particle is traced independently on the static input; only
#      its final cell is written, so shared tails do not interact and no trail is visible.
#    - Combo behavior:
#      1. process particles from top to bottom;
#      2. when one particle finishes, only its endpoint is inserted into the working grid as a
#         real obstacle for later particles;
#      3. every moved particle paints all visited background cells onto the output, so the
#         visible result is a set of full magenta trajectories.
#    - Small trace:
#      * A (higher row) and B (lower row) both turn into the same target row T.
#      * Original: A and B independently finish at the far-right stop cell of T; output still
#        shows just one endpoint there.
#      * Combo: A is processed first and occupies that stop cell in the working grid. B later
#        enters T, advances right, sees A directly ahead, and either turns at that blocker if a
#        legal perpendicular corridor exists or stops one cell earlier. Both A and B leave full
#        trails in the output.
#
# b) Complexity calibration
#    - The interaction is constrained to one clean shared-tail cluster with two forced source
#      rows, plus a few additional ordinary real corridors. This keeps the new rule obvious
#      without turning every sample into a dense traffic jam.
#    - Only one visible ordering principle exists: top-to-bottom row order. No color-id or
#      hidden timestamp is used.
#    - Trail painting uses a single fixed color (magenta), avoiding extra mapping burden.
#
# c) Implicit assumption audit of the original main()
#    - Original _has_corridor accepted PARTICLE as traversable; that breaks sequential blocking,
#      so the combo version allows only BG corridor cells on the working grid.
#    - Original _trace returned only a final coordinate; the combo needs the full path so output
#      can show continuous trails.
#    - Original main traced every particle on the immutable input and wrote to an output copy;
#      that assumes independence. The combo must maintain a separate working grid whose only
#      dynamic additions are settled endpoints.
#    - Painting trails directly into the working grid would create false obstacles and make
#      later particles block on earlier trails, which is not the intended fusion. Therefore the
#      working grid and painted output must stay separate.
#    - Real-channel detection must still use the base input structure, not the mutated working
#      grid, otherwise an occupied endpoint could incorrectly suppress a later particle before it
#      even attempts to move.
#
# d) Placement legality
#    - The generator reuses the original L/U corridor builders, which already enforce non-
#      ambiguous turns and in-bounds vertical shafts.
#    - A special cluster reserves two source channels, one fake spacer channel, and one target
#      channel. The spacer prevents the target-entrance break from corrupting the neighboring
#      source entrance.
#    - The two special sources enter the same target tail through distinct vertical columns, so
#      their gadgets do not overlap and the arrival order stays geometry-driven.
#
# e) Visual observability
#    - Across examples, a solver can see three stable cues: multiple particles are processed,
#      their full paths are painted, and a later path is visibly shortened or redirected near an
#      earlier endpoint on the same tail.
#    - Because the blocker is a visible magenta endpoint and the order is visible by row
#      position, the sequential rule is inferable from 2-3 examples.
#
# f) Visual cleanliness
#    - The forced interaction is localized to one target row; the rest of the board keeps normal
#      straight/L/U examples for contrast.
#    - Trails occupy corridor background cells only and do not overwrite colored bars, so the
#      route geometry remains readable.
#
# g) Inferability of changes
#    - There is no conditional recoloring beyond "visited background becomes magenta".
#    - The generator rejects degenerate samples where the sequential result matches the parallel
#      result on the special rows or where the painted trails are too short to be noticeable.
#
# New solver-side assumptions introduced here and guaranteed by the generator
#    - At least one shared-tail pair exists whose parallel endpoints coincide but whose
#      sequential endpoints differ by one step on the target row.
#    - The special interaction stays acyclic; no head-on swaps or simultaneous motion logic is
#      needed because particles move one by one.
#    - Any visible trail cell was background in the base input. Only settled endpoints become
#      dynamic obstacles.


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != BG:
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


def _trace_path(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    """Trace one particle on the current working grid and return every visited non-start cell."""
    h, w = grid.shape
    direction = "right"
    path: list[tuple[int, int]] = []
    for _ in range(h * w * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) == BG:
            r, c = nr, nc
            path.append((r, c))
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))
    return path


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first same-color corridor entrance for row r, or None."""
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
    """Check whether row r is a genuine movable corridor in the base grid."""
    w = grid.shape[1]
    entrance_col = _find_entrance_col(grid, r)
    if entrance_col is None:
        return False
    above_color = int(grid[r - 1, entrance_col])
    below_color = int(grid[r + 1, entrance_col])
    bar_above = next((c for c in range(w) if int(grid[r - 1, c]) == above_color), w)
    bar_below = next((c for c in range(w) if int(grid[r + 1, c]) == below_color), w)
    return above_color == below_color and bar_above == bar_below and bar_above <= 3


def _paint_path(out: np.ndarray, base: np.ndarray, path: list[tuple[int, int]]) -> None:
    """Paint visited background cells in the visible output."""
    for pr, pc in path:
        if int(base[pr, pc]) == BG:
            out[pr, pc] = PARTICLE


def _simulate(
    grid: np.ndarray,
    *,
    sequential: bool,
    paint_trails: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Common engine for sequential/parallel and endpoint-only/trail rendering."""
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
        path = _trace_path(trace_grid, r, 0)
        moved = bool(path)
        end = path[-1] if moved else (r, 0)
        moves.append({"row": r, "path": path, "end": end, "moved": moved})
        if not moved:
            continue

        if sequential:
            end_r, end_c = end
            work[r, 0] = BG
            work[end_r, end_c] = PARTICLE

        out[r, 0] = BG
        if paint_trails:
            _paint_path(out, base, path)
        else:
            end_r, end_c = end
            out[end_r, end_c] = PARTICLE

    return (work if sequential and not paint_trails else out), moves


def _parallel_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, sequential=False, paint_trails=True)[0]


def _endpoint_only_parallel_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, sequential=False, paint_trails=False)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Process particles top-to-bottom, block later ones with settled endpoints, and paint trails.

    Input structure:
    - bg=7 and particle=6. Column 0 contains visible particles; column 1 is bg.
    - Even rows carry colored bars; odd rows are candidate corridor rows.
    - A real corridor is a bg route flanked by same-color bars, with optional L/U turns.

    Rule:
    - Scan movable particles from top to bottom.
    - Trace each particle on a working grid where only previously settled endpoints remain as
      new particle-colored obstacles.
    - When a particle moves, clear its start cell, paint every visited background cell magenta in
      the output, and keep its final cell occupied so later particles may be truncated or turned.

    Key insight:
    - The fusion separates dynamics from rendering: trails are only visual evidence in the output,
      while endpoint occupancy alone drives the sequential blocking behavior.
    """
    return _simulate(grid, sequential=True, paint_trails=True)[0]


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
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
_MIN_EXTRA_TRAIL_CELLS = 12
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


def _pick_special_cluster(n_channels: int) -> dict[str, object]:
    hub = random.randint(3, n_channels - 2)
    return {"sources": [hub - 3, hub - 2], "spacer": hub - 1, "target": hub}


def _trail_stats(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    moved = sum(
        1
        for r in range(inp.shape[0])
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    painted_bg = int(np.sum((inp == BG) & (out == PARTICLE)))
    return moved, painted_bg - moved


def _fusion_metrics(
    g: np.ndarray,
    special: dict[str, object],
    bar_rows: list[int],
) -> dict[str, Any]:
    seq_out, seq_moves = _simulate(g, sequential=True, paint_trails=True)
    par_out, par_moves = _simulate(g, sequential=False, paint_trails=True)
    seq_by_row = {int(m["row"]): m for m in seq_moves}
    par_by_row = {int(m["row"]): m for m in par_moves}
    interaction_rows = sorted(r for r in seq_by_row if tuple(seq_by_row[r]["end"]) != tuple(par_by_row[r]["end"]))
    moved, extra_trail = _trail_stats(g, seq_out)

    source_channels = [int(x) for x in special["sources"]]
    source_rows = [bar_rows[i] + 1 for i in source_channels]
    target_row = bar_rows[int(special["target"])] + 1
    painted_on_target = int(np.sum((g[target_row] == BG) & (seq_out[target_row] == PARTICLE)))

    return {
        "seq_out": seq_out,
        "par_out": par_out,
        "seq_by_row": seq_by_row,
        "par_by_row": par_by_row,
        "interaction_rows": interaction_rows,
        "source_rows": source_rows,
        "target_row": target_row,
        "moved": moved,
        "extra_trail": extra_trail,
        "painted_on_target": painted_on_target,
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
    sources = [int(x) for x in special["sources"]]
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
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w), min(n_decor, w - bar_start - 1))
        for c_decor in cols:
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        seq_out = main(g)
        _parallel_main(g)
        _endpoint_only_parallel_main(g)
    except Exception:
        return None

    metrics = _fusion_metrics(g, special, bar_rows)
    seq_by_row = metrics["seq_by_row"]
    par_by_row = metrics["par_by_row"]
    src0, src1 = metrics["source_rows"]
    if src0 not in seq_by_row or src1 not in seq_by_row:
        return None

    first_seq = seq_by_row[src0]
    second_seq = seq_by_row[src1]
    first_par = par_by_row[src0]
    second_par = par_by_row[src1]

    if not all(bool(m["moved"]) for m in (first_seq, second_seq, first_par, second_par)):
        return None
    if tuple(first_par["end"]) != tuple(second_par["end"]):
        return None
    if tuple(first_seq["end"]) != tuple(first_par["end"]):
        return None
    if tuple(second_seq["end"]) == tuple(second_par["end"]):
        return None
    if tuple(second_seq["end"])[0] != int(metrics["target_row"]):
        return None
    if tuple(second_seq["end"])[1] != tuple(second_par["end"])[1] - 1:
        return None
    if not metrics["interaction_rows"]:
        return None
    if metrics["painted_on_target"] < 3:
        return None

    if not (_MIN_MOVED <= int(metrics["moved"]) <= _MAX_MOVED):
        return None
    if int(metrics["extra_trail"]) < _MIN_EXTRA_TRAIL_CELLS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if np.array_equal(g, seq_out):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
