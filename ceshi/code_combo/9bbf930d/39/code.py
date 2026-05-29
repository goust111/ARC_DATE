# Combo variant of 9bbf930d (index 39)
# combo_spec: D13/0+D7/1+D20/1
# Name: 逐个顺序处理+终点改色为走廊色+Z形走廊

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) concrete_behavior_and_trace:
#   - Input keeps the striped layout of the base task: even rows are colored bars,
#     odd rows are channel rows, col 0 is filled with particles, col 1 is bg.
#   - The geometric 2-turn motif is now Z-shaped: a particle goes right, turns
#     vertically, then turns right again on an offset target channel.
#   - Processing is sequential from top to bottom. After a particle settles, its
#     endpoint is immediately written back into the working grid, so later particles
#     see it as a real colored obstacle.
#   - The settled color is not magenta: it is the color of the terminal corridor
#     in the current working grid. Therefore a first particle entering a Z target
#     row can leave behind a colored endpoint that visually merges with that row.
#   - Concrete trace:
#       source A (higher row) Z-turns down into target row T and reaches T's
#       rightmost free cell. The solver writes that cell with T's corridor color.
#       source B (lower row) later Z-turns into the same T. In the original task,
#       A and B would both independently end at the same far-right cell. In this
#       combo, B reaches the cell just left of A because A's recolored endpoint is
#       already a blocking colored cell in the work grid.
#
# b) complexity_calibration:
#   - The fusion stays in one object vocabulary: no new symbols, no hidden ordering,
#     only a changed execution schedule, a changed endpoint color, and a new 2-turn
#     shape. The core visible motifs repeat across examples.
#   - To keep the fusion observable, the generator forces a special two-source
#     Z-merge cluster. This prevents degeneration into "just recolor endpoints" or
#     "just add a Z path with no order dependence".
#   - Extra real channels are limited to straight/L paths so the grids stay legible.
#
# c) implicit_assumption_audit_of_main:
#   - Original `_has_corridor` accepted PARTICLE as traversable. That breaks the
#     sequential rule, so here only BG can be entered; settled colored endpoints are
#     solid obstacles.
#   - Original tracing on the immutable input assumed particles do not interact. The
#     combo must trace on a mutable work grid, otherwise the cascade disappears.
#   - Original endpoint writing used a constant magenta color, which no longer holds.
#     The combo must recover the terminal corridor color from the current work grid.
#   - Entrance legality must still be checked on the original structure, not on the
#     mutated work grid, so that a later source is still considered a valid source
#     even after earlier particles have altered nearby destinations.
#   - With Z paths, the second turn is only unambiguous if the target channel's left
#     entrance is broken and the unused branch is sealed. The generator guarantees
#     this so `_trace_path` can keep the same "at most one turn option" assert.
#
# d) placement_legality:
#   - Each vertical shaft reserves a 3-column window around its wall column, so
#     separate shafts do not touch and create ambiguous side turns.
#   - The forced target channel is isolated by fake channels above and below; this
#     lets the generator break the target's entrance without damaging another real
#     source and keeps the shared Z tail visually clean.
#   - Special sources use distinct shaft columns and the target row's own entrance
#     is broken, so only the intended particles can enter the shared tail.
#
# e) visual_observability:
#   - Humans can read the rule from a few examples because all three changes are
#     visible: Z geometry, top-to-bottom order dependence, and endpoints matching
#     corridor color rather than staying magenta.
#   - The forced cluster makes the most diagnostic pattern repeat: the higher source
#     occupies the far-right tail cell, the lower source stops one cell earlier, and
#     both settled cells take the target corridor color.
#
# f) visual_cleanliness:
#   - No stacked markers are used. The combo effect is expressed by ordinary colored
#     cells inside existing corridors, so the output remains sparse and structured.
#   - Fake decorations stay light, and the isolated target row avoids crowded overlap
#     around the special interaction region.
#
# g) color_change_inferability:
#   - Endpoint recolor is determined by visible local structure only: the matched bar
#     pair around the final corridor cell in the current work grid.
#   - The generator rejects cases where all moved particles recolor to one color, so
#     the "endpoint becomes corridor color" rule is not hidden behind a constant.
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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a BG cell flanked by a same-color bar pair."""
    H, W = grid.shape
    if int(grid[r, c]) != BG:
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
    """Return the color of the visible corridor containing the endpoint cell."""
    H, W = grid.shape

    if direction in {"left", "right"} and 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            return a
    if direction in {"up", "down"} and 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            return a

    candidates: list[int] = []
    if 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    if 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    assert len(set(candidates)) == 1, "final position is not on a unique corridor"
    return int(candidates[0])


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first legal real-corridor entrance on row r, or None."""
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
    return int(entrance_col)


def _is_real_channel(grid: np.ndarray, r: int) -> bool:
    return _find_entrance_col(grid, r) is not None


def _trace_path(grid: np.ndarray, r: int, c: int) -> tuple[int, int, str, list[tuple[str, str]]]:
    """Trace one particle on the current grid and record turn transitions."""
    H, W = grid.shape
    direction = "right"
    turns: list[tuple[str, str]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) == BG:
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
        next_r, next_c, next_dir = options[0]
        turns.append((direction, next_dir))
        r, c, direction = next_r, next_c, next_dir
    return r, c, direction, turns


def _classify_turns(turns: list[tuple[str, str]]) -> str:
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


def _simulate(
    grid: np.ndarray,
    *,
    sequential: bool,
    settled_mode: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Shared engine for combo solver and comparison baselines."""
    assert settled_mode in {"corridor", "particle"}
    base = np.array(grid, dtype=int)
    H, _ = base.shape
    out = base.copy()
    work = base.copy()
    moves: list[dict[str, object]] = []

    for r in range(1, H - 1):
        if int(work[r, 0]) != PARTICLE:
            continue
        if not _is_real_channel(base, r):
            continue
        trace_grid = work if sequential else base
        end_r, end_c, end_dir, turns = _trace_path(trace_grid, r, 0)
        moved = (end_r, end_c) != (r, 0)
        color = None
        if moved:
            color = _corridor_color(trace_grid, end_r, end_c, end_dir)
        moves.append(
            {
                "row": r,
                "end": (end_r, end_c),
                "moved": moved,
                "dir": end_dir,
                "turn_kind": _classify_turns(turns),
                "color": color,
            }
        )
        if not moved:
            continue

        settled_value = color if settled_mode == "corridor" else PARTICLE
        if sequential:
            work[r, 0] = BG
            work[end_r, end_c] = int(settled_value)
        else:
            out[r, 0] = BG
            out[end_r, end_c] = int(settled_value)

    return (work if sequential else out), moves


def _parallel_combo_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, sequential=False, settled_mode="corridor")[0]


def _sequential_particle_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, sequential=True, settled_mode="particle")[0]


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Summarize visible path topologies on the unmodified input."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    metrics = {"moved": 0, "straight": 0, "l": 0, "z": 0, "u": 0, "other": 0}
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if not _is_real_channel(g, r):
            continue
        end_r, end_c, _end_dir, turns = _trace_path(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
        metrics["moved"] += 1
        kind = _classify_turns(turns)
        if kind in metrics:
            metrics[kind] += 1
        else:
            metrics["other"] += 1
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """Sequentially move particles through real corridors and recolor settled endpoints.

    Input structure:
    - bg=7 and particles=6 fill col 0; col 1 is bg; even rows hold colored bars and odd rows hold channels.
    - Real corridors are BG cells flanked by same-color bar pairs; fake corridors use mismatched colors.
    - Two-turn real corridors may be Z-shaped, so a path can go right, then vertical, then right again.

    Rule:
    - Process source rows from top to bottom.
    - On the current working grid, a particle moves through BG only; when blocked by a colored cell, it may turn 90 degrees into the unique perpendicular BG cell whose flanking bars match.
    - If it moves, erase its start and write its final cell with the color of the corridor it ends in. That recolored cell remains in the working grid and can block or mislead later particles.

    Key insight:
    - The output is not a parallel overlay. Earlier particles permanently rewrite the maze with corridor-colored endpoints, so later particles can be stopped by cells that visually blend into the Z-shaped corridor system.
    """
    return _simulate(grid, sequential=True, settled_mode="corridor")[0]


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.18
_L_INTERNAL_PROB = 0.32

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_ENDPOINT_COLOR_DIVERSITY = 2
_MAX_RETRIES = 500


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


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Assign only straight or L plans to non-special real channels."""
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
    """Build an L-turn with the standard sealed-corner geometry."""
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
) -> None:
    """Build a Z path: right -> vertical -> right on an offset target channel."""
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
    """Render all planned corridors onto the grid."""
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
            )


def _pick_special_cluster(n_channels: int) -> dict[str, object]:
    """Pick 5 channels: source0, source1, bridge fake, target real, guard fake."""
    target = random.randint(3, n_channels - 2)
    return {
        "sources": [target - 3, target - 2],
        "bridge": target - 1,
        "target": target,
        "guard": target + 1,
    }


def _interaction_metrics(
    g: np.ndarray,
    bar_rows: list[int],
    special: dict[str, object],
) -> dict[str, object]:
    seq_out, seq_moves = _simulate(g, sequential=True, settled_mode="corridor")
    par_out, par_moves = _simulate(g, sequential=False, settled_mode="corridor")
    seq_particle_out, seq_particle_moves = _simulate(g, sequential=True, settled_mode="particle")

    seq_by_row = {int(m["row"]): m for m in seq_moves}
    par_by_row = {int(m["row"]): m for m in par_moves}
    seq_particle_by_row = {int(m["row"]): m for m in seq_particle_moves}

    source_rows = [bar_rows[i] + 1 for i in special["sources"]]
    target_row = bar_rows[int(special["target"])] + 1
    interaction_rows = sorted(r for r in seq_by_row if seq_by_row[r]["end"] != par_by_row[r]["end"])

    endpoint_colors = [int(m["color"]) for m in seq_moves if m["moved"]]
    return {
        "seq_out": seq_out,
        "par_out": par_out,
        "seq_particle_out": seq_particle_out,
        "seq_by_row": seq_by_row,
        "par_by_row": par_by_row,
        "seq_particle_by_row": seq_particle_by_row,
        "source_rows": source_rows,
        "target_row": target_row,
        "interaction_rows": interaction_rows,
        "endpoint_colors": endpoint_colors,
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the combo variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    special = _pick_special_cluster(n_channels)
    sources = list(special["sources"])
    bridge = int(special["bridge"])
    target = int(special["target"])
    guard = int(special["guard"])

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels - 2)

    forced_real = set(sources + [target])
    forced_fake = {bridge, guard}
    if n_real < len(forced_real):
        n_real = len(forced_real)

    pool = [i for i in range(n_channels) if i not in forced_real and i not in forced_fake]
    extra_needed = n_real - len(forced_real)
    extra_reals = set(random.sample(pool, extra_needed))
    real_set = forced_real | extra_reals
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    channel_spec[bridge] = "fake"
    channel_spec[guard] = "fake"

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
        g[br, bar_start:W] = bar_color[br]

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {target: {"kind": "z_target"}}

    special_cols: list[int] = []
    for _ in sources:
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            return None
        special_cols.append(c_v)
    special_cols.sort()
    for src, c_v in zip(sources, special_cols):
        plans[src] = {
            "kind": "z",
            "target": target,
            "direction": "down",
            "c_v": c_v,
            "break_side": "bottom",
        }

    remaining_reals = [i for i, s in enumerate(channel_spec) if s == "real" and i not in plans]
    _plan_remaining(remaining_reals, bar_start, W, used_vcols, plans)
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
        seq_out = main(g)
        par_out = _parallel_combo_main(g)
        seq_particle_out = _sequential_particle_main(g)
    except Exception:
        return None

    metrics = _behavior_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["z"] < 2 or metrics["u"] != 0 or metrics["other"] != 0:
        return None

    interaction = _interaction_metrics(g, bar_rows, special)
    source_rows = interaction["source_rows"]
    seq_by_row = interaction["seq_by_row"]
    par_by_row = interaction["par_by_row"]
    seq_particle_by_row = interaction["seq_particle_by_row"]
    src0, src1 = source_rows
    if src0 not in seq_by_row or src1 not in seq_by_row:
        return None

    first_seq = seq_by_row[src0]
    second_seq = seq_by_row[src1]
    first_par = par_by_row[src0]
    second_par = par_by_row[src1]
    second_seq_particle = seq_particle_by_row[src1]

    if not all(bool(m["moved"]) for m in (first_seq, second_seq, first_par, second_par, second_seq_particle)):
        return None
    if first_par["turn_kind"] != "z" or second_par["turn_kind"] != "z":
        return None
    if tuple(first_par["end"]) != tuple(second_par["end"]):
        return None
    if tuple(first_seq["end"]) != tuple(first_par["end"]):
        return None
    if tuple(second_seq["end"])[0] != int(interaction["target_row"]):
        return None
    if tuple(second_seq["end"])[1] != tuple(second_par["end"])[1] - 1:
        return None
    if tuple(second_seq_particle["end"]) != tuple(second_seq["end"]):
        return None
    if first_seq["color"] is None or second_seq["color"] is None:
        return None
    if int(first_seq["color"]) != int(second_seq["color"]):
        return None

    target_row = int(interaction["target_row"])
    far_c = int(first_seq["end"][1])
    near_c = int(second_seq["end"][1])
    target_color = int(first_seq["color"])
    if far_c - near_c != 1:
        return None
    if int(seq_out[target_row, far_c]) != target_color or int(seq_out[target_row, near_c]) != target_color:
        return None
    if np.array_equal(seq_out, par_out):
        return None
    if np.array_equal(seq_out, seq_particle_out):
        return None
    if not interaction["interaction_rows"]:
        return None
    if len(set(interaction["endpoint_colors"])) < _MIN_ENDPOINT_COLOR_DIVERSITY:
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
