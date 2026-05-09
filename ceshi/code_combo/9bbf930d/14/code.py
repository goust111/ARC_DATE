# Combo variant of 9bbf930d (index 14)
# combo_spec: D5/1+D15/1
# Name: 粒子位于走廊内部+方向由相邻走廊决定

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
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}

# Design analysis for combo D5/1 + D15/1
# ---------------------------------------
# fused_rule:
#   old:
#     - every potentially moving particle sits on the left border in col 0
#     - a valid row is detected by searching for a horizontal same-color corridor
#     - once accepted, the launch direction is always "right"
#   new:
#     - particles are sparse interior objects, never boundary launchers
#     - each particle is placed on a real corridor cell somewhere along a traced path
#     - the initial heading is inferred locally by comparing the available corridor
#       continuation lengths in the neighboring cardinal directions; the unique
#       longest corridor arm wins
#   concrete_visual_form:
#     - the striped bar/corridor language from the original task is preserved
#     - the full magenta left column disappears; col 0 becomes ordinary background
#     - every particle sits inside a corridor, often not centered on its local
#       straight segment, so one side of that segment visibly extends farther
#     - some starts are horizontal and some are vertical, making the "not always
#       rightward" change visually obvious without introducing extra symbols
#   behavior_trace_example:
#     - original:
#       1. particle at (5, 0) enters row 5 and moves right
#       2. at the blocker near col 12 it turns upward, then stops at the edge
#     - fused:
#       1. remove the edge wall of particles; place one particle at (9, 11), already
#          on the vertical part of a real corridor
#       2. from that cell, the upward straight corridor arm has length 4 while the
#          downward arm has length 1, so the start direction is inferred as "up"
#       3. after that first choice, tracing uses the original move-straight / turn-90°
#          rule unchanged until the particle reaches its final cell
#   complexity_calibration:
#     - the fusion is visible in two independent ways: starts are interior, and the
#       chosen direction varies with local corridor geometry instead of being fixed
#     - the cue is still low-entropy and human-readable: compare a few corridor-arm
#       lengths, rather than infer any hidden color ordering or multi-step logic
#     - generator validation rejects ties, so the launch cue remains crisp
#   implicit_assumptions_review:
#     - `if g[r, 0] != PARTICLE`: invalid, because starts are no longer confined to
#       the left edge
#     - entrance search from the border: invalid, because the particle is already
#       embedded in the corridor network
#     - `_trace(..., direction="right")`: invalid, because starts may launch up/down
#       as well as right
#     - `out[r, 0] = BG`: invalid, because the source cell is arbitrary
#     - a new assumption appears: the unique longest adjacent corridor arm determines
#       the start direction. The generator guarantees this by selecting only cells
#       with a strict longest arm and validating every finished grid with `main()`.
#   placement_legality:
#     - starts are chosen from traced real-path cells that are strictly interior and
#       have at least two opposite traversable directions on the same corridor axis
#     - one particle is placed per traced real path, and the whole grid is rechecked
#       after all placements so overlaps / ambiguities are rejected
#     - no extra blockers are inserted; the underlying corridor topology remains legal
#   observability:
#     - from a few examples, a human can spot that magenta particles begin inside
#       corridors and move toward whichever side visibly offers the longer hallway
#     - requiring at least one vertical launch direction prevents collapse back to
#       the original "always go right" behavior
#   cleanliness:
#     - the grid still contains only background, sparse particles, corridor bars,
#       and a few existing distractors; there is no new overlay or annotation
#     - starts avoid crossroads with 4 equally plausible exits, keeping local cues clean


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars perpendicular to direction."""
    h, w = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
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


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    """Trace one particle starting from (r, c) with an explicit initial direction."""
    h, w = grid.shape
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _trace_with_history(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[list[tuple[int, int]], list[str]]:
    """Trace one particle and record visited cells plus the direction of each move."""
    h, w = grid.shape
    path = [(r, c)]
    dirs: list[str] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            path.append((r, c))
            dirs.append(direction)
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))
        dirs.append(direction)
    return path, dirs


def _corridor_run_length(grid: np.ndarray, r: int, c: int, direction: str) -> int:
    """Count visible straight hallway cells from a structurally valid launch cell."""
    if not _has_corridor(grid, r, c, direction):
        return 0
    h, w = grid.shape
    dr, dc = _DELTAS[direction]
    rr, cc = r, c
    length = 0
    while True:
        nr, nc = rr + dr, cc + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) not in {PARTICLE, BG}:
            break
        length += 1
        rr, cc = nr, nc
    return length


def _direction_lengths(grid: np.ndarray, r: int, c: int) -> dict[str, int]:
    """Return launchable directions and their visible straight-corridor lengths."""
    lengths = {}
    for direction in ("left", "right", "up", "down"):
        run = _corridor_run_length(grid, r, c, direction)
        if run > 0:
            lengths[direction] = run
    return lengths


def _find_initial_direction(grid: np.ndarray, r: int, c: int) -> str | None:
    """Choose the unique longest adjacent corridor arm around an internal particle."""
    if int(grid[r, c]) != PARTICLE:
        return None
    lengths = _direction_lengths(grid, r, c)
    if not lengths:
        return None
    best = max(lengths.values())
    winners = [direction for direction, run in lengths.items() if run == best]
    if len(winners) != 1:
        return None
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move internal particles according to the longest adjacent corridor arm.

    Input structure:
    - bg=7 and particle=6. The striped colored-bar corridor system is unchanged.
    - Particles are sparse interior cells placed directly inside real corridors, not on
      a fixed boundary column.

    Rule:
    - For each particle, inspect the launchable corridor directions touching its cell.
    - The initial direction is the unique direction whose adjacent straight corridor arm
      extends farther than the others.
    - After that launch choice, move straight through bg/particle cells; when blocked by
      a colored cell, turn 90 degrees into the unique perpendicular valid corridor.
    - Stop at the grid edge or when no valid turn exists.

    Key insight:
    - The particle no longer tells you its heading by placement on the left edge. Instead,
      the heading is encoded geometrically by unequal corridor continuation lengths around
      the particle, so the first inference is local and the rest is the original tracing rule.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    starts = [(r, c) for r, c in zip(*np.where(g == PARTICLE))]

    for r, c in starts:
        direction = _find_initial_direction(g, r, c)
        if direction is None:
            continue
        end_r, end_c = _trace(g, r, c, direction)
        if (end_r, end_c) != (r, c):
            out[r, c] = BG
            out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

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
_MAX_RETRIES = 300
_MIN_DIRECTION_MARGIN = 2
_PLACEMENT_RETRIES = 40


def _find_legacy_source_rows(grid: np.ndarray) -> list[int]:
    """Rows that the original edge-based solver would accept as real starts."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    rows = []
    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next(
            (c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), w
        )
        bar_below = next(
            (c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), w
        )
        if bar_above != bar_below or bar_above > 3:
            continue
        rows.append(r)
    return rows


def _solve_from_edge(grid: np.ndarray) -> np.ndarray:
    """Original solver behavior, used only to generate canonical base layouts."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for r in _find_legacy_source_rows(g):
        end_r, end_c = _trace(g, r, 0, "right")
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans):
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


def _plan_remaining(real_indices, bar_start, w, used_vcols, plans):
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


def _build_l_turn(g, h, top, bot, row, color, c_v, direction, bar_color, stop_at):
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
    g, w, top, bot, row, color, c_v, direction, j_top, j_bot, bar_color, bar_start, extras, internal_stop
):
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


def _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, extras):
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
                    cands = list(range(bot + 2, h - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, h, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                w,
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


def _count_moved(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.sum((inp == PARTICLE) & (out != PARTICLE)))


def _gen_base_attempt() -> np.ndarray | None:
    """Generate one valid original-style instance before internalizing the starts."""
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
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
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, w, used_vcols, plans)
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
        out = _solve_from_edge(g)
    except Exception:
        return None
    n_moved = _count_moved(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def _start_candidates_for_path(
    grid: np.ndarray, path: list[tuple[int, int]], dirs: list[str]
) -> list[dict[str, object]]:
    """Collect legal interior starts whose launch direction is the unique longest arm."""
    h, w = grid.shape
    candidates: list[dict[str, object]] = []
    for i in range(1, len(path) - 1):
        r, c = path[i]
        if r in {0, h - 1} or c in {0, w - 1}:
            continue
        lengths = _direction_lengths(grid, r, c)
        if len(lengths) != 2:
            continue
        forward = dirs[i]
        backward = _OPPOSITE[forward]
        if forward not in lengths or backward not in lengths:
            continue
        if set(lengths) != {forward, backward}:
            continue
        margin = int(lengths[forward]) - int(lengths[backward])
        if margin < _MIN_DIRECTION_MARGIN:
            continue
        if lengths[forward] < 2 or lengths[backward] < 1:
            continue
        axis = "vertical" if forward in {"up", "down"} else "horizontal"
        candidates.append(
            {
                "cell": (r, c),
                "direction": forward,
                "axis": axis,
                "margin": margin,
                "forward_len": int(lengths[forward]),
                "backward_len": int(lengths[backward]),
            }
        )
    return candidates


def _convert_to_combo_variant(base: np.ndarray) -> np.ndarray | None:
    """Replace edge starts with interior starts inferred from unequal corridor arms."""
    paths: list[tuple[list[tuple[int, int]], list[str]]] = []
    path_candidates: list[list[dict[str, object]]] = []
    for r in _find_legacy_source_rows(base):
        path, dirs = _trace_with_history(base, r, 0, "right")
        if len(path) <= 1:
            continue
        candidates = _start_candidates_for_path(base, path, dirs)
        if not candidates:
            return None
        paths.append((path, dirs))
        path_candidates.append(candidates)

    if not (_MIN_MOVED <= len(paths) <= _MAX_MOVED):
        return None

    vertical_paths = [i for i, cands in enumerate(path_candidates) if any(c["axis"] == "vertical" for c in cands)]
    if not vertical_paths:
        return None

    for _ in range(_PLACEMENT_RETRIES):
        g = base.copy()
        g[g == PARTICLE] = BG

        chosen: list[dict[str, object]] = []
        forced_vertical_index = random.choice(vertical_paths)
        used_cells: set[tuple[int, int]] = set()
        success = True

        order = list(range(len(path_candidates)))
        random.shuffle(order)
        if forced_vertical_index in order:
            order.remove(forced_vertical_index)
        order.insert(0, forced_vertical_index)

        for idx in order:
            cands = path_candidates[idx]
            local = cands
            if idx == forced_vertical_index:
                vertical = [cand for cand in cands if cand["axis"] == "vertical"]
                if vertical:
                    local = vertical
            local = sorted(local, key=lambda cand: (cand["margin"], cand["forward_len"]), reverse=True)
            top_band = local[: min(4, len(local))]
            random.shuffle(top_band)
            pick = next((cand for cand in top_band if cand["cell"] not in used_cells), None)
            if pick is None:
                pick = next((cand for cand in local if cand["cell"] not in used_cells), None)
            if pick is None:
                success = False
                break
            chosen.append(pick)
            used_cells.add(pick["cell"])

        if not success:
            continue

        for pick in chosen:
            r, c = pick["cell"]
            g[r, c] = PARTICLE

        starts = [(r, c) for r, c in zip(*np.where(g == PARTICLE))]
        if len(starts) != len(paths):
            continue
        directions = []
        for r, c in starts:
            direction = _find_initial_direction(g, r, c)
            if direction is None:
                success = False
                break
            lengths = _direction_lengths(g, r, c)
            sorted_lengths = sorted(lengths.values(), reverse=True)
            if len(sorted_lengths) < 2 or sorted_lengths[0] - sorted_lengths[1] < _MIN_DIRECTION_MARGIN:
                success = False
                break
            directions.append(direction)
        if not success:
            continue
        if not any(direction in {"up", "down"} for direction in directions):
            continue
        if len(set(directions)) < 2:
            continue

        try:
            out = main(g)
        except Exception:
            continue
        if _count_moved(g, out) != len(paths):
            continue
        return g
    return None


def _gen_attempt() -> np.ndarray | None:
    base = _gen_base_attempt()
    if base is None:
        return None
    return _convert_to_combo_variant(base)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
