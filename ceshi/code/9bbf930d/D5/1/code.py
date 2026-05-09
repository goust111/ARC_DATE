# Variant of 9bbf930d
# Dimension: D5 — 源对象：粒子定义 (src)
# Change: 品红色粒子位于最左列（col 0），从有效走廊入口的奇数行出发，初始方向固定向右。... → 粒子位于走廊内部

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

# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old: every row starts with a magenta particle in col 0, and real particles
#       are discovered by checking whether that row admits a horizontal entrance.
#       Once accepted, the particle always starts with heading "right".
#     new: only 3-6 magenta particles exist, and each one is already placed inside
#       a real horizontal corridor. The particle is not on the grid edge. Its start
#       direction must be inferred from the local corridor shape: the cell itself is
#       inside a same-color horizontal corridor, and one horizontal side is sealed by
#       a same-color blocker while the other side stays open, so the unique heading is
#       visible as the only open continuation of that corridor.
#   concrete_visual_form:
#     - the striped corridor/bar layout stays the same
#     - the old solid magenta col 0 disappears; col 0 becomes ordinary background
#     - each real path receives one internal particle placed on an odd-row horizontal
#       corridor cell, plus a same-color blocker immediately behind it
#     - the suffix of the original path stays unchanged, so the particle follows the
#       same corridor logic after its internal launch point
#   behavior_trace_example:
#     old:
#       1. particle at (5, 0) enters the row-5 corridor and moves right
#       2. it hits a colored blocker at col 13, turns upward, then finishes elsewhere
#     new:
#       1. remove the edge particle; place the particle at (5, 8)
#       2. recolor (5, 7) with the row-5 corridor color, sealing the left side
#       3. because (5, 8) is visibly inside a horizontal corridor and only its right
#          neighbor remains open, the initial heading is uniquely "right"
#       4. from then on, tracing is the same suffix as the original path
#   complexity_calibration:
#     - the variant is visually obvious because the distinctive col-0 magenta wall is
#       gone and the remaining particles are sparse interior objects
#     - local direction inference uses only a small, repeated pattern:
#       "particle in corridor + one side blocked"
#     - we do not introduce hidden color-id logic; the blocker color is simply the
#       same visible corridor color already flanking that row
#   old_main_assumptions_that_break:
#     - `if g[r, 0] != PARTICLE`: invalid because starts are no longer in col 0
#     - entrance search from col 2 onward: invalid because the particle is already
#       inside the corridor rather than outside the left boundary
#     - `_trace(... direction='right')`: invalid because some internal starts head
#       left and others head right
#     - `out[r, 0] = BG`: invalid because the source cell is no longer fixed
#   placement_legality:
#     - starts are only chosen from traced real-path cells on odd rows where the path
#       currently moves horizontally
#     - the particle's "back" neighbor on that row is recolored as a blocker, which
#       makes the local heading unique without changing the path suffix
#     - the generator validates the transformed grid with the new solver, so bad
#       placements, overlaps, or ambiguous starts are rejected
#   visual_observability:
#     - from a few examples, a human can see that particles begin inside corridors,
#       not at the left edge, and that each one moves away from a nearby blocker
#     - the same corridor-following rule after launch is preserved, which keeps the
#       task readable instead of inventing a second unrelated mechanic
#   new_solver_assumptions:
#     - every particle lies on a same-color horizontal corridor cell
#     - exactly one horizontal continuation is open from that cell
#     - particles are placed on disjoint suffixes so moving one does not invalidate
#       another; generator enforces this by construction and by whole-grid validation


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
    """Trace one particle and record every visited cell and the direction of each step."""
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


def _find_start_directions(grid: np.ndarray, r: int, c: int) -> list[str]:
    """Infer the internal particle's initial direction from its local corridor structure."""
    if int(grid[r, c]) != PARTICLE:
        return []
    options = []
    for direction in ("left", "right", "up", "down"):
        if not _has_corridor(grid, r, c, direction):
            continue
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if int(grid[nr, nc]) in {PARTICLE, BG}:
                options.append(direction)
    return options


def main(grid: np.ndarray) -> np.ndarray:
    """Move internal magenta particles along the same corridor network to their final cells."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    starts = [(r, c) for r, c in zip(*np.where(g == PARTICLE))]

    for r, c in starts:
        options = _find_start_directions(g, r, c)
        if len(options) != 1:
            continue
        end_r, end_c = _trace(g, r, c, options[0])
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


def _find_legacy_source_rows(grid: np.ndarray) -> list[int]:
    """Rows that the original edge-based solver would consider valid starts."""
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
    """Original solver behavior, kept only to validate and transform base instances."""
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


def _gen_base_attempt() -> np.ndarray | None:
    """Generate one valid original-style instance before converting its particle starts."""
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
    n_moved = sum(1 for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def _horizontal_block_color(grid: np.ndarray, r: int, c: int) -> int:
    """Use the same visible corridor color as the start row's flanking bars."""
    above = int(grid[r - 1, c]) if r > 0 else BG
    below = int(grid[r + 1, c]) if r + 1 < grid.shape[0] else BG
    if above == below and above not in {BG, PARTICLE}:
        return above
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            val = int(grid[nr, nc])
            if val not in {BG, PARTICLE}:
                return val
    raise ValueError("no suitable blocker color around start cell")


def _convert_to_internal_variant(base: np.ndarray) -> np.ndarray | None:
    """Turn edge-launched particles into internal horizontal starts with local blockers."""
    h, w = base.shape
    paths = []
    for r in _find_legacy_source_rows(base):
        path, dirs = _trace_with_history(base, r, 0, "right")
        if len(path) <= 1:
            continue
        paths.append((path, dirs))

    if not (_MIN_MOVED <= len(paths) <= _MAX_MOVED):
        return None

    g = base.copy()
    g[g == PARTICLE] = BG
    random.shuffle(paths)

    for path, dirs in paths:
        candidates = []
        for i, direction in enumerate(dirs):
            if direction not in {"left", "right"}:
                continue
            r, c = path[i]
            if not (0 < r < h - 1 and 2 <= c <= w - 3):
                continue
            if len(path) - i < 4:
                continue
            dr, dc = _DELTAS[direction]
            if path[i + 1] != (r + dr, c + dc):
                continue
            if not _has_corridor(base, r, c, direction):
                continue
            back_c = c - 1 if direction == "right" else c + 1
            if not (0 <= back_c < w):
                continue
            candidates.append((i, direction, back_c))

        random.shuffle(candidates)
        placed = False
        for i, direction, back_c in candidates:
            r, c = path[i]
            blocker = _horizontal_block_color(base, r, c)
            prev_back = int(g[r, back_c])
            prev_here = int(g[r, c])
            g[r, back_c] = blocker
            g[r, c] = PARTICLE
            options = _find_start_directions(g, r, c)
            end_r, end_c = _trace(g, r, c, direction) if options == [direction] else (r, c)
            if options == [direction] and (end_r, end_c) != (r, c):
                placed = True
                break
            g[r, back_c] = prev_back
            g[r, c] = prev_here
        if not placed:
            return None

    particle_cells = list(zip(*np.where(g == PARTICLE)))
    if len(particle_cells) != len(paths):
        return None
    if any(r in {0, h - 1} or c in {0, w - 1} for r, c in particle_cells):
        return None
    if any(len(_find_start_directions(g, r, c)) != 1 for r, c in particle_cells):
        return None
    try:
        out = main(g)
    except Exception:
        return None
    moved = sum(1 for r, c in particle_cells if int(out[r, c]) != PARTICLE)
    if moved != len(paths):
        return None
    return g


def _gen_attempt() -> np.ndarray | None:
    base = _gen_base_attempt()
    if base is None:
        return None
    variant = _convert_to_internal_variant(base)
    if variant is None:
        return None
    if random.random() < 0.5:
        variant = np.fliplr(variant).copy()
        try:
            _ = main(variant)
        except Exception:
            return None
    return variant


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
