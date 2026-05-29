# Variant of 9bbf930d
# Dimension: D6 — 终点条件 (dst)
# Change: 粒子在到达网格边缘或遇到完全阻塞（无有效垂直走廊可转）时停止。... → 到达目标色像素停止
#
# Design analysis for this variant:
# - Concrete form: keep the original striped corridor system unchanged, but add
#   isolated target-color pixels directly on traversable cells of real particle
#   paths. The particle may enter a target pixel; upon entering it, motion ends
#   immediately and the output shows the particle on that cell.
# - Behavior trace: in the original task a path might continue right, turn up,
#   then stop at an edge or dead-end. In this variant, if a target pixel sits on
#   an earlier cell of that same path, the particle stops there instead. The
#   corridor geometry, entrance detection, and turn logic remain the same.
# - Complexity calibration: there is exactly one fixed target color per puzzle
#   and one target per moving real path. This keeps the new cue visible without
#   introducing extra branching rules or hidden per-color semantics.
# - Implicit-assumption audit:
#   1) Original `_has_corridor` and straight-step logic assumed that only bg and
#      particle cells are traversable. That is no longer true, so target cells
#      must also count as traversable corridor cells.
#   2) `main()` still detects corridor entrances by scanning for early bg cells.
#      Therefore the generator must not place targets on the first entrance cell
#      or immediately around it, otherwise a valid real channel could become
#      invisible to the solver.
#   3) Output writing in `main()` overwrites the destination cell with the
#      particle. This was harmless for bg endpoints and remains correct here:
#      reaching a target consumes its visual marker in the output.
# - Placement legality: targets are chosen only from already-traced real paths,
#   excluding the start cell, the natural no-target endpoint, boundary cells,
#   and the early entrance segment. This guarantees no overlap, no ambiguity,
#   and that the solver can legally reach the chosen target.
# - Visual observability: the new rule is legible from examples because several
#   identical isolated target dots appear inside true corridors and disappear in
#   the output exactly where particles stop, often before the old edge/dead-end.
# - Visual cleanliness: targets are single pixels placed on otherwise open path
#   cells, so they do not destroy the bar/corridor layout or pile multiple
#   markings onto the same local structure.
# - Color inference: the target color is fixed and reserved; corridor bars never
#   use it. The mapping is therefore directly visible rather than encoded by an
#   invisible color-id ordering trick.
# - Reference-task note: the requested source task `195c6913` exists in this
#   workspace only as JSON; no solver code was available to read, so the stop-on-
#   target behavior below is implemented from the textual variant description.

from __future__ import annotations

import numpy as np
import random

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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal corridor cell for row r, if any."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG, TARGET}:
            entrance_col = c
            break
    if entrance_col is None:
        return None
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == int(grid[r - 1, entrance_col])), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])), W)
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
    return a == b and a not in {PARTICLE, BG, TARGET}


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    stop_on_target: bool = True,
    return_path: bool = False,
) -> tuple[int, int] | list[tuple[int, int]]:
    """Trace a particle from (r,c) going right.

    Variant rule: a TARGET cell is traversable like bg, but entering it ends the
    motion immediately when `stop_on_target` is true.
    """
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
    return path if return_path else (r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along colored corridors until a target is reached.

    Differences from the original task:
    - target pixels (TARGET) are traversable corridor cells
    - entering a target cell ends the motion immediately
    - otherwise, turning/blocking behavior is unchanged
    """
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator (sa_max_effort) ---------------

# === CFG: tunable generation parameters ===

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 3  # target color is reserved separately

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

# Target placement
_MIN_TARGET_STOPS = 3

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
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


def _choose_target_cells(g: np.ndarray) -> dict[int, tuple[int, int]] | None:
    """Pick one interior target cell on each moving real path."""
    H, W = g.shape
    target_cells = {}
    used_cells = set()

    for r in range(1, H - 1):
        entrance_col = _find_entrance_col(g, r)
        if entrance_col is None:
            continue
        try:
            path = _trace(g, r, 0, stop_on_target=False, return_path=True)
        except AssertionError:
            return None
        if len(path) < 6:
            return None

        min_idx = max(entrance_col + 2, len(path) // 2)
        candidates = []
        for idx in range(min_idx, len(path) - 1):
            rr, cc = path[idx]
            if (rr, cc) in used_cells:
                continue
            if not (0 < rr < H - 1 and 1 < cc < W - 1):
                continue
            if int(g[rr, cc]) != BG:
                continue
            candidates.append((rr, cc))
        if not candidates:
            return None

        choice = random.choice(candidates)
        target_cells[r] = choice
        used_cells.add(choice)

    return target_cells if len(target_cells) >= _MIN_TARGET_STOPS else None


def _variant_stats(g: np.ndarray) -> dict[str, int]:
    """Quantify whether the target-stop mechanic is actually active."""
    moved = 0
    target_stops = 0
    shortened = 0
    for r in range(1, g.shape[0] - 1):
        if _find_entrance_col(g, r) is None:
            continue
        try:
            natural_end = _trace(g, r, 0, stop_on_target=False)
            target_end = _trace(g, r, 0, stop_on_target=True)
        except AssertionError:
            return {"moved": 0, "target_stops": 0, "shortened": 0}
        if natural_end != (r, 0):
            moved += 1
        if target_end != natural_end and int(g[target_end]) == TARGET:
            target_stops += 1
            shortened += 1
    return {"moved": moved, "target_stops": target_stops, "shortened": shortened}


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D6 target-stop variant."""
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

    target_cells = _choose_target_cells(g)
    if target_cells is None:
        return None
    for rr, cc in target_cells.values():
        g[rr, cc] = TARGET

    try:
        out = main(g)
    except Exception:
        return None

    stats = _variant_stats(g)
    if stats["target_stops"] < _MIN_TARGET_STOPS or stats["shortened"] < _MIN_TARGET_STOPS:
        return None

    n_moved = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
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


def _build_u_turn(g, W, top, bot, row, color, c_v, direction, j_top, j_bot, bar_color, bar_start, extras, internal_stop):
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
