# Variant of 9bbf930d
# Dimension: D20 — L/U/S形走廊的拓扑变化 (path)
# Change: 走廊支持0-3次转弯：直通（0转）、L形（1转）、U形（2转，同色通道由垂直条连接）。... → Z形走廊（连续两次同向转弯）
#
# Design analysis
# ===============
# a) Concrete behavior
#    - The original family already uses striped bar rows (even) and channel rows (odd).
#      A particle starts at (odd_row, 0), moves right through background cells, and only
#      considers turning when the next cell is colored.
#    - The D20 variant changes the 2-turn topology. Instead of a U-path
#      `right -> up/down -> left`, the generator now builds a Z-path
#      `right -> up/down -> right`. Visually, the continuation corridor is shifted onto
#      another same-color channel row and resumes to the RIGHT of the vertical shaft,
#      so the two horizontal segments are parallel and offset.
#    - Small trace example:
#      original U:
#        source row hits wall at c=10, turns up via column 9, reaches target row,
#        is blocked above, then turns LEFT into column 8.
#      new Z:
#        source row hits wall at c=10, turns up via column 9, reaches target row,
#        is blocked above, then turns RIGHT into column 10 and continues rightward.
#      The solver mechanism is unchanged; only the local gap/seal pattern around the
#      second turn changes.
#
# b) Complexity calibration
#    - Keeping straight and L paths preserves familiar examples.
#    - Replacing U with Z adds a visibly different 2-turn motif without introducing
#      too many new cases. The generator requires at least one genuine Z-trace per grid
#      and rejects accidental U-traces, so the variant is observable in a few examples.
#
# c) Implicit assumption audit of main()
#    - `_has_corridor` only assumes that a candidate turn cell is BG/particle and that
#      its perpendicular flanks match. Z-corridors satisfy the same local condition, so
#      this logic remains valid.
#    - `_trace_path` assumes the next state is fully determined by local walls/gaps and
#      asserts that at most one perpendicular turn is legal. For Z, the generator must
#      still seal the unused first-turn branch and must break the unused LEFT branch at
#      the second turn; otherwise the assert would fire.
#    - Entrance detection assumes a real corridor has aligned left starts on both bar
#      rows. A Z target row is a destination only, so the generator must deliberately
#      break its left entrance while preserving the right-side continuation segment.
#
# d) Placement legality
#    - Every vertical shaft still reserves a 3-column window around its wall column,
#      preventing adjacent shafts from creating ambiguous side turns.
#    - Z targets are chosen among same-color real channels, and their left entrance is
#      broken on a bar row that preferably borders a fake/edge neighbor, reducing
#      collateral damage to other real channels.
#    - Internal dead-end walls for Z are only placed at columns that are not used as
#      other shaft wall columns, preventing unintended third turns.
#
# e) Visual observability
#    - The distinguishing cue is geometric, not color-id based: the destination channel
#      lies above/below the source and resumes on the right side of the shaft. Across a
#      few examples, the human solver can see that 2-turn paths offset rightward rather
#      than folding back leftward.
#
# f) Visual cleanliness
#    - The grid keeps the original striped layout, one-cell channels, and sparse fake
#      decorations. The only added structure is the shifted second horizontal leg, so
#      the image remains readable.
#
# g) Color inference
#    - No hidden color ordering is used. Same-color channel pairs visually signal which
#      two horizontal segments belong to one Z-corridor.

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
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


def _trace_path(grid: np.ndarray, r: int, c: int) -> tuple[int, int, list[tuple[str, str]]]:
    """Trace a particle from (r, c) going right and record direction changes."""
    H, W = grid.shape
    direction = "right"
    turns: list[tuple[str, str]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
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
    return r, c, turns


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid same-color corridor cell on row r, or None."""
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


def _classify_turns(turns: list[tuple[str, str]]) -> str:
    """Classify the traced topology for validation and variant metrics."""
    if not turns:
        return "straight"
    if turns == [("right", "up")] or turns == [("right", "down")]:
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


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Summarize how many visible paths of each topology the input actually contains."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    metrics = {"moved": 0, "straight": 0, "l": 0, "z": 0, "u": 0, "other": 0}
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns = _trace_path(g, r, 0)
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
    """Move magenta particles along same-color corridors, including Z-shaped offsets."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, _ = _trace_path(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator (D20 variant) ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_Z_INTERNAL_STOP_PROB = 0.55

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
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical shaft and return its wall column."""
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
    """Choose the entrance-breaking side that least likely damages another real channel."""
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
    """Pair same-color real channels into Z-turn plans."""
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
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
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
    """Build a Z-turn: source goes right, turns vertically, then resumes right on target."""
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
    """Render all planned turns onto the grid."""
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
    """Generate one candidate input for the D20 Z-corridor variant."""
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

    metrics = _behavior_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["z"] < 1 or metrics["u"] != 0 or metrics["other"] != 0:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    # Keep the output-first validation style: generated inputs must be solvable and
    # must actually move particles away from the left edge.
    if sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE) != metrics["moved"]:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
