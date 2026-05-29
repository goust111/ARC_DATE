# Combo variant of 9bbf930d (index 18)
# combo_spec: D7/0+D6/0
# Name: 终点扩散+目标色停止
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Base task: a magenta particle starts at (r, 0), follows same-color corridors,
#      and normally stops only at an edge or a hard block; output writes one magenta
#      pixel at that natural endpoint.
#    - Fused task: color 8 is a visible target marker placed on traversable cells of
#      real paths. The particle may step onto that target cell exactly like onto bg,
#      but the moment it enters the target it stops, even if the corridor continues.
#      Output then paints a magenta cross centered on that stopping target cell.
#    - Example: if a path would naturally continue to (5, 15) but a target is placed
#      earlier at (5, 11), the fused solver stops at (5, 11) and paints the clipped
#      cross {(5,11), (4,11), (6,11), (5,10), (5,12)} instead of a single pixel at
#      the old natural endpoint.
# b) Complexity calibration:
#    - The target mechanic uses one fixed reserved color, not a per-sample mapping.
#    - The output mechanic uses one fixed 4-neighbor cross, not a sample-dependent
#      choice between multiple spread shapes.
#    - The generator places one target on each moving real path so the stop rule is
#      repeated several times in every puzzle and the burst pattern is visually obvious.
# c) Hidden-assumption audit:
#    - Original straight-step logic assumed only {particle, bg} were traversable.
#      Fused solver widens traversability to include TARGET.
#    - Original entrance detection scanned for early bg cells. That still holds, so
#      the generator must never place a target on the first valid entrance segment.
#    - Original output mutation could safely happen row-by-row because each move wrote
#      only one endpoint. With bursts, painting must happen after all moved starts are
#      cleared, otherwise a later start-clear could erase an earlier burst arm.
#    - Original corridor checks treated any non-bg/non-particle as a wall color. In
#      the fused task TARGET must not count as a wall, so `_has_corridor` excludes it.
# d) Placement legality:
#    - Targets are chosen only from already traced real paths, excluding the start,
#      the natural no-target endpoint, early entrance cells, borders, and non-channel
#      rows. This keeps targets reachable and solver-visible.
#    - The generator also enforces disjoint burst footprints so output crosses do not
#      merge into a messy blob.
# e) Visual observability:
#    - Input shows several isolated target-colored dots embedded in true corridors.
#    - Output shows the corresponding particles stopping at those dots and expanding
#      into fixed magenta crosses, which is directly inferable from 2-4 examples.
# f) Visual cleanliness:
#    - Restricting targets to odd corridor rows keeps them legible as dots inside open
#      channels rather than hidden in bar rows.
#    - Disjoint bursts prevent endpoint markers from collapsing into one ambiguous mass.
# g) Change inferability:
#    - The target color is structurally singled out because it appears as isolated dots
#      on traversable paths and disappears under the terminal magenta crosses.
#    - The spread rule is positional rather than palette-dependent: wherever a particle
#      stops, the same fixed cross is drawn.
# h) Variant-strength metrics used by the generator:
#    - `target_stops`: how many particles stop on target cells rather than at the old
#      natural endpoints.
#    - `burst_gain`: how many magenta cells the cross rendering adds beyond the same
#      target-stop endpoints rendered as single pixels.
# i) New solver assumptions and generator guarantees:
#    - Assumption: target cells lie on genuine traversable real paths.
#      Guarantee: targets are sampled from `_trace(..., stop_on_target=False)` paths.
#    - Assumption: target-stop endpoints remain interior enough for visible spreading.
#      Guarantee: targets are restricted away from the border.
#    - Assumption: burst overlap is not needed to express the rule.
#      Guarantee: target selection rejects overlapping burst footprints.

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
    return a == b and a not in {PARTICLE, BG, TARGET}


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    stop_on_target: bool = True,
    return_path: bool = False,
) -> tuple[int, int] | list[tuple[int, int]]:
    """Trace a particle from (r,c) going right."""
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


def _collect_moves(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Collect `(start_row, end_row, end_col)` for particles that truly move."""
    H = grid.shape[0]
    moves: list[tuple[int, int, int]] = []
    for r in range(1, H - 1):
        if _find_entrance_col(grid, r) is None:
            continue
        end_r, end_c = _trace(grid, r, 0, stop_on_target=True)
        if (end_r, end_c) != (r, 0):
            moves.append((r, end_r, end_c))
    return moves


def _burst_cells(H: int, W: int, r: int, c: int) -> set[tuple[int, int]]:
    """Return the in-bounds cross centered on (r, c)."""
    cells = set()
    for dr, dc in _BURST_DELTAS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            cells.add((rr, cc))
    return cells


def _paint_burst(out: np.ndarray, r: int, c: int) -> None:
    """Paint the terminal particle burst as a clipped 4-neighbor cross."""
    H, W = out.shape
    for rr, cc in _burst_cells(H, W, r, c):
        out[rr, cc] = PARTICLE


def _single_point_output(grid: np.ndarray, moves: list[tuple[int, int, int]]) -> np.ndarray:
    """Render target-stop endpoints without the D7 burst, for validation only."""
    out = np.array(grid, dtype=int, copy=True)
    for start_r, end_r, end_c in moves:
        out[start_r, 0] = BG
        out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles until target contact, then mark endpoints as crosses.

    Input structure:
    - bg=7, particle=6, target=8.
    - Col 0 is filled with particles; real corridors are bg cells between same-color
      bar pairs, and some interior real-path cells contain isolated target dots.

    Rule:
    - A valid particle enters the first real corridor on its row and follows the base
      movement rule: go straight through traversable cells, and when blocked by a bar
      color, turn 90 degrees into the unique perpendicular same-color corridor.
    - Target cells are traversable like bg, but entering one ends the motion
      immediately, even if the corridor would continue further.
    - After all moved starts are cleared, each terminal cell is painted as a clipped
      magenta cross consisting of center, up, down, left, and right.

    Key insight:
    - The endpoint is no longer determined only by geometry; a visible target dot can
      truncate the path early, and that new endpoint is emphasized by a fixed cross.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    moves = _collect_moves(g)
    for start_r, _, _ in moves:
        out[start_r, 0] = BG
    for _, end_r, end_c in moves:
        _paint_burst(out, end_r, end_c)
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

# Target placement / validation
_MIN_TARGET_STOPS = 3
_MIN_BURST_GAIN = 10

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
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


def _choose_target_cells(g: np.ndarray) -> dict[int, tuple[int, int]] | None:
    """Pick one interior, visually separated target cell on each moving real path."""
    H, W = g.shape
    path_info: list[tuple[int, list[tuple[int, int]], int]] = []

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
        path_info.append((r, path, entrance_col))

    random.shuffle(path_info)
    target_cells: dict[int, tuple[int, int]] = {}
    used_burst_cells: set[tuple[int, int]] = set()

    for r, path, entrance_col in path_info:
        min_idx = max(entrance_col + 2, len(path) // 2)
        candidates = []
        for idx in range(min_idx, len(path) - 1):
            rr, cc = path[idx]
            if rr % 2 == 0:
                continue
            if not (0 < rr < H - 1 and 2 < cc < W - 2):
                continue
            if int(g[rr, cc]) != BG:
                continue
            burst = _burst_cells(H, W, rr, cc)
            if burst & used_burst_cells:
                continue
            candidates.append((rr, cc))
        if not candidates:
            return None
        rr, cc = random.choice(candidates)
        target_cells[r] = (rr, cc)
        used_burst_cells.update(_burst_cells(H, W, rr, cc))

    return target_cells if len(target_cells) >= _MIN_TARGET_STOPS else None


def _variant_stats(g: np.ndarray) -> dict[str, int]:
    """Quantify whether both target-stop and burst behavior are active."""
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
            return {"moved": 0, "target_stops": 0, "shortened": 0, "burst_gain": 0}
        if natural_end != (r, 0):
            moved += 1
        if target_end != natural_end and int(g[target_end]) == TARGET:
            target_stops += 1
            shortened += 1

    moves = _collect_moves(g)
    single_point = _single_point_output(g, moves)
    combo_out = main(g)
    burst_gain = int(np.sum((combo_out == PARTICLE) & (single_point != PARTICLE)))
    return {
        "moved": moved,
        "target_stops": target_stops,
        "shortened": shortened,
        "burst_gain": burst_gain,
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused target-stop + burst variant."""
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
        _ = main(g)
    except Exception:
        return None

    stats = _variant_stats(g)
    if stats["target_stops"] < _MIN_TARGET_STOPS or stats["shortened"] < _MIN_TARGET_STOPS:
        return None
    if stats["burst_gain"] < _MIN_BURST_GAIN:
        return None
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


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
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
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


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
