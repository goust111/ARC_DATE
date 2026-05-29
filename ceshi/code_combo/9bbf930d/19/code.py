# Combo variant of 9bbf930d (index 19)
# combo_spec: D17/0+D10/1
# Name: 仅2种走廊色+假走廊为陷阱
#
# Design analysis:
# a) Concrete behavior trace:
#    Original 9bbf930d launches particles from odd rows whose horizontal channel is flanked by the same color on
#    both even-row bar lines. A moving particle keeps going through background cells and only turns when a blocker
#    is immediately ahead and exactly one perpendicular mouth is a same-color corridor. The fused variant changes
#    two linked facts. First, every corridor bar in the whole grid is built from exactly two colors A/B. Second,
#    an unequal-color mouth is no longer inert: if a blocker is ahead and the only available perpendicular mouth is
#    A|BG|B or B|BG|A, the particle steps into that mouth cell and stops there. Example: a row between A/A bars
#    moves right until an A blocker. If the upward mouth is A-B mismatched, the original solver stays on the row
#    before the blocker, while the fused solver moves one cell upward into the mouth and gets trapped.
# b) Complexity calibration:
#    The palette is aggressively simplified but not collapsed. Using exactly two corridor colors makes real A/A and
#    B/B channels visually repeat across the sample, while fake A/B and B/A channels also repeat. That gives human
#    support for the rule "equality means true, inequality means trap" without relying on many colors. The
#    generator still emits a mix of straight, real-turn, U-turn, and trap rows so the task does not degenerate
#    into a trivial single-branch puzzle.
# c) Hidden-assumption audit of the original solver:
#    - `_has_corridor` assumed only same-color mouths mattered. That is false now, so the solver must classify
#      mouths into real/fake/none instead of boolean valid/invalid.
#    - `_trace` assumed "no real turn" means stop in place. That is false when the only mouth is fake, because the
#      particle must enter and stop at the fake mouth.
#    - `main()` launch-row detection assumed many bar colors might exist, but its real requirement is narrower:
#      launch only from rows whose horizontal channel is same-color. That still holds and is preserved.
#    - The original generator assumed fake rows were harmless decoration. That no longer holds because fake
#      branches can affect the output, so the generator must deliberately attach at least one reachable fake mouth
#      to a moving real row.
# d) Placement legality:
#    The generator keeps the original striped geometry. Even rows are bars, odd rows are channel rows, and every
#    vertical structure reserves a 3-column window around its wall column. Real turns, U-turns, and trap mouths
#    all use the same reservation scheme, so they do not overlap and they stay inside bounds. The opposite side of
#    each turn mouth is sealed, preventing simultaneous up/down choices at the same blocker.
# e) Visual observability:
#    From 2-4 examples a human can see two strong regularities. Real horizontal launch rows are always between
#    identical colors, while non-launch fake rows are between alternating colors. In outputs, some particles end on
#    odd-row channel endpoints after following real turns, but at least one particle ends on an even-row mouth
#    cell, revealing that an unequal-color branch acts as a trap rather than as a valid turn or inert decoration.
# f) Visual cleanliness:
#    No extra decorator colors are added. The entire puzzle uses only corridor colors A/B plus BG and the magenta
#    particle, so the scene stays readable. Trap branches are kept short and local; the fake pattern is already
#    visible from the full-width mismatched horizontal channels, so no noisy ornament is needed.
# g) Change inferability:
#    The color rule is structural, not value-based: same-color pairs are true corridors, different-color pairs are
#    fake corridors. Because only two corridor colors are present, fake pairs can only be AB or BA, which makes
#    the distinction subtle but still inferable from repeated same-vs-different examples.
# 9) New solver assumptions guaranteed by the generator:
#    - Every generated grid uses exactly two non-BG/non-particle corridor colors, and both colors occur multiple
#      times on bar rows.
#    - At any blocker there is at most one perpendicular corridor-like mouth, whether real or fake.
#    - At least one moving particle is trapped by a fake mouth, so the fused D10 behavior changes the output.
#    - At least two fake horizontal channels remain elsewhere, so the D17 two-color equality/inequality cue is
#      visible beyond the trapped branch itself.

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


def _other_color(color: int, corridor_colors: tuple[int, int]) -> int:
    return corridor_colors[1] if color == corridor_colors[0] else corridor_colors[0]


def _corridor_kind(grid: np.ndarray, r: int, c: int, direction: str) -> str | None:
    """Classify a corridor mouth as 'real', 'fake', or None."""
    h, w = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a in {PARTICLE, BG} or b in {PARTICLE, BG}:
        return None
    return "real" if a == b else "fake"


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle under the fused two-color trap rule."""
    h, w = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue

        real_options = []
        fake_options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            kind = _corridor_kind(grid, tr, tc, turn_dir)
            if kind == "real":
                real_options.append((tr, tc, turn_dir))
            elif kind == "fake":
                fake_options.append((tr, tc, turn_dir))

        total = len(real_options) + len(fake_options)
        assert total <= 1, "ambiguous turn: multiple perpendicular mouths"
        if real_options:
            r, c, direction = real_options[0]
            continue
        if fake_options:
            return fake_options[0][0], fake_options[0][1]
        break
    return r, c


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original tracing rule used to reject degenerate generators."""
    h, w = grid.shape
    direction = "right"
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
            if 0 <= tr < h and 0 <= tc < w and _corridor_kind(grid, tr, tc, turn_dir) == "real":
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _solve_with_trace(grid: np.ndarray, trace_fn) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            if _corridor_kind(g, r, c, "right") == "real":
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next((c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), w)
        bar_below = next((c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), w)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = trace_fn(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


def _launch_rows(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    rows: list[int] = []
    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            if _corridor_kind(g, r, c, "right") == "real":
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), w)
        bar_below = next((c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), w)
        if bar_above == bar_below and bar_above <= 3:
            rows.append(r)
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through two-color corridors with fake-mouth traps.

    Input structure:
    - bg=7 and particle=6.
    - Even rows are bar rows and odd rows are channel rows.
    - Every corridor in the grid uses exactly two corridor colors; real horizontal channels have equal colors
      above/below, while fake horizontal channels have different colors above/below.

    Rule:
    - Launch a particle only from rows whose horizontal entrance is a real same-color corridor.
    - A moving particle goes straight through background cells.
    - When blocked by a colored cell, inspect the two perpendicular mouths. If exactly one real mouth exists,
      turn into it and continue. If no real mouth exists but exactly one fake mouth exists, enter that mouth and
      stop there. Otherwise stop.

    Key insight:
    - With only two corridor colors, the decisive cue is not which color appears but whether the two flank colors
      around a mouth match or differ; mismatched mouths are traps.
    """
    return _solve_with_trace(grid, _trace)


def _main_original(grid: np.ndarray) -> np.ndarray:
    return _solve_with_trace(grid, _trace_original)


# --------------- generator (sa_max_effort) ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [3, 3, 2]
_MIN_FAKE_CHANNELS = 2

_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.75
_TRAP_COUNT_RANGE = (1, 2)

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_MIN_MOVED = 3
_MAX_MOVED = 6
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
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 3:
            continue
        random.shuffle(chans)
        max_pairs = (len(chans) - 1) // 2
        n_pairs = random.randint(0, max_pairs)
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


def _plan_traps(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    available = [i for i in real_indices if i not in plans]
    if not available:
        return
    random.shuffle(available)
    lo, hi = _TRAP_COUNT_RANGE
    n_traps = min(len(available), random.randint(lo, min(hi, len(available))))
    for i in available[:n_traps]:
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            break
        plans[i] = {"kind": "trap", "direction": random.choice(["up", "down"]), "c_v": c_v}


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
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
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": random.choice(["up", "down"]), "c_v": c_v}


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
    corridor_colors: tuple[int, int],
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

    g[j_top, bar_start] = _other_color(color, corridor_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, w - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _build_fake_trap(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    source_color: int,
    c_v: int,
    direction: str,
    corridor_colors: tuple[int, int],
) -> None:
    """Build a single mismatched turn mouth that traps the particle at entry."""
    trap_color = _other_color(source_color, corridor_colors)
    g[row, c_v] = source_color
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = source_color
    g[turn_bar, c_v] = trap_color
    g[opp_bar, c_v - 1] = source_color


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    corridor_colors: tuple[int, int],
) -> None:
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in ("straight", "u_target"):
            continue

        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, h - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, h, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif kind == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
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
                corridor_colors,
                bool(plan.get("internal_stop", False)),
            )
        elif kind == "trap":
            _build_fake_trap(
                g,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                corridor_colors,
            )


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    bar_rows = list(range(0, h, 2))
    corridor_colors = {
        int(g[r, c])
        for r in bar_rows
        for c in range(2, g.shape[1])
        if int(g[r, c]) not in {BG, PARTICLE}
    }
    real_channels = 0
    fake_channels = 0
    trap_captures = 0
    moved_particles = 0
    even_row_landings = 0
    for i in range(len(bar_rows) - 1):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if int(g[top, 2]) == int(g[bot, 2]):
            real_channels += 1
        else:
            fake_channels += 1
    for r in _launch_rows(g):
        new_end = _trace(g, r, 0)
        old_end = _trace_original(g, r, 0)
        if new_end != (r, 0):
            moved_particles += 1
        if new_end != old_end:
            trap_captures += 1
        if new_end[0] % 2 == 0 and new_end != (r, 0):
            even_row_landings += 1
    return {
        "corridor_colors": len(corridor_colors),
        "real_channels": real_channels,
        "fake_channels": fake_channels,
        "trap_captures": trap_captures,
        "moved_particles": moved_particles,
        "even_row_landings": even_row_landings,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    corridor_colors = tuple(random.sample([c for c in range(10) if c not in (BG, PARTICLE)], 2))

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1

    max_real = n_channels - _MIN_FAKE_CHANNELS
    if max_real < min(_N_REAL_OPTIONS):
        return None
    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, max_real)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(corridor_colors)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = _other_color(bar_color[top], corridor_colors)

    if len(set(bar_color.values())) != 2:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_traps(real_indices, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, w, used_vcols, plans)
    if not any(str(plan["kind"]) == "trap" for plan in plans.values()):
        return None
    _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, corridor_colors)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        old_out = _main_original(g)
    except Exception:
        return None
    if np.array_equal(out, old_out):
        return None

    metrics = _behavior_metrics(g)
    if metrics["corridor_colors"] != 2:
        return None
    if metrics["fake_channels"] < _MIN_FAKE_CHANNELS:
        return None
    if metrics["trap_captures"] < 1 or metrics["even_row_landings"] < 1:
        return None
    if not (_MIN_MOVED <= metrics["moved_particles"] <= _MAX_MOVED):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
