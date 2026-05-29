# Variant of 9bbf930d
# Dimension: D5 — 源对象：粒子定义 (src)
# Change: 品红色粒子位于最左列（col 0），从有效走廊入口的奇数行出发，初始方向固定向右。
#   → 多色粒子分组：粒子有多种颜色，每种颜色的粒子只能进入对应颜色的走廊
#     （颜色必须匹配条对色）；因此“能否进入/转入走廊”的约束由粒子自身颜色决定。
#
# Design analysis
# ===============
# Reference-task note
# - The prompt cites `b5ca7ac4` for color-grouping inspiration.
# - In this workspace there is no local solver file for `b5ca7ac4`; only
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json`
#   is available, so the implementation below uses only the observed idea
#   "same-colored sources belong to the same behavioral group" and does not
#   import any unseen solver logic.
#
# a) Concrete behavior and trace
# - The base layout stays striped: even rows are colored bar rows, odd rows are
#   channel rows, and horizontal/vertical corridors are still detected by
#   same-color bar pairs around a 1-cell background path.
# - The visual change is concentrated in the source column: odd-row particles in
#   col 0 now use several colors taken from the corridor palette instead of a
#   single magenta color.
# - A particle may start only if the first reachable horizontal corridor on its
#   row has the same color as the particle itself. Later turns are constrained in
#   the same way: among perpendicular candidates, only corridors flanked by that
#   particle color are legal.
# - Small example:
#   original: row 5 had a particle `6`, the bars above/below were blue, and the
#   solver still entered because color was ignored after the "is it magenta?"
#   check.
#   variant: row 5 may contain a red source pixel. If the bars above/below are
#   blue, the entrance is invalid and the particle stays in col 0. On another
#   row with a red corridor, the same red source does move.
# - This changes the pipeline at two places: solver entrance/turn filtering and
#   generator source-color assignment.
#
# b) Complexity calibration
# - The variant is intentionally narrow: corridor geometry, turn shapes, and
#   stopping rules are unchanged. Only the source identity becomes multi-valued.
# - To keep the change visible, the generator enforces:
#   1. at least two real corridor colors,
#   2. at least one movable real corridor whose source color matches,
#   3. at least one other movable real corridor whose source color mismatches.
# - This avoids a degenerate dataset where all sources behave the same or every
#   source has a unique color with no grouping support.
#
# c) Old implicit assumptions that break under D5
# - Original `PARTICLE = 6` assumed source identity and source color were the
#   same thing. That fails here because multiple colors must be treated as
#   particles on odd rows.
# - Original `_has_corridor()` only checked "same colored walls" and did not
#   know which color the current particle was allowed to use. That would let a
#   red source enter a blue corridor.
# - Original `_trace()` treated `{BG, PARTICLE}` as passable and every other
#   color as a blocker. In the variant the source color is per-particle, so the
#   passable set must become `{BG, current_particle_color}`.
# - Original `main()` cleared the start and wrote back the constant `PARTICLE`.
#   That would overwrite the correct final source color. The variant must write
#   back the moving particle's own color.
#
# d) Placement legality
# - The generator keeps the original corridor-building routines, so geometric
#   legality (no out-of-bounds turns, sealed L-turns, non-ambiguous U-turns) is
#   preserved.
# - Source colors are assigned after corridor geometry is fixed and only affect
#   col 0. They therefore cannot create bar overlaps or invalidate turn carving.
# - Rows that serve as U-turn targets still have their entrance broken exactly as
#   in the base task; their source color is also set to a mismatching color so
#   they are visibly non-moving.
#
# e) Visual observability
# - Humans can now compare several odd rows and notice that only the rows whose
#   left-edge particle color matches the corridor color actually move.
# - The generator deliberately includes both moving and non-moving particles that
#   share colors with real corridors elsewhere, so the rule is inferable from
#   2-4 examples without any hidden numeric ordering.
#
# f) Visual cleanliness
# - The grid stays stripe-based and low-overlap. The only new overlay is a
#   colored source column, which is spatially isolated from the corridors by the
#   preserved background col 1.
# - No new markers or annotations are added inside the corridor field.
#
# g) Color-rule inferability
# - Particle colors are drawn from the same visible palette as the corridors.
#   There is no invisible mapping, arithmetic on color ids, or sort-by-id rule.
# - A mismatch means "source color differs from the bar-pair color on that row",
#   which is directly readable from the pixels.
#
# h) Generator-side behavioral metric
# - We quantify the variant with `mismatched_real_sources`: real, geometrically
#   movable channels whose source color does not match their corridor color.
# - If this metric is zero, the sample visually collapses back toward the base
#   corridor behavior, so the generator rejects it.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: only odd rows are source rows. Generator guarantees source
#   placement on odd rows and keeps col 1 as pure background.
# - Solver assumption: a legal turn corridor must have wall color equal to the
#   current particle color. Generator guarantees every real multi-turn path keeps
#   one consistent corridor color along the whole path, exactly like the base
#   task's same-color U/L construction.
# - Solver assumption: passable non-background cells, if any, must match the
#   moving particle color. Generator keeps all traveled path cells background
#   except for the source itself, so this remains safe.

from __future__ import annotations

import random

import numpy as np

BG = 7
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(
    grid: np.ndarray, r: int, c: int, direction: str, particle_color: int
) -> bool:
    """True iff (r,c) is passable and flanked by bars of the particle color."""
    H, W = grid.shape
    cell = int(grid[r, c])
    if cell not in {BG, particle_color}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b == particle_color


def _trace(grid: np.ndarray, r: int, c: int, particle_color: int) -> tuple[int, int]:
    """Trace one colored particle from (r, c) going right."""
    H, W = grid.shape
    direction = "right"
    passable = {BG, particle_color}
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in passable:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W:
                if _has_corridor(grid, tr, tc, turn_dir, particle_color):
                    options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move each odd-row source only through corridors of the same color."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1, 2):
        particle_color = int(g[r, 0])
        if particle_color == BG:
            continue

        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below == particle_color:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == particle_color), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == particle_color), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0, particle_color)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = particle_color

    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 5

# Channel assignment
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
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
_MAX_RETRIES = 400
_MIN_MISMATCHED_REAL_SOURCES = 1


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall col."""
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
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
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
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn while keeping the original non-ambiguous sealing rules."""
    color = bar_color[top]
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
    """Build a same-color U-turn between two real channels."""
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


def _choose_alt_color(colors: list[int], forbidden: int) -> int | None:
    options = [c for c in colors if c != forbidden]
    return random.choice(options) if options else None


def _candidate_source_rows(
    g: np.ndarray, bar_rows: list[int]
) -> list[tuple[int, int, int]]:
    """Rows whose geometry exposes a valid same-color left entrance.

    Returns `(channel_index, row, corridor_color)` tuples using only corridor
    geometry, independent of whatever source color will later be placed in col 0.
    """
    H, W = g.shape
    candidates: list[tuple[int, int, int]] = []
    for i, top in enumerate(bar_rows[:-1]):
        row = top + 1
        if row <= 0 or row >= H - 1:
            continue
        entrance_col = None
        corridor_color = None
        for c in range(2, W):
            if int(g[row, c]) != BG:
                continue
            above, below = int(g[row - 1, c]), int(g[row + 1, c])
            if above == below and above != BG:
                entrance_col = c
                corridor_color = above
                break
        if entrance_col is None or corridor_color is None:
            continue
        bar_above = next((c for c in range(W) if int(g[row - 1, c]) == corridor_color), W)
        bar_below = next((c for c in range(W) if int(g[row + 1, c]) == corridor_color), W)
        if bar_above == bar_below and bar_above <= 3:
            candidates.append((i, row, corridor_color))
    return candidates


def _assign_particle_colors(
    g: np.ndarray,
    channel_spec: list[str],
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
) -> tuple[dict[int, int], dict[str, int]] | tuple[None, None]:
    """Assign odd-row source colors so some real rows match and some mismatch."""
    real_colors = sorted({bar_color[bar_rows[i]] for i in real_indices})
    if len(real_colors) < 2:
        return None, None

    candidates = _candidate_source_rows(g, bar_rows)
    candidate_indices = [i for i, _, _ in candidates]
    candidate_colors = {corridor_color for _, _, corridor_color in candidates}
    if len(candidate_indices) < _MIN_MOVED + _MIN_MISMATCHED_REAL_SOURCES:
        return None, None
    if len(candidate_colors) < 2:
        return None, None

    max_match = min(_MAX_MOVED, len(candidate_indices) - _MIN_MISMATCHED_REAL_SOURCES)
    if max_match < _MIN_MOVED:
        return None, None
    n_match = random.randint(_MIN_MOVED, max_match)
    matched = set(random.sample(candidate_indices, n_match))

    particle_color: dict[int, int] = {}
    for i in range(len(channel_spec)):
        corridor_color = bar_color[bar_rows[i]]
        if i in matched:
            particle_color[i] = corridor_color
            continue

        alt = _choose_alt_color(real_colors, corridor_color)
        if alt is None:
            return None, None
        particle_color[i] = alt

    mismatched_real = sum(
        1 for i in candidate_indices if particle_color[i] != bar_color[bar_rows[i]]
    )
    matched_real_colors = {particle_color[i] for i in matched}
    if mismatched_real < _MIN_MISMATCHED_REAL_SOURCES:
        return None, None
    if len(matched_real_colors) < 2:
        return None, None

    stats = {
        "candidate_real": len(candidate_indices),
        "matched_real": len(matched),
        "mismatched_real": mismatched_real,
        "matched_real_colors": len(matched_real_colors),
    }
    return particle_color, stats


def _paint_sources(
    g: np.ndarray, bar_rows: list[int], particle_color: dict[int, int]
) -> None:
    """Place colored particles in col 0 while keeping col 1 as background."""
    g[:, 0] = BG
    for i, top in enumerate(bar_rows[:-1]):
        row = top + 1
        g[row, 0] = particle_color[i]
    g[:, 1] = BG


def _count_moved_sources(inp: np.ndarray, out: np.ndarray) -> int:
    H = inp.shape[0]
    return sum(
        1
        for r in range(1, H - 1, 2)
        if int(inp[r, 0]) != BG and int(out[r, 0]) == BG
    )


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D5 color-grouped variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c != BG], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
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

    particle_color, stats = _assign_particle_colors(
        g, channel_spec, real_indices, bar_rows, bar_color
    )
    if particle_color is None or stats is None:
        return None

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

    _paint_sources(g, bar_rows, particle_color)

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = _count_moved_sources(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if n_moved != stats["matched_real"]:
        return None
    if stats["mismatched_real"] < _MIN_MISMATCHED_REAL_SOURCES:
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
