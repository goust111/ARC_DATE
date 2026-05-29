# Variant of 9bbf930d
# Dimension: D15 — 初始方向 (path)
# Change: 所有粒子初始方向固定向右。... → 初始方向由边缘标记指定

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
ARROW = 2
PASSABLE = {BG, PARTICLE, ARROW}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


# Design analysis for D15
# a) Concrete behavior:
#    The original puzzle has a global striped corridor system and all particles sit on the
#    left border, implicitly meaning "start by moving right". In this variant each moving
#    particle still sits on the border, but it now has one dedicated adjacent arrow-marker
#    pixel (color ARROW) one cell inside the grid. The relative offset particle->marker is
#    the initial direction: marker on the right means start right, marker below means start
#    down, etc. After the first direction is chosen, the old corridor rule is unchanged:
#    move straight through passable cells; when a colored wall blocks movement, try the two
#    perpendicular corridor continuations and turn iff exactly one continuation is valid.
#    Small trace example:
#    - Original: particle at (r, 0) always calls _trace(..., "right"), so the row enters the
#      horizontal corridor regardless of any visible marker.
#    - Variant: particle at (0, c) with marker at (1, c) calls _trace(..., "down"). This is
#      the same physical corridor after a 90° rotation, but now the start direction is read
#      from the grid instead of being hard-coded in main().
# b) Complexity calibration:
#    A fully mixed-orientation puzzle within one grid would require rebuilding the striped
#    corridor generator into several locally rotated sublayouts, which would change much more
#    than D15 and create visual clutter. To keep the task close to 9bbf930d, generation is
#    still done in the original canonical "enter from the left" layout; then the entire grid
#    is rotated by 0/90/180/270 degrees and every start gets an explicit arrow marker. This
#    makes the direction change clearly observable while preserving the original path logic.
# c) Original main() assumptions that break under D15:
#    - Looping only over rows in col 0 assumes all starts are on the left border. Not valid.
#    - Finding a horizontal entrance using rows above/below assumes initial motion is rightward.
#      Not valid once starts can come from top/right/bottom.
#    - _trace() hard-coding direction="right" is the core broken assumption.
#    - Treating only {BG, PARTICLE} as passable would make the arrow marker act like a wall.
#      In this variant the marker is decorative/passable and only encodes direction.
# d) Placement legality:
#    Generator only places starts on real source channels from the canonical generator
#    (straight / L / U-source, never U-target). In canonical orientation those starts are
#    odd rows on the left border, never corners; after whole-grid rotation they remain edge
#    starts and the marker remains one cell inward. Because the grid is rotated as a whole,
#    all corridor geometry, seals, turn disambiguation and bounds remain legal automatically.
# e) Visual observability:
#    The explicit ARROW marker is the only new symbol. Across samples the whole layout appears
#    entering from different edges; the marker sits between the particle and the corridor, so
#    a human can see "the adjacent marker indicates where the particle initially goes". This is
#    much easier to infer from 2-3 examples than a hidden orientation parameter.
# f) Visual cleanliness:
#    Using one dedicated arrow color avoids overloading corridor colors. Markers only appear
#    next to starts on the outer border, never inside interior turns, so they do not overlap
#    with bar logic or decorate fake corridors.
# g) Color inference:
#    No color mapping rule is introduced. Direction is encoded only by relative position of a
#    single fixed-color marker, which is visually explicit and does not depend on hidden color
#    ids or ordering.
# New solver assumptions and generator guarantees:
#    - Each movable particle is on the border and has exactly one 4-neighbor arrow marker.
#    - The arrow marker color appears only in these local start pairs.
#    - Starts are pairwise disjoint and come from real source channels, so their traces do not
#      require tie-breaking beyond the original "at most one perpendicular corridor" rule.


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color bars perpendicular to direction."""
    h, w = grid.shape
    if int(grid[r, c]) not in PASSABLE:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in PASSABLE


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    """Trace a particle from (r,c) using the given initial direction."""
    h, w = grid.shape
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in PASSABLE:
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


def _edge_starts(grid: np.ndarray) -> list[tuple[int, int, str]]:
    """Return border particles with a unique adjacent arrow marker."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts: list[tuple[int, int, str]] = []
    seen = set()
    edge_cells = []
    for c in range(w):
        edge_cells.append((0, c))
        edge_cells.append((h - 1, c))
    for r in range(1, h - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, w - 1))

    for r, c in edge_cells:
        if int(g[r, c]) != PARTICLE or (r, c) in seen:
            continue
        seen.add((r, c))
        dirs = []
        for name, (dr, dc) in _DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == ARROW:
                dirs.append(name)
        if len(dirs) == 1:
            starts.append((r, c, dirs[0]))
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through same-color corridors; initial direction comes from edge arrows."""
    g = np.array(grid, dtype=int)
    out = g.copy()

    starts = _edge_starts(g)
    end_positions = []
    for r, c, direction in starts:
        end_positions.append(_trace(g, r, c, direction))

    for r, c, _ in starts:
        out[r, c] = BG
    for end_r, end_c in end_positions:
        out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_BAR_COLORS = 3

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
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
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
    """Assign straight or L-turn plans to real channels not yet planned."""
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
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    bar_colors: list[int],
    internal_stop: bool,
) -> None:
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot)."""
    g[row, c_v] = bar_color[top]

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

    other_colors = [c for c in bar_colors if c != bar_color[top]]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, w - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = bar_color[top]


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    bar_colors: list[int],
) -> None:
    """Render all planned turns onto the canonical grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = int(plan["c_v"])
        if kind in {"l_edge", "l_internal"}:
            stop_at = None
            direction = str(plan["direction"])
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, h - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, h, top, bot, row, c_v, direction, bar_color, stop_at)
        elif kind == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                w,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                bar_colors,
                bool(plan.get("internal_stop", False)),
            )


def _source_channels(plans: dict[int, dict[str, object]], real_indices: list[int]) -> list[int]:
    """Channels that receive a starting particle in the canonical orientation."""
    return [i for i in real_indices if str(plans[i]["kind"]) != "u_target"]


def _place_starts_canonical(
    g: np.ndarray,
    bar_rows: list[int],
    source_channels: list[int],
) -> None:
    """Canonical D15 starts: particle on left edge, marker one cell to the right."""
    for i in source_channels:
        row = bar_rows[i] + 1
        g[row, 0] = PARTICLE
        g[row, 1] = ARROW


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D15 variant."""
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    pool = [c for c in range(10) if c not in (BG, PARTICLE, ARROW)]
    bar_colors = random.sample(pool, _N_BAR_COLORS)

    g = np.full((h, w), BG, dtype=int)
    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(bar_colors)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in bar_colors if c != bar_color[top]])
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, w, used_vcols, plans)
    _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, bar_colors)

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

    sources = _source_channels(plans, real_indices)
    if not (_MIN_MOVED <= len(sources) <= _MAX_MOVED):
        return None
    _place_starts_canonical(g, bar_rows, sources)

    rot_k = random.randrange(4)
    g = np.rot90(g, rot_k)

    starts = _edge_starts(g)
    if len(starts) != len(sources):
        return None
    if len({direction for _, _, direction in starts}) != 1:
        return None

    try:
        out = main(g)
    except Exception:
        return None

    moved = 0
    for r, c, _ in starts:
        if int(out[r, c]) != PARTICLE:
            moved += 1
    if moved != len(starts):
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
