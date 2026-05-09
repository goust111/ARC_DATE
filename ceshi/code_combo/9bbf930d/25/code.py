# Combo variant of 9bbf930d (index 25)
# combo_spec: D17/1+D7/1
# Name: 每走廊唯一色+终点改色为走廊色
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace point
# - Base task: magenta particles start in col 0, move right through BG, and only turn when a
#   colored blocker is directly ahead and exactly one perpendicular same-color corridor exists.
# - D7/1 changes the output write: the endpoint is no longer magenta; it becomes the color of the
#   terminal corridor containing the final BG cell.
# - D17/1 changes the input construction: each movable real corridor gets its own unique color.
#   The color may appear on multiple bar rows only when those rows belong to one connected logical
#   corridor; it is never reused by a different movable corridor.
# - Small trace example: a particle starts on a row flanked by color 4, moves right, hits a color-4
#   blocker, turns upward into a vertical color-4 corridor, and stops near the top edge. Originally
#   the output would place 6 at that stop cell. In this combo it places 4 there, and because no
#   other movable corridor uses 4, that endpoint color uniquely identifies the finished corridor.
#
# b) Complexity calibration
# - The fused task keeps the original corridor-tracing mechanics, fake corridors, edge stops, and
#   internal dead ends. The simplification is only that real entrances are isolated by fake rows so
#   independent movable corridors do not accidentally share a bar color through the shared-row
#   striped layout.
# - The generator still mixes straight and L-turn paths, so the solver must reason about direction
#   changes; the task is not reduced to pure recoloring.
#
# c) Hidden assumptions broken by the fusion
# - Original `_trace()` returned only coordinates because the output color was constant. D7/1 breaks
#   that: the solver must also return the final travel direction so the endpoint corridor color can
#   be read from the correct perpendicular bar pair.
# - Original generator freely reused colors across independent real channels and even relied on that
#   for same-color grouping. D17/1 breaks that assumption. In the fused generator, a real channel
#   may share its color only with bar rows that are part of the same logical corridor; independent
#   real entrances are separated by fake channels.
# - The solver now assumes the endpoint sits on a unique real corridor, so `_corridor_color()` reads
#   from the original input grid `g`, not the partially edited output grid `out`.
#
# d) Placement legality
# - Real entrances are chosen from non-adjacent channel rows. This is necessary because adjacent
#   channels share one bar row in the striped representation; if both were independent real
#   corridors they would be forced to share a color, violating D17/1.
# - L-turn vertical columns are reserved with a 3-column exclusion window, preventing overlaps,
#   accidental adjacent blockers, and ambiguous perpendicular choices.
# - Fake rows always have mismatched top/bottom colors at the entrance, so their particles never
#   qualify as movable real-corridor starts.
#
# e) Visual observability
# - Across examples, moved endpoints are no longer magenta. Each one blends into the surrounding
#   corridor bars and different moved particles usually end in different colors.
# - Because each movable corridor color is unique in that example, a human can infer that the final
#   color is not arbitrary recoloring: it is determined by the corridor the particle ends in.
#
# f) Visual cleanliness
# - No extra markers are introduced. The only output change is that the moved particle adopts the
#   local corridor color, which is visually simpler than overlaying a fixed magenta token.
# - Isolating real entrances with fake rows reduces stacked same-color structures that would make it
#   hard to tell whether repeated colors indicate one corridor or many.
#
# g) Color-change inferability
# - The recolor rule is structurally grounded. The endpoint color is always visible on the bar pair
#   flanking the final segment, and D17/1 ensures that this color identifies one movable corridor in
#   the sample instead of several unrelated ones.

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
AVAILABLE_COLORS = [0, 1, 2, 3, 4, 5, 8, 9]

_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]

# Turn probabilities
_STRAIGHT_PROB = 0.20
_L_INTERNAL_PROB = 0.35

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_TURNED = 1
_MIN_EXTRA_BAR_COLORS = 1
_MAX_RETRIES = 400


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg-like cell flanked by same-color bars perpendicular to direction."""
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


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int:
    """Return the bar-pair color of the real corridor containing the endpoint cell."""
    H, W = grid.shape

    if direction in {"left", "right"} and 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            return a
    if direction in {"up", "down"} and 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            return a

    candidates = []
    if 0 < r < H - 1:
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    if 0 < c < W - 1:
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
        if a == b and a not in {PARTICLE, BG}:
            candidates.append(a)
    assert len(set(candidates)) == 1, "final position is not on a unique real corridor"
    return int(candidates[0])


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Find the first valid real-corridor entrance for row r, or None."""
    _H, W = grid.shape
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


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, str]:
    """Trace a particle from (r,c) going right. Return final position and final direction."""
    H, W = grid.shape
    direction = "right"
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
        r, c, direction = options[0]
    return r, c, direction


def _moved_traces(grid: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    """Return moved particle traces as (start_row, end_r, end_c, corridor_color, end_dir)."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    traces = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, end_dir = _trace(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
        traces.append((r, end_r, end_c, _corridor_color(g, end_r, end_c, end_dir), end_dir))
    return traces


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles along uniquely colored corridors and recolor them at their final corridor.

    Input structure:
    - bg=7 and particles=6 fill col 0; col 1 is bg.
    - Even rows are colored bar rows, odd rows are channel rows.
    - Real movable corridors are bg cells flanked by same-color bars; fake rows use mismatched bars.

    Rule:
    - Each particle that has a valid real-corridor entrance moves right through bg cells.
    - When blocked by a colored cell, it turns 90 degrees into the unique perpendicular corridor
      whose two flanking bars match in color; otherwise it stops.
    - The start cell becomes bg, and the final cell is colored with the bar-pair color of the
      terminal corridor instead of staying magenta.

    Key insight:
    - The endpoint color is read from the corridor that contains the particle's final position, and
      in this combo each movable corridor color is unique, so the output color identifies which
      corridor the particle ended in.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c, end_dir = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = _corridor_color(g, end_r, end_c, end_dir)

    return out


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


def _choose_isolated_real_indices(n_channels: int, n_real: int) -> list[int] | None:
    """Choose non-adjacent channel indices for independent real corridors."""
    for _ in range(80):
        pool = list(range(n_channels))
        chosen = []
        while pool and len(chosen) < n_real:
            idx = random.choice(pool)
            chosen.append(idx)
            banned = {idx - 1, idx, idx + 1}
            pool = [p for p in pool if p not in banned]
        if len(chosen) == n_real:
            return sorted(chosen)
    return None


def _assign_bar_colors(
    n_channels: int,
    real_indices: list[int],
    real_colors: dict[int, int],
    fake_palette: list[int],
    bar_rows: list[int],
) -> dict[int, int] | None:
    """Assign a color to each bar row while keeping real corridors unique and fake corridors fake."""
    real_set = set(real_indices)
    if not real_indices:
        return None

    bar_color = {}
    first_real = real_indices[0]
    if first_real == 0:
        current = real_colors[0]
    else:
        prefer = [c for c in fake_palette if c != real_colors[first_real]]
        fallback = [c for c in AVAILABLE_COLORS if c != real_colors[first_real]]
        choices = prefer or fallback
        if not choices:
            return None
        current = random.choice(choices)
    bar_color[bar_rows[0]] = current

    for i in range(n_channels):
        top = bar_rows[i]
        bot = bar_rows[i + 1]
        top_color = int(bar_color[top])
        if i in real_set:
            if top_color != real_colors[i]:
                return None
            bar_color[bot] = top_color
            continue

        if i + 1 in real_set:
            bot_color = real_colors[i + 1]
            if bot_color == top_color:
                return None
            bar_color[bot] = bot_color
            continue

        prefer = [c for c in fake_palette if c != top_color]
        fallback = [c for c in AVAILABLE_COLORS if c != top_color]
        choices = prefer or fallback
        if not choices:
            return None
        bar_color[bot] = random.choice(choices)

    return bar_color


def _plan_real_corridors(
    real_indices: list[int], bar_start: int, H: int, W: int
) -> dict[int, dict[str, int | str | None]]:
    """Assign straight or L-turn behavior to each real corridor."""
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | None]] = {}
    for i in real_indices:
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        plans[i] = {
            "kind": "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge",
            "direction": random.choice(["up", "down"]),
            "c_v": c_v,
        }

    if all(plan["kind"] == "straight" for plan in plans.values()):
        for i in random.sample(real_indices, len(real_indices)):
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                continue
            plans[i] = {
                "kind": "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge",
                "direction": random.choice(["up", "down"]),
                "c_v": c_v,
            }
            break

    return plans


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


def _execute_plans(
    g: np.ndarray,
    H: int,
    plans: dict[int, dict[str, int | str | None]],
    bar_rows: list[int],
    bar_color: dict[int, int],
) -> None:
    """Render the planned real-corridor turns onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] == "straight":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])
        direction = str(plan["direction"])
        stop_at = None
        if plan["kind"] == "l_internal":
            if direction == "up":
                cands = list(range(2, top, 2))
            else:
                cands = list(range(bot + 2, H - 1, 2))
            stop_at = random.choice(cands) if cands else None
        _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D17/1 + D7/1 fused variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    max_real = min(len(AVAILABLE_COLORS) - 1, (n_channels + 1) // 2, _MAX_MOVED)
    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(max_real, n_real)
    if n_real < _MIN_MOVED:
        return None

    real_indices = _choose_isolated_real_indices(n_channels, n_real)
    if real_indices is None:
        return None

    real_palette = random.sample(AVAILABLE_COLORS, n_real)
    fake_palette = [c for c in AVAILABLE_COLORS if c not in real_palette]
    random.shuffle(real_palette)
    real_colors = {idx: color for idx, color in zip(real_indices, real_palette)}

    bar_color = _assign_bar_colors(n_channels, real_indices, real_colors, fake_palette, bar_rows)
    if bar_color is None:
        return None

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    plans = _plan_real_corridors(real_indices, bar_start, H, W)
    _execute_plans(g, H, plans, bar_rows, bar_color)

    for i in range(n_channels):
        if i in real_colors:
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
        traces = _moved_traces(g)
        _out = main(g)
    except Exception:
        return None

    if not (_MIN_MOVED <= len(traces) <= _MAX_MOVED):
        return None

    endpoint_colors = [color for _sr, _er, _ec, color, _dir in traces]
    if len(set(endpoint_colors)) != len(endpoint_colors):
        return None

    turned = sum(1 for start_r, end_r, _end_c, _color, end_dir in traces if end_r != start_r or end_dir != "right")
    if turned < _MIN_TURNED:
        return None

    bar_palette = {int(v) for v in bar_color.values()}
    if len(bar_palette - set(real_palette)) < _MIN_EXTRA_BAR_COLORS:
        return None

    if set(real_colors.values()) != set(endpoint_colors):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
