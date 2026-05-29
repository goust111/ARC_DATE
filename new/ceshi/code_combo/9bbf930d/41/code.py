# Combo variant of 9bbf930d (index 41)
# combo_spec: D9/2+D14/0
# Name: 交叉口信号灯+边缘图例

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
TURN_SIGNAL = 8
STRAIGHT_SIGNAL = 9
LEGEND_H = 2

_TRAVERSABLE = {BG, PARTICLE, TURN_SIGNAL, STRAIGHT_SIGNAL}
_DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
_DIR_ORDER = ["up", "right", "down", "left"]
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_TURN_FROM_SHAPE = {
    frozenset({(0, 0), (1, 0), (1, 1)}): "ccw",
    frozenset({(0, 1), (1, 0), (1, 1)}): "cw",
}


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
# - The base geometry stays the same as 9bbf930d: the maze occupies the top H-2 rows,
#   even rows are colored bars, odd rows are particle channels, column 0 is magenta,
#   and column 1 is orange background.
# - Two visible symbol systems are added:
#   1. Local signal lights inside the maze:
#      TURN_SIGNAL=8 means "at this crossing, do not continue straight; turn now".
#      STRAIGHT_SIGNAL=9 means "at this crossing, ignore side branches; continue".
#   2. A bottom-edge 2-row legend band:
#      for each corridor color, a same-color 2x2 L-shape encodes the global turn
#      preference of that color: CCW or CW.
# - Concrete trace on a horizontal corridor of color C:
#   original task:
#     ... BG BG [color wall] ...
#     the particle looks at the blocker, then turns only if geometry exposes a unique
#     perpendicular mouth.
#   fused variant:
#     ... BG [signal] BG ...
#     at the signal cell, straight and vertical continuation are both geometrically
#     visible. The signal decides straight-vs-turn. If it says turn, the legend entry
#     of corridor color C decides whether the turn is CCW or CW relative to the current
#     heading. Thus:
#       STRAIGHT_SIGNAL => pass through the crossing.
#       TURN_SIGNAL + legend(C)=CCW => turn up when moving right.
#       TURN_SIGNAL + legend(C)=CW  => turn down when moving right.
# - The crucial difference is that neither symbol system is sufficient by itself:
#   signal lights answer "turn or not", while the legend answers "which side".
#
# b) complexity calibration
# - The local alphabet is kept binary: only two signal colors.
# - The global legend is also binary per color: only CCW or CW.
# - Each sample reuses the same corridor color on multiple bar rows, so a human can
#   observe repeated evidence that "this color always turns the same way when a turn
#   signal appears".
# - Turn signals are rendered as real crossings with open forward continuation and both
#   immediate vertical mouths open. Straight signals are rendered on comparable crossings
#   with at least one available side branch. This makes the visual difference salient.
#
# c) implicit assumptions audit for main()
# - Original _has_corridor assumed only BG/PARTICLE were traversable. The fused task
#   puts signal pixels on traversed cells, so both signal colors must count as passable.
# - Original _trace treated the next non-BG cell as a blocker that triggers a turn test.
#   That breaks here because signal cells are meant to be entered, not treated as walls.
# - D14 already broke the assumption that geometry alone disambiguates turns; the combo
#   keeps that. At a TURN_SIGNAL crossing both up and down mouths can be valid, and the
#   legend selects one. Solver logic must therefore parse the legend first.
# - The legend occupies a dedicated bottom band, so main() must slice the maze and keep
#   the legend unchanged in the output.
#
# d) placement legality
# - The legend is isolated in the bottom 2 rows; the maze is generated only above it.
# - Signal columns are reserved with a 3-column exclusion window, preventing overlapping
#   crossings from sharing the same vertical carve-out.
# - Every signal column is placed far enough from the left and right edges so the signal
#   cell has visible approach/exit context.
# - Turn branches are carved only within bounds. Internal-stop turns deliberately leave a
#   colored bar farther along the chosen direction so the particle stops without leaking
#   into another corridor.
#
# e) visual observability
# - A solver looking at examples can see three repeated motifs:
#   1. identical 2x2 color legends along the bottom edge,
#   2. local 8/9 single-pixel lights on crossings,
#   3. particles of the same corridor color turning the same global way whenever the
#      local light asks them to turn.
# - No rule depends on numeric color ordering. Only visible shape, placement, and color
#   repetition matter.
#
# f) visual cleanliness
# - Signals occupy single channel cells; legends occupy a separate band.
# - Crossings are spaced apart horizontally, and fake decorations never use signal colors.
# - The two symbol systems therefore do not overlap or camouflage each other.
#
# g) change inferability
# - Corridor colors are not remapped. The legend uses the same color as the controlled
#   corridor, so the binding is explicit and localizable by eye.
# - The only semantic new colors are 8 and 9, and they recur with consistent roles.
#
# Quantitative behavior checks used by the generator
# - _count_turn_signal_controlled_junctions(): counts turn-signal crossings where the
#   particle could also continue straight and both perpendicular mouths are immediately
#   valid, so the output depends jointly on local signal + global legend.
# - _count_straight_signal_overrides(): counts straight-signal crossings where a side
#   branch exists, so the local signal genuinely suppresses a visible turn option.
#
# New solver assumptions introduced here and guaranteed by generate_input()
# - Every traversed TURN_SIGNAL belongs to a real corridor whose color appears in the
#   bottom legend.
# - The legend band is exactly 2 rows high and each legend entry is a 3-pixel 2x2 L.
# - Active signal crossings are only placed on horizontal corridors; the solver supports
#   other headings generically, but the generator does not rely on them.


def _rotate(direction: str, turn: str) -> str:
    idx = _DIR_ORDER.index(direction)
    step = -1 if turn == "ccw" else 1
    return _DIR_ORDER[(idx + step) % 4]


def _same_color_component(
    legend: np.ndarray, start_r: int, start_c: int, seen: np.ndarray
) -> list[tuple[int, int]]:
    color = int(legend[start_r, start_c])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in _DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < legend.shape[0] and 0 <= nc < legend.shape[1] and not seen[nr, nc]:
                if int(legend[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return cells


def _parse_legend(legend: np.ndarray) -> dict[int, str]:
    assert legend.shape[0] == LEGEND_H
    seen = np.zeros_like(legend, dtype=bool)
    turn_map: dict[int, str] = {}
    for r in range(LEGEND_H):
        for c in range(legend.shape[1]):
            color = int(legend[r, c])
            if color == BG or seen[r, c]:
                continue
            comp = _same_color_component(legend, r, c, seen)
            if len(comp) != 3:
                continue
            rs = [rr for rr, _ in comp]
            cs = [cc for _, cc in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if (r0, r1) != (0, 1) or c1 - c0 != 1:
                continue
            shape = frozenset((rr - r0, cc - c0) for rr, cc in comp)
            turn = _TURN_FROM_SHAPE.get(shape)
            if turn is not None:
                turn_map[color] = turn
    return turn_map


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    H, W = grid.shape
    if int(grid[r, c]) not in _TRAVERSABLE:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in _TRAVERSABLE:
        return a
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    return _corridor_color(grid, r, c, direction) is not None


def _trace(grid: np.ndarray, turn_map: dict[int, str], r: int, c: int) -> tuple[int, int]:
    H, W = grid.shape
    direction = "right"
    seen: set[tuple[int, int, str]] = set()
    for _ in range(700):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break

        nxt = int(grid[nr, nc])
        if nxt in {BG, PARTICLE}:
            r, c = nr, nc
            continue

        if nxt == STRAIGHT_SIGNAL:
            r, c = nr, nc
            continue

        if nxt == TURN_SIGNAL:
            color = _corridor_color(grid, r, c, direction)
            turn = turn_map.get(color) if color is not None else None
            if turn is None:
                break
            target_dir = _rotate(direction, turn)
            if target_dir not in _PERPENDICULAR[direction]:
                break
            tdr, tdc = _DELTAS[target_dir]
            tr, tc = nr + tdr, nc + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, target_dir):
                r, c = nr, nc
                direction = target_dir
                continue
            break

        turn = turn_map.get(nxt)
        if turn is None:
            break
        target_dir = _rotate(direction, turn)
        if target_dir not in _PERPENDICULAR[direction]:
            break
        tdr, tdc = _DELTAS[target_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, target_dir):
            r, c = tr, tc
            direction = target_dir
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through striped corridors using signals and a color legend.

    Input structure:
    - The top H-2 rows are the corridor maze from 9bbf930d.
    - The bottom 2 rows are a legend band made of same-color 2x2 L-shapes.
    - Particles=6 fill column 0 of the maze. BG=7. Signal cells inside the maze are
      TURN_SIGNAL=8 and STRAIGHT_SIGNAL=9.

    Rule:
    - A particle starts on an odd maze row and moves right through traversable cells.
    - On STRAIGHT_SIGNAL it keeps its current direction.
    - On TURN_SIGNAL it must turn 90 degrees; the turn side is determined by the legend
      entry of the current corridor color.
    - On an ordinary colored blocker, the legend likewise gives the preferred side; if
      that side is not a legal corridor, the particle stops.
    - The particle stops at the grid edge or whenever the required continuation is absent.

    Key insight:
    - The local signal answers "turn here or go straight here", while the bottom legend
      answers "if this corridor color turns, which side does it prefer".
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    assert H > LEGEND_H + 2

    maze = g[:-LEGEND_H, :]
    legend = g[-LEGEND_H:, :]
    turn_map = _parse_legend(legend)

    out = g.copy()
    for r in range(1, maze.shape[0] - 1):
        if int(maze[r, 0]) != PARTICLE:
            continue

        entrance_col = None
        for c in range(2, W):
            if int(maze[r, c]) != BG:
                continue
            above, below = int(maze[r - 1, c]), int(maze[r + 1, c])
            if above == below and above not in _TRAVERSABLE:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_color = int(maze[r - 1, entrance_col])
        bar_above = next((c for c in range(W) if int(maze[r - 1, c]) == bar_color), W)
        bar_below = next((c for c in range(W) if int(maze[r + 1, c]) == bar_color), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(maze, turn_map, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_PLAY_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_SIGNAL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)

_PLAIN_STRAIGHT_PROB = 0.2
_SIGNAL_STRAIGHT_PROB = 0.3
_TURN_INTERNAL_PROB = 0.35

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 7
_MIN_TURN_SIGNAL_CONTROLLED = 1
_MIN_STRAIGHT_SIGNAL_OVERRIDES = 1
_MAX_RETRIES = 500


def _allocate_signal_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    c_min = bar_start + 3
    c_max = W - 4
    candidates = [
        c for c in range(c_min, c_max + 1)
        if all((c + d) not in used_cols for d in _SIGNAL_WINDOW)
    ]
    if not candidates:
        return None
    x = random.choice(candidates)
    used_cols.update({x + d for d in _SIGNAL_WINDOW})
    return x


def _open_crossing_mouths(
    maze: np.ndarray, top: int, bot: int, x: int
) -> None:
    maze[top, x] = BG
    maze[bot, x] = BG


def _carve_branch(
    maze: np.ndarray, top: int, bot: int, x: int, direction: str, stop_at: int | None
) -> None:
    H = maze.shape[0]
    if direction == "up":
        for r_br in range(0, top + 1, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            maze[r_br, x] = BG
    else:
        for r_br in range(bot, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            maze[r_br, x] = BG


def _build_turn_signal_plan(
    maze: np.ndarray,
    top: int,
    bot: int,
    row: int,
    x: int,
    direction: str,
    kind: str,
) -> None:
    maze[row, x] = TURN_SIGNAL
    _open_crossing_mouths(maze, top, bot, x)
    stop_at = None
    if kind == "turn_internal":
        if direction == "up":
            cands = list(range(0, top, 2))
        else:
            cands = list(range(bot + 2, maze.shape[0], 2))
        stop_at = random.choice(cands) if cands else None
    _carve_branch(maze, top, bot, x, direction, stop_at)


def _build_straight_signal_plan(
    maze: np.ndarray,
    top: int,
    bot: int,
    row: int,
    x: int,
) -> None:
    maze[row, x] = STRAIGHT_SIGNAL
    _open_crossing_mouths(maze, top, bot, x)


def _place_legend(
    grid: np.ndarray, maze_h: int, colors: list[int], turn_pref: dict[int, str]
) -> None:
    band = grid[maze_h:, :]
    band[:, :] = BG
    legend_colors = colors[:]
    random.shuffle(legend_colors)
    total_w = len(legend_colors) * 2 + max(0, len(legend_colors) - 1)
    max_start = grid.shape[1] - total_w
    start = 0 if max_start <= 0 else random.randint(0, max_start)
    c = start
    for color in legend_colors:
        shape = (
            {(0, 0), (1, 0), (1, 1)}
            if turn_pref[color] == "ccw"
            else {(0, 1), (1, 0), (1, 1)}
        )
        for dr, dc in shape:
            band[dr, c + dc] = color
        c += 3


def _count_turn_signal_controlled_junctions(grid: np.ndarray) -> int:
    maze = grid[:-LEGEND_H, :]
    turn_map = _parse_legend(grid[-LEGEND_H:, :])
    H, W = maze.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(maze[r, c]) != TURN_SIGNAL:
                continue
            color = None
            if c - 1 >= 0:
                color = _corridor_color(maze, r, c - 1, "right")
            if color is None and c + 1 < W:
                color = _corridor_color(maze, r, c + 1, "right")
            turn = turn_map.get(color) if color is not None else None
            if turn is None:
                continue
            forward_ok = c + 1 < W and int(maze[r, c + 1]) in _TRAVERSABLE
            up_ok = _has_corridor(maze, r - 1, c, "up")
            down_ok = _has_corridor(maze, r + 1, c, "down")
            preferred = "up" if turn == "ccw" else "down"
            preferred_ok = up_ok if preferred == "up" else down_ok
            if forward_ok and up_ok and down_ok and preferred_ok:
                count += 1
    return count


def _count_straight_signal_overrides(grid: np.ndarray) -> int:
    maze = grid[:-LEGEND_H, :]
    H, W = maze.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(maze[r, c]) != STRAIGHT_SIGNAL:
                continue
            forward_ok = c + 1 < W and int(maze[r, c + 1]) in _TRAVERSABLE
            up_ok = _has_corridor(maze, r - 1, c, "up")
            down_ok = _has_corridor(maze, r + 1, c, "down")
            if forward_ok and (up_ok or down_ok):
                count += 1
    return count


def _plan_real_channels(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    turn_pref: dict[int, str],
) -> dict[int, dict[str, object]] | None:
    if len(real_indices) < 2:
        return None

    plans: dict[int, dict[str, object]] = {}
    used_cols: set[int] = set()
    must_turn = random.choice(real_indices)
    remaining = [i for i in real_indices if i != must_turn]
    must_straight = random.choice(remaining)
    priority_order = [must_turn, must_straight] + [
        i for i in real_indices if i not in {must_turn, must_straight}
    ]

    for i in priority_order:
        top = bar_rows[i]
        color = bar_color[top]
        direction = "up" if turn_pref[color] == "ccw" else "down"

        if i == must_turn:
            kind = "turn_internal" if random.random() < _TURN_INTERNAL_PROB else "turn_edge"
        elif i == must_straight:
            kind = "signal_straight"
        else:
            p = random.random()
            if p < _PLAIN_STRAIGHT_PROB:
                kind = "plain_straight"
            elif p < _PLAIN_STRAIGHT_PROB + _SIGNAL_STRAIGHT_PROB:
                kind = "signal_straight"
            else:
                kind = "turn_internal" if random.random() < _TURN_INTERNAL_PROB else "turn_edge"

        if kind == "plain_straight":
            plans[i] = {"kind": kind}
            continue

        x = _allocate_signal_col(bar_start, W, used_cols)
        if x is None:
            if i in {must_turn, must_straight}:
                return None
            plans[i] = {"kind": "plain_straight"}
            continue
        plans[i] = {"kind": kind, "x": x, "direction": direction}

    return plans


def _execute_plans(
    maze: np.ndarray,
    bar_rows: list[int],
    plans: dict[int, dict[str, object]],
) -> None:
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind == "plain_straight":
            continue

        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        x = int(plan["x"])

        if kind == "signal_straight":
            _build_straight_signal_plan(maze, top, bot, row, x)
        else:
            _build_turn_signal_plan(
                maze, top, bot, row, x, str(plan["direction"]), kind
            )


def _gen_attempt() -> np.ndarray | None:
    maze_h = random.choice(_PLAY_HEIGHT_OPTIONS)
    H = maze_h + LEGEND_H
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    maze = g[:-LEGEND_H, :]
    maze[:, 0] = PARTICLE

    bar_rows = list(range(0, maze_h, 2))
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
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    turn_pref = {color: random.choice(["ccw", "cw"]) for color in set(bar_color.values())}
    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        maze[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    plans = _plan_real_channels(real_indices, bar_rows, bar_color, bar_start, W, turn_pref)
    if plans is None:
        return None
    _execute_plans(maze, bar_rows, plans)

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
            if int(maze[r, c_decor]) == BG:
                maze[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    maze[:, 0] = PARTICLE
    maze[:, 1] = BG
    _place_legend(g, maze_h, sorted(set(bar_color.values())), turn_pref)

    try:
        out = main(g)
    except Exception:
        return None

    moved = sum(
        1
        for r in range(maze_h)
        if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None

    signal_values = {int(v) for v in g.flatten() if int(v) in {TURN_SIGNAL, STRAIGHT_SIGNAL}}
    if signal_values != {TURN_SIGNAL, STRAIGHT_SIGNAL}:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if _count_turn_signal_controlled_junctions(g) < _MIN_TURN_SIGNAL_CONTROLLED:
        return None
    if _count_straight_signal_overrides(g) < _MIN_STRAIGHT_SIGNAL_OVERRIDES:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
