# Combo variant of 9bbf930d (index 5)
# combo_spec: D6/0+D14/0
# Name: 目标色停止+边缘图例指示转弯
#
# Design analysis for this combo:
# 1) Concrete form and behavior trace
#    - I realize the "edge legend" using a fixed 2-row band at the BOTTOM. The
#      upper area remains the striped bar/corridor maze from 9bbf930d.
#    - Each corridor color receives a same-color 2x2 L-shaped legend glyph:
#      {(0,0),(1,0),(1,1)} means CCW, {(0,1),(1,0),(1,1)} means CW.
#    - Real moving paths are generated so their first non-straight junction is
#      structurally ambiguous: both immediate vertical mouths are legal. The
#      particle therefore must read the legend color of the blocking cell to
#      decide which side to enter.
#    - A fixed target color is added as isolated dots on traversable cells that
#      lie AFTER the first legend-controlled turn of each moving path. Once the
#      particle enters such a target cell, it stops immediately even if the path
#      could continue.
#    - Small example: moving right in row r, the particle reaches column c_v-1
#      with color 4 blocking at (r,c_v). Original task: exactly one of up/down
#      corridors exists. This combo: both exist, legend color 4 says CCW/CW, so
#      the particle turns accordingly; several steps later it lands on target 8
#      and halts there instead of continuing to the old edge/dead-end.
#
# 2) Complexity calibration
#    - The new symbolic freedom is deliberately binary (CW vs CCW). The target
#      cue is also binary (target or not). This keeps the puzzle inferable while
#      still forcing a two-step chain: choose the branch, then stop early.
#    - The same corridor colors repeat across multiple rows and the legend shows
#      all used corridor colors, so humans get repeated support for the mapping.
#    - Real paths still mix L-turns and U-turns. I avoid many extra markings in
#      the maze itself; only one target dot is placed per moving real path.
#
# 3) Implicit-assumption audit against the original solver
#    - Original `_trace()` assumed `len(options) <= 1` at a turn. That premise
#      is intentionally broken here, so `_trace()` must instead decode the legend
#      color and test only the preferred perpendicular direction.
#    - Original `_has_corridor()` and forward motion assumed only BG/PARTICLE are
#      traversable. The target cell must now also count as traversable.
#    - Original `main()` treated the whole grid as maze. Here the bottom 2 rows
#      are reserved for legend parsing, so solving happens only on `grid[:-2]`.
#    - Original entrance detection looked specifically for early BG cells. That
#      remains valid because the generator never places targets on the entrance
#      segment or inside the legend band.
#
# 4) Placement legality
#    - The legend occupies only the bottom 2 rows and is spaced with background
#      gaps, so it never overlaps the maze.
#    - Turn columns use a reserved 3-column window to avoid corridor collisions.
#    - Targets are sampled only from already traced real paths, excluding starts,
#      natural endpoints, boundary cells, and the pre-turn entrance segment.
#
# 5) Visual observability
#    - The input visibly differs from the original task in two ways: same-color
#      edge symbols exist, and isolated target dots appear inside real corridors.
#    - The output visibly demonstrates both mechanics: particles choose branches
#      consistent with the legend color, then disappear onto target dots before
#      the old geometric endpoint.
#
# 6) Visual cleanliness
#    - The legend is isolated in its own band; targets are single dots on open
#      corridor cells. No object stack shares the same local pixels.
#
# 7) Color inference
#    - No hidden color ordering is used. Corridor color -> legend symbol mapping
#      is directly visible by same-color matching. The target uses one reserved,
#      non-corridor color.
#
# 8) Quantified non-degeneracy
#    - `_variant_stats()` measures `dual_controlled`: paths whose final stop is a
#      target, whose no-target path would continue further, and whose endpoint
#      changes if the legend directions are flipped. Requiring several such paths
#      prevents degeneration to either the plain target-stop or plain legend-only
#      variants.
#
# 9) New solver assumptions and generator guarantees
#    - Assumption A: the legend band height is exactly 2. Generator always uses 2.
#    - Assumption B: every corridor color used in the maze has exactly one legend
#      glyph. Generator places one glyph per used corridor color.
#    - Assumption C: every moving real path encounters at least one legend-read
#      turn before its target. Generator enforces this through plan choices and
#      target placement after the first controlled turn.
#    - Assumption D: target cells are not part of the entrance test. Generator
#      excludes the early entrance segment from target candidates.

from __future__ import annotations

import random
import numpy as np

BG = 7
PARTICLE = 6
TARGET = 8
LEGEND_H = 2

_TRAVERSABLE = {PARTICLE, BG, TARGET}
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


def _rotate(direction: str, turn: str) -> str:
    idx = _DIR_ORDER.index(direction)
    step = -1 if turn == "ccw" else 1
    return _DIR_ORDER[(idx + step) % 4]


def _flip_turn_map(turn_map: dict[int, str]) -> dict[int, str]:
    return {color: ("cw" if turn == "ccw" else "ccw") for color, turn in turn_map.items()}


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
    """Read bottom-edge 2x2 L symbols as color -> turn preference."""
    assert legend.shape[0] == LEGEND_H
    seen = np.zeros_like(legend, dtype=bool)
    turn_map: dict[int, str] = {}
    for r in range(legend.shape[0]):
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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal corridor cell for row r, if any."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG, TARGET}:
            bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == above), W)
            bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == below), W)
            if bar_above == bar_below and bar_above <= 3:
                return c
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) is traversable and flanked by same-color bars."""
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
    turn_map: dict[int, str],
    r: int,
    c: int,
    *,
    stop_on_target: bool = True,
    return_meta: bool = False,
):
    """Trace a particle with legend-directed turns and optional target-stop logic."""
    H, W = grid.shape
    direction = "right"
    path = [(r, c)]
    seen: set[tuple[int, int, str]] = set()
    controlled_turns = 0
    first_controlled_index: int | None = None

    for _ in range(600):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break

        ahead = int(grid[nr, nc])
        if ahead in _TRAVERSABLE:
            r, c = nr, nc
            path.append((r, c))
            if stop_on_target and ahead == TARGET:
                break
            continue

        turn = turn_map.get(ahead)
        if turn is None:
            break
        target_dir = _rotate(direction, turn)
        if target_dir not in _PERPENDICULAR[direction]:
            break

        tdr, tdc = _DELTAS[target_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, target_dir):
            r, c, direction = tr, tc, target_dir
            path.append((r, c))
            controlled_turns += 1
            if first_controlled_index is None:
                first_controlled_index = len(path) - 1
            if stop_on_target and int(grid[r, c]) == TARGET:
                break
            continue
        break

    if return_meta:
        return {
            "end": (r, c),
            "path": path,
            "controlled_turns": controlled_turns,
            "first_controlled_index": first_controlled_index,
        }
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through a legend-guided corridor maze with target stops.

    Input structure:
    - The upper area is the original striped maze: even rows are colored bars and
      odd rows are particle channels; col 0 is particle and col 1 is background.
    - The bottom 2 rows form a separate legend band of same-color L symbols.
    - Target-color pixels appear on traversable maze cells, never inside the legend.

    Rule:
    - A particle moves straight through background and target cells.
    - If it enters a target cell, it stops immediately.
    - When a colored blocker is ahead, the blocker color is looked up in the edge
      legend to decide whether the 90-degree turn preference is CCW or CW; the
      particle takes only that preferred perpendicular corridor if it exists.
    - If the preferred turn does not exist, or the particle exits the maze, motion stops.

    Key insight:
    - Geometry alone no longer determines the turn because the generator opens
      both immediate mouths at real junctions; the legend resolves that choice,
      and the target color then short-circuits the otherwise longer legend-chosen path.
    """
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    assert H > LEGEND_H + 2

    maze = g[:-LEGEND_H, :]
    legend = g[-LEGEND_H:, :]
    turn_map = _parse_legend(legend)

    out = g.copy()
    for r in range(1, maze.shape[0] - 1):
        if _find_entrance_col(maze, r) is None:
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
_N_EXTRA_COLORS = 3

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [2, 3, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.0
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_TARGET_STOPS = 3
_MIN_LEGEND_CONTROLLED = 3
_MIN_DUAL_CONTROLLED = 3
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
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
    plans: dict[int, dict[str, object]],
    turn_pref: dict[int, str],
) -> None:
    """Pair some same-color channels into U-turns without dropping moved paths below the target-stop minimum."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)

    pair_budget = max(0, len(real_indices) - _MIN_TARGET_STOPS)
    pairs_used = 0
    for color, chans in by_color.items():
        if len(chans) < 2 or pairs_used >= pair_budget:
            continue
        random.shuffle(chans)
        max_pairs_here = min(len(chans) // 2, pair_budget - pairs_used)
        n_pairs = random.randint(0, max_pairs_here)
        for p in range(n_pairs):
            a, b = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if turn_pref[color] == "ccw":
                source, target, direction = b, a, "up"
            else:
                source, target, direction = a, b, "down"
            plans[source] = {
                "kind": "u",
                "target": target,
                "direction": direction,
                "c_v": c_v,
                "internal_stop": internal_stop,
            }
            plans[target] = {"kind": "u_target"}
            pairs_used += 1
            if pairs_used >= pair_budget:
                break


def _plan_remaining(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
    turn_pref: dict[int, str],
) -> None:
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
        color = bar_color[bar_rows[i]]
        direction = "up" if turn_pref[color] == "ccw" else "down"
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


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
    """Open both immediate mouths; only the legend-selected side extends into a real path."""
    g[row, c_v] = color
    for br in (top, bot):
        g[br, c_v - 1] = BG
        g[br, c_v - 2] = bar_color[br]
        g[br, c_v] = bar_color[br]

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
    H: int,
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
    """Open both source mouths; the legend-selected side leads to the paired channel."""
    g[row, c_v] = color

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = BG
        g[top, c_v - 2] = bar_color[top]
        g[top, c_v] = bar_color[top]
        g[bot, c_v - 2] = bar_color[bot]
        g[bot, c_v] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = BG
        g[top, c_v - 2] = bar_color[top]
        g[top, c_v] = bar_color[top]
        g[bot, c_v - 2] = bar_color[bot]
        g[bot, c_v] = bar_color[bot]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


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
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in {"l_edge", "l_internal"}:
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        else:
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                H,
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


def _place_legend(g: np.ndarray, maze_h: int, colors: list[int], turn_pref: dict[int, str]) -> None:
    band = g[maze_h:, :]
    band[:, :] = BG

    legend_colors = colors[:]
    random.shuffle(legend_colors)
    total_w = len(legend_colors) * 2 + max(0, len(legend_colors) - 1)
    max_start = g.shape[1] - total_w
    start = 0 if max_start <= 0 else random.randint(0, max_start)
    c = start
    for color in legend_colors:
        shape = {(0, 0), (1, 0), (1, 1)} if turn_pref[color] == "ccw" else {(0, 1), (1, 0), (1, 1)}
        for dr, dc in shape:
            band[dr, c + dc] = color
        c += 3


def _choose_target_cells(grid: np.ndarray) -> dict[int, tuple[int, int]] | None:
    """Pick one target after the first legend-controlled turn of each moving real path."""
    maze = grid[:-LEGEND_H, :]
    legend = grid[-LEGEND_H:, :]
    turn_map = _parse_legend(legend)
    H, W = maze.shape
    target_cells: dict[int, tuple[int, int]] = {}
    used_cells: set[tuple[int, int]] = set()

    for r in range(1, H - 1):
        entrance_col = _find_entrance_col(maze, r)
        if entrance_col is None:
            continue
        meta = _trace(maze, turn_map, r, 0, stop_on_target=False, return_meta=True)
        if meta["end"] == (r, 0):
            continue
        if meta["controlled_turns"] < 1 or meta["first_controlled_index"] is None:
            return None

        min_idx = max(entrance_col + 2, int(meta["first_controlled_index"]) + 1)
        candidates: list[tuple[int, int]] = []
        for idx in range(min_idx, len(meta["path"]) - 1):
            rr, cc = meta["path"][idx]
            if (rr, cc) in used_cells:
                continue
            if not (0 < rr < H - 1 and 1 < cc < W - 1):
                continue
            if int(maze[rr, cc]) != BG:
                continue
            candidates.append((rr, cc))
        if not candidates:
            return None

        pool = candidates[len(candidates) // 2 :] or candidates
        choice = random.choice(pool)
        target_cells[r] = choice
        used_cells.add(choice)

    return target_cells if len(target_cells) >= _MIN_TARGET_STOPS else None


def _count_legend_controlled_junctions(grid: np.ndarray) -> int:
    maze = grid[:-LEGEND_H, :]
    turn_map = _parse_legend(grid[-LEGEND_H:, :])
    H, W = maze.shape
    count = 0
    for r in range(H):
        for c in range(W):
            if int(maze[r, c]) not in _TRAVERSABLE:
                continue
            for direction, (dr, dc) in _DELTAS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                ahead = int(maze[nr, nc])
                if ahead in _TRAVERSABLE or ahead not in turn_map:
                    continue
                options = 0
                for turn_dir in _PERPENDICULAR[direction]:
                    tdr, tdc = _DELTAS[turn_dir]
                    tr, tc = r + tdr, c + tdc
                    if 0 <= tr < H and 0 <= tc < W and _has_corridor(maze, tr, tc, turn_dir):
                        options += 1
                if options == 2:
                    count += 1
    return count


def _variant_stats(grid: np.ndarray) -> dict[str, int]:
    """Measure whether both legend control and target-stop materially affect the sample."""
    maze = grid[:-LEGEND_H, :]
    turn_map = _parse_legend(grid[-LEGEND_H:, :])
    flipped_map = _flip_turn_map(turn_map)

    moved = 0
    target_stops = 0
    legend_paths = 0
    dual_controlled = 0

    for r in range(1, maze.shape[0] - 1):
        if _find_entrance_col(maze, r) is None:
            continue
        natural = _trace(maze, turn_map, r, 0, stop_on_target=False, return_meta=True)
        combo = _trace(maze, turn_map, r, 0, stop_on_target=True, return_meta=True)
        alt_combo = _trace(maze, flipped_map, r, 0, stop_on_target=True, return_meta=True)

        if natural["end"] != (r, 0):
            moved += 1
        if natural["controlled_turns"] > 0:
            legend_paths += 1

        combo_end = combo["end"]
        is_target_stop = combo_end != natural["end"] and int(maze[combo_end]) == TARGET
        if is_target_stop:
            target_stops += 1
        if is_target_stop and natural["controlled_turns"] > 0 and alt_combo["end"] != combo_end:
            dual_controlled += 1

    return {
        "moved": moved,
        "target_stops": target_stops,
        "legend_paths": legend_paths,
        "dual_controlled": dual_controlled,
    }


def _gen_attempt() -> np.ndarray | None:
    maze_h = random.choice(_PLAY_HEIGHT_OPTIONS)
    H = maze_h + LEGEND_H
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE, TARGET)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    maze = g[:maze_h, :]
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
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    used_colors = sorted(set(bar_color.values()))
    turn_pref = {color: random.choice(["ccw", "cw"]) for color in used_colors}
    if len(used_colors) >= 2 and len(set(turn_pref.values())) == 1:
        turn_pref[random.choice(used_colors)] = "cw" if next(iter(turn_pref.values())) == "ccw" else "ccw"

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        maze[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans, turn_pref)
    _plan_remaining(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans, turn_pref)
    _execute_plans(maze, maze_h, W, plans, bar_rows, bar_color, bar_start, extras)

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
    _place_legend(g, maze_h, used_colors, turn_pref)

    target_cells = _choose_target_cells(g)
    if target_cells is None:
        return None
    for rr, cc in target_cells.values():
        maze[rr, cc] = TARGET

    try:
        out = main(g)
    except Exception:
        return None

    stats = _variant_stats(g)
    if stats["target_stops"] != stats["moved"]:
        return None
    if stats["legend_paths"] != stats["moved"]:
        return None
    if stats["target_stops"] < _MIN_TARGET_STOPS:
        return None
    if stats["dual_controlled"] < _MIN_DUAL_CONTROLLED:
        return None

    n_moved = sum(
        1 for r in range(maze_h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if _count_legend_controlled_junctions(g) < _MIN_LEGEND_CONTROLLED:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
