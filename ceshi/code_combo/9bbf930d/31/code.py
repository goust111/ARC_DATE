# Combo variant of 9bbf930d (index 31)
# combo_spec: D2/0+D9/1+D20/2
# Name: 固定手性+交叉口强制转弯+螺旋走廊

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and step trace
#    - The visual language stays close to 9bbf930d: bg=7, particle=6 in col 0,
#      orange col 1, colored bars on even rows, and only a few odd rows have a
#      valid same-color entrance near the left edge.
#    - The fused change is that every real path is now a clockwise T-junction
#      spiral. The particle starts on the middle odd row of a 5-row band:
#        1. move right on the middle row;
#        2. at an open T-junction, it must turn down instead of going straight;
#        3. at the next open T-junction, it must turn left;
#        4. then turn up at a third junction;
#        5. then turn right at a fourth junction and stop at an internal wall.
#    - Small local trace at the first corner:
#        old task:
#          current = (r, c-1), next = (r, c) is bg, so the particle just enters
#          (r, c) and keeps going; only a colored blocker could trigger a turn.
#        combo task:
#          current = (r, c-1), next = (r, c) is bg, the straight-ahead cell
#          (r, c+1) is also bg, the clockwise branch (r+1, c) is bg, and the
#          counterclockwise side (r-1, c) is blocked; therefore (r, c) is a
#          one-sided T-junction and the particle must immediately turn down into
#          (r+1, c). The turn direction itself is fixed by chirality, not by
#          color matching.
#    - Difference from the original rule:
#        original turning was "blocked cell + probe both perpendicular same-color
#        corridors"; combo turning is "at a one-sided open intersection, turn in
#        the fixed clockwise direction". Blocked turns still exist as a fallback,
#        also using the same clockwise chirality, but the generator mainly uses
#        forced open-junction turns so the D9 component is observable.
#
# b) Complexity calibration
#    - A single forced turn would be visually too weak and would look like an
#      ordinary L-corridor. Each real path therefore performs exactly four
#      successful clockwise turns inside one band.
#    - The grid always contains three such paths, all following the same
#      chirality and the same T-junction semantics. Repetition across three rows
#      makes the fused rule inferable from 2-4 ARC examples.
#    - Free parameters are intentionally small: each band only chooses three
#      columns `(x1, x2, x3)` subject to `x2 < x3 < x1`. This gives variation
#      without making the shape family unstructured.
#
# c) Hidden-assumption audit of `main()`
#    - Entrance detection from the original task is retained. Assumption:
#      intended active rows still have a same-color corridor entrance at col 2-3.
#      Generator guarantee: only the three middle rows of the spiral bands have
#      matching bar colors on both sides near the entrance; the partner rows in
#      the band are forced to have mismatched bar rows.
#    - Original `_trace()` assumed turning only happened on colored blockage and
#      that perpendicular legality depended on same-color bar pairs. That no
#      longer holds. The combo tracer replaces this with:
#        1. detect a forced junction ahead using local occupancy only;
#        2. if present, turn clockwise immediately;
#        3. otherwise move straight through free cells;
#        4. if blocked, attempt a clockwise blocked-turn fallback.
#    - New solver assumption: a forced junction should be one-sided, otherwise a
#      broad open area could create accidental turns. Generator guarantee: every
#      intended turn is encoded as a T where the clockwise branch is open and the
#      counterclockwise branch is colored.
#    - New solver assumption: loops should not occur. Generator guarantee: each
#      band is acyclic and the final segment ends at a blocker whose clockwise
#      side is also blocked.
#
# d) Placement legality
#    - Each spiral occupies one private 5-row band: odd rows `(top, mid, bot)`
#      plus the even rows between them. Bands are centered on rows 3, 9, 15 and
#      do not overlap.
#    - Column choices satisfy `bar_start+3 <= x2 < x3 < x1 <= W-4`, leaving room
#      for the forced-junction forward cell and for internal stop walls.
#    - Vertical BG carvings are only placed on specific columns, so junctions do
#      not merge across bands or go out of bounds.
#
# e) Visual observability
#    - The rule is readable by eye: at each corner the straight path visibly
#      continues, yet the output particle ends up on the clockwise branch. That
#      is the distinctive D9 signal.
#    - The endpoint also differs from a pure D2 "turn only on blockers" solver:
#      without the forced-turn rule the particle would overshoot and stop on the
#      middle row. Generator validation explicitly rejects such degeneracy.
#
# f) Visual cleanliness
#    - Real paths are separated into three wide bands; fake rows only get sparse
#      odd-row decoration outside those bands.
#    - Obstacles on odd rows are used in short contiguous runs to mark the closed
#      side of each T-junction, so the picture remains legible instead of noisy.
#
# g) Inference / color audit
#    - Colors are not used as ids or for hidden ordering. They only indicate
#      "blocked" versus "bar pair" structure.
#    - The only semantic distinction the human needs is geometric: which rows are
#      real entrances, and how the particle behaves at repeated one-sided open
#      intersections.
# ---------------------------------------------------------------------------

BG = 7
PARTICLE = 6

_DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
_RIGHT_TURN = {"right": "down", "down": "left", "left": "up", "up": "right"}
_LEFT_TURN = {v: k for k, v in _RIGHT_TURN.items()}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _is_free(value: int) -> bool:
    return int(value) in {BG, PARTICLE}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Original-task corridor predicate, kept for degeneration checks."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if not _is_free(int(grid[r, c])):
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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    h, w = grid.shape
    if not (0 < r < h - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _is_active_start(grid: np.ndarray, r: int) -> bool:
    h, w = grid.shape
    if not (0 < r < h - 1) or int(grid[r, 0]) != PARTICLE:
        return False
    entrance_col = _find_entrance_col(grid, r)
    if entrance_col is None:
        return False
    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(w) if int(grid[r - 1, c]) == color), w)
    bar_below = next((c for c in range(w) if int(grid[r + 1, c]) == color), w)
    return bar_above == bar_below and bar_above <= 3


def _forced_turn_ahead(
    grid: np.ndarray,
    r: int,
    c: int,
    direction: str,
    turn_map: dict[str, str] = _RIGHT_TURN,
    opposite_map: dict[str, str] = _LEFT_TURN,
) -> tuple[int, int, str] | None:
    """Detect a one-sided open T-junction ahead and return the forced turn."""
    h, w = grid.shape
    dr, dc = _DELTAS[direction]
    cr, cc = r + dr, c + dc
    if not (0 <= cr < h and 0 <= cc < w) or not _is_free(int(grid[cr, cc])):
        return None

    fr, fc = cr + dr, cc + dc
    if not (0 <= fr < h and 0 <= fc < w) or not _is_free(int(grid[fr, fc])):
        return None

    turn_dir = turn_map[direction]
    tdr, tdc = _DELTAS[turn_dir]
    tr, tc = cr + tdr, cc + tdc
    if not (0 <= tr < h and 0 <= tc < w) or not _is_free(int(grid[tr, tc])):
        return None

    other_dir = opposite_map[direction]
    odr, odc = _DELTAS[other_dir]
    orr, occ = cr + odr, cc + odc
    if 0 <= orr < h and 0 <= occ < w and _is_free(int(grid[orr, occ])):
        return None

    return tr, tc, turn_dir


def _trace_combo(
    grid: np.ndarray,
    r: int,
    c: int,
    turn_map: dict[str, str] = _RIGHT_TURN,
    opposite_map: dict[str, str] = _LEFT_TURN,
) -> tuple[int, int, int, int]:
    """Trace with fixed chirality, mandatory one-sided T-junction turns, and fallback blocked turns."""
    h, w = grid.shape
    direction = "right"
    forced_turns = 0
    blocked_turns = 0
    seen: set[tuple[int, int, str]] = set()
    for _ in range(h * w * 8):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        forced = _forced_turn_ahead(grid, r, c, direction, turn_map, opposite_map)
        if forced is not None:
            r, c, direction = forced
            forced_turns += 1
            continue

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if _is_free(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        direction = turn_map[direction]
        blocked_turns += 1
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if not _is_free(int(grid[nr, nc])):
            break
        r, c = nr, nc

    return r, c, forced_turns, blocked_turns


def _trace_fixed_no_force(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """D2-style baseline: only blocked cells trigger a clockwise turn."""
    h, w = grid.shape
    direction = "right"
    seen: set[tuple[int, int, str]] = set()
    for _ in range(h * w * 8):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if _is_free(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        direction = _RIGHT_TURN[direction]
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if not _is_free(int(grid[nr, nc])):
            break
        r, c = nr, nc
    return r, c


def _trace_counterclockwise_combo(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Counterfactual baseline used to verify chirality is actually controlling motion."""
    return _trace_combo(grid, r, c, turn_map=_LEFT_TURN, opposite_map=_RIGHT_TURN)[:2]


def _trace_original_rule(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original 9bbf930d rule for degeneration checks."""
    h, w = grid.shape
    direction = "right"
    seen: set[tuple[int, int, str]] = set()
    for _ in range(h * w * 8):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if _is_free(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) != 1:
            break
        r, c, direction = options[0]
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along clockwise spiral corridors with forced T-junction turns.

    Input structure:
    - bg=7, particle=6. Column 0 contains particles; column 1 is bg.
    - Even rows carry colored stripe bars. Only selected odd rows have a valid
      same-color entrance near the left edge and act as real spiral paths.

    Rule:
    - The particle starts moving right.
    - If the next cell is a one-sided open T-junction, it must turn clockwise
      into the branch instead of going straight.
    - Otherwise it goes straight through free cells; if straight ahead is blocked,
      it tries one clockwise blocked-turn as a fallback. It stops at the edge or
      when the clockwise continuation is unavailable.

    Key insight:
    - Motion is controlled by geometry, not color ids: real paths are repeated
      clockwise spirals whose open intersections still allow straight travel, but
      the particle always diverts into the clockwise branch.
    """
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if not _is_active_start(g, r):
            continue
        end_r, end_c, _, _ = _trace_combo(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_HEIGHT_OPTIONS = [19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START = 2
_N_EXTRA_COLORS = 4
_START_ROWS = [3, 9, 15]
_FAKE_DECOR_RANGE = (0, 2)
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _pick_color(extras: list[int], *exclude: int) -> int:
    banned = set(map(int, exclude))
    choices = [c for c in extras if c not in banned]
    return random.choice(choices if choices else extras)


def _build_row_colors(h: int, extras: list[int], start_rows: list[int]) -> dict[int, int]:
    even_rows = list(range(0, h, 2))
    offset = random.randrange(len(extras))
    row_colors = {r: extras[(offset + i) % len(extras)] for i, r in enumerate(even_rows)}
    locked: set[int] = set()

    for i, start_row in enumerate(start_rows):
        color = extras[i % len(extras)]
        top, bot = start_row - 1, start_row + 1
        row_colors[top] = color
        row_colors[bot] = color
        locked.update({top, bot})

    for odd_row in range(1, h - 1, 2):
        if odd_row in start_rows:
            continue
        top, bot = odd_row - 1, odd_row + 1
        if row_colors[top] != row_colors[bot]:
            continue
        if top not in locked:
            row_colors[top] = _pick_color(extras, row_colors[bot])
        elif bot not in locked:
            row_colors[bot] = _pick_color(extras, row_colors[top])
        else:
            row_colors[bot] = _pick_color(extras, row_colors[top])
    return row_colors


def _plan_spiral_columns(w: int) -> tuple[int, int, int] | None:
    if w < 16:
        return None
    for _ in range(80):
        x1 = random.randint(max(8, w - 6), w - 4)
        x2 = random.randint(5, x1 - 4)
        x3 = random.randint(x2 + 2, x1 - 2)
        if x2 - 1 > 1 and x3 + 1 < w:
            return x1, x2, x3
    return None


def _carve_vertical(g: np.ndarray, col: int, rows: list[int]) -> None:
    for rr in rows:
        g[rr, col] = BG


def _fill_row_segment(g: np.ndarray, row: int, start: int, end: int, color: int) -> None:
    if start <= end:
        g[row, start : end + 1] = color


def _build_spiral_band(
    g: np.ndarray,
    extras: list[int],
    start_row: int,
    columns: tuple[int, int, int],
) -> dict[str, int]:
    """Render one clockwise 4-turn spiral using one-sided open T-junctions."""
    h, w = g.shape
    del h
    top = start_row - 2
    mid = start_row
    bot = start_row + 2
    x1, x2, x3 = columns

    blocker_a = random.choice(extras)
    blocker_b = _pick_color(extras, blocker_a)
    blocker_c = _pick_color(extras, blocker_a, blocker_b)

    # Encode the closed side of each T-junction on the odd rows.
    _fill_row_segment(g, top, 2, x2 - 1, blocker_a)
    _fill_row_segment(g, top, x3 + 1, w - 1, blocker_b)
    _fill_row_segment(g, mid, x1 + 2, w - 1, blocker_c)
    _fill_row_segment(g, bot, x1 + 1, w - 1, blocker_b)

    # Carve vertical continuations used by the forced-junction detector.
    _carve_vertical(g, x1, [mid + 1, bot, bot + 1])
    _carve_vertical(g, x2, [top - 1, top, top + 1, mid, bot - 1])

    # Preserve the free horizontal runs that the particle actually uses or could
    # have used under the old inertia rule.
    g[mid, 1 : x1 + 2] = BG
    g[bot, x2 - 1 : x1 + 1] = BG
    g[top, x2 : x3 + 1] = BG

    # Keep the intended stopping cell blocked both ahead and on the clockwise side.
    g[top + 1, x3] = blocker_c

    return {"x1": x1, "x2": x2, "x3": x3, "end_row": top, "end_col": x3}


def _add_fake_decor(
    g: np.ndarray,
    extras: list[int],
    row_colors: dict[int, int],
    protected_rows: set[int],
) -> None:
    h, w = g.shape
    for r in range(1, h - 1, 2):
        if r in protected_rows:
            continue
        n = random.randint(*_FAKE_DECOR_RANGE)
        if n == 0:
            continue
        cols = random.sample(range(max(_BAR_START + 3, w // 2), w), k=min(n, w - max(_BAR_START + 3, w // 2)))
        for c in cols:
            if g[r, c] != BG:
                continue
            g[r, c] = random.choice([row_colors[r - 1], row_colors[r + 1], random.choice(extras)])


def _candidate_start_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(1, grid.shape[0] - 1, 2) if _is_active_start(grid, r)]


def _moved_rows(inp: np.ndarray, out: np.ndarray) -> list[int]:
    rows = []
    for r in range(inp.shape[0]):
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE:
            rows.append(r)
    return rows


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    start_rows = [r for r in _START_ROWS if r + 2 < h]
    if len(start_rows) != 3:
        return None

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    row_colors = _build_row_colors(h, extras, start_rows)
    for br, color in row_colors.items():
        g[br, _BAR_START:w] = color

    band_infos = []
    protected_rows: set[int] = set()
    for start_row in start_rows:
        columns = _plan_spiral_columns(w)
        if columns is None:
            return None
        band_infos.append(_build_spiral_band(g, extras, start_row, columns))
        protected_rows.update({start_row - 2, start_row, start_row + 2})

    _add_fake_decor(g, extras, row_colors, protected_rows)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    candidate_rows = _candidate_start_rows(g)
    if candidate_rows != start_rows:
        return None

    out = main(g)
    moved = _moved_rows(g, out)
    if moved != start_rows:
        return None

    endpoints = set()
    for row, info in zip(start_rows, band_infos):
        end_r, end_c, forced_turns, blocked_turns = _trace_combo(g, row, 0)
        if (end_r, end_c) != (info["end_row"], info["end_col"]):
            return None
        if forced_turns != 4 or blocked_turns != 1:
            return None
        if _trace_fixed_no_force(g, row, 0) == (end_r, end_c):
            return None
        if _trace_original_rule(g, row, 0) == (end_r, end_c):
            return None
        if _trace_counterclockwise_combo(g, row, 0) == (end_r, end_c):
            return None
        endpoints.add((end_r, end_c))

    if len(endpoints) != len(start_rows):
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
