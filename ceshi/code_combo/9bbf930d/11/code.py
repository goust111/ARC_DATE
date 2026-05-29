# Combo variant of 9bbf930d (index 11)
# combo_spec: D2/0+D20/2
# Name: 固定手性转弯+螺旋走廊

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    - The left-edge entrance test stays recognizable from the original task:
#      column 0 is all particles, column 1 is bg, and only some odd rows have a
#      same-color bar pair starting at column 2 or 3.
#    - After entry, the rule changes completely. When the particle sees a
#      colored blocker in front of it, it does not inspect both perpendicular
#      corridors. It always rotates in one fixed chirality: clockwise / right.
#    - Each real path is rendered as a compact clockwise spiral with five turn
#      attempts:
#        row mid: move right until blocker at x1+1, then turn down
#        row bot: move left until blocker at x2-1, then turn up
#        row top: move right until blocker at x3+1, then try to turn down
#        final down-step is blocked, so the particle stops on the top arm
#      The path therefore has four successful 90-degree turns plus one final
#      blocked chirality attempt. This is already outside the original 0-2 turn
#      family and also makes the fixed-hand rule visually obvious.
#    - Difference from the original task:
#      original: a colored blocker triggers a search for a same-color
#      perpendicular corridor; exactly one side must satisfy that semantic test.
#      combo: a colored blocker triggers a deterministic clockwise turn; wall
#      colors at the corner are irrelevant.
#
# b) Complexity calibration
#    - A single L-turn would be too close to the source task. Each moved particle
#      therefore performs exactly five turn attempts in a 3-row spiral ribbon.
#    - The same clockwise spiral motif is repeated three times per grid. This
#      repetition gives the solver enough evidence, from only a few examples, to
#      infer both the fixed chirality and the multi-turn spiral topology.
#    - Free parameters are kept narrow: only three spiral columns vary per path,
#      and the turn direction never varies inside a sample.
#
# c) Implicit-assumption audit for main()
#    - Entrance detection in `main()` assumes real rows can still be recognized
#      by a same-color bar pair near the left edge. The generator preserves that.
#    - The old `_trace()` assumed a turn was legal only if a same-color corridor
#      existed on one perpendicular side. That assumption is removed; the new
#      trace only assumes:
#        1. the clockwise side cell at each successful corner is free;
#        2. the final clockwise side cell is blocked to terminate the spiral;
#        3. no loop is intended, so a state-repeat guard is still prudent.
#    - Because turns no longer depend on side-wall colors, the generator must
#      explicitly prevent accidental degeneration back to the original rule. It
#      does so by perturbing the even-row flank colors near each corner and by
#      rejecting any sample where the original solver would reach the same end.
#
# d) Placement legality
#    - Each spiral occupies a private 5-row band centered on one odd start row:
#      odd rows `(start-2, start, start+2)` plus the two even rows between them.
#    - Start rows are spaced by 6, so different spirals do not overlap.
#    - Spiral columns satisfy `bar_start+2 <= x2 < x3 < x1 < W-1`, which leaves
#      room for blockers and keeps all turn cells in-bounds.
#
# e) Visual observability
#    - The moved particles end on the upper arm of clockwise spirals, not at the
#      end of same-color L/U corridors. This is easy to see directly in the
#      input/output pairs.
#    - The generator rejects any instance where the old rule and the combo rule
#      agree on a real path endpoint, so the examples visibly teach the new law.
#
# f) Visual cleanliness
#    - The grid keeps the original striped visual language: colored even-row bars
#      plus sparse blockers on odd rows.
#    - Decorative noise is only placed on odd rows outside the active spiral
#      bands, so it does not visually collide with the real paths.
#
# g) Change inferability
#    - No hidden color mapping is introduced. Colors only mark obstacles/bars.
#    - The inferable structure is geometric: "on every blockage, turn the same
#      way" and "the real corridors can coil inward for more than three turns".
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
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _is_free(value: int) -> bool:
    return int(value) in {BG, PARTICLE}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Original-task corridor predicate, kept only for degeneration checks."""
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
    if not (0 < r < h - 1):
        return None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _candidate_start_rows(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    rows = []
    for r in range(1, h - 1, 2):
        if int(grid[r, 0]) != PARTICLE:
            continue
        entrance_col = _find_entrance_col(grid, r)
        if entrance_col is None:
            continue
        color = int(grid[r - 1, entrance_col])
        bar_above = next((c for c in range(w) if int(grid[r - 1, c]) == color), w)
        bar_below = next((c for c in range(w) if int(grid[r + 1, c]) == color), w)
        if bar_above == bar_below and bar_above <= 3:
            rows.append(r)
    return rows


def _trace_fixed_turn(
    grid: np.ndarray, r: int, c: int, direction: str = "right"
) -> tuple[int, int, int]:
    """Trace with the combo rule: on blockage, always rotate clockwise once."""
    h, w = grid.shape
    turns = 0
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
        turns += 1
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if not _is_free(int(grid[nr, nc])):
            break
        r, c = nr, nc

    return r, c, turns


def _trace_original_rule(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original trace logic, but ambiguity simply stops instead of asserting."""
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
    """Move selected particles through striped corridors with fixed clockwise turns.

    Input structure:
    - `7` is bg and `6` is the particle color.
    - Column 0 is filled with particles and column 1 is bg.
    - Even rows carry colored bars; only odd rows whose adjacent bar rows match
      near the left edge are active entrances.

    Rule:
    - An active particle starts at `(row, 0)` and moves through bg/particle cells.
    - If the forward cell is colored, the particle always turns clockwise
      (`right -> down -> left -> up -> right`) and attempts exactly that side.
    - If the turned cell is blocked or out of bounds, the particle stops there.

    Key insight:
    - The true corridors are not defined by same-color corner matching anymore.
      The consistent cue is a repeated clockwise spiral, so solving the task
      means following a fixed chirality through multi-turn paths.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = _find_entrance_col(g, r)
        if entrance_col is None:
            continue

        color = int(g[r - 1, entrance_col])
        bar_above = next((c for c in range(w) if int(g[r - 1, c]) == color), w)
        bar_below = next((c for c in range(w) if int(g[r + 1, c]) == color), w)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c, _ = _trace_fixed_turn(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_HEIGHT_OPTIONS = [19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_START_ROW_PATTERNS = {
    19: [[3, 9, 15]],
    21: [[3, 9, 15], [5, 11, 17]],
    23: [[3, 9, 15], [5, 11, 17]],
}
_FAKE_DECOR_RANGE = (0, 2)
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _pick_color(extras: list[int], *exclude: int) -> int:
    banned = set(map(int, exclude))
    choices = [c for c in extras if c not in banned]
    return random.choice(choices if choices else extras)


def _choose_start_rows(h: int) -> list[int]:
    return random.choice(_START_ROW_PATTERNS[h])


def _build_row_colors(h: int, extras: list[int], start_rows: list[int]) -> dict[int, int]:
    even_rows = list(range(0, h, 2))
    offset = random.randrange(len(extras))
    row_colors = {r: extras[(offset + i) % len(extras)] for i, r in enumerate(even_rows)}
    locked = set()

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


def _plan_spiral_columns(bar_start: int, w: int) -> tuple[int, int, int] | None:
    left_min = bar_start + 2
    x1_min = max(left_min + 6, w - 6)
    x1_max = w - 3
    if x1_min > x1_max:
        return None

    x1 = random.randint(x1_min, x1_max)
    x2_low = left_min
    x2_high = x1 - 5
    if x2_low > x2_high:
        return None
    x2 = random.randint(x2_low, x2_high)

    x3_low = x2 + 2
    x3_high = x1 - 3
    if x3_low > x3_high:
        return None
    x3 = random.randint(x3_low, x3_high)
    return x1, x2, x3


def _break_old_vertical_turn(
    g: np.ndarray, row_colors: dict[int, int], extras: list[int], even_row: int, col: int
) -> None:
    """Invalidate the original same-color corner test near one turn column."""
    if not (0 <= even_row < g.shape[0] and 1 < col < g.shape[1] - 1):
        return
    neighbor = col - 1 if col - 1 > 1 else col + 1
    opposite = col + 1 if neighbor == col - 1 else col - 1
    avoid = {row_colors.get(even_row, BG), int(g[even_row, opposite]), BG, PARTICLE}
    g[even_row, neighbor] = _pick_color(extras, *avoid)


def _carve_spiral(
    g: np.ndarray,
    row_colors: dict[int, int],
    extras: list[int],
    start_row: int,
    columns: tuple[int, int, int],
) -> None:
    """Carve a 5-turn clockwise spiral inside one 5-row private band."""
    top, mid, bot = start_row - 2, start_row, start_row + 2
    x1, x2, x3 = columns

    g[mid, 1 : x1 + 1] = BG
    for rr in range(mid + 1, bot + 1):
        g[rr, x1] = BG
    g[bot, x2 : x1 + 1] = BG
    for rr in range(top, bot):
        g[rr, x2] = BG
    g[top, x2 : x3 + 1] = BG

    block1 = _pick_color(extras, row_colors[mid - 1], row_colors[mid + 1])
    block2 = _pick_color(extras, row_colors[bot - 1], row_colors[bot + 1], block1)
    block3 = _pick_color(
        extras, row_colors[top - 1], row_colors[top + 1], block1, block2
    )
    g[mid, x1 + 1] = block1
    g[bot, x2 - 1] = block2
    g[top, x3 + 1] = block3

    for even_row in (mid - 1, mid + 1):
        _break_old_vertical_turn(g, row_colors, extras, even_row, x1)
    for even_row in (bot - 1, bot + 1):
        _break_old_vertical_turn(g, row_colors, extras, even_row, x2)


def _add_fake_row_decor(g: np.ndarray, extras: list[int], start_rows: list[int]) -> None:
    h, w = g.shape
    protected = set()
    for r in start_rows:
        protected.update({r - 2, r, r + 2})

    for r in range(1, h - 1, 2):
        if r in protected:
            continue
        n = random.randint(*_FAKE_DECOR_RANGE)
        for _ in range(n):
            c = random.randint(max(w // 2, 5), w - 2)
            if g[r, c] == BG:
                g[r, c] = random.choice(extras)


def _moved_rows(inp: np.ndarray, out: np.ndarray) -> list[int]:
    rows = []
    for r in range(inp.shape[0]):
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE:
            rows.append(r)
    return rows


def _variant_metrics(g: np.ndarray) -> dict[str, object]:
    """Quantify whether the combo behavior is present on the generated sample."""
    rows = _candidate_start_rows(g)
    turn_counts = []
    diverged = 0
    end_positions = []
    for r in rows:
        new_end_r, new_end_c, turns = _trace_fixed_turn(g, r, 0)
        old_end = _trace_original_rule(g, r, 0)
        turn_counts.append(turns)
        end_positions.append((new_end_r, new_end_c))
        diverged += int((new_end_r, new_end_c) != old_end)
    return {
        "rows": rows,
        "turn_counts": turn_counts,
        "diverged": diverged,
        "unique_ends": len(set(end_positions)),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    start_rows = _choose_start_rows(h)
    row_colors = _build_row_colors(h, extras, start_rows)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    for br, color in row_colors.items():
        g[br, bar_start:w] = color

    for start_row in start_rows:
        columns = _plan_spiral_columns(bar_start, w)
        if columns is None:
            return None
        _carve_spiral(g, row_colors, extras, start_row, columns)

    _add_fake_row_decor(g, extras, start_rows)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    candidate_rows = _candidate_start_rows(g)
    if candidate_rows != start_rows:
        return None

    out = main(g)
    moved = _moved_rows(g, out)
    if moved != start_rows:
        return None

    metrics = _variant_metrics(g)
    if metrics["diverged"] != len(start_rows):
        return None
    if metrics["unique_ends"] != len(start_rows):
        return None
    if any(turns != 5 for turns in metrics["turn_counts"]):
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
