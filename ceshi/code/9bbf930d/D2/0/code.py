# Variant of 9bbf930d
# Dimension: D2 — 转弯规则 (path)
# Change: 被彩色格挡住时检查两个垂直方向，选择有同色条对走廊的一侧转90度；必须恰好一个方向有效（无歧义断裂）。... → 固定手性转弯

from __future__ import annotations

import random
import numpy as np

# ---------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# 0) Reference-task note
#    - The prompt cites `e87109e9` as the turning-rule inspiration.
#    - Under the allowed paths for this task, no readable solver `.py` for
#      `e87109e9` was available, so that code-reading step had to be skipped.
#    - The implementation therefore follows the prompt's turning-rule description
#      directly: preserve the striped 9bbf930d layout, but replace "search for a
#      matching perpendicular corridor" with "always rotate in one fixed
#      chirality". Here the fixed chirality is clockwise / right-turn.
#
# 1) Concrete visual form and behavior trace
#    - The input still looks like the original family: orange bg (7), magenta
#      particles (6) in column 0, orange column 1, colored horizontal stripe-bars
#      on even rows starting at column 2.
#    - Only three odd rows are "real entrances": their bar rows immediately above
#      and below share the same color at the left entrance, so `main()` still
#      selects them as particles that should move.
#    - The crucial change is inside the maze body. Instead of building vertical
#      turns that are recognizable by same-color side walls, the generator carves
#      a clockwise polyline:
#        start row: move right
#        hit a colored blocker -> turn down
#        hit a colored blocker -> turn left
#        hit a colored blocker -> turn up
#        hit a colored blocker -> turn right
#        hit a colored blocker -> try down, then stop
#    - Example at a first corner:
#        old rule on row r, col x:
#          obstacle at (r, x+1); solver probes both (r-1, x) and (r+1, x);
#          only a target whose side walls match in color is legal.
#        new rule on the same pixels:
#          obstacle at (r, x+1); solver ignores wall colors and simply tries
#          the clockwise side, i.e. (r+1, x) when moving right.
#        difference:
#          the generator deliberately breaks the old "same-color side walls"
#          around those even-row turn cells, so the old rule would stop (or be
#          ambiguous), while the new rule still follows the carved spiral.
#
# 2) Complexity calibration
#    - A single right-turn would be too weak and visually close to the original
#      L-turn samples. Each moving particle therefore performs a full 5-turn
#      clockwise trace: four successful chirality changes plus one final blocked
#      chirality attempt that terminates the path.
#    - The spiral is repeated three times per grid. Repetition matters: with
#      2-4 training pairs, a human can infer that the turning choice is global
#      and chirality-based, not color-matching-based.
#    - Colors are *not* used as hidden ids. They remain purely visual obstacles;
#      the only stable cue is the repeated clockwise geometry.
#
# 3) Old-code dependency audit
#    - `_has_corridor(...)` encoded the old task's semantic gate. It is kept only
#      for degeneration checks in the generator, not for actual tracing.
#    - `_trace(...)` in the original depended on:
#        a) the blocked-ahead cell being colored;
#        b) at least one perpendicular cell being a same-color corridor;
#        c) ambiguity being illegal.
#      In this variant only (a) remains. The new trace:
#        a) moves through bg/particle cells;
#        b) on colored blockage, rotates clockwise exactly once;
#        c) enters the turned cell iff that cell is free; otherwise stops.
#    - `main()` originally used entrance detection tied to same-color bars near
#      the left edge. That part remains valid and intentionally unchanged, because
#      it selects which particles are active without depending on the turn rule.
#    - The generator must therefore guarantee:
#        a) exactly the intended start rows still satisfy the old entrance test;
#        b) later turns do *not* accidentally recreate old-style matched
#           perpendicular corridors.
#
# 4) Placement legality
#    - Each spiral occupies a private 5-row band: odd rows (top, mid, bot) and
#      the two even rows between them. Different spirals are separated by one
#      unused even row, so no path cells overlap.
#    - Columns satisfy `2 < x2 < x3 < x1 < W-1`, leaving room for blockers and
#      for the clockwise turns to be within bounds.
#    - All carved path cells are bg, all blockers are colored, and the final
#      right-turn attempt is forced to fail because the downward side cell is left
#      colored. That prevents loops.
#
# 5) Visual observability
#    - The new rule is visible because particles now settle at the top arm of a
#      clockwise spiral rather than at old-style matched L/U corridor endpoints.
#    - The generator explicitly rejects degenerate samples where the old rule and
#      the new rule would yield the same endpoints.
#
# 6) Visual cleanliness
#    - The grid stays in one visual language: striped bars plus sparse blockers.
#      No legends or extra symbols are introduced.
#    - Decorative odd-row noise is allowed only outside the 5-row spiral bands, so
#      it cannot truncate a real spiral or create visually confusing overlaps.
#
# 7) New implicit assumptions and how the generator enforces them
#    - Assumption: the clockwise side cell at each corner is free.
#      Enforcement: each spiral explicitly carves that cell.
#    - Assumption: the final attempted clockwise turn is blocked, so the particle
#      terminates instead of entering another corridor.
#      Enforcement: the side cell at the last corner is left colored.
#    - Assumption: only intended rows pass entrance detection.
#      Enforcement: only the three mid rows have same-color bar pairs at the left
#      entrance; all other odd rows are constructed to fail that test.
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
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    if not _is_free(int(grid[r, c])):
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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    H, W = grid.shape
    if not (0 < r < H - 1):
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _trace_fixed_turn(
    grid: np.ndarray, r: int, c: int, direction: str = "right"
) -> tuple[int, int, int]:
    """Trace with the new D2 rule: on colored blockage, always turn clockwise."""
    H, W = grid.shape
    turns = 0
    seen: set[tuple[int, int, str]] = set()
    for _ in range(H * W * 8):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if _is_free(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        direction = _RIGHT_TURN[direction]
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        turns += 1
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if not _is_free(int(grid[nr, nc])):
            break
        r, c = nr, nc
    return r, c, turns


def _trace_original_rule(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original trace logic, but ambiguity simply stops instead of asserting."""
    H, W = grid.shape
    direction = "right"
    seen: set[tuple[int, int, str]] = set()
    for _ in range(H * W * 8):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
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


def _candidate_start_rows(grid: np.ndarray) -> list[int]:
    H, W = grid.shape
    rows = []
    for r in range(1, H - 1, 2):
        if int(grid[r, 0]) != PARTICLE:
            continue
        entrance_col = _find_entrance_col(grid, r)
        if entrance_col is None:
            continue
        color = int(grid[r - 1, entrance_col])
        bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == color), W)
        bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            rows.append(r)
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """Move selected particles through the striped maze using fixed clockwise turns."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = _find_entrance_col(g, r)
        if entrance_col is None:
            continue

        color = int(g[r - 1, entrance_col])
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == color), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == color), W)
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
_WIDTH_RANGE = (15, 21)
_N_EXTRA_COLORS = 4
_BAR_START = 2
_START_ROWS = [3, 9, 15]
_MAX_RETRIES = 300
_MIN_DISTINCT_COLORS = 6


def _pick_color(extras: list[int], *exclude: int) -> int:
    banned = set(map(int, exclude))
    choices = [c for c in extras if c not in banned]
    return random.choice(choices if choices else extras)


def _build_row_colors(H: int, extras: list[int], start_rows: list[int]) -> dict[int, int]:
    even_rows = list(range(0, H, 2))
    offset = random.randrange(len(extras))
    row_colors = {r: extras[(offset + i) % len(extras)] for i, r in enumerate(even_rows)}
    locked = set()

    for i, start_row in enumerate(start_rows):
        color = extras[i % len(extras)]
        top, bot = start_row - 1, start_row + 1
        row_colors[top] = color
        row_colors[bot] = color
        locked.update({top, bot})

    for odd_row in range(1, H - 1, 2):
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


def _plan_spiral_columns(W: int) -> tuple[int, int, int]:
    x1 = random.randint(max(8, W - 6), W - 3)
    x2 = random.randint(4, x1 - 5)
    x3 = random.randint(x2 + 2, x1 - 3)
    return x1, x2, x3


def _break_old_vertical_turn(
    g: np.ndarray, row_colors: dict[int, int], extras: list[int], even_row: int, col: int
) -> None:
    """Invalidate the original same-color side-wall test at one even-row target."""
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
    block3 = _pick_color(extras, row_colors[top - 1], row_colors[top + 1], block1, block2)
    g[mid, x1 + 1] = block1
    g[bot, x2 - 1] = block2
    g[top, x3 + 1] = block3

    for even_row in (mid - 1, mid + 1):
        _break_old_vertical_turn(g, row_colors, extras, even_row, x1)
    for even_row in (bot - 1, bot + 1):
        _break_old_vertical_turn(g, row_colors, extras, even_row, x2)


def _add_fake_row_decor(g: np.ndarray, extras: list[int], start_rows: list[int]) -> None:
    H, W = g.shape
    protected = set()
    for r in start_rows:
        protected.update({r - 2, r, r + 2})
    for r in range(1, H - 1, 2):
        if r in protected:
            continue
        n = random.randint(0, 2)
        for _ in range(n):
            c = random.randint(max(W // 2, 5), W - 2)
            if g[r, c] == BG:
                g[r, c] = random.choice(extras)


def _moved_rows(inp: np.ndarray, out: np.ndarray) -> list[int]:
    rows = []
    for r in range(inp.shape[0]):
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE:
            rows.append(r)
    return rows


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    start_rows = [r for r in _START_ROWS if r + 2 < H]
    if len(start_rows) != 3:
        return None

    row_colors = _build_row_colors(H, extras, start_rows)
    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    for br, color in row_colors.items():
        g[br, _BAR_START:W] = color

    for start_row in start_rows:
        _carve_spiral(g, row_colors, extras, start_row, _plan_spiral_columns(W))

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

    diff_count = 0
    turn_counts = []
    end_positions = set()
    for r in start_rows:
        new_end = _trace_fixed_turn(g, r, 0)
        old_end = _trace_original_rule(g, r, 0)
        diff_count += int((new_end[0], new_end[1]) != old_end)
        turn_counts.append(new_end[2])
        end_positions.add((new_end[0], new_end[1]))

    if diff_count != len(start_rows):
        return None
    if any(turns != 5 for turns in turn_counts):
        return None
    if len(end_positions) != len(start_rows):
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
