# Variant of 9bbf930d
# Dimension: D16 — 惯性与间隙穿越 (path)
# Change: 粒子保持当前方向不变地穿越背景间隙（包括交叉口间隙），直到遇到彩色像素被阻挡才尝试转弯。... → 间隙中减速
#
# Design analysis
# - Source-task note: the prompt points to 16de56c4 for the gcd-spacing idea, but
#   no solver file for 16de56c4 exists in the allowed local workspace, so direct
#   solver reading is not possible and is skipped.
# - Concrete behavior:
#   - The visual layout stays close to the original task: odd rows are horizontal
#     channels, even rows are colored bars, col 0 is filled with magenta particles,
#     and real channels are still recognized by equal colors above and below.
#   - The D16 change is implemented as jump motion inside background gaps. Along a
#     straight run, the particle can only land every second background cell. It
#     therefore advances by two cells at a time through passable background, and if
#     only one background cell remains before a wall or the boundary, that odd
#     position is not a legal resting place.
#   - Example on a real row:
#       original: c=0 -> 1 -> 2 -> 3 -> 4, then wall at 5 causes a turn.
#       variant : c=0 -> 2 -> 4, then wall at 5 causes the same turn.
#       difference: cells 1 and 3 are crossed conceptually but never reachable.
# - Complexity calibration:
#   - Multi-turn U-shapes from the original generator rely on vertical motion being
#     able to rest on odd channel rows. Under jump motion starting from the original
#     turn cell, vertical landings stay on even rows, so that re-entry geometry is
#     no longer reliable.
#   - To keep the task readable rather than fragile, this variant keeps straight
#     paths and one-turn L-paths only. That is the minimum structural change needed
#     to make D16 observable without introducing hidden parity traps.
# - Implicit assumptions audited in main():
#   - _has_corridor still assumes turn targets are background cells flanked by
#     equal non-background colors. This remains valid.
#   - The original _trace assumed every background cell is reachable; that no
#     longer holds. The new _trace only moves when both the intermediate cell and
#     the landing cell are passable.
#   - The original entrance scan accepted any early corridor cell. The generator now
#     fixes the bar start at col 2 so the first usable landing is parity-compatible.
#   - Turning is still attempted only when the immediately adjacent cell is colored.
#     Therefore generator walls that are meant to trigger turns must sit one cell
#     beyond a reachable landing site; the generator enforces odd wall columns.
# - Placement legality:
#   - All true turn walls use odd columns c_v, so the last horizontal landing cell
#     c_v-1 is even and reachable from col 0 by repeated two-cell jumps.
#   - Vertical gaps are carved on even bar rows exactly as before; with jump motion,
#     the particle lands on those even rows and can reach the top or bottom edge.
#   - Decorative cells on fake rows are forbidden on reserved vertical-corridor
#     columns, otherwise they would accidentally block a real jump path.
# - Visual observability:
#   - Widths are chosen even so many straight channels stop one cell short of the
#     right border under the new rule, while the original solver would continue
#     farther. L-turns also terminate one reachable cell earlier when parity blocks
#     the last odd cell.
#   - Generation explicitly rejects samples where the new output matches the
#     original solver output, so the variant is exercised in the training data.

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


def _is_passable(value: int) -> bool:
    return value in {PARTICLE, BG}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
    h, w = grid.shape
    if not _is_passable(int(grid[r, c])):
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


def _choose_turn(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int, str] | None:
    options = []
    h, w = grid.shape
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))
    assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
    return options[0] if options else None


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace the particle with jump motion through background gaps.

    The particle still keeps its direction until a colored cell directly blocks it,
    but straight motion only lands on every second passable cell. Operationally:
    - adjacent colored cell -> try a perpendicular turn from the current position
    - adjacent bg + landing bg two cells away -> jump to the landing cell
    - adjacent bg but blocked / out of bounds on the second cell -> stop
    """
    h, w = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        mr, mc = r + dr, c + dc
        if not (0 <= mr < h and 0 <= mc < w):
            break
        if not _is_passable(int(grid[mr, mc])):
            turn = _choose_turn(grid, r, c, direction)
            if turn is None:
                break
            r, c, direction = turn
            continue
        lr, lc = mr + dr, mc + dc
        if not (0 <= lr < h and 0 <= lc < w):
            break
        if not _is_passable(int(grid[lr, lc])):
            break
        r, c = lr, lc
    return r, c


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original 9bbf930d movement, used only to reject degenerate generated samples."""
    h, w = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if _is_passable(int(grid[nr, nc])):
            r, c = nr, nc
            continue
        turn = _choose_turn(grid, r, c, direction)
        if turn is None:
            break
        r, c, direction = turn
    return r, c


def _main_common(grid: np.ndarray, trace_fn) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if c % 2 == 1:
                continue
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
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


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along striped corridors with jump-style gap traversal."""
    return _main_common(grid, _trace)


def _main_original(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, _trace_original)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_OPTIONS = [14, 16, 18, 20]
_BAR_START = 2
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.25
_L_INTERNAL_PROB = 0.35

_VCOL_LEFT_PAD = 3
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400
_MIN_L_TURNS = 1
_MIN_VARIANT_DIFF_CELLS = 2


def _allocate_vcol(bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    """Reserve an odd wall column so the landing cell c_v-1 is even and reachable."""
    c_min = max(bar_start + _VCOL_LEFT_PAD, 5)
    c_max = w - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if c % 2 == 1 and all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _plan_real_channels(real_indices: list[int], bar_start: int, w: int) -> dict[int, dict[str, int | str]]:
    plans: dict[int, dict[str, int | str]] = {}
    used_vcols: set[int] = set()
    min_turns = _MIN_L_TURNS if len(real_indices) < 4 else 2
    turn_target = random.randint(min_turns, len(real_indices))
    turn_rows = set(random.sample(real_indices, turn_target))
    actual_turns = 0

    for i in real_indices:
        if i not in turn_rows or random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        actual_turns += 1
        plans[i] = {
            "kind": "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge",
            "direction": random.choice(["up", "down"]),
            "c_v": c_v,
        }

    if actual_turns < min_turns:
        return {}
    return plans


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


def _execute_plans(
    g: np.ndarray,
    h: int,
    plans: dict[int, dict[str, int | str]],
    bar_rows: list[int],
    bar_color: dict[int, int],
) -> set[int]:
    reserved_cols: set[int] = set()
    for i, plan in plans.items():
        if plan["kind"] == "straight":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = int(plan["c_v"])
        reserved_cols.add(c_v - 1)
        stop_at = None
        if plan["kind"] == "l_internal":
            if plan["direction"] == "up":
                cands = list(range(0, top, 2))
            else:
                cands = list(range(bot + 2, h, 2))
            stop_at = random.choice(cands) if cands else None
        _build_l_turn(g, h, top, bot, row, c_v, str(plan["direction"]), bar_color, stop_at)
    return reserved_cols


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.choice(_WIDTH_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
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

    for br in bar_rows:
        g[br, _BAR_START:w] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    plans = _plan_real_channels(real_indices, _BAR_START, w)
    if not plans:
        return None
    reserved_vertical_cols = _execute_plans(g, h, plans, bar_rows, bar_color)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        free_cols = [c for c in range(_BAR_START + 1, w) if c not in reserved_vertical_cols]
        if not free_cols:
            continue
        cols = random.sample(free_cols, min(n_decor, len(free_cols)))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        out_original = _main_original(g)
    except Exception:
        return None

    n_moved = sum(1 for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if int(np.count_nonzero(out != out_original)) < _MIN_VARIANT_DIFF_CELLS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
