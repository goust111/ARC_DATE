# Variant of 9bbf930d
# Dimension: D1 — 路径类型：走廊跟随 (path)
# Change: 粒子在同色条对形成的1格宽走廊中移动，遇障碍时转入垂直方向的同色条对走廊
#         （corridor_following + inertia）→ 直线射线路径（straight ray）

from __future__ import annotations

import random
import numpy as np


BG = 7
PARTICLE = 6


"""
Design analysis for the D1 variant
==================================

1. Concrete behavior and traced example
   - The striped layout is preserved: even rows are colored bar rows, odd rows are
     horizontal lanes, col 0 is filled with magenta particles, col 1 is background.
   - A "real" lane is still recognized exactly as before: some column >= 2 on that odd
     row is a background cell whose above/below cells have the same non-bg color, and
     those bars begin together at col 2 or 3.
   - The core change happens after entrance recognition.
     Original:
       particle walks right; if the next cell is colored, it probes perpendicular
       corridor cells and may turn into a vertical lane.
     Variant:
       particle walks right as a straight ray only; the first non-bg/non-particle
       cell is a hard stop. The particle lands on the cell immediately before it.
   - Small concrete example:
       row r is a real lane, blocker at (r, 10), and there is an old-style upward
       turn opening at (r-1, 9). Original solver would stop at c=9, turn upward,
       and continue vertically. Variant solver stops at c=9 and never inspects the
       turn opening. That is the visible behavioral difference.

2. Complexity calibration
   - I keep the original striped bars and real/fake lane distinction so the puzzle is
     not reduced to "move each particle to the first colored cell on its row" with no
     structure.
   - I also keep decoy turn structures. They are visually strong because they look
     exactly like old turn opportunities, but they are irrelevant under the new rule.
   - Freedom is controlled: real lane count stays 3-6, bar start stays 2-3, colors
     stay at 6 total, and lane plans use only three row-level outcomes:
       edge ray, blocker stop, turn-bait stop.

3. Original-code assumptions that no longer hold
   - The old `_trace()` assumes a colored blocker may trigger a perpendicular search.
     That assumption is removed entirely.
   - The old solver's correctness relied on `_has_corridor()` and on there being at
     most one valid perpendicular turn. In this variant, those turn-validity checks
     are irrelevant and intentionally absent.
   - The old generator spent effort building traversable turns/U-turns because solver
     behavior depended on them. The new generator only needs them as visible decoys,
     not as actual path segments.

4. Placement legality
   - Bars still start at col 2 or 3 and stay inside the grid.
   - Turn-bait columns are allocated with a reserved 3-column window so carved gaps
     and obstacle cells do not collide.
   - Row blockers are placed at columns >= bar_start + 3, so every moved particle
     travels a visible distance and never stops immediately at the entrance.
   - Vertical decoy gaps are carved only on even rows and stay within bounds.

5. Visual observability
   - Humans can compare examples and see that particles always stay on their original
     row, even when a same-color turn opening is available right before a blocker.
   - The rule does not depend on color ids, ordering, or hidden metadata; only visible
     geometry matters.

6. Visual cleanliness
   - The scene is still row-striped and sparse.
   - Decoy vertical gaps are limited to a few reserved columns, so the picture remains
     readable instead of becoming a dense maze.

7. Color-change inference
   - There is no color remapping in this variant. Colors only define visible bar pairs
     and blockers, which is easy to infer from a few examples.

8. New solver assumptions and how generator guarantees them
   - Assumption: only real odd rows should move.
     Guarantee: real lanes use same-color bar pairs; fake lanes use mismatched pairs.
   - Assumption: a moved row has a valid aligned entrance whose bars start at col 2/3.
     Guarantee: full-width bars begin at a shared `bar_start`, and decorations never
     alter col 0/1 or the entrance columns.
   - Assumption: the ray endpoint is well-defined by the first colored cell or edge.
     Guarantee: blockers are single colored cells on the same row, and the grid border
     naturally terminates edge rays.
"""


def _find_entrance(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid entrance column for a real lane, else None."""
    H, W = grid.shape
    if not (0 < r < H - 1):
        return None
    if int(grid[r, 0]) != PARTICLE:
        return None

    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above = int(grid[r - 1, c])
        below = int(grid[r + 1, c])
        if above == below and above not in {BG, PARTICLE}:
            entrance_col = c
            break
    if entrance_col is None:
        return None

    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == color), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == color), W)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _ray_endpoint(grid: np.ndarray, r: int, c0: int = 0) -> tuple[int, int]:
    """Move horizontally until the first colored pixel or the right boundary."""
    _, W = grid.shape
    c = c0
    while True:
        nc = c + 1
        if nc >= W:
            break
        if int(grid[r, nc]) in {BG, PARTICLE}:
            c = nc
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move each real-lane particle as a straight horizontal ray."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1, 2):
        if _find_entrance(g, r) is None:
            continue
        end_r, end_c = _ray_endpoint(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_BLOCKER_MIN_OFFSET = 3
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 300


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a turn-bait column window and return the blocker column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _build_turn_bait(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
) -> None:
    """Build an old-style L-turn opening that the new straight-ray solver ignores."""
    H, _ = g.shape
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
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG


def _place_row_blocker(g: np.ndarray, row: int, min_col: int, colors: list[int]) -> None:
    """Place a same-row colored blocker so the ray stops before it."""
    _, W = g.shape
    c = random.randint(min_col, W - 2)
    g[row, c] = random.choice(colors)


def _choose_bait_direction(i: int, n_channels: int) -> str:
    """Prefer a direction with at least some visible room."""
    if i == 0:
        return "down"
    if i == n_channels - 1:
        return "up"
    return random.choice(["up", "down"])


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
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

    real_indices = sorted(i for i, tag in enumerate(channel_spec) if tag == "real")
    if not real_indices:
        return None

    n_bait = random.randint(1, min(3, len(real_indices)))
    bait_indices = set(random.sample(real_indices, n_bait))
    remaining = [i for i in real_indices if i not in bait_indices]
    n_edge = random.randint(0, len(remaining))
    edge_indices = set(random.sample(remaining, n_edge))

    used_vcols: set[int] = set()

    for i in real_indices:
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        blocker_palette = [c for c in extras if c != BG]

        if i in bait_indices:
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                _place_row_blocker(g, row, bar_start + _BLOCKER_MIN_OFFSET, blocker_palette)
            else:
                direction = _choose_bait_direction(i, n_channels)
                _build_turn_bait(g, top, bot, row, color, c_v, direction, bar_color)
        elif i in edge_indices:
            pass
        else:
            _place_row_blocker(g, row, bar_start + _BLOCKER_MIN_OFFSET, blocker_palette)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        row = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        top_color = bar_color[bar_rows[i]]
        bot_color = bar_color[bar_rows[i + 1]]
        for c in cols:
            if g[row, c] == BG:
                g[row, c] = random.choice([top_color, bot_color])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    out = main(g)
    n_moved = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    bait_rows_realized = 0
    for i in bait_indices:
        row = bar_rows[i] + 1
        moved_cols = np.where(out[row] == PARTICLE)[0]
        if moved_cols.size != 1:
            continue
        c_end = int(moved_cols[0])
        if c_end + 1 < W and int(g[row, c_end + 1]) not in {BG, PARTICLE}:
            bait_rows_realized += 1
    if bait_rows_realized == 0:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
