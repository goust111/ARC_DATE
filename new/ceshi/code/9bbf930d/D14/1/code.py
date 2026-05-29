# Variant of 9bbf930d
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无图例或符号编码，所有规则由网格结构隐式表达。原题要求“遇阻时至多只有一个合法转弯”，
# 这里改为“右上角/右下角角落调色板编码走廊优先级”；当交叉口同时存在多个合法 90° 转弯时，
# 根据调色板中颜色出现的先后顺序选择对应颜色的走廊。

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


# ---------------------------------------------------------------------------
# Design analysis for the D14 variant
#
# a) Concrete appearance and behavior
# - The original task has no explicit symbol system: the particle simply takes the
#   only available perpendicular corridor when blocked.
# - This variant adds a visible legend in the top-right and bottom-right corners.
#   Each legend is a short horizontal color sequence, e.g. [red, green, blue].
#   Earlier colors mean higher turn priority.
# - The corridor layout stays striped: even rows are colored bar rows and odd rows
#   are channels. The particle still enters from column 0 and moves through BG.
# - The new visual event is a "fork row": the particle travels through a row whose
#   upper and lower bars intentionally have different colors near the junction.
#   At the blocker, both upward and downward turns are structurally legal. The
#   upper turn inherits the upper bar color; the lower turn inherits the lower bar
#   color. The legend decides which branch is taken.
# - Small trace example:
#   original logic: particle reaches a blocker on a fake-looking horizontal strip;
#   both up/down turns exist -> old solver would assert because ambiguity was
#   assumed impossible.
#   variant logic: the same geometry is allowed; if upper branch color is the
#   first color in the corner palette and lower branch color is second, the
#   particle turns upward. Reversing only the palette reverses the chosen branch.
#
# b) Complexity calibration
# - The new symbol mechanism is intentionally compact: exactly one short palette
#   sequence, re-used by multiple forks in the same grid.
# - The generator ensures at least two particles encounter priority-sensitive
#   forks, so the new rule is not a rare corner case.
# - The palette uses 3 colors. That is enough to support repeated evidence while
#   staying easy to infer from 2-4 ARC examples.
#
# c) Old main() assumptions that no longer hold
# - Old _trace() had `assert len(options) <= 1`. This relied on the generator
#   forbidding ambiguity. That assumption is invalid now.
# - Old logic implicitly treated structural uniqueness as the rule. In the
#   variant, geometry alone is insufficient; the solver must read an explicit
#   legend and use it as a lookup table.
# - Entrance detection in main() can stay unchanged because only turn selection,
#   not the notion of a "real entrance", is altered.
#
# d) Placement legality
# - The generator still uses non-adjacent active channel rows to reduce conflicts.
# - Fork / L-turn columns are allocated with spacing so their blockers do not
#   overlap.
# - A clean right-side legend strip is reserved so the palette is visible and does
#   not collide with turn construction.
#
# e) Visual observability
# - The palette colors are directly copied from corridor colors already present in
#   the maze; no invisible numeric ordering is used.
# - Multiple forks reuse palette colors, so a human can compare branch choices
#   across examples and infer that palette order matters.
#
# f) Visual cleanliness
# - The legend occupies a blank strip on the far right, duplicated on the top and
#   bottom edges. This keeps the symbolic cue isolated from the corridor mass.
# - The rest of the grid preserves the striped look of the source task.
#
# g) Color change inferability
# - No arbitrary color remapping is introduced. The only new color-based rule is:
#   "among simultaneously valid turns, prefer the branch whose corridor color
#   appears earlier in the corner palette."
#
# h) New solver assumptions and generator guarantees
# - Assumption: the top-right row ends with a contiguous run of legend colors and
#   a BG separator immediately to its left. Generator guarantees this.
# - Assumption: if multiple turn options share the same best legend rank, the
#   input is malformed. Generator avoids equal-color competing branches.
# - Assumption: at least two particles are priority-sensitive. Generator validates
#   this by tracing each active row under normal and reversed palette order.
# ---------------------------------------------------------------------------


def _read_priority(grid: np.ndarray) -> list[int]:
    """Read the contiguous non-BG suffix on the top row as the priority palette."""
    top = np.array(grid[0], dtype=int)
    colors: list[int] = []
    c = len(top) - 1
    while c >= 0 and int(top[c]) not in {BG, PARTICLE}:
        colors.append(int(top[c]))
        c -= 1
    colors.reverse()
    assert len(colors) >= 2, "missing corner priority palette"
    return colors


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Return the corridor wall color for a legal turn cell, else None."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in {PARTICLE, BG}:
        return a
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    return _corridor_color(grid, r, c, direction) is not None


def _choose_turn(
    grid: np.ndarray,
    options: list[tuple[int, int, str]],
    priority_rank: dict[int, int],
) -> tuple[int, int, str]:
    scored: list[tuple[int, int, int, str]] = []
    default_rank = len(priority_rank) + 10
    for tr, tc, turn_dir in options:
        color = _corridor_color(grid, tr, tc, turn_dir)
        assert color is not None, "invalid turn candidate"
        scored.append((priority_rank.get(int(color), default_rank), tr, tc, turn_dir))
    best_rank = min(item[0] for item in scored)
    best = [item for item in scored if item[0] == best_rank]
    assert len(best) == 1, "ambiguous turn remains after priority lookup"
    _, tr, tc, turn_dir = best[0]
    return tr, tc, turn_dir


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    priority_rank: dict[int, int],
) -> tuple[int, int]:
    """Trace one particle using corridor geometry plus palette-based tie-breaking."""
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
        if not options:
            break
        if len(options) == 1:
            r, c, direction = options[0]
        else:
            r, c, direction = _choose_turn(grid, options, priority_rank)
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through striped corridors; palette resolves multi-turn forks."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    priority = _read_priority(g)
    priority_rank = {int(color): i for i, color in enumerate(priority)}

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0, priority_rank)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (15, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_LEGEND_LEN = 3
_MIN_MOVED = 3
_MAX_MOVED = 5
_MAX_RETRIES = 500


def _pick_active_channels(n_channels: int, count: int) -> list[int]:
    order = list(range(n_channels))
    random.shuffle(order)
    chosen: list[int] = []
    for idx in order:
        if all(abs(idx - other) > 1 for other in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _allocate_turn_col(bar_start: int, path_right: int, used_cols: set[int]) -> int | None:
    candidates = []
    for c_v in range(bar_start + 5, path_right - 1):
        window = {c_v - 2, c_v - 1, c_v, c_v + 1}
        if window & used_cols:
            continue
        candidates.append(c_v)
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_cols.update({c_v - 2, c_v - 1, c_v, c_v + 1})
    return c_v


def _paint_base_bars(g: np.ndarray, bar_rows: list[int], bar_start: int, sep: int, extras: list[int]) -> None:
    palette = extras[:]
    random.shuffle(palette)
    for i, br in enumerate(bar_rows):
        g[br, bar_start:sep] = int(palette[i % len(palette)])


def _apply_real_entrance(g: np.ndarray, top: int, bot: int, bar_start: int, stop: int, color: int) -> None:
    g[top, bar_start:stop] = int(color)
    g[bot, bar_start:stop] = int(color)


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
) -> None:
    g[row, c_v] = int(color)
    if direction == "up":
        g[top, c_v - 1] = BG
        g[bot, c_v - 1] = int(color)
        for r_br in range(0, top + 1, 2):
            g[r_br, c_v - 1] = BG
    else:
        g[bot, c_v - 1] = BG
        g[top, c_v - 1] = int(color)
        for r_br in range(bot, H, 2):
            g[r_br, c_v - 1] = BG


def _build_priority_fork(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    bar_start: int,
    sep: int,
    entry_color: int,
    c_v: int,
    switch_c: int,
    up_color: int,
    down_color: int,
) -> None:
    _apply_real_entrance(g, top, bot, bar_start, switch_c, entry_color)
    g[top, switch_c:sep] = int(up_color)
    g[bot, switch_c:sep] = int(down_color)
    g[row, c_v] = int(entry_color)
    g[top, c_v - 1] = BG
    g[bot, c_v - 1] = BG
    for r_br in range(0, top + 1, 2):
        g[r_br, c_v - 1] = BG
    for r_br in range(bot, H, 2):
        g[r_br, c_v - 1] = BG


def _start_rows(grid: np.ndarray) -> list[int]:
    H = grid.shape[0]
    rows = []
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        for c in range(2, grid.shape[1]):
            if int(grid[r, c]) != BG:
                continue
            above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                bar_above = next((cc for cc in range(grid.shape[1]) if int(grid[r - 1, cc]) == above), grid.shape[1])
                bar_below = next((cc for cc in range(grid.shape[1]) if int(grid[r + 1, cc]) == below), grid.shape[1])
                if bar_above == bar_below and bar_above <= 3:
                    rows.append(r)
                break
    return rows


def _count_priority_sensitive_paths(grid: np.ndarray, rows: list[int]) -> int:
    priority = _read_priority(grid)
    forward = {int(color): i for i, color in enumerate(priority)}
    backward = {int(color): i for i, color in enumerate(reversed(priority))}
    sensitive = 0
    for r in rows:
        try:
            end_a = _trace(grid, r, 0, forward)
            end_b = _trace(grid, r, 0, backward)
        except Exception:
            return 0
        if end_a != end_b:
            sensitive += 1
    return sensitive


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)

    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], _N_EXTRA_COLORS)
    priority = random.sample(extras[:], _LEGEND_LEN)
    sep = W - _LEGEND_LEN - 1
    path_right = sep - 1
    if path_right - bar_start < 8:
        return None

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG
    g[:, sep:] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    max_non_adjacent = (n_channels + 1) // 2
    n_particles = random.randint(_MIN_MOVED, min(_MAX_MOVED, max_non_adjacent))
    active = _pick_active_channels(n_channels, n_particles)
    if len(active) < _MIN_MOVED:
        return None

    _paint_base_bars(g, bar_rows, bar_start, sep, extras)

    used_cols: set[int] = set()
    fork_target = min(2, len(active))
    fork_rows = set(random.sample(active, fork_target))

    for idx in active:
        top, bot = bar_rows[idx], bar_rows[idx + 1]
        row = top + 1
        entry_color = random.choice(extras)

        if idx in fork_rows:
            c_v = _allocate_turn_col(bar_start, path_right, used_cols)
            if c_v is None:
                return None
            switch_low = bar_start + 2
            switch_high = c_v - 2
            if switch_low > switch_high:
                return None
            switch_c = random.randint(switch_low, switch_high)
            up_color, down_color = random.sample(priority, 2)
            _build_priority_fork(
                g,
                H,
                top,
                bot,
                row,
                bar_start,
                sep,
                entry_color,
                c_v,
                switch_c,
                up_color,
                down_color,
            )
            continue

        kind = random.choice(["straight", "l_up", "l_down"])
        _apply_real_entrance(g, top, bot, bar_start, sep, entry_color)
        if kind == "straight":
            continue
        c_v = _allocate_turn_col(bar_start, path_right, used_cols)
        if c_v is None:
            continue
        direction = "up" if kind == "l_up" else "down"
        _build_l_turn(g, H, top, bot, row, entry_color, c_v, direction)

    # Force corner palette visibility with a BG separator to its left.
    g[0, sep] = BG
    g[H - 1, sep] = BG
    g[0, W - _LEGEND_LEN : W] = np.array(priority, dtype=int)
    g[H - 1, W - _LEGEND_LEN : W] = np.array(priority, dtype=int)

    try:
        out = main(g)
    except Exception:
        return None

    active_rows = [bar_rows[idx] + 1 for idx in active]
    moved_rows = [r for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE]
    if len(moved_rows) != len(active):
        return None
    if sorted(moved_rows) != sorted(active_rows):
        return None
    if _count_priority_sensitive_paths(g, active_rows) < fork_target:
        return None
    if len(set(g.flatten().tolist())) < 6:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
