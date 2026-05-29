# Combo variant of 9bbf930d (index 37)
# combo_spec: D12/2+D5/3
# Name: 按颜色分批次处理+多色粒子匹配走廊

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for combo D12/2 + D5/3
======================================

a) Concrete behavior and trace
   - The base striped geometry is preserved: even rows are colored bar rows, odd rows are
     horizontal travel rows, and turns are created by carving single-column gaps in the bars.
   - The fusion uses two visible particle/corridor colors only:
     `PHASE_A = 6` and `PHASE_B = 8`.
     These are not hidden ids; they are the actual corridor colors seen on the grid.
   - Each particle may move only inside corridors whose flanking bars have the same color as the
     particle itself. This applies both at the initial entrance and at every later 90-degree turn.
   - Phase order is fixed and visible:
     1. move every color-6 source using only color-6 corridors,
     2. commit their endpoints simultaneously,
     3. then move every color-8 source using only color-8 corridors on the modified grid.
   - Small concrete trace:
     a color-6 source starts on a color-6 row, turns downward into a vertical color-6 shaft, and
     stops on a color-8 horizontal crossing cell. In the original task that crossing is just a bg
     cell, so a later traveler would pass through it. In this combo, when phase B starts, that
     crossing cell is already occupied by a settled color-6 particle, so the color-8 particle on
     that row stops earlier than it would on the untouched input. A different color-8 source placed
     on a color-6 row never enters at all, because entrance color-matching fails before phase order
     even matters.

b) Complexity calibration
   - Only two operational colors are used for real corridors. That is enough to create repeated
     support for both "same-color access" and "process color 6 before color 8" without exploding
     the number of cases humans must infer.
   - Every sample contains four visible sources with repeated groups:
     two color-6 sources, two color-8 sources. Among them:
     1. one color-6 source causes the cascade,
     2. one color-8 source is truncated by the settled color-6 endpoint,
     3. one mismatched color-8-on-color-6 row stays put.
   - This avoids a degenerate dataset where phase order exists but color matching is irrelevant,
     or vice versa.

c) Implicit-assumption audit against the original solver
   - Original `_has_corridor()` assumed the moving particle color was irrelevant. That breaks:
     the combo must require the corridor wall color to equal the current particle color.
   - Original `_trace()` treated `{BG, PARTICLE}` as passable. That is no longer safe once
     corridor bars themselves may have the same color as the moving particle. If same-colored wall
     pixels were passable, the particle would tunnel through blockers instead of turning. Therefore
     the combo trace treats only `BG` as traversable; any colored pixel, including a same-colored
     settled particle or bar, is an obstacle.
   - Original `main()` performed one global sweep and wrote endpoints directly. That assumed
     particles do not affect later particles. The combo needs phase-local snapshot tracing and
     batched commits because color-6 endpoints must become obstacles before color-8 moves.
   - Original start detection assumed a single dedicated particle color. The combo must instead
     detect starts by phase color and validate that the first real corridor on the row matches it.

d) Placement legality
   - The generator chooses four real channels whose indices are pairwise non-adjacent. That lets
     each real channel own its adjacent bar-row pair without conflicting with another real channel.
   - The forced cascade uses one reserved vertical column with the original 3-column exclusion
     window, so the vertical shaft does not overlap later turn structures.
   - The source color-6 shaft stops on the color-8 row by leaving the lower bar closed at the
     target channel; this creates a legal dead-end with no out-of-bounds movement.
   - Fake channels are filled only after the real-channel bar pairs are fixed, and each fake pair is
     explicitly forced to have different colors, so they cannot accidentally become new real
     corridors.

e) Visual observability
   - From a few input/output pairs a human can see:
     1. only same-colored left-edge sources enter same-colored corridors,
     2. color-6 outputs appear first and remain on the board,
     3. a color-8 path is shortened by an already-settled color-6 endpoint.
   - The order rule does not depend on invisible numeric sorting of arbitrary colors; the same two
     visible colors are reused across all samples.

f) Visual cleanliness
   - The fusion keeps the original stripe vocabulary and introduces only one cross-phase blockage per
     grid. The rest of the scene stays mostly straight, which keeps the cause of the truncated
     color-8 path readable.
   - Sources remain isolated in column 0 with column 1 preserved as pure background, so start colors
     are visually separate from the corridor field.

g) Change inferability
   - There is no hidden recoloring or lookup table. A source's color is preserved from input to
     output, and the same visible color simultaneously encodes:
     1. which corridors that source may use,
     2. which batch the source belongs to.
   - The generator enforces both key signals quantitatively:
     at least one mismatched real source and at least one color-8 endpoint changed by the color-6
     phase. If either metric disappears, the sample is rejected as degenerate.

Post-change solver assumptions audited against the generator
------------------------------------------------------------
   - Solver assumption: the only movable source colors are 6 and 8.
     Generator guarantee: every source in column 0 is one of those two colors.
   - Solver assumption: a legal path cell is background, never a colored wall pixel.
     Generator guarantee: all intended travel cells remain `BG`; only walls and settled endpoints
     are colored.
   - Solver assumption: phase destinations are unique within a phase.
     Generator guarantee: the forced cascade endpoint, the straight endpoints, and the non-mover are
     all distinct by construction.
   - Solver assumption: the first usable entrance still begins at columns 2-3 as in the base task.
     Generator guarantee: full-width bars start at column 2 or 3 and every real corridor keeps that
     left entrance unobstructed.
"""

BG = 7
PHASE_A = 6
PHASE_B = 8
PHASE_COLORS = (PHASE_A, PHASE_B)
_PHASE_SET = set(PHASE_COLORS)

_DELTAS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(
    grid: np.ndarray, r: int, c: int, direction: str, corridor_color: int
) -> bool:
    """True iff (r,c) is a bg path cell flanked by bars of `corridor_color`."""
    H, W = grid.shape
    if int(grid[r, c]) != BG:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b == int(corridor_color)


def _trace(grid: np.ndarray, r: int, c: int, particle_color: int) -> tuple[int, int]:
    """Trace one particle, treating every colored pixel as a blocker."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) == BG:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W:
                if _has_corridor(grid, tr, tc, turn_dir, particle_color):
                    options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return int(r), int(c)


def _find_entrance_col(grid: np.ndarray, r: int, particle_color: int) -> int | None:
    """Return the first valid horizontal entrance cell whose walls match `particle_color`."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below == int(particle_color):
            return int(c)
    return None


def _is_valid_start(grid: np.ndarray, r: int, particle_color: int) -> bool:
    """True iff row r has a phase-colored source and a matching original-style entrance."""
    if int(grid[r, 0]) != int(particle_color):
        return False
    entrance_col = _find_entrance_col(grid, r, particle_color)
    if entrance_col is None:
        return False
    _H, W = grid.shape
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == int(particle_color)), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == int(particle_color)), W)
    return bar_above == bar_below and bar_above <= 3


def _collect_moves(grid: np.ndarray, particle_color: int) -> dict[int, tuple[int, int]]:
    """Trace all starts of one color phase from a fixed snapshot."""
    H, _W = grid.shape
    moves: dict[int, tuple[int, int]] = {}
    for r in range(1, H - 1, 2):
        if not _is_valid_start(grid, r, particle_color):
            continue
        end_r, end_c = _trace(grid, r, 0, particle_color)
        if (end_r, end_c) != (r, 0):
            moves[int(r)] = (int(end_r), int(end_c))
    return moves


def _apply_moves(
    grid: np.ndarray, particle_color: int, moves: dict[int, tuple[int, int]]
) -> np.ndarray:
    """Commit one phase after all destinations have been computed."""
    out = np.array(grid, dtype=int, copy=True)
    for r in moves:
        out[int(r), 0] = BG
    seen: set[tuple[int, int]] = set()
    for end_r, end_c in moves.values():
        assert (end_r, end_c) not in seen, "phase destinations must be unique"
        seen.add((int(end_r), int(end_c)))
        out[int(end_r), int(end_c)] = int(particle_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move color-matched particles in two visible phases.

    Input structure:
    - bg=7. Sources appear only in col 0 and use two visible colors: 6 then 8.
    - Even rows are bar rows; odd rows are travel rows. Real corridors are same-color bar pairs.

    Rule:
    - A particle may enter and turn only inside corridors whose flanking bars have the same color
      as that particle.
    - Process all color-6 particles first and keep their final positions on the grid.
    - Then process all color-8 particles on the modified grid. Any settled color-6 endpoint is now
      a blocking obstacle.
    - A particle stops at the boundary or when the next colored blocker offers no legal matching
      perpendicular turn.

    Key insight:
    - The same visible color carries two meanings at once: it selects which corridors are legal for
      that particle and it selects which batch moves first, so earlier-color endpoints can shorten
      later-color paths.
    """
    out = np.array(grid, dtype=int)
    for particle_color in PHASE_COLORS:
        moves = _collect_moves(out, particle_color)
        out = _apply_moves(out, particle_color, moves)
    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_EXTRA_COLORS = (0, 1, 2, 3, 4, 5, 9)

_REAL_COUNT = 4
_MAX_RETRIES = 500
_MIN_DISTINCT_COLORS = 6
_MIN_B_PHASE_SHIFT = 3
_FAKE_DECOR_RANGE = (0, 2)

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for one vertical shaft."""
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
    return int(c_v)


def _pick_real_indices(n_channels: int) -> list[int] | None:
    """Pick four pairwise non-adjacent real channels."""
    candidates = list(range(n_channels))
    for _ in range(80):
        chosen = sorted(random.sample(candidates, _REAL_COUNT))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _build_bar_colors(
    n_channels: int, real_color_by_channel: dict[int, int], extras: list[int]
) -> list[int] | None:
    """Assign bar-row colors so chosen channels are real and all others are fake."""
    bar_colors: list[int | None] = [None] * (n_channels + 1)
    for idx, color in real_color_by_channel.items():
        for pos in (idx, idx + 1):
            current = bar_colors[pos]
            if current is not None and current != int(color):
                return None
            bar_colors[pos] = int(color)

    for idx in range(n_channels):
        if idx in real_color_by_channel:
            continue
        top = bar_colors[idx]
        bot = bar_colors[idx + 1]
        if top is None and bot is None:
            top = random.choice(extras)
            bot = random.choice([c for c in extras if c != top])
            bar_colors[idx] = int(top)
            bar_colors[idx + 1] = int(bot)
        elif top is None:
            choices = [c for c in extras if c != int(bot)]
            if not choices:
                return None
            bar_colors[idx] = int(random.choice(choices))
        elif bot is None:
            choices = [c for c in extras if c != int(top)]
            if not choices:
                return None
            bar_colors[idx + 1] = int(random.choice(choices))
        elif int(top) == int(bot):
            return None

    out = [int(c) for c in bar_colors]
    fake_palette = {
        out[idx]
        for idx in range(n_channels)
        if idx not in real_color_by_channel
    } | {
        out[idx + 1]
        for idx in range(n_channels)
        if idx not in real_color_by_channel
    }
    if len(fake_palette) < 3:
        return None
    return out


def _paint_bars(g: np.ndarray, bar_rows: list[int], bar_colors: list[int], bar_start: int) -> None:
    """Paint full-width stripe bars using the selected row colors."""
    for i, br in enumerate(bar_rows):
        g[br, bar_start:] = int(bar_colors[i])


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color_by_row: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build one sealed L-turn, optionally stopping before the outer edge."""
    color = int(bar_color_by_row[top])
    g[row, c_v] = color

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = int(bar_color_by_row[turn_bar])
    g[turn_bar, c_v] = int(bar_color_by_row[turn_bar])
    g[opp_bar, c_v - 1] = int(bar_color_by_row[opp_bar])

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


def _row_corridor_color(grid: np.ndarray, r: int) -> int | None:
    """Return the real corridor color available on row r, ignoring the source color."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above != below or above == BG:
            continue
        bar_above = next((k for k in range(W) if int(grid[r - 1, k]) == above), W)
        bar_below = next((k for k in range(W) if int(grid[r + 1, k]) == below), W)
        if bar_above == bar_below and bar_above <= 3:
            return int(above)
    return None


def _paint_sources(
    g: np.ndarray, bar_rows: list[int], source_color_by_channel: dict[int, int]
) -> None:
    """Place left-edge sources while keeping column 1 as background."""
    g[:, 0] = BG
    for idx, color in source_color_by_channel.items():
        g[bar_rows[idx] + 1, 0] = int(color)
    g[:, 1] = BG


def _count_moved_sources(inp: np.ndarray, out: np.ndarray) -> int:
    """Count how many visible sources actually left column 0."""
    H = inp.shape[0]
    return sum(
        1
        for r in range(1, H - 1, 2)
        if int(inp[r, 0]) in _PHASE_SET and int(out[r, 0]) == BG
    )


def _measure_combo_behavior(grid: np.ndarray) -> dict[str, int]:
    """Quantify the two fused signals: mismatch and staged blockage."""
    g = np.array(grid, dtype=int)

    mismatched_real_sources = 0
    for r in range(1, g.shape[0] - 1, 2):
        source_color = int(g[r, 0])
        if source_color not in _PHASE_SET:
            continue
        corridor_color = _row_corridor_color(g, r)
        if corridor_color is not None and corridor_color != source_color:
            mismatched_real_sources += 1

    independent_b = _collect_moves(g, PHASE_B)
    after_a = _apply_moves(g, PHASE_A, _collect_moves(g, PHASE_A))
    staged_b = _collect_moves(after_a, PHASE_B)

    changed_b = 0
    max_b_shift = 0
    for r, independent_end in independent_b.items():
        staged_end = staged_b.get(r, (r, 0))
        if staged_end != independent_end:
            changed_b += 1
            max_b_shift = max(max_b_shift, int(independent_end[1]) - int(staged_end[1]))

    return {
        "mismatched_real_sources": int(mismatched_real_sources),
        "independent_b": int(len(independent_b)),
        "changed_b": int(changed_b),
        "max_b_shift": int(max_b_shift),
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one combo instance with one forced A->B cascade and one mismatched source."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)

    g = np.full((H, W), BG, dtype=int)
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    real_indices = _pick_real_indices(n_channels)
    if real_indices is None:
        return None
    extra_palette = random.sample(list(_EXTRA_COLORS), 4)

    extra_idx = real_indices[0]
    source_idx = real_indices[1]
    target_idx = real_indices[2]
    mismatch_idx = real_indices[3]

    real_color_by_channel = {
        extra_idx: PHASE_A,
        source_idx: PHASE_A,
        target_idx: PHASE_B,
        mismatch_idx: PHASE_A,
    }
    bar_colors = _build_bar_colors(n_channels, real_color_by_channel, extra_palette)
    if bar_colors is None:
        return None

    _paint_bars(g, bar_rows, bar_colors, bar_start)
    bar_color_by_row = {bar_rows[i]: int(bar_colors[i]) for i in range(len(bar_rows))}

    used_vcols: set[int] = set()
    c_v = _allocate_vcol(bar_start, W, used_vcols)
    if c_v is None:
        return None

    source_top, source_bot = bar_rows[source_idx], bar_rows[source_idx + 1]
    source_row = source_top + 1
    target_top, target_bot = bar_rows[target_idx], bar_rows[target_idx + 1]

    if source_idx < target_idx:
        direction = "down"
        stop_at = target_bot
    else:
        direction = "up"
        stop_at = target_top

    _build_l_turn(
        g,
        H,
        source_top,
        source_bot,
        source_row,
        c_v,
        direction,
        bar_color_by_row,
        stop_at,
    )

    for idx in range(n_channels):
        if idx in real_color_by_channel:
            continue
        row = bar_rows[idx] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c in cols:
            if g[row, c] == BG:
                g[row, c] = random.choice([bar_colors[idx], bar_colors[idx + 1]])

    source_color_by_channel = {
        extra_idx: PHASE_A,
        source_idx: PHASE_A,
        target_idx: PHASE_B,
        mismatch_idx: PHASE_B,
    }
    _paint_sources(g, bar_rows, source_color_by_channel)

    cascade_cell = (target_top + 1, c_v - 1)
    if int(g[cascade_cell]) != BG:
        return None

    try:
        out = main(g)
    except Exception:
        return None

    if _count_moved_sources(g, out) != 3:
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None

    metrics = _measure_combo_behavior(g)
    if metrics["mismatched_real_sources"] < 1:
        return None
    if metrics["independent_b"] < 1 or metrics["changed_b"] < 1:
        return None
    if metrics["max_b_shift"] < _MIN_B_PHASE_SHIFT:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
