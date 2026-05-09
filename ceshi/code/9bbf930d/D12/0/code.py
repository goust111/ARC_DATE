# Variant of 9bbf930d
# Dimension: D12 — 粒子数量与路径多样性 (meta)
# Change: 3-6个粒子同时存在，每个粒子有0-3次转弯，路径长度14-46格。... → 单粒子深度路径
#
# Design analysis
# spec:
#   core_visual:
#     input still uses the original striped language: even rows are colored bar rows,
#     odd rows are channels/background, bg=7, particle=6.
#     The variant collapses the original "many short independent corridors" into one
#     dominant same-color corridor that snakes through most of the board.
#     The particle still starts on the left edge and still obeys the original local
#     motion rule: go straight through bg, and only turn when a colored blocker is
#     directly ahead and exactly one perpendicular same-color corridor is available.
#   concrete_shape:
#     a contiguous stack of odd rows is chosen as the active path band.
#     All adjacent even rows in that band share one corridor color.
#     On successive active odd rows, the open run alternates:
#       rightward segment -> down turn near the right side
#       leftward segment  -> down turn near the left side
#     This creates a long snake with 8-20 turns instead of 0-3 turns.
#     Other rows may contain decorative colored bars, but only one particle exists.
#   behavior_trace_example:
#     Original:
#       row 1 particle goes right, maybe turns once/down, stops quickly; other particles
#       on other odd rows do the same independently.
#     Variant:
#       only one particle exists. It enters the top active corridor, runs almost the
#       full width, hits a blocker, turns down, immediately re-enters the next corridor
#       in the opposite horizontal direction, and repeats across many rows.
#       The output moves that single particle to the snake tail.
#     Difference:
#       inference shifts from "which of several particles moves where" to "follow one
#       long trajectory without losing orientation across many repeated turns".
#   complexity_calibration:
#     The snake uses a fixed pair of inner turn columns (bar_start on the left,
#     W-2 on the right). Repetition makes the rule legible, while a long contiguous
#     band of 5-11 horizontal segments guarantees that the D12 change is visually obvious.
#     We avoid adding many free parameters: one path color, one active band, one particle.
#   implicit_assumption_audit_for_main:
#     - Original main() assumed many particles in col 0 and searched each row for a
#       matching entrance. That is harmless for the original generator but too loose here:
#       the variant must treat "exactly one particle" as the normal case.
#     - Original entrance logic relied on bars starting at col 2-3 and on every moving
#       particle having its own independent horizontal corridor. The variant still keeps
#       bar_start in {2,3}, but path rows now revisit the same global corridor color many
#       times, so main() should not depend on per-row independence.
#     - _trace() only needed a modest loop cap before. The variant deliberately creates
#       much longer paths, so the cap is scaled with grid size.
#   placement_legality:
#     The active path band is contiguous in row space, so every vertical turn lands in
#     the immediately adjacent channel row and never jumps. Left and right turn columns
#     are fixed away from the borders, which prevents overlap between opposite-side turns.
#     Decorative bars are only placed outside the active band, so they cannot create
#     ambiguous branches for the traced particle.
#   visual_observability:
#     Humans can see a single magenta source on the left and one dominant snake corridor
#     occupying most of the grid. Across examples, only the active band height, bar_start,
#     and board size vary; the repeated serpentine motif makes the D12 change easy to infer.
#   cleanliness:
#     The main path is one strong stripe system with sparse off-band decorations, so the
#     grid stays readable. No color-id-dependent ordering or invisible metadata is used.
#   new_main_assumptions_checked_against_generator:
#     - exactly one particle is generated
#     - the particle is on col 0
#     - a valid horizontal corridor appears within the first few columns
#     - the snake produces an unambiguous trace with many turns

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars."""
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


def _trace_with_stats(grid: np.ndarray, r: int, c: int) -> tuple[int, int, int, int]:
    """Trace one particle and return (end_r, end_c, path_len, turns)."""
    H, W = grid.shape
    direction = "right"
    steps = 0
    turns = 0
    for _ in range(H * W * 6):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            steps += 1
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
        steps += 1
        turns += 1
    return r, c, steps, turns


def _has_start_access(grid: np.ndarray, r: int) -> bool:
    """The variant still begins with a rightward horizontal corridor near col 2-4."""
    H, W = grid.shape
    if not (0 <= r < H) or W < 5:
        return False
    if int(grid[r, 0]) != PARTICLE or int(grid[r, 1]) not in {PARTICLE, BG}:
        return False
    for c in range(2, min(W, 5)):
        if _has_corridor(grid, r, c, "right"):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Move the single particle along the unique long corridor."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    particles = [tuple(pos) for pos in np.argwhere(g == PARTICLE)]
    if not particles:
        return out

    # Normal variant case: exactly one particle.
    if len(particles) == 1:
        r, c = particles[0]
        if c != 0 or not _has_start_access(g, r):
            return out
        end_r, end_c, _, _ = _trace_with_stats(g, r, c)
        if (end_r, end_c) != (r, c):
            out[r, c] = BG
            out[end_r, end_c] = PARTICLE
        return out

    # Fallback: preserve original multi-particle behavior if such an input appears.
    for r, c in particles:
        if c != 0 or not _has_start_access(g, r):
            continue
        end_r, end_c, _, _ = _trace_with_stats(g, r, c)
        if (end_r, end_c) != (r, c):
            out[r, c] = BG
            out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_MAX_RETRIES = 300
_MIN_SEGMENTS = 5


def _paint_decorations(
    g: np.ndarray,
    active_rows: set[int],
    active_even_rows: set[int],
    bar_start: int,
    colors: list[int],
) -> None:
    """Add off-band distractors without touching the active snake."""
    H, W = g.shape
    inactive_even = [r for r in range(0, H, 2) if r not in active_even_rows]
    inactive_odd = [r for r in range(1, H, 2) if r not in active_rows]

    for r in inactive_even:
        color = random.choice(colors)
        c0 = random.randint(bar_start, max(bar_start, W - 5))
        c1 = random.randint(min(W - 1, c0 + 3), W - 1)
        g[r, c0 : c1 + 1] = color

    for color in colors:
        placed = False
        for _ in range(20):
            r = random.randrange(H)
            c = random.randrange(bar_start, W)
            if r in active_rows or r in active_even_rows:
                continue
            if g[r, c] != BG:
                continue
            g[r, c] = color
            placed = True
            break
        if not placed and inactive_even:
            r = inactive_even[0]
            g[r, min(W - 1, bar_start + colors.index(color))] = color

    for r in inactive_odd:
        for _ in range(random.randint(0, 2)):
            c = random.randint(bar_start, W - 1)
            if g[r, c] == BG:
                g[r, c] = random.choice(colors)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    particles = [tuple(pos) for pos in np.argwhere(grid == PARTICLE)]
    if len(particles) != 1:
        return {"particle_count": len(particles), "path_length": 0, "turns": 0}
    r, c = particles[0]
    end_r, end_c, path_length, turns = _trace_with_stats(grid, r, c)
    moved = int((end_r, end_c) != (r, c))
    return {
        "particle_count": 1,
        "path_length": path_length,
        "turns": turns,
        "moved": moved,
        "end_r": end_r,
        "end_c": end_c,
    }


def _build_snake_grid() -> np.ndarray:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], 4)
    path_color = extras[0]
    deco_colors = extras[1:]

    g = np.full((H, W), BG, dtype=int)
    n_channels = (H - 1) // 2
    max_odd_segments = n_channels if n_channels % 2 == 1 else n_channels - 1
    n_segments = max_odd_segments
    if n_segments - 2 >= _MIN_SEGMENTS and random.random() < 0.35:
        n_segments -= 2
    top_channel = random.randint(0, n_channels - n_segments)

    active_rows = {2 * (top_channel + k) + 1 for k in range(n_segments)}
    active_even_rows = {2 * top_channel + 2 * k for k in range(n_segments + 1)}

    for r in active_even_rows:
        g[r, bar_start:W] = path_color

    right_wall = W - 2
    right_turn_col = W - 3
    left_wall = bar_start
    left_turn_col = bar_start + 1
    rows_sorted = sorted(active_rows)

    for idx, r in enumerate(rows_sorted):
        moving_right = idx % 2 == 0
        is_final = idx == len(rows_sorted) - 1
        if moving_right:
            if idx > 0:
                g[r, left_wall] = path_color
            if not is_final:
                g[r, right_wall] = path_color
                g[r + 1, right_turn_col] = BG
        else:
            g[r, right_wall] = path_color
            if not is_final:
                g[r, left_wall] = path_color
                g[r + 1, left_turn_col] = BG

    start_row = rows_sorted[0]
    g[start_row, 0] = PARTICLE
    g[start_row, 1] = BG

    _paint_decorations(g, active_rows, active_even_rows, bar_start, deco_colors)
    return g


def _gen_attempt() -> np.ndarray | None:
    g = _build_snake_grid()
    metrics = _variant_metrics(g)
    if metrics.get("particle_count") != 1 or metrics.get("moved") != 1:
        return None
    if metrics["turns"] < 8:
        return None
    if metrics["path_length"] < int(0.33 * g.size):
        return None

    out = main(g)
    if np.count_nonzero(out == PARTICLE) != 1:
        return None
    if int(out[metrics["end_r"], metrics["end_c"]]) != PARTICLE:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
