# Combo variant of 9bbf930d (index 12)
# combo_spec: D13/1+D20/1
# Name: 链式反应+Z形走廊

from __future__ import annotations

from collections import deque
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
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}

# ---------------------------------------------------------------------------
# Design analysis for the combo variant D13/1 + D20/1
#
# a) Concrete behavior and trace
# - Base task: each valid seed in col 0 launches one particle to the right and
#   only its final endpoint is written once.
# - Combo: when a particle stops, that endpoint stays magenta and immediately
#   launches a new particle in the opposite terminal direction. The corridor set
#   also includes genuine Z offsets: `right -> down -> right` and the mirrored
#   `left -> down -> left`, plus the reverse traversals those same gadgets imply.
# - One chain band in this generator looks like:
#   seed row 1 goes right -> Z-shift downward onto row 3 -> L-turn upward to E1;
#   E1 launches downward -> row 3 continues left -> mirrored Z-shift downward
#   onto row 5 -> mirrored L-turn upward to E2; E2 launches downward -> row 5
#   continues right to the far edge. Original logic would only mark E1; the
#   combo marks E1, E2, and the final right-edge endpoint, while the Z offsets
#   are the geometry that make the chain continue on parallel lanes.
#
# b) Complexity calibration
# - A fully random planner would make the cascade fragile: reverse launches can
#   bounce back immediately or produce ambiguous side turns.
# - Instead the generator uses two repeated, color-separated chain bands with a
#   fixed 3-step reaction template. The repeated structure makes the cascade and
#   the Z motif visible from a few examples without introducing too many cases.
#
# c) Implicit assumption audit against the original main()
# - Clearing `out[r, 0]` assumed movement is a one-shot transfer. Invalid now:
#   particles accumulate, so seeds remain.
# - Calling `_trace()` once per row assumed independent particles. Invalid now:
#   endpoints spawn later launches, so processing must be queue-based.
# - Returning only `(end_r, end_c)` assumed the terminal direction was useless.
#   Invalid now: the chain needs the terminal direction to emit the opposite one.
# - The local turn rule itself still works. `_has_corridor()` only checks the
#   flanking bars of the candidate turn cell, so Z corridors are legal without
#   changing the solver, provided the generator seals the unused branch.
#
# d) Placement legality
# - Each chain band occupies four even bar rows, i.e. three channel rows, and
#   the two bands start at bar rows 0 and 8. This fits every allowed odd height.
# - Turn wall columns are fixed relative to `bar_start`, so the mirrored gadgets
#   do not overlap. Lower channels have their left entrance signature broken so
#   they are destinations/reaction lanes rather than independent seeds.
#
# e) Visual observability
# - The output visibly differs from the source task because several magenta
#   particles accumulate along one same-color band instead of a single transfer.
# - The Z motif is visible as a parallel offset: a lane hits a wall, drops down
#   a shaft, then continues in the same horizontal direction on another row.
#
# f) Visual cleanliness
# - Real bands each use one color and are vertically separated. Remaining rows
#   alternate two fake colors plus sparse distractor cells, so the active paths
#   stay legible instead of being buried under overlapping structures.
#
# g) Human-inferable change
# - No rule depends on color ids or hidden ordering. Same-color bars still mean
#   "real corridor"; the new inference is geometric and temporal: Z offsets
#   create the next lane, and every resting point emits one more traversal.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: every launch state has at most one legal turn. Generator
#   guarantee: each gadget seals the opposite side of the shaft.
# - Solver assumption: chain termination on an already-occupied endpoint is
#   enough to prevent loops. Generator guarantee: each band intentionally loops
#   back to a previously occupied endpoint after the visible chain is complete.
# - Solver assumption: only the two intended seed rows have valid left entrances.
#   Generator guarantee: `_break_unwanted_seeds()` corrupts every other entrance.
# ---------------------------------------------------------------------------

# Generator configuration
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 200

# Verified 3-step reaction template. Values are written for bar_start=2 and are
# shifted by +1 when bars start at col 3.
_CHAIN_WALLS = {"z1": 8, "l1": 11, "z2": 5, "l2": 3}
_CHAIN_BAR_STARTS = (0, 8)


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color bars."""
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


def _trace(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[int, int, str, list[tuple[str, str]]]:
    """Trace one launch until its next endpoint and record turns."""
    H, W = grid.shape
    turns: list[tuple[str, str]] = []
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
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, next_dir = options[0]
        turns.append((direction, next_dir))
        direction = next_dir
    return r, c, direction, turns


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid same-color corridor cell on row r, or None."""
    H, W = grid.shape
    if r <= 0 or r >= H - 1:
        return None
    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
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


def _seed_rows(grid: np.ndarray) -> list[int]:
    """Return rows whose left edge matches the original-task corridor entrance."""
    H, _ = grid.shape
    rows: list[int] = []
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(grid, r) is not None:
            rows.append(r)
    return rows


def _count_z_motifs(turns: list[tuple[str, str]]) -> int:
    """Count consecutive same-direction two-turn patterns such as right-down-right."""
    total = 0
    for (a, b), (c, d) in zip(turns, turns[1:]):
        if b == c and a == d and a != b:
            total += 1
    return total


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify both the chain reaction and the Z-corridor usage."""
    g = np.array(grid, dtype=int)
    seed_rows = _seed_rows(g)
    queue: deque[tuple[int, int, str, int]] = deque((r, 0, "right", 0) for r in seed_rows)
    launched = {(r, 0, "right") for r in seed_rows}
    occupied = {(r, 0) for r in range(g.shape[0]) if int(g[r, 0]) == PARTICLE}

    spawned = 0
    max_depth = 0
    launch_count = 0
    z_launches = 0
    z_motifs = 0

    while queue:
        r, c, direction, depth = queue.popleft()
        end_r, end_c, end_dir, turns = _trace(g, r, c, direction)
        launch_count += 1
        motifs = _count_z_motifs(turns)
        if motifs > 0:
            z_launches += 1
            z_motifs += motifs
        if (end_r, end_c) == (r, c) or (end_r, end_c) in occupied:
            continue
        occupied.add((end_r, end_c))
        spawned += 1
        max_depth = max(max_depth, depth + 1)
        next_state = (end_r, end_c, _OPPOSITE[end_dir])
        if next_state not in launched:
            launched.add(next_state)
            queue.append((end_r, end_c, _OPPOSITE[end_dir], depth + 1))

    baseline = 0
    seen_endpoints: set[tuple[int, int]] = set()
    seed_z_motifs = 0
    for r in seed_rows:
        end_r, end_c, _end_dir, turns = _trace(g, r, 0, "right")
        seed_z_motifs += _count_z_motifs(turns)
        if (end_r, end_c) != (r, 0) and (end_r, end_c) not in seen_endpoints:
            seen_endpoints.add((end_r, end_c))
            baseline += 1

    return {
        "seed_count": len(seed_rows),
        "launch_count": launch_count,
        "spawned": spawned,
        "max_depth": max_depth,
        "extra_over_single_step": spawned - baseline,
        "z_launches": z_launches,
        "z_motifs": z_motifs,
        "extra_z_over_seed_step": z_motifs - seed_z_motifs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Accumulate magenta particles through chain-reacting Z-offset corridors.

    Input structure:
    - bg=7 and particle=6. Col 0 is filled with particles, col 1 is bg.
    - Even rows are colored bar rows; odd rows are corridor channels. Real
      corridors are bg cells flanked by same-color bars, and some real lanes
      include Z-shaped offsets into a parallel channel.

    Rule:
    - Every valid seed row launches one particle to the right through bg cells.
    - When the next cell is colored, the particle turns into the unique legal
      perpendicular corridor cell. This handles straight, L, and Z paths using
      the same local corridor test.
    - When a particle stops, the endpoint keeps a particle and emits a new one
      in the opposite terminal direction. The chain stops when a launch cannot
      move or would end on an already occupied endpoint.

    Key insight:
    - Same-color bars still define where turns are legal, but endpoints are no
      longer terminal output markers. Each endpoint becomes the start of the
      next leg, and the Z offsets provide the parallel continuation lanes that
      make the cascade visibly longer than the original one-step task.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    queue: deque[tuple[int, int, str]] = deque()
    launched: set[tuple[int, int, str]] = set()

    for r in _seed_rows(g):
        state = (r, 0, "right")
        queue.append(state)
        launched.add(state)

    while queue:
        r, c, direction = queue.popleft()
        end_r, end_c, end_dir, _turns = _trace(g, r, c, direction)
        if (end_r, end_c) == (r, c):
            continue
        if int(out[end_r, end_c]) == PARTICLE:
            continue
        out[end_r, end_c] = PARTICLE
        next_state = (end_r, end_c, _OPPOSITE[end_dir])
        if next_state not in launched:
            launched.add(next_state)
            queue.append(next_state)

    return out


def _build_l_up_right(
    g: np.ndarray, top: int, bot: int, wall_col: int, color: int, ceiling: int
) -> None:
    """Right-moving lane rises to a ceiling; reverse descent exits left."""
    row = top + 1
    shaft = wall_col - 1
    g[row, wall_col] = color
    g[top, shaft] = BG
    g[top, shaft - 1] = color
    g[top, shaft + 1] = color
    g[bot, shaft] = color
    g[bot, shaft + 1] = BG
    for r_br in range(ceiling, top, 2):
        g[r_br, shaft] = BG
    if ceiling > 0:
        g[ceiling - 1, shaft] = color


def _build_l_up_left(
    g: np.ndarray, top: int, bot: int, wall_col: int, color: int, ceiling: int
) -> None:
    """Mirrored up-L: left-moving lane rises to a ceiling; reverse exits right."""
    row = top + 1
    shaft = wall_col + 1
    g[row, wall_col] = color
    g[top, shaft] = BG
    g[top, shaft - 1] = color
    g[top, shaft + 1] = color
    g[bot, shaft] = color
    g[bot, shaft - 1] = BG
    for r_br in range(ceiling, top, 2):
        g[r_br, shaft] = BG
    if ceiling > 0:
        g[ceiling - 1, shaft] = color


def _build_z_down_right(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Right-moving lane descends via a Z offset and keeps moving right."""
    row = top + 1
    shaft = wall_col - 1
    g[row, wall_col] = color
    g[mid, shaft] = BG
    g[top, shaft] = color
    g[bot, shaft - 1] = BG


def _build_z_down_left(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Left-moving lane descends via a mirrored Z offset and keeps moving left."""
    row = top + 1
    shaft = wall_col + 1
    g[row, wall_col] = color
    g[mid, shaft] = BG
    g[top, shaft] = color
    g[bot, shaft + 1] = BG


def _apply_chain_band(
    g: np.ndarray,
    bar_row0: int,
    color: int,
    break_color_1: int,
    break_color_2: int,
    bar_start: int,
    ceiling: int,
) -> None:
    """Render one verified 3-step reaction band with explicit Z offsets."""
    shift = bar_start - 2
    walls = {name: col + shift for name, col in _CHAIN_WALLS.items()}

    top0 = bar_row0
    top1 = bar_row0 + 2
    top2 = bar_row0 + 4
    top3 = bar_row0 + 6

    _build_z_down_right(g, top0, top1, top2, walls["z1"], color)
    _build_l_up_right(g, top1, top2, walls["l1"], color, ceiling)
    _build_z_down_left(g, top1, top2, top3, walls["z2"], color)
    _build_l_up_left(g, top2, top3, walls["l2"], color, ceiling)

    # Lower channels are reaction lanes, not independent seeds.
    g[top2, bar_start] = break_color_1
    g[top3, bar_start] = break_color_2


def _paint_base_bars(
    H: int, W: int, colors: list[int], bar_start: int
) -> tuple[np.ndarray, dict[int, int]]:
    """Paint alternating fake bars first; real chain bands overwrite them later."""
    g = np.full((H, W), BG, dtype=int)
    bar_rows = list(range(0, H, 2))
    row_colors: dict[int, int] = {}
    fake_cycle = [colors[2], colors[3]]
    for i, br in enumerate(bar_rows):
        row_colors[br] = fake_cycle[i % len(fake_cycle)]
        g[br, bar_start:W] = row_colors[br]
    return g, row_colors


def _decorate_fake_rows(
    g: np.ndarray, row_colors: dict[int, int], bar_start: int, colors: list[int]
) -> None:
    """Add sparse distractor cells on mismatched channels."""
    H, W = g.shape
    for top in range(0, H - 2, 2):
        bot = top + 2
        if row_colors[top] == row_colors[bot]:
            continue
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c in cols:
            if int(g[row, c]) == BG:
                g[row, c] = random.choice([row_colors[top], row_colors[bot], colors[2], colors[3]])


def _break_unwanted_seeds(
    g: np.ndarray, desired_rows: set[int], bar_start: int, colors: list[int]
) -> None:
    """Corrupt every left entrance except the intended chain seeds."""
    for r in _seed_rows(g):
        if r in desired_rows:
            continue
        top = r - 1
        bot = r + 1
        for alt in colors:
            if alt != int(g[top, bar_start]):
                g[bot, bar_start] = alt
                break


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the chain-reaction + Z-corridor combo."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    colors = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g, row_colors = _paint_base_bars(H, W, colors, bar_start)

    chain_color_1, chain_color_2 = colors[0], colors[1]
    for br in range(_CHAIN_BAR_STARTS[0], _CHAIN_BAR_STARTS[0] + 8, 2):
        row_colors[br] = chain_color_1
        g[br, bar_start:W] = chain_color_1
    for br in range(_CHAIN_BAR_STARTS[1], _CHAIN_BAR_STARTS[1] + 8, 2):
        if br >= H:
            return None
        row_colors[br] = chain_color_2
        g[br, bar_start:W] = chain_color_2

    _apply_chain_band(
        g, _CHAIN_BAR_STARTS[0], chain_color_1, colors[2], colors[3], bar_start, ceiling=0
    )
    _apply_chain_band(
        g,
        _CHAIN_BAR_STARTS[1],
        chain_color_2,
        colors[3],
        colors[2],
        bar_start,
        ceiling=_CHAIN_BAR_STARTS[1],
    )
    _decorate_fake_rows(g, row_colors, bar_start, colors)

    g[:, 0] = PARTICLE
    g[:, 1] = BG
    _break_unwanted_seeds(g, desired_rows={1, 9}, bar_start=bar_start, colors=colors)

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _combo_metrics(g)
    if metrics["seed_count"] != 2:
        return None
    if metrics["spawned"] < 6:
        return None
    if metrics["max_depth"] < 3:
        return None
    if metrics["extra_over_single_step"] < 4:
        return None
    if metrics["z_launches"] < 4 or metrics["z_motifs"] < 6:
        return None
    if metrics["extra_z_over_seed_step"] < 4:
        return None
    if len(set(g.flatten().tolist())) != 6:
        return None
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
