# Variant of 9bbf930d
# Dimension: D13 — 多步处理与级联 (multi_step)
# Change: 单步处理——所有粒子独立计算终点，互不影响，结果一次性写入输出。... → 粒子到达终点后发射新粒子

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
# Design analysis for the D13 variant
#
# a) Concrete behavior and flow impact
# - Original task: every valid seed in col 0 traces once to a final resting cell;
#   the start cell is cleared and only that single endpoint is written.
# - Variant: a traced endpoint becomes a persistent particle, and that endpoint
#   immediately launches one more moving particle in the opposite direction of
#   the terminal corridor. This repeats until the next endpoint is either the
#   launch cell itself, or a cell that already contains a particle.
# - In pixels this means outputs now accumulate several magenta particles per
#   corridor system. The original bar/corridor geometry is unchanged; the change
#   is entirely in the event ordering and in the fact that endpoints persist.
# - Example on one generated 3-lane band:
#   row1 seed goes right -> U-turn into row3 -> L-turn upward to top endpoint E1.
#   From E1 a new particle goes downward -> on row3 it now travels left ->
#   mirrored U-turn into row5 -> mirrored L-turn upward to top endpoint E2.
#   From E2 a new particle goes downward -> on row5 it now travels right to the
#   band's far edge and stops. Original logic would only mark E1; variant marks
#   E1, E2, and the final row5 endpoint.
#
# b) Complexity calibration
# - If we simply reused the original generator, reverse launches almost always
#   bounce back to the seed and the D13 behavior becomes invisible.
# - To make the chain observable without introducing branching ambiguity, the
#   generator builds explicit 3-step reaction bands using the same L/U corridor
#   vocabulary as the source task, plus horizontal mirror images of those turns.
# - Each puzzle contains two such bands with different colors, so the reaction
#   repeats in the same sample and is visible from only a few examples.
#
# c) Implicit assumptions audited from the original main()
# - `out[r, 0] = BG` relied on "movement is a transfer". That is invalid now:
#   starts must remain because the visible change is multiplication, not motion.
# - A single `_trace(g, r, 0)` per row relied on independence between particles.
#   That is invalid now: endpoints create new launches, so processing must become
#   queue-based and stateful.
# - Using the static geometry grid is still safe because particles are passable
#   in both the original code and this variant (`PARTICLE` is treated like `BG`
#   when tracing and corridor-checking).
# - The original code only needed the endpoint position; the variant also needs
#   the terminal direction to know which opposite direction is emitted next.
#
# d) Placement legality
# - Real reaction bands occupy four consecutive even bar rows, producing three
#   channel rows. The first channel is the only legal seed; the next two have
#   their entrance signature broken at the left edge so the solver will not
#   start there directly.
# - Turn gadgets are spaced so their disambiguation cells do not overwrite each
#   other. The generator validates this by running the new solver metrics.
#
# e) Visual observability
# - The variant is visible because outputs contain several magenta particles
#   along the same colored band, not just one relocated particle.
# - Every sample contains two independent chain bands, so "endpoint creates a
#   new run" is repeated rather than shown as a one-off.
#
# f) Visual cleanliness
# - Bands use only one color per real chain, fake rows use the remaining colors,
#   and chains are vertically separated. There is no heavy overdraw or mixing of
#   incompatible structures in the same local area.
#
# g) Color inference
# - No rule depends on color ids. Colors only mark corridor membership exactly
#   as in the original task; the new behavior is driven by geometry and order.
# ---------------------------------------------------------------------------

# Generator configuration
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 200

# A verified 3-step chain template. Values are written for bar_start=2 and are
# shifted by +1 when bars start at col 3.
_CHAIN_WALLS = {"u1": 8, "l1": 11, "u2": 5, "l2": 3}
_CHAIN_BAR_STARTS = (0, 8)  # two vertically separated bands; valid for all H>=17


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


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int, str]:
    """Trace one moving particle until it reaches its next endpoint."""
    H, W = grid.shape
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
        r, c, direction = options[0]
    return r, c, direction


def _seed_rows(grid: np.ndarray) -> list[int]:
    """Return original-task seed rows whose left entrance is valid."""
    H, W = grid.shape
    rows: list[int] = []
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(grid[r, c]) != BG:
                continue
            above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next(
            (c for c in range(W) if int(grid[r - 1, c]) == int(grid[r - 1, entrance_col])),
            W,
        )
        bar_below = next(
            (c for c in range(W) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])),
            W,
        )
        if bar_above == bar_below and bar_above <= 3:
            rows.append(r)
    return rows


def _reaction_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify the cascade for validation and post-generation auditing."""
    g = np.array(grid, dtype=int)
    seed_rows = _seed_rows(g)
    queue: deque[tuple[int, int, str, int]] = deque((r, 0, "right", 0) for r in seed_rows)
    launched = {(r, 0, "right") for r in seed_rows}
    occupied = {(r, 0) for r in range(g.shape[0]) if int(g[r, 0]) == PARTICLE}

    spawned = 0
    max_depth = 0
    while queue:
        r, c, direction, depth = queue.popleft()
        end_r, end_c, end_dir = _trace(g, r, c, direction)
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
    for r in seed_rows:
        end_r, end_c, _ = _trace(g, r, 0, "right")
        if (end_r, end_c) != (r, 0) and (end_r, end_c) not in seen_endpoints:
            seen_endpoints.add((end_r, end_c))
            baseline += 1

    return {
        "seed_count": len(seed_rows),
        "spawned": spawned,
        "max_depth": max_depth,
        "extra_over_single_step": spawned - baseline,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Run the multi-step chain reaction on corridor endpoints."""
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
        end_r, end_c, end_dir = _trace(g, r, c, direction)
        if (end_r, end_c) == (r, c):
            continue
        # New particles accumulate. Reaching an already-occupied endpoint ends
        # the chain, which prevents infinite ping-pong on symmetric corridors.
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
    g[bot, shaft + 1] = BG  # reverse descent may only exit to the left
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
    g[bot, shaft - 1] = BG  # reverse descent may only exit to the right
    for r_br in range(ceiling, top, 2):
        g[r_br, shaft] = BG
    if ceiling > 0:
        g[ceiling - 1, shaft] = color


def _build_u_down_right(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Original U-turn: right-moving source enters the lower lane still moving right."""
    row = top + 1
    shaft = wall_col - 1
    g[row, wall_col] = color
    g[mid, shaft] = BG
    g[top, shaft] = color
    g[bot, shaft - 1] = BG  # only the target-side re-entry stays legal


def _build_u_down_left(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Horizontal mirror of the original U-turn for left-moving re-entry."""
    row = top + 1
    shaft = wall_col + 1
    g[row, wall_col] = color
    g[mid, shaft] = BG
    g[top, shaft] = color
    g[bot, shaft + 1] = BG  # only the target-side re-entry stays legal


def _apply_chain_band(
    g: np.ndarray,
    bar_row0: int,
    color: int,
    break_color_1: int,
    break_color_2: int,
    bar_start: int,
    ceiling: int,
) -> None:
    """Render one verified 3-step reaction band spanning four even bar rows."""
    shift = bar_start - 2
    walls = {name: col + shift for name, col in _CHAIN_WALLS.items()}

    top0 = bar_row0
    top1 = bar_row0 + 2
    top2 = bar_row0 + 4
    top3 = bar_row0 + 6

    _build_u_down_right(g, top0, top1, top2, walls["u1"], color)
    _build_l_up_right(g, top1, top2, walls["l1"], color, ceiling)
    _build_u_down_left(g, top1, top2, top3, walls["u2"], color)
    _build_l_up_left(g, top2, top3, walls["l2"], color, ceiling)

    # Break direct left entrances for the two lower channels without affecting
    # the upper seed channel, which uses the bar row above them.
    g[top2, bar_start] = break_color_1
    g[top3, bar_start] = break_color_2


def _paint_base_bars(H: int, W: int, colors: list[int], bar_start: int) -> tuple[np.ndarray, dict[int, int]]:
    """Paint alternating fake bars first, then real chains will overwrite them."""
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
    """Add a few distractor cells on mismatched channels."""
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
    """Keep only the intended seed rows active at the left edge."""
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
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    colors = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g, row_colors = _paint_base_bars(H, W, colors, bar_start)

    # Two independent 3-step real chains. Their colors differ so the in-between
    # channels remain fake and visually separated.
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

    metrics = _reaction_metrics(g)
    if metrics["seed_count"] != 2:
        return None
    if metrics["spawned"] < 6:
        return None
    if metrics["max_depth"] < 3:
        return None
    if metrics["extra_over_single_step"] < 4:
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
