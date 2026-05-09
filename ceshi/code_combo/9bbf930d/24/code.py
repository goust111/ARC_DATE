# Combo variant of 9bbf930d (index 24)
# combo_spec: D7/3+D13/1
# Name: 擦除走廊条对+链式反应
#
# Design analysis for this combo
# a) Concrete behavior and execution trace
# - The source task still uses striped same-color bar pairs as corridors and launches
#   seeds from column 0 to the right. The combo changes two points in the event order:
#   each completed segment erases the wall pair flanking every traveled corridor cell,
#   and the endpoint then launches one more segment in the opposite direction.
# - The important timing detail is that erasure is applied after a whole segment is
#   traced, not step-by-step. Otherwise a segment could destroy its own next turn
#   before reaching it, which would contradict the spec's "first segment affects later
#   linked segments" reading.
# - In the reused chain-band template, one seed travels right, turns, and reaches a top
#   endpoint. Its traveled wall pairs are then erased. The relaunched particle can still
#   move straight back down the already-cleared shaft because straight motion only checks
#   that the next cell is passable. But when it reaches the old turning area, the
#   same-color flank pair is gone, so that old turn is no longer legal and the chain
#   stops earlier than in the chain-only variant.
# b) Complexity calibration
# - A naive merge would often collapse to either "erase only" or "chain only". To keep
#   both effects visible, the generator uses two repeated 3-row chain bands. Each band
#   produces more than one endpoint, but fewer than the no-erasure chain-only baseline.
# - This keeps the new freedom small and repeated: every sample shows the same causal
#   pattern twice, so the rule is inferable from a few examples instead of depending on
#   one-off geometry.
# c) Hidden assumptions audited against the original main()
# - The original code traced each seed independently on the untouched input grid. That
#   is no longer valid because later launches must see previously erased walls.
# - The D7 single-variant kept tracing on the untouched grid; that would break the combo.
#   Here we maintain a mutable live grid and only mutate it between segments.
# - The D13 single-variant assumed starts remain and endpoints accumulate. We keep that:
#   clearing starts would hide the chain-reaction multiplication signal.
# - The original solver only needed the final cell. The combo also needs terminal
#   direction, because the next launched state depends on it.
# d) Placement legality
# - The generator reuses the verified D13 chain-band gadgets, which already avoid
#   ambiguous perpendicular turns and fit within the original 17-23 by 14-21 limits.
# - Unwanted seed rows are still broken at the left entrance, and the combo metrics
#   validate the finished grid before it is accepted.
# e) Visual observability
# - The output now shows two simultaneous signals: missing colored bar pixels along the
#   first traveled segment, and extra magenta particles created by the relaunch.
# - Comparing against the chain-only baseline, the combo bands stop one launch earlier,
#   so the missing-wall effect has an observable consequence rather than being cosmetic.
# f) Visual cleanliness
# - Real chain bands use one color each and are vertically separated. Fake rows keep the
#   remaining colors, so erased walls and spawned particles stay easy to localize.
# g) Color inference
# - No rule depends on absolute color ids. Colors only mark corridor membership; the
#   transformation depends on corridor structure, travel order, and whether wall pairs
#   remain intact.

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color walls for this direction."""
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


def _corridor_walls(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the two colored wall cells flanking (r,c) in the travel direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = (r - 1, c), (r + 1, c)
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = (r, c - 1), (r, c + 1)
    va, vb = int(grid[a]), int(grid[b])
    if va != vb or va in {PARTICLE, BG}:
        return None
    return a, b


def _trace_segment(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[int, int, str, set[tuple[int, int]]]:
    """Trace one segment on the current live grid and collect wall cells to erase."""
    eaten: set[tuple[int, int]] = set()
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            walls = _corridor_walls(grid, r, c, direction)
            if walls is not None:
                eaten.update(walls)
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]:
                if _has_corridor(grid, tr, tc, turn_dir):
                    options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        walls = _corridor_walls(grid, r, c, direction)
        if walls is not None:
            eaten.update(walls)
    return r, c, direction, eaten


def _seed_rows(grid: np.ndarray) -> list[int]:
    """Return source-task seed rows whose left entrance is a valid real corridor."""
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


def _simulate(grid: np.ndarray, erase_between_segments: bool) -> tuple[np.ndarray, dict[str, int]]:
    """Run the chain reaction, optionally erasing traversed wall pairs between segments."""
    g = np.array(grid, dtype=int)
    live = g.copy()
    out = g.copy()
    seed_rows = _seed_rows(g)
    queue: deque[tuple[int, int, str, int]] = deque((r, 0, "right", 0) for r in seed_rows)
    launched = {(r, 0, "right") for r in seed_rows}
    occupied = {(r, 0) for r in range(g.shape[0]) if int(g[r, 0]) == PARTICLE}

    spawned = 0
    max_depth = 0
    erased_walls = 0
    moved_segments = 0
    erasing_segments = 0

    while queue:
        r, c, direction, depth = queue.popleft()
        end_r, end_c, end_dir, eaten = _trace_segment(live, r, c, direction)
        if (end_r, end_c) == (r, c):
            continue
        moved_segments += 1
        erased_walls += len(eaten)
        if eaten:
            erasing_segments += 1
        if erase_between_segments:
            for rr, cc in eaten:
                live[rr, cc] = BG
                out[rr, cc] = BG
        if (end_r, end_c) in occupied:
            continue
        occupied.add((end_r, end_c))
        out[end_r, end_c] = PARTICLE
        spawned += 1
        max_depth = max(max_depth, depth + 1)
        next_state = (end_r, end_c, _OPPOSITE[end_dir])
        if next_state not in launched:
            launched.add(next_state)
            queue.append((end_r, end_c, _OPPOSITE[end_dir], depth + 1))

    return out, {
        "seed_count": len(seed_rows),
        "spawned": spawned,
        "max_depth": max_depth,
        "erased_walls": erased_walls,
        "moved_segments": moved_segments,
        "erasing_segments": erasing_segments,
    }


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    """Measure whether the generated sample really uses both erasure and chaining."""
    _, combo = _simulate(grid, erase_between_segments=True)
    _, chain_only = _simulate(grid, erase_between_segments=False)
    combo["static_spawned"] = chain_only["spawned"]
    combo["static_max_depth"] = chain_only["max_depth"]
    combo["spawn_loss_vs_chain_only"] = chain_only["spawned"] - combo["spawned"]
    combo["depth_loss_vs_chain_only"] = chain_only["max_depth"] - combo["max_depth"]
    return combo


def main(grid: np.ndarray) -> np.ndarray:
    """Run the erase-and-relaunch corridor combo.

    Input structure:
    - bg=7 and particle=6.
    - Column 0 contains seed particles, column 1 is bg, and striped colored bars form
      horizontal corridors plus L/U-turn gadgets exactly as in the source task.

    Rule:
    - Every valid seed launches rightward and traces one segment on the current grid.
    - After that segment finishes, each traveled corridor cell erases its two same-color
      flanking wall pixels to bg.
    - The endpoint keeps a particle and emits one new segment in the opposite direction
      of the terminal corridor.
    - Later launches use the already-erased live grid, so they may still move straight
      through bg but can no longer take turns whose same-color wall pairs were eaten.

    Key insight:
    - The path geometry is stable within a segment, but not across segments: the first
      pass leaves visible holes in the corridor walls, and those missing wall pairs are
      exactly what shortens the later chain reaction.
    """
    out, _ = _simulate(grid, erase_between_segments=True)
    return out


# --------------- generator (sa_max_effort) ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 200

# Verified D13 chain-band template. In the combo it produces one successful relaunch
# per band before the erased first segment disables the next turn.
_CHAIN_WALLS = {"u1": 8, "l1": 11, "u2": 5, "l2": 3}
_CHAIN_BAR_STARTS = (0, 8)
_MIN_ERASED_WALLS = 28


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


def _build_u_down_right(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Original U-turn: right-moving source enters the lower lane still moving right."""
    row = top + 1
    shaft = wall_col - 1
    g[row, wall_col] = color
    g[mid, shaft] = BG
    g[top, shaft] = color
    g[bot, shaft - 1] = BG


def _build_u_down_left(
    g: np.ndarray, top: int, mid: int, bot: int, wall_col: int, color: int
) -> None:
    """Horizontal mirror of the original U-turn for left-moving re-entry."""
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
    """Render one 3-step chain-only band; in the combo it collapses to 2 visible steps."""
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

    g[top2, bar_start] = break_color_1
    g[top3, bar_start] = break_color_2


def _paint_base_bars(H: int, W: int, colors: list[int], bar_start: int) -> tuple[np.ndarray, dict[int, int]]:
    """Paint alternating fake bars first; real chain bands overwrite selected rows."""
    g = np.full((H, W), BG, dtype=int)
    row_colors: dict[int, int] = {}
    fake_cycle = [colors[2], colors[3]]
    for i, br in enumerate(range(0, H, 2)):
        row_colors[br] = fake_cycle[i % len(fake_cycle)]
        g[br, bar_start:W] = row_colors[br]
    return g, row_colors


def _decorate_fake_rows(
    g: np.ndarray, row_colors: dict[int, int], bar_start: int, colors: list[int]
) -> None:
    """Add a few distractor cells on rows whose flanking bars mismatch."""
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
    """Keep only the intended chain-band seed rows active at the left edge."""
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

    chain_color_1, chain_color_2 = colors[0], colors[1]
    for br in range(_CHAIN_BAR_STARTS[0], _CHAIN_BAR_STARTS[0] + 8, 2):
        row_colors[br] = chain_color_1
        g[br, bar_start:W] = chain_color_1
    for br in range(_CHAIN_BAR_STARTS[1], _CHAIN_BAR_STARTS[1] + 8, 2):
        if br >= H:
            return None
        row_colors[br] = chain_color_2
        g[br, bar_start:W] = chain_color_2

    _apply_chain_band(g, _CHAIN_BAR_STARTS[0], chain_color_1, colors[2], colors[3], bar_start, 0)
    _apply_chain_band(
        g,
        _CHAIN_BAR_STARTS[1],
        chain_color_2,
        colors[3],
        colors[2],
        bar_start,
        _CHAIN_BAR_STARTS[1],
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
    if metrics["spawned"] < 4:
        return None
    if metrics["max_depth"] < 2:
        return None
    if metrics["static_spawned"] - metrics["spawned"] < 2:
        return None
    if metrics["erased_walls"] < _MIN_ERASED_WALLS:
        return None
    if metrics["erasing_segments"] < 2:
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
