# Combo variant of 9bbf930d (index 15)
# combo_spec: D16/0+D8/0
# Name: 间隙宽度限制+条对部分断裂

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
MAX_GAP_WIDTH = 2

_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}

# ---------------------------------------------------------------------------
# Design analysis for combo D16/0 + D8/0
#
# a) Concrete behavior and traced example
#    Original task:
#    - Horizontal real corridors are recognized only when the two bar rows have
#      the same color in the exact same column.
#    - The particle coasts through any number of BG cells until it meets a
#      colored blocker, then it tries a perpendicular turn.
#    Combo task:
#    - Horizontal bar pairs are partially broken, so the top and bottom rails
#      often look like staggered dashed lines instead of solid strips.
#    - A horizontal corridor cell is still valid if each bar row shows the same
#      color within radius 1 of that column. This lets the solver treat a small
#      rail crack as part of one corridor.
#    - While moving straight, we count consecutive cells that are no longer
#      supported by such local rails. The particle may traverse at most N=2
#      unsupported cells. The third unsupported cell is terminal.
#    Small example:
#      top : C . C . . . . C . C
#      mid : . . . . . . . . . .
#      bot : . C . C . . . . C .
#      cols: 0 1 2 3 4 5 6 7 8 9
#    - Around cols 0-3 and 7-9, radius-1 support exists, so the row is a real
#      dashed corridor.
#    - The longer simultaneous break at cols 4-7 creates unsupported cells in
#      the middle. Original logic would ignore this and keep moving; the combo
#      logic stops once the unsupported streak would exceed 2.
#
# b) Complexity calibration
#    - The dashed-rail rule is repeated across many rows, so the D8 change is
#      visible globally rather than being hidden in one local exception.
#    - Every generated grid contains at least one traversable long-looking break
#      and one wider break that stops a particle. This makes N inferable from a
#      few examples instead of looking like random blockage.
#    - We keep only straight corridors and L-turns in the generator. That keeps
#      the routing space rich enough for ARC while avoiding unnecessary overlap
#      between U-turn geometry and broken-rail geometry.
#
# c) Implicit-assumption review of the original main()
#    - Original horizontal corridor test assumed exact same-column overlap.
#      Broken rails violate that, so `_has_corridor()` must switch to local
#      radius-1 support for left/right motion.
#    - Original entrance detection assumed the first exact overlap column marks
#      a real corridor. Here many real rows have no exact overlap anywhere, so
#      entrance detection must also use the local-support test.
#    - Original tracing assumed all BG ahead is safe. That fails after adding a
#      width cap, so `_trace()` must inspect every traversed cell and maintain a
#      direction-specific unsupported run length.
#    - Vertical turn recognition still relies on exact left/right wall pixels.
#      The generator therefore preserves those turn flanks exactly so no new
#      ambiguity is introduced.
#
# d) Placement legality
#    - Vertical turns reserve a 3-column window `[c_v-2, c_v]`, keeping all
#      required flank cells inside the grid and separating turns from each other.
#    - Horizontal width-feature breaks are carved only before the first blocker
#      of a channel, never through the protected turn cells.
#    - Decorative fake-channel cells are added only on unprotected BG cells.
#
# e) Visual observability
#    - A human sees many same-colored dashed rail pairs and can infer that small
#      cracks do not destroy corridor identity.
#    - Within those same dashed corridors, some wider rail absences are crossed
#      while still wider absences stop the particle earlier. The difference is
#      visible directly in the broken-rail lengths.
#
# f) Visual cleanliness
#    - Each even row still uses one bar color only; breaks are just BG holes.
#    - Turns keep the original L-shape vocabulary, so the scene stays readable:
#      dashed horizontal rails, exact vertical turn flanks, and a few channel
#      blockers.
#
# g) Inferability of the change
#    - No hidden ordering by color id is used.
#    - The only new variable is visibly geometric: how long a shared break in
#      both rails becomes before the particle stops. Repeated short/long cases
#      make the fused rule recoverable from 2-4 examples.
#
# Solver-side new assumptions and generator guarantees:
# - `_local_bar_support_color()` assumes each bar row shows at most one non-BG
#   color in a small window; the generator keeps every even row monochromatic.
# - `_trace()` assumes a perpendicular turn target is itself a supported cell;
#   `_build_l_turn()` preserves exact flank cells there.
# - `_iter_active_rows()` assumes the first visible bars of a real corridor
#   still begin near columns 2-4; the generator preserves the entrance region
#   and never carves width features that early.
# ---------------------------------------------------------------------------


def _local_bar_support_color(
    grid: np.ndarray, bar_row: int, c: int, radius: int = 1
) -> int | None:
    """Return the unique visible bar color within a small horizontal window."""
    H, W = grid.shape
    if not (0 <= bar_row < H):
        return None
    colors = set()
    for cc in range(max(0, c - radius), min(W - 1, c + radius) + 1):
        value = int(grid[bar_row, cc])
        if value not in {BG, PARTICLE}:
            colors.add(value)
    if len(colors) == 1:
        return next(iter(colors))
    return None


def _first_visible_bar_col(grid: np.ndarray, bar_row: int, color: int) -> int | None:
    for c in range(grid.shape[1]):
        if int(grid[bar_row, c]) == color:
            return c
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) belongs to a corridor under the fused rule."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        top = _local_bar_support_color(grid, r - 1, c, radius=1)
        bot = _local_bar_support_color(grid, r + 1, c, radius=1)
        return top == bot and top is not None
    if c == 0 or c == W - 1:
        return False
    left = int(grid[r, c - 1])
    right = int(grid[r, c + 1])
    return left == right and left not in {PARTICLE, BG}


def _trace_unbounded_local(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """D8-style trace: broken rails are recognized, but gap width is unbounded."""
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
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace with dashed-rail recognition and a width cap on unsupported streaks."""
    H, W = grid.shape
    direction = "right"
    unsupported_run = 0
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            next_run = unsupported_run + 1
            if _has_corridor(grid, nr, nc, direction):
                next_run = 0
            if next_run > MAX_GAP_WIDTH:
                break
            r, c = nr, nc
            unsupported_run = next_run
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
        unsupported_run = 0
    return r, c


def _iter_active_rows(grid: np.ndarray):
    """Yield particle rows that have a valid real-corridor entrance."""
    H, W = grid.shape
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if _has_corridor(grid, r, c, "right"):
                entrance_col = c
                break
        if entrance_col is None:
            continue
        top_color = _local_bar_support_color(grid, r - 1, entrance_col, radius=1)
        bot_color = _local_bar_support_color(grid, r + 1, entrance_col, radius=1)
        if top_color is None or top_color != bot_color:
            continue
        top_start = _first_visible_bar_col(grid, r - 1, top_color)
        bot_start = _first_visible_bar_col(grid, r + 1, bot_color)
        if top_start is None or bot_start is None or max(top_start, bot_start) > 4:
            continue
        yield r


def _main_from_trace(grid: np.ndarray, tracer) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    for r in _iter_active_rows(g):
        end_r, end_c = tracer(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through broken rail corridors with a width cap.

    Input structure:
    - bg=7 and particles=6 fill column 0; column 1 is background.
    - Even rows contain colored rails; odd rows are candidate channels.
    - Real horizontal corridors are same-colored rail pairs, but the rails may
      be broken into staggered dashed segments.

    Rule:
    - A horizontal channel is valid if both adjacent rail rows show the same
      color within radius 1 of the current column.
    - The particle moves straight through BG cells and only turns when a colored
      blocker forces a 90-degree turn into the unique supported corridor.
    - While moving straight, it may cross at most 2 consecutive unsupported
      cells; a wider break becomes the particle's stopping point.

    Key insight:
    - Same-colored dashed rails still define one corridor, but only short joint
      breaks can be coasted through. Wider shared breaks are visible dead ends.
    """
    return _main_from_trace(grid, _trace)


# ---------------------------------------------------------------------------
# Legacy/oracle helpers used only for generator validation and combo metrics.
# ---------------------------------------------------------------------------


def _legacy_has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
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


def _legacy_trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
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
            if 0 <= tr < H and 0 <= tc < W and _legacy_has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) > 1:
            raise AssertionError("ambiguous turn: both perpendicular directions valid")
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
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
        bar_above = next(
            (c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])),
            W,
        )
        bar_below = next(
            (c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])),
            W,
        )
        if bar_above != bar_below or bar_above > 3:
            continue
        end_r, end_c = _legacy_trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def measure_combo_behavior(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether both fused mechanisms are active on a sample."""
    g = np.array(grid, dtype=int)
    legacy_out = _legacy_main(g)
    variant_only_rows = 0
    width_limited_rows = 0
    active_rows = 0
    for r in _iter_active_rows(g):
        active_rows += 1
        new_end = _trace(g, r, 0)
        unbounded_end = _trace_unbounded_local(g, r, 0)
        if new_end != unbounded_end:
            width_limited_rows += 1
        moved_new = new_end != (r, 0)
        moved_old = int(legacy_out[r, 0]) != PARTICLE
        if moved_new and not moved_old:
            variant_only_rows += 1
    return {
        "active_rows": active_rows,
        "variant_only_rows": variant_only_rows,
        "width_limited_rows": width_limited_rows,
    }


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_STRAIGHT_PROB = 0.45

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)
_SHORT_BREAK_OPTIONS = [3, 4]
_LONG_BREAK_OPTIONS = [5, 6]
_MAX_EXTRA_BREAKS = 2

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_VARIANT_ONLY_MOVED = 2
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW) and c - 2 >= bar_start
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _build_row_colors(
    bar_rows: list[int], channel_spec: list[str], extras: list[int]
) -> dict[int, int]:
    row_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            row_color[bot] = row_color[top]
        else:
            row_color[bot] = random.choice([c for c in extras if c != row_color[top]])
    return row_color


def _register_vertical_segment(
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
    rows: list[int],
    c_v: int,
) -> None:
    for br in rows:
        must_bg[br].add(c_v - 1)
        must_keep[br].update({c_v - 2, c_v})


def _build_l_turn(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    g[row, c_v] = color
    if direction == "up":
        turn_bar, opp_bar = top, bot
        must_bg[turn_bar].add(c_v - 1)
        must_keep[turn_bar].update({c_v - 2, c_v})
        must_keep[opp_bar].add(c_v - 1)
        _register_vertical_segment(must_keep, must_bg, list(range(0, top, 2)), c_v)
    else:
        turn_bar, opp_bar = bot, top
        must_bg[turn_bar].add(c_v - 1)
        must_keep[turn_bar].update({c_v - 2, c_v})
        must_keep[opp_bar].add(c_v - 1)
        _register_vertical_segment(
            must_keep, must_bg, list(range(bot + 2, g.shape[0], 2)), c_v
        )


def _render_bar_rows(
    g: np.ndarray,
    bar_rows: list[int],
    row_color: dict[int, int],
    bar_start: int,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    W = g.shape[1]
    for idx, br in enumerate(bar_rows):
        color = row_color[br]
        phase = idx % 2
        for c in range(bar_start, W):
            if c in must_bg[br]:
                g[br, c] = BG
            elif c in must_keep[br] or (c - bar_start) % 2 == phase:
                g[br, c] = color
            else:
                g[br, c] = BG


def _segment_max_end(plan: dict[str, int | str], W: int) -> int | None:
    if plan["kind"] == "straight":
        return W - 2
    if plan["kind"] == "l":
        return int(plan["c_v"]) - 2
    return None


def _choose_feature_channels(candidates: list[dict]) -> tuple[dict, dict] | None:
    long_pool = [
        c
        for c in candidates
        if c["max_end"] - c["min_start"] + 1 >= min(_LONG_BREAK_OPTIONS)
    ]
    if not long_pool:
        return None
    long_choice = random.choice(long_pool)
    short_pool = [
        c
        for c in candidates
        if c["i"] != long_choice["i"]
        and c["max_end"] - c["min_start"] + 1 >= min(_SHORT_BREAK_OPTIONS)
    ]
    non_adjacent = [c for c in short_pool if abs(c["i"] - long_choice["i"]) > 1]
    if non_adjacent:
        short_pool = non_adjacent
    if not short_pool:
        return None
    short_choice = random.choice(short_pool)
    return long_choice, short_choice


def _carve_bar_break(g: np.ndarray, top: int, bot: int, start: int, length: int) -> None:
    g[top, start : start + length] = BG
    g[bot, start : start + length] = BG


def _apply_width_features(
    g: np.ndarray,
    W: int,
    plans: dict[int, dict[str, int | str]],
    bar_rows: list[int],
    real_indices: list[int],
    bar_start: int,
) -> list[dict] | None:
    candidates = []
    for i in real_indices:
        plan = plans.get(i, {"kind": "straight"})
        max_end = _segment_max_end(plan, W)
        if max_end is None:
            continue
        min_start = bar_start + 4
        if max_end < min_start:
            continue
        candidates.append({"i": i, "min_start": min_start, "max_end": max_end})

    picked = _choose_feature_channels(candidates)
    if picked is None:
        return None
    long_choice, short_choice = picked
    features: list[dict] = []

    for label, choice, options in [
        ("long", long_choice, _LONG_BREAK_OPTIONS),
        ("short", short_choice, _SHORT_BREAK_OPTIONS),
    ]:
        feasible_lengths = [
            length
            for length in options
            if choice["max_end"] - choice["min_start"] + 1 >= length
        ]
        if not feasible_lengths:
            return None
        length = random.choice(feasible_lengths)
        start_max = choice["max_end"] - length + 1
        if start_max < choice["min_start"]:
            return None
        start = random.randint(choice["min_start"], start_max)
        top, bot = bar_rows[choice["i"]], bar_rows[choice["i"] + 1]
        _carve_bar_break(g, top, bot, start, length)
        features.append(
            {
                "kind": label,
                "channel": choice["i"],
                "row": top + 1,
                "start": start,
                "length": length,
            }
        )

    remaining = [c for c in candidates if c["i"] not in {long_choice["i"], short_choice["i"]}]
    random.shuffle(remaining)
    extra_added = 0
    for choice in remaining:
        if extra_added >= _MAX_EXTRA_BREAKS or random.random() >= 0.45:
            continue
        length = random.randint(1, 6)
        if choice["max_end"] - choice["min_start"] + 1 < length:
            continue
        start_max = choice["max_end"] - length + 1
        if start_max < choice["min_start"]:
            continue
        start = random.randint(choice["min_start"], start_max)
        top, bot = bar_rows[choice["i"]], bar_rows[choice["i"] + 1]
        _carve_bar_break(g, top, bot, start, length)
        extra_added += 1

    return features


def _verify_width_features(g: np.ndarray, features: list[dict]) -> bool:
    saw_long = False
    saw_short = False
    for feat in features:
        row = int(feat["row"])
        new_end = _trace(g, row, 0)
        old_end = _trace_unbounded_local(g, row, 0)
        gap_end = int(feat["start"] + feat["length"] - 1)
        if feat["kind"] == "short":
            if new_end != old_end:
                return False
            if old_end[0] == row and old_end[1] <= gap_end:
                return False
            saw_short = True
        else:
            if new_end == old_end:
                return False
            if new_end[0] != row:
                return False
            if new_end[1] >= gap_end:
                return False
            if old_end[0] == row and old_end[1] <= gap_end:
                return False
            saw_long = True
    return saw_long and saw_short


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choice(_N_REAL_OPTIONS), n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    row_color = _build_row_colors(bar_rows, channel_spec, extras)
    if len(set(row_color.values())) < 4:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    must_keep = {br: set() for br in bar_rows}
    must_bg = {br: set() for br in bar_rows}

    real_indices = [i for i in range(n_channels) if channel_spec[i] == "real"]
    random.shuffle(real_indices)
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str]] = {}

    for i in real_indices:
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = row_color[top]
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        plans[i] = {"kind": "l", "direction": direction, "c_v": c_v}
        _build_l_turn(g, top, bot, row, color, c_v, direction, must_keep, must_bg)

    _render_bar_rows(g, bar_rows, row_color, bar_start, must_keep, must_bg)

    features = _apply_width_features(g, W, plans, bar_rows, real_indices, bar_start)
    if not features:
        return None

    protected_cols = {0, 1}
    for cols in must_keep.values():
        protected_cols.update(cols)
    for cols in must_bg.values():
        protected_cols.update(cols)
    protected_cols.update(np.argwhere(g != BG)[:, 1].tolist())

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        row = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        allowed = [
            c
            for c in range(bar_start + 1, W)
            if c not in protected_cols and int(g[row, c]) == BG
        ]
        if not allowed:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        for c in random.sample(allowed, min(n_decor, len(allowed))):
            g[row, c] = random.choice([row_color[top], row_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        unbounded_out = _main_from_trace(g, _trace_unbounded_local)
    except Exception:
        return None

    if not _verify_width_features(g, features):
        return None
    if np.array_equal(out, unbounded_out):
        return None

    metrics = measure_combo_behavior(g)
    if metrics["variant_only_rows"] < _MIN_VARIANT_ONLY_MOVED:
        return None
    if metrics["width_limited_rows"] < 1:
        return None

    n_moved = sum(1 for r in range(H) if int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
