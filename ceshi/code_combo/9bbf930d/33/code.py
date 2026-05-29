from __future__ import annotations

# Combo variant of 9bbf930d (index 33)
# combo_spec: D19/2+D16/0
# Name: 多列缓冲区+间隙宽度限制
#
# Design analysis
# - Concrete behavior:
#   The original task had a 1-cell launch buffer at col 1 and then a corridor
#   beginning at col 2 or 3. This combo widens the launch buffer to cols 1-3
#   and shifts every horizontal bar start to col 4. A real particle therefore
#   begins with a 3-cell unsupported orange run before it reaches the first
#   supported corridor cell at col 4.
# - Small traced example:
#   On a real channel, cols 1-3 are orange buffer, col 4 is the first supported
#   corridor cell, cols 8-10 might be another orange gap, and cols 11+ restore
#   the corridor. Original logic would cross both the launch gap and the later
#   gap regardless of width. This combo uses N=3: the particle may traverse the
#   3-cell launch buffer and also any internal gap of width 3, but it stops on
#   the third cell of a width-4/5 internal gap because attempting the next cell
#   would make the unsupported streak length 4>N.
# - Complexity calibration:
#   Every generated grid contains both:
#   (1) a real particle that visibly crosses the 3-cell launch buffer and a
#       width-3 internal gap, and
#   (2) another real particle that enters through the same 3-cell buffer but is
#       stopped by a width-4 or width-5 internal gap.
#   This makes the fused rule inferable as "3 is allowed, 4 is not" instead of
#   looking like a generic hole-blocking rule or a pure layout shift.
# - Implicit assumptions audit:
#   The original `main()` assumed the first real entrance sits at col 2 or 3 and
#   that any background run is traversable. Both assumptions break here. The new
#   solver therefore:
#   (a) requires cols 1-3 to be plain background,
#   (b) requires the first supported real corridor cell to be exactly at col 4,
#   (c) tracks unsupported-run length during tracing.
#   Turn logic still assumes that a valid turn target is itself a supported
#   corridor cell; the generator preserves that by never carving width features
#   onto the turn cell.
# - Placement legality:
#   The generator paints cols 1-3 as background only after all corridor and fake
#   decorations are rendered, so no bar pixel can leak into the buffer. Width
#   features are reserved on two forced-straight real channels, which prevents
#   overlap with vertical turn structures and guarantees enough horizontal room.
# - Visual observability:
#   Humans can see the left margin difference immediately because every example
#   has a 3-column orange buffer before any bars begin. They can also compare
#   one row that crosses a 3-wide hole against another row that stops at a 4/5-
#   wide hole. The repeated coexistence of both situations is the evidence for
#   the exact threshold.
# - Cleanliness:
#   The scene remains the original striped corridor world. The only added visual
#   changes are the wider left launch buffer and a small number of bar erasures
#   placed on dedicated rows, so the grid stays readable instead of becoming a
#   pile-up of overlapping markers.

import random
import numpy as np

BG = 7
PARTICLE = 6
BUFFER_WIDTH = 3
CORRIDOR_START = 1 + BUFFER_WIDTH
MAX_GAP_WIDTH = 3

_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is supported by matching same-color walls for `direction`."""
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


def _trace_with_limit(
    grid: np.ndarray, r: int, c: int, max_gap_width: int | None
) -> tuple[int, int]:
    """Trace a particle while capping consecutive unsupported BG cells."""
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
            if max_gap_width is not None and next_run > max_gap_width:
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


def _trace_unbounded(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original behavior for comparison: coast through arbitrarily long BG runs."""
    return _trace_with_limit(grid, r, c, None)


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Combo behavior: unsupported traversal is capped at width 3."""
    return _trace_with_limit(grid, r, c, MAX_GAP_WIDTH)


def _has_buffered_entrance(grid: np.ndarray, r: int) -> bool:
    """True iff row r has the fixed 3-cell buffer and a real entrance at col 4."""
    H, _ = grid.shape
    if not (0 < r < H - 1):
        return False
    if any(int(grid[r, c]) != BG for c in range(1, CORRIDOR_START)):
        return False
    return _has_corridor(grid, r, CORRIDOR_START, "right")


def _main_from_trace(grid: np.ndarray, tracer) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE or not _has_buffered_entrance(g, r):
            continue
        end_r, end_c = tracer(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through shifted corridors with bounded gap crossing.

    Input structure:
    - bg=7. Particles=6 in col 0. Cols 1-3 are a pure background buffer.
    - Colored bars on even rows start at col 4 and form real/fake corridors on odd rows.

    Rule:
    - A particle may move through background while keeping direction, but only
      across unsupported runs of width at most 3.
    - When blocked by a colored cell, it turns 90 degrees into the unique
      perpendicular supported corridor, if one exists. Otherwise it stops.

    Key insight:
    - The 3-cell launch buffer is itself part of the width-limit evidence:
      real particles must survive that exact gap to enter the corridor, but
      wider later gaps still stop them.
    """
    return _main_from_trace(grid, _trace)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START = CORRIDOR_START
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.12
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 6
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_SHORT_GAP_OPTIONS = [3]
_LONG_GAP_OPTIONS = [4, 5]
_MAX_EXTRA_LONG_GAPS = 1

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
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
    return c_v


def _pick_feature_channels(real_indices: list[int]) -> tuple[int, int]:
    short_i, long_i = random.sample(real_indices, 2)
    return short_i, long_i


def _plan_u_turns(
    real_indices: list[int],
    protected: set[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        if i in protected:
            continue
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if random.random() < 0.5:
                plans[j] = {
                    "kind": "u",
                    "target": i,
                    "direction": "up",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[j] = {"kind": "u_target"}


def _plan_remaining(
    real_indices: list[int],
    forced_straight: set[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    for i in real_indices:
        if i in plans:
            continue
        if i in forced_straight or random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _build_l_turn(
    g: np.ndarray,
    H: int,
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
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    g[row, c_v] = bar_color[top]

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = bar_color[top]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != bar_color[top]]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = bar_color[top]


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = int(plan["c_v"])

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _carve_gap(g: np.ndarray, top: int, bot: int, start: int, length: int) -> None:
    g[top, start : start + length] = BG
    g[bot, start : start + length] = BG


def _apply_width_features(
    g: np.ndarray,
    W: int,
    bar_rows: list[int],
    short_channel: int,
    long_channel: int,
) -> list[dict[str, int]] | None:
    features = []
    rows = [("short", short_channel, _SHORT_GAP_OPTIONS), ("long", long_channel, _LONG_GAP_OPTIONS)]
    min_start = CORRIDOR_START + 2
    max_end = W - 3
    if max_end < min_start:
        return None

    used_spans: list[tuple[int, int]] = []
    for label, channel, options in rows:
        feasible = [length for length in options if max_end - min_start + 1 >= length]
        if not feasible:
            return None
        length = random.choice(feasible)
        starts = list(range(min_start, max_end - length + 2))
        random.shuffle(starts)
        start = None
        for cand in starts:
            span = (cand, cand + length - 1)
            if any(not (span[1] < lo or span[0] > hi) for lo, hi in used_spans):
                continue
            start = cand
            used_spans.append(span)
            break
        if start is None:
            return None
        top, bot = bar_rows[channel], bar_rows[channel + 1]
        _carve_gap(g, top, bot, start, length)
        features.append(
            {
                "kind": label,
                "channel": channel,
                "row": top + 1,
                "start": start,
                "length": length,
            }
        )

    return features


def _segment_max_end(plan: dict[str, object], W: int) -> int | None:
    if plan["kind"] == "straight":
        return W - 3
    if plan["kind"] in {"l_edge", "l_internal", "u"}:
        return int(plan["c_v"]) - 2
    return None


def _add_extra_long_gap(
    g: np.ndarray,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    real_indices: list[int],
    reserved: set[int],
) -> None:
    candidates = []
    for i in real_indices:
        if i in reserved:
            continue
        plan = plans.get(i, {"kind": "straight"})
        if plan["kind"] == "u_target":
            continue
        max_end = _segment_max_end(plan, W)
        min_start = CORRIDOR_START + 2
        if max_end is None or max_end - min_start + 1 < min(_LONG_GAP_OPTIONS):
            continue
        candidates.append((i, min_start, max_end))
    random.shuffle(candidates)
    if not candidates or random.random() >= 0.4:
        return
    i, min_start, max_end = candidates[0]
    length = random.choice([x for x in _LONG_GAP_OPTIONS if max_end - min_start + 1 >= x])
    start = random.randint(min_start, max_end - length + 1)
    top, bot = bar_rows[i], bar_rows[i + 1]
    _carve_gap(g, top, bot, start, length)


def _verify_width_features(g: np.ndarray, features: list[dict[str, int]]) -> bool:
    saw_long_divergence = False
    saw_short_match = False
    for feat in features:
        row = feat["row"]
        try:
            new_end = _trace(g, row, 0)
            old_end = _trace_unbounded(g, row, 0)
        except Exception:
            return False
        if feat["kind"] == "short":
            if new_end != old_end:
                return False
            if old_end[1] <= feat["start"] + feat["length"] - 1:
                return False
            saw_short_match = True
        else:
            expected_col = feat["start"] + MAX_GAP_WIDTH - 1
            if new_end == old_end:
                return False
            if new_end != (row, expected_col):
                return False
            if old_end[1] <= feat["start"] + feat["length"] - 1:
                return False
            saw_long_divergence = True
    return saw_long_divergence and saw_short_match


def _verify_buffer_synergy(g: np.ndarray, real_indices: list[int], bar_rows: list[int]) -> bool:
    """Ensure the 3-cell buffer matters: budget-2 would fail at the entrance."""
    strict_budget = MAX_GAP_WIDTH - 1
    for i in real_indices:
        row = bar_rows[i] + 1
        if not _has_buffered_entrance(g, row):
            continue
        combo_end = _trace(g, row, 0)
        strict_end = _trace_with_limit(g, row, 0, strict_budget)
        if combo_end[1] > CORRIDOR_START and strict_end == (row, strict_budget):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
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
        g[br, _BAR_START:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    short_channel, long_channel = _pick_feature_channels(real_indices)
    protected = {short_channel, long_channel}

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, protected, bar_rows, bar_color, _BAR_START, W, used_vcols, plans)
    _plan_remaining(real_indices, protected, _BAR_START, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, _BAR_START, extras)

    features = _apply_width_features(g, W, bar_rows, short_channel, long_channel)
    if not features:
        return None
    _add_extra_long_gap(g, W, plans, bar_rows, real_indices, protected)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(_BAR_START + 1, W), min(n_decor, W - _BAR_START - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1:CORRIDOR_START] = BG

    try:
        out = main(g)
        out_unbounded = _main_from_trace(g, _trace_unbounded)
    except Exception:
        return None

    if not _verify_width_features(g, features):
        return None
    if not _verify_buffer_synergy(g, real_indices, bar_rows):
        return None
    if np.array_equal(out, out_unbounded):
        return None

    n_moved = sum(
        1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
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
