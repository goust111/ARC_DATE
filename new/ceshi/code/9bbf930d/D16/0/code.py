from __future__ import annotations

# Variant of 9bbf930d
# Dimension: D16 — 惯性与间隙穿越 (path)
# Change: 粒子保持当前方向不变地穿越背景间隙（包括交叉口间隙），直到遇到彩色像素被阻挡才尝试转弯。... → 间隙宽度限制
#
# Design analysis
# - Concrete behavior:
#   The original task already allows a particle to coast through any number of
#   unsupported background cells, as long as the cells ahead are still BG.
#   In this variant, "unsupported" means the current cell is not flanked by a
#   same-color bar pair for the particle's current direction. The particle may
#   stay inertial through at most N such cells; here N=2. The third consecutive
#   unsupported cell is terminal, so the particle stops on the second one.
# - Small traced example:
#   Suppose a real horizontal corridor is present at columns 2-4, then both bar
#   rows are erased at columns 5-8, then the corridor resumes at columns 9-11.
#   Original logic: the particle crosses 5-8, reaches 9, then continues/turns.
#   Variant logic: the particle reaches 5, then 6, but attempting 7 would make
#   the unsupported streak length 3>N, so it stops at 6.
# - Complexity calibration:
#   To keep the new rule visible, every generated instance contains both:
#   (1) at least one short gap of width 1-2 that is traversed successfully, and
#   (2) at least one long gap of width 3-4 that stops the particle early.
#   This makes the threshold inferable from a few examples instead of looking
#   like a generic "holes block movement" rule.
# - Implicit assumptions in main():
#   The original `_trace()` relied on "all BG ahead is safe to traverse".
#   That is no longer valid. We therefore inspect every traversed BG cell and
#   maintain a run length of direction-specific unsupported cells. Turn logic is
#   unchanged and still assumes that a turn target must itself be a supported
#   corridor cell. The entrance logic in `main()` still assumes the first real
#   corridor begins at column 2 or 3; the generator preserves that.
# - Placement legality:
#   Gap features are carved only on horizontal segments with guaranteed room
#   before the first blocker/turn, so they do not overwrite turn cells. Feature
#   channels are chosen from channels that actually move, and the generator
#   validates each feature by tracing it with both the bounded and unbounded
#   solvers.
# - Visual observability:
#   Long gaps are visible as obvious orange holes inside an otherwise valid
#   corridor. Short and long gaps coexist in the same grid, so a human can see
#   that some holes are crossed while wider holes stop the particle.
# - Cleanliness:
#   The layout remains the original striped corridor world. The only new visual
#   mark is a small number of bar erasures, which are spatially localized and do
#   not introduce extra symbols or hidden color-based encodings.

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


def _trace_unbounded(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original D16 behavior: coast through arbitrarily long BG runs."""
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
    """Trace particle with a width-limited unsupported-gap traversal budget."""
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


def _main_from_trace(grid: np.ndarray, tracer) -> np.ndarray:
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
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = tracer(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles, but inertial BG traversal is capped at width 2."""
    return _main_from_trace(grid, _trace)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_SHORT_GAP_OPTIONS = [1, 2]
_LONG_GAP_OPTIONS = [3, 4]
_MAX_EXTRA_LONG_GAPS = 1

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start, W, used_vcols):
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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans):
    by_color = {}
    for i in real_indices:
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


def _plan_remaining(real_indices, bar_start, W, used_vcols, plans):
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras):
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = plan["c_v"]

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = plan["direction"]
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at):
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


def _build_u_turn(g, W, top, bot, row, c_v, direction, j_top, j_bot,
                  bar_color, bar_start, extras, internal_stop):
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


def _segment_max_end(plan, W):
    if plan["kind"] == "straight":
        return W - 3
    if plan["kind"] in {"l_edge", "l_internal", "u"}:
        return plan["c_v"] - 2
    return None


def _choose_gap_channels(candidates):
    random.shuffle(candidates)
    long_choice = None
    short_choice = None

    long_pool = [c for c in candidates if c["max_end"] - c["min_start"] + 1 >= min(_LONG_GAP_OPTIONS)]
    if not long_pool:
        return None
    long_choice = random.choice(long_pool)

    short_pool = [
        c
        for c in candidates
        if c["i"] != long_choice["i"] and c["max_end"] - c["min_start"] + 1 >= min(_SHORT_GAP_OPTIONS)
    ]
    non_adjacent_short = [c for c in short_pool if abs(c["i"] - long_choice["i"]) > 1]
    if non_adjacent_short:
        short_pool = non_adjacent_short
    if not short_pool:
        return None
    short_choice = random.choice(short_pool)
    return long_choice, short_choice


def _carve_gap(g, top, bot, start, length):
    g[top, start : start + length] = BG
    g[bot, start : start + length] = BG


def _apply_width_features(g, W, plans, bar_rows, real_indices, bar_start):
    candidates = []
    for i in real_indices:
        plan = plans.get(i, {"kind": "straight"})
        if plan["kind"] == "u_target":
            continue
        max_end = _segment_max_end(plan, W)
        if max_end is None:
            continue
        min_start = bar_start + 2
        if max_end < min_start:
            continue
        candidates.append({"i": i, "plan": plan, "min_start": min_start, "max_end": max_end})

    picked = _choose_gap_channels(candidates)
    if picked is None:
        return None
    long_choice, short_choice = picked
    features = []

    for label, choice, options in [
        ("long", long_choice, _LONG_GAP_OPTIONS),
        ("short", short_choice, _SHORT_GAP_OPTIONS),
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
        _carve_gap(g, top, bot, start, length)
        features.append(
            {
                "kind": label,
                "channel": choice["i"],
                "row": top + 1,
                "start": start,
                "length": length,
            }
        )

    remaining = [
        c
        for c in candidates
        if c["i"] not in {long_choice["i"], short_choice["i"]}
    ]
    random.shuffle(remaining)
    extra_added = 0
    for choice in remaining:
        if extra_added >= _MAX_EXTRA_LONG_GAPS:
            break
        if random.random() >= 0.4:
            continue
        feasible_lengths = [
            length
            for length in _LONG_GAP_OPTIONS
            if choice["max_end"] - choice["min_start"] + 1 >= length
        ]
        if not feasible_lengths:
            continue
        length = random.choice(feasible_lengths)
        start_max = choice["max_end"] - length + 1
        if start_max < choice["min_start"]:
            continue
        start = random.randint(choice["min_start"], start_max)
        top, bot = bar_rows[choice["i"]], bar_rows[choice["i"] + 1]
        _carve_gap(g, top, bot, start, length)
        features.append(
            {
                "kind": "long",
                "channel": choice["i"],
                "row": top + 1,
                "start": start,
                "length": length,
            }
        )
        extra_added += 1

    return features


def _verify_width_features(g, features):
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

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    features = _apply_width_features(g, W, plans, bar_rows, real_indices, bar_start)
    if not features:
        return None

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        out_unbounded = _main_from_trace(g, _trace_unbounded)
    except Exception:
        return None
    if not _verify_width_features(g, features):
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
