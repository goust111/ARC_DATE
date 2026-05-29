# Combo variant of 9bbf930d (index 28)
# combo_spec: D8/0+D16/0+D10/2
# Name: 断裂条对+间隙宽度限制+间距区分真假

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
# Design analysis for combo D8/0 + D16/0 + D10/2
#
# a) Concrete behavior trace
# - We keep the original "horizontal bars + magenta particles entering from the
#   left" visual world, but the geometry changes in three coupled ways.
# - Truth is no longer color-based. Real horizontal corridors are the rows that
#   sit between two adjacent bar rows (gap height 1). Fake horizontal corridors
#   are wider, with two BG rows between bars (gap height 2). Likewise, real
#   vertical turns are 1 column wide; fake shafts are 2 columns wide.
# - Bar rows are no longer solid. They are dashed, so exact top/bottom overlap
#   is often absent. A horizontal corridor cell is considered supported if each
#   adjacent bar row shows some colored cell within horizontal radius 1.
# - Straight motion now tracks how many consecutive cells lack such support for
#   the current direction. At most 2 unsupported cells may be crossed; the 3rd
#   is terminal.
# - Small example on a real row:
#     top bar:    1 . 1 . . . . 1
#     corridor:   . . . . . . . .
#     bottom bar: . 2 . 2 . . . 2
#   On columns near the left, the particle still sees a real corridor because
#   each side has a nearby dash within radius 1 even though colors differ and
#   exact overlap is missing. If both bars are erased for 4 columns, the inner
#   2 cells become unsupported and are still traversable; if erased for 5+, the
#   unsupported run exceeds 2 and the particle stops inside the break.
#
# b) Complexity calibration
# - Width only has two levels: real=1, fake=2. This is visually obvious and is
#   repeated across many lanes in every sample.
# - Dashed support only tolerates radius-1 local gaps. This keeps D8 visible
#   without introducing a continuum of possible support radii.
# - The generator forces both a short traversable break and a longer stopping
#   break, so the MAX_GAP_WIDTH=2 rule is inferable from 2-3 examples.
# - The generator also forces at least one real-but-different-color lane and at
#   least one fake-but-same-color lane, so color matching is visibly invalid.
#
# c) Implicit assumptions broken from the original main()
# - Original horizontal corridor recognition assumed exact same-color cells
#   above and below the channel. Broken by both dashed bars and width-based
#   truth, so the fused solver uses local visible support instead.
# - Original entrance filtering assumed the top and bottom bars begin at the
#   same exact column. Broken by alternating dash phase, so we only require
#   both bar rows to become visible near the left edge.
# - Original straight motion assumed arbitrary BG runs are safe. Broken by the
#   width limit, so tracing now counts consecutive unsupported cells.
# - Original truth depended on color equality everywhere. Broken by D10/2, so
#   colors become decorative for truth and only geometry matters.
#
# d) Placement legality
# - Lanes are built by alternating bar rows with corridor heights 1 or 2, so no
#   overlap or out-of-bounds placement is needed to encode real vs fake.
# - True turns reserve a 3-column window and record must_keep/must_bg cells per
#   bar row before dashed rendering, which prevents the dash pattern from
#   deleting required shaft flanks or turn gaps.
# - Fake shafts reserve a wider window and are placed after true shafts so they
#   cannot accidentally create an ambiguous second turn option.
#
# e) Visual observability
# - Humans can see three independent repeated cues:
#   1. narrow lanes move, wide lanes do not;
#   2. some real lanes have mismatched colors while some fake lanes have matched
#      colors, so color cannot be the criterion;
#   3. broken bars sometimes still count, but only up to a limited break size.
# - No rule depends on color ids, ordering, or any hidden metadata.
#
# f) Visual cleanliness
# - The scene remains a small set of repeated elements: bar rows, narrow/wide
#   lanes, a few 1-column true shafts, and a few 2-column fake shafts.
# - Dashed bars are regular instead of noisy; long breaks are localized and only
#   a few per grid, so the image stays readable.
#
# g) New assumptions the fused solver introduces
# - A real horizontal lane must have bar rows immediately above and below it.
# - A real vertical shaft must have exact colored flank cells left and right on
#   each carved bar-row gap.
# - Unsupported spans between successive supported shaft cells never exceed 2
#   rows, which the lane layout guarantees because corridor heights are only 1
#   or 2.
# - The generator explicitly validates against three ablations:
#   exact-support only (no D8), unbounded inertia (no D16), and color-based
#   truth (no D10).
# ---------------------------------------------------------------------------


def _is_colored(value: int) -> bool:
    return int(value) not in {BG, PARTICLE}


def _local_support_color(grid: np.ndarray, bar_row: int, c: int, radius: int = 1) -> int | None:
    if not (0 <= bar_row < grid.shape[0]):
        return None
    colors = {
        int(grid[bar_row, cc])
        for cc in range(max(0, c - radius), min(grid.shape[1] - 1, c + radius) + 1)
        if _is_colored(int(grid[bar_row, cc]))
    }
    if len(colors) == 1:
        return next(iter(colors))
    return None


def _has_local_bar_support(grid: np.ndarray, bar_row: int, c: int, radius: int = 1) -> bool:
    return _local_support_color(grid, bar_row, c, radius=radius) is not None


def _has_true_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Real corridor = width-1 geometry plus local support for dashed bars."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        return _has_local_bar_support(grid, r - 1, c) and _has_local_bar_support(grid, r + 1, c)
    if c == 0 or c == W - 1:
        return False
    return _is_colored(int(grid[r, c - 1])) and _is_colored(int(grid[r, c + 1]))


def _has_exact_width_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Ablation without D8: width-based truth but no dashed-gap tolerance."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        return _is_colored(int(grid[r - 1, c])) and _is_colored(int(grid[r + 1, c]))
    if c == 0 or c == W - 1:
        return False
    return _is_colored(int(grid[r, c - 1])) and _is_colored(int(grid[r, c + 1]))


def _has_color_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Ablation without D10: dashed support kept, but truth again depends on color."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a = _local_support_color(grid, r - 1, c)
        b = _local_support_color(grid, r + 1, c)
        return a == b and a is not None
    if c == 0 or c == W - 1:
        return False
    a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and _is_colored(a)


def _first_visible_bar_col(row: np.ndarray) -> int:
    for c, value in enumerate(row):
        if _is_colored(int(value)):
            return c
    return len(row)


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    corridor_fn,
    *,
    max_gap_width: int | None,
) -> tuple[int, int]:
    H, W = grid.shape
    direction = "right"
    unsupported_run = 0
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            if max_gap_width is None:
                r, c = nr, nc
                continue
            next_run = unsupported_run + 1
            if corridor_fn(grid, nr, nc, direction):
                next_run = 0
            if next_run > max_gap_width:
                break
            r, c = nr, nc
            unsupported_run = next_run
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and corridor_fn(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        unsupported_run = 0
    return r, c


def _main_from_trace(grid: np.ndarray, corridor_fn, *, max_gap_width: int | None) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if corridor_fn(g, r, c, "right"):
                entrance_col = c
                break
        if entrance_col is None:
            continue
        if _first_visible_bar_col(g[r - 1]) > 4 or _first_visible_bar_col(g[r + 1]) > 4:
            continue

        end_r, end_c = _trace(g, r, 0, corridor_fn, max_gap_width=max_gap_width)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


def _final_col_per_row(inp: np.ndarray, out: np.ndarray) -> list[int]:
    finals = []
    for r in range(inp.shape[0]):
        if int(inp[r, 0]) != PARTICLE:
            continue
        cols = np.flatnonzero(out[r] == PARTICLE)
        finals.append(int(cols[-1]) if len(cols) else -1)
    return finals


def _rowwise_output_diff(inp: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    return sum(x != y for x, y in zip(_final_col_per_row(inp, a), _final_col_per_row(inp, b)))


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through dashed narrow corridors with bounded gap crossing.

    Input structure:
    - col 0 is all particle, col 1 is bg.
    - Colored rows act as horizontal bars. Real horizontal lanes have height 1; fake
      lanes have height 2. Bars themselves are dashed rather than solid.

    Rule:
    - A particle starts on each row and tries to move right.
    - It may keep going through unsupported bg cells, but only for 2 consecutive cells.
    - When blocked by a colored cell, it may turn 90 degrees only into a 1-cell-wide
      vertical shaft whose left and right flanks are colored.

    Key insight:
    - Corridor truth is encoded by geometry, not by color matching. Dashed bars still
      count if each side is locally visible within radius 1, but overly long breaks do not.
    """
    return _main_from_trace(grid, _has_true_corridor, max_gap_width=MAX_GAP_WIDTH)


def _exact_support_main(grid: np.ndarray) -> np.ndarray:
    return _main_from_trace(grid, _has_exact_width_corridor, max_gap_width=MAX_GAP_WIDTH)


def _color_truth_main(grid: np.ndarray) -> np.ndarray:
    return _main_from_trace(grid, _has_color_corridor, max_gap_width=MAX_GAP_WIDTH)


def _unbounded_main(grid: np.ndarray) -> np.ndarray:
    return _main_from_trace(grid, _has_true_corridor, max_gap_width=None)


# ------------------------------ generator -----------------------------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_STRAIGHT_PROB = 0.30
_FAKE_SHAFT_RANGE = (1, 3)

_SHORT_BREAK_OPTIONS = [3, 4]
_LONG_BREAK_OPTIONS = [5, 6]

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 500


def _row_layout_solutions(H: int) -> list[tuple[int, int]]:
    solutions = []
    for n_true in range(_MIN_MOVED, _MAX_MOVED + 1):
        rem = (H - 1) - 2 * n_true
        if rem >= 0 and rem % 3 == 0:
            n_fake = rem // 3
            if n_fake >= 2:
                solutions.append((n_true, n_fake))
    return solutions


def _build_lanes(H: int) -> tuple[list[dict], list[int]]:
    n_true, n_fake = random.choice(_row_layout_solutions(H))
    widths = [1] * n_true + [2] * n_fake
    random.shuffle(widths)

    lanes = []
    bar_rows = [0]
    cur = 0
    for idx, width in enumerate(widths):
        top = cur
        bottom = cur + width + 1
        lane = {
            "index": idx,
            "width": width,
            "top": top,
            "bottom": bottom,
            "rows": list(range(top + 1, bottom)),
        }
        if width == 1:
            lane["row"] = top + 1
        lanes.append(lane)
        bar_rows.append(bottom)
        cur = bottom
    assert cur == H - 1
    return lanes, bar_rows


def _lane_relations(lanes: list[dict]) -> list[str]:
    relations = [random.choice(["same", "diff"]) for _ in lanes]
    true_indices = [lane["index"] for lane in lanes if lane["width"] == 1]
    fake_indices = [lane["index"] for lane in lanes if lane["width"] == 2]
    relations[random.choice(true_indices)] = "diff"
    relations[random.choice(fake_indices)] = "same"
    return relations


def _assign_bar_colors(bar_rows: list[int], lanes: list[dict], extras: list[int]) -> dict[int, int]:
    relations = _lane_relations(lanes)
    colors = {bar_rows[0]: random.choice(extras)}
    for lane, relation in zip(lanes, relations):
        top = lane["top"]
        bottom = lane["bottom"]
        if relation == "same":
            colors[bottom] = colors[top]
            continue
        choices = [c for c in extras if c != colors[top]]
        unseen = [c for c in choices if c not in colors.values()]
        colors[bottom] = random.choice(unseen or choices)
    return colors


def _allocate_true_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    candidates = []
    for c in range(bar_start + 4, W - 2):
        if all(col not in used_cols for col in range(c - 1, c + 2)):
            candidates.append(c)
    if not candidates:
        return None
    c = random.choice(candidates)
    used_cols.update(range(c - 1, c + 2))
    return c


def _allocate_fake_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    candidates = []
    for c in range(bar_start + 4, W - 3):
        if all(col not in used_cols for col in range(c - 1, c + 4)):
            candidates.append(c)
    if not candidates:
        return None
    c = random.choice(candidates)
    used_cols.update(range(c - 1, c + 4))
    return c


def _plan_true_lanes(lanes: list[dict], H: int, bar_start: int, W: int) -> tuple[dict[int, dict], set[int]]:
    plans: dict[int, dict] = {}
    used_cols: set[int] = set()
    for lane in lanes:
        if lane["width"] != 1:
            continue
        directions = []
        if lane["top"] > 0:
            directions.append("up")
        if lane["bottom"] < H - 1:
            directions.append("down")
        if not directions or random.random() < _STRAIGHT_PROB:
            plans[lane["index"]] = {"kind": "straight"}
            continue
        c = _allocate_true_col(bar_start, W, used_cols)
        if c is None:
            plans[lane["index"]] = {"kind": "straight"}
            continue
        plans[lane["index"]] = {"kind": "l_edge", "direction": random.choice(directions), "c": c}
    return plans, used_cols


def _register_true_gap(
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
    bar_rows: list[int],
    *,
    c: int,
    top_limit: int | None = None,
    bottom_limit: int | None = None,
) -> None:
    for br in bar_rows:
        if top_limit is not None and br >= top_limit:
            continue
        if bottom_limit is not None and br <= bottom_limit:
            continue
        must_bg[br].add(c)
        must_keep[br].update({c - 1, c + 1})


def _build_l_turn(
    lane: dict,
    bar_rows: list[int],
    bar_color: dict[int, int],
    plan: dict,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
    blockers: dict[tuple[int, int], int],
) -> None:
    top, bottom, row = lane["top"], lane["bottom"], lane["row"]
    c = plan["c"]
    direction = plan["direction"]

    turn_bar = top if direction == "up" else bottom
    opp_bar = bottom if direction == "up" else top
    blockers[(row, c + 1)] = bar_color[turn_bar]

    must_bg[turn_bar].add(c)
    must_keep[turn_bar].update({c - 1, c + 1})
    must_keep[opp_bar].add(c)

    if direction == "up":
        _register_true_gap(must_keep, must_bg, bar_rows, c=c, top_limit=top)
    else:
        _register_true_gap(must_keep, must_bg, bar_rows, c=c, bottom_limit=bottom)


def _segment_max_end(plan: dict, W: int) -> int:
    if plan["kind"] == "straight":
        return W - 3
    return plan["c"] - 2


def _render_bar_rows(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    W = g.shape[1]
    for idx, br in enumerate(bar_rows):
        phase = idx % 2
        color = bar_color[br]
        for c in range(bar_start, W):
            if c in must_bg[br]:
                g[br, c] = BG
            elif c in must_keep[br] or (c - bar_start) % 2 == phase:
                g[br, c] = color
            else:
                g[br, c] = BG


def _carve_break(g: np.ndarray, lane: dict, start: int, length: int) -> None:
    g[lane["top"], start : start + length] = BG
    g[lane["bottom"], start : start + length] = BG


def _choose_break_lanes(candidates: list[dict]) -> tuple[dict, dict] | None:
    long_pool = [c for c in candidates if c["max_end"] - c["min_start"] + 1 >= min(_LONG_BREAK_OPTIONS)]
    if not long_pool:
        return None
    long_choice = random.choice(long_pool)
    short_pool = [
        c for c in candidates
        if c["lane"]["index"] != long_choice["lane"]["index"]
        and c["max_end"] - c["min_start"] + 1 >= min(_SHORT_BREAK_OPTIONS)
    ]
    if not short_pool:
        return None
    short_choice = random.choice(short_pool)
    return long_choice, short_choice


def _apply_width_features(
    g: np.ndarray,
    lanes: list[dict],
    plans: dict[int, dict],
    bar_start: int,
    W: int,
) -> list[dict] | None:
    candidates = []
    for lane in lanes:
        if lane["width"] != 1:
            continue
        plan = plans[lane["index"]]
        min_start = bar_start + 2
        max_end = _segment_max_end(plan, W)
        if max_end < min_start:
            continue
        candidates.append({"lane": lane, "min_start": min_start, "max_end": max_end})

    picked = _choose_break_lanes(candidates)
    if picked is None:
        return None
    long_choice, short_choice = picked
    features = []

    for kind, choice, options in [
        ("long", long_choice, _LONG_BREAK_OPTIONS),
        ("short", short_choice, _SHORT_BREAK_OPTIONS),
    ]:
        feasible = [length for length in options if choice["max_end"] - choice["min_start"] + 1 >= length]
        if not feasible:
            return None
        length = random.choice(feasible)
        start_max = choice["max_end"] - length + 1
        if start_max < choice["min_start"]:
            return None
        start = random.randint(choice["min_start"], start_max)
        _carve_break(g, choice["lane"], start, length)
        features.append(
            {
                "kind": kind,
                "row": choice["lane"]["row"],
                "start": start,
                "length": length,
            }
        )
    return features


def _add_fake_shafts(
    g: np.ndarray,
    bar_rows: list[int],
    bar_start: int,
    W: int,
    used_cols: set[int],
) -> None:
    for _ in range(random.randint(*_FAKE_SHAFT_RANGE)):
        c = _allocate_fake_col(bar_start, W, used_cols)
        if c is None:
            return
        top_idx = random.randint(0, len(bar_rows) - 2)
        bottom_idx = random.randint(top_idx + 1, len(bar_rows) - 1)
        for br in bar_rows[top_idx : bottom_idx + 1]:
            g[br, c : c + 2] = BG


def _verify_width_features(g: np.ndarray, features: list[dict]) -> bool:
    saw_long = False
    saw_short = False
    for feat in features:
        row = feat["row"]
        new_end = _trace(g, row, 0, _has_true_corridor, max_gap_width=MAX_GAP_WIDTH)
        old_end = _trace(g, row, 0, _has_true_corridor, max_gap_width=None)
        if feat["kind"] == "short":
            if new_end != old_end:
                return False
            if new_end[1] <= feat["start"] + feat["length"] - 1:
                return False
            saw_short = True
        else:
            if new_end == old_end:
                return False
            if new_end[0] != row:
                return False
            if new_end[1] > feat["start"] + feat["length"] - 1:
                return False
            if old_end[1] <= feat["start"] + feat["length"] - 1:
                return False
            saw_long = True
    return saw_long and saw_short


def _dominant_bar_color(row: np.ndarray) -> int | None:
    vals = [int(v) for v in row if _is_colored(int(v))]
    if not vals:
        return None
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bar_rows = [r for r in range(H) if int(np.sum((g[r] != BG) & (g[r] != PARTICLE))) >= 2]

    true_lanes = 0
    fake_lanes = 0
    true_diff = 0
    fake_same = 0
    for top, bottom in zip(bar_rows, bar_rows[1:]):
        gap = bottom - top - 1
        top_color = _dominant_bar_color(g[top])
        bottom_color = _dominant_bar_color(g[bottom])
        if gap == 1:
            true_lanes += 1
            if top_color is not None and bottom_color is not None and top_color != bottom_color:
                true_diff += 1
        elif gap == 2:
            fake_lanes += 1
            if top_color is not None and bottom_color is not None and top_color == bottom_color:
                fake_same += 1

    out = main(g)
    exact_out = _exact_support_main(g)
    color_out = _color_truth_main(g)
    unbounded_out = _unbounded_main(g)
    moved = int(np.sum((g[:, 0] == PARTICLE) & (out[:, 0] != PARTICLE)))

    return {
        "true_lanes": true_lanes,
        "fake_lanes": fake_lanes,
        "true_diff": true_diff,
        "fake_same": fake_same,
        "moved": moved,
        "new_vs_exact_rows": _rowwise_output_diff(g, out, exact_out),
        "new_vs_color_rows": _rowwise_output_diff(g, out, color_out),
        "new_vs_unbounded_rows": _rowwise_output_diff(g, out, unbounded_out),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], _N_EXTRA_COLORS)

    lanes, bar_rows = _build_lanes(H)
    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_color = _assign_bar_colors(bar_rows, lanes, extras)
    if len(set(bar_color.values())) < _N_EXTRA_COLORS:
        return None

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    plans, used_cols = _plan_true_lanes(lanes, H, bar_start, W)
    must_keep = {br: set() for br in bar_rows}
    must_bg = {br: set() for br in bar_rows}
    blockers: dict[tuple[int, int], int] = {}

    for lane in lanes:
        if lane["width"] != 1:
            continue
        plan = plans[lane["index"]]
        if plan["kind"] == "l_edge":
            _build_l_turn(lane, bar_rows, bar_color, plan, must_keep, must_bg, blockers)

    _render_bar_rows(g, bar_rows, bar_color, bar_start, must_keep, must_bg)
    for (r, c), color in blockers.items():
        g[r, c] = color

    features = _apply_width_features(g, lanes, plans, bar_start, W)
    if not features:
        return None

    _add_fake_shafts(g, bar_rows, bar_start, W, used_cols)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    moved = int(np.sum((g[:, 0] == PARTICLE) & (out[:, 0] != PARTICLE)))
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None
    if not _verify_width_features(g, features):
        return None

    metrics = analyze_variant(g)
    if metrics["true_diff"] < 1 or metrics["fake_same"] < 1:
        return None
    if metrics["new_vs_exact_rows"] < 1:
        return None
    if metrics["new_vs_color_rows"] < 1:
        return None
    if metrics["new_vs_unbounded_rows"] < 1:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
