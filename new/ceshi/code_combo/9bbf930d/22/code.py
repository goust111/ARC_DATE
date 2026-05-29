# Combo variant of 9bbf930d (index 22)
# combo_spec: D13/2+D6/2
# Name: 两轮移动+穿越固定数量走廊停止

from __future__ import annotations

import random
from typing import Any

import numpy as np

# Design analysis
# ---------------
# a) Concretization and behavior trace
# - Base geometry stays the same as 9bbf930d: even rows are colored bar rows, odd rows are
#   horizontal channel rows, column 0 is filled with particle color 6, and column 1 is bg 7.
# - The fused rule changes the movement operator itself. One round is no longer "trace until edge
#   or dead end"; one round is "launch rightward, but stop before entering corridor segment 3".
# - Corridor counting is local to each launch:
#   1. the horizontal segment entered from the start cell counts as corridor 1,
#   2. the first perpendicular turn enters corridor 2,
#   3. if another legal turn would enter corridor 3, the particle stops on the current cell.
# - After every active particle finishes this capped first round, the resulting interior particle
#   positions are treated as fresh starts and the same capped movement rule is applied once more.
# - Concrete fused example:
#   1. Round 1 starts at a left-edge real row, moves right, turns down into a vertical shaft.
#   2. At the bottom of that shaft there is a legal re-entry turn into another same-color
#      horizontal row, but this would be corridor 3, so the particle stops there because of D6/2.
#   3. Original task would continue into that target row. Plain D6/2 would stop there forever.
#   4. The fused task launches round 2 from that interior stop cell; the particle now moves right
#      along the target row and possibly takes one more capped turn later.
# - Visible difference:
#   compared with the source task, particles undershoot on round 1 and then continue again; compared
#   with plain D13/2, the intermediate stop positions are forced by a repeated corridor-limit motif.
#
# b) Complexity calibration
# - The corridor limit is fixed at N=2. Using N=1 would erase most turning structure; using larger
#   N would make the cap too hard to see in 2-4 examples.
# - The generator reuses a small menu of path types: two U-turn sources that reliably hit the round-1
#   corridor cap, plus 1-2 singleton L-turn sources. This keeps repeated evidence while avoiding a
#   cluttered combinatorial layout.
# - The second round is made visible by the U-turn construction itself: the capped round-1 endpoint
#   lands at the mouth of a target horizontal corridor, so re-launching to the right is visually
#   obvious in the output.
#
# c) Implicit-assumption audit for main()
# - Original code assumed only column-0 particles can be active. This breaks under two rounds, so
#   active-particle detection must accept interior particle cells as round-2 starts.
# - Original trace assumed every unique legal turn is always taken. This breaks under the corridor
#   cap, so the turn step must stop when `corridors_used >= 2` even if a legal turn exists.
# - Original entrance validation for column-0 particles still matters: even-row particles in the
#   left column are decorative and must remain inert. That check is preserved only for `c == 0`.
# - Parallel writeback is still required because all particles in a round move independently from the
#   same obstacle map. Clearing starts first and then writing destinations preserves that behavior.
#
# d) Placement legality
# - Vertical shafts still use a reserved 3-column window, so two turn structures do not overlap.
# - U-turn targets keep their left entrance broken so the left-edge particle on that row does not
#   start moving by itself; that row is only activated by a capped interior stop from round 1.
# - All accepted samples are validated by actually running the fused solver, so out-of-bounds or
#   ambiguous-turn constructions are rejected.
#
# e) Visual observability
# - A human can infer the rule from repeated evidence: several particles stop after two segments in
#   round 1, yet their final output positions lie farther right because a second launch occurs.
# - No hidden metadata is used. The only cues are corridor geometry, repeated interior stop points,
#   and the mismatch between one-round and two-round outcomes.
#
# f) Visual cleanliness
# - The grid remains the original striped corridor drawing. No labels, markers, or recoloring are
#   added.
# - Using only two U-turn source rows plus a small number of singleton L-turn rows avoids dense
#   overlaid shafts and keeps the target rows visually separable.
#
# g) Change inferability
# - There is no new color mapping. Particle color stays 6 and bars keep their original colors.
# - The only hidden variable a solver must infer is the fixed corridor cap N=2, and that cap is
#   repeated across multiple U-turn examples per grid.
#
# 9) New assumptions introduced by the fused main()
# - Any interior particle is assumed to be a legitimate previous-round endpoint and is relaunched
#   rightward. The generator guarantees interior particle starts only arise through round-1 traces.
# - The corridor count resets at the start of each round. The generator relies on this by placing
#   round-1 limit-hit endpoints at the entrance of a fresh horizontal segment.
# - Multiple particles are assumed not to need collision tie-breaking beyond standard ARC overwrite
#   semantics. The generator uses separated channel rows so destination collisions are rare.

BG = 7
PARTICLE = 6
MAX_CORRIDORS = 2

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


def _find_left_entrance(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid real-channel entrance column for row r, or None."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
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
    bar_color_above = int(grid[r - 1, entrance_col])
    bar_color_below = int(grid[r + 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == bar_color_above), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == bar_color_below), W)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _is_active_particle(grid: np.ndarray, r: int, c: int) -> bool:
    """Column-0 particles need a legal entrance; interior particles are round-2 starts."""
    if int(grid[r, c]) != PARTICLE:
        return False
    if c == 0:
        return _find_left_entrance(grid, r) is not None
    return True


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    max_corridors: int | None,
    return_stats: bool = False,
) -> Any:
    """Trace a right-launched particle, optionally capping how many corridors it may enter."""
    H, W = grid.shape
    direction = "right"
    corridors_used = 1
    turns_taken = 0
    stop_reason = "start"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            stop_reason = "edge"
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
            stop_reason = "blocked"
            break
        if max_corridors is not None and corridors_used >= max_corridors:
            stop_reason = "limit"
            break
        r, c, direction = options[0]
        corridors_used += 1
        turns_taken += 1

    if return_stats:
        return r, c, {
            "corridors_used": corridors_used,
            "turns_taken": turns_taken,
            "stop_reason": stop_reason,
        }
    return r, c


def _move_once(
    grid: np.ndarray,
    *,
    max_corridors: int | None,
    return_stats: bool = False,
):
    """Apply one simultaneous movement round on the current obstacle map."""
    g = np.array(grid, dtype=int)
    starts: list[tuple[int, int]] = []
    dests: list[tuple[int, int]] = []
    traces: list[dict[str, Any]] = []
    for r, c in np.argwhere(g == PARTICLE):
        rr, cc = int(r), int(c)
        if not _is_active_particle(g, rr, cc):
            continue
        end_r, end_c, meta = _trace(g, rr, cc, max_corridors=max_corridors, return_stats=True)
        start = (rr, cc)
        dest = (int(end_r), int(end_c))
        starts.append(start)
        dests.append(dest)
        traces.append({"start": start, "dest": dest, "meta": meta})

    out = g.copy()
    for r, c in starts:
        out[r, c] = BG
    moved = 0
    limit_hits = 0
    for trace in traces:
        if trace["dest"] != trace["start"]:
            moved += 1
        if trace["meta"]["stop_reason"] == "limit":
            limit_hits += 1
        out[trace["dest"]] = PARTICLE

    if return_stats:
        return out, {
            "starts": starts,
            "dests": dests,
            "traces": traces,
            "moved": moved,
            "limit_hits": limit_hits,
        }
    return out


def _analyze_combo_behavior(grid: np.ndarray) -> dict[str, Any]:
    """Return fused-variant diagnostics for one input grid."""
    limited_first, first_stats = _move_once(grid, max_corridors=MAX_CORRIDORS, return_stats=True)
    limited_second, second_stats = _move_once(
        limited_first, max_corridors=MAX_CORRIDORS, return_stats=True
    )
    unlimited_first, unlimited_first_stats = _move_once(
        grid, max_corridors=None, return_stats=True
    )

    second_by_start = {tuple(item["start"]): item for item in second_stats["traces"]}
    cascade_limit_movers = 0
    for item in first_stats["traces"]:
        if item["meta"]["stop_reason"] != "limit":
            continue
        follow = second_by_start.get(tuple(item["dest"]))
        if follow is not None and follow["dest"] != follow["start"]:
            cascade_limit_movers += 1

    round1_changed_vs_unlimited = sum(
        1
        for limited_item, unlimited_item in zip(
            first_stats["traces"], unlimited_first_stats["traces"]
        )
        if limited_item["dest"] != unlimited_item["dest"]
    )

    return {
        "first_round": limited_first,
        "second_round": limited_second,
        "first_stats": first_stats,
        "second_stats": second_stats,
        "unlimited_first": unlimited_first,
        "round1_changed_vs_unlimited": round1_changed_vs_unlimited,
        "cascade_limit_movers": cascade_limit_movers,
        "final_differs_from_first": not np.array_equal(limited_first, limited_second),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through the corridor network for exactly two capped rounds.

    Input structure:
    - bg=7, particle=6. Column 0 contains particles, column 1 is bg.
    - Even rows hold colored bars; odd rows are horizontal channels between bar pairs.

    Rule:
    - In each round, every active particle launches to the right.
    - It travels through bg/particle cells and may turn 90 degrees only when blocked by a colored
      cell and exactly one perpendicular same-color corridor is available.
    - Each launch may enter at most two corridor segments. If a legal turn would enter segment 3,
      the particle stops on its current cell instead of taking that turn.
    - After all particles finish round 1 simultaneously, the resulting particle positions become
      the starts for an identical round 2.

    Key insight:
    - The final output is not the one-round capped stop state. You must first stop particles after
      two corridor segments, then relaunch those new positions to the right and apply the same
      two-corridor cap again.
    """
    first = _move_once(np.array(grid, dtype=int), max_corridors=MAX_CORRIDORS)
    second = _move_once(first, max_corridors=MAX_CORRIDORS)
    return second


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_SINGLETON_REAL_OPTIONS = [1, 2]
_L_INTERNAL_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.0

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_LIMIT_HITS = 2
_MIN_CASCADE_LIMIT_MOVERS = 2
_MIN_SECOND_ROUND_MOVED = 2
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for one vertical corridor."""
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


def _pick_pair_starts(n_channels: int) -> list[int] | None:
    """Pick two length-2 real blocks with at least one fake gap between them."""
    starts = list(range(n_channels - 1))
    random.shuffle(starts)
    for first in starts:
        others = [s for s in starts if s >= first + 3 or s <= first - 3]
        random.shuffle(others)
        for second in others:
            if {first, first + 1}.isdisjoint({second, second + 1}):
                return sorted([first, second])
    return None


def _choose_real_layout(n_channels: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    """Choose two U-turn pairs plus 1-2 singleton real channels."""
    pair_starts = _pick_pair_starts(n_channels)
    if pair_starts is None:
        return None
    pairs = [(s, s + 1) for s in pair_starts]
    blocked = {idx for pair in pairs for idx in pair}
    reserved_for_singletons = set(blocked)
    for idx in blocked:
        if idx - 1 >= 0:
            reserved_for_singletons.add(idx - 1)
        if idx + 1 < n_channels:
            reserved_for_singletons.add(idx + 1)
    singleton_pool = [i for i in range(n_channels) if i not in reserved_for_singletons]
    if not singleton_pool:
        return None

    n_singletons = min(random.choice(_N_SINGLETON_REAL_OPTIONS), len(singleton_pool))
    if n_singletons == 0:
        return None
    singletons = random.sample(singleton_pool, n_singletons)
    return pairs, singletons


def _build_channel_spec(
    n_channels: int,
) -> tuple[list[str], dict[int, dict[str, Any]], list[list[int]]] | None:
    """Return channel roles and movement plans."""
    layout = _choose_real_layout(n_channels)
    if layout is None:
        return None
    pairs, singletons = layout

    channel_spec = ["fake"] * n_channels
    plans: dict[int, dict[str, Any]] = {}
    real_groups: list[list[int]] = []

    for upper, lower in pairs:
        channel_spec[upper] = "real"
        channel_spec[lower] = "real"
        real_groups.append([upper, lower])
        if random.random() < 0.5:
            plans[upper] = {"kind": "u", "target": lower, "direction": "down"}
            plans[lower] = {"kind": "u_target"}
        else:
            plans[lower] = {"kind": "u", "target": upper, "direction": "up"}
            plans[upper] = {"kind": "u_target"}

    for idx in singletons:
        channel_spec[idx] = "real"
        real_groups.append([idx])
        plans[idx] = {
            "kind": "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge",
            "direction": random.choice(["up", "down"]),
        }

    return channel_spec, plans, real_groups


def _build_bar_colors(
    n_channels: int,
    channel_spec: list[str],
    extras: list[int],
    real_groups: list[list[int]],
) -> list[int] | None:
    """Construct bar-row colors while keeping real structures color-separated."""
    if len(real_groups) > len(extras):
        return None

    required: dict[int, int] = {}
    group_colors = random.sample(extras, len(real_groups))
    for group, color in zip(real_groups, group_colors):
        start = min(group)
        end = max(group) + 1
        for row_idx in range(start, end + 1):
            prev = required.get(row_idx)
            if prev is not None and prev != color:
                return None
            required[row_idx] = color

    def _dfs(row_idx: int, acc: list[int]) -> list[int] | None:
        if row_idx == n_channels + 1:
            return acc.copy()

        candidates = [required[row_idx]] if row_idx in required else extras[:]
        random.shuffle(candidates)
        for color in candidates:
            if row_idx > 0:
                prev = acc[row_idx - 1]
                if channel_spec[row_idx - 1] == "real" and color != prev:
                    continue
                if channel_spec[row_idx - 1] != "real" and color == prev:
                    continue
            if row_idx < n_channels and row_idx + 1 in required:
                next_color = required[row_idx + 1]
                if channel_spec[row_idx] == "real" and next_color != color:
                    continue
                if channel_spec[row_idx] != "real" and next_color == color:
                    continue
            acc.append(color)
            result = _dfs(row_idx + 1, acc)
            if result is not None:
                return result
            acc.pop()
        return None

    return _dfs(0, [])


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    color: int,
    c_v: int,
    direction: str,
    color_by_row: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn, matching the original geometry."""
    row = top + 1
    g[row, c_v] = color

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = color_by_row[turn_bar]
    g[turn_bar, c_v] = color_by_row[turn_bar]
    g[opp_bar, c_v - 1] = color_by_row[opp_bar]

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
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    color_by_row: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    """Build a U-turn between two same-color real channels."""
    g[row, c_v] = color

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = color_by_row[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = color_by_row[top]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, Any]],
    bar_rows: list[int],
    bar_colors: list[int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    color_by_row = {row: bar_colors[i] for i, row in enumerate(bar_rows)}
    for i, plan in plans.items():
        kind = plan["kind"]
        if kind in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = color_by_row[top]
        c_v = plan["c_v"]

        if kind in ("l_edge", "l_internal"):
            stop_at = None
            if kind == "l_internal":
                if plan["direction"] == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, color, c_v, plan["direction"], color_by_row, stop_at)
        elif kind == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                color_by_row,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input whose output requires both combo components."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    channel_data = _build_channel_spec(n_channels)
    if channel_data is None:
        return None
    channel_spec, plans, real_groups = channel_data

    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_colors = _build_bar_colors(n_channels, channel_spec, extras, real_groups)
    if bar_colors is None or len(set(bar_colors)) < 3:
        return None

    for br, color in zip(bar_rows, bar_colors):
        g[br, bar_start:W] = color

    used_vcols: set[int] = set()
    for _, plan in plans.items():
        if plan["kind"] == "u_target":
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            return None
        plan["c_v"] = c_v
        if plan["kind"] == "u":
            plan["internal_stop"] = random.random() < _U_INTERNAL_STOP_PROB

    _execute_plans(g, H, W, plans, bar_rows, bar_colors, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_colors[i], bar_colors[i + 1]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        analysis = _analyze_combo_behavior(g)
        out = main(g)
    except Exception:
        return None

    first_stats = analysis["first_stats"]
    second_stats = analysis["second_stats"]
    moved = first_stats["moved"]
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None
    if first_stats["limit_hits"] < _MIN_LIMIT_HITS:
        return None
    if second_stats["moved"] < _MIN_SECOND_ROUND_MOVED:
        return None
    if analysis["cascade_limit_movers"] < _MIN_CASCADE_LIMIT_MOVERS:
        return None
    if analysis["round1_changed_vs_unlimited"] < _MIN_LIMIT_HITS:
        return None
    if not analysis["final_differs_from_first"]:
        return None
    if not np.array_equal(out, analysis["second_round"]):
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
