# Variant of 9bbf930d
# Dimension: D13 — 多步处理与级联 (multi_step)
# Change: 单步处理——所有粒子独立计算终点，互不影响，结果一次性写入输出。→ 两轮移动：
# first apply the original movement operator once, then treat the intermediate particle positions
# as fresh starts and apply the same right-start movement rule a second time.

from __future__ import annotations

import random

import numpy as np

# Design analysis
# ----------------
# Reference-task note:
# - The prompt asks to read solver code for source task 28a6681f first.
# - In this workspace I could find
#   /home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/28a6681f.json
#   plus mechanism notes tagging it as multi_step / iterative movement, but no local solver file
#   named 28a6681f.py under the allowed paths. I therefore use its visible iterative behavior
#   as the baseline: the output is obtained by repeated state updates, not a single closed-form move.
#
# a) Concretization and behavior trace
# - The input geometry stays the same as the original task: full-width colored bar rows on even
#   rows, bg channels on odd rows, magenta particles in column 0, fake channels from mismatched
#   bar pairs, and real channels from same-color bar pairs.
# - The D13 change is not a new object type; it is a new staging rule. The first round still uses
#   the original corridor-following trace. The second round starts from the new particle positions
#   produced by round 1 and launches the same right-start trace again.
# - Concrete cascade example:
#   1. Round 1: a particle starts at (r,0), goes right, turns down at column c_v, and the vertical
#      corridor is intentionally terminated at the bottom bar of a fake (mismatched) channel row j.
#   2. The particle therefore stops on j's odd row, at column c_v-1. Because the bars around that
#      row mismatch, vertical motion cannot legally turn left/right there, so round 1 really stops.
#   3. Original task output would stop here. Variant output runs round 2: from that stopped cell,
#      the particle starts moving right again and slides through bg toward the right edge.
# - The visible difference is therefore "particle ends farther along a second corridor segment than
#   the one-round solver would allow", without changing the bar artwork itself.
#
# b) Complexity calibration
# - The rule is still a small finite set: exactly two rounds, both using the same deterministic
#   movement operator. There is no unbounded iteration.
# - To make the difference observable, the generator explicitly creates 1-2 cascade sources whose
#   first-round endpoints land on fake rows with open bg to the right, so round 2 usually moves at
#   least one particle further.
# - I avoid adding many new subcases; the only new plan class is "cascade", a restricted L-turn
#   whose internal stop is aligned with a target fake row.
#
# c) Review of original main() assumptions that break under D13
# - Original `for r in range(1, H - 1)` + `g[r,0] == PARTICLE` assumes only left-edge odd-row
#   particles can ever be active. Under D13, round-2 starts are interior positions, so this must
#   become a scan over all particle cells.
# - Original entrance check is still needed for column-0 particles, because even-row bar particles
#   on the left edge are inert decoration and must not start moving. I keep the original entrance
#   logic only for `c == 0`.
# - Original `_trace()` always starts with direction "right". I keep that on purpose for both
#   rounds; otherwise round-1 endpoints would usually be dead ends and the second round would be
#   invisible. The second round therefore means "relaunch the same operator from the new position".
# - Original writeback `out[r,0] = BG; out[end_r,end_c] = PARTICLE` assumed one round only. Under
#   D13 I need a reusable one-round helper that clears all movable starts, then writes all
#   destinations, so a destination that is also a source cell is handled correctly.
#
# d) Placement legality
# - A cascade source uses the existing L-turn builder and only changes `stop_at` so the vertical
#   corridor stops exactly at the top or bottom bar of a fake (mismatched) channel row.
# - This preserves non-overlap guarantees: the same reserved vertical-column window logic is used,
#   and the stop row is one of the striped bar rows already aligned to the task grammar.
# - Fake stop rows are chosen because vertical motion can then stop there without the original
#   solver's "two perpendicular turns available" ambiguity; round 2 still has visible motion
#   because particles travel straight through bg even on fake rows.
#
# e) Visual observability
# - The new behavior is inferable from 2-4 ARC examples because particles finish deeper in the
#   network than the single-pass corridor trace predicts.
# - No hidden color-id ordering or invisible metadata is used. The second pass is revealed by path
#   geometry alone.
#
# f) Visual cleanliness
# - No extra markings are added. The grid remains a striped corridor drawing, and cascades reuse
#   existing rows/columns instead of overlaying labels.
# - Reserved vertical windows keep nearby turns separated, avoiding unreadable clutter.
#
# g) Color-change inferability
# - There is no new recoloring rule. Particle color stays magenta; bars keep their original colors.
#
# 9) New assumptions introduced by the new main()
# - Interior particle starts are assumed to be fresh round-1 particle positions and are therefore
#   always relaunched, even if they sit on plain bg or fake rows. The generator guarantees these
#   interior starts come only from legal round-1 endpoints.
# - Column-0 particles are still assumed to be active only if their entrance is a genuine real
#   channel with aligned bar starts <= 3. The generator preserves that original layout.
# - Multiple particles are assumed not to require collision tie-breaking beyond "destination cells
#   remain particle-colored". The generator keeps endpoints distinct enough that this does not
#   become ambiguous in accepted samples.

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
    """True if (r,c) is a bg/particle cell flanked by same-color bars perpendicular to direction."""
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
    """Round start test.

    Column-0 particles must satisfy the original entrance test.
    Interior particles are always relaunched in round 2 from their new positions.
    """
    if int(grid[r, c]) != PARTICLE:
        return False
    if c == 0:
        return _find_left_entrance(grid, r) is not None
    return True


def _trace(grid: np.ndarray, r: int, c: int, direction: str = "right") -> tuple[int, int]:
    """Trace one particle until it hits the edge or a blocker with no legal perpendicular turn."""
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
    return r, c


def _move_once(grid: np.ndarray, return_stats: bool = False):
    """Apply one parallel movement round using the current grid as the obstacle map."""
    g = np.array(grid, dtype=int)
    starts: list[tuple[int, int]] = []
    dests: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == PARTICLE):
        rr, cc = int(r), int(c)
        if not _is_active_particle(g, rr, cc):
            continue
        starts.append((rr, cc))
        dests.append(_trace(g, rr, cc, "right"))

    out = g.copy()
    for r, c in starts:
        out[r, c] = BG
    moved = 0
    for start, dest in zip(starts, dests):
        if dest != start:
            moved += 1
        out[dest] = PARTICLE

    if return_stats:
        return out, {"starts": starts, "dests": dests, "moved": moved}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through the corridor network for exactly two rounds.

    Round rule:
    - A start cell is active if it is a legitimate original left-edge entry, or if it is any
      interior particle produced by the previous round.
    - Each active particle launches to the right, moves through bg/particle cells, and turns 90
      degrees only when blocked by a colored cell and exactly one perpendicular corridor is valid.
    - All particles in the round move independently and write back simultaneously.

    Variant change:
    - Apply that round rule twice in sequence.
    """
    first = _move_once(np.array(grid, dtype=int))
    second = _move_once(first)
    return second


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.70
_U_INTERNAL_STOP_PROB = 0.80
_CASCADE_COUNT_OPTIONS = [1, 1, 2]

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_SECOND_ROUND_MOVED = 1
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
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


def _choose_cascade_target(i: int, candidates: list[int]) -> tuple[str, int] | None:
    """Pick a nearby target row above or below source i."""
    options: list[tuple[str, int, int]] = []
    for j in candidates:
        if j == i:
            continue
        if j < i:
            options.append(("up", j, i - j))
        else:
            options.append(("down", j, j - i))
    if not options:
        return None
    min_dist = min(dist for _, _, dist in options)
    near = [(direction, j) for direction, j, dist in options if dist <= min_dist + 1]
    return random.choice(near)


def _plan_cascades(
    real_indices: list[int],
    fake_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Create 1-2 sources whose round-1 endpoint lands on a fake row for round-2 relaunch."""
    if not real_indices or not fake_indices:
        return
    desired = min(random.choice(_CASCADE_COUNT_OPTIONS), len(real_indices), len(fake_indices))
    sources = real_indices[:]
    random.shuffle(sources)
    made = 0
    reserved_targets: set[int] = set()
    for i in sources:
        if made >= desired or i in plans or i in reserved_targets:
            continue
        target_candidates = [j for j in fake_indices if j not in reserved_targets]
        choice = _choose_cascade_target(i, target_candidates)
        if choice is None:
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            break
        direction, target = choice
        plans[i] = {"kind": "cascade", "direction": direction, "c_v": c_v, "target": target}
        reserved_targets.add(target)
        made += 1


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Pair same-color real channels into U-turn plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        if i in plans:
            continue
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            if i in plans or j in plans:
                continue
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
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Assign straight or simple edge-reaching L-turn plans to remaining real channels."""
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
        plans[i] = {"kind": "l_edge", "direction": direction, "c_v": c_v}


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind == "cascade":
            target = int(plan["target"])
            j_top, j_bot = bar_rows[target], bar_rows[target + 1]
            stop_at = j_top if str(plan["direction"]) == "up" else j_bot
            _build_l_turn(g, H, top, bot, row, color, c_v, str(plan["direction"]), bar_color, stop_at)
        elif kind in {"l_edge", "l_internal"}:
            stop_at = None
            if kind == "l_internal":
                if str(plan["direction"]) == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(
                g,
                H,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                bar_color,
                stop_at,
            )
        elif kind == "u":
            target = int(plan["target"])
            j_top, j_bot = bar_rows[target], bar_rows[target + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn and optionally terminate the vertical run at a chosen bar row."""
    g[row, c_v] = color

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
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    """Build a U-turn between source and target real channels."""
    g[row, c_v] = color

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

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _count_left_edge_departures(inp: np.ndarray, out: np.ndarray) -> int:
    """Count original active particles that left column 0."""
    H, _ = inp.shape
    moved = 0
    for r in range(1, H - 1):
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE:
            moved += 1
    return moved


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input whose output requires the second round."""
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
    fake_indices = [i for i, s in enumerate(channel_spec) if s == "fake"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_cascades(real_indices, fake_indices, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    cascade_targets = {int(plan["target"]) for plan in plans.values() if str(plan["kind"]) == "cascade"}
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        if i in cascade_targets:
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
        first_round, stats1 = _move_once(g, return_stats=True)
        second_round, stats2 = _move_once(first_round, return_stats=True)
        out = second_round
    except Exception:
        return None

    n_moved = _count_left_edge_departures(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if stats1["moved"] < _MIN_MOVED:
        return None
    if stats2["moved"] < _MIN_SECOND_ROUND_MOVED:
        return None
    if np.array_equal(first_round, second_round):
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
