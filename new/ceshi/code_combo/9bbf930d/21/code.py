# Combo variant of 9bbf930d (index 21)
# combo_spec: D9/1+D20/1
# Name: 交叉口强制转弯+Z形走廊

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

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The base task uses striped geometry: even rows are colored bars, odd rows are
#      horizontal travel rows, and particles start in column 0 on odd rows.
#    - D9 changes a pass-through crossing into a mandatory branch point. Concretely,
#      if a particle is about to enter a BG crossing cell and that crossing exposes
#      exactly one perpendicular same-color corridor, the particle must turn into it
#      immediately instead of moving straight across the crossing.
#    - D20 changes the 2-turn topology from a U-return to a Z-offset. In this combo,
#      the Z is encoded as two open crossings:
#        1. source row moves right to a BG junction and is forced into a vertical shaft;
#        2. the vertical shaft reaches a target row at another BG junction and is forced
#           right into the offset horizontal continuation.
#      Example: on a lower real row, the old solver would pass through the first open
#      crossing and continue right forever. The combo solver instead turns up at that
#      crossing, then turns right again at the upper crossing, producing a zig-zag
#      `right -> up -> right` trace.
#
# b) Complexity calibration
#    - To keep the visual rule clear, the generator only uses straight rows plus
#      forced-Z rows. It does not mix in old U-turn blockers or extra branch styles.
#    - Z paths are required in every generated grid, while the remaining real rows stay
#      straight. This gives humans contrast and repeated evidence without introducing
#      too many topologies at once.
#
# c) Implicit assumption audit of main()
#    - `_has_corridor()` still assumes legal turns are recognized locally by matching
#      same-color flanks around the candidate turn cell. The generator preserves that.
#    - The original solver only checked turns after hitting a colored blocker. That
#      assumption no longer holds. `_forced_turn_ahead()` now checks the next BG cell
#      before each move and applies to both horizontal and vertical motion, because the
#      second corner of a Z path is also an open crossing.
#    - `_trace_path()` still asserts that at most one perpendicular option exists. The
#      generator guarantees this by making every forced junction one-sided and by only
#      letting shafts cross fake rows or their intended target row.
#    - `_find_entrance_col()` assumes genuine source rows start at column 2 or 3 with
#      aligned same-color bars. Target rows of Z paths violate source-ness, so the
#      generator deliberately breaks their left entrance alignment while keeping the
#      right continuation intact.
#
# d) Placement legality
#    - Every shaft reserves a 3-column window so separate Z structures do not merge.
#    - A forced Z pair is only allowed when every channel between source and target is
#      fake, so the vertical shaft never crosses another active real row and therefore
#      cannot create accidental mandatory turns for other particles.
#    - The target row is broken only near the left entrance and on the left side of the
#      target junction; this preserves the intended rightward continuation.
#
# e) Visual observability
#    - The distinguishing visual cue is structural: a horizontal corridor meets an open
#      one-sided crossing and the particle relocates onto a parallel row to the right of
#      the shaft. Across a few examples, the zig-zag offset is visible without relying
#      on hidden color ordering.
#
# f) Visual cleanliness
#    - The image keeps the original striped grammar. Fake rows remain mismatched-color
#      distractors, and only a small number of shafts are introduced. Because shafts are
#      separated in both row intervals and columns, intersections stay readable.
#
# g) Color inference
#    - Colors are only used for local same-color pairing. No global color remapping is
#      required. Repeated same-color bar pairs on source and target rows visually cue
#      that those two horizontal segments belong to one Z corridor.

# Generator configuration
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 1]
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 500
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_Z_PATHS = 1


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
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


def _forced_turn_ahead(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[int, int, str] | None:
    """Return a forced turn through the next open crossing, if any."""
    H, W = grid.shape
    dr, dc = _DELTAS[direction]
    cr, cc = r + dr, c + dc
    if not (0 <= cr < H and 0 <= cc < W):
        return None
    if int(grid[cr, cc]) not in {PARTICLE, BG}:
        return None

    options = []
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = cr + tdr, cc + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))
    assert len(options) <= 1, "ambiguous forced junction: both perpendicular turns valid"
    return options[0] if options else None


def _trace_path(
    grid: np.ndarray, r: int, c: int, use_forced: bool = True
) -> tuple[int, int, list[tuple[str, str]], int]:
    """Trace a particle and record turn topology plus forced-junction count."""
    H, W = grid.shape
    direction = "right"
    turns: list[tuple[str, str]] = []
    forced_turns = 0

    for _ in range(500):
        if use_forced:
            forced = _forced_turn_ahead(grid, r, c, direction)
            if forced is not None:
                next_r, next_c, next_dir = forced
                turns.append((direction, next_dir))
                forced_turns += 1
                r, c, direction = next_r, next_c, next_dir
                continue

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
        assert len(options) <= 1, "ambiguous blocked turn: both perpendicular turns valid"
        if not options:
            break
        next_r, next_c, next_dir = options[0]
        turns.append((direction, next_dir))
        r, c, direction = next_r, next_c, next_dir

    return r, c, turns, forced_turns


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original base-task tracing, used only for degeneration checks."""
    end_r, end_c, _, _ = _trace_path(grid, r, c, use_forced=False)
    return end_r, end_c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid same-color source corridor cell on row r."""
    H, W = grid.shape
    if r <= 0 or r >= H - 1 or int(grid[r, 0]) != PARTICLE:
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


def _classify_turns(turns: list[tuple[str, str]]) -> str:
    """Classify turn topology for generator-side validation."""
    if not turns:
        return "straight"
    if turns in ([("right", "up")], [("right", "down")]):
        return "l"
    if turns in (
        [("right", "up"), ("up", "right")],
        [("right", "down"), ("down", "right")],
    ):
        return "z"
    if turns in (
        [("right", "up"), ("up", "left")],
        [("right", "down"), ("down", "left")],
    ):
        return "u"
    return f"{len(turns)}-turn"


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Summarize the visible combo behavior present in an input."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    metrics = {
        "moved": 0,
        "straight": 0,
        "l": 0,
        "z": 0,
        "u": 0,
        "other": 0,
        "forced_l": 0,
        "forced_z": 0,
        "forced_junctions": 0,
        "divergent_vs_base": 0,
    }
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns, forced_turns = _trace_path(g, r, 0, use_forced=True)
        if (end_r, end_c) == (r, 0):
            continue
        metrics["moved"] += 1
        kind = _classify_turns(turns)
        if kind in metrics:
            metrics[kind] += 1
        else:
            metrics["other"] += 1
        if kind == "l" and forced_turns == 1:
            metrics["forced_l"] += 1
        if kind == "z" and forced_turns == 2:
            metrics["forced_z"] += 1
        metrics["forced_junctions"] += forced_turns
        if (end_r, end_c) != _trace_original(g, r, 0):
            metrics["divergent_vs_base"] += 1
    return metrics


def _solve_with_trace(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, _, _ = _trace_path(g, r, 0, use_forced=True)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through forced-junction corridors with Z offsets.

    Input structure:
    - bg=7 and particle=6.
    - Even rows are colored bars; odd rows are horizontal travel rows.
    - Real corridors are BG cells between same-color bar pairs; fake rows use mismatched bars.

    Rule:
    - A particle normally advances through BG cells.
    - If the next BG cell is an open crossing with exactly one perpendicular corridor,
      the particle must turn into that branch immediately instead of going straight.
    - If the next cell is colored, the particle falls back to the original 90-degree
      blocked-turn rule. The particle stops at the grid edge or when no turn is legal.

    Key insight:
    - Z corridors are encoded as two one-sided open crossings, so the particle zig-zags
      `right -> up/down -> right` by mandatory turns rather than by hitting blockers.
    """
    return _solve_with_trace(grid)


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for one vertical shaft."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1) if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _preferred_break_side(target: int, direction: str, channel_spec: list[str]) -> str:
    """Break the left entrance on the safer side of the target row."""
    n_channels = len(channel_spec)
    preferred = "top" if direction == "up" else "bottom"
    top_safe = target == 0 or channel_spec[target - 1] == "fake"
    bottom_safe = target == n_channels - 1 or channel_spec[target + 1] == "fake"
    if preferred == "top":
        if top_safe:
            return "top"
        if bottom_safe:
            return "bottom"
    else:
        if bottom_safe:
            return "bottom"
        if top_safe:
            return "top"
    return preferred


def _plan_forced_z_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    channel_spec: list[str],
) -> dict[int, dict[str, object]] | None:
    """Build one or more disjoint forced-Z plans."""
    candidates: list[tuple[int, int, str, str]] = []
    for upper_pos, upper in enumerate(real_indices):
        for lower in real_indices[upper_pos + 1 :]:
            if lower - upper < 2:
                continue
            if any(channel_spec[k] == "real" for k in range(upper + 1, lower)):
                continue
            if bar_color[bar_rows[upper]] != bar_color[bar_rows[lower]]:
                continue
            if random.random() < 0.5:
                direction = "up"
                source, target = lower, upper
            else:
                direction = "down"
                source, target = upper, lower
            break_side = _preferred_break_side(target, direction, channel_spec)
            candidates.append((source, target, direction, break_side))

    if not candidates:
        return None

    random.shuffle(candidates)
    plans: dict[int, dict[str, object]] = {}
    reserved_rows: set[int] = set()
    built = 0
    for source, target, direction, break_side in candidates:
        if source in reserved_rows or target in reserved_rows:
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            break
        plans[source] = {
            "kind": "forced_z",
            "target": target,
            "direction": direction,
            "c_v": c_v,
            "break_side": break_side,
            "internal_stop": random.random() < 0.50,
        }
        plans[target] = {"kind": "z_target"}
        reserved_rows.update({source, target})
        built += 1

    if built < _MIN_Z_PATHS:
        return None
    return plans


def _plan_remaining(real_indices: list[int], plans: dict[int, dict[str, object]]) -> None:
    """Give every unpaired real row a simple visible role."""
    for i in real_indices:
        if i not in plans:
            plans[i] = {"kind": "straight"}


def _break_target_entrance(
    g: np.ndarray,
    j_top: int,
    j_bot: int,
    bar_start: int,
    color: int,
    extras: list[int],
    break_side: str,
) -> None:
    other_colors = [c for c in extras if c != color]
    if not other_colors:
        return
    if break_side == "top":
        g[j_top, bar_start] = random.choice(other_colors)
    else:
        g[j_bot, bar_start] = random.choice(other_colors)


def _build_forced_z_turn(
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
    bar_start: int,
    extras: list[int],
    break_side: str,
    internal_stop: bool,
    wall_cols: set[int],
) -> None:
    """Build an open-crossing Z: right -> up/down -> right."""
    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v] = BG
        g[j_top, c_v - 1] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v] = BG
        g[j_bot, c_v - 1] = BG

    _break_target_entrance(g, j_top, j_bot, bar_start, color, extras, break_side)

    if internal_stop:
        j_row = j_top + 1
        candidates = [c for c in range(c_v + 3, W) if c not in wall_cols]
        if candidates:
            g[j_row, random.choice(candidates)] = color


def _add_fake_decor(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    channel_spec: list[str],
    bar_start: int,
) -> None:
    """Add sparse distractor blockers inside fake rows."""
    _, W = g.shape
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        usable = [c for c in range(bar_start + 1, W) if int(g[r, c]) == BG]
        if not usable:
            continue
        for c_decor in random.sample(usable, min(n_decor, len(usable))):
            g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the forced-turn Z-corridor combo."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])
    if len(set(bar_color.values())) != _N_EXTRA_COLORS:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = sorted(real_set)
    used_vcols: set[int] = set()
    plans = _plan_forced_z_turns(
        real_indices, bar_rows, bar_color, bar_start, W, used_vcols, channel_spec
    )
    if plans is None:
        return None
    _plan_remaining(real_indices, plans)

    wall_cols = {int(plan["c_v"]) for plan in plans.values() if "c_v" in plan}
    for i, plan in plans.items():
        if plan["kind"] != "forced_z":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        target = int(plan["target"])
        j_top, j_bot = bar_rows[target], bar_rows[target + 1]
        _build_forced_z_turn(
            g,
            W,
            top,
            bot,
            row,
            color,
            int(plan["c_v"]),
            str(plan["direction"]),
            j_top,
            j_bot,
            bar_start,
            extras,
            str(plan["break_side"]),
            bool(plan["internal_stop"]),
            wall_cols,
        )

    _add_fake_decor(g, bar_rows, bar_color, channel_spec, bar_start)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _behavior_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["forced_z"] < _MIN_Z_PATHS:
        return None
    if metrics["z"] != metrics["forced_z"] or metrics["u"] != 0 or metrics["other"] != 0:
        return None
    if metrics["divergent_vs_base"] < metrics["forced_z"]:
        return None
    if len(set(map(int, g.flatten().tolist()))) != 6:
        return None

    moved_again = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if moved_again != metrics["moved"]:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
