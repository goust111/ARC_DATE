# Combo variant of 9bbf930d (index 3)
# combo_spec: D4/3+D5/3
# Name: 走廊色轨迹+多色粒子匹配走廊
#
# Design analysis
# - Concrete behavior:
#   The striped corridor geometry stays the same as the base task: even rows
#   are colored bar rows, odd rows are channel rows, and turns are created by
#   carving vertical same-color connectors. The fusion changes two linked parts.
#   First, col 0 sources are no longer one fixed magenta color: each odd-row
#   source has a corridor-palette color, and it may move only through corridors
#   whose two flanking bars have exactly that color. Second, every visited path
#   cell is painted in the output with the current corridor color. Because the
#   admissible corridor color must equal the particle color, the same visible
#   color now controls both "may enter" and "what trail is drawn".
# - Small trace:
#   In the base task, a source on row 5 entered the first real corridor on that
#   row regardless of source color and only the final cell changed. Here, if the
#   source at (5, 0) is red and the first real corridor is blue, the row does
#   not move at all. If another row has a red source and red corridor bars, that
#   source moves right, may turn inside red vertical connectors, and every
#   visited channel cell becomes red in the output. The difference is therefore
#   visible both at the left border (who moves) and along the route (painted
#   trail instead of endpoint-only relocation).
# - Complexity calibration:
#   I keep the original geometric family: straight channels, L-turns, and
#   same-color U-turns. The fusion is made observable by generator constraints
#   rather than extra mechanics: accepted samples must include 3-6 movers, at
#   least one turning mover, at least two distinct moving source colors, at
#   least one geometrically valid real row whose source color mismatches, and a
#   minimum amount of painted trail. This avoids degenerating into either the
#   plain D5 grouping task (if trails are too short) or the plain D4 trail task
#   (if all sources happen to match).
# - Implicit-assumption review of the original main():
#   1. `PARTICLE = 6` no longer identifies every source. The solver must treat
#      any non-background odd-row col-0 color as a source and carry that color
#      through tracing and final painting.
#   2. Original corridor tests only required "same non-bg bars". That is no
#      longer enough; every entrance and turn must also match the current source
#      color, or a red source could enter a blue corridor.
#   3. Original tracing only needed coordinates. Trail painting requires keeping
#      an active corridor color so vertical gap cells remain paintable even when
#      the local flanks are temporarily absent at a crossing/gap.
#   4. Original output writes (`out[start]=BG`, `out[end]=PARTICLE`) assumed no
#      intermediate path state. Here all movement decisions read from the input
#      copy, then the finished trail is painted onto the output copy afterward.
# - Placement legality:
#   The generator reuses the proven base-task bar constructors. Particle colors
#   are assigned only after corridor geometry is fixed, so source coloring cannot
#   create overlaps or break turn sealing. Vertical corridor columns are still
#   reserved with a 3-column exclusion window, keeping turns non-overlapping.
# - Visual observability and cleanliness:
#   The leftmost colored sources are isolated from the maze by background col 1,
#   so humans can compare source color with nearby corridor-bar color directly.
#   Trails reuse existing corridor colors instead of adding a new marker layer,
#   which keeps outputs clean. Because multiple moving colors are required and
#   some real rows are deliberately mismatched, 2-4 examples are enough to infer
#   both halves of the rule: match to move, and matched movement paints the
#   traversed channel.
# - New solver assumptions introduced here and guaranteed by the generator:
#   1. Only odd rows host sources; even rows keep col 0 as background.
#   2. Every moving path starts with a genuine same-color horizontal entrance
#      whose first bar position is at col 2 or 3.
#   3. Real multi-turn paths keep one consistent corridor color along the whole
#      route, so color-matched turning remains unambiguous.

from __future__ import annotations

import random

import numpy as np

BG = 7
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _corridor_color(
    grid: np.ndarray, r: int, c: int, direction: str, particle_color: int
) -> int | None:
    """Return the corridor color for a legal cell, else None."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    if int(grid[r, c]) != BG:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b == particle_color:
        return particle_color
    return None


def _find_entrance_col(grid: np.ndarray, r: int, particle_color: int) -> int | None:
    """Find the first legal same-color horizontal corridor entrance."""
    _, W = grid.shape
    for c in range(2, W):
        color = _corridor_color(grid, r, c, "right", particle_color)
        if color is None:
            continue
        bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == color), W)
        bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            return c
    return None


def _trace_path(
    grid: np.ndarray, start_r: int, start_c: int, particle_color: int
) -> tuple[int, int, list[tuple[int, int, int]], int]:
    """Trace one colored particle and record its painted trail."""
    H, W = grid.shape
    r, c = start_r, start_c
    direction = "right"
    active_color: int | None = None
    trail: list[tuple[int, int, int]] = []
    turns = 0

    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) == BG:
            step_color = _corridor_color(grid, nr, nc, direction, particle_color)
            if step_color is not None:
                active_color = step_color
            if active_color is not None:
                trail.append((nr, nc, active_color))
            r, c = nr, nc
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            turn_color = _corridor_color(grid, tr, tc, turn_dir, particle_color)
            if turn_color is not None:
                options.append((tr, tc, turn_dir, turn_color))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break

        r, c, direction, active_color = options[0]
        trail.append((r, c, active_color))
        turns += 1

    return r, c, trail, turns


def _collect_particle_paths(grid: np.ndarray) -> list[dict]:
    """Collect path metadata for all odd-row colored sources."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    paths: list[dict] = []
    for r in range(1, H - 1, 2):
        particle_color = int(g[r, 0])
        if particle_color == BG:
            continue
        entrance_col = _find_entrance_col(g, r, particle_color)
        if entrance_col is None:
            paths.append(
                {
                    "start": (r, 0),
                    "end": (r, 0),
                    "particle_color": particle_color,
                    "trail": [],
                    "turns": 0,
                    "moved": False,
                }
            )
            continue
        end_r, end_c, trail, turns = _trace_path(g, r, 0, particle_color)
        paths.append(
            {
                "start": (r, 0),
                "end": (end_r, end_c),
                "particle_color": particle_color,
                "trail": trail,
                "turns": turns,
                "moved": (end_r, end_c) != (r, 0),
            }
        )
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """Move color-grouped particles and paint their matching-corridor trails.

    Input structure:
    - bg=7 fills open cells. Odd rows of col 0 contain colored source particles;
      col 1 stays background. Even rows carry corridor bars.
    - Real corridors are same-color bar pairs around a 1-cell path; fake rows
      may contain mismatched bars or decorative blockers.

    Rule:
    - A source can enter or turn only into corridors whose flanking bar color
      exactly matches the source color.
    - Every visited travel cell is painted with the current corridor color.
    - Movement stops at the grid edge or when no matching perpendicular turn is
      available after a blockage.

    Key insight:
    - The same visible color now has two jobs: it authorizes movement into a
      corridor and it becomes the trail color, so matched rows both move and
      draw while mismatched rows stay unchanged.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    for path in _collect_particle_paths(g):
        if not path["moved"]:
            continue
        start_r, start_c = path["start"]
        end_r, end_c = path["end"]
        particle_color = path["particle_color"]
        out[start_r, start_c] = BG
        for tr, tc, color in path["trail"]:
            out[tr, tc] = color
        out[end_r, end_c] = particle_color

    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 5

# Channel assignment
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_MISMATCHED_REAL_SOURCES = 1
_MIN_MOVED_COLOR_GROUPS = 2
_MIN_TURNING_PATHS = 1
_MIN_TRAIL_CELLS = 18
_ASSIGNMENT_ATTEMPTS = 40
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, width: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall col."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = width - _VCOL_RIGHT_PAD
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


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    width: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Pair same-color real channels into U-turn plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, width, used_vcols)
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
    width: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Assign straight or L-turn plans to real channels not yet planned."""
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, width, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _build_l_turn(
    g: np.ndarray,
    height: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn while preserving the original sealing rule."""
    color = bar_color[top]
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
        for r_br in range(bot + 2, height, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    width: int,
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
    """Build a same-color U-turn between two real channels."""
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
        c_stop_min, c_stop_max = c_v + 3, width - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(
    g: np.ndarray,
    height: int,
    width: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    candidates = list(range(2, top, 2))
                else:
                    candidates = list(range(bot + 2, height - 1, 2))
                stop_at = random.choice(candidates) if candidates else None
            _build_l_turn(g, height, top, bot, row, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                width,
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


def _choose_alt_color(colors: list[int], forbidden: int) -> int | None:
    """Choose a source color different from the corridor color on that row."""
    options = [c for c in colors if c != forbidden]
    return random.choice(options) if options else None


def _candidate_source_rows(
    g: np.ndarray, bar_rows: list[int]
) -> list[tuple[int, int, int]]:
    """Rows whose fixed geometry exposes a legal same-color left entrance."""
    height, width = g.shape
    candidates: list[tuple[int, int, int]] = []
    for i, top in enumerate(bar_rows[:-1]):
        row = top + 1
        if row <= 0 or row >= height - 1:
            continue
        corridor_color = None
        entrance_col = None
        for c in range(2, width):
            if int(g[row, c]) != BG:
                continue
            above, below = int(g[row - 1, c]), int(g[row + 1, c])
            if above == below and above != BG:
                corridor_color = above
                entrance_col = c
                break
        if entrance_col is None or corridor_color is None:
            continue
        bar_above = next((c for c in range(width) if int(g[row - 1, c]) == corridor_color), width)
        bar_below = next((c for c in range(width) if int(g[row + 1, c]) == corridor_color), width)
        if bar_above == bar_below and bar_above <= 3:
            candidates.append((i, row, corridor_color))
    return candidates


def _assign_particle_colors(
    g: np.ndarray,
    channel_spec: list[str],
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    plans: dict[int, dict[str, int | str | bool]],
) -> tuple[dict[int, int], dict[str, int]] | tuple[None, None]:
    """Assign source colors so some real rows match and some mismatch."""
    real_colors = sorted({bar_color[bar_rows[i]] for i in real_indices})
    if len(real_colors) < _MIN_MOVED_COLOR_GROUPS:
        return None, None

    all_candidates = _candidate_source_rows(g, bar_rows)
    real_candidates = [entry for entry in all_candidates if channel_spec[entry[0]] == "real"]
    if len(real_candidates) < _MIN_MOVED + _MIN_MISMATCHED_REAL_SOURCES:
        return None, None
    if len({corridor_color for _, _, corridor_color in real_candidates}) < _MIN_MOVED_COLOR_GROUPS:
        return None, None

    candidate_indices = [i for i, _, _ in real_candidates]
    candidate_color = {i: corridor_color for i, _, corridor_color in real_candidates}
    turning_candidates = {
        i
        for i in candidate_indices
        if plans.get(i, {"kind": "straight"})["kind"] not in ("straight", "u_target")
    }
    if not turning_candidates:
        return None, None
    max_match = min(_MAX_MOVED, len(candidate_indices) - _MIN_MISMATCHED_REAL_SOURCES)
    if max_match < _MIN_MOVED:
        return None, None

    for _ in range(_ASSIGNMENT_ATTEMPTS):
        n_match = random.randint(_MIN_MOVED, max_match)
        matched = set(random.sample(candidate_indices, n_match))
        matched_colors = {candidate_color[i] for i in matched}
        if not matched.intersection(turning_candidates):
            continue
        if len(matched_colors) < _MIN_MOVED_COLOR_GROUPS:
            continue

        particle_color: dict[int, int] = {}
        for i in range(len(channel_spec)):
            corridor = bar_color[bar_rows[i]]
            if i in matched:
                particle_color[i] = corridor
                continue
            alt = _choose_alt_color(real_colors, corridor)
            if alt is None:
                particle_color = {}
                break
            particle_color[i] = alt
        if not particle_color:
            continue

        mismatched_real = sum(1 for i in candidate_indices if particle_color[i] != candidate_color[i])
        if mismatched_real < _MIN_MISMATCHED_REAL_SOURCES:
            continue

        stats = {
            "candidate_real": len(candidate_indices),
            "matched_real": len(matched),
            "mismatched_real": mismatched_real,
            "matched_real_colors": len(matched_colors),
        }
        return particle_color, stats

    return None, None


def _paint_sources(g: np.ndarray, bar_rows: list[int], particle_color: dict[int, int]) -> None:
    """Place colored particles in col 0 and preserve bg in col 1."""
    g[:, 0] = BG
    for i, top in enumerate(bar_rows[:-1]):
        g[top + 1, 0] = particle_color[i]
    g[:, 1] = BG


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Measure whether the fused behavior is visibly present."""
    moved_paths = [path for path in _collect_particle_paths(grid) if path["moved"]]
    return {
        "moved": len(moved_paths),
        "turning": sum(1 for path in moved_paths if path["turns"] > 0),
        "trail_cells": sum(len(path["trail"]) for path in moved_paths),
        "moved_colors": len({path["particle_color"] for path in moved_paths}),
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused D4/3 + D5/3 variant."""
    height = random.choice(_HEIGHT_OPTIONS)
    width = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c != BG], _N_EXTRA_COLORS)

    g = np.full((height, width), BG, dtype=int)
    bar_rows = list(range(0, height, 2))
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
        g[br, bar_start:width] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, width, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, width, used_vcols, plans)
    _execute_plans(g, height, width, plans, bar_rows, bar_color, bar_start, extras)

    particle_color, stats = _assign_particle_colors(
        g, channel_spec, real_indices, bar_rows, bar_color, plans
    )
    if particle_color is None or stats is None:
        return None

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, width), min(n_decor, width - bar_start - 1))
        for c_decor in cols:
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    _paint_sources(g, bar_rows, particle_color)

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _variant_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["moved"] != stats["matched_real"]:
        return None
    if stats["mismatched_real"] < _MIN_MISMATCHED_REAL_SOURCES:
        return None
    if metrics["turning"] < _MIN_TURNING_PATHS:
        return None
    if metrics["moved_colors"] < _MIN_MOVED_COLOR_GROUPS:
        return None
    if metrics["trail_cells"] < _MIN_TRAIL_CELLS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
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
