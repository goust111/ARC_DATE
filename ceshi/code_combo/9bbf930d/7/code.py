# Combo variant of 9bbf930d (index 7)
# combo_spec: D7/1+D4/3
# Name: 终点改色为走廊色+走廊色轨迹
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace point
# - The input geometry stays the same as the source task: column 0 is filled
#   with particles (6), even rows carry colored bars, real corridors are BG
#   cells flanked by same-color bar pairs, fake corridors use mismatched colors.
# - Original behavior: erase the start cell and place one magenta particle at
#   the final traced position; all intermediate cells stay unchanged.
# - Fused behavior: every visited path cell is painted with the active corridor
#   color, and the endpoint keeps that final corridor color instead of reverting
#   to magenta.
# - Small trace example:
#   a particle starts on a horizontal color-4 corridor, moves right leaving
#   color 4, hits a blocker, turns upward into a color-2 vertical corridor, and
#   stops near the top edge. The fused output shows a color-4 horizontal trail,
#   then a color-2 vertical trail, and the endpoint is also color 2. Relative to
#   the source task, both the path and the endpoint become informative.
#
# b) Complexity calibration
# - The movement rule itself is unchanged; only output rendering is enriched.
# - To avoid samples collapsing into "just D7" or "just D4", the generator
#   requires at least one moved path that both turns and changes trail color, and
#   also requires endpoint color diversity across moved particles.
# - The palette remains in the original narrow regime: four non-bg/non-particle
#   colors are reused for corridor bars, trails, and recolored endpoints.
#
# c) Hidden assumptions in the original main()
# - The source `_trace()` returned only coordinates. That breaks here because the
#   endpoint color depends on the final corridor segment and the trail depends on
#   every intermediate segment. The fused tracer therefore returns trail cells,
#   turn count, final direction, and endpoint color.
# - The source code could write directly to `out` during iteration because only
#   one terminal pixel changed. That is no longer safe: trail painting must read
#   movement decisions from the untouched input `g`, otherwise earlier painted
#   cells could affect later particles. The fused solver traces on `g` and only
#   then writes to `out`.
# - Some vertical-gap cells are traversable BG cells without locally exposing a
#   same-color pair in the movement direction. The new tracer therefore carries
#   forward an `active_color` instead of demanding a fresh local corridor match
#   on every step.
# - The source endpoint write assumed a constant particle color. In the fused
#   task the endpoint is overwritten with the final active corridor color.
#
# d) Placement legality
# - The generator preserves the original path-first corridor builders for
#   straight, L, and U paths, so objects stay in bounds and non-overlapping.
# - Vertical-corridor columns are still reserved with a 3-column exclusion
#   window, preventing ambiguous adjacent lanes.
# - Extra validation only rejects generated instances; it does not mutate an
#   accepted layout in ways that could violate the original solver invariants.
#
# e) Visual observability
# - A human can now see two linked cues: path segments inherit surrounding bar
#   colors, and the final marker does not stay magenta but matches the last
#   corridor segment. Because at least one accepted path is multicolor, examples
#   visibly demonstrate that different corridor segments contribute different
#   trail colors.
# - Requiring at least two distinct endpoint colors prevents the recolor rule
#   from looking like a fixed replacement of magenta by one arbitrary color.
#
# f) Visual cleanliness
# - Trails reuse existing corridor colors instead of introducing a new overlay
#   color, so each painted segment remains visually anchored to nearby bars.
# - The endpoint is not an extra icon; it is simply the last trail cell, kept in
#   the same color as the final corridor. This avoids a cluttered "trail plus
#   unrelated marker" look.
#
# g) Color-change inferability
# - The recolor depends only on visible local structure: which matched bar pair
#   currently defines the corridor segment. No ordering by numeric color value,
#   hidden ranking, or off-grid metadata is used.
# - The generator guarantees repeated support for the rule by enforcing at least
#   one multicolor turning path per sample and multiple endpoint colors overall.
#
# New solver assumptions introduced by the fused variant and guaranteed by the
# generator:
# - Every moved particle reaches at least one real corridor cell before any
#   painting occurs; `_find_entrance_col()` enforces the original entrance gate.
# - Ambiguous turns remain forbidden; the original builders still ensure at most
#   one perpendicular real-corridor option exists at each blocker.
# - The final active trail color is the intended endpoint color, even when the
#   stopping cell is an open gap whose local perpendicular pair is not explicit.

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


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Return the same-color flank pair for this corridor cell and travel direction."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in {PARTICLE, BG}:
        return a
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    return _corridor_color(grid, r, c, direction) is not None


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Find the first source-visible real horizontal corridor for row r."""
    _H, W = grid.shape
    for c in range(2, W):
        color = _corridor_color(grid, r, c, "right")
        if color is None:
            continue
        bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == color), W)
        bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            return c
    return None


def _endpoint_color(
    grid: np.ndarray, r: int, c: int, direction: str, active_color: int | None
) -> int:
    """Resolve the final corridor color, preferring the carried segment color."""
    directional = _corridor_color(grid, r, c, direction)
    if directional is not None:
        return int(directional)
    if active_color is not None:
        return int(active_color)

    candidates = []
    horiz = _corridor_color(grid, r, c, "right")
    vert = _corridor_color(grid, r, c, "down")
    if horiz is not None:
        candidates.append(int(horiz))
    if vert is not None:
        candidates.append(int(vert))
    assert candidates and len(set(candidates)) == 1, "final position is not on a unique real corridor"
    return int(candidates[0])


def _trace_path(
    grid: np.ndarray, start_r: int, start_c: int
) -> tuple[int, int, str, list[tuple[int, int, int]], int, int]:
    """Trace one particle, recording a corridor-colored trail and final color."""
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

        if int(grid[nr, nc]) in {PARTICLE, BG}:
            step_color = _corridor_color(grid, nr, nc, direction)
            if step_color is not None:
                active_color = int(step_color)
            if active_color is not None:
                trail.append((nr, nc, active_color))
            r, c = nr, nc
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            turn_color = _corridor_color(grid, tr, tc, turn_dir)
            if turn_color is not None:
                options.append((tr, tc, turn_dir, int(turn_color)))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break

        r, c, direction, active_color = options[0]
        trail.append((r, c, active_color))
        turns += 1

    end_color = _endpoint_color(grid, r, c, direction, active_color)
    return r, c, direction, trail, turns, end_color


def _collect_particle_paths(grid: np.ndarray) -> list[dict]:
    """Collect path metadata for all solver-visible particles in column 0."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    paths = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c, end_dir, trail, turns, end_color = _trace_path(g, r, 0)
        moved = (end_r, end_c) != (r, 0)
        colors = [color for _, _, color in trail]
        color_transitions = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i - 1])
        paths.append(
            {
                "start": (r, 0),
                "end": (end_r, end_c),
                "end_dir": end_dir,
                "trail": trail,
                "turns": turns,
                "moved": moved,
                "endpoint_color": end_color,
                "distinct_colors": len(set(colors)),
                "color_transitions": color_transitions,
            }
        )
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each particle's route with corridor colors and recolor the endpoint to the final corridor color.

    Input structure:
    - bg=7 and particles=6 fill column 0; real horizontal corridors are BG cells between same-color bars.
    - Vertical connections are carved from the same corridor colors, while fake channels use mismatched bar colors.

    Rule:
    - Trace each eligible particle exactly as in the source task: move straight through open cells, and when blocked,
      turn 90 degrees only into the unique perpendicular real corridor if one exists.
    - Every moved-to path cell is painted with the currently active corridor color, carrying that color through open
      gap cells until a new corridor segment explicitly changes it.
    - The start cell becomes BG, and the endpoint keeps the final active corridor color instead of reverting to magenta.

    Key insight:
    - The output is determined by segment ownership, not just by the final coordinate: the same corridor-color state
      that paints the last trail segment also determines the endpoint color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    for path in _collect_particle_paths(g):
        if not path["moved"]:
            continue
        start_r, start_c = path["start"]
        end_r, end_c = path["end"]
        out[start_r, start_c] = BG
        for tr, tc, color in path["trail"]:
            out[tr, tc] = color
        out[end_r, end_c] = int(path["endpoint_color"])

    return out


# --------------- generator (sa_max_effort) ---------------

# === CFG: tunable generation parameters ===

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
_MIN_TURNING_PATHS = 1
_MIN_MULTICOLOR_PATHS = 1
_MIN_COLOR_TRANSITIONS = 2
_MIN_TRAIL_CELLS = 18
_MIN_ENDPOINT_COLOR_DIVERSITY = 2
_MIN_COMBO_VISIBLE_PATHS = 1
_MAX_RETRIES = 300


def _allocate_vcol(bar_start, W, used_vcols):
    """Reserve a 3-column window for one vertical corridor wall column."""
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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans):
    """Pair same-color real channels into U-turn plans."""
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
    """Assign straight or L-turn plans to real channels not already planned."""
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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the fused behavior is clearly visible on this input."""
    paths = [p for p in _collect_particle_paths(grid) if p["moved"]]
    combo_visible = 0
    for p in paths:
        trail = p["trail"]
        if not trail:
            continue
        if p["turns"] > 0 and p["distinct_colors"] > 1 and p["endpoint_color"] == trail[-1][2]:
            combo_visible += 1

    return {
        "moved": len(paths),
        "turning": sum(1 for p in paths if p["turns"] > 0),
        "multicolor": sum(1 for p in paths if p["distinct_colors"] > 1),
        "trail_cells": sum(len(p["trail"]) for p in paths),
        "color_transitions": sum(p["color_transitions"] for p in paths),
        "endpoint_color_diversity": len({int(p["endpoint_color"]) for p in paths}),
        "combo_visible": combo_visible,
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused colored-trail + recolored-endpoint variant."""
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
    except Exception:
        return None

    metrics = _variant_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["turning"] < _MIN_TURNING_PATHS:
        return None
    if metrics["multicolor"] < _MIN_MULTICOLOR_PATHS:
        return None
    if metrics["color_transitions"] < _MIN_COLOR_TRANSITIONS:
        return None
    if metrics["trail_cells"] < _MIN_TRAIL_CELLS:
        return None
    if metrics["endpoint_color_diversity"] < _MIN_ENDPOINT_COLOR_DIVERSITY:
        return None
    if metrics["combo_visible"] < _MIN_COMBO_VISIBLE_PATHS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if np.array_equal(out, g):
        return None
    return g


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras):
    """Render all planned turns onto the grid."""
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
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
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
                bar_color,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at):
    """Build an L-turn at channel (top, bot) with a blocking wall at (row, c_v)."""
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
    g, W, top, bot, row, color, c_v, direction, j_top, j_bot, bar_color, bar_start, extras, internal_stop
):
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot)."""
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


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
