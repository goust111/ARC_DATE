# Variant of 9bbf930d
# Dimension: D4 — 轨迹涂色 (trajectory)
# Change: 无轨迹——粒子移动后只在终点出现，路径上不留任何痕迹。... → 走廊色轨迹
#
# Design analysis
# - Reference-task check: the prompt names 16de56c4 as the inspiration for
#   "color_pairing". I searched the workspace for a solver file for 16de56c4
#   before coding; none exists here, only the ARC evaluation JSON. Per prompt,
#   the missing solver means that code-reading step is skipped and documented.
# - Concrete behavior: the input grid is unchanged structurally. Particles still
#   start in column 0 and move through same-color corridors. The output now keeps
#   a painted trail: every visited travel cell is recolored with the color of the
#   corridor segment currently carrying the particle, and the final cell is then
#   overwritten back to particle color 6.
# - Small execution trace: in the original task, a particle on a real horizontal
#   channel might go right, turn upward at a blocker, then stop at the top edge;
#   only the final cell becomes 6. In this variant, the horizontal segment is
#   painted with its channel-pair color, the upward segment keeps the most recent
#   corridor color through open gap cells, and updates color again whenever the
#   local flanking pair changes at a new corridor row. The endpoint still ends as
#   6, so the difference is visible all along the route instead of only at the
#   terminal cell.
# - Complexity calibration: I keep the original movement rule and corridor test.
#   The only state added to tracing is "active trail color". This avoids adding
#   new branching logic while still making turns visually meaningful. Generator
#   validation rejects samples where all moved paths stay visually close to the
#   original no-trail task.
# - Implicit assumption review for main():
#   1. Original _trace() assumed the output grid could ignore intermediate cells.
#      That is no longer safe, so tracing is separated from painting: all motion
#      decisions read from the untouched input copy `g`, while `out` is written
#      afterward.
#   2. Original straight-step branch only needed positions. Here it must also
#      update/persist corridor color. A cell may be open without locally exposing
#      flanking bars (for example an odd-row cell inside a vertical run), so the
#      code cannot require a fresh corridor match every step; it carries forward
#      the last known corridor color until a new explicit corridor color appears.
#   3. Original `out[r,0] = BG; out[end] = PARTICLE` was harmless because no
#      other writes touched the route. In the variant, this would erase trail
#      state if done in-place during tracing, hence painting happens after the
#      complete path is known.
# - Placement legality: the generator keeps the original geometry builders for
#   straight/L/U plans, so corridors remain non-overlapping, in-bounds, and
#   solver-compatible. Additional validation only filters generated inputs; it
#   does not mutate geometry in risky ways.
# - Visual observability: each accepted sample must include at least one turning
#   path and at least one moved path whose trail uses more than one color. This
#   makes D4 visible across 2-4 examples instead of degenerating to "same as
#   original, plus a short monochrome smear".
# - Visual cleanliness: trail colors reuse existing corridor-bar colors, so every
#   painted path is legible as belonging to nearby bar pairs. No new marker color
#   or extra overlay layer is introduced.
# - Color inference: the color choice is structural, not numeric. A human can see
#   that a trail segment takes the color of the bars that flank the traversed
#   corridor. The generator limits palette size to the original regime and forces
#   repeated support for the effect.
# - New solver assumptions introduced by the variant and guaranteed by generator:
#   1. Paths traced from column 0 do not need to cross each other in accepted
#      samples; the original generator already avoids overlapping vertical lanes.
#   2. A moved path has at least one genuine corridor cell before any color is
#      painted; `_find_entrance_col()` preserves the original entrance gate.
#   3. Ambiguous turns remain forbidden; the geometry builders still enforce the
#      original "at most one perpendicular option" invariant.

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
    """Return the flanking-pair color for a valid corridor cell, else None."""
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
    """Original entrance filter: first real horizontal corridor starting at col <= 3."""
    _, W = grid.shape
    for c in range(2, W):
        color = _corridor_color(grid, r, c, "right")
        if color is None:
            continue
        bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == color), W)
        bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            return c
    return None


def _trace_path(
    grid: np.ndarray, start_r: int, start_c: int
) -> tuple[int, int, list[tuple[int, int, int]], int]:
    """Trace one particle and record a colored trail.

    The trail stores moved-to cells only, never the starting cell. The active
    color is refreshed when the current cell explicitly exposes a corridor pair
    for the current direction; otherwise the previous active color is carried
    through open gap cells inside the same segment.
    """
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
                active_color = step_color
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
                options.append((tr, tc, turn_dir, turn_color))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break

        r, c, direction, active_color = options[0]
        trail.append((r, c, active_color))
        turns += 1

    return r, c, trail, turns


def _collect_particle_paths(grid: np.ndarray) -> list[dict]:
    """Collect solver-visible path metadata for all particles in column 0."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    paths = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = _find_entrance_col(g, r)
        if entrance_col is None:
            continue
        end_r, end_c, trail, turns = _trace_path(g, r, 0)
        colors = [color for _, _, color in trail]
        color_transitions = sum(
            1 for i in range(1, len(colors)) if colors[i] != colors[i - 1]
        )
        paths.append(
            {
                "start": (r, 0),
                "end": (end_r, end_c),
                "trail": trail,
                "turns": turns,
                "moved": (end_r, end_c) != (r, 0),
                "distinct_colors": len(set(colors)),
                "color_transitions": color_transitions,
            }
        )
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles as before, but preserve a corridor-colored trail in output."""
    g = np.array(grid, dtype=int)
    out = g.copy()

    for path in _collect_particle_paths(g):
        start_r, start_c = path["start"]
        end_r, end_c = path["end"]
        if not path["moved"]:
            continue
        out[start_r, start_c] = BG
        for tr, tc, color in path["trail"]:
            out[tr, tc] = color
        out[end_r, end_c] = PARTICLE

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
_MAX_RETRIES = 300


def _allocate_vcol(bar_start, W, used_vcols):
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
    """Assign straight or L-turn plans to real channels not yet planned."""
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


def _variant_metrics(grid: np.ndarray) -> dict:
    """Quantify whether the D4 behavior is visibly present on this input."""
    paths = [p for p in _collect_particle_paths(grid) if p["moved"]]
    return {
        "moved": len(paths),
        "turning": sum(1 for p in paths if p["turns"] > 0),
        "multicolor": sum(1 for p in paths if p["distinct_colors"] > 1),
        "trail_cells": sum(len(p["trail"]) for p in paths),
        "color_transitions": sum(p["color_transitions"] for p in paths),
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for task 9bbf930d with colored-trail outputs."""
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
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
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
