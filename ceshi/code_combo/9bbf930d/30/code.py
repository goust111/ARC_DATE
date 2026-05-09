# Combo variant of 9bbf930d (index 30)
# combo_spec: D7/1+D17/1+D4/3
# Name: 唯一走廊色+终点改色+走廊色轨迹
#
# Design analysis
# ---------------
# a) Concrete behavior and trace point
# - The source task has particles in column 0 on odd rows. They move right through
#   horizontal corridors, turn only when the next cell is colored, and enter a
#   perpendicular corridor if that turn cell is a BG cell flanked by a same-color pair.
# - This combo changes three linked pieces at once:
#   1. Each real corridor segment uses a unique color.
#   2. Every visited travel cell is painted with the current corridor color.
#   3. The endpoint is not magenta anymore; it keeps the final corridor color.
# - Concrete trace example:
#   a row-13 particle moves right inside corridor color A, hits a blocker, turns up
#   into vertical corridor color B, then turns right again into row-7 corridor color C.
#   Original output would erase the start and place a magenta endpoint only.
#   The combo output erases the start, paints the first segment A, the second segment B,
#   the third segment C, and the endpoint stays C.
# - The visible difference is not just "recolor the last cell": because corridor colors
#   are unique, the trail color sequence itself reveals the ordered corridor list.
#
# b) Complexity calibration
# - Simply recoloring endpoints would underuse D17 and D4. Simply painting trails while
#   reusing corridor colors would make the path-to-corridor mapping ambiguous.
# - The generator therefore constructs three moved particles with mixed path shapes:
#   straight, L-turn, and one two-turn path. This gives repeated support for
#   the rule while keeping the geometry small enough to stay legible.
# - The palette is expanded to the full eight non-BG/non-particle colors, and each sample
#   draws a unique-color subset from that pool for its real corridor segments.
#
# c) Hidden assumptions in the original main()
# - Original `_trace()` returned only a final coordinate because the output ignored the
#   route. That assumption breaks under D4/3, so tracing now records the full path.
# - Original `out[end] = PARTICLE` assumes endpoint color is fixed and independent of the
#   final corridor. That is false under D7/1, so the final write must use the last active
#   corridor color instead.
# - Original generation paired same-color channels to build U-turns. D17/1 forbids that:
#   neighboring real segments may no longer share a color. The generator is therefore
#   changed from "pair equal-color rows" to explicit path construction with unique colors.
# - The new tracer carries `active_color` across open crossing gaps. This is necessary
#   because vertical pass-through gaps can temporarily hide local same-color flanks even
#   though the particle is still inside the same conceptual segment.
#
# d) Placement legality
# - Active source rows are spaced four rows apart so their primary horizontal corridors do
#   not require the same bar row to hold two different colors over the same span.
# - Vertical lanes are allocated with a reserved 3-column window, preventing adjacent walls
#   from collapsing into ambiguous multi-lane turns.
# - The two-turn target corridor starts only to the right of both neighboring turn columns,
#   so shared bar rows are split into non-overlapping colored spans rather than conflicting.
# - Validation reruns the solver on every generated input; any ambiguity or malformed path
#   rejects the sample immediately.
#
# e) Visual observability
# - Across 2-4 examples a human can see:
#   1. starts disappear,
#   2. routes are painted,
#   3. route colors change at turns,
#   4. endpoints match the last painted segment rather than magenta.
# - Because every real corridor color is unique, the multicolor path is interpretable:
#   "this particle visited these exact corridors in this exact order".
# - The generator enforces at least one three-color path and at least three distinct endpoint
#   colors so the fusion does not collapse into a near-original dataset.
#
# f) Visual cleanliness
# - The output uses only existing corridor colors. No extra marker layer is introduced.
# - The generator limits the number of moved particles to three and uses fixed row spacing,
#   which avoids a tangled overlay of many trails in the same region.
# - Crossings remain clean because pass-through gaps are kept at one cell wide.
#
# g) Color-change inferability
# - The color change is structural, not numeric: trail color always comes from the visibly
#   flanking corridor pair, and the final cell keeps the last such color.
# - Unique corridor colors reduce ambiguity rather than increasing it: when a trail changes
#   from A to B to C, a solver can infer that the particle changed corridors in that order.
# - New solver assumptions guaranteed by the generator:
#   1. Every moved path encounters a real entrance with first bar occurrence at column 2 or 3.
#   2. At every blocker, at most one perpendicular corridor is valid.
#   3. Every moved path paints at least one trail cell, so endpoint color can be read from
#      the last trail segment without consulting a mutated output grid.

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
_ALL_REAL_COLORS = [0, 1, 2, 3, 4, 5, 8, 9]


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
    """Find the first solver-visible real-corridor entrance for row r."""
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
) -> tuple[int, int, str, list[tuple[int, int, int]], int]:
    """Trace one particle and record its corridor-colored trail."""
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

    return r, c, direction, trail, turns


def _collect_particle_paths(grid: np.ndarray) -> list[dict]:
    """Collect path metadata for all solver-visible particles."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    paths = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, end_dir, trail, turns = _trace_path(g, r, 0)
        if trail:
            final_color = int(trail[-1][2])
        else:
            final_color = _corridor_color(g, end_r, end_c, end_dir)
        colors = [color for _, _, color in trail]
        color_transitions = sum(
            1 for i in range(1, len(colors)) if colors[i] != colors[i - 1]
        )
        paths.append(
            {
                "start": (r, 0),
                "end": (end_r, end_c),
                "final_direction": end_dir,
                "trail": trail,
                "turns": turns,
                "moved": (end_r, end_c) != (r, 0),
                "final_color": final_color,
                "distinct_colors": len(set(colors)),
                "color_transitions": color_transitions,
            }
        )
    return paths


def _baseline_output(grid: np.ndarray) -> np.ndarray:
    """Original-task style output: erase start, keep only a magenta endpoint."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for path in _collect_particle_paths(g):
        if not path["moved"]:
            continue
        start_r, start_c = path["start"]
        end_r, end_c = path["end"]
        out[start_r, start_c] = BG
        out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move particles through uniquely colored corridors and leave a corridor-colored trail.

    Input structure:
    - bg=7 and particle=6. Odd rows may host moving particles from column 0.
    - Real corridors are BG travel cells flanked by same-color bar pairs; each real corridor
      segment uses a distinct color. Some rows contain only distractors or unreachable segments.

    Rule:
    - A particle moves straight through open cells.
    - When its next cell is colored, it turns 90 degrees into the unique perpendicular corridor
      whose turn cell is BG and flanked by a same-color pair.
    - Every visited path cell is recolored to the currently occupied corridor color, and the
      endpoint remains that final corridor color rather than magenta.

    Key insight:
    - The output is a color-coded route log: because corridor colors are unique, the painted
      trail and same-colored endpoint together identify the exact ordered corridor sequence.
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
        assert path["final_color"] is not None, "moved path must end with a corridor color"
        out[end_r, end_c] = int(path["final_color"])

    return out


# --------------- generator (combo-specific) ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_MAX_RETRIES = 400

_N_REAL_SEGMENT_COLORS = 6
_N_MOVED = 3
_MIN_TURNING_PATHS = 2
_MIN_MULTICOLOR_PATHS = 2
_MIN_THREE_COLOR_PATHS = 1
_MIN_COLOR_TRANSITIONS = 3
_MIN_TRAIL_CELLS = 22
_MIN_ENDPOINT_DIVERSITY = 3
_MIN_DIFF_FROM_BASELINE = 15


def _paint_h_corridor(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        return
    grid[row - 1, c0 : c1 + 1] = color
    grid[row + 1, c0 : c1 + 1] = color


def _paint_v_corridor(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, col] = BG
    grid[lo : hi + 1, col - 1] = color
    grid[lo : hi + 1, col + 1] = color


def _allocate_vcols(bar_start: int, W: int, count: int) -> list[int] | None:
    """Allocate `count` vertical corridor columns with a 3-column exclusion window."""
    candidates = list(range(bar_start + 3, W - 4))
    if not candidates:
        return None
    for _ in range(200):
        random.shuffle(candidates)
        picked: list[int] = []
        used: set[int] = set()
        for c in candidates:
            if any((c + d) in used for d in (-1, 0, 1)):
                continue
            picked.append(c)
            used.update({c - 1, c, c + 1})
            if len(picked) == count:
                return sorted(picked)
    return None


def _odd_rows(H: int) -> list[int]:
    return list(range(1, H - 1, 2))


def _pick_layout_rows(H: int) -> tuple[int, int, int, int]:
    """Return (target_row, two_turn_row, straight_row, l_down_row)."""
    odds = _odd_rows(H)
    required = [1, 5, 9, 13]
    assert all(r in odds for r in required), "grid too short for combo layout"
    return 1, 5, 9, 13


def _build_combo_layout(g: np.ndarray, bar_start: int, colors: list[int]) -> dict:
    """Construct the fixed combo-supporting set of real corridors."""
    H, W = g.shape
    target_row, two_turn_row, straight_row, l_down_row = _pick_layout_rows(H)
    vcols = _allocate_vcols(bar_start, W, 2)
    if vcols is None:
        raise ValueError("failed to allocate vertical lanes")
    v_two, v_down = vcols

    (
        c_two_h1,
        c_two_v,
        c_two_h2,
        c_straight,
        c_l_down_h,
        c_l_down_v,
    ) = colors

    # L-turn downward path.
    _paint_h_corridor(g, l_down_row, bar_start, v_down, c_l_down_h)
    g[l_down_row, v_down + 1] = c_l_down_h
    _paint_v_corridor(g, v_down, l_down_row + 1, H - 1, c_l_down_v)

    # Two-turn path: right on row `two_turn_row`, up to `target_row`, then right again.
    _paint_h_corridor(g, two_turn_row, bar_start, v_two, c_two_h1)
    g[two_turn_row, v_two + 1] = c_two_h1
    _paint_v_corridor(g, v_two, target_row + 1, two_turn_row - 1, c_two_v)
    g[target_row, v_two] = BG
    _paint_h_corridor(g, target_row, v_two + 1, W - 1, c_two_h2)
    g[target_row - 1, v_two] = c_two_h2

    # Straight path stays isolated from all vertical walls.
    _paint_h_corridor(g, straight_row, bar_start, W - 1, c_straight)

    # Preserve the required left border.
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    # Add a small amount of harmless decoration on unused rows so examples still look like
    # puzzle instances rather than three isolated training traces.
    active_like = {straight_row, l_down_row, two_turn_row, target_row}
    used_colors = list(colors)
    for r in _odd_rows(H):
        if r in active_like:
            continue
        n = random.randint(0, 2)
        for _ in range(n):
            c = random.randint(bar_start + 1, W - 2)
            if g[r, c] != BG:
                continue
            g[r, c] = random.choice(used_colors)

    return {
        "rows": {
            "target": target_row,
            "two_turn": two_turn_row,
            "straight": straight_row,
            "l_down": l_down_row,
        },
        "vcols": {"two_turn": v_two, "l_down": v_down},
        "real_colors": set(colors),
    }


def _variant_metrics(grid: np.ndarray) -> dict:
    """Quantify whether the combo behavior is present and visible."""
    paths = [p for p in _collect_particle_paths(grid) if p["moved"]]
    out = main(grid)
    base = _baseline_output(grid)
    endpoint_colors = [int(p["final_color"]) for p in paths]
    trail_colors = {color for p in paths for _, _, color in p["trail"]}
    return {
        "moved": len(paths),
        "turning": sum(1 for p in paths if p["turns"] > 0),
        "multicolor": sum(1 for p in paths if p["distinct_colors"] > 1),
        "three_color": sum(1 for p in paths if p["distinct_colors"] >= 3),
        "color_transitions": sum(p["color_transitions"] for p in paths),
        "trail_cells": sum(len(p["trail"]) for p in paths),
        "endpoint_diversity": len(set(endpoint_colors)),
        "trail_color_union": len(trail_colors),
        "non_particle_endpoints": sum(1 for color in endpoint_colors if color != PARTICLE),
        "diff_from_baseline": int(np.sum(out != base)),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    colors = random.sample(_ALL_REAL_COLORS, k=_N_REAL_SEGMENT_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        layout = _build_combo_layout(g, bar_start, colors)
        out = main(g)
    except Exception:
        return None

    paths = [p for p in _collect_particle_paths(g) if p["moved"]]
    metrics = _variant_metrics(g)

    if len(layout["real_colors"]) != _N_REAL_SEGMENT_COLORS:
        return None
    if len(paths) != _N_MOVED:
        return None
    if metrics["turning"] < _MIN_TURNING_PATHS:
        return None
    if metrics["multicolor"] < _MIN_MULTICOLOR_PATHS:
        return None
    if metrics["three_color"] < _MIN_THREE_COLOR_PATHS:
        return None
    if metrics["color_transitions"] < _MIN_COLOR_TRANSITIONS:
        return None
    if metrics["trail_cells"] < _MIN_TRAIL_CELLS:
        return None
    if metrics["endpoint_diversity"] < _MIN_ENDPOINT_DIVERSITY:
        return None
    if metrics["non_particle_endpoints"] != _N_MOVED:
        return None
    if metrics["diff_from_baseline"] < _MIN_DIFF_FROM_BASELINE:
        return None
    if metrics["trail_color_union"] < 6:
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
