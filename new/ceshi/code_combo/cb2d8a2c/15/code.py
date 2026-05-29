# Combo variant of cb2d8a2c (index 15)
# combo_spec: D8/0+D20/0
# Name: 蓝色位置编码方向+蓝色标记通行口
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and small trace:
#    - In canonical top-start orientation, the grid keeps one green border start and several
#      1-cell-thick horizontal obstacle bars, but the bars are made floating instead of
#      side-attached. That adjustment is necessary: if a bar touches a wall, the bypass side
#      is forced by geometry and the D8 direction signal would collapse into noise.
#    - Red cells are solid barrier cells. Blue cells are literal holes in the bar: the green
#      route may stand on a blue cell and pierce the bar there.
#    - The same blue cells also encode a bypass side. If the blue centroid is left of the
#      bar midpoint, the recommended detour side is left; if the centroid is right of the
#      midpoint, the recommended detour side is right.
#    - The local decision is:
#         * if the current vertical track meets the bar and some blue hole lies within one
#           sideways step of that track, the route nudges to that nearby blue hole and crosses;
#         * otherwise it ignores the holes as crossings and goes around the end indicated by
#           the blue bias.
#    - Small example: current track at column 8, bar spans columns 4..11.
#         * blue at {7, 9}: one hole is adjacent to the track, so the path crosses there.
#         * blue at {4, 6}: no hole is near column 8, the blue centroid is left-biased, so
#           the path detours around the left end instead of piercing the bar.
# b) Complexity calibration:
#    - The alignment threshold is only one cell, giving a small and repeatable discrete rule.
#      That keeps the "cross or bypass?" choice legible rather than turning it into fuzzy
#      global optimization.
#    - Every generated sample uses 3-4 bars, contains at least one gate-crossed bar and at
#      least two detoured bars, and the detours include both left and right directions. This
#      gives direct support for both halves of the combo from only a few examples.
# c) Hidden-assumption audit against the original solver:
#    - Original assumption 1: blue count encodes a keep-out radius. Broken. Blue now marks
#      passable cells, so there is no Manhattan flood-fill forbidden halo.
#    - Original assumption 2: all bar cells are blocked. Broken. Only red cells remain
#      blocked; blue cells may be traversed and become green only if the chosen path uses them.
#    - Original assumption 3: wall attachment determines the turn side. Broken by D8/0, so
#      the generator switches to floating bars with space on both sides.
#    - Original assumption 4: all blue pixels become red in the output. Broken. Unused blue
#      gate markers stay blue so the opening structure remains visible.
# d) Placement legality:
#    - Bars are single-row 4-connected components, separated along the main axis by at least
#      three rows, so each bar can be handled by a clean row-above / row-through-or-around /
#      row-below maneuver without overlapping the next one.
#    - Each bar leaves at least one free column on both sides, and detour bars leave an extra
#      column on the chosen side so the path can land there and still be intercepted by later bars.
#    - The generator places every bar so the current track column lies inside that bar's span.
#      Therefore each bar genuinely forces a cross-vs-detour choice instead of being ignorable.
# e) Visual observability:
#    - The D20 behavior is visible because some output paths clearly turn specific blue pixels
#      into green while leaving other blue pixels untouched.
#    - The D8 behavior is visible because other bars keep all blue pixels unused, and the
#      detour side matches the blue cluster's left/right bias.
#    - Because both crossed and bypassed bars appear in every sample, the task does not reduce
#      to either "always use holes" or "always read side bias".
# f) Visual cleanliness:
#    - The scene still uses only azure background, red bars, blue markers, and one green path.
#    - Floating bars are spaced and long enough to read as separate obstacles; no legends or
#      extra helper objects are added.
# g) Change inferability / anti-degeneracy:
#    - The generator rejects any candidate whose fused output matches a counterfactual
#      "always cross through the nearest blue hole" solver or a counterfactual
#      "always detour according to blue bias" solver.
#    - It also rejects candidates unless the fused route actually contains both crossing and
#      detouring events, so the intended mixed behavior is guaranteed to be present.

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
BAR_CROSS_PREFERRED_LO = 48
START_GAP = 2
BAR_ROW_STEP = 3
END_GAP = 1
GATE_REACH = 1
MAX_GEN_RETRIES = 600
MAX_LAYOUT_RETRIES = 120


def _start_edge(grid: np.ndarray) -> tuple[str, int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return "top", r, c
    if c == w - 1:
        return "right", r, c
    if r == h - 1:
        return "bottom", r, c
    if c == 0:
        return "left", r, c
    raise AssertionError("green start must be on a border edge")


def _to_canonical_top(grid: np.ndarray) -> tuple[np.ndarray, str]:
    edge, _, _ = _start_edge(grid)
    if edge == "top":
        return np.array(grid, copy=True), edge
    if edge == "right":
        return np.rot90(grid, 1).copy(), edge
    if edge == "bottom":
        return np.rot90(grid, 2).copy(), edge
    return np.rot90(grid, -1).copy(), edge


def _from_canonical_top(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "right":
        return np.rot90(grid, -1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, 1).copy()


def _segment_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r0 == r1 or c0 == c1, "segments must be axis-aligned"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    seg = _segment_coords(r0, c0, r1, c1)
    if path:
        assert path[-1] == seg[0], "segments must connect"
        path.extend(seg[1:])
    else:
        path.extend(seg)


def _bias_side(blue_cols: list[int], c0: int, c1: int) -> str:
    midpoint = (c0 + c1) / 2.0
    centroid = sum(blue_cols) / len(blue_cols)
    return "left" if centroid < midpoint else "right"


def _decoded_bar_infos(grid: np.ndarray) -> list[dict[str, object]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    infos: list[dict[str, object]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        pts = np.array(comp, dtype=int)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        assert r0 == r1, "canonical bars must be horizontal"
        blue_cols = sorted(int(c) for r, c in pts if grid[int(r), int(c)] == BLUE)
        assert blue_cols, "each bar must contain at least one blue gate"
        infos.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "blue_cols": blue_cols,
                "side": _bias_side(blue_cols, c0, c1),
            }
        )
    infos.sort(key=lambda info: int(info["row"]))
    return infos


def _choose_gate(current_col: int, blue_cols: list[int], side: str, max_shift: int | None) -> int | None:
    candidates = [c for c in blue_cols if max_shift is None or abs(c - current_col) <= max_shift]
    if not candidates:
        return None
    if side == "left":
        return min(candidates, key=lambda c: (abs(c - current_col), c))
    return min(candidates, key=lambda c: (abs(c - current_col), -c))


def _bar_target(
    current_col: int,
    info: dict[str, object],
    behavior: str,
) -> tuple[str, int]:
    c0 = int(info["c0"])
    c1 = int(info["c1"])
    blue_cols = list(info["blue_cols"])
    side = str(info["side"])

    if not (c0 <= current_col <= c1):
        return "straight", current_col

    if behavior == "cross_only":
        gate = _choose_gate(current_col, blue_cols, side, max_shift=None)
        assert gate is not None
        return "cross", gate

    if behavior == "detour_only":
        return ("detour", c0 - 1) if side == "left" else ("detour", c1 + 1)

    if behavior != "fused":
        raise ValueError(f"unknown behavior: {behavior}")

    gate = _choose_gate(current_col, blue_cols, side, max_shift=GATE_REACH)
    if gate is not None:
        return "cross", gate
    return ("detour", c0 - 1) if side == "left" else ("detour", c1 + 1)


def _plan_actions(
    grid: np.ndarray,
    infos: list[dict[str, object]],
    behavior: str,
) -> list[tuple[str, int]]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the start on the top edge"

    current_col = sc
    actions: list[tuple[str, int]] = []
    for info in infos:
        action, target_col = _bar_target(current_col, info, behavior)
        actions.append((action, target_col))
        current_col = target_col
    return actions


def _build_canonical_path(
    grid: np.ndarray,
    infos: list[dict[str, object]],
    behavior: str = "fused",
) -> tuple[list[tuple[int, int]], list[str]]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the start on the top edge"

    path: list[tuple[int, int]] = [(sr, sc)]
    current_r, current_col = sr, sc
    actions: list[str] = []

    for info in infos:
        row = int(info["row"])
        _append_segment(path, current_r, current_col, row - 1, current_col)
        current_r = row - 1

        action, target_col = _bar_target(current_col, info, behavior)
        if action != "straight":
            _append_segment(path, current_r, current_col, current_r, target_col)
        _append_segment(path, current_r, target_col, row + 1, target_col)
        current_r, current_col = row + 1, target_col
        actions.append(action)

    h, _ = grid.shape
    _append_segment(path, current_r, current_col, h - 1, current_col)
    return path, actions


def _paint_solution(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        assert out[r, c] != RED, "path entered a red obstacle cell"
        out[r, c] = GREEN
    return out


def _solve_canonical(grid: np.ndarray, behavior: str = "fused") -> tuple[np.ndarray, list[str]]:
    infos = _decoded_bar_infos(grid)
    path, actions = _build_canonical_path(grid, infos, behavior=behavior)
    out = _paint_solution(grid, path)
    return out, actions


def _safe_solve_canonical(grid: np.ndarray, behavior: str) -> tuple[np.ndarray, list[str]] | None:
    try:
        return _solve_canonical(grid, behavior=behavior)
    except Exception:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green border path to the opposite edge while blue cells on each obstacle bar serve as both pass-through gates and side-bias markers.

    Input structure:
    - The background is azure (8), with several 1-cell-thick red bars that also contain one or more blue (1) cells.
    - Exactly one green (3) pixel sits on a border edge and marks the path start.
    - The bars are floating obstacles: red cells block movement, while blue cells are holes in those bars.

    Rule:
    - Process the bars in order from the start edge toward the opposite edge.
    - If the current track hits a bar and some blue hole on that bar is at most one column/row away from the current track, shift to the nearest such blue cell and cross through it.
    - Otherwise bypass the bar on the side indicated by the blue placement: blue mass leaning toward the left/up half means detour left/up, and leaning toward the right/down half means detour right/down. Used blue holes become green; unused blue holes stay blue.

    Key insight:
    The same blue pixels carry two meanings at once. Individually they are literal openings, but collectively their position also chooses the detour side when no opening is close enough to use directly.
    """
    canonical, edge = _to_canonical_top(np.asarray(grid, dtype=int))
    out, _ = _solve_canonical(canonical, behavior="fused")
    return _from_canonical_top(out, edge)


def _min_main_len(count: int) -> int:
    return START_GAP + (count - 1) * BAR_ROW_STEP + END_GAP + 1


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
        (BAR_CROSS_PREFERRED_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - 2,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _strict_offset_side(offsets: tuple[int, ...], length: int) -> str | None:
    midpoint = (length - 1) / 2.0
    centroid = sum(offsets) / len(offsets)
    if centroid < midpoint:
        return "left"
    if centroid > midpoint:
        return "right"
    return None


def _non_adjacent_combos(length: int, gate_count: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(range(length), gate_count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]


def _choose_bar_spec(
    cross_size: int,
    current_col: int,
    mode: str,
    desired_side: str | None,
) -> dict[str, object] | None:
    gate_counts = list(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1))
    random.shuffle(gate_counts)

    for gate_count in gate_counts:
        bounds = _length_bounds(cross_size, gate_count)
        if bounds is None:
            continue

        lengths = list(range(bounds[0], bounds[1] + 1))
        random.shuffle(lengths)
        for length in lengths:
            combos = _non_adjacent_combos(length, gate_count)
            if not combos:
                continue

            c0_min = 1
            c0_max = cross_size - length - 1
            if mode == "detour" and desired_side == "left":
                c0_min = 2
            if mode == "detour" and desired_side == "right":
                c0_max = cross_size - length - 2
            if c0_min > c0_max:
                continue

            candidate_c0 = list(range(c0_min, c0_max + 1))
            random.shuffle(candidate_c0)
            for c0 in candidate_c0:
                c1 = c0 + length - 1
                if not (c0 <= current_col <= c1):
                    continue
                track_offset = current_col - c0

                valid: list[tuple[tuple[int, ...], str]] = []
                for offsets in combos:
                    side = _strict_offset_side(offsets, length)
                    if side is None:
                        continue
                    if mode == "cross":
                        if any(abs(off - track_offset) <= GATE_REACH for off in offsets):
                            valid.append((offsets, side))
                    else:
                        if side != desired_side:
                            continue
                        if all(abs(off - track_offset) > GATE_REACH for off in offsets):
                            valid.append((offsets, side))

                if not valid:
                    continue

                offsets, side = random.choice(valid)
                blue_cols = [c0 + off for off in offsets]
                if mode == "cross":
                    target_col = _choose_gate(current_col, blue_cols, side, max_shift=GATE_REACH)
                    assert target_col is not None
                else:
                    target_col = c0 - 1 if side == "left" else c1 + 1

                return {
                    "c0": c0,
                    "length": length,
                    "blue_cols": blue_cols,
                    "side": side,
                    "target_col": target_col,
                }

    return None


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, blue_cols: list[int]) -> None:
    colors = np.full(length, RED, dtype=int)
    for c in blue_cols:
        colors[c - c0] = BLUE
    grid[row, c0 : c0 + length] = colors


def _choose_modes(count: int) -> tuple[str, ...]:
    choices = [
        combo
        for combo in itertools.product(("cross", "detour"), repeat=count)
        if combo.count("cross") >= 1 and combo.count("detour") >= 2
    ]
    return random.choice(choices)


def _choose_detour_sides(detour_count: int) -> tuple[str, ...]:
    choices = [
        combo
        for combo in itertools.product(("left", "right"), repeat=detour_count)
        if "left" in combo and "right" in combo
    ]
    return random.choice(choices)


def _canonical_grid(main_size: int, cross_size: int, modes: tuple[str, ...]) -> np.ndarray | None:
    count = len(modes)
    slack = main_size - _min_main_len(count)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = START_GAP + extras[0]
    rows = [row]
    for i in range(1, count):
        row += BAR_ROW_STEP + extras[i]
        rows.append(row)

    start_lo = max(2, cross_size // 4)
    start_hi = min(cross_size - 3, cross_size - 1 - cross_size // 4)
    if start_lo > start_hi:
        start_lo, start_hi = 1, cross_size - 2

    for _ in range(MAX_LAYOUT_RETRIES):
        grid = np.full((main_size, cross_size), BG, dtype=int)
        start_col = random.randint(start_lo, start_hi)
        current_col = start_col
        placed = True

        side_iter = iter(_choose_detour_sides(modes.count("detour")))
        for row, mode in zip(rows, modes):
            desired_side = next(side_iter) if mode == "detour" else None
            spec = _choose_bar_spec(cross_size, current_col, mode, desired_side)
            if spec is None:
                placed = False
                break
            _paint_bar(grid, row, int(spec["c0"]), int(spec["length"]), list(spec["blue_cols"]))
            current_col = int(spec["target_col"])

        if not placed:
            continue

        grid[0, start_col] = GREEN

        infos = _decoded_bar_infos(grid)
        if len(infos) != count:
            continue
        actions = [action for action, _ in _plan_actions(grid, infos, behavior="fused")]
        if actions.count("cross") < 1 or actions.count("detour") < 2:
            continue
        detour_dirs = [str(info["side"]) for info, action in zip(infos, actions) if action == "detour"]
        if "left" not in detour_dirs or "right" not in detour_dirs:
            continue
        return grid

    return None


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "right":
        return bool(np.any(out[:, 0] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    return bool(np.any(out[:, -1] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "right", "bottom", "left"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    modes = _choose_modes(count)
    main_size = random.randint(max(main_lo, _min_main_len(count)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)

    canonical = _canonical_grid(main_size, cross_size, modes)
    if canonical is None:
        return None

    actual = _safe_solve_canonical(canonical, behavior="fused")
    cross_only = _safe_solve_canonical(canonical, behavior="cross_only")
    detour_only = _safe_solve_canonical(canonical, behavior="detour_only")
    if actual is None or cross_only is None or detour_only is None:
        return None
    actual_out, actual_actions = actual
    if np.array_equal(actual_out, cross_only[0]):
        return None
    if np.array_equal(actual_out, detour_only[0]):
        return None
    if actual_actions.count("cross") < 1 or actual_actions.count("detour") < 2:
        return None

    inp = _from_canonical_top(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(out, _from_canonical_top(actual_out, edge)):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
