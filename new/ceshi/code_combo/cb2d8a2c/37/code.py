# Combo variant of cb2d8a2c (index 37)
# combo_spec: D8/2+D16/2
# Name: 单向障碍+蓝色聚集一端
#
# Design analysis for the fused variant:
# a) Concrete behavior and traced example:
#    - Canonical view: rotate/flip so the unique green seed is on the top border and the path
#      travels downward. In that view each obstacle is a floating horizontal bar.
#    - Original task: blue cells were scattered along a wall-attached bar; their count gave one
#      symmetric keep-out radius, so the path stayed that far from both bar ends.
#    - Fused task: blue cells form one contiguous block at exactly one bar end. The block length
#      still gives the distance, but only that same side is distance-constrained.
#    - Example: current trunk column = 9, bar spans cols 7..14, blue block length = 3.
#      If blue is on the left end, the left lane is col 3 (`7 - 3 - 1`) while the right lane is
#      col 15 (`14 + 1`). The path therefore slips around the right side unless its current column
#      is already much closer to the left. If the same bar moves the blue block to the right end,
#      the right lane becomes col 18 and the left lane becomes col 6, so the preferred detour flips.
#    - Difference from the original: the same blue count no longer inflates both sides equally.
#      Blue placement now matters, not just blue amount.
# b) Complexity calibration:
#    - Bars remain 1-cell-thick, axis-aligned, and few in number (2-4), so ARC examples stay easy
#      to parse visually.
#    - Distance values are still only 1..4. That is a small repeated vocabulary and keeps the green
#      detours visually comparable across examples.
#    - Blue blocks are always contiguous at one end, never scattered. This makes the D8/2 signal
#      obvious without adding extra symbols or colors.
# c) Hidden-assumption audit against the original main():
#    - Original `blue_count = sum(...)` remains valid, but `count` no longer defines a symmetric
#      halo. The solver must read which end owns the blue block.
#    - Original wall-attachment is dropped. If bars stayed attached to a wall, one of the two
#      left/right one-sided distance states would often be inert. Floating bars make the directional
#      signal visible and usable.
#    - Original Dijkstra over a symmetric forbidden region is no longer the right model. The fused
#      rule is "two candidate side lanes per bar, one expanded and one hugging the bar", so the
#      solver now evaluates those two lanes explicitly.
# d) Placement legality:
#    - Every generated bar is contiguous, horizontal in canonical view, and fully inside bounds.
#    - The chosen path column always lies inside each bar span, so every bar really blocks the
#      current downward trunk and must be interpreted.
#    - Rows are spaced by at least 2, guaranteeing enough room for the `down -> sideways -> down`
#      maneuver around each bar.
# e) Visual observability:
#    - D8/2 is visible directly in the input because blue cells form a solid endpoint patch.
#    - D16/2 is visible in the output because the path hugs one side of a bar but leaves a larger
#      gap on the blue-marked side.
#    - The generator requires both left-marked and right-marked bars, so examples visibly support
#      the directional reading instead of collapsing to one repeated orientation.
# f) Visual cleanliness:
#    - Only four colors are used: azure background, red/blue obstacle bars, green path.
#    - No element overlaps ambiguously: the green path is only painted on background cells, and the
#      blue-to-red recolor happens after routing.
# g) Change inferability:
#    - From a few examples, a human can test "blue count expands both sides" and see it fail,
#      because one side stays flush to the bar while the other side gets pushed farther away.
#    - The same examples also show that a longer blue block means a wider gap on that marked side.
# h) Quantitative fusion metric used after implementation:
#    - `analyze_fused_behavior()` reports the per-sample blue-count set, how many bars changed their
#      chosen side-lane relative to a symmetric-distance baseline, and whether the full output still
#      matches that baseline.
#    - The generator rejects samples unless:
#      `distinct_blue_counts >= 2`, both marker directions appear, `directional_shift_bars >= 1`,
#      and the fused output differs from the symmetric baseline.
# i) New solver assumptions re-checked against the generator:
#    - Exactly one green seed lies on one border edge.
#    - After canonicalization every obstacle component is one horizontal contiguous bar.
#    - Blue cells on a bar form a contiguous prefix or suffix and encode the side-specific radius.
#    - Each next bar covers the current path column; the generator enforces this before accepting a
#      sample and validates by re-running the solver plus the baseline comparison.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
MAX_GEN_RETRIES = 800


def _detect_start_edge(grid: np.ndarray) -> str:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("green start must be on a border edge")


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.array(grid, copy=True)
    edge = _detect_start_edge(g)
    if edge == "top":
        return g.copy(), edge
    if edge == "bottom":
        return np.flipud(g).copy(), edge
    if edge == "left":
        return g.T.copy(), edge
    return np.fliplr(g).T.copy(), edge


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _paint_hline(out: np.ndarray, row: int, c0: int, c1: int) -> None:
    if c0 > c1:
        c0, c1 = c1, c0
    for col in range(c0, c1 + 1):
        assert out[row, col] in (BG, GREEN), "path cannot overwrite obstacle cells"
        out[row, col] = GREEN


def _paint_vline(out: np.ndarray, col: int, r0: int, r1: int) -> None:
    if r0 > r1:
        r0, r1 = r1, r0
    for row in range(r0, r1 + 1):
        assert out[row, col] in (BG, GREEN), "path cannot overwrite obstacle cells"
        out[row, col] = GREEN


def _extract_bars(canonical: np.ndarray) -> list[dict]:
    bar_mask = (canonical == BLUE) | (canonical == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict] = []
    for comp in comps:
        rows = sorted({int(r) for r, _ in comp})
        cols = sorted(int(c) for _, c in comp)
        assert len(rows) == 1, "canonical obstacles must be horizontal bars"
        row = rows[0]
        c0 = cols[0]
        c1 = cols[-1]
        length = c1 - c0 + 1
        assert len(comp) == length, "bars must be contiguous with no gaps"
        segment = canonical[row, c0 : c1 + 1]
        blue_idx = np.flatnonzero(segment == BLUE)
        assert blue_idx.size >= 1, "each bar must contain blue cells"
        assert np.all(np.diff(blue_idx) == 1), "blue cells must be one contiguous block"
        blue_lo = int(blue_idx[0])
        blue_hi = int(blue_idx[-1])
        blue_count = int(blue_idx.size)
        if blue_lo == 0:
            assert np.all(segment[:blue_count] == BLUE)
            assert np.all(segment[blue_count:] == RED)
            marker_side = "left"
        elif blue_hi == length - 1:
            assert np.all(segment[: length - blue_count] == RED)
            assert np.all(segment[length - blue_count :] == BLUE)
            marker_side = "right"
        else:
            raise AssertionError("blue block must touch exactly one bar end")
        bars.append(
            {
                "row": row,
                "c0": c0,
                "c1": c1,
                "length": length,
                "blue_count": blue_count,
                "marker_side": marker_side,
            }
        )
    assert bars, "grid must contain at least one obstacle bar"
    bars.sort(key=lambda bar: int(bar["row"]))
    return bars


def _lane_columns(bar: dict, symmetric: bool) -> tuple[int, int]:
    radius = int(bar["blue_count"])
    c0 = int(bar["c0"])
    c1 = int(bar["c1"])
    if symmetric:
        return c0 - radius - 1, c1 + radius + 1
    left_gap = radius + 1 if bar["marker_side"] == "left" else 1
    right_gap = radius + 1 if bar["marker_side"] == "right" else 1
    return c0 - left_gap, c1 + right_gap


def _choose_side(entry_col: int, bar: dict, symmetric: bool) -> tuple[str, int, int, int]:
    left_col, right_col = _lane_columns(bar, symmetric=symmetric)
    left_cost = entry_col - left_col
    right_cost = right_col - entry_col
    assert left_cost > 0 and right_cost > 0, "current trunk must hit the bar span"
    if left_cost < right_cost:
        return "left", left_col, left_cost, right_cost
    if right_cost < left_cost:
        return "right", right_col, left_cost, right_cost
    # Deterministic tie-break: prefer the unmarked side, then left.
    if not symmetric and bar["marker_side"] == "right":
        return "left", left_col, left_cost, right_cost
    if not symmetric and bar["marker_side"] == "left":
        return "right", right_col, left_cost, right_cost
    return "left", left_col, left_cost, right_cost


def _solve_canonical(canonical: np.ndarray, *, symmetric: bool) -> tuple[np.ndarray, list[dict]]:
    out = np.array(canonical, copy=True)
    bars = _extract_bars(canonical)
    start = np.argwhere(canonical == GREEN)
    assert start.shape[0] == 1, "canonical grid must contain one green seed"
    sr, sc = map(int, start[0])
    assert sr == 0, "canonical view must start from the top border"
    h, w = canonical.shape

    current_row = sr
    current_col = sc
    trace: list[dict] = []
    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        assert current_row <= row - 1, "bars must be ordered along the travel axis"
        assert c0 <= current_col <= c1, "each bar must cover the current downward trunk"
        left_col, right_col = _lane_columns(bar, symmetric=symmetric)
        assert 0 <= left_col < w and 0 <= right_col < w, "detour lanes must stay in bounds"
        chosen_side, exit_col, left_cost, right_cost = _choose_side(current_col, bar, symmetric=symmetric)

        _paint_vline(out, current_col, current_row, row - 1)
        _paint_hline(out, row - 1, current_col, exit_col)
        _paint_vline(out, exit_col, row - 1, row + 1)

        trace.append(
            {
                "row": row,
                "c0": c0,
                "c1": c1,
                "length": int(bar["length"]),
                "blue_count": int(bar["blue_count"]),
                "marker_side": str(bar["marker_side"]),
                "entry_col": current_col,
                "left_col": left_col,
                "right_col": right_col,
                "chosen_side": chosen_side,
                "exit_col": exit_col,
                "left_cost": left_cost,
                "right_cost": right_cost,
            }
        )
        current_row = row + 1
        current_col = exit_col

    _paint_vline(out, current_col, current_row, h - 1)
    out[out == BLUE] = RED
    return out, trace


def _symmetric_baseline_output(grid: np.ndarray) -> np.ndarray:
    canonical, edge = _to_canonical(grid)
    solved, _ = _solve_canonical(canonical, symmetric=True)
    return _from_canonical(solved, edge)


def analyze_fused_behavior(grid: np.ndarray) -> dict:
    canonical, _ = _to_canonical(grid)
    fused_out, fused_trace = _solve_canonical(canonical, symmetric=False)
    _, symmetric_trace = _solve_canonical(canonical, symmetric=True)
    baseline_out = _symmetric_baseline_output(grid)

    directional_shift = 0
    for fused_step, base_step in zip(fused_trace, symmetric_trace):
        if (
            fused_step["chosen_side"] != base_step["chosen_side"]
            or fused_step["exit_col"] != base_step["exit_col"]
        ):
            directional_shift += 1

    blue_counts = tuple(int(step["blue_count"]) for step in fused_trace)
    marker_sides = tuple(str(step["marker_side"]) for step in fused_trace)
    return {
        "bar_count": len(fused_trace),
        "blue_counts": blue_counts,
        "distinct_blue_counts": len(set(blue_counts)),
        "left_markers": marker_sides.count("left"),
        "right_markers": marker_sides.count("right"),
        "directional_shift_bars": directional_shift,
        "baseline_matches_variant": bool(np.array_equal(_from_canonical(fused_out, _detect_start_edge(grid)), baseline_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green seed to the opposite edge while each bar has one distance-heavy side and one hugging side.

    Input structure:
    - The background is azure (8), with one green (3) border seed and several horizontal/vertical red-blue obstacle bars.
    - In canonical top-start view, each obstacle is a 1-cell-thick horizontal bar.
    - On every bar, blue pixels form one solid block at exactly one end rather than being sprinkled through the bar.

    Rule:
    - Rotate/flip to canonical top-start view. Read each bar in top-to-bottom order.
    - The blue-block length gives the extra sidestep size for that same end: if blue is on the left/up end, the left lane must stay `blue_count` cells farther away while the right lane may pass adjacent to the bar; if blue is on the right/down end, the right lane gets that larger offset instead.
    - Around each bar, compare the two legal side lanes and take the shorter orthogonal detour from the current trunk column.
    - Paint the full green route and then recolor every blue pixel to red.

    Key insight:
    Blue now carries two pieces of information at once: how wide the detour must be, and which side of the bar is the constrained side. The opposite side is the cheap hugging lane.
    """
    canonical, edge = _to_canonical(grid)
    solved, _ = _solve_canonical(canonical, symmetric=False)
    return _from_canonical(solved, edge)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blue_counts(count: int) -> tuple[int, ...]:
    while True:
        blues = tuple(random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count))
        if len(set(blues)) >= 2:
            return blues


def _min_main_len(count: int) -> int:
    return 2 * count + 1


def _row_positions(main_size: int, count: int) -> list[int]:
    slack = main_size - _min_main_len(count)
    assert slack >= 0
    extras = _random_composition(slack, count + 1)
    row = 1 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += 2 + extras[i]
        rows.append(row)
    return rows


def _candidate_range(
    entry_col: int,
    width: int,
    length: int,
    marker_side: str,
    chosen_side: str,
    blue_count: int,
) -> range:
    left_gap = blue_count + 1 if marker_side == "left" else 1
    right_gap = blue_count + 1 if marker_side == "right" else 1
    if chosen_side == "left":
        lo = max(0, entry_col - left_gap - length + 1)
        hi = min(entry_col - 1, width - left_gap - length)
        return range(lo, hi + 1) if lo <= hi else range(0)
    lo = max(entry_col + 1, right_gap + length - 1)
    hi = min(width - 1, entry_col + right_gap + length - 1)
    return range(lo, hi + 1) if lo <= hi else range(0)


def _bar_from_choice(
    entry_col: int,
    width: int,
    length: int,
    marker_side: str,
    chosen_side: str,
    exit_col: int,
    blue_count: int,
) -> dict | None:
    left_gap = blue_count + 1 if marker_side == "left" else 1
    right_gap = blue_count + 1 if marker_side == "right" else 1
    if chosen_side == "left":
        c0 = exit_col + left_gap
    else:
        c0 = exit_col - right_gap - length + 1
    c1 = c0 + length - 1
    if not (0 <= c0 and c1 < width):
        return None
    if not (c0 <= entry_col <= c1):
        return None
    bar = {
        "c0": c0,
        "c1": c1,
        "length": length,
        "blue_count": blue_count,
        "marker_side": marker_side,
    }
    chosen_now, _, left_cost, right_cost = _choose_side(entry_col, bar, symmetric=False)
    if left_cost == right_cost:
        return None
    if chosen_now != chosen_side:
        return None
    return bar


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, marker_side: str, blue_count: int) -> None:
    segment = np.full(length, RED, dtype=int)
    if marker_side == "left":
        segment[:blue_count] = BLUE
    else:
        segment[-blue_count:] = BLUE
    grid[row, c0 : c0 + length] = segment


def _build_canonical_candidate(
    main_size: int,
    cross_size: int,
    rows: list[int],
    blue_counts: tuple[int, ...],
) -> np.ndarray | None:
    grid = np.full((main_size, cross_size), BG, dtype=int)
    start_col = random.randint(max(1, cross_size // 3), min(cross_size - 2, cross_size - 1 - cross_size // 3))
    current_col = start_col
    left_markers = 0
    right_markers = 0

    for i, (row, blue_count) in enumerate(zip(rows, blue_counts)):
        feasible: list[tuple[int, int, str, str, int, int, int]] = []
        remaining = len(rows) - i - 1
        for length in range(BAR_LEN_MIN, min(BAR_LEN_MAX, cross_size - 2) + 1):
            for marker_side in ("left", "right"):
                for chosen_side in ("left", "right"):
                    for exit_col in _candidate_range(
                        current_col,
                        cross_size,
                        length,
                        marker_side,
                        chosen_side,
                        blue_count,
                    ):
                        bar = _bar_from_choice(
                            current_col,
                            cross_size,
                            length,
                            marker_side,
                            chosen_side,
                            exit_col,
                            blue_count,
                        )
                        if bar is None:
                            continue
                        sym_side, sym_exit, _, _ = _choose_side(current_col, bar, symmetric=True)
                        shift = int(sym_side != chosen_side or sym_exit != exit_col)
                        anti_marker = int(chosen_side != marker_side)
                        marker_bonus = 0
                        if left_markers == 0 and marker_side == "left":
                            marker_bonus += 1
                        if right_markers == 0 and marker_side == "right":
                            marker_bonus += 1
                        if remaining == 0:
                            if left_markers == 0 and marker_side != "left":
                                continue
                            if right_markers == 0 and marker_side != "right":
                                continue
                        score = 4 * shift + 2 * anti_marker + marker_bonus
                        feasible.append((score, exit_col, marker_side, chosen_side, length, bar["c0"], shift))
        if not feasible:
            return None

        feasible.sort(key=lambda item: (item[0], random.random()), reverse=True)
        top = feasible[: max(1, min(12, len(feasible)))]
        score, exit_col, marker_side, _chosen_side, length, c0, _shift = random.choice(top)
        _paint_bar(grid, row, c0, length, marker_side, blue_count)
        if marker_side == "left":
            left_markers += 1
        else:
            right_markers += 1
        current_col = exit_col

    if left_markers == 0 or right_markers == 0:
        return None
    grid[0, start_col] = GREEN
    return grid


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blue_counts = _choose_blue_counts(count)
    main_size = random.randint(max(main_lo, _min_main_len(count)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    rows = _row_positions(main_size, count)

    canonical = _build_canonical_candidate(main_size, cross_size, rows, blue_counts)
    if canonical is None:
        return None
    inp = _from_canonical(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        metrics = analyze_fused_behavior(inp)
        if metrics["distinct_blue_counts"] < 2:
            return None
        if metrics["left_markers"] == 0 or metrics["right_markers"] == 0:
            return None
        if metrics["directional_shift_bars"] < 1:
            return None
        if metrics["baseline_matches_variant"]:
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
