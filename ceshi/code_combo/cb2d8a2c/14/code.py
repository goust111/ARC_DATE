# Combo variant of cb2d8a2c (index 14)
# combo_spec: D3/2+D8/2
# Name: 条长度编码距离+蓝色聚集一端
#
# Design analysis for the fused variant:
# a) Concrete behavior and traced example:
#    - Canonical view: rotate/flip so the single green seed is on the top border and the route
#      travels downward. In that view every obstacle is a floating horizontal bar.
#    - Original task: a bar's blue-count b creates a Manhattan keep-out radius b, and the path
#      takes whichever legal detour the shortest-path search prefers.
#    - Fused task: the keep-out radius no longer comes from blue-count. It is computed from the
#      total bar length `L` by `ceil(L / 4)`. Meanwhile the blue pixels are no longer scattered:
#      they form one contiguous block at exactly one end of the bar.
#    - The blue end is now directional. If the blue block is on the left end, the path must bypass
#      the bar via the left safe column `c0 - radius - 1`; if the blue block is on the right end,
#      it must bypass via `c1 + radius + 1`.
#    - Small trace:
#      current column = 10; bar = cols 7..15 (length 9 => radius 3); blue block on the left.
#      Original logic with 2 blue cells would have used radius 2 and might choose either side.
#      New logic uses radius 3 because the bar length is 9, so the legal top row is `row - 4`,
#      and because blue is on the left, the route must shift to col 3, descend past the bar, and
#      continue from there. The same geometric bar with the blue block moved to the right would
#      force the continuation column to become 19 instead.
# b) Complexity calibration:
#    - Bars remain 1-cell-thick, axis-aligned, and 2..4 in count, so the object family stays easy
#      to parse from a few ARC examples.
#    - Lengths are restricted to 6..15, which under `ceil(L / 4)` yields only radii 2, 3, or 4.
#      That is a small discrete set with repeated support, but it is visually different enough that
#      longer bars clearly induce wider green detours.
#    - Blue-block length is fixed, so humans are not misled into reusing the old "count blue"
#      strategy. What changes from sample to sample is blue-block side, not blue-block size.
# c) Hidden-assumption audit against the original main():
#    - Original `blue_count = sum(...)` encoded radius directly. Broken: blue is now only a marker.
#    - Original solver allowed any shortest legal bypass. Broken: the combo needs side-specific
#      bypassing, so the route must read the bar-end marker.
#    - Original generator assumed wall-attached alternating bars. Relaxed: this combo uses floating
#      bars so that both left and right bypasses are physically possible and the blue-end signal is
#      actually meaningful.
#    - Original Dijkstra alone cannot prove the marker matters, because it may choose the same side
#      even when the marker is ignored. The new generator therefore compares against a baseline
#      length-only shortest-path solver and rejects degenerate samples.
# d) Placement legality:
#    - Every canonical bar is a contiguous horizontal segment `row, c0..c1` with at least one cell
#      of margin on both sides after accounting for its length-derived radius.
#    - Rows satisfy `next_row >= row + prev_radius + next_radius + 3`, so the guided path can leave
#      one bar's safe band before entering the next bar's safe band.
#    - Bars are chosen sequentially from the current path column, so each accepted sample guarantees
#      that every bar really intercepts the current downward trunk and must be reacted to.
# e) Visual observability:
#    - The D8/2 part is visible directly in the input: blue no longer alternates through a bar but
#      appears as one solid endpoint patch.
#    - The D3/2 part is visible in the output: longer bars cause wider offsets, even when the blue
#      block length stays constant.
#    - Because the marker sometimes sends the path toward the locally longer side, input/output
#      examples visibly distinguish "read the blue end" from "just take the nearest escape".
# f) Visual cleanliness:
#    - Only four colors remain: azure background, red obstacle body, blue endpoint marker, green
#      path. No extra annotation color is introduced.
#    - Bars do not overlap each other, and the green path never paints over obstacle pixels, so the
#      detour geometry stays legible.
# g) Change inferability:
#    - Humans can test two hypotheses from a few examples:
#      1. blue-count controls distance: false, because blue-block size is constant;
#      2. bar length controls distance and blue endpoint controls detour side: true across samples.
# h) Quantitative fusion metric used after implementation:
#    - `analyze_fused_behavior()` reports:
#      `distinct_radii`: how many different length-derived distance buckets appear,
#      `anti_shortest_bars`: how many bars force the path to take the locally longer side,
#      `baseline_matches_variant`: whether ignoring endpoint markers gives the same output.
#    - The generator accepts only samples with `distinct_radii >= 2`, `anti_shortest_bars >= 1`,
#      and `baseline_matches_variant == False`, preventing collapse back to the original behavior.
# i) New solver assumptions re-checked against the generator:
#    - Exactly one green seed lies on one border edge.
#    - After canonicalization every obstacle component is one contiguous horizontal bar.
#    - Blue cells on a bar form one contiguous prefix or suffix, never an interior block.
#    - The current path column is inside each upcoming bar span, so each bar is actually relevant.
#    - The generator constructs only such inputs and validates them by re-running both solvers.

from __future__ import annotations

import heapq
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 15, 29
COLS_LO, COLS_HI = 18, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BLUE_BLOCK_LEN = 2
LENGTH_RADIUS_DIVISOR = 4
MAX_GEN_RETRIES = 800

RADIUS_TO_LENGTHS: dict[int, tuple[int, int]] = {
    2: (6, 8),
    3: (9, 12),
    4: (13, 15),
}


def _detect_start_edge(grid: np.ndarray) -> str:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    h, w = grid.shape
    if sr == 0:
        return "top"
    if sr == h - 1:
        return "bottom"
    if sc == 0:
        return "left"
    if sc == w - 1:
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


def _to_canonical_by_edge(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def _length_to_radius(length: int) -> int:
    return (int(length) + LENGTH_RADIUS_DIVISOR - 1) // LENGTH_RADIUS_DIVISOR


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
        c0, c1 = cols[0], cols[-1]
        length = c1 - c0 + 1
        assert len(comp) == length, "bars must be contiguous with no gaps"
        segment = canonical[row, c0 : c1 + 1]
        blue_pos = np.flatnonzero(segment == BLUE)
        assert blue_pos.size >= 1, "each bar must contain a blue endpoint marker"
        assert np.all(np.diff(blue_pos) == 1), "blue pixels must form one contiguous block"
        blue_lo = int(blue_pos[0])
        blue_hi = int(blue_pos[-1])
        if blue_lo == 0:
            assert np.all(segment[: blue_hi + 1] == BLUE)
            assert np.all(segment[blue_hi + 1 :] == RED)
            marker_side = "left"
        elif blue_hi == length - 1:
            assert np.all(segment[:blue_lo] == RED)
            assert np.all(segment[blue_lo:] == BLUE)
            marker_side = "right"
        else:
            raise AssertionError("blue block must occupy exactly one bar end")
        bars.append(
            {
                "row": row,
                "c0": c0,
                "c1": c1,
                "length": length,
                "radius": _length_to_radius(length),
                "marker_side": marker_side,
            }
        )
    assert bars, "grid must contain at least one obstacle bar"
    bars.sort(key=lambda bar: int(bar["row"]))
    return bars


def _guided_solve_canonical(canonical: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    bars = _extract_bars(canonical)
    out = np.array(canonical, copy=True)
    start_pts = np.argwhere(canonical == GREEN)
    assert start_pts.shape[0] == 1, "canonical grid must have one green start"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical start must be on the top row"
    current_row = sr
    current_col = sc
    h, w = canonical.shape

    trace: list[dict] = []
    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        radius = int(bar["radius"])
        top = row - radius - 1
        bottom = row + radius + 1
        assert 0 <= top < bottom < h, "bar clearance must stay in bounds"
        assert c0 - radius <= current_col <= c1 + radius, "each bar must block the current trunk"
        if bar["marker_side"] == "left":
            side_col = c0 - radius - 1
        else:
            side_col = c1 + radius + 1
        assert 0 <= side_col < w, "detour column must stay in bounds"
        assert current_row <= top, "bars must be ordered along the path direction"

        trace.append(
            {
                "entry_col": current_col,
                "exit_col": side_col,
                "row": row,
                "c0": c0,
                "c1": c1,
                "radius": radius,
                "marker_side": bar["marker_side"],
                "length": int(bar["length"]),
            }
        )

        _paint_vline(out, current_col, current_row, top)
        _paint_hline(out, top, current_col, side_col)
        _paint_vline(out, side_col, top, bottom)
        current_row = bottom
        current_col = side_col

    _paint_vline(out, current_col, current_row, h - 1)
    out[out == BLUE] = RED
    return out, trace


def _bar_halo_mask(shape: tuple[int, int], row: int, c0: int, c1: int, radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for dr in range(-radius, radius + 1):
        rr = row + dr
        if not (0 <= rr < h):
            continue
        rem = radius - abs(dr)
        lo = max(0, c0 - rem)
        hi = min(w - 1, c1 + rem)
        mask[rr, lo : hi + 1] = True
    return mask


def _dijkstra_path_canonical(canonical: np.ndarray, bars: list[dict]) -> list[tuple[int, int]]:
    h, w = canonical.shape
    start_pts = np.argwhere(canonical == GREEN)
    assert start_pts.shape[0] == 1, "canonical grid must have one green start"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical start must be on the top row"

    blocked = np.zeros((h, w), dtype=bool)
    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        radius = int(bar["radius"])
        blocked |= _bar_halo_mask((h, w), row, c0, c1, radius)
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, 2): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, 2)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if r == h - 1 and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(DIRS_URDL):
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "baseline shortest path must exist"
    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    path.reverse()
    return path


def _paint_path_and_recolor(canonical: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(canonical, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _baseline_length_only_output(grid: np.ndarray) -> np.ndarray:
    canonical, edge = _to_canonical(grid)
    bars = _extract_bars(canonical)
    path = _dijkstra_path_canonical(canonical, bars)
    return _from_canonical(_paint_path_and_recolor(canonical, path), edge)


def analyze_fused_behavior(grid: np.ndarray) -> dict:
    canonical, _ = _to_canonical(grid)
    bars = _extract_bars(canonical)
    _, trace = _guided_solve_canonical(canonical)
    baseline_matches = np.array_equal(main(grid), _baseline_length_only_output(grid))
    radii = [int(bar["radius"]) for bar in bars]
    anti_shortest = 0
    for step in trace:
        chosen = abs(int(step["exit_col"]) - int(step["entry_col"]))
        opposite = int(step["length"]) + 2 * int(step["radius"]) + 1 - chosen
        if chosen > opposite:
            anti_shortest += 1
    return {
        "bar_count": len(bars),
        "radii": tuple(radii),
        "distinct_radii": len(set(radii)),
        "left_markers": sum(1 for bar in bars if bar["marker_side"] == "left"),
        "right_markers": sum(1 for bar in bars if bar["marker_side"] == "right"),
        "anti_shortest_bars": anti_shortest,
        "baseline_matches_variant": bool(baseline_matches),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green seed to the opposite edge, but now each bar's total length sets the safe offset and the blue endpoint chooses the bypass side.

    Input structure:
    - bg = azure (8). Red/blue obstacle bars (colors 2/1) are 1-cell-thick line segments perpendicular to the travel direction.
    - A single green (3) pixel lies on one border edge and marks the route start.
    - On each bar, blue pixels form one contiguous block at exactly one end instead of being scattered through the bar.

    Rule:
    - Canonicalize to a top-start view. For a bar of total length `L`, compute its keep-out radius as `ceil(L / 4)`.
    - The path must remain farther than that Manhattan radius from the bar. The legal detour lane is therefore one cell beyond that halo.
    - If the blue block sits on the left/up end of the bar, the path exits around that left/up side; if the blue block sits on the right/down end, it exits around that right/down side.
    - After drawing the green path, all blue pixels become red.

    Key insight:
    Blue is no longer a counter. Bar length determines how wide the avoidance must be, while the contiguous blue endpoint acts as a direction marker that tells the path which side of the bar to use.
    """
    canonical, edge = _to_canonical(grid)
    solved, _ = _guided_solve_canonical(canonical)
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


def _choose_radii(count: int, main_hi: int) -> tuple[int, ...] | None:
    choices: list[tuple[int, ...]] = []
    for _ in range(200):
        radii = tuple(random.randint(2, 4) for _ in range(count))
        if len(set(radii)) < 2:
            continue
        if _min_main_len(radii) <= main_hi:
            choices.append(radii)
    if not choices:
        return None
    return random.choice(choices)


def _length_for_radius(radius: int) -> int:
    lo, hi = RADIUS_TO_LENGTHS[int(radius)]
    return random.randint(lo, hi)


def _min_main_len(radii: tuple[int, ...]) -> int:
    row = radii[0] + 2
    for prev_radius, next_radius in zip(radii, radii[1:]):
        row += prev_radius + next_radius + 3
    return row + radii[-1] + 3


def _row_positions(main_size: int, radii: tuple[int, ...]) -> list[int]:
    slack = main_size - _min_main_len(radii)
    assert slack >= 0
    extras = _random_composition(slack, len(radii) + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(radii)):
        row += radii[i - 1] + radii[i] + 3 + extras[i]
        rows.append(row)
    return rows


def _candidate_exit_cols(current_col: int, width: int, radius: int, length: int, side: str) -> list[int]:
    if side == "left":
        lo = max(0, current_col - radius - length)
        hi = min(current_col - radius - 1, width - radius - length - 1)
    else:
        lo = max(current_col + radius + 1, radius + length)
        hi = min(width - 1, current_col + radius + length)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, marker_side: str) -> None:
    segment = np.full(length, RED, dtype=int)
    if marker_side == "left":
        segment[:BLUE_BLOCK_LEN] = BLUE
    else:
        segment[-BLUE_BLOCK_LEN:] = BLUE
    grid[row, c0 : c0 + length] = segment


def _build_canonical_candidate(
    main_size: int,
    cross_size: int,
    rows: list[int],
    radii: tuple[int, ...],
    lengths: tuple[int, ...],
) -> np.ndarray | None:
    grid = np.full((main_size, cross_size), BG, dtype=int)
    start_col = random.randint(cross_size // 3, cross_size - 1 - cross_size // 3)
    current_col = start_col
    used_left = 0
    used_right = 0

    for i, (row, radius, length) in enumerate(zip(rows, radii, lengths)):
        feasible: list[tuple[str, int, int]] = []
        for side in ("left", "right"):
            for exit_col in _candidate_exit_cols(current_col, cross_size, radius, length, side):
                chosen = abs(exit_col - current_col)
                opposite = length + 2 * radius + 1 - chosen
                anti_shortest = 1 if chosen > opposite else 0
                feasible.append((side, exit_col, anti_shortest))
        if not feasible:
            return None

        remaining = len(rows) - i - 1
        need_left = used_left == 0 and remaining >= 1
        need_right = used_right == 0 and remaining >= 1

        candidates = feasible
        if need_left:
            left_only = [item for item in feasible if item[0] == "left"]
            if left_only:
                candidates = left_only
        if need_right and any(item[0] == "right" for item in candidates):
            right_only = [item for item in candidates if item[0] == "right"]
            if right_only:
                candidates = right_only

        candidates.sort(key=lambda item: (item[2], random.random()), reverse=True)
        side, exit_col, _ = random.choice(candidates[: max(1, min(6, len(candidates)))])

        if side == "left":
            c0 = exit_col + radius + 1
        else:
            c0 = exit_col - radius - length
        if not (0 <= c0 and c0 + length <= cross_size):
            return None
        if not (c0 <= current_col <= c0 + length - 1):
            return None

        _paint_bar(grid, row, c0, length, side)
        if side == "left":
            used_left += 1
        else:
            used_right += 1
        current_col = exit_col

    if used_left == 0 or used_right == 0:
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
    radii = _choose_radii(count, main_hi)
    if radii is None:
        return None
    lengths = tuple(_length_for_radius(radius) for radius in radii)
    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    rows = _row_positions(main_size, radii)

    canonical = _build_canonical_candidate(main_size, cross_size, rows, radii, lengths)
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
        if metrics["distinct_radii"] < 2:
            return None
        if metrics["anti_shortest_bars"] < 1:
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
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
