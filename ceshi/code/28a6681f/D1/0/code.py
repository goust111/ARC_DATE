# Variant of 28a6681f
# Dimension: D1 — 水流路径机制 (path)
# Change: 重力驱动的水流模拟：水像素先向下直落，被阻挡时通过唯一开放侧的对角方向下滑，支撑后水平滑向最近可落点... → 直线射线路径

from __future__ import annotations

import random

import numpy as np

# Design analysis:
# spec:
#   preserved_backbone:
#     grid: fixed 10x10 as in the original task
#     solids: 2 basin colors still build bottom-anchored walls / L-shapes / short stair blocks
#     water: color 1 remains the only moving color
#   D1_path_change:
#     old: each blue pixel searched for a deeper resting place by repeated down / diagonal / horizontal motion
#     new: blue pixels are emitted along one straight ray determined by an upper guide, and they stop only when the
#       next step on that ray would leave the grid or hit a solid / earlier-settled water pixel
#   concrete_visualization:
#     guide:
#       exactly one non-bottom color appears in the upper half
#       it is a straight 8-connected line entering from the left or right edge
#       its interior endpoint is the nozzle tip; the ray direction is the guide direction continued forward
#     source:
#       a contiguous run of blue pixels starts immediately after the guide tip on the same line
#       all source pixels belong to that one emitter line; no extra blue regions are used
#     target:
#       a bottom-anchored wall on the ray acts as the first collision point
#       output water becomes a packed run right before that wall, preserving water count
#   behavior_trace:
#     example:
#       guide: (2,0)-(3,1)
#       source: (4,2),(5,3)
#       blocker: vertical wall occupying (8,6),(9,6)
#       old_solver_on_same_input: would drop the two blue pixels downward into lower cavities
#       new_solver: clears the source run, marches both pixels along (+1,+1), and repacks them into (6,4),(7,5)
#     difference:
#       the key motion is no longer "find a lower basin"; it is "continue the visible guide as a straight ray"
#   complexity_calibration:
#     why_not_too_simple:
#       both horizontal and diagonal rays are generated
#       left and right emitters both occur
#       source length varies from 2 to 3, so collision packing is visible rather than a single-point teleport
#     why_not_too_free:
#       only one emitter exists per grid
#       the guide always encodes one unambiguous unit step from {right, down-right, left, down-left}
#   implicit_assumptions_review:
#     replaced_old_assumptions:
#       original input checks required every water cell to be supported from below; that is invalid here because
#       source water intentionally floats on the ray
#       original settle() depended on local gravity choices (down / diagonal / horizontal); every one of those
#       branches assumes vertical descent is meaningful, so the whole routine is replaced by guide parsing + ray march
#     new_assumptions:
#       exactly one guide color is present in the upper half and absent from the bottom row
#       the guide is a single straight chain with one border endpoint
#       all blue pixels form one contiguous source run immediately after the tip
#     generator_guarantees:
#       bottom structures use only bottom colors, so guide detection by "upper non-bottom color" is reliable
#       travel cells between source and blocker stay background, so the ray has visible motion
#   placement_legality:
#     guide and source are chosen from one lattice line fully inside bounds
#     the blocker column is bottom-anchored
#     decorative structures are only placed beyond the blocker relative to ray travel, so they cannot create an
#       earlier accidental collision
#   observability_and_cleanliness:
#     humans can read the direction directly from the colored guide
#     output water remains on the same line, making the ray mechanism visually explicit in 2-4 examples
#     overlapping layers are avoided: guide, source, travel corridor, and basin walls occupy disjoint cells in input


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _sgn(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _point(start: tuple[int, int], step: tuple[int, int], t: int) -> tuple[int, int]:
    return int(start[0] + step[0] * t), int(start[1] + step[1] * t)


def _trace_line(start: tuple[int, int], step: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    r, c = start
    while _in_bounds(r, c, h, w):
        pts.append((int(r), int(c)))
        r += int(step[0])
        c += int(step[1])
    return pts


def _paint_bottom_structure(
    grid: np.ndarray,
    cols: list[int],
    heights: list[int],
    color: int,
    forbidden: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    for col, height in zip(cols, heights):
        if not (0 <= int(col) < int(w)):
            continue
        hh = max(1, min(int(h), int(height)))
        for row in range(int(h - hh), int(h)):
            cell = (int(row), int(col))
            if cell not in forbidden:
                grid[cell] = int(color)


def _guide_info(grid: np.ndarray, bg: int, water: int) -> tuple[int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    bottom_colors = set(map(int, grid[h - 1]))
    candidate_colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(water)} - bottom_colors)
    valid: list[tuple[int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []

    for color in candidate_colors:
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        if not pts or not any(r < 7 for r, _c in pts):
            continue

        ptset = set(pts)
        neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {pt: [] for pt in pts}
        for r, c in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    nxt = (int(r + dr), int(c + dc))
                    if nxt in ptset:
                        neighbors[(r, c)].append(nxt)

        endpoints = [pt for pt, adj in neighbors.items() if len(adj) == 1]
        if len(endpoints) != 2 or any(len(adj) > 2 for adj in neighbors.values()):
            continue

        border_endpoints = [pt for pt in endpoints if pt[1] in (0, w - 1)]
        if len(border_endpoints) != 1:
            continue

        start = border_endpoints[0]
        first = neighbors[start][0]
        step = (_sgn(first[0] - start[0]), _sgn(first[1] - start[1]))
        if step not in {(0, 1), (1, 1), (0, -1), (1, -1)}:
            continue

        ordered = [start]
        prev: tuple[int, int] | None = None
        cur = start
        ok = True
        while True:
            nxts = [pt for pt in neighbors[cur] if pt != prev]
            if not nxts:
                break
            if len(nxts) != 1:
                ok = False
                break
            nxt = nxts[0]
            got = (_sgn(nxt[0] - cur[0]), _sgn(nxt[1] - cur[1]))
            if got != step:
                ok = False
                break
            ordered.append(nxt)
            prev, cur = cur, nxt

        if ok and len(ordered) == len(pts):
            valid.append((int(color), ordered, ordered[-1], step))

    assert len(valid) == 1, f"expected exactly one guide candidate, found {[item[0] for item in valid]}"
    return valid[0]


def _source_after_tip(grid: np.ndarray, tip: tuple[int, int], step: tuple[int, int], water: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    r = int(tip[0] + step[0])
    c = int(tip[1] + step[1])
    while _in_bounds(r, c, h, w) and int(grid[r, c]) == int(water):
        out.append((int(r), int(c)))
        r += int(step[0])
        c += int(step[1])
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Move blue source pixels along a guide-defined straight ray until they pack against the first collision wall.

    Input structure:
    - bg = 0. Water pixels = color 1.
    - Bottom-anchored solids still form the basin-like scenery, but one extra non-bottom color in the upper half is a
      straight guide entering from the left or right edge.
    - A contiguous source run of water begins immediately after the guide tip on the same lattice line.

    Rule:
    - Read the guide orientation from its border endpoint toward its interior tip.
    - Remove the source run and re-emit its pixels one by one along that same unit step.
    - Each pixel keeps marching in a straight line through background cells until the next step would leave the grid or
      hit a solid / earlier-settled water pixel, then it stops there.

    Key insight: The changing part is no longer gravity or overflow; the guide acts as a nozzle, so the output is a
    packed ray segment at the first collision site.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    bg = 0
    water = 1

    _guide_color, guide_pts, tip, step = _guide_info(g, bg, water)
    source = _source_after_tip(g, tip, step, water)
    assert source, "expected water source immediately after guide tip"

    if check_input:
        water_pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(water))}
        assert water_pts == set(source), "all water must belong to the single ray source"

        line_after_tip = _trace_line((int(tip[0] + step[0]), int(tip[1] + step[1])), step, h, w)
        assert line_after_tip[: len(source)] == source, "source must be contiguous on the emitter line"

        moved_probe = False
        probe_r = int(source[-1][0] + step[0])
        probe_c = int(source[-1][1] + step[1])
        while _in_bounds(probe_r, probe_c, h, w) and int(g[probe_r, probe_c]) == int(bg):
            moved_probe = True
            probe_r += int(step[0])
            probe_c += int(step[1])
        assert moved_probe, "generator should leave visible travel space after the source"

    for r, c in source:
        g[int(r), int(c)] = int(bg)

    moved = False
    for r, c in reversed(source):
        rr = int(r)
        cc = int(c)
        while True:
            nr = int(rr + step[0])
            nc = int(cc + step[1])
            if not _in_bounds(nr, nc, h, w) or int(g[nr, nc]) != int(bg):
                break
            rr, cc = nr, nc
        moved = bool(moved or ((rr, cc) != (int(r), int(c))))
        g[int(rr), int(cc)] = int(water)

    if check_output:
        final_water = [(int(r), int(c)) for r, c in np.argwhere(g == int(water))]
        assert len(final_water) == len(source), "water count must be conserved"

        line = _trace_line((int(tip[0] + step[0]), int(tip[1] + step[1])), step, h, w)
        pos = {pt: idx for idx, pt in enumerate(line)}
        idxs = sorted(pos[pt] for pt in final_water)
        assert idxs == list(range(idxs[0], idxs[-1] + 1)), "output water must form one contiguous ray segment"

        last = line[idxs[-1]]
        nr = int(last[0] + step[0])
        nc = int(last[1] + step[1])
        assert not _in_bounds(nr, nc, h, w) or int(g[nr, nc]) != int(bg), "ray must terminate at a collision"
        assert moved, "generated examples should show actual ray travel"

    return g


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = w = 10
    bg = 0
    water = 1
    palette = [2, 3, 4, 5, 6, 7, 8, 9]
    guide_color, wall_color, extra_color = rng.sample(palette, 3)

    side, step = rng.choice(
        [
            ("left", (0, 1)),
            ("left", (1, 1)),
            ("right", (0, -1)),
            ("right", (1, -1)),
        ]
    )

    if step[0] == 0:
        start = (rng.randint(4, 6), 0 if side == "left" else w - 1)
    else:
        start = (rng.randint(0, 2), 0 if side == "left" else w - 1)

    tmax = len(_trace_line(start, step, h, w)) - 1
    params: list[tuple[int, int, int, int]] = []
    for guide_len in (2, 3):
        for source_len in (2, 3):
            for gap in (1, 2):
                hit_t = int(guide_len + 2 * source_len + gap)
                if hit_t <= int(tmax):
                    params.append((guide_len, source_len, gap, hit_t))
    if not params:
        return None

    guide_len, source_len, gap, hit_t = rng.choice(params)
    source_ts = list(range(int(guide_len), int(guide_len + source_len)))
    output_ts = list(range(int(hit_t - source_len), int(hit_t)))
    blocker = _point(start, step, hit_t)

    grid = np.full((h, w), bg, dtype=int)

    guide_pts = [_point(start, step, t) for t in range(int(guide_len))]
    source_pts = [_point(start, step, t) for t in source_ts]
    output_pts = [_point(start, step, t) for t in output_ts]
    travel_pts = [_point(start, step, t) for t in range(int(guide_len + source_len), int(hit_t))]

    for r, c in guide_pts:
        grid[int(r), int(c)] = int(guide_color)
    for r, c in source_pts:
        grid[int(r), int(c)] = int(water)

    forbidden_before_hit = set(guide_pts) | set(source_pts) | set(travel_pts)

    blocker_r, blocker_c = blocker
    for row in range(int(blocker_r), h):
        grid[int(row), int(blocker_c)] = int(wall_color)

    base_span = rng.randint(2, 4)
    if side == "left":
        for col in range(int(blocker_c), min(w, int(blocker_c + base_span))):
            grid[h - 1, int(col)] = int(wall_color)
    else:
        for col in range(max(0, int(blocker_c - base_span + 1)), int(blocker_c + 1)):
            grid[h - 1, int(col)] = int(wall_color)

    beyond_cols: list[int]
    if step[1] > 0:
        beyond_cols = list(range(int(blocker_c + 2), w))
    else:
        beyond_cols = list(range(0, int(blocker_c - 1)))
    if beyond_cols:
        span_len = min(len(beyond_cols), rng.randint(2, 3))
        if step[1] > 0:
            cols = beyond_cols[:span_len]
            heights = [rng.randint(1, 3) + i for i in range(span_len)]
        else:
            cols = beyond_cols[-span_len:]
            heights = [rng.randint(1, 3) + (span_len - 1 - i) for i in range(span_len)]
        _paint_bottom_structure(grid, cols, heights, extra_color, forbidden_before_hit)

    if rng.random() < 0.65:
        far_col = w - 1 if step[1] > 0 else 0
        far_height = rng.randint(2, 4)
        _paint_bottom_structure(grid, [far_col], [far_height], extra_color, forbidden_before_hit)

    if any(int(grid[r, c]) != int(bg) for r, c in travel_pts):
        return None
    if any(int(grid[r, c]) != int(wall_color) for r, c in [blocker]):
        return None
    if any(int(grid[r, c]) != int(bg) for r, c in output_pts):
        return None
    if guide_color in set(map(int, grid[h - 1])):
        return None

    expected = grid.copy()
    for r, c in source_pts:
        expected[int(r), int(c)] = int(bg)
    for r, c in output_pts:
        expected[int(r), int(c)] = int(water)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        case = _build_case(rng)
        if case is not None:
            return case
    raise RuntimeError("generate_input: failed after 500 attempts")
