# Combo variant of cb2d8a2c (index 5)
# combo_spec: D3/1+D8/0
# Name: 蓝色间距编码距离+蓝色位置编码方向
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and small trace:
#    - In canonical top-start orientation, the input still contains red/blue 1-cell-thick
#      horizontal obstacle bars plus one green border start, but the bars are now floating
#      rather than wall-attached. That change is necessary: if a bar touches a side wall,
#      the bypass side is geometrically forced and D8/0 would be visually redundant.
#    - Each bar carries exactly three blue pixels. The route no longer counts them.
#      Instead, the minimum Manhattan clearance radius is:
#          radius = maximum number of non-blue cells between consecutive blue pixels.
#      Example: blue columns [5, 7, 12] yield gaps 1 and 4, so radius = 4.
#    - The same three blue pixels also encode bypass direction. If their centroid is left
#      of the bar midpoint, the path detours on the left; if it is right of the midpoint,
#      the path detours on the right.
#    - Relative to the original task, the visible difference is immediate:
#      two bars can both have three blue pixels, yet one forces a wider berth because one
#      blue gap is larger, and the green bend can go to opposite sides depending on which
#      end the blue cluster favors.
# b) Complexity calibration:
#    - Radius stays in 1..4, matching the original scale.
#    - Every generated sample uses 2-3 bars, includes both left and right detours, and
#      includes at least two distinct radii. This gives repeated support for both fused
#      subrules without exploding the number of cases a human must infer.
# c) Hidden-assumption audit against the original solver:
#    - Original assumption 1: side-wall attachment determines the only legal turn side.
#      Broken by D8/0, so the generator switches to floating bars with room on both sides.
#    - Original assumption 2: radius = blue_count. Broken by D3/1, so `main()` decodes
#      radius from sorted blue gaps instead of counting blue cells.
#    - Original assumption 3: generic shortest-path search is enough. Here the generator
#      guarantees a bar-by-bar monotone detour layout, so `main()` explicitly reconstructs
#      the orthogonal route implied by the decoded side and radius, then validates it
#      against the true Manhattan keep-out mask.
# d) Placement legality:
#    - Bars are single-row 4-connected components, separated enough in the main axis so
#      each halo can be entered and exited monotonically:
#          row_gap(i,i+1) >= radius_i + radius_{i+1} + 1  (in empty rows between bars)
#    - Each floating bar keeps at least `radius + 1` free cells on both left and right, so
#      either detour side is physically available and the decoded side is a real choice.
#    - While generating bars, the current path column is forced to lie inside every bar's
#      inflated horizontal span, guaranteeing that every bar actually affects the route.
# e) Visual observability:
#    - Blue count is constant (three per bar), so counting cannot solve the puzzle.
#    - The largest internal blue gap is directly visible and controls how far the green
#      path stands off from the bar.
#    - The blue cluster's bias toward the left or right end is also directly visible and
#      controls the handedness of the bend.
# f) Visual cleanliness:
#    - No extra colors or legend regions are introduced.
#    - Objects remain simple: azure background, red bars with three blue markers, and one
#      green orthogonal path. Bars are spaced so their halos do not visually merge.
# g) Change inferability / anti-degeneracy:
#    - The generator rejects samples whose output matches a counterfactual solver that uses
#      the old blue-count distance rule, and also rejects samples whose output matches a
#      counterfactual solver that ignores blue-position direction and simply detours to the
#      nearer side. This prevents silent collapse back to the source behavior.

from collections import deque
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
RADIUS_LO, RADIUS_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500
MAX_LAYOUT_RETRIES = 80


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


def _decoded_bar_infos(grid: np.ndarray) -> list[dict[str, object]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    infos: list[dict[str, object]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        pts = np.array(comp, dtype=int)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        assert r0 == r1, "canonical bars must be horizontal"

        blue_cols = sorted(int(c) for r, c in pts if grid[int(r), int(c)] == BLUE)
        assert len(blue_cols) >= 2, "each bar must contain at least two blue pixels"
        gaps = [b - a - 1 for a, b in zip(blue_cols, blue_cols[1:])]
        radius = max(gaps)
        assert RADIUS_LO <= radius <= RADIUS_HI, "decoded radius out of range"

        midpoint = (c0 + c1) / 2.0
        centroid = sum(blue_cols) / len(blue_cols)
        side = "left" if centroid < midpoint else "right"
        cells = [(r0, c) for c in range(c0, c1 + 1)]
        infos.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "blue_cols": blue_cols,
                "blue_count": len(blue_cols),
                "radius": radius,
                "side": side,
                "cells": cells,
            }
        )
    infos.sort(key=lambda info: int(info["row"]))
    return infos


def _radius_for(info: dict[str, object], radius_mode: str) -> int:
    if radius_mode == "decoded":
        return int(info["radius"])
    if radius_mode == "count":
        return int(info["blue_count"])
    raise ValueError(f"unknown radius_mode: {radius_mode}")


def _choose_side(current_col: int, info: dict[str, object], radius: int, side_mode: str) -> str:
    if side_mode == "decoded":
        return str(info["side"])

    left_col = int(info["c0"]) - radius - 1
    right_col = int(info["c1"]) + radius + 1
    left_cost = abs(current_col - left_col)
    right_cost = abs(current_col - right_col)
    return "left" if left_cost <= right_cost else "right"


def _build_blocked_mask(shape: tuple[int, int], infos: list[dict[str, object]], radius_mode: str) -> np.ndarray:
    h, w = shape
    obstacle = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)

    for info in infos:
        radius = _radius_for(info, radius_mode)
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in info["cells"]:
            obstacle[r, c] = True
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            blocked[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    return obstacle | blocked


def _build_canonical_path(
    grid: np.ndarray,
    infos: list[dict[str, object]],
    radius_mode: str = "decoded",
    side_mode: str = "decoded",
) -> list[tuple[int, int]]:
    h, w = grid.shape
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the start on the top edge"

    path: list[tuple[int, int]] = [(sr, sc)]
    current_r, current_col = sr, sc

    for info in infos:
        radius = _radius_for(info, radius_mode)
        top_row = int(info["row"]) - radius - 1
        bottom_row = int(info["row"]) + radius + 1
        assert top_row >= current_r, "bars must be ordered with enough vertical spacing"

        _append_segment(path, current_r, current_col, top_row, current_col)
        current_r = top_row

        left_hit = int(info["c0"]) - radius
        right_hit = int(info["c1"]) + radius
        if left_hit <= current_col <= right_hit:
            side = _choose_side(current_col, info, radius, side_mode)
            side_col = int(info["c0"]) - radius - 1 if side == "left" else int(info["c1"]) + radius + 1
            assert 0 <= side_col < w, "detour column must stay inside the grid"
            _append_segment(path, current_r, current_col, current_r, side_col)
            current_col = side_col

        _append_segment(path, current_r, current_col, bottom_row, current_col)
        current_r = bottom_row

    _append_segment(path, current_r, current_col, h - 1, current_col)
    return path


def _paint_solution(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _solve_canonical(
    grid: np.ndarray,
    radius_mode: str = "decoded",
    side_mode: str = "decoded",
) -> np.ndarray:
    infos = _decoded_bar_infos(grid)
    path = _build_canonical_path(grid, infos, radius_mode=radius_mode, side_mode=side_mode)

    blocked = _build_blocked_mask(grid.shape, infos, radius_mode=radius_mode)
    start = path[0]
    blocked[start] = False
    for r, c in path:
        assert not blocked[r, c], "decoded path entered a forbidden cell"

    return _paint_solution(grid, path)


def _safe_solve_canonical(
    grid: np.ndarray,
    radius_mode: str = "decoded",
    side_mode: str = "decoded",
) -> np.ndarray | None:
    try:
        return _solve_canonical(grid, radius_mode=radius_mode, side_mode=side_mode)
    except Exception:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border green path across the grid while each obstacle bar uses blue spacing to set clearance and blue bias to choose the bypass side.

    Input structure:
    - The background is azure (8), with 1-cell-thick red bars containing a few blue (1) marker pixels.
    - Exactly one green (3) start pixel lies on a border edge.
    - In canonical top-start view, every obstacle bar is horizontal and floating, leaving traversable space on both its left and right.

    Rule:
    - For each bar, sort its blue pixels along the bar. The minimum clearance radius equals the largest number of non-blue cells between consecutive blue pixels.
    - The same blue pixels also set direction: if their center of mass leans toward the left end, the path detours on the left; if it leans toward the right end, the path detours on the right.
    - The green route advances orthogonally from the start edge to the opposite edge, making one rectangular detour around each bar's Manhattan keep-out halo, then all blue pixels become red.

    Key insight:
    The blue markers are no longer counted. Their internal spacing determines how far away the path must stay, while their overall left/right bias determines which side the path chooses, so the same three blue pixels encode two independent control signals.
    """
    canonical, edge = _to_canonical_top(np.asarray(grid, dtype=int))
    out = _solve_canonical(canonical, radius_mode="decoded", side_mode="decoded")
    return _from_canonical_top(out, edge)


def _min_main_len(radii: tuple[int, ...]) -> int:
    row = radii[0] + 1
    for a, b in zip(radii, radii[1:]):
        row += a + b + 2
    return row + radii[-1] + 2


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


def _length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, radius + 4, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - 2 * (radius + 1), (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _blue_offsets(length: int, radius: int, side: str) -> tuple[int, int, int]:
    if side == "left":
        offsets = (0, 2, radius + 3)
    else:
        offsets = (length - radius - 4, length - 3, length - 1)
    assert 0 <= offsets[0] < offsets[1] < offsets[2] < length
    return offsets


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, radius: int, side: str) -> None:
    colors = np.full(length, RED, dtype=int)
    for off in _blue_offsets(length, radius, side):
        colors[off] = BLUE
    grid[row, c0 : c0 + length] = colors


def _choose_radii(count: int, cross_size: int, main_hi: int) -> tuple[int, ...] | None:
    choices = []
    for radii in itertools.product(range(RADIUS_LO, RADIUS_HI + 1), repeat=count):
        if len(set(radii)) < 2:
            continue
        if _min_main_len(radii) > main_hi:
            continue
        if all(_length_bounds(cross_size, radius) is not None for radius in radii):
            choices.append(radii)
    return random.choice(choices) if choices else None


def _choose_sides(count: int) -> tuple[str, ...]:
    choices = [combo for combo in itertools.product(("left", "right"), repeat=count) if len(set(combo)) == 2]
    return random.choice(choices)


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    sides: tuple[str, ...],
) -> np.ndarray | None:
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 1 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    start_margin = max(2, cross_size // 4)
    start_lo = start_margin
    start_hi = cross_size - start_margin - 1
    if start_lo > start_hi:
        start_lo, start_hi = 1, cross_size - 2

    for _ in range(MAX_LAYOUT_RETRIES):
        grid = np.full((main_size, cross_size), BG, dtype=int)
        start_col = random.randint(start_lo, start_hi)
        current_col = start_col
        placed = True

        for row, radius, side in zip(rows, radii, sides):
            bounds = _length_bounds(cross_size, radius)
            if bounds is None:
                placed = False
                break

            lengths = list(range(bounds[0], bounds[1] + 1))
            random.shuffle(lengths)
            found = False

            for length in lengths:
                c0_lo = max(radius + 1, current_col - (length - 1 + radius))
                c0_hi = min(cross_size - length - radius - 1, current_col + radius)
                if c0_lo > c0_hi:
                    continue

                candidates = list(range(c0_lo, c0_hi + 1))
                random.shuffle(candidates)
                for c0 in candidates:
                    c1 = c0 + length - 1
                    next_col = c0 - radius - 1 if side == "left" else c1 + radius + 1
                    if not (0 <= next_col < cross_size):
                        continue
                    _paint_bar(grid, row, c0, length, radius, side)
                    current_col = next_col
                    found = True
                    break
                if found:
                    break

            if not found:
                placed = False
                break

        if not placed:
            continue

        grid[0, start_col] = GREEN
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
    cross_size = random.randint(cross_lo, cross_hi)
    radii = _choose_radii(count, cross_size, main_hi)
    if radii is None:
        return None
    sides = _choose_sides(count)

    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    canonical = _canonical_grid(main_size, cross_size, radii, sides)
    if canonical is None:
        return None

    actual = _solve_canonical(canonical, radius_mode="decoded", side_mode="decoded")
    count_baseline = _safe_solve_canonical(canonical, radius_mode="count", side_mode="decoded")
    side_baseline = _safe_solve_canonical(canonical, radius_mode="decoded", side_mode="nearest")
    if count_baseline is not None and np.array_equal(actual, count_baseline):
        return None
    if side_baseline is not None and np.array_equal(actual, side_baseline):
        return None

    inp = _from_canonical_top(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(out, main(inp)):
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
