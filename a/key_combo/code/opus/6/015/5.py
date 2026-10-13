from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 9
PALETTE = (1, 2, 3, 4, 5, 8)
RETRY_LIMIT = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _make_rect(name: str, color: int, r0: int, c0: int, h: int, w: int) -> dict:
    return {
        "name": str(name),
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
        "area": int(h * w),
    }


def _copy_rect(rect: dict) -> dict:
    return _make_rect(rect["name"], rect["color"], rect["r0"], rect["c0"], rect["h"], rect["w"])


def _paint_rect(grid: np.ndarray, rect: dict) -> None:
    grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(rect["color"])


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(np.asarray(grid) == MARKER)
    assert pts.shape[0] == 1, "expected exactly one maroon direction marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert r in (0, h - 1) or c in (0, w - 1), "marker must lie on the border"
    return r, c


def _rotation_to_down(marker: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = map(int, marker)
    h, w = map(int, shape)
    if r == h - 1:
        return 0
    if r == 0:
        return 2
    if c == 0:
        return 1
    if c == w - 1:
        return 3
    raise AssertionError("marker must be on exactly one edge")


def _rotate_back(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(np.asarray(grid), -int(k)).copy()


def _rect_center(rect: dict) -> tuple[float, float]:
    return ((rect["r0"] + rect["r1"]) / 2.0, (rect["c0"] + rect["c1"]) / 2.0)


def _dist2_to_grid_center(rect: dict, shape: tuple[int, int]) -> float:
    cr, cc = _rect_center(rect)
    gr = (shape[0] - 1) / 2.0
    gc = (shape[1] - 1) / 2.0
    return (cr - gr) ** 2 + (cc - gc) ** 2


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, MARKER) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert area == len(cells), "each foreground object must be a solid rectangle"
            rects.append(_make_rect(f"rect_{len(rects)}", color, r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    return rects


def _nearest_obstacle(mover: dict, stopped: list[dict]) -> tuple[int, dict] | None:
    candidates: list[tuple[int, int, int, int, dict]] = []
    mover_mid = mover["c0"] + mover["c1"]
    for idx, obstacle in enumerate(stopped):
        if obstacle["r0"] <= mover["r1"]:
            continue
        if not _interval_overlap(mover["c0"], mover["c1"], obstacle["c0"], obstacle["c1"]):
            continue
        gap = obstacle["r0"] - mover["r1"] - 1
        mid_delta = abs((obstacle["c0"] + obstacle["c1"]) - mover_mid)
        candidates.append((gap, obstacle["r0"], mid_delta, idx, obstacle))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:-1])
    _gap, _r0, _mid_delta, idx, obstacle = candidates[0]
    return int(idx), obstacle


def _split_obstacle(obstacle: dict, mover: dict, shape: tuple[int, int], split_id: int) -> tuple[list[dict], str]:
    h, w = map(int, shape)
    left_w = mover["c0"] - obstacle["c0"]
    right_w = obstacle["c1"] - mover["c1"]
    assert left_w > 0 and right_w > 0, "split path must leave two non-empty obstacle halves"

    left = _make_rect(
        f"{obstacle['name']}_L{split_id}",
        obstacle["color"],
        obstacle["r0"],
        obstacle["c0"] - 1,
        obstacle["h"],
        left_w,
    )
    right = _make_rect(
        f"{obstacle['name']}_R{split_id}",
        obstacle["color"],
        obstacle["r0"],
        mover["c1"] + 2,
        obstacle["h"],
        right_w,
    )
    assert 0 <= left["c0"] and right["c1"] < w, "shifted split halves must stay in bounds"

    left_d = _dist2_to_grid_center(left, shape)
    right_d = _dist2_to_grid_center(right, shape)
    if left_d <= right_d:
        keep_side = "left"
        right["color"] = int(mover["color"])
    else:
        keep_side = "right"
        left["color"] = int(mover["color"])
    return [left, right], keep_side


def _simulate_down(rects: list[dict], shape: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    h, _w = map(int, shape)
    order = sorted(rects, key=lambda rect: (-rect["area"], rect["r0"], rect["c0"], rect["color"]))
    stopped: list[dict] = []
    events: list[dict] = []
    split_id = 0

    for src in order:
        mover = _copy_rect(src)
        while True:
            hit = _nearest_obstacle(mover, stopped)
            if hit is None:
                shift = h - 1 - mover["r1"]
                mover["r0"] += shift
                mover["r1"] += shift
                break

            obstacle_idx, obstacle = hit
            gap = obstacle["r0"] - mover["r1"] - 1
            mover["r0"] += gap
            mover["r1"] += gap

            if mover["h"] < obstacle["h"]:
                events.append(
                    {
                        "type": "stop",
                        "mover": mover["name"],
                        "obstacle": obstacle["name"],
                        "mover_h": mover["h"],
                        "obstacle_h": obstacle["h"],
                    }
                )
                break

            split_id += 1
            obstacle = stopped.pop(obstacle_idx)
            pieces, keep_side = _split_obstacle(obstacle, mover, shape, split_id)
            stopped.extend(pieces)
            events.append(
                {
                    "type": "split",
                    "mover": mover["name"],
                    "obstacle": obstacle["name"],
                    "mover_h": mover["h"],
                    "obstacle_h": obstacle["h"],
                    "keep_side": keep_side,
                }
            )

        stopped.append(mover)

    stopped.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["name"]))
    return stopped, events


def _render_down(shape: tuple[int, int], rects: list[dict], marker_c: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        _paint_rect(out, rect)
    out[shape[0] - 1, marker_c] = MARKER
    return out


def _solve_down(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    marker_r, marker_c = _find_marker(g)
    assert marker_r == h - 1, "canonical form expects the direction marker on the bottom edge"

    rects = _parse_rectangles(g)
    assert 4 <= len(rects) <= 6, "expected four to six rectangles"
    assert len({rect["color"] for rect in rects}) == len(rects), "input rectangle colors must be unique"
    areas = [rect["area"] for rect in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct"
    assert all(rect["w"] > rect["h"] for rect in rects), "canonical form expects horizontal bars"

    final_rects, _events = _simulate_down(rects, g.shape)
    return _render_down(g.shape, final_rects, marker_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank solid rectangles by area, slide them toward the marker edge, and split thinner-cross-section obstacles into recolored side halves.

    Input structure:
    - The background is black and exactly one maroon border pixel marks the common slide direction.
    - There are four to six non-touching monochrome solid rectangles, each in a unique foreground color.
    - All rectangles share one bar orientation; after rotating toward the marker direction they become horizontal bars with distinct areas.
    - No foreground component uses the marker color, and each component is a single solid rectangle.

    Rule:
    - Conceptually rotate the grid so the maroon marker points downward, then sort the rectangles by area from largest to smallest.
    - Process the bars in that rank order: each one slides downward until it reaches the border or a previously stopped bar.
    - On contact, compare bar thicknesses along the slide axis. If the mover is thinner, it stops flush against the obstacle.
    - Otherwise the mover passes through: remove the obstacle cells directly under the mover span, shift the left and right obstacle remnants one cell outward, keep the half nearer the grid center in the old color, recolor the farther half with the mover color, and continue sliding.
    - After all bars are processed, rotate the result back and keep the maroon marker pixel present in the output.

    Key insight:
    - Area controls processing order, but the collision branch depends on cross-section thickness, and split recoloring depends on which remnant half ends up closer to the grid center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker = _find_marker(g)
    k = _rotation_to_down(marker, g.shape)
    canonical = np.rot90(g, k).copy()
    solved = _solve_down(canonical)
    return _rotate_back(solved, k)


def _marker_candidates(width: int) -> list[int]:
    center = (width - 1) / 2.0
    out = []
    for c in range(1, width - 1):
        if abs(c - center) > 2.5:
            continue
        left_width = c - 2
        right_width = width - c - 3
        if max(left_width, right_width) >= 6:
            out.append(c)
    return out


def _choose_config(main_width: int) -> dict | None:
    choices = []
    if main_width >= 6:
        w_a = random.choice([w for w in (6, 7) if w <= main_width])
        w_b = random.choice([w for w in (4, 3) if 1 <= 1 and w <= w_a - 2])
        w_c = random.choice([w for w in (w_a, w_a - 1) if w >= 5])
        d_choices = [w for w in (4, 3) if w <= w_c - 2]
        for w_d in d_choices:
            rects = {
                "A": (2, w_a),
                "B": (2, w_b),
                "C": (1, w_c),
                "D": (1, w_d),
            }
            areas = [h * w for h, w in rects.values()]
            if len(set(areas)) == 4 and areas[0] > areas[1] > areas[2] > areas[3]:
                choices.append(rects)
    if main_width >= 7:
        rects = {
            "A": (2, 7),
            "B": (3, 4),
            "C": (1, 7),
            "D": (2, 3),
        }
        choices.append(rects)
    if not choices:
        return None
    return random.choice(choices)


def _span_inside(lo: int, hi: int, width: int) -> tuple[int, int]:
    start = random.randint(lo, hi - width + 1)
    return start, start + width - 1


def _contained_span(container: tuple[int, int], width: int) -> tuple[int, int]:
    lo, hi = container
    min_start = lo + 1
    max_start = hi - width
    assert min_start <= max_start
    start = random.randint(min_start, max_start)
    return start, start + width - 1


def _place_rects_for_chain(
    shape: tuple[int, int],
    marker_c: int,
    main_side: str,
    config: dict,
    colors: list[int],
) -> list[dict] | None:
    h, w = shape
    if main_side == "left":
        main_lo, main_hi = 1, marker_c - 2
    else:
        main_lo, main_hi = marker_c + 2, w - 2
    main_width = main_hi - main_lo + 1

    h_a, w_a = config["A"]
    h_b, w_b = config["B"]
    h_c, w_c = config["C"]
    h_d, w_d = config["D"]
    if max(w_a, w_c) > main_width:
        return None

    span_a = _span_inside(main_lo, main_hi, w_a)
    span_b = _contained_span(span_a, w_b)

    c_lo_min = max(main_lo, span_b[1] - w_c + 1)
    c_lo_max = min(span_b[0], main_hi - w_c + 1)
    if c_lo_min > c_lo_max:
        return None
    c0_c = random.randint(c_lo_min, c_lo_max)
    span_c = (c0_c, c0_c + w_c - 1)
    span_d = _contained_span(span_c, w_d)

    top_margin = random.randint(0, 1)
    gap_dc = random.randint(1, 2)
    gap_cb = random.randint(1, 2)
    gap_ba = random.randint(1, 2)
    r0_d = top_margin
    r0_c = r0_d + h_d + gap_dc
    r0_b = r0_c + h_c + gap_cb
    r0_a = r0_b + h_b + gap_ba
    if r0_a + h_a > h - 2:
        return None

    rects = [
        _make_rect("A", colors[0], r0_a, span_a[0], h_a, w_a),
        _make_rect("B", colors[1], r0_b, span_b[0], h_b, w_b),
        _make_rect("C", colors[2], r0_c, span_c[0], h_c, w_c),
        _make_rect("D", colors[3], r0_d, span_d[0], h_d, w_d),
    ]
    return rects


def _extra_candidates(max_width: int, used_areas: set[int]) -> list[tuple[int, int]]:
    out = []
    for h, w in ((1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4)):
        if w > h and w <= max_width and h * w not in used_areas:
            out.append((h, w))
    return out


def _add_extras(
    rects: list[dict],
    shape: tuple[int, int],
    marker_c: int,
    main_side: str,
    colors: list[int],
) -> list[dict]:
    h, w = shape
    if main_side == "left":
        extra_lo, extra_hi = marker_c + 1, w - 1
    else:
        extra_lo, extra_hi = 0, marker_c - 1
    extra_width = extra_hi - extra_lo + 1
    if extra_width < 2:
        return rects

    target_count = random.randint(0, 2)
    if target_count == 0:
        return rects

    occupied: list[tuple[int, int]] = []
    used_areas = {rect["area"] for rect in rects}
    next_color = 4
    for _ in range(target_count):
        if next_color >= len(colors):
            break
        candidates = _extra_candidates(extra_width, used_areas)
        random.shuffle(candidates)
        placed = False
        for h_e, w_e in candidates:
            starts = list(range(extra_lo, extra_hi - w_e + 2))
            random.shuffle(starts)
            for c0 in starts:
                c1 = c0 + w_e - 1
                if any(not (c1 + 1 < lo or hi + 1 < c0) for lo, hi in occupied):
                    continue
                r0 = random.randint(0, max(0, h - h_e - 3))
                rects.append(_make_rect(f"X{len(occupied)}", colors[next_color], r0, c0, h_e, w_e))
                occupied.append((c0, c1))
                used_areas.add(h_e * w_e)
                next_color += 1
                placed = True
                break
            if placed:
                break
    return rects


def _build_canonical_input() -> np.ndarray | None:
    width = random.randint(14, 18)
    marker_options = _marker_candidates(width)
    if not marker_options:
        return None
    marker_c = random.choice(marker_options)

    sides = []
    if marker_c - 2 >= 6:
        sides.append("left")
    if width - marker_c - 3 >= 6:
        sides.append("right")
    if not sides:
        return None
    main_side = random.choice(sides)
    main_width = (marker_c - 2) if main_side == "left" else (width - marker_c - 3)
    config = _choose_config(main_width)
    if config is None:
        return None

    base_height = 2 + 1 + 3 + 2 + 6
    height = random.randint(max(14, base_height), 18)
    n_rects = 4 + random.randint(0, 2)
    colors = random.sample(PALETTE, n_rects)

    rects = _place_rects_for_chain((height, width), marker_c, main_side, config, colors)
    if rects is None:
        return None
    if n_rects > 4:
        rects = _add_extras(rects, (height, width), marker_c, main_side, colors)
        if len(rects) < n_rects:
            return None
        rects = rects[:n_rects]

    grid = np.zeros((height, width), dtype=int)
    for rect in rects:
        _paint_rect(grid, rect)
    grid[height - 1, marker_c] = MARKER

    final_rects, events = _simulate_down(rects, grid.shape)
    if len(events) < 2:
        return None
    if sum(event["type"] == "split" for event in events) < 1:
        return None
    if sum(event["type"] == "stop" for event in events) < 1:
        return None

    expected = _render_down(grid.shape, final_rects, marker_c)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _rotate_input(grid: np.ndarray) -> np.ndarray:
    k = random.randrange(4)
    return np.rot90(np.asarray(grid), -k).copy()


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        canonical = _build_canonical_input()
        if canonical is None:
            continue
        grid = _rotate_input(canonical)
        try:
            out = main(grid)
        except Exception:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
