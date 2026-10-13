from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
CYAN = 8
PAIR_COLORS = (BLUE, RED, YELLOW)

_ORIENTATIONS = {
    "down": ((1, -1), (1, 1)),
    "up": ((-1, -1), (-1, 1)),
    "left": ((-1, -1), (1, -1)),
    "right": ((-1, 1), (1, 1)),
}

_GEN_ATTEMPTS = 600
_WEDGE_ATTEMPTS = 160
_GRID_SIDE_MIN = 15
_GRID_SIDE_MAX = 22


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ray_step(apex: tuple[int, int], point: tuple[int, int], direction: tuple[int, int]) -> int | None:
    ar, ac = apex
    pr, pc = point
    dr, dc = direction
    delta_r = pr - ar
    delta_c = pc - ac
    if delta_r == 0 and delta_c == 0:
        return 0
    if delta_r != dr * delta_c:
        return None
    if dr != 0:
        t = delta_r // dr
    else:
        t = delta_c // dc
    if t <= 0:
        return None
    if ar + dr * t != pr or ac + dc * t != pc:
        return None
    return int(t)


def _wedge_mask(
    shape: tuple[int, int],
    apex: tuple[int, int],
    orientation: str,
    height: int,
) -> np.ndarray:
    ar, ac = apex
    rr, cc = np.indices(shape)
    if orientation == "down":
        return (rr >= ar) & (rr <= ar + height) & (np.abs(cc - ac) <= rr - ar)
    if orientation == "up":
        return (rr <= ar) & (rr >= ar - height) & (np.abs(cc - ac) <= ar - rr)
    if orientation == "right":
        return (cc >= ac) & (cc <= ac + height) & (np.abs(rr - ar) <= cc - ac)
    if orientation == "left":
        return (cc <= ac) & (cc >= ac - height) & (np.abs(rr - ar) <= ac - cc)
    raise ValueError(f"unknown orientation {orientation}")


def _candidate_wedges(greens: set[tuple[int, int]], shape: tuple[int, int]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple] = set()
    for apex in greens:
        for orientation, rays in _ORIENTATIONS.items():
            ray_steps = [[], []]
            for point in greens:
                if point == apex:
                    continue
                for idx, direction in enumerate(rays):
                    step = _ray_step(apex, point, direction)
                    if step is not None:
                        ray_steps[idx].append(step)
                        break
            if not ray_steps[0] or not ray_steps[1]:
                continue
            height_left = max(ray_steps[0])
            height_right = max(ray_steps[1])
            if height_left != height_right or height_left < 2:
                continue
            height = int(height_left)
            markers = {apex}
            for point in greens:
                if point == apex:
                    continue
                on_side = False
                for direction in rays:
                    step = _ray_step(apex, point, direction)
                    if step is not None and step <= height:
                        on_side = True
                        break
                if on_side:
                    markers.add(point)
            if not any(_ray_step(apex, p, rays[0]) == height for p in markers if p != apex):
                continue
            if not any(_ray_step(apex, p, rays[1]) == height for p in markers if p != apex):
                continue
            signature = (apex, orientation, height, tuple(sorted(markers)))
            if signature in seen:
                continue
            seen.add(signature)
            mask = _wedge_mask(shape, apex, orientation, height)
            candidates.append(
                {
                    "apex": apex,
                    "orientation": orientation,
                    "height": height,
                    "markers": frozenset(markers),
                    "mask": mask,
                    "bbox": _bbox_coords(np.argwhere(mask)),
                }
            )
    return candidates


def _detect_wedges(grid: np.ndarray) -> list[dict]:
    greens = {tuple(map(int, cell)) for cell in np.argwhere(grid == GREEN)}
    assert 4 <= len(greens) <= 8, "expected 4..8 green markers"
    candidates = _candidate_wedges(greens, grid.shape)
    solutions: list[list[dict]] = []
    for count in (1, 2):
        for combo in combinations(candidates, count):
            used: set[tuple[int, int]] = set()
            valid = True
            for wedge in combo:
                if used & wedge["markers"]:
                    valid = False
                    break
                used |= set(wedge["markers"])
            if valid and used == greens:
                solutions.append(list(combo))
    assert len(solutions) == 1, "green markers must define a unique 1- or 2-wedge cover"
    return solutions[0]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment must be axis-aligned")


def _all_pairings(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[tuple[int, int], tuple[int, int]], ...]]:
    if not coords:
        return [tuple()]
    first = coords[0]
    out = []
    for idx in range(1, len(coords)):
        second = coords[idx]
        if first[0] != second[0] and first[1] != second[1]:
            continue
        pair = tuple(sorted((first, second)))
        remaining = coords[1:idx] + coords[idx + 1 :]
        for rest in _all_pairings(remaining):
            out.append(tuple(sorted((pair,) + rest)))
    return out


def _unique_pairs_for_color(coords: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered = tuple(sorted(coords))
    assert len(ordered) % 2 == 0 and ordered, "each color must have an even non-zero count"
    matchings = {matching for matching in _all_pairings(ordered)}
    assert len(matchings) == 1, "same-color endpoints must admit a unique row/column pairing"
    return [tuple(pair) for pair in next(iter(matchings))]


def _scene_analysis(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
    wedges = _detect_wedges(grid)
    permitted = np.zeros_like(grid, dtype=bool)
    membership = np.zeros_like(grid, dtype=int)
    for idx, wedge in enumerate(wedges, start=1):
        mask = wedge["mask"]
        assert not np.any(permitted & mask), "wedge interiors must stay disjoint"
        permitted |= mask
        membership[mask] = idx

    pairs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREEN}):
        coords = [tuple(map(int, cell)) for cell in np.argwhere(grid == color)]
        for a, b in _unique_pairs_for_color(coords):
            cells = _segment_cells(a, b)
            pairs.append(
                {
                    "color": int(color),
                    "endpoints": (a, b),
                    "cells": tuple(cells),
                    "axis": "h" if a[0] == b[0] else "v",
                    "allowed": all(bool(permitted[r, c]) for r, c in cells),
                    "inside_a": bool(permitted[a]),
                    "inside_b": bool(permitted[b]),
                    "wedge_a": int(membership[a]),
                    "wedge_b": int(membership[b]),
                }
            )
    return wedges, permitted, membership, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wedge-permitted regions and connect only same-color endpoint pairs whose full row/column path stays inside them.

    Input structure:
    - The background is black.
    - Green marker pixels lie on the diagonal sides of one or two 45-degree triangular wedges; together they uniquely determine each wedge's apex, side directions, and height.
    - Blue, red, and yellow endpoint pixels appear in same-color row/column pairs, with black gaps between paired endpoints.

    Rule:
    - Reconstruct every wedge from the green markers and treat the wedge interior, including the green side markers, as the permitted region.
    - For each non-green color, uniquely pair same-color endpoints that share a row or a column.
    - If every cell on that pair's straight row/column segment lies inside the permitted region, fill the black interior cells of the segment with that color; otherwise leave the pair unchanged.
    - After all valid connections, paint every remaining black permitted-region cell cyan and keep the green markers green.

    Key insight:
    - Green markers do not directly draw the answer; they define a geometric filter, so identical-looking endpoint pairs behave differently depending on whether their entire path stays inside the wedge-permitted area.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _wedges, permitted, _membership, pairs = _scene_analysis(g)

    for pair in pairs:
        if not pair["allowed"]:
            continue
        for r, c in pair["cells"][1:-1]:
            if out[r, c] == BG:
                out[r, c] = pair["color"]

    out[permitted & (out == BG)] = CYAN
    out[g == GREEN] = GREEN
    return out


def _expanded_bbox(mask: np.ndarray, gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox_coords(np.argwhere(mask))
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _boxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _sample_apex(shape: tuple[int, int], orientation: str, height: int) -> tuple[int, int] | None:
    h, w = shape
    if orientation == "down":
        r_values = range(1, h - height - 1)
        c_values = range(height + 1, w - height - 1)
    elif orientation == "up":
        r_values = range(height + 1, h - 1)
        c_values = range(height + 1, w - height - 1)
    elif orientation == "right":
        r_values = range(height + 1, h - height - 1)
        c_values = range(1, w - height - 1)
    else:
        r_values = range(height + 1, h - height - 1)
        c_values = range(height + 1, w - 1)
    r_values = list(r_values)
    c_values = list(c_values)
    if not r_values or not c_values:
        return None
    return random.choice(r_values), random.choice(c_values)


def _marker_points(
    apex: tuple[int, int],
    orientation: str,
    height: int,
    target_count: int,
) -> set[tuple[int, int]]:
    rays = _ORIENTATIONS[orientation]
    markers = {apex}
    for direction in rays:
        dr, dc = direction
        markers.add((apex[0] + dr * height, apex[1] + dc * height))
    optional = []
    for ray_idx, direction in enumerate(rays):
        dr, dc = direction
        for step in range(1, height):
            optional.append((ray_idx, step, (apex[0] + dr * step, apex[1] + dc * step)))
    random.shuffle(optional)
    for _ray_idx, _step, point in optional:
        if len(markers) >= target_count:
            break
        markers.add(point)
    return markers


def _place_wedges(shape: tuple[int, int], num_wedges: int) -> list[dict] | None:
    if num_wedges == 1:
        target_counts = [random.randint(4, 6)]
    else:
        target_counts = [random.choice((3, 4)), random.choice((3, 4))]
        while sum(target_counts) < 6 or sum(target_counts) > 8:
            target_counts = [random.choice((3, 4)), random.choice((3, 4))]

    wedges = []
    for idx in range(num_wedges):
        target_count = target_counts[idx]
        placed = False
        for _ in range(_WEDGE_ATTEMPTS):
            orientation = random.choice(tuple(_ORIENTATIONS))
            max_height = 5 if num_wedges == 1 else 4
            height = random.randint(3, max_height)
            apex = _sample_apex(shape, orientation, height)
            if apex is None:
                continue
            mask = _wedge_mask(shape, apex, orientation, height)
            bbox = _expanded_bbox(mask, gap=2)
            if any(not _boxes_disjoint(bbox, other["expanded_bbox"]) for other in wedges):
                continue
            markers = _marker_points(apex, orientation, height, target_count)
            if len(markers) != target_count:
                continue
            wedges.append(
                {
                    "apex": apex,
                    "orientation": orientation,
                    "height": height,
                    "mask": mask,
                    "markers": markers,
                    "expanded_bbox": bbox,
                }
            )
            placed = True
            break
        if not placed:
            return None

    green_grid = np.zeros(shape, dtype=int)
    for wedge in wedges:
        for r, c in wedge["markers"]:
            green_grid[r, c] = GREEN
    try:
        detected = _detect_wedges(green_grid)
    except AssertionError:
        return None

    expected_markers = {frozenset(w["markers"]) for w in wedges}
    detected_markers = {w["markers"] for w in detected}
    if expected_markers != detected_markers:
        return None
    return wedges


def _enumerate_segments(
    shape: tuple[int, int],
    permitted: np.ndarray,
    membership: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    segments: list[dict] = []
    for r in range(h):
        for c0 in range(w):
            for delta in range(3, 9):
                c1 = c0 + delta
                if c1 >= w:
                    break
                cells = tuple((r, c) for c in range(c0, c1 + 1))
                if any(cell in forbidden for cell in cells):
                    continue
                a, b = cells[0], cells[-1]
                segments.append(
                    {
                        "endpoints": (a, b),
                        "cells": cells,
                        "axis": "h",
                        "allowed": all(bool(permitted[x, y]) for x, y in cells),
                        "inside_a": bool(permitted[a]),
                        "inside_b": bool(permitted[b]),
                        "wedge_a": int(membership[a]),
                        "wedge_b": int(membership[b]),
                    }
                )
    for c in range(w):
        for r0 in range(h):
            for delta in range(3, 9):
                r1 = r0 + delta
                if r1 >= h:
                    break
                cells = tuple((r, c) for r in range(r0, r1 + 1))
                if any(cell in forbidden for cell in cells):
                    continue
                a, b = cells[0], cells[-1]
                segments.append(
                    {
                        "endpoints": (a, b),
                        "cells": cells,
                        "axis": "v",
                        "allowed": all(bool(permitted[x, y]) for x, y in cells),
                        "inside_a": bool(permitted[a]),
                        "inside_b": bool(permitted[b]),
                        "wedge_a": int(membership[a]),
                        "wedge_b": int(membership[b]),
                    }
                )
    return segments


def _build_specs(num_wedges: int) -> list[dict]:
    repeated = random.choice(PAIR_COLORS)
    others = [color for color in PAIR_COLORS if color != repeated]
    repeated_allowed = random.choice((True, False))
    forced_opposite = random.choice(others)
    base_status = {
        repeated: repeated_allowed,
        forced_opposite: not repeated_allowed,
        next(color for color in others if color != forced_opposite): random.choice((True, False)),
    }
    repeated_axis = random.choice(("h", "v"))
    specs = [
        {"color": repeated, "allowed": base_status[repeated], "axis": repeated_axis, "blocked_mode": None},
        {
            "color": repeated,
            "allowed": not base_status[repeated],
            "axis": "v" if repeated_axis == "h" else "h",
            "blocked_mode": None,
        },
    ]
    for color in others:
        specs.append(
            {
                "color": color,
                "allowed": base_status[color],
                "axis": random.choice(("h", "v")),
                "blocked_mode": None,
            }
        )
    blocked_specs = [spec for spec in specs if not spec["allowed"]]
    if num_wedges == 2:
        random.choice(blocked_specs)["blocked_mode"] = "cross_wedges"
    else:
        random.choice(blocked_specs)["blocked_mode"] = "single_inside"
    specs.sort(key=lambda spec: (spec["allowed"], spec["blocked_mode"] is None))
    return specs


def _segment_matches_spec(segment: dict, spec: dict) -> bool:
    if segment["allowed"] != spec["allowed"]:
        return False
    if segment["axis"] != spec["axis"]:
        return False
    if spec["blocked_mode"] == "cross_wedges":
        return (
            segment["inside_a"]
            and segment["inside_b"]
            and segment["wedge_a"] > 0
            and segment["wedge_b"] > 0
            and segment["wedge_a"] != segment["wedge_b"]
        )
    if spec["blocked_mode"] == "single_inside":
        return segment["inside_a"] ^ segment["inside_b"]
    if segment["allowed"]:
        return segment["inside_a"] and segment["inside_b"] and segment["wedge_a"] == segment["wedge_b"]
    return segment["inside_a"] or segment["inside_b"]


def _assign_segments(specs: list[dict], segments: list[dict]) -> list[dict] | None:
    chosen: dict[int, dict] = {}
    occupied: set[tuple[int, int]] = set()

    def backtrack() -> bool:
        if len(chosen) == len(specs):
            return True
        candidate_options = None
        spec_index = None
        for idx, spec in enumerate(specs):
            if idx in chosen:
                continue
            options = [
                segment
                for segment in segments
                if _segment_matches_spec(segment, spec) and not any(cell in occupied for cell in segment["cells"])
            ]
            if not options:
                return False
            if candidate_options is None or len(options) < len(candidate_options):
                candidate_options = options
                spec_index = idx
        assert candidate_options is not None and spec_index is not None
        random.shuffle(candidate_options)
        for segment in candidate_options:
            chosen[spec_index] = segment
            occupied.update(segment["cells"])
            if backtrack():
                return True
            for cell in segment["cells"]:
                occupied.remove(cell)
            del chosen[spec_index]
        return False

    if not backtrack():
        return None
    return [{**specs[idx], **chosen[idx]} for idx in range(len(specs))]


def _paint_input(shape: tuple[int, int], wedges: list[dict], assigned_segments: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for wedge in wedges:
        for r, c in wedge["markers"]:
            grid[r, c] = GREEN
    for spec in assigned_segments:
        (r0, c0), (r1, c1) = spec["endpoints"]
        grid[r0, c0] = spec["color"]
        grid[r1, c1] = spec["color"]
    return grid


def _semantic_checks(grid: np.ndarray, expected_wedge_count: int) -> bool:
    try:
        wedges, permitted, membership, pairs = _scene_analysis(grid)
    except AssertionError:
        return False
    if len(wedges) != expected_wedge_count:
        return False
    if {pair["color"] for pair in pairs} != set(PAIR_COLORS):
        return False
    connectable = [pair for pair in pairs if pair["allowed"]]
    blocked = [pair for pair in pairs if not pair["allowed"]]
    if not connectable or not blocked:
        return False
    per_color = {}
    for pair in pairs:
        per_color.setdefault(pair["color"], []).append(pair["allowed"])
    if not any(len(states) == 2 and set(states) == {False, True} for states in per_color.values()):
        return False
    if not np.any(permitted):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.any(out == CYAN):
        return False

    fill_only = grid.copy()
    fill_only[permitted & (fill_only == BG)] = CYAN
    fill_only[grid == GREEN] = GREEN
    if np.array_equal(fill_only, out):
        return False

    connect_all = grid.copy()
    for pair in pairs:
        for r, c in pair["cells"][1:-1]:
            if connect_all[r, c] == BG:
                connect_all[r, c] = pair["color"]
    connect_all[permitted & (connect_all == BG)] = CYAN
    connect_all[grid == GREEN] = GREEN
    if np.array_equal(connect_all, out):
        return False

    if expected_wedge_count == 2 and not any(
        pair["inside_a"]
        and pair["inside_b"]
        and pair["wedge_a"] != pair["wedge_b"]
        and not pair["allowed"]
        for pair in pairs
    ):
        return False
    if expected_wedge_count == 1 and not any((pair["inside_a"] ^ pair["inside_b"]) and not pair["allowed"] for pair in pairs):
        return False
    if len({pair["axis"] for pair in pairs}) < 2:
        return False
    return np.array_equal(main(grid), out) and np.all((membership == 0) | permitted)


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX), random.randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX))
    num_wedges = 2 if random.random() < 0.9 else 1
    wedges = _place_wedges(shape, num_wedges)
    if wedges is None:
        return None

    green_markers = set().union(*(wedge["markers"] for wedge in wedges))
    permitted = np.zeros(shape, dtype=bool)
    membership = np.zeros(shape, dtype=int)
    for idx, wedge in enumerate(wedges, start=1):
        permitted |= wedge["mask"]
        membership[wedge["mask"]] = idx

    segments = _enumerate_segments(shape, permitted, membership, green_markers)
    specs = _build_specs(num_wedges)
    assigned = _assign_segments(specs, segments)
    if assigned is None:
        return None

    grid = _paint_input(shape, wedges, assigned)
    if not _semantic_checks(grid, num_wedges):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid wedge-filter connection input")
