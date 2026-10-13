from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

OBJECT_COLORS = (RED, GREEN, YELLOW, PINK)
CARRIER_TO_SIDE = {BLUE: "right", TEAL: "left"}
SIDES = ("left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _all_objects(grid: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for color in OBJECT_COLORS:
        out.extend(_components_of_color(grid, color))
    return out


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected a grey frame"
    return _bbox(coords)


def _gap_cells(side: str, row: int, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "left":
        c = c0
        return [(row, c), (row - 1, c), (row + 1, c), (row, c - 1), (row, c + 1)]
    c = c1
    return [(row, c), (row - 1, c), (row + 1, c), (row, c - 1), (row, c + 1)]


def _detect_gaps(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[str, list[dict[str, object]]]:
    r0, c0, r1, c1 = frame_bbox
    found: dict[str, list[dict[str, object]]] = {"left": [], "right": []}
    for side, col in (("left", c0), ("right", c1)):
        used: set[tuple[int, int]] = set()
        for row in range(r0 + 1, r1):
            cells = _gap_cells(side, row, frame_bbox)
            if any(not _in_bounds(grid.shape, rr, cc) for rr, cc in cells):
                continue
            if not all(int(grid[rr, cc]) == BLACK for rr, cc in cells):
                continue
            cell_set = set(cells)
            if used & cell_set:
                continue
            found[side].append({"row": int(row), "center": (int(row), int(col)), "cells": cells})
            used |= cell_set
    return found


def _object_info(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], coords: np.ndarray
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    color = int(grid[coords[0, 0], coords[0, 1]])

    carriers = set()
    for r, c in coords:
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r + dr), int(c + dc)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in CARRIER_TO_SIDE:
                carriers.add(int(grid[nr, nc]))
    assert len(carriers) == 1, "each object must touch exactly one carrier color"
    carrier = next(iter(carriers))
    side = CARRIER_TO_SIDE[carrier]

    inner_left = int(frame_bbox[1] + 1)
    inner_right = int(frame_bbox[3] - 1)
    if side == "right":
        distance = int(inner_right - c1)
    else:
        distance = int(r0 * 0 + c0 - inner_left)
    assert distance >= 1, "object must not start already flush"

    return {
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": color,
        "carrier": int(carrier),
        "side": side,
        "distance": int(distance),
        "row_top": int(r0),
        "row_center": float(r0 + r1) / 2.0,
    }


def _assigned_fillers(
    objects: list[dict[str, object]], gaps: dict[str, list[dict[str, object]]]
) -> list[tuple[list[tuple[int, int]], int]]:
    fills: list[tuple[list[tuple[int, int]], int]] = []
    for side in SIDES:
        side_gaps = sorted(gaps[side], key=lambda item: int(item["row"]))
        if not side_gaps:
            continue
        side_objects = [obj for obj in objects if obj["side"] == side]
        if not side_objects:
            continue

        if len(side_objects) == 1:
            obj = side_objects[0]
            gap = min(
                side_gaps,
                key=lambda item: (abs(float(obj["row_center"]) - int(item["row"])), int(item["row"])),
            )
            fills.append((list(gap["cells"]), int(obj["color"])))
            continue

        ranked_objects = sorted(
            side_objects,
            key=lambda obj: (-int(obj["distance"]), int(obj["row_top"])),
        )
        for gap, obj in zip(side_gaps, ranked_objects):
            fills.append((list(gap["cells"]), int(obj["color"])))
    return fills


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each carrier-marked object toward the frame and paint the reached side's notch with that object's color.

    Input structure:
    - Black background with one grey 1-cell-thick rectangular frame near the center.
    - The left and right frame sides may contain 1-2 five-pixel cross-shaped black notches.
    - Inside the frame are 3-5 small monochrome objects in red, green, yellow, or pink, each embedded in a blue or teal rectangular carrier.

    Rule:
    - Use the carrier color to decide the object's horizontal slide direction: blue moves right, teal moves left.
    - Slide each object until its front edge would touch the frame's inner wall, and record that slide distance.
    - On each reached side, fill notches with object colors: a lone arriving object fills the nearest notch; multiple arriving objects are ranked by larger slide distance, then by higher row on ties, and assigned to notches from top to bottom.
    - Clear all carriers and original object pixels, leaving only the grey frame, the recolored notch cells, and black background.

    Key insight:
    - The carrier color does not choose an output color directly; it determines slide direction, which determines the target side and the distance-based priority for coloring that side's notch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _frame_bbox(g)
    gaps = _detect_gaps(g, frame_bbox)
    objects = [_object_info(g, frame_bbox, coords) for coords in _all_objects(g)]
    assert objects, "expected at least one object"

    out = np.zeros_like(g)
    out[g == GREY] = GREY
    for cells, color in _assigned_fillers(objects, gaps):
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _choose_side_counts(n_objects: int, rng: random.Random) -> tuple[int, int]:
    choices = [(left, n_objects - left) for left in range(1, n_objects)]
    choices = [pair for pair in choices if max(pair) >= 2]
    return rng.choice(choices)


def _choose_colors(n_objects: int, rng: random.Random) -> list[int]:
    base = list(OBJECT_COLORS)
    if n_objects <= len(base):
        colors = rng.sample(base, n_objects)
    else:
        colors = base + [rng.choice(base)]
    rng.shuffle(colors)
    return colors


def _sample_distances(max_distances: list[int], rng: random.Random) -> list[int]:
    count = len(max_distances)
    assert count >= 1
    distances = [0] * count
    if count >= 2 and rng.random() < 0.35:
        a, b = rng.sample(range(count), 2)
        shared = rng.randint(2, min(max_distances[a], max_distances[b]))
        distances[a] = shared
        distances[b] = shared
    used: set[int] = set(d for d in distances if d)
    for idx, max_d in enumerate(max_distances):
        if distances[idx]:
            continue
        choices = [d for d in range(2, max_d + 1) if d not in used]
        if not choices:
            choices = list(range(2, max_d + 1))
        distances[idx] = rng.choice(choices)
        used.add(distances[idx])
    return distances


def _choose_gap_rows(
    side_rows: list[int], all_gap_rows: list[int], gap_count: int, rng: random.Random
) -> list[int]:
    side_rows = sorted(side_rows)
    if gap_count == 1:
        pool = side_rows if rng.random() < 0.7 else all_gap_rows
        return [int(rng.choice(pool))]

    if len(side_rows) == 1:
        row = side_rows[0]
        others = [candidate for candidate in all_gap_rows if abs(candidate - row) >= 4]
        others.sort(key=lambda candidate: (-abs(candidate - row), candidate))
        if others:
            return sorted([int(row), int(others[0])])
        raise AssertionError("expected a second distinct lane row")

    pairs = [(a, b) for a in all_gap_rows for b in all_gap_rows if a < b and b - a >= 4]
    assert pairs, "expected at least one valid two-gap placement"
    scored: list[tuple[tuple[int, int], tuple[int, int]]] = []
    top_row = side_rows[0]
    bottom_row = side_rows[-1]
    for a, b in pairs:
        score = (abs(a - top_row) + abs(b - bottom_row), abs(a - top_row), abs(b - bottom_row))
        scored.append((score, (a, b)))
    best_score = min(score for score, _pair in scored)
    best_pairs = [pair for score, pair in scored if score == best_score]
    return list(rng.choice(best_pairs))


def _can_place_two_gaps(side_rows: list[int], all_gap_rows: list[int]) -> bool:
    side_rows = sorted(side_rows)
    if len(side_rows) == 1:
        row = side_rows[0]
        return any(abs(candidate - row) >= 4 for candidate in all_gap_rows if candidate != row)
    return any(a < b and b - a >= 4 for a in all_gap_rows for b in all_gap_rows)


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _draw_gaps(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], gap_rows: dict[str, list[int]]
) -> None:
    for side in SIDES:
        for row in gap_rows[side]:
            for rr, cc in _gap_cells(side, row, frame_bbox):
                grid[rr, cc] = BLACK


def _expected_output_from_meta(meta: dict[str, object]) -> np.ndarray:
    h = int(meta["height"])
    w = int(meta["width"])
    frame_bbox = meta["frame_bbox"]
    gap_rows = meta["gap_rows"]
    objects = meta["objects"]

    out = np.zeros((h, w), dtype=int)
    _draw_frame(out, frame_bbox)
    _draw_gaps(out, frame_bbox, gap_rows)

    gaps = {
        side: [
            {"row": row, "cells": _gap_cells(side, row, frame_bbox)}
            for row in sorted(gap_rows[side])
        ]
        for side in SIDES
    }
    fillers = _assigned_fillers(objects, gaps)
    for cells, color in fillers:
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _check_paths_clear(objects: list[dict[str, object]], frame_bbox: tuple[int, int, int, int]) -> None:
    inner_left = int(frame_bbox[1] + 1)
    inner_right = int(frame_bbox[3] - 1)
    occupied_by_row: dict[int, set[int]] = {}
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        for row in range(int(r0), int(r1) + 1):
            occupied_by_row.setdefault(row, set()).update(range(int(c0), int(c1) + 1))

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if obj["side"] == "right":
            segment = range(int(c1) + 1, inner_right + 1)
        else:
            segment = range(inner_left, int(c0))
        for row in range(int(r0), int(r1) + 1):
            blockers = occupied_by_row[row] - set(range(int(c0), int(c1) + 1))
            assert not (set(segment) & blockers), "horizontal path must be free of other objects"


def _validate_meta(meta: dict[str, object], inp: np.ndarray, expected: np.ndarray) -> None:
    frame_bbox = meta["frame_bbox"]
    objects = meta["objects"]
    gap_rows = meta["gap_rows"]

    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert sum(len(gap_rows[side]) for side in SIDES) >= 2
    assert any(len([obj for obj in objects if obj["side"] == side]) >= 2 for side in SIDES)

    _check_paths_clear(objects, frame_bbox)

    for side in SIDES:
        side_gaps = sorted(gap_rows[side])
        side_objects = [obj for obj in objects if obj["side"] == side]
        assert side_gaps and side_objects, "both sides must matter in generated samples"
        if len(side_objects) == 1 and len(side_gaps) > 1:
            row = float(side_objects[0]["row_center"])
            distances = sorted(abs(row - gap_row) for gap_row in side_gaps)
            assert distances[0] < distances[1], "nearest-notch choice must be unique"

    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output must match intended semantics"
    assert np.count_nonzero(expected != GREY) > 0


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(3, 5)
    total_lanes = n_objects + rng.randint(0, 1)
    frame_inner_h = 3 * total_lanes
    frame_h = frame_inner_h + 2
    frame_w = rng.randint(13, 16)
    frame_inner_w = frame_w - 2

    if frame_inner_w < 6:
        return None

    h = frame_h + rng.randint(4, 7)
    w = frame_w + rng.randint(4, 8)
    if max(h, w) > 30:
        return None

    top = rng.randint(2, h - frame_h - 2)
    left = rng.randint(2, w - frame_w - 2)
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    inner_left = left + 1
    inner_right = left + frame_w - 2

    lane_rows = [top + 2 + 3 * idx for idx in range(total_lanes)]
    gap_candidate_rows = list(range(top + 2, top + frame_h - 2))
    chosen_lanes = sorted(rng.sample(range(total_lanes), n_objects))
    object_rows = [lane_rows[idx] for idx in chosen_lanes]

    left_count, right_count = _choose_side_counts(n_objects, rng)
    sides = ["left"] * left_count + ["right"] * right_count
    rng.shuffle(sides)
    colors = _choose_colors(n_objects, rng)
    widths = [rng.randint(2, 4) for _ in range(n_objects)]

    objects: list[dict[str, object]] = []
    for row, side, color, width in sorted(zip(object_rows, sides, colors, widths), key=lambda item: item[0]):
        max_distance = frame_inner_w - 1 - int(width)
        if max_distance < 2:
            return None
        objects.append(
            {
                "row_top": int(row),
                "row_center": float(row),
                "side": side,
                "color": int(color),
                "width": int(width),
                "max_distance": int(max_distance),
            }
        )

    for side in SIDES:
        side_objects = [obj for obj in objects if obj["side"] == side]
        side_distances = _sample_distances([int(obj["max_distance"]) for obj in side_objects], rng)
        for obj, distance in zip(side_objects, side_distances):
            obj["distance"] = int(distance)
            row = int(obj["row_top"])
            width = int(obj["width"])
            if side == "right":
                c1 = inner_right - int(distance)
                c0 = c1 - width + 1
                right_margin = rng.randint(1, min(2, int(distance) - 1))
                left_margin = rng.randint(1, min(2, c0 - inner_left))
            else:
                c0 = inner_left + int(distance)
                c1 = c0 + width - 1
                left_margin = rng.randint(1, min(2, int(distance) - 1))
                right_margin = rng.randint(1, min(2, inner_right - c1))
            if min(left_margin, right_margin) < 1:
                return None

            obj["bbox"] = (row, int(c0), row, int(c1))
            obj["carrier_bbox"] = (row - 1, int(c0 - left_margin), row + 1, int(c1 + right_margin))
            obj["carrier"] = BLUE if side == "right" else TEAL

    gap_rows: dict[str, list[int]] = {}
    for side in SIDES:
        side_rows = [int(obj["row_top"]) for obj in objects if obj["side"] == side]
        if len(side_rows) == 1:
            want_two = rng.random() < 0.6
        else:
            want_two = rng.random() < 0.65
        gap_count = 2 if want_two and _can_place_two_gaps(side_rows, gap_candidate_rows) else 1
        gap_rows[side] = _choose_gap_rows(side_rows, gap_candidate_rows, gap_count, rng)

    gap_cells = {
        (rr, cc)
        for side in SIDES
        for row in gap_rows[side]
        for rr, cc in _gap_cells(side, row, frame_bbox)
    }
    for obj in objects:
        r0, c0, r1, c1 = obj["carrier_bbox"]
        carrier_cells = {
            (rr, cc)
            for rr in range(int(r0), int(r1) + 1)
            for cc in range(int(c0), int(c1) + 1)
        }
        if carrier_cells & gap_cells:
            return None

    inp = np.zeros((h, w), dtype=int)
    _draw_frame(inp, frame_bbox)
    _draw_gaps(inp, frame_bbox, gap_rows)

    for obj in objects:
        cr0, cc0, cr1, cc1 = obj["carrier_bbox"]
        inp[int(cr0) : int(cr1) + 1, int(cc0) : int(cc1) + 1] = int(obj["carrier"])
    for obj in objects:
        r0, c0, _r1, c1 = obj["bbox"]
        inp[int(r0), int(c0) : int(c1) + 1] = int(obj["color"])

    meta = {
        "height": int(h),
        "width": int(w),
        "frame_bbox": frame_bbox,
        "gap_rows": gap_rows,
        "objects": objects,
    }
    expected = _expected_output_from_meta(meta)
    _validate_meta(meta, inp, expected)
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
