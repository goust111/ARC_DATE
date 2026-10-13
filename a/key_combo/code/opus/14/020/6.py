from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LINE_COLOR = 1
SIDE_B_COLOR = 2
SIDE_A_COLOR = 3
INPUT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)

STEP_OPTIONS = (
    (1, 1),
    (1, -1),
    (1, 2),
    (1, -2),
    (2, 1),
    (2, -1),
    (1, 3),
    (1, -3),
    (3, 1),
    (3, -1),
)

GRID_SIDE_RANGE = (10, 15)
SINGLETON_COUNT_OPTIONS = (4, 5, 6)
GENERATION_ATTEMPTS = 800


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sort_coords(coords: np.ndarray) -> np.ndarray:
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def _extract_markers(grid: np.ndarray) -> np.ndarray:
    markers = np.argwhere(grid == LINE_COLOR)
    assert len(markers) == 3, "expected exactly three blue markers"
    markers = _sort_coords(markers.astype(int))
    deltas = np.diff(markers, axis=0)
    assert np.array_equal(deltas[0], deltas[1]), "markers must be equally spaced and collinear"
    dr, dc = map(int, deltas[0])
    assert dr != 0 and dc != 0, "the arithmetic line must be slanted"
    return markers


def _extend_line(shape: tuple[int, int], first_marker: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    r, c = first_marker
    while _in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc

    line: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        line.append((r, c))
        r += dr
        c += dc
    return line


def _cross_value(point: tuple[int, int], anchor: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = point
    r0, c0 = anchor
    dr, dc = step
    return (r - r0) * dc - (c - c0) * dr


def _nearest_line_index(point: tuple[int, int], line: list[tuple[int, int]]) -> tuple[int, int]:
    distances = [abs(point[0] - r) + abs(point[1] - c) for r, c in line]
    best = min(distances)
    indices = [i for i, dist in enumerate(distances) if dist == best]
    assert len(indices) == 1, "nearest line pixel must be unique"
    return indices[0], best


def _nearest_line_index_or_none(point: tuple[int, int], line: list[tuple[int, int]]) -> tuple[int, int] | None:
    distances = [abs(point[0] - r) + abs(point[1] - c) for r, c in line]
    best = min(distances)
    indices = [i for i, dist in enumerate(distances) if dist == best]
    if len(indices) != 1:
        return None
    return indices[0], best


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _assign_singletons(
    singletons: list[tuple[int, int, int]],
    line: list[tuple[int, int]],
    anchor: tuple[int, int],
    step: tuple[int, int],
) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    for r, c, color in singletons:
        side_value = _cross_value((r, c), anchor, step)
        assert side_value != 0, "singleton cannot lie on the divider line"
        target_idx, distance = _nearest_line_index((r, c), line)
        records.append(
            {
                "r": r,
                "c": c,
                "color": color,
                "side": 1 if side_value > 0 else -1,
                "target_idx": target_idx,
                "distance": distance,
            }
        )

    records.sort(key=lambda rec: (rec["distance"], rec["r"], rec["c"], rec["color"]))
    occupied: set[int] = set()
    for rec in records:
        place_idx = rec["target_idx"]
        while place_idx in occupied:
            place_idx -= 1
        assert place_idx >= 0, "collision resolution ran out of line pixels"
        rec["place_idx"] = place_idx
        occupied.add(place_idx)
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the arithmetic blue marker line, snap each singleton onto it, and recolor by the singleton's original side.

    Input structure:
    - The background is black.
    - Exactly three blue pixels mark an equally spaced slanted arithmetic line.
    - Several other non-black singletons of arbitrary colors lie on both sides of that line.

    Rule:
    - Sort the three blue markers in row-major order and use the repeated difference vector as the line step.
    - Extend that arithmetic line in both directions until the next step would leave the grid, and paint the full extended line blue.
    - Clear every original singleton, then move each one to its unique nearest line pixel by Manhattan distance.
    - Recolor a moved singleton green if its original position had positive cross product against the line direction, or red if the cross product was negative.
    - If several singletons want the same line pixel, keep the nearer one there and shift the others along the line toward the first marker.

    Key insight:
    - The blue arithmetic line is both the geometric divider that decides the final color and the snapping target that receives every moved singleton.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = _extract_markers(g)
    first_marker = tuple(map(int, markers[0]))
    step = tuple(map(int, markers[1] - markers[0]))
    line = _extend_line(g.shape, first_marker, step)

    singletons = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BACKGROUND) & (g != LINE_COLOR))
    ]
    assert 4 <= len(singletons) <= 6, "expected 4 to 6 singleton pixels"

    assignments = _assign_singletons(singletons, line, first_marker, step)

    out = np.full_like(g, BACKGROUND)
    for r, c in line:
        out[r, c] = LINE_COLOR
    for rec in assignments:
        rr, cc = line[rec["place_idx"]]
        out[rr, cc] = SIDE_A_COLOR if rec["side"] > 0 else SIDE_B_COLOR
    return out


def _line_candidates(shape: tuple[int, int], step: tuple[int, int]) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    lines: list[list[tuple[int, int]]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            line = _extend_line(shape, (r, c), step)
            key = tuple(line)
            if len(line) >= 6 and key not in seen:
                seen.add(key)
                lines.append(line)
    return lines


def _candidate_records(
    shape: tuple[int, int],
    line: list[tuple[int, int]],
    anchor: tuple[int, int],
    step: tuple[int, int],
) -> dict[int, list[dict[str, int]]]:
    line_set = set(line)
    records = {1: [], -1: []}
    for r in range(shape[0]):
        for c in range(shape[1]):
            point = (r, c)
            if point in line_set:
                continue
            side_value = _cross_value(point, anchor, step)
            if side_value == 0:
                continue
            nearest = _nearest_line_index_or_none(point, line)
            if nearest is None:
                continue
            target_idx, distance = nearest
            if distance < 2:
                continue
            if min(_chebyshev(point, lp) for lp in line) < 2:
                continue
            records[1 if side_value > 0 else -1].append(
                {"r": r, "c": c, "target_idx": target_idx, "distance": distance}
            )
    return records


def _choose_side_counts(total: int) -> tuple[int, int]:
    options = [(a, total - a) for a in range(2, total - 1) if total - a >= 2]
    return random.choice(options)


def _backtrack_pick(
    side_records: dict[int, list[dict[str, int]]],
    needed: dict[int, int],
) -> list[dict[str, int]] | None:
    ordered_sides = [1, -1] if needed[1] <= needed[-1] else [-1, 1]
    for side in ordered_sides:
        if needed[side] > 0:
            focus_side = side
            break
    else:
        return []

    candidates = side_records[focus_side][:]
    random.shuffle(candidates)
    for rec in candidates:
        if rec.get("used"):
            continue
        next_needed = dict(needed)
        next_needed[focus_side] -= 1
        snapshot = []
        rejected = False
        for side in (1, -1):
            for other in side_records[side]:
                if other.get("chosen"):
                    if other["distance"] == rec["distance"] or other["target_idx"] == rec["target_idx"]:
                        rejected = True
                        break
                    if _chebyshev((other["r"], other["c"]), (rec["r"], rec["c"])) <= 1:
                        rejected = True
                        break
            if rejected:
                break
        if rejected:
            continue

        rec["chosen"] = True
        snapshot.append(rec)
        result = _backtrack_pick(side_records, next_needed)
        if result is not None:
            return [rec] + result
        for item in snapshot:
            item.pop("chosen", None)
    return None


def _pick_singletons(
    side_records: dict[int, list[dict[str, int]]],
    total: int,
) -> list[dict[str, int]] | None:
    need_a, need_b = _choose_side_counts(total)
    chosen = _backtrack_pick(side_records, {1: need_a, -1: need_b})
    if chosen is None:
        return None
    return chosen


def _assign_input_colors(chosen: list[dict[str, int]]) -> None:
    side_a = [rec for rec in chosen if rec["side"] == 1]
    side_b = [rec for rec in chosen if rec["side"] == -1]
    color_pool = [color for color in INPUT_COLORS if color not in (2, 3)]
    random.shuffle(color_pool)

    side_a[0]["color"] = 2
    side_b[0]["color"] = 3
    remaining = [rec for rec in chosen if "color" not in rec]
    extra_colors = color_pool[: len(remaining)]
    for rec, color in zip(remaining, extra_colors):
        rec["color"] = color


def _build_input(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    chosen: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r, c in markers:
        grid[r, c] = LINE_COLOR
    for rec in chosen:
        grid[rec["r"], rec["c"]] = rec["color"]
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    markers = _extract_markers(grid)
    first_marker = tuple(map(int, markers[0]))
    step = tuple(map(int, markers[1] - markers[0]))
    line = _extend_line(grid.shape, first_marker, step)
    marker_indices = [line.index(tuple(map(int, marker))) for marker in markers]
    assert marker_indices[0] > 0 and marker_indices[-1] < len(line) - 1, "markers must leave extension room on both ends"

    singletons = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != BACKGROUND) & (grid != LINE_COLOR))
    ]
    assert 4 <= len(singletons) <= 6

    sides = []
    distances = set()
    targets = set()
    for i, (r, c, _color) in enumerate(singletons):
        side_value = _cross_value((r, c), first_marker, step)
        assert side_value != 0
        sides.append(1 if side_value > 0 else -1)
        target_idx, distance = _nearest_line_index((r, c), line)
        assert distance >= 2
        assert distance not in distances
        assert target_idx not in targets
        distances.add(distance)
        targets.add(target_idx)
        assert min(_chebyshev((r, c), lp) for lp in line) >= 2
        for j in range(i):
            pr, pc, _ = singletons[j]
            assert _chebyshev((r, c), (pr, pc)) >= 2
    assert sides.count(1) >= 2 and sides.count(-1) >= 2

    out = main(grid)
    assignments = _assign_singletons(singletons, line, first_marker, step)
    occupied = {rec["place_idx"]: rec for rec in assignments}
    assert out.shape == grid.shape
    for r, c, _ in singletons:
        assert out[r, c] == BACKGROUND
    for idx, point in enumerate(line):
        expected = LINE_COLOR
        if idx in occupied:
            expected = SIDE_A_COLOR if occupied[idx]["side"] > 0 else SIDE_B_COLOR
        assert out[point] == expected
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) not in set(line):
                assert out[r, c] == BACKGROUND


def _generate_once() -> np.ndarray | None:
    shape = (
        random.randrange(*GRID_SIDE_RANGE),
        random.randrange(*GRID_SIDE_RANGE),
    )
    step = random.choice(STEP_OPTIONS)
    lines = _line_candidates(shape, step)
    if not lines:
        return None

    line = random.choice(lines)
    start_idx = random.randrange(1, len(line) - 3)
    markers = line[start_idx : start_idx + 3]
    anchor = markers[0]

    side_records = _candidate_records(shape, line, anchor, step)
    if len(side_records[1]) < 2 or len(side_records[-1]) < 2:
        return None

    total = random.choice(SINGLETON_COUNT_OPTIONS)
    chosen = _pick_singletons(side_records, total)
    if chosen is None:
        return None
    for rec in chosen:
        rec["side"] = 1 if _cross_value((rec["r"], rec["c"]), anchor, step) > 0 else -1
    _assign_input_colors(chosen)

    grid = _build_input(shape, markers, chosen)
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arithmetic-divider puzzle")
