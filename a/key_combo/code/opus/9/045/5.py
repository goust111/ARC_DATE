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
ORANGE = 7

MARKER_INFO = {
    "nw": (BLUE, (-1, -1)),
    "ne": (RED, (-1, 2)),
    "sw": (GREEN, (2, -1)),
    "se": (YELLOW, (2, 2)),
}
REVERSE_MARKERS = {"nw", "se"}
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _collect_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _detect_squares(grid: np.ndarray) -> list[tuple[int, int]]:
    squares: list[tuple[int, int]] = []
    for comp in _collect_components(grid, GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(comp) == 4, "each grey object must be a 2x2 square"
        assert r1 == r0 + 1 and c1 == c0 + 1, "grey object must fill a 2x2 bbox"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey square must be solid"
        squares.append((r0, c0))
    squares.sort()
    return squares


def _marker_cells(top_left: tuple[int, int]) -> dict[str, tuple[int, int, int]]:
    r, c = top_left
    return {
        kind: (r + dr, c + dc, color)
        for kind, (color, (dr, dc)) in MARKER_INFO.items()
    }


def _build_marker_map(shape: tuple[int, int], squares: list[tuple[int, int]], grid: np.ndarray):
    marker_map: dict[tuple[int, int], tuple[str, int]] = {}
    square_cells: set[tuple[int, int]] = set()
    zones: list[tuple[int, int, int, int]] = []
    for r, c in squares:
        assert 1 <= r <= shape[0] - 4 and 1 <= c <= shape[1] - 4, "square must leave room for all four markers"
        zone = (r - 1, c - 1, r + 2, c + 2)
        for zr0, zc0, zr1, zc1 in zones:
            assert zone[2] < zr0 or zr1 < zone[0] or zone[3] < zc0 or zc1 < zone[1], "square zones must not overlap"
        zones.append(zone)
        for rr in (r, r + 1):
            for cc in (c, c + 1):
                square_cells.add((rr, cc))
        for kind, (mr, mc, color) in _marker_cells((r, c)).items():
            assert _in_bounds(shape, mr, mc), "marker must stay inside the grid"
            assert int(grid[mr, mc]) == BLACK, "marker cells must be empty in the input"
            assert (mr, mc) not in marker_map, "marker cells must be unique"
            marker_map[(mr, mc)] = (kind, color)
    return marker_map, square_cells


def _reflect(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind in REVERSE_MARKERS:
        return -dr, -dc
    if kind == "ne":
        return dr, -dc
    if kind == "sw":
        return -dr, dc
    raise AssertionError(f"unknown marker kind: {kind}")


def _trace_ray(
    shape: tuple[int, int],
    source: tuple[int, int],
    marker_map: dict[tuple[int, int], tuple[str, int]],
    square_cells: set[tuple[int, int]],
) -> dict:
    current = source
    direction = (1, 1)
    segment_color = GREY
    path: list[tuple[int, int, int]] = []
    bounces: list[dict] = []
    visited_order: list[tuple[int, int]] = []
    stop_reason = ""

    while True:
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(shape, nr, nc):
            stop_reason = "boundary"
            break
        if (nr, nc) in marker_map:
            kind, hit_color = marker_map[(nr, nc)]
            bounces.append(
                {
                    "marker": (nr, nc),
                    "kind": kind,
                    "incoming": direction,
                    "pivot": current,
                    "next_color": hit_color,
                }
            )
            if len(bounces) == 5:
                stop_reason = "bounce_limit"
                break
            direction = _reflect(direction, kind)
            segment_color = hit_color
            continue
        assert (nr, nc) not in square_cells, "ray must only bounce on decorated corners, not enter a grey block"
        current = (nr, nc)
        path.append((nr, nc, segment_color))
        visited_order.append((nr, nc))

    return {
        "path": path,
        "visited_order": visited_order,
        "bounces": bounces,
        "stop": current,
        "stop_reason": stop_reason,
    }


def _render_output(
    grid: np.ndarray,
    squares: list[tuple[int, int]],
    source: tuple[int, int],
    trace: dict,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for square in squares:
        for _kind, (r, c, color) in _marker_cells(square).items():
            out[r, c] = color
    for r, c, color in trace["path"]:
        if (r, c) != source:
            out[r, c] = color
    out[-1, trace["stop"][1]] = YELLOW
    out[source] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decorate grey 2x2 blocks, bounce a diagonal ray off the corner markers, recolor each post-bounce segment, and project the stop column to the bottom row.

    Input structure:
    - The grid has a black background, one orange single-pixel ray source, and 3-5 separated grey 2x2 solid squares.
    - Every grey square leaves its four diagonal outside-corner cells empty so those cells can become markers.
    - The last row is black in the input and contains no pre-existing foreground pixels.

    Rule:
    - Keep the orange source and every grey square in place, then decorate each square with fixed corner markers: NW blue, NE red, SW green, SE yellow.
    - Emit one ray from the orange source in the down-right diagonal direction; paint each traversed cell grey until a marker would be entered.
    - When the next step is a marker, do not paint over the marker; instead bounce at the previous cell: NW/SE markers reverse both direction components, NE flips only the horizontal component, and SW flips only the vertical component.
    - After each bounce, recolor all later ray cells with the color of the marker that caused that bounce; stop at the grid edge or immediately after the fifth bounce, then paint one yellow projection pixel on the bottom row in the stop column.

    Key insight:
    - The grey squares matter only through their virtual corner decoration: each decorated corner is both a geometric reflector and the color source for the next ray segment, so the final bottom projection is determined by the full bounce trace rather than by any one object alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert np.all(g[-1] == BLACK), "the last row must be empty in the input"

    source_cells = np.argwhere(g == ORANGE)
    assert len(source_cells) == 1, "expected exactly one orange source"
    source = tuple(map(int, source_cells[0]))

    allowed = {BLACK, GREY, ORANGE}
    assert set(map(int, np.unique(g))).issubset(allowed), "input may contain only black, grey, and orange"

    squares = _detect_squares(g)
    assert 3 <= len(squares) <= 5, "expected 3-5 grey squares"
    marker_map, square_cells = _build_marker_map(g.shape, squares, g)
    assert source not in square_cells and source not in marker_map, "source must be separate from every square"
    assert source[0] < g.shape[0] - 1, "source must not lie on the bottom row"

    trace = _trace_ray(g.shape, source, marker_map, square_cells)
    return _render_output(g, squares, source, trace)


def _square_from_marker(kind: str, marker: tuple[int, int]) -> tuple[int, int]:
    r, c = marker
    if kind == "nw":
        return r + 1, c + 1
    if kind == "ne":
        return r + 1, c - 2
    if kind == "sw":
        return r - 2, c + 1
    if kind == "se":
        return r - 2, c - 2
    raise AssertionError(f"unknown marker kind: {kind}")


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = start
    dr, dc = direction
    row_steps = shape[0] - 1 - r if dr > 0 else r
    col_steps = shape[1] - 1 - c if dc > 0 else c
    return min(row_steps, col_steps)


def _zone_of_square(square: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = square
    return r - 1, c - 1, r + 2, c + 2


def _disjoint(zone: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    return zone[2] < other[0] or other[2] < zone[0] or zone[3] < other[1] or other[3] < zone[1]


def _plan_case() -> tuple[np.ndarray, list[dict]] | None:
    height = random.randint(16, 22)
    width = random.randint(16, 22)
    source = (random.randint(1, max(1, height // 2 - 3)), random.randint(1, max(1, width // 4)))
    total_squares = random.randint(3, 5)
    bounce_count = random.choices([2, 3, 4, 5], weights=[2, 4, 3, 1])[0]
    bounce_count = min(bounce_count, total_squares)

    used_squares: list[tuple[int, int]] = []
    used_zones: list[tuple[int, int, int, int]] = []
    reserved: set[tuple[int, int]] = {source}
    planned_bounces: list[dict] = []
    current = source
    direction = (1, 1)

    for bounce_idx in range(bounce_count):
        remaining = bounce_count - bounce_idx - 1
        kinds = ["ne", "sw"]
        if bounce_idx == bounce_count - 1 and random.random() < 0.3:
            kinds += ["nw", "se"]
        candidates: list[dict] = []
        for kind in kinds:
            for distance in range(2, 8):
                marker = (current[0] + direction[0] * distance, current[1] + direction[1] * distance)
                pivot = (current[0] + direction[0] * (distance - 1), current[1] + direction[1] * (distance - 1))
                square = _square_from_marker(kind, marker)
                sr, sc = square
                if not (1 <= sr <= height - 4 and 1 <= sc <= width - 4):
                    continue
                zone = _zone_of_square(square)
                if any(not _disjoint(zone, other) for other in used_zones):
                    continue
                markers = _marker_cells(square)
                square_cells = {(sr, sc), (sr + 1, sc), (sr, sc + 1), (sr + 1, sc + 1)}
                segment_cells = [
                    (current[0] + direction[0] * step, current[1] + direction[1] * step)
                    for step in range(1, distance)
                ]
                if not all(_in_bounds((height, width), r, c) for r, c in segment_cells + [marker]):
                    continue
                blocked = square_cells | {(r, c) for r, c, _ in markers.values()}
                if any(cell in reserved or cell in blocked for cell in segment_cells):
                    continue
                if marker in reserved or marker in square_cells:
                    continue
                other_markers = {(r, c) for mk, (r, c, _clr) in markers.items() if mk != kind}
                if other_markers & (reserved | set(segment_cells)):
                    continue
                new_dir = _reflect(direction, kind)
                if remaining > 0 and _steps_to_edge((height, width), pivot, new_dir) < 2:
                    continue
                candidates.append(
                    {
                        "kind": kind,
                        "marker": marker,
                        "pivot": pivot,
                        "square": square,
                        "zone": zone,
                        "segment_cells": segment_cells,
                        "new_dir": new_dir,
                    }
                )
        if not candidates:
            return None
        choice = random.choice(candidates)
        used_squares.append(choice["square"])
        used_zones.append(choice["zone"])
        reserved.update(choice["segment_cells"])
        planned_bounces.append(choice)
        current = choice["pivot"]
        direction = choice["new_dir"]

    trace_cells = set(reserved)
    extra_needed = total_squares - len(used_squares)
    extra_candidates = [(r, c) for r in range(1, height - 3) for c in range(1, width - 3)]
    random.shuffle(extra_candidates)
    extras: list[tuple[int, int]] = []
    for square in extra_candidates:
        if len(extras) == extra_needed:
            break
        zone = _zone_of_square(square)
        if any(not _disjoint(zone, other) for other in used_zones):
            continue
        markers = _marker_cells(square)
        square_cells = {(square[0], square[1]), (square[0] + 1, square[1]), (square[0], square[1] + 1), (square[0] + 1, square[1] + 1)}
        occupied = square_cells | {(r, c) for r, c, _ in markers.values()}
        if occupied & trace_cells:
            continue
        extras.append(square)
        used_zones.append(zone)

    if len(extras) != extra_needed:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c in used_squares + extras:
        grid[r : r + 2, c : c + 2] = GREY
    grid[source] = ORANGE
    return grid, planned_bounces


def _valid_generated_case(grid: np.ndarray, planned_bounces: list[dict]) -> bool:
    squares = _detect_squares(grid)
    marker_map, square_cells = _build_marker_map(grid.shape, squares, grid)
    source = tuple(map(int, np.argwhere(grid == ORANGE)[0]))
    trace = _trace_ray(grid.shape, source, marker_map, square_cells)

    if len(trace["bounces"]) != len(planned_bounces):
        return False
    if len(trace["bounces"]) < 2:
        return False
    if [b["marker"] for b in trace["bounces"]] != [b["marker"] for b in planned_bounces]:
        return False
    if [b["kind"] for b in trace["bounces"]] != [b["kind"] for b in planned_bounces]:
        return False
    if len(trace["path"]) < 6:
        return False

    visits = trace["visited_order"]
    repeated = len(set(visits)) != len(visits)
    if repeated and not any(b["kind"] in REVERSE_MARKERS for b in trace["bounces"]):
        return False

    colors_used = {color for _r, _c, color in trace["path"]}
    if len(colors_used) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if len(np.unique(out)) < 4:
        return False
    if np.count_nonzero(out[-1] == YELLOW) != 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        planned = _plan_case()
        if planned is None:
            continue
        grid, planned_bounces = planned
        try:
            if _valid_generated_case(grid, planned_bounces):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid arc case")
