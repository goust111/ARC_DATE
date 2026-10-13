from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

TURN_DELTA = {
    RED: 1,
    BLUE: -1,
    GREEN: 0,
    YELLOW: 2,
}
TURN_COLORS = tuple(TURN_DELTA)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}

GEN_ATTEMPTS = 500
_GENERATION_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_passable(color: int) -> bool:
    return color == GREY or color in TURN_DELTA


def _turn(direction: tuple[int, int], color: int) -> tuple[int, int]:
    idx = DIR_TO_INDEX[direction]
    return DIRS[(idx + TURN_DELTA[color]) % 4]


def _find_unique_color(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) == 1, f"expected exactly one cell of color {color}, got {len(coords)}"
    return tuple(map(int, coords[0]))


def _start_from_marker(grid: np.ndarray, marker: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    mr, mc = marker
    for dr, dc in DIRS:
        nr, nc = mr + dr, mc + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            candidates.append(((nr, nc), (dr, dc)))
    assert len(candidates) == 1, f"marker at {marker} should touch exactly one grey endpoint"
    return candidates[0]


def _trace_signal(grid: np.ndarray, source_marker: tuple[int, int]) -> dict:
    pos, direction = _start_from_marker(grid, source_marker)
    grey_cells = [pos]
    turn_events: list[tuple[tuple[int, int], int]] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction[0], direction[1])
        if state in seen_states:
            return {
                "loop": True,
                "grey_cells": grey_cells,
                "turn_events": turn_events,
                "target_marker": None,
                "target_color": None,
            }
        seen_states.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not _in_bounds(grid.shape, nr, nc):
            return {
                "loop": False,
                "grey_cells": grey_cells,
                "turn_events": turn_events,
                "target_marker": None,
                "target_color": None,
            }

        color = int(grid[nr, nc])
        if color == GREY:
            pos = (nr, nc)
            grey_cells.append(pos)
            continue
        if color in TURN_DELTA:
            pos = (nr, nc)
            turn_events.append((pos, color))
            direction = _turn(direction, color)
            continue
        if color in (MAROON, ORANGE):
            return {
                "loop": False,
                "grey_cells": grey_cells,
                "turn_events": turn_events,
                "target_marker": (nr, nc),
                "target_color": color,
            }
        return {
            "loop": False,
            "grey_cells": grey_cells,
            "turn_events": turn_events,
            "target_marker": None,
            "target_color": None,
        }


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the orange signal through the colored crossroad network, paint its travelled grey path azure, and light the reached terminal marker.

    Input structure:
    - The background is the dominant black color, and grey one-cell-wide horizontal and vertical segments form a rectilinear path network.
    - Some four-way crossings have their center grey pixel replaced by one of red, blue, green, or yellow.
    - Exactly one orange marker sits just outside a path endpoint, and the other marked endpoints use maroon.

    Rule:
    - Start from the orange marker, enter the adjacent grey endpoint, and move in the direction from the marker into the path.
    - Recolor every grey path pixel visited by the signal to azure.
    - When the signal reaches a colored crossing, keep the crossing color unchanged but update the travel direction using red=right turn, blue=left turn, green=straight, and yellow=reverse.
    - When the signal reaches an endpoint whose outward neighbor is a maroon marker, recolor that marker to orange and leave every other pixel unchanged.

    Key insight:
    - The colored crossing centers are directional control nodes: once their relative turn behavior is applied while tracing the unique signal route, the lit destination is determined by the resulting endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source_marker = _find_unique_color(g, ORANGE)
    trace = _trace_signal(g, source_marker)
    assert not trace["loop"], "signal should not loop forever"
    assert trace["target_marker"] is not None, "signal must end at a marked endpoint"
    assert trace["target_color"] == MAROON, "signal must light a maroon terminal, not return to the source"

    out = g.copy()
    for r, c in trace["grey_cells"]:
        out[r, c] = AZURE

    tr, tc = trace["target_marker"]
    out[tr, tc] = ORANGE
    return out


def _sample_spaced_positions(start: int, end: int, count: int, min_gap: int = 2) -> list[int] | None:
    adjusted_end = end - (count - 1) * (min_gap - 1)
    if adjusted_end < start:
        return None
    base = sorted(random.sample(range(start, adjusted_end + 1), count))
    return [value + idx * (min_gap - 1) for idx, value in enumerate(base)]


def _build_grid(
    height: int,
    width: int,
    top_end: int,
    bottom_end: int,
    left_end: int,
    right_end: int,
    h_rows: list[int],
    v_cols: list[int],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)

    for r in h_rows:
        grid[r, left_end : right_end + 1] = GREY
    for c in v_cols:
        grid[top_end : bottom_end + 1, c] = GREY
    for coord, color in colors.items():
        grid[coord] = color

    for r in h_rows:
        grid[r, left_end - 1] = MAROON
        grid[r, right_end + 1] = MAROON
    for c in v_cols:
        grid[top_end - 1, c] = MAROON
        grid[bottom_end + 1, c] = MAROON

    return grid


def _endpoint_markers(
    top_end: int,
    bottom_end: int,
    left_end: int,
    right_end: int,
    h_rows: list[int],
    v_cols: list[int],
) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for r in h_rows:
        markers.append((r, left_end - 1))
        markers.append((r, right_end + 1))
    for c in v_cols:
        markers.append((top_end - 1, c))
        markers.append((bottom_end + 1, c))
    return markers


def _trace_candidates(
    grid: np.ndarray,
    markers: list[tuple[int, int]],
    required_turn: int,
) -> list[tuple[tuple[int, int], dict]]:
    matches: list[tuple[tuple[int, int], dict]] = []
    for marker in markers:
        trace = _trace_signal(grid, marker)
        if trace["loop"]:
            continue
        target = trace["target_marker"]
        if target is None or target == marker:
            continue
        if trace["target_color"] != MAROON:
            continue
        if len(trace["turn_events"]) < 2:
            continue
        path_turn_colors = [color for _coord, color in trace["turn_events"]]
        if required_turn not in path_turn_colors:
            continue
        matches.append((marker, trace))
    return matches


def _random_turn_assignment(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    assert len(coords) >= 4
    palette = [RED, BLUE, GREEN, YELLOW]
    while len(palette) < len(coords):
        palette.append(random.choice(TURN_COLORS))
    random.shuffle(palette)
    return dict(zip(coords, palette))


def _gen_attempt(required_turn: int) -> np.ndarray | None:
    num_h = random.randint(2, 4)
    num_v = random.randint(2, 4)

    vspan = random.randint(2 * num_h + 2, min(18, 2 * num_h + 9))
    hspan = random.randint(2 * num_v + 2, min(18, 2 * num_v + 9))

    top_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    left_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)

    height = top_margin + vspan + 3 + bottom_margin
    width = left_margin + hspan + 3 + right_margin
    if height > 30 or width > 30:
        return None

    top_end = top_margin + 1
    bottom_end = top_end + vspan
    left_end = left_margin + 1
    right_end = left_end + hspan

    h_rows = _sample_spaced_positions(top_end + 2, bottom_end - 2, num_h, min_gap=2)
    v_cols = _sample_spaced_positions(left_end + 2, right_end - 2, num_v, min_gap=2)
    if h_rows is None or v_cols is None:
        return None

    intersection_coords = [(r, c) for r in h_rows for c in v_cols]
    color_map = _random_turn_assignment(intersection_coords)
    base_grid = _build_grid(
        height=height,
        width=width,
        top_end=top_end,
        bottom_end=bottom_end,
        left_end=left_end,
        right_end=right_end,
        h_rows=h_rows,
        v_cols=v_cols,
        colors=color_map,
    )

    markers = _endpoint_markers(top_end, bottom_end, left_end, right_end, h_rows, v_cols)
    candidates = _trace_candidates(base_grid, markers, required_turn)
    if not candidates:
        return None

    source_marker, _trace = random.choice(candidates)
    grid = base_grid.copy()
    grid[source_marker] = ORANGE

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATION_COUNT

    required_turns = [RED, BLUE, GREEN, YELLOW]
    required_turn = required_turns[_GENERATION_COUNT % len(required_turns)]
    _GENERATION_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(required_turn)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-network puzzle")
