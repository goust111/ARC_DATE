from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
PINK = 6
MAROON = 9

BAND_COLORS = (RED, BLUE, YELLOW)
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_inside_interior(cell: tuple[int, int], frame_box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = frame_box
    return r0 < r < r1 and c0 < c < c1


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _find_frame_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert frame_coords, "expected a grey frame"
    r0, c0, r1, c1 = _bbox(frame_coords)
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    assert expected == set(frame_coords), "grey pixels must form one hollow rectangle"
    assert (r1 - r0 - 1) >= 10 and (c1 - c0 - 1) >= 10, "frame interior must be at least 10x10"
    return r0, c0, r1, c1


def _arrow_cells_from_start(
    start: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    mr, mc = sr - dr, sc - dc
    pts = {
        (mr, mc),
        (mr, sc),
        (sr, mc),
        (sr, sc),
    }
    pts.remove((mr, mc))
    return pts


def _find_arrow(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == PINK)]
    assert len(coords) == 3, "expected one 3-pixel arrow"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 1 and c1 - c0 == 1, "arrow must occupy a 2x2 bbox with one missing corner"
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    pts = set(coords)
    missing = list(corners - pts)
    assert len(missing) == 1
    mr, mc = missing[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1
    deg = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
    start = next(cell for cell, degree in deg.items() if degree == 2)
    return pts, (int(start[0]), int(start[1])), (int(dr), int(dc))


def _band_color(r: int, c: int, frame_box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_box
    depth = min(r - r0, r1 - r, c - c0, c1 - c)
    assert depth >= 1
    return BAND_COLORS[(depth - 1) % len(BAND_COLORS)]


def _trace_until_event(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    wall_cells: set[tuple[int, int]],
) -> dict:
    H, W = shape
    r0, c0, r1, c1 = frame_box
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(direction[0]), int(direction[1])
    cells: list[tuple[int, int]] = []
    unsafe: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return {
                "event": "loop",
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": (dr, dc),
                "trigger": None,
            }
        seen.add(state)

        nr, nc = cur[0] + dr, cur[1] + dc
        if 0 <= nr < H and 0 <= nc < W:
            unsafe.add((nr, nc))
        if not (0 <= nr < H and 0 <= nc < W):
            return {
                "event": "out",
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": (dr, dc),
                "trigger": None,
            }
        if (nr, nc) in arrow_cells:
            return {
                "event": "arrow",
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": (dr, dc),
                "trigger": None,
            }
        if nr in (r0, r1) or nc in (c0, c1):
            return {
                "event": "frame",
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": (dr, dc),
                "trigger": None,
            }
        if (nr, nc) in wall_cells:
            return {
                "event": "head",
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": (-dr, -dc),
                "trigger": (nr, nc),
            }

        cur = (nr, nc)
        cells.append(cur)

        side_r = (cur[0] + dr, cur[1])
        side_c = (cur[0], cur[1] + dc)
        if 0 <= side_r[0] < H and 0 <= side_r[1] < W:
            unsafe.add(side_r)
        if 0 <= side_c[0] < H and 0 <= side_c[1] < W:
            unsafe.add(side_c)

        hit_r = side_r in wall_cells
        hit_c = side_c in wall_cells
        if hit_r or hit_c:
            if hit_r and hit_c:
                event = "side_both"
                trigger = frozenset((side_r, side_c))
                new_dir = (-dr, -dc)
            elif hit_r:
                event = "side_r"
                trigger = side_r
                new_dir = (-dr, dc)
            else:
                event = "side_c"
                trigger = side_c
                new_dir = (dr, -dc)
            return {
                "event": event,
                "cells": cells,
                "unsafe": unsafe,
                "end_cur": cur,
                "end_dir": new_dir,
                "trigger": trigger,
            }


def _trace_full(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    wall_cells: set[tuple[int, int]],
) -> dict:
    cur = start
    dr, dc = direction
    painted: list[tuple[int, int]] = []
    unsafe: set[tuple[int, int]] = set()
    bounce_count = 0
    events: list[str] = []
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return {
                "painted": painted,
                "unsafe": unsafe,
                "bounce_count": bounce_count,
                "stop": "loop",
                "events": events,
            }
        seen.add(state)
        segment = _trace_until_event(shape, frame_box, arrow_cells, cur, (dr, dc), wall_cells)
        painted.extend(segment["cells"])
        unsafe |= segment["unsafe"]
        event = segment["event"]
        if event in {"frame", "arrow", "out", "loop"}:
            return {
                "painted": painted,
                "unsafe": unsafe,
                "bounce_count": bounce_count,
                "stop": event,
                "events": events,
            }
        bounce_count += 1
        events.append(event)
        cur = segment["end_cur"]
        dr, dc = segment["end_dir"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonal ray through a hollow frame, recoloring the path from concentric distance bands.

    Input structure:
    - A black background contains one grey 1-cell-thick hollow rectangle whose interior is at least 10x10.
    - Inside the frame are 3-5 isolated maroon wall pixels and one magenta 3-pixel L-shaped arrow in a 2x2 bbox.
    - The arrow sits near a frame corner and its missing corner points inward, determining the diagonal launch direction.

    Rule:
    - Keep the grey frame, the magenta arrow, and every maroon wall unchanged.
    - Build an invisible distance field over the frame interior: cells at distance 1, 2, 3, ... from the inner frame wall cycle through red, blue, yellow.
    - Fire one diagonal ray from the arrow's right-angle pixel away from the missing corner; head-on wall hits reverse both axes, side-contact wall hits reverse only the blocked axis.
    - Paint each visited black cell with the distance-band color at that position, and stop when the next step would reach the grey frame or the arrow itself.

    Key insight:
    - The geometry is determined by the arrow plus bounce walls, but the ray's visible color is determined independently by its current distance to the surrounding rectangular frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_box = _find_frame_box(g)
    arrow_cells, start, direction = _find_arrow(g)
    wall_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)}

    trace = _trace_full(g.shape, frame_box, arrow_cells, start, direction, wall_cells)

    out = g.copy()
    for r, c in trace["painted"]:
        if out[r, c] == BLACK:
            out[r, c] = _band_color(r, c, frame_box)
    return out


def _frame_candidates(rng: random.Random) -> tuple[int, int, int, int, int, int]:
    H = rng.randint(14, 18)
    W = rng.randint(14, 18)
    for _ in range(40):
        top = rng.randint(0, 2)
        bottom = rng.randint(0, 2)
        left = rng.randint(0, 2)
        right = rng.randint(0, 2)
        if H - top - bottom >= 12 and W - left - right >= 12:
            return H, W, top, left, H - bottom - 1, W - right - 1
    return H, W, 1, 1, H - 2, W - 2


def _start_options(
    frame_box: tuple[int, int, int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_box
    dr, dc = direction
    if (dr, dc) == (1, 1):
        rows = range(r0 + 2, min(r0 + 5, r1))
        cols = range(c0 + 2, min(c0 + 5, c1))
    elif (dr, dc) == (1, -1):
        rows = range(r0 + 2, min(r0 + 5, r1))
        cols = range(max(c0 + 1, c1 - 4), c1)
    elif (dr, dc) == (-1, 1):
        rows = range(max(r0 + 1, r1 - 4), r1)
        cols = range(c0 + 2, min(c0 + 5, c1))
    else:
        rows = range(max(r0 + 1, r1 - 4), r1)
        cols = range(max(c0 + 1, c1 - 4), c1)

    options: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            arrow = _arrow_cells_from_start((r, c), direction)
            if all(_is_inside_interior(cell, frame_box) for cell in arrow):
                options.append((r, c))
    return options


def _bounce_candidates(
    frame_box: tuple[int, int, int, int],
    cur: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[str, tuple[int, int], int]]:
    r0, c0, r1, c1 = frame_box
    dr, dc = direction
    row_steps = (r1 - 1 - cur[0]) if dr == 1 else (cur[0] - (r0 + 1))
    col_steps = (c1 - 1 - cur[1]) if dc == 1 else (cur[1] - (c0 + 1))
    max_steps = min(row_steps, col_steps)
    options: list[tuple[str, tuple[int, int], int]] = []

    for steps in range(1, max_steps + 1):
        cell = (cur[0] + dr * steps, cur[1] + dc * steps)
        head = (cell[0] + dr, cell[1] + dc)
        side_r = (cell[0] + dr, cell[1])
        side_c = (cell[0], cell[1] + dc)
        if _is_inside_interior(head, frame_box):
            options.append(("head", head, steps))
        if _is_inside_interior(side_r, frame_box):
            options.append(("side_r", side_r, steps))
        if _is_inside_interior(side_c, frame_box):
            options.append(("side_c", side_c, steps))
    return options


def _plan_bounce_walls(
    rng: random.Random,
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    target_bounces: int,
    require_head: bool = False,
) -> set[tuple[int, int]] | None:
    def search(
        cur: tuple[int, int],
        cur_dir: tuple[int, int],
        bounce_index: int,
        wall_cells: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
        unsafe_prefix: set[tuple[int, int]],
        has_head: bool,
    ) -> set[tuple[int, int]] | None:
        if bounce_index == target_bounces:
            if require_head and not has_head:
                return None
            return set(wall_cells)

        options = _bounce_candidates(frame_box, cur, cur_dir)
        rng.shuffle(options)
        options.sort(
            key=lambda item: (
                0 if item[0] == "head" else 1,
                -item[2] if bounce_index < target_bounces - 1 else item[2],
                rng.random(),
            )
        )
        for event_name, wall, expected_len in options:
            if wall in occupied or wall in wall_cells or wall in unsafe_prefix:
                continue
            if any(_chebyshev_close(wall, other) for other in wall_cells):
                continue

            event = _trace_until_event(
                shape,
                frame_box,
                arrow_cells,
                cur,
                cur_dir,
                wall_cells | {wall},
            )
            if event["event"] != event_name:
                continue
            if event["trigger"] != wall:
                continue
            if len(event["cells"]) != expected_len:
                continue
            segment_cells = set(event["cells"])
            if segment_cells & occupied:
                continue
            if len(segment_cells) != len(event["cells"]):
                continue
            result = search(
                event["end_cur"],
                event["end_dir"],
                bounce_index + 1,
                wall_cells | {wall},
                occupied | {wall} | segment_cells,
                unsafe_prefix | event["unsafe"],
                has_head or event_name == "head",
            )
            if result is not None:
                return result
        return None

    return search(start, direction, 0, set(), set(arrow_cells), set(), False)


def _place_distractors(
    rng: random.Random,
    frame_box: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    unsafe: set[tuple[int, int]],
    need: int,
) -> set[tuple[int, int]] | None:
    if need == 0:
        return wall_cells

    r0, c0, r1, c1 = frame_box
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in arrow_cells
        and (r, c) not in path_cells
        and (r, c) not in wall_cells
        and (r, c) not in unsafe
    ]
    rng.shuffle(candidates)
    extra: set[tuple[int, int]] = set()

    for cell in candidates:
        if any(_chebyshev_close(cell, other) for other in wall_cells | extra):
            continue
        if any(_chebyshev_close(cell, arrow) for arrow in arrow_cells):
            continue
        extra.add(cell)
        if len(extra) == need:
            return wall_cells | extra
    return None


def _render_input(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> np.ndarray:
    H, W = shape
    r0, c0, r1, c1 = frame_box
    grid = np.zeros((H, W), dtype=int)
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for r, c in arrow_cells:
        grid[r, c] = PINK
    for r, c in wall_cells:
        grid[r, c] = MAROON
    return grid


def _expected_output(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in path:
        if out[r, c] == BLACK:
            out[r, c] = _band_color(r, c, frame_box)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H, W, r0, c0, r1, c1 = _frame_candidates(rng)
    frame_box = (r0, c0, r1, c1)
    direction = rng.choice(DIAGONALS)
    starts = _start_options(frame_box, direction)
    if not starts:
        return None
    start = rng.choice(starts)
    arrow_cells = _arrow_cells_from_start(start, direction)

    total_walls = rng.randint(3, 5)
    target_bounces = rng.randint(2, min(3, total_walls))
    require_head = rng.random() < 0.45
    bounce_walls = _plan_bounce_walls(
        rng,
        (H, W),
        frame_box,
        arrow_cells,
        start,
        direction,
        target_bounces,
        require_head=require_head,
    )
    if bounce_walls is None:
        return None

    initial_trace = _trace_full((H, W), frame_box, arrow_cells, start, direction, bounce_walls)
    if initial_trace["stop"] != "frame":
        return None
    if initial_trace["bounce_count"] != target_bounces:
        return None
    if require_head and "head" not in initial_trace["events"]:
        return None
    if len(initial_trace["painted"]) < 6:
        return None
    if len(set(initial_trace["painted"])) != len(initial_trace["painted"]):
        return None

    band_count = len({_band_color(r, c, frame_box) for r, c in initial_trace["painted"]})
    if band_count < 2:
        return None

    wall_cells = _place_distractors(
        rng,
        frame_box,
        arrow_cells,
        set(initial_trace["painted"]),
        bounce_walls,
        initial_trace["unsafe"],
        total_walls - len(bounce_walls),
    )
    if wall_cells is None:
        return None

    final_trace = _trace_full((H, W), frame_box, arrow_cells, start, direction, wall_cells)
    if final_trace["stop"] != "frame":
        return None
    if final_trace["bounce_count"] < 2:
        return None
    if require_head and "head" not in final_trace["events"]:
        return None
    if len(set(final_trace["painted"])) != len(final_trace["painted"]):
        return None

    grid = _render_input((H, W), frame_box, arrow_cells, wall_cells)
    expected = _expected_output(grid, frame_box, final_trace["painted"])
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 1")
