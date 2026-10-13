from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SHELF = 5

MARKER_BLOCK = np.array([[RED, BLUE], [GREEN, YELLOW]], dtype=int)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_marker_block(grid: np.ndarray) -> tuple[int, int]:
    H, W = map(int, grid.shape)
    matches: list[tuple[int, int]] = []
    for r in range(H - 1):
        for c in range(W - 1):
            if np.array_equal(grid[r : r + 2, c : c + 2], MARKER_BLOCK):
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "expected exactly one 2x2 marker block"
    return matches[0]


def _shelf_segments(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    H, W = map(int, grid.shape)
    rows: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        segs: list[tuple[int, int]] = []
        c = 0
        while c < W:
            if int(grid[r, c]) != SHELF:
                c += 1
                continue
            c0 = c
            while c + 1 < W and int(grid[r, c + 1]) == SHELF:
                c += 1
            segs.append((int(c0), int(c)))
            c += 1
        if segs:
            rows[int(r)] = segs
    return rows


def _span_bounds(axis: int, width: int, dh: int) -> tuple[int, int]:
    width = int(width)
    axis = int(axis)
    if int(dh) > 0:
        left = axis - (width - 1) // 2
        right = axis + width // 2
    else:
        left = axis - width // 2
        right = axis + (width - 1) // 2
    return int(left), int(right)


def _root_states(marker_r: int, marker_c: int) -> list[dict[str, int]]:
    return [
        {
            "src_r": marker_r,
            "src_c": marker_c,
            "color": RED,
            "dv": 1,
            "dh": 1,
            "row": marker_r + 2,
            "axis": marker_c + 1,
            "width": 2,
            "dist": 1,
            "generation": 0,
            "family": 0,
        },
        {
            "src_r": marker_r,
            "src_c": marker_c + 1,
            "color": BLUE,
            "dv": 1,
            "dh": -1,
            "row": marker_r + 2,
            "axis": marker_c,
            "width": 2,
            "dist": 1,
            "generation": 0,
            "family": 1,
        },
        {
            "src_r": marker_r + 1,
            "src_c": marker_c,
            "color": GREEN,
            "dv": -1,
            "dh": 1,
            "row": marker_r - 1,
            "axis": marker_c + 1,
            "width": 2,
            "dist": 1,
            "generation": 0,
            "family": 2,
        },
        {
            "src_r": marker_r + 1,
            "src_c": marker_c + 1,
            "color": YELLOW,
            "dv": -1,
            "dh": -1,
            "row": marker_r - 1,
            "axis": marker_c,
            "width": 2,
            "dist": 1,
            "generation": 0,
            "family": 3,
        },
    ]


def _simulate(grid: np.ndarray, *, want_trace: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    marker_r, marker_c = _find_marker_block(g)
    shelves = _shelf_segments(g)
    H, W = map(int, g.shape)

    out = g.copy()
    claimed: dict[tuple[int, int], tuple[tuple[int, int, int], int]] = {}
    stack: list[dict[str, int]] = list(reversed(_root_states(marker_r, marker_c)))

    trace: dict[str, object] = {
        "painted_cells": 0,
        "painted_colors": set(),
        "shelf_hits": 0,
        "branch_events": 0,
        "overflow_streams": 0,
        "max_generation": 0,
        "hit_rows": set(),
        "source_cells": set(),
    }

    while stack:
        state = stack.pop()
        row = int(state["row"])
        axis = int(state["axis"])
        width = int(state["width"])
        dist = int(state["dist"])
        dv = int(state["dv"])
        dh = int(state["dh"])
        color = int(state["color"])
        src_r = int(state["src_r"])
        src_c = int(state["src_c"])
        generation = int(state["generation"])
        trace["max_generation"] = max(int(trace["max_generation"]), generation)

        while 0 <= row < H:
            left, right = _span_bounds(axis, width, dh)
            if right < 0 or left >= W:
                break

            overlaps = [
                seg
                for seg in shelves.get(row, [])
                if not (int(seg[1]) < left or int(seg[0]) > right)
            ]
            assert len(overlaps) <= 1, "expected at most one relevant shelf per row"
            if overlaps:
                a, b = overlaps[0]
                children: list[dict[str, int]] = []
                for overflow_c in (a - 1, b + 1):
                    if 0 <= overflow_c < W and int(g[row, overflow_c]) == BG:
                        child = {
                            "src_r": src_r,
                            "src_c": src_c,
                            "color": color,
                            "dv": dv,
                            "dh": dh,
                            "row": row,
                            "axis": overflow_c,
                            "width": 1,
                            "dist": dist,
                            "generation": generation + 1,
                            "family": int(state["family"]),
                        }
                        children.append(child)
                if children:
                    trace["shelf_hits"] = int(trace["shelf_hits"]) + 1
                    trace["branch_events"] = int(trace["branch_events"]) + 1
                    trace["overflow_streams"] = int(trace["overflow_streams"]) + len(children)
                    hit_rows = trace["hit_rows"]
                    assert isinstance(hit_rows, set)
                    hit_rows.add(int(row))
                    for child in reversed(children):
                        stack.append(child)
                break

            for c in range(max(0, left), min(W - 1, right) + 1):
                if int(g[row, c]) != BG:
                    continue
                priority = (dist, src_c, src_r)
                prev = claimed.get((row, c))
                if prev is None or priority < prev[0]:
                    claimed[(row, c)] = (priority, color)
            row += dv
            axis += dh
            width += 1
            dist += 1

    for (r, c), (_priority, color) in claimed.items():
        out[r, c] = int(color)

    if want_trace:
        painted = {(int(r), int(c), int(color)) for (r, c), (_p, color) in claimed.items()}
        trace["painted_cells"] = len(painted)
        trace["painted_colors"] = {int(color) for _r, _c, color in painted}
        trace["source_cells"] = {
            (int(priority[1]), int(priority[2])) for priority, _color in claimed.values()
        }
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four diagonally directed wedges from a 2x2 marker block, splitting each wedge at grey shelves into recursive child wedges.

    Input structure:
    - The grid is black background with exactly one 2x2 marker block whose colors are fixed as red/blue on top and green/yellow below.
    - Grey horizontal shelf segments of length 3-7 sit away from the marker block, with free cells beside their endpoints.
    - No other foreground objects are present.

    Rule:
    - Keep the marker block and every shelf unchanged.
    - The red, blue, green, and yellow marker pixels launch wedges into the diagonally opposite quadrants; each wedge advances one row at a time in its direction, starts with width 2, and widens by 1 per row while only painting black cells.
    - If a wedge would enter a row containing a shelf segment, that row is not painted; instead, overflow cells one step left of the shelf and one step right of the shelf each start a same-color child wedge of width 1 that continues in the original direction, and recursive conflicts on black cells are resolved by nearer-to-block rows first, then smaller source column, then smaller source row.

    Key insight:
    - The grey shelves are not merely blockers: every hit shelf converts one widening diagonal wedge into two narrower descendant wedges, so the final pattern is a branching wedge tree controlled jointly by the 2x2 diagonal legend and shelf-triggered overflow.
    """
    out, _trace = _simulate(grid, want_trace=False)
    return out


def _project_state(state: dict[str, int], step: int) -> dict[str, int]:
    step = int(step)
    return {
        **state,
        "row": int(state["row"]) + int(state["dv"]) * step,
        "axis": int(state["axis"]) + int(state["dh"]) * step,
        "width": int(state["width"]) + step,
        "dist": int(state["dist"]) + step,
    }


def _candidate_segment_positions(
    row: int,
    span_left: int,
    span_right: int,
    width: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    min_len = 3
    max_len = min(7, width - 2)
    if max_len < min_len:
        return positions
    for seg_len in range(min_len, max_len + 1):
        for start in range(1, width - seg_len):
            end = start + seg_len - 1
            if end < span_left or start > span_right:
                continue
            positions.append((int(start), int(end)))
    random.shuffle(positions)
    return positions


def _place_shelf_on_stream(
    grid: np.ndarray,
    state: dict[str, int],
    used_rows: set[int],
    family_rows: dict[int, list[int]],
    shelves: list[tuple[int, int, int]],
    marker_r: int,
) -> tuple[bool, list[dict[str, int]]]:
    H, W = map(int, grid.shape)
    family = int(state["family"])
    dv = int(state["dv"])
    vertical_room = (H - 1 - int(state["row"])) if dv > 0 else int(state["row"])
    if vertical_room < 2:
        return False, []

    candidate_steps = list(range(2, vertical_room + 1))
    random.shuffle(candidate_steps)
    for step in candidate_steps:
        projected = _project_state(state, step)
        row = int(projected["row"])
        if row in used_rows:
            continue
        if any(abs(row - prior) < 2 for prior in family_rows[family]):
            continue
        if dv < 0 and row > marker_r - 3:
            continue
        if dv > 0 and row < marker_r + 4:
            continue

        span_left, span_right = _span_bounds(
            int(projected["axis"]), int(projected["width"]), int(projected["dh"])
        )
        if span_right < 1 or span_left > W - 2:
            continue
        clipped_left = max(1, span_left)
        clipped_right = min(W - 2, span_right)
        positions = _candidate_segment_positions(row, clipped_left, clipped_right, W)
        for a, b in positions:
            if np.any(grid[row, a : b + 1] != BG):
                continue
            left_ok = int(grid[row, a - 1]) == BG
            right_ok = int(grid[row, b + 1]) == BG
            if not (left_ok or right_ok):
                continue
            grid[row, a : b + 1] = SHELF
            used_rows.add(row)
            family_rows[family].append(row)
            shelves.append((row, a, b))
            children: list[dict[str, int]] = []
            for overflow_c in (a - 1, b + 1):
                if 0 <= overflow_c < W and int(grid[row, overflow_c]) == BG:
                    children.append(
                        {
                            "src_r": int(state["src_r"]),
                            "src_c": int(state["src_c"]),
                            "color": int(state["color"]),
                            "dv": int(state["dv"]),
                            "dh": int(state["dh"]),
                            "row": row,
                            "axis": overflow_c,
                            "width": 1,
                            "dist": int(projected["dist"]),
                            "generation": int(state["generation"]) + 1,
                            "family": family,
                        }
                    )
            return True, children
    return False, []


def _generation_requirements(
    inp: np.ndarray, out: np.ndarray, trace: dict[str, object], shelf_count: int, need_recursive: bool
) -> bool:
    painted_colors = trace["painted_colors"]
    source_cells = trace["source_cells"]
    assert isinstance(painted_colors, set)
    assert isinstance(source_cells, set)
    if shelf_count < 2:
        return False
    if int(trace["shelf_hits"]) != shelf_count:
        return False
    if int(trace["branch_events"]) < 2:
        return False
    if int(trace["painted_cells"]) < 24:
        return False
    if int(trace["painted_cells"]) > int(inp.size * 0.48):
        return False
    if len(painted_colors) < 3:
        return False
    if len(source_cells) < 3:
        return False
    if np.array_equal(inp, out):
        return False
    if need_recursive and int(trace["max_generation"]) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 26)
    W = random.randint(18, 26)
    marker_r = random.randint(6, min(10, H - 8))
    marker_c = random.randint(5, min(9, W - 8))

    grid = np.zeros((H, W), dtype=int)
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = MARKER_BLOCK

    roots = _root_states(marker_r, marker_c)
    used_rows: set[int] = set()
    family_rows: dict[int, list[int]] = {i: [] for i in range(4)}
    shelves: list[tuple[int, int, int]] = []
    frontier: list[dict[str, int]] = []

    root_order = roots[:]
    random.shuffle(root_order)
    first_level = random.choice([2, 3, 3, 4])

    down_roots = [root for root in root_order if int(root["dv"]) > 0]
    up_roots = [root for root in root_order if int(root["dv"]) < 0]
    selected: list[dict[str, int]] = []
    if down_roots:
        selected.append(random.choice(down_roots))
    if up_roots:
        selected.append(random.choice(up_roots))
    for root in root_order:
        if len(selected) >= first_level:
            break
        if root not in selected:
            selected.append(root)

    for root in selected:
        placed, children = _place_shelf_on_stream(
            grid, root, used_rows, family_rows, shelves, marker_r
        )
        if placed:
            frontier.extend(children)

    need_recursive = random.random() < 0.7
    second_level_target = random.choice([1, 1, 2]) if need_recursive else random.choice([0, 1])
    random.shuffle(frontier)
    second_level_done = 0
    for state in frontier:
        if second_level_done >= second_level_target:
            break
        if int(state["generation"]) != 1:
            continue
        placed, children = _place_shelf_on_stream(
            grid, state, used_rows, family_rows, shelves, marker_r
        )
        if placed:
            second_level_done += 1
            frontier.extend(children)

    if not shelves:
        return None

    out, trace = _simulate(grid, want_trace=True)
    if not _generation_requirements(grid, out, trace, len(shelves), need_recursive):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
