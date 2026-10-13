from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
GREY = 5
MAROON = 9
SOURCE_COLORS = [1, 2, 4, 6, 7, 8]
SPECIAL = {BG, GREEN, GREY, MAROON}

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
GRID_MAX = 30
SCENE_TYPES = [(1, 1), (2, 1), (1, 2)]
LANE_COUNT_OPTIONS = [2, 3]
LANE_GAP = 3


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _orientation_map(cells: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    orient: dict[tuple[int, int], str] = {}
    for i, cell in enumerate(cells):
        if i == 0:
            nxt = cells[1]
            orient[cell] = "H" if nxt[0] == cell[0] else "V"
        elif i == len(cells) - 1:
            prv = cells[-2]
            orient[cell] = "H" if prv[0] == cell[0] else "V"
        else:
            prv, nxt = cells[i - 1], cells[i + 1]
            if prv[0] == nxt[0] == cell[0]:
                orient[cell] = "H"
            elif prv[1] == nxt[1] == cell[1]:
                orient[cell] = "V"
            else:
                orient[cell] = "HV"
    return orient


def _next_step(grid: np.ndarray, prev: tuple[int, int], cur: tuple[int, int]) -> tuple[int, int] | None:
    if int(grid[cur]) == GREY:
        dr = cur[0] - prev[0]
        dc = cur[1] - prev[1]
        cand = (cur[0] + dr, cur[1] + dc)
        assert _in_bounds(grid.shape, *cand) and int(grid[cand]) != BG
        return cand

    candidates = []
    for nr, nc in _neighbors4(*cur):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG and (nr, nc) != prev:
            candidates.append((nr, nc))

    assert len(candidates) <= 1, "non-crossing path cells should have a unique continuation"
    return candidates[0] if candidates else None


def _trace_path(grid: np.ndarray, component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for cell in component:
        if not _is_border(grid.shape, cell):
            continue
        deg = sum(
            1
            for nr, nc in _neighbors4(*cell)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG
        )
        if deg == 1:
            starts.append(cell)

    if not starts:
        starts = [cell for cell in component if _is_border(grid.shape, cell)]
    assert starts, "expected a colored border entry"

    start = starts[0]
    neigh = [
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG
    ]
    assert len(neigh) == 1, "entry endpoint should have exactly one interior continuation"

    path = [start, neigh[0]]
    seen = {start}
    while True:
        prev, cur = path[-2], path[-1]
        assert cur not in seen, "a pipe should not revisit cells"
        seen.add(cur)
        nxt = _next_step(grid, prev, cur)
        if nxt is None:
            break
        path.append(nxt)
        assert len(path) <= grid.size, "pipe trace exceeded the grid size"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve over/under crossings in orthogonal pipes and recolor each whole pipe by its entry color.

    Input structure:
    - Background is black.
    - Colored 1px-wide orthogonal pipes enter from the grid edge and exit at another edge.
    - Each pipe has a 2-pixel colored entry segment; the rest is green except for grey crossings
      and maroon bridge markers beside the upper pipe.

    Rule:
    - Trace each pipe from its colored border entry to its exit.
    - At grey crossings, keep moving straight through in the incoming direction.
    - Repaint every pipe with its own entry color, with the crossing pixel finally showing the
      color of the upper pipe indicated by the maroon marker orientation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    border_components: list[tuple[int, list[tuple[int, int]]]] = []
    visited_border_color_cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not _is_border(g.shape, (r, c)):
                continue
            if (r, c) in visited_border_color_cells:
                continue
            color = int(g[r, c])
            if color in SPECIAL:
                continue
            for comp in _components(g == color):
                comp_set = set(comp)
                if (r, c) in comp_set:
                    visited_border_color_cells |= comp_set
                    border_components.append((color, comp))
                    break

    traced = []
    dedupe: set[tuple[int, frozenset[tuple[int, int]]]] = set()
    for color, comp in border_components:
        cells = _trace_path(g, comp)
        key = (int(color), frozenset(cells))
        if key in dedupe:
            continue
        dedupe.add(key)
        traced.append(
            {
                "color": int(color),
                "cells": cells,
                "orient": _orientation_map(cells),
            }
        )

    out = np.full_like(g, BG)
    crossing_map: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for path in traced:
        for cell in path["cells"]:
            if int(g[cell]) == GREY:
                crossing_map.setdefault(cell, []).append((path["orient"][cell], path["color"]))
            else:
                out[cell] = path["color"]

    for cell in map(tuple, np.argwhere(g == GREY)):
        maroons = [
            (nr, nc)
            for nr, nc in _neighbors4(*cell)
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == MAROON
        ]
        assert len(maroons) == 1, "each crossing should have exactly one bridge marker"
        top_axis = "H" if maroons[0][0] == cell[0] else "V"
        colors = [color for axis, color in crossing_map.get(cell, []) if axis == top_axis]
        assert len(colors) == 1, "the bridge marker must identify exactly one upper pipe"
        out[cell] = colors[0]

    return out


def _sample_spaced(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(low, high))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(500):
        cand = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(cand, cand[1:])):
            return cand
    return None


def _build_horizontal_path(
    height: int,
    width: int,
    rows_seq: list[int],
    turn_cols: list[int],
    reverse: bool,
) -> list[tuple[int, int]]:
    cells = [(rows_seq[0], 0)]
    r, c = rows_seq[0], 0
    for i, turn_c in enumerate(turn_cols):
        for nc in range(c + 1, turn_c + 1):
            cells.append((r, nc))
        nr = rows_seq[i + 1]
        step = 1 if nr > r else -1
        for rr in range(r + step, nr + step, step):
            cells.append((rr, turn_c))
        r, c = nr, turn_c
    for nc in range(c + 1, width):
        cells.append((r, nc))
    return list(reversed(cells)) if reverse else cells


def _build_vertical_path(
    height: int,
    width: int,
    cols_seq: list[int],
    turn_rows: list[int],
    reverse: bool,
) -> list[tuple[int, int]]:
    cells = [(0, cols_seq[0])]
    r, c = 0, cols_seq[0]
    for i, turn_r in enumerate(turn_rows):
        for nr in range(r + 1, turn_r + 1):
            cells.append((nr, c))
        nc = cols_seq[i + 1]
        step = 1 if nc > c else -1
        for cc in range(c + step, nc + step, step):
            cells.append((turn_r, cc))
        r, c = turn_r, nc
    for nr in range(r + 1, height):
        cells.append((nr, c))
    return list(reversed(cells)) if reverse else cells


def _can_add_path(
    path: dict,
    placed: list[dict],
    height: int,
    width: int,
) -> bool:
    cells = path["cells"]
    if len(set(cells)) != len(cells):
        return False

    occupied: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for other in placed:
        for cell in other["cells"]:
            occupied.setdefault(cell, []).append((other["id"], other["orient"][cell]))

    shared: set[tuple[int, int]] = set()
    for cell in cells:
        if cell not in occupied:
            continue
        if len(occupied[cell]) != 1 or _is_border((height, width), cell):
            return False
        other_orient = occupied[cell][0][1]
        this_orient = path["orient"][cell]
        if this_orient not in {"H", "V"} or other_orient not in {"H", "V"} or this_orient == other_orient:
            return False
        shared.add(cell)

    consecutive = set()
    for a, b in zip(cells, cells[1:]):
        consecutive.add((a, b))
        consecutive.add((b, a))

    for cell in cells:
        if cell in shared:
            continue
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) not in occupied:
                continue
            if (nr, nc) in shared and (cell, (nr, nc)) in consecutive:
                continue
            return False
    return True


def _choose_markers(
    paths: list[dict],
    occupancy: dict[tuple[int, int], list[int]],
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], tuple[int, int]]] | tuple[None, None]:
    cross_cells = [cell for cell, ids in occupancy.items() if len(ids) == 2]
    random.shuffle(cross_cells)
    cross_set = set(cross_cells)
    used_markers: set[tuple[int, int]] = set()
    top_axis: dict[tuple[int, int], str] = {}
    marker_for_cross: dict[tuple[int, int], tuple[int, int]] = {}

    for cell in cross_cells:
        p0, p1 = occupancy[cell]
        axis0 = paths[p0]["orient"][cell]
        axis1 = paths[p1]["orient"][cell]
        assert {axis0, axis1} == {"H", "V"}
        chosen_axis = random.choice(["H", "V"])
        top_axis[cell] = chosen_axis
        top_id = p0 if axis0 == chosen_axis else p1
        idx = paths[top_id]["index"][cell]
        candidates = [paths[top_id]["cells"][idx - 1], paths[top_id]["cells"][idx + 1]]
        random.shuffle(candidates)

        pick = None
        for cand in candidates:
            if cand in cross_set or cand in used_markers:
                continue
            if paths[top_id]["index"][cand] < 2:
                continue
            if paths[top_id]["orient"][cand] == "HV":
                continue
            pick = cand
            break
        if pick is None:
            return None, None

        used_markers.add(pick)
        marker_for_cross[cell] = pick

    return top_axis, marker_for_cross


def _render_input(paths: list[dict], shape: tuple[int, int], marker_for_cross: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    crossing_cells = set(marker_for_cross)
    marker_cells = set(marker_for_cross.values())

    for path in paths:
        for cell in path["cells"]:
            grid[cell] = GREEN
        for cell in path["cells"][:2]:
            grid[cell] = path["color"]

    for cell in marker_cells:
        grid[cell] = MAROON
    for cell in crossing_cells:
        grid[cell] = GREY
    return grid


def _render_output(paths: list[dict], shape: tuple[int, int], top_axis: dict[tuple[int, int], str]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    occupancy: dict[tuple[int, int], list[int]] = {}
    for path in paths:
        for cell in path["cells"]:
            occupancy.setdefault(cell, []).append(path["id"])
            out[cell] = path["color"]

    for cell, ids in occupancy.items():
        if len(ids) != 2:
            continue
        winner = next(pid for pid in ids if paths[pid]["orient"][cell] == top_axis[cell])
        out[cell] = paths[winner]["color"]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_h, n_v = random.choice(SCENE_TYPES)
        h_lane_counts = [random.choice(LANE_COUNT_OPTIONS) for _ in range(n_h)]
        v_lane_counts = [random.choice(LANE_COUNT_OPTIONS) for _ in range(n_v)]

        total_row_slots = sum(h_lane_counts) + sum(k - 1 for k in v_lane_counts)
        total_col_slots = sum(v_lane_counts) + sum(k - 1 for k in h_lane_counts)

        min_h = max(14, LANE_GAP * total_row_slots + 3)
        min_w = max(14, LANE_GAP * total_col_slots + 3)
        height = random.randint(min_h, GRID_MAX)
        width = random.randint(min_w, GRID_MAX)

        row_slots = _sample_spaced(2, height - 2, total_row_slots, LANE_GAP)
        col_slots = _sample_spaced(2, width - 2, total_col_slots, LANE_GAP)
        if row_slots is None or col_slots is None:
            continue

        random.shuffle(row_slots)
        random.shuffle(col_slots)
        h_rows = row_slots[: sum(h_lane_counts)]
        v_turn_rows = row_slots[sum(h_lane_counts) :]
        v_cols = col_slots[: sum(v_lane_counts)]
        h_turn_cols = col_slots[sum(v_lane_counts) :]

        random.shuffle(h_rows)
        random.shuffle(v_turn_rows)
        random.shuffle(v_cols)
        random.shuffle(h_turn_cols)

        path_colors = random.sample(SOURCE_COLORS, n_h + n_v)
        paths: list[dict] = []
        ok = True

        row_idx = 0
        turn_idx = 0
        for lane_count in h_lane_counts:
            rows_seq = h_rows[row_idx : row_idx + lane_count]
            random.shuffle(rows_seq)
            turn_cols = sorted(h_turn_cols[turn_idx : turn_idx + lane_count - 1])
            cells = _build_horizontal_path(
                height,
                width,
                rows_seq,
                turn_cols,
                reverse=random.choice([False, True]),
            )
            path = {
                "id": len(paths),
                "color": path_colors[len(paths)],
                "cells": cells,
            }
            path["orient"] = _orientation_map(cells)
            path["index"] = {cell: i for i, cell in enumerate(cells)}
            if not _can_add_path(path, paths, height, width):
                ok = False
                break
            paths.append(path)
            row_idx += lane_count
            turn_idx += lane_count - 1
        if not ok:
            continue

        col_idx = 0
        turn_idx = 0
        for lane_count in v_lane_counts:
            cols_seq = v_cols[col_idx : col_idx + lane_count]
            random.shuffle(cols_seq)
            turn_rows = sorted(v_turn_rows[turn_idx : turn_idx + lane_count - 1])
            cells = _build_vertical_path(
                height,
                width,
                cols_seq,
                turn_rows,
                reverse=random.choice([False, True]),
            )
            path = {
                "id": len(paths),
                "color": path_colors[len(paths)],
                "cells": cells,
            }
            path["orient"] = _orientation_map(cells)
            path["index"] = {cell: i for i, cell in enumerate(cells)}
            if not _can_add_path(path, paths, height, width):
                ok = False
                break
            paths.append(path)
            col_idx += lane_count
            turn_idx += lane_count - 1
        if not ok:
            continue

        occupancy: dict[tuple[int, int], list[int]] = {}
        for path in paths:
            for cell in path["cells"]:
                occupancy.setdefault(cell, []).append(path["id"])

        crossings = [cell for cell, ids in occupancy.items() if len(ids) == 2]
        if not crossings:
            continue

        cross_count_by_path = [0] * len(paths)
        for cell in crossings:
            for pid in occupancy[cell]:
                cross_count_by_path[pid] += 1
        if min(cross_count_by_path) == 0:
            continue

        top_axis, marker_for_cross = _choose_markers(paths, occupancy)
        if top_axis is None or marker_for_cross is None:
            continue

        inp = _render_input(paths, (height, width), marker_for_cross)
        expected = _render_output(paths, (height, width), top_axis)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid pipe-overpass scene")
