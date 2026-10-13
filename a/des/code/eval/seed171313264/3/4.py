import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _grey_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    coords = np.argwhere(grid == GREY)
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    comp_index: dict[tuple[int, int], int] = {}

    for rr, cc in coords:
        start = (int(rr), int(cc))
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        comp_cells: list[tuple[int, int]] = []
        while queue:
            cur = queue.popleft()
            comp_cells.append(cur)
            r, c = cur
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if not _in_bounds(grid.shape, nr, nc) or nxt in seen or grid[nr, nc] != GREY:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        comp_id = len(components)
        components.append(comp_cells)
        for cell in comp_cells:
            comp_index[cell] = comp_id
    return components, comp_index


def _grey_neighbor_map(cells: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cell_set = set(cells)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        cur_neighbors = []
        for dr, dc in CARDINALS:
            nxt = (r + dr, c + dc)
            if nxt in cell_set:
                cur_neighbors.append(nxt)
        neighbors[(r, c)] = cur_neighbors
    return neighbors


def _trace_path(
    cells: list[tuple[int, int]],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        next_cells = [nxt for nxt in neighbors[cur] if nxt != prev]
        if not next_cells:
            break
        if len(next_cells) != 1:
            raise AssertionError("path must not branch")
        prev, cur = cur, next_cells[0]
        ordered.append(cur)
    if len(ordered) != len(cells):
        raise AssertionError("expected a simple path component")
    return ordered


def _segment_bounds(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) == 1:
        return [(0, 0)]
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(len(dirs)):
        if i == len(dirs) - 1 or dirs[i] != dirs[i + 1]:
            bounds.append((start, i + 1))
            start = i + 2
    return bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked grey polyline segment-by-segment according to the order in which
    the path passes side-attached color gates.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    components, comp_index = _grey_components(g)
    if not components:
        return out

    neighbor_maps = [_grey_neighbor_map(cells) for cells in components]
    degrees = [{cell: len(neighbors[cell]) for cell in cells} for cells, neighbors in zip(components, neighbor_maps)]
    marker_for_component: dict[int, tuple[tuple[int, int], int]] = {}
    gate_colors: dict[tuple[int, int], int] = {}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BLACK, GREY):
                continue
            adjacent_grey = []
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == GREY:
                    adjacent_grey.append((nr, nc))
            if len(adjacent_grey) != 1:
                continue
            grey_cell = adjacent_grey[0]
            comp_id = comp_index[grey_cell]
            if degrees[comp_id][grey_cell] == 1 and comp_id not in marker_for_component:
                marker_for_component[comp_id] = (grey_cell, color)
            else:
                gate_colors[grey_cell] = color

    for comp_id, cells in enumerate(components):
        marker = marker_for_component.get(comp_id)
        if marker is None:
            continue
        start_cell, current_color = marker
        path = _trace_path(cells, neighbor_maps[comp_id], start_cell)
        for seg_start, seg_end in _segment_bounds(path):
            segment_color = current_color
            for idx in range(seg_start, seg_end + 1):
                r, c = path[idx]
                out[r, c] = segment_color
                if (r, c) in gate_colors:
                    current_color = gate_colors[(r, c)]

    return out


def _sample_positions(length: int, count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(1, length - 1))
    if count <= 0 or len(candidates) < count:
        return None
    for _ in range(80):
        chosen = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _build_horizontal_box(height: int, width: int, rng: random.Random) -> dict | None:
    max_levels = min(5, (height - 2 + 1) // 2)
    if max_levels < 2:
        return None
    levels = rng.randint(2, max_levels)
    rows = _sample_positions(height, levels, 2, rng)
    if rows is None:
        return None

    left = 1
    right = width - 2
    cur_r = rows[0]
    cur_c = left if rng.random() < 0.5 else right
    ordered = [(cur_r, cur_c)]
    for next_r in rows[1:]:
        next_c = right if cur_c == left else left
        step_c = 1 if next_c > cur_c else -1
        for cc in range(cur_c + step_c, next_c + step_c, step_c):
            ordered.append((cur_r, cc))
        step_r = 1 if next_r > cur_r else -1
        for rr in range(cur_r + step_r, next_r + step_r, step_r):
            ordered.append((rr, next_c))
        cur_r, cur_c = next_r, next_c

    return {"height": height, "width": width, "ordered": ordered}


def _build_vertical_box(height: int, width: int, rng: random.Random) -> dict | None:
    max_levels = min(5, (width - 2 + 1) // 2)
    if max_levels < 2:
        return None
    levels = rng.randint(2, max_levels)
    cols = _sample_positions(width, levels, 2, rng)
    if cols is None:
        return None

    top = 1
    bottom = height - 2
    cur_r = top if rng.random() < 0.5 else bottom
    cur_c = cols[0]
    ordered = [(cur_r, cur_c)]
    for next_c in cols[1:]:
        next_r = bottom if cur_r == top else top
        step_r = 1 if next_r > cur_r else -1
        for rr in range(cur_r + step_r, next_r + step_r, step_r):
            ordered.append((rr, cur_c))
        step_c = 1 if next_c > cur_c else -1
        for cc in range(cur_c + step_c, next_c + step_c, step_c):
            ordered.append((next_r, cc))
        cur_r, cur_c = next_r, next_c

    return {"height": height, "width": width, "ordered": ordered}


def _make_relative_path(rng: random.Random) -> dict | None:
    for _ in range(100):
        height = rng.randint(7, 12)
        width = rng.randint(7, 12)
        if rng.random() < 0.5:
            spec = _build_horizontal_box(height, width, rng)
        else:
            spec = _build_vertical_box(height, width, rng)
        if spec is None:
            continue
        cells = spec["ordered"]
        if len(cells) < 6:
            continue
        return spec
    return None


def _expanded_clear(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not blocked[r0:r1, c0:c1].any()


def _place_box(spec: dict, blocked: np.ndarray, rng: random.Random) -> dict | None:
    height = int(spec["height"])
    width = int(spec["width"])
    grid_h, grid_w = blocked.shape
    if height > grid_h or width > grid_w:
        return None
    for _ in range(100):
        top = rng.randint(0, grid_h - height)
        left = rng.randint(0, grid_w - width)
        if not _expanded_clear(blocked, top, left, height, width):
            continue
        blocked[top : top + height, left : left + width] = True
        ordered = [(top + r, left + c) for r, c in spec["ordered"]]
        return {"top": top, "left": left, "height": height, "width": width, "ordered": ordered}
    return None


def _marker_cell(path: list[tuple[int, int]]) -> tuple[int, int]:
    if len(path) < 2:
        raise AssertionError("marker needs at least two path cells")
    dr = path[0][0] - path[1][0]
    dc = path[0][1] - path[1][1]
    return path[0][0] + dr, path[0][1] + dc


def _segment_orientation(path: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    if end <= start:
        if start == 0:
            dr = path[1][0] - path[0][0]
            dc = path[1][1] - path[0][1]
        else:
            dr = path[start][0] - path[start - 1][0]
            dc = path[start][1] - path[start - 1][1]
    else:
        dr = path[start + 1][0] - path[start][0]
        dc = path[start + 1][1] - path[start][1]
    return dr, dc


def _gate_slots_for_path(path: list[tuple[int, int]]) -> list[dict]:
    bounds = _segment_bounds(path)
    if len(bounds) <= 1:
        return []
    slots = []
    path_set = set(path)
    used_marker = {_marker_cell(path)}
    for seg_index, (seg_start, seg_end) in enumerate(bounds[:-1]):
        candidates = []
        dr, dc = _segment_orientation(path, seg_start, seg_end)
        for idx in range(seg_start, seg_end):
            if idx == 0:
                continue
            r, c = path[idx]
            if dr == 0:
                options = [(r - 1, c), (r + 1, c)]
            else:
                options = [(r, c - 1), (r, c + 1)]
            valid_options = []
            for gr, gc in options:
                if (gr, gc) in path_set or (gr, gc) in used_marker:
                    continue
                valid_options.append((gr, gc))
            if valid_options:
                candidates.append({"index": idx, "gate_cells": valid_options})
        if candidates:
            slots.append({"segment": seg_index, "candidates": candidates})
    return slots


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 28)
    width = rng.randint(16, 28)
    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    target_paths = rng.randint(1, 4)
    placed_paths = []
    for _ in range(target_paths):
        spec = _make_relative_path(rng)
        if spec is None:
            continue
        placed = _place_box(spec, blocked, rng)
        if placed is not None:
            placed_paths.append(placed)

    if not placed_paths:
        return None

    if len(placed_paths) == 1:
        marked_flags = [True]
    else:
        marked_flags = [rng.random() < 0.7 for _ in placed_paths]
        if not any(marked_flags):
            marked_flags[rng.randrange(len(marked_flags))] = True
        if all(marked_flags) and rng.random() < 0.5:
            marked_flags[rng.randrange(len(marked_flags))] = False

    path_infos = []
    for placed, marked in zip(placed_paths, marked_flags):
        path = list(placed["ordered"])
        if marked and rng.random() < 0.5:
            path.reverse()
        marker_color = rng.choice(COLOR_POOL) if marked else None
        path_infos.append({"path": path, "marked": marked, "marker_color": marker_color})

    gate_slots = []
    for path_id, info in enumerate(path_infos):
        if not info["marked"]:
            continue
        slots = _gate_slots_for_path(info["path"])
        for slot in slots:
            slot = dict(slot)
            slot["path_id"] = path_id
            gate_slots.append(slot)

    if gate_slots:
        if rng.random() < 0.5:
            gate_count = rng.randint(1, min(len(COLOR_POOL), len(gate_slots), 5))
        else:
            gate_count = rng.randint(0, min(len(COLOR_POOL), len(gate_slots), 2))
    else:
        gate_count = 0
    chosen_slots = rng.sample(gate_slots, gate_count) if gate_count > 0 else []
    gate_colors = rng.sample(COLOR_POOL, gate_count) if gate_count > 0 else []

    placements_by_path: dict[int, list[tuple[tuple[int, int], int]]] = {i: [] for i in range(len(path_infos))}
    occupied_color_cells: set[tuple[int, int]] = set()
    for slot, color in zip(chosen_slots, gate_colors):
        candidate = rng.choice(slot["candidates"])
        gate_options = [cell for cell in candidate["gate_cells"] if cell not in occupied_color_cells]
        if not gate_options:
            return None
        gate_cell = rng.choice(gate_options)
        occupied_color_cells.add(gate_cell)
        placements_by_path[slot["path_id"]].append((gate_cell, color))

    for info in path_infos:
        _paint_path(grid, info["path"])

    for path_id, info in enumerate(path_infos):
        if info["marked"]:
            marker = _marker_cell(info["path"])
            if not _in_bounds(grid.shape, marker[0], marker[1]) or grid[marker] != BLACK or marker in occupied_color_cells:
                return None
            occupied_color_cells.add(marker)
            grid[marker] = int(info["marker_color"])
        for (r, c), color in placements_by_path[path_id]:
            if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
                return None
            grid[r, c] = color

    # Reject accidental color cells that do not point to a unique path cell.
    for r in range(height):
        for c in range(width):
            if grid[r, c] in (BLACK, GREY):
                continue
            adjacent = 0
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == GREY:
                    adjacent += 1
            if adjacent != 1:
                return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
