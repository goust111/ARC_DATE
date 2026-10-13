from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (18, 28)
ROW_MARGIN = 5
CARRIER_ROW_GAP = 6
MIN_CARRIER_LEN = 8
MIN_BRANCH_GAP = 4
MAX_PATH_COUNT = 6
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _background_color(grid: np.ndarray) -> int:
    return BG if np.any(grid == BG) else _mode_color(grid)


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out.append((rr, cc))
    return out


def _expand_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(points) >= 2
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        steps = abs(r1 - r0) + abs(c1 - c0)
        segment = [(r0 + dr * k, c0 + dc * k) for k in range(steps + 1)]
        if cells:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _path_neighbors(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    shape = mask.shape
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        neigh[cell] = [n for n in _neighbors4(shape, cell) if mask[n]]
    return neigh


def _find_hints(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    full_mask = grid != bg
    full_neighbors = _path_neighbors(full_mask)
    hint_cells: set[tuple[int, int]] = set()
    crossing_hint: dict[tuple[int, int], int] = {}

    for rr, cc in np.argwhere((grid != bg) & (grid != GREY)):
        cell = (int(rr), int(cc))
        neigh = full_neighbors.get(cell, [])
        if len(neigh) != 1:
            continue
        parent = neigh[0]
        if int(grid[parent]) != GREY:
            continue
        other_neighbors = [n for n in full_neighbors[parent] if n != cell]
        if len(other_neighbors) < 3:
            continue
        hint_cells.add(cell)
        crossing_hint[parent] = int(grid[cell])

    return hint_cells, crossing_hint


def _next_step(
    current: tuple[int, int],
    previous: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int] | None:
    neigh = neighbors[current]
    assert previous in neigh
    if len(neigh) <= 1:
        return None
    if len(neigh) == 2:
        return neigh[0] if neigh[1] == previous else neigh[1]

    dr = current[0] - previous[0]
    dc = current[1] - previous[1]
    opposite = (current[0] + dr, current[1] + dc)
    return opposite if opposite in neigh else None


def _trace_paths(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], list[tuple[int, int]]]]:
    neighbors = _path_neighbors(mask)
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []
    max_steps = int(mask.size) + 5

    for start, neighs in neighbors.items():
        for nxt in neighs:
            edge = tuple(sorted((start, nxt)))
            if edge in visited_edges:
                continue

            left = [start]
            previous, current = nxt, start
            steps = 0
            while True:
                cont = _next_step(current, previous, neighbors)
                if cont is None:
                    break
                left.append(cont)
                previous, current = current, cont
                steps += 1
                assert steps <= max_steps, "path tracing should not cycle"

            right = [nxt]
            previous, current = start, nxt
            steps = 0
            while True:
                cont = _next_step(current, previous, neighbors)
                if cont is None:
                    break
                right.append(cont)
                previous, current = current, cont
                steps += 1
                assert steps <= max_steps, "path tracing should not cycle"

            path = list(reversed(left)) + right
            for a, b in zip(path, path[1:]):
                visited_edges.add(tuple(sorted((a, b))))
            paths.append(path)

    return paths, neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each independent orthogonal path by its marker color, using the
    crossing-side hint to decide which path shows on top at shared pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    hint_cells, crossing_hint = _find_hints(g, bg)
    hint_mask = np.zeros_like(g, dtype=bool)
    for cell in hint_cells:
        hint_mask[cell] = True

    path_mask = (g != bg) & ~hint_mask
    paths, neighbors = _trace_paths(path_mask)
    out = g.copy()
    out[hint_mask] = bg

    cell_colors: dict[tuple[int, int], set[int]] = {}
    for path in paths:
        colors = {int(g[r, c]) for r, c in path if int(g[r, c]) not in {bg, GREY}}
        assert len(colors) == 1, "each path must have a unique marker color"
        color = colors.pop()
        for cell in path:
            cell_colors.setdefault(cell, set()).add(color)
            if int(g[cell]) == GREY:
                out[cell] = color

    crossings = {cell for cell, neigh in neighbors.items() if len(neigh) >= 3}
    assert crossings == set(crossing_hint.keys()), "each crossing needs one side hint"

    for cell, hint_color in crossing_hint.items():
        assert hint_color in cell_colors[cell], "hint color must match an incident path"
        out[cell] = hint_color

    return out


def _placeable(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    owners: dict[tuple[int, int], set[int]],
    blocked: set[tuple[int, int]],
    allowed_overlap: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for cell in cells:
        if not _in_bounds(shape, cell):
            return False
        if cell in blocked and cell not in allowed_overlap:
            return False
        existing = owners.get(cell, set())
        if not existing:
            pass
        elif cell not in allowed_overlap or len(existing) != 1:
            return False

        for neigh in _neighbors4(shape, cell):
            if neigh in cell_set:
                continue
            if neigh not in owners:
                continue
            if cell in allowed_overlap or neigh in allowed_overlap:
                continue
            return False
    return True


def _add_owner(owners: dict[tuple[int, int], set[int]], cells: list[tuple[int, int]], path_idx: int) -> None:
    for cell in cells:
        owners.setdefault(cell, set()).add(path_idx)


def _choose_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(ROW_MARGIN, height - ROW_MARGIN))
    if len(choices) < count:
        return None
    for _ in range(80):
        rows = sorted(random.sample(choices, count))
        if all(b - a >= CARRIER_ROW_GAP for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _build_carrier(height: int, width: int, row: int) -> dict[str, object] | None:
    for _ in range(80):
        c0 = random.randint(2, width - MIN_CARRIER_LEN - 3)
        c1 = random.randint(c0 + MIN_CARRIER_LEN, width - 3)
        left_len = random.choice([0, 0, 1, 2, 3])
        right_len = random.choice([0, 0, 1, 2, 3])

        points: list[tuple[int, int]] = []
        if left_len:
            dirs = [sgn for sgn in (-1, 1) if 1 <= row + sgn * left_len < height - 1]
            if not dirs:
                continue
            points.append((row + random.choice(dirs) * left_len, c0))
        points.append((row, c0))
        points.append((row, c1))
        if right_len:
            dirs = [sgn for sgn in (-1, 1) if 1 <= row + sgn * right_len < height - 1]
            if not dirs:
                continue
            points.append((row + random.choice(dirs) * right_len, c1))

        cells = _expand_polyline(points)
        if len(set(cells)) != len(cells):
            continue
        return {"points": points, "cells": cells, "row": row, "c0": c0, "c1": c1}
    return None


def _build_branch(
    height: int,
    width: int,
    carrier: dict[str, object],
    side: str,
    crossing_col: int,
    owners: dict[tuple[int, int], set[int]],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    row = int(carrier["row"])
    crossing = (row, int(crossing_col))

    if side == "N":
        if row <= 3:
            return None
        start_row = random.randint(1, row - 3)
        hint = (row + 1, crossing_col)
        must_use = (row - 1, crossing_col)
    else:
        if row >= height - 4:
            return None
        start_row = random.randint(row + 3, height - 2)
        hint = (row - 1, crossing_col)
        must_use = (row + 1, crossing_col)

    if not _in_bounds((height, width), hint) or hint in owners or hint in blocked:
        return None
    if any(neigh in owners or neigh in blocked for neigh in _neighbors4((height, width), hint) if neigh != crossing):
        return None

    shape_types = ["straight", "bent"]
    random.shuffle(shape_types)
    for shape_kind in shape_types:
        if shape_kind == "straight":
            points = [(start_row, crossing_col), crossing]
            candidate_sets = [points]
        else:
            cols = [c for c in range(1, width - 1) if abs(c - crossing_col) >= 2]
            random.shuffle(cols)
            candidate_sets = [[(start_row, start_col), (start_row, crossing_col), crossing] for start_col in cols[:10]]

        for points in candidate_sets:
            cells = _expand_polyline(points)
            if must_use not in cells:
                continue
            if not _placeable((height, width), cells, owners, blocked, {crossing}):
                continue
            if any(cell in owners or cell in blocked for cell in cells[:-1]):
                continue
            return {
                "points": points,
                "cells": cells,
                "crossing": crossing,
                "hint": hint,
                "side": side,
                "carrier_idx": None,
            }
    return None


def _path_colorings(
    height: int,
    width: int,
    carriers: list[dict[str, object]],
    branches: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    paths = carriers + branches
    if not (2 <= len(paths) <= MAX_PATH_COUNT):
        return None

    colors = random.sample(PATH_COLORS, len(paths))
    for idx, (path, color) in enumerate(zip(paths, colors)):
        path["path_idx"] = idx
        path["color"] = int(color)

    crossings = {tuple(branch["crossing"]) for branch in branches}
    crossing_info: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for branch in branches:
        carrier = carriers[int(branch["carrier_idx"])]
        top_path = random.choice([int(branch["path_idx"]), int(carrier["path_idx"])])
        crossing_info[tuple(branch["crossing"])] = (tuple(branch["hint"]), int(paths[top_path]["color"]))

    grid = np.full((height, width), BG, dtype=int)
    for path in paths:
        for cell in path["cells"]:
            grid[cell] = GREY

    for path in paths:
        candidates = []
        for cell in path["cells"]:
            if cell in crossings:
                continue
            if any(abs(cell[0] - cr[0]) + abs(cell[1] - cr[1]) <= 1 for cr in crossings):
                continue
            candidates.append(cell)
        if not candidates:
            return None

        marker_count = 1
        if len(candidates) >= 4 and random.random() < 0.5:
            marker_count = 2
        markers = random.sample(candidates, marker_count)
        path["markers"] = markers
        for cell in markers:
            grid[cell] = int(path["color"])

    for crossing, (hint_cell, hint_color) in crossing_info.items():
        if grid[hint_cell] != BG:
            return None
        grid[hint_cell] = int(hint_color)

    expected = grid.copy()
    for hint_cell, _hint_color in crossing_info.values():
        expected[hint_cell] = BG
    for path in paths:
        color = int(path["color"])
        for cell in path["cells"]:
            if int(expected[cell]) == GREY:
                expected[cell] = color
    for crossing, (_hint_cell, hint_color) in crossing_info.items():
        expected[crossing] = int(hint_color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_carriers = random.randint(1, 2)
    n_branches = random.randint(max(1, n_carriers), 4)

    rows = _choose_rows(height, n_carriers)
    if rows is None:
        return None

    owners: dict[tuple[int, int], set[int]] = {}
    blocked: set[tuple[int, int]] = set()
    carriers: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        placed = None
        for _ in range(120):
            carrier = _build_carrier(height, width, row)
            if carrier is None:
                continue
            cells = list(carrier["cells"])
            if not _placeable((height, width), cells, owners, blocked, set()):
                continue
            carrier["path_idx"] = idx
            placed = carrier
            _add_owner(owners, cells, idx)
            carriers.append(placed)
            break
        if placed is None:
            return None

    if n_branches < len(carriers):
        return None

    carrier_targets = list(range(len(carriers)))
    while len(carrier_targets) < n_branches:
        carrier_targets.append(random.randrange(len(carriers)))
    random.shuffle(carrier_targets)

    north_count = random.choice([n_branches // 2, (n_branches + 1) // 2])
    side_plan = ["N"] * north_count + ["S"] * (n_branches - north_count)
    random.shuffle(side_plan)

    used_cols: dict[int, list[int]] = {idx: [] for idx in range(len(carriers))}
    branches: list[dict[str, object]] = []
    next_path_idx = len(carriers)

    for carrier_idx, side in zip(carrier_targets, side_plan):
        carrier = carriers[carrier_idx]
        c0 = int(carrier["c0"])
        c1 = int(carrier["c1"])
        columns = [c for c in range(c0 + 2, c1 - 1) if all(abs(c - prev) >= MIN_BRANCH_GAP for prev in used_cols[carrier_idx])]
        random.shuffle(columns)
        placed = None
        for crossing_col in columns:
            for _ in range(30):
                branch = _build_branch(height, width, carrier, side, crossing_col, owners, blocked)
                if branch is None:
                    continue
                branch["carrier_idx"] = carrier_idx
                branch["path_idx"] = next_path_idx
                placed = branch
                break
            if placed is not None:
                break

        if placed is None:
            return None

        used_cols[carrier_idx].append(int(placed["crossing"][1]))
        _add_owner(owners, list(placed["cells"]), next_path_idx)
        branches.append(placed)
        next_path_idx += 1

        hint = tuple(placed["hint"])
        blocked.add(hint)
        for neigh in _neighbors4((height, width), hint):
            if neigh != tuple(placed["crossing"]):
                blocked.add(neigh)

    overlap_cells = {cell for cell, path_ids in owners.items() if len(path_ids) > 1}
    if overlap_cells != {tuple(branch["crossing"]) for branch in branches}:
        return None

    return _path_colorings(height, width, carriers, branches)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _generate_scene()
        if scene is None:
            continue
        grid, _expected = scene
        return grid
    raise RuntimeError("generate_input failed after repeated attempts")
