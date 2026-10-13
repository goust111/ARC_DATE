from __future__ import annotations

import random

import numpy as np


BG = 0
PATH = 5
MARKER_COLORS = (1, 2, 3, 4, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    path_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == PATH)]
    assert path_cells
    path_set = set(path_cells)

    degrees: dict[tuple[int, int], int] = {}
    for r, c in path_cells:
        degrees[(r, c)] = sum((r + dr, c + dc) in path_set for dr, dc in DIRS4)

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2
    start = min(endpoints)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        r, c = cur
        nxt = [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in path_set and (r + dr, c + dc) != prev]
        if not nxt:
            break
        assert len(nxt) == 1
        prev, cur = cur, nxt[0]

    assert len(order) == len(path_cells)
    return order


def _extract_markers(grid: np.ndarray) -> list[dict[str, object]]:
    path_set = {tuple(map(int, rc)) for rc in np.argwhere(grid == PATH)}
    markers: list[dict[str, object]] = []

    marker_coords = np.argwhere((grid != BG) & (grid != PATH))
    for r, c in marker_coords:
        rr, cc = int(r), int(c)
        adjacent = [(rr + dr, cc + dc) for dr, dc in DIRS4 if (rr + dr, cc + dc) in path_set]
        assert len(adjacent) == 1
        markers.append(
            {
                "pos": (rr, cc),
                "color": int(grid[rr, cc]),
                "anchor": adjacent[0],
            }
        )

    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the gray snake and use its visitation order to turn colored singletons into a cyclic halo system.

    Input structure:
    - The grid has black background, one gray 4-connected snake path with exactly two endpoints, and 3-5 colored singleton markers.
    - Each marker is off the path, sits at least one cell from the border, and is 4-neighbor-adjacent to exactly one gray path cell.
    - Marker colors are distinct, and different markers attach to different path cells.

    Rule:
    - Start from the endpoint with smaller row, breaking ties by smaller column, and trace the gray snake in path order.
    - Record the colors of the markers whose attached path cells are visited, then build a cyclic permutation from each recorded color to the next one.
    - Remove the entire gray snake, paint each marker's 3x3 neighborhood with its permuted color in traversal order, and finally restore every marker's original color at its own cell.

    Key insight:
    - The gray snake is only an ordering device: its endpoint rule fixes a unique traversal, that traversal fixes the color cycle, and the cycle determines each marker's halo color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    path_order = _trace_path(g)
    markers = _extract_markers(g)

    anchor_to_marker = {marker["anchor"]: marker for marker in markers}
    visited = [anchor_to_marker[cell] for cell in path_order if cell in anchor_to_marker]
    colors = [int(marker["color"]) for marker in visited]
    assert len(colors) >= 3
    assert len(colors) == len(set(colors))

    permutation = {colors[i]: colors[(i + 1) % len(colors)] for i in range(len(colors))}
    marker_cells = {tuple(marker["pos"]): int(marker["color"]) for marker in markers}

    out[out == PATH] = BG
    for marker in visited:
        r, c = marker["pos"]
        halo_color = permutation[int(marker["color"])]
        for rr in range(max(0, r - 1), min(out.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(out.shape[1], c + 2)):
                if (rr, cc) not in marker_cells:
                    out[rr, cc] = halo_color

    for (r, c), color in marker_cells.items():
        out[r, c] = color
    return out


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    if len(path) != len(set(path)):
        return False
    path_set = set(path)
    degrees = []
    for r, c in path:
        degree = sum((r + dr, c + dc) in path_set for dr, dc in DIRS4)
        if degree not in (1, 2):
            return False
        degrees.append(degree)
    return degrees.count(1) == 2


def _build_horizontal_snake(height: int, width: int) -> list[tuple[int, int]]:
    max_lanes = min(5, (height - 3) // 2)
    lane_count = random.randint(3, max_lanes)
    span = random.randint(5, min(8, width - 4))
    top = random.randint(2, height - 3 - 2 * (lane_count - 1))
    left = random.randint(2, width - span - 2)
    right = left + span - 1

    path: list[tuple[int, int]] = []
    for lane in range(lane_count):
        row = top + 2 * lane
        if lane == 0:
            for col in range(left, right + 1):
                path.append((row, col))
            continue

        connector_col = path[-1][1]
        path.append((row - 1, connector_col))
        path.append((row, connector_col))
        if lane % 2 == 1:
            for col in range(connector_col - 1, left - 1, -1):
                path.append((row, col))
        else:
            for col in range(connector_col + 1, right + 1):
                path.append((row, col))

    assert _is_simple_path(path)
    assert path[0] == min(path[0], path[-1])
    return path


def _build_vertical_snake(height: int, width: int) -> list[tuple[int, int]]:
    lane_choices = [lanes for lanes in (4, 6) if lanes <= min(6, (width - 3) // 2)]
    lane_count = random.choice(lane_choices)
    span = random.randint(5, min(8, height - 4))
    left = random.randint(2, width - 3 - 2 * (lane_count - 1))
    top = random.randint(2, height - span - 2)
    bottom = top + span - 1

    path: list[tuple[int, int]] = []
    for lane in range(lane_count):
        col = left + 2 * lane
        if lane == 0:
            for row in range(top, bottom + 1):
                path.append((row, col))
            continue

        connector_row = path[-1][0]
        path.append((connector_row, col - 1))
        path.append((connector_row, col))
        if lane % 2 == 1:
            for row in range(connector_row - 1, top - 1, -1):
                path.append((row, col))
        else:
            for row in range(connector_row + 1, bottom + 1):
                path.append((row, col))

    assert _is_simple_path(path)
    assert path[0] == min(path[0], path[-1])
    assert path[0][0] == path[-1][0]
    return path


def _marker_candidates(path: list[tuple[int, int]], shape: tuple[int, int]) -> list[dict[str, object]]:
    path_set = set(path)
    seen_positions: set[tuple[int, int]] = set()
    candidates: list[dict[str, object]] = []

    for index, (r, c) in enumerate(path):
        for dr, dc in DIRS4:
            mr, mc = r + dr, c + dc
            if not (1 <= mr < shape[0] - 1 and 1 <= mc < shape[1] - 1):
                continue
            if (mr, mc) in path_set or (mr, mc) in seen_positions:
                continue
            neighbors = sum((mr + adr, mc + adc) in path_set for adr, adc in DIRS4)
            if neighbors != 1:
                continue
            seen_positions.add((mr, mc))
            candidates.append(
                {
                    "index": index,
                    "anchor": (r, c),
                    "marker": (mr, mc),
                }
            )

    return candidates


def _non_overlapping(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) > 2 or abs(a[1] - b[1]) > 2


def _choose_markers(candidates: list[dict[str, object]], count: int) -> list[dict[str, object]] | None:
    for _ in range(40):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        chosen: list[dict[str, object]] = []
        used_anchors: set[tuple[int, int]] = set()

        def dfs(start: int) -> bool:
            if len(chosen) == count:
                return True
            remaining = len(shuffled) - start
            if len(chosen) + remaining < count:
                return False

            for idx in range(start, len(shuffled)):
                cand = shuffled[idx]
                anchor = cand["anchor"]
                marker = cand["marker"]
                if anchor in used_anchors:
                    continue
                if any(not _non_overlapping(marker, prev["marker"]) for prev in chosen):
                    continue
                used_anchors.add(anchor)
                chosen.append(cand)
                if dfs(idx + 1):
                    return True
                chosen.pop()
                used_anchors.remove(anchor)
            return False

        if dfs(0):
            return sorted(chosen, key=lambda item: int(item["index"]))
    return None


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    path_order = _trace_path(grid)
    markers = _extract_markers(grid)
    assert 3 <= len(markers) <= 5
    assert len({int(marker["color"]) for marker in markers}) == len(markers)
    assert len({tuple(marker["anchor"]) for marker in markers}) == len(markers)

    marker_positions = [tuple(marker["pos"]) for marker in markers]
    for r, c in marker_positions:
        assert 1 <= r < grid.shape[0] - 1
        assert 1 <= c < grid.shape[1] - 1
    for i in range(len(marker_positions)):
        for j in range(i + 1, len(marker_positions)):
            assert _non_overlapping(marker_positions[i], marker_positions[j])

    anchor_to_marker = {marker["anchor"]: marker for marker in markers}
    visited = [anchor_to_marker[cell] for cell in path_order if cell in anchor_to_marker]
    assert len(visited) == len(markers)

    output = main(grid)
    assert output.shape == grid.shape
    assert max(output.shape) <= 30
    assert np.any(output != grid)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(11, 16)
    width = random.randint(11, 16)
    kind = random.choice(("horizontal", "vertical"))

    if kind == "horizontal":
        if min(height, width) < 11:
            return None
        path = _build_horizontal_snake(height, width)
    else:
        if width < 11 or height < 11:
            return None
        path = _build_vertical_snake(height, width)

    marker_count = random.randint(3, 5)
    candidates = _marker_candidates(path, (height, width))
    chosen = _choose_markers(candidates, marker_count)
    if chosen is None:
        return None

    colors = random.sample(list(MARKER_COLORS), marker_count)
    grid = np.zeros((height, width), dtype=int)
    for r, c in path:
        grid[r, c] = PATH
    for cand, color in zip(chosen, colors):
        mr, mc = cand["marker"]
        grid[mr, mc] = color

    _validate_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid snake-ring puzzle")
