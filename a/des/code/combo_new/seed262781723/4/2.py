from __future__ import annotations

import math
from itertools import combinations

import numpy as np


BG = 0
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_LEFT = {DIRS[i]: DIRS[(i - 1) % 4] for i in range(4)}
TURN_RIGHT = {DIRS[i]: DIRS[(i + 1) % 4] for i in range(4)}

RNG = np.random.default_rng()

COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
SEGMENT_COUNT_RANGE = (2, 4)
SEGMENT_LENGTH_RANGE = (2, 5)
GAP_LENGTH_RANGE = (1, 4)
LOCAL_BBOX_LIMIT = 10
GRID_SIDE_MAX = 30
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """Reconnect same-color path fragments, then recolor each bend red.

    Input structure:
    - The grid has a black background and a single red marker pixel in the
      bottom-right corner.
    - Every non-red foreground color appears as 2-4 disconnected monochrome
      horizontal or vertical segments, each 2-5 cells long.
    - For each color, the segments' endpoints can be paired by row/column
      alignment through black cells so that all pieces form one simple
      orthogonal path.

    Rule:
    - For each non-red color, find the unique way to connect its segment
      endpoints with straight same-color lines through the black gaps.
    - Paint those missing connector cells to complete the full path while
      leaving the rest of the grid unchanged.
    - In each completed path, recolor every non-terminal turn cell red; cells
      on straight runs and the two path endpoints keep their original color.

    Key insight:
    - The endpoints of each colored segment define a small alignment graph; the
      unique subset of endpoint-to-endpoint links that yields one simple path
      reveals the hidden connectors, and bend cells are exactly the degree-2
      path cells whose two neighbors are not collinear.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BG, RED))

    for color in colors:
        path_graph = _solve_color_graph(g, color)
        for cell in path_graph:
            out[cell] = color
        for cell, neighbors in path_graph.items():
            if len(neighbors) != 2:
                continue
            n1, n2 = tuple(neighbors)
            if not ((n1[0] == cell[0] == n2[0]) or (n1[1] == cell[1] == n2[1])):
                out[cell] = RED

    return out


def _solve_color_graph(grid: np.ndarray, color: int) -> dict[tuple[int, int], set[tuple[int, int]]]:
    valid_graphs = _valid_color_graphs(grid, color, stop_after=2)
    assert len(valid_graphs) == 1, "segment connections must be uniquely solvable"
    return valid_graphs[0]


def _valid_color_graphs(
    grid: np.ndarray, color: int, stop_after: int
) -> list[dict[tuple[int, int], set[tuple[int, int]]]]:
    segments = _extract_segments(grid, color)
    assert 2 <= len(segments) <= 4, "each non-red color must have 2-4 source segments"

    candidates = []
    for i, left in enumerate(segments):
        for j in range(i + 1, len(segments)):
            right = segments[j]
            for ei, p in enumerate(left["endpoints"]):
                for ej, q in enumerate(right["endpoints"]):
                    if _clear_between(grid, p, q):
                        candidates.append(
                            {
                                "segments": (i, j),
                                "endpoints": ((i, ei), (j, ej)),
                                "line": _line_cells(p, q),
                            }
                        )

    valid_graphs = []
    for subset in combinations(candidates, len(segments) - 1):
        if not _uses_unique_endpoints(subset):
            continue
        if not _segments_connected(len(segments), subset):
            continue
        graph = _build_path_graph(segments, subset)
        if graph is None:
            continue
        valid_graphs.append(graph)
        if len(valid_graphs) >= stop_after:
            break
    return valid_graphs


def _extract_segments(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    mask = grid == color
    components = _components(mask)
    segments: list[dict[str, object]] = []

    for cells in components:
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        assert len(cells) >= 2, "segments must be at least length 2"

        if len(rows) == 1:
            ordered = sorted(cells, key=lambda cell: cell[1])
            cols_sorted = [c for _, c in ordered]
            assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)), "horizontal segment must be contiguous"
        else:
            assert len(cols) == 1, "segment must be horizontal or vertical"
            ordered = sorted(cells, key=lambda cell: cell[0])
            rows_sorted = [r for r, _ in ordered]
            assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)), "vertical segment must be contiguous"

        assert 2 <= len(ordered) <= 5, "source segment length must be 2-5"
        segments.append({"cells": ordered, "endpoints": (ordered[0], ordered[-1])})

    return segments


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []

        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(comp)

    return comps


def _clear_between(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return c1 > c0 and np.all(grid[r, c0 + 1 : c1] == BG)
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return r1 > r0 and np.all(grid[r0 + 1 : r1, c] == BG)
    return False


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    assert a[1] == b[1], "line must be horizontal or vertical"
    step = 1 if b[0] > a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _uses_unique_endpoints(subset: tuple[dict[str, object], ...]) -> bool:
    used = set()
    for edge in subset:
        for endpoint in edge["endpoints"]:
            if endpoint in used:
                return False
            used.add(endpoint)
    return True


def _segments_connected(count: int, subset: tuple[dict[str, object], ...]) -> bool:
    adjacency = {idx: set() for idx in range(count)}
    for edge in subset:
        a, b = edge["segments"]
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == count


def _build_path_graph(
    segments: list[dict[str, object]], subset: tuple[dict[str, object], ...]
) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()

    for segment in segments:
        cells = segment["cells"]
        for cell in cells:
            graph.setdefault(cell, set())
            occupied.add(cell)
        for u, v in zip(cells, cells[1:]):
            if not _add_edge(graph, u, v):
                return None

    for edge in subset:
        line = edge["line"]
        if any(cell in occupied for cell in line[1:-1]):
            return None
        for u, v in zip(line, line[1:]):
            if not _add_edge(graph, u, v):
                return None
        occupied.update(line[1:-1])

    degrees = [len(neighbors) for neighbors in graph.values()]
    if any(degree not in (1, 2) for degree in degrees):
        return None
    if degrees.count(1) != 2:
        return None

    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    if len(seen) != len(graph):
        return None
    return graph


def _add_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]], u: tuple[int, int], v: tuple[int, int]
) -> bool:
    graph.setdefault(u, set()).add(v)
    graph.setdefault(v, set()).add(u)
    return len(graph[u]) <= 2 and len(graph[v]) <= 2


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _ordered_bends(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    bends = set()
    for idx in range(1, len(cells) - 1):
        prev_cell = cells[idx - 1]
        cur_cell = cells[idx]
        next_cell = cells[idx + 1]
        if not (
            (prev_cell[0] == cur_cell[0] == next_cell[0])
            or (prev_cell[1] == cur_cell[1] == next_cell[1])
        ):
            bends.add(cur_cell)
    return bends


def _extend_path(
    current: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    occupied: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    limit: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    dr, dc = direction
    prev = current
    cur = current
    new_cells: list[tuple[int, int]] = []
    trial_occupied = set(occupied)
    min_r, max_r, min_c, max_c = bounds

    for _ in range(steps):
        cur = (cur[0] + dr, cur[1] + dc)
        if cur in trial_occupied:
            return None
        for rr, cc in DIR4:
            adj = (cur[0] + rr, cur[1] + cc)
            if adj in trial_occupied and adj != prev:
                return None
        new_cells.append(cur)
        trial_occupied.add(cur)
        min_r = min(min_r, cur[0])
        max_r = max(max_r, cur[0])
        min_c = min(min_c, cur[1])
        max_c = max(max_c, cur[1])
        if max_r - min_r + 1 > limit or max_c - min_c + 1 > limit:
            return None
        prev = cur

    return new_cells, (min_r, max_r, min_c, max_c)


def _shuffled(options: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    order = rng.permutation(len(options))
    return [options[int(idx)] for idx in order]


def _generate_local_object(
    rng: np.random.Generator, require_turn: bool
) -> dict[str, object] | None:
    n_segments = int(rng.integers(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1] + 1))
    piece_count = 2 * n_segments - 1

    visible_lengths = [int(rng.integers(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1)) for _ in range(n_segments)]
    gap_lengths = [int(rng.integers(GAP_LENGTH_RANGE[0], GAP_LENGTH_RANGE[1] + 1)) for _ in range(n_segments - 1)]

    full_cells = [(0, 0)]
    occupied = {(0, 0)}
    bounds = (0, 0, 0, 0)
    directions: list[tuple[int, int]] = []
    visible_segments: list[list[tuple[int, int]]] = []
    turn_count = 0

    for piece_idx in range(piece_count):
        is_visible = piece_idx % 2 == 0
        if is_visible:
            length = visible_lengths[piece_idx // 2]
            steps = length - 1
        else:
            gap = gap_lengths[piece_idx // 2]
            steps = gap + 1

        if piece_idx == 0:
            options = _shuffled(DIRS, rng)
        else:
            prev_dir = directions[-1]
            base = [prev_dir, TURN_LEFT[prev_dir], TURN_RIGHT[prev_dir]]
            options = _shuffled(base, rng)
            if require_turn and turn_count == 0 and piece_idx == piece_count - 1:
                options.sort(key=lambda direction: direction == prev_dir)

        chosen = None
        chosen_cells: list[tuple[int, int]] | None = None
        chosen_bounds = bounds
        current = full_cells[-1]

        for direction in options:
            attempt = _extend_path(current, direction, steps, occupied, bounds, LOCAL_BBOX_LIMIT)
            if attempt is None:
                continue
            new_cells, new_bounds = attempt
            chosen = direction
            chosen_cells = new_cells
            chosen_bounds = new_bounds
            break

        if chosen is None or chosen_cells is None:
            return None

        if directions and chosen != directions[-1]:
            turn_count += 1
        directions.append(chosen)

        start_idx = len(full_cells) - 1
        full_cells.extend(chosen_cells)
        occupied.update(chosen_cells)
        bounds = chosen_bounds

        if is_visible:
            visible_segments.append(full_cells[start_idx : len(full_cells)])

    min_r = min(r for r, _ in full_cells)
    min_c = min(c for _, c in full_cells)
    full_shifted = _shift_cells(full_cells, -min_r, -min_c)
    visible_shifted = [_shift_cells(segment, -min_r, -min_c) for segment in visible_segments]

    height = max(r for r, _ in full_shifted) + 1
    width = max(c for _, c in full_shifted) + 1
    if height > LOCAL_BBOX_LIMIT or width > LOCAL_BBOX_LIMIT:
        return None

    local_grid = np.zeros((height, width), dtype=int)
    for segment in visible_shifted:
        for r, c in segment:
            local_grid[r, c] = 1

    expected_cells = set(full_shifted)
    expected_bends = _ordered_bends(full_shifted)
    solved_graphs = _valid_color_graphs(local_grid, 1, stop_after=2)
    if len(solved_graphs) != 1:
        return None
    solved_graph = solved_graphs[0]
    if set(solved_graph) != expected_cells:
        return None
    solved_bends = {
        cell
        for cell, neighbors in solved_graph.items()
        if len(neighbors) == 2 and not (
            tuple(neighbors)[0][0] == cell[0] == tuple(neighbors)[1][0]
            or tuple(neighbors)[0][1] == cell[1] == tuple(neighbors)[1][1]
        )
    }
    if solved_bends != expected_bends:
        return None
    if require_turn and not expected_bends:
        return None

    return {
        "bbox": (height, width),
        "ordered_full_cells": full_shifted,
        "visible_segments": visible_shifted,
        "bends": expected_bends,
    }


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_objects(
    rng: np.random.Generator, objects: list[dict[str, object]], height: int, width: int
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = [(-1, -1)] * len(objects)
    expanded_rects: list[tuple[int, int, int, int]] = []
    order = list(rng.permutation(len(objects)))

    for obj_idx in order:
        obj_h, obj_w = objects[obj_idx]["bbox"]
        max_top = height - obj_h - 2
        max_left = width - obj_w - 2
        if max_top < 1 or max_left < 1:
            return None

        placed = False
        for _ in range(300):
            top = int(rng.integers(1, max_top + 1))
            left = int(rng.integers(1, max_left + 1))
            rect = (top, left, top + obj_h - 1, left + obj_w - 1)
            expanded = (rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1)
            if any(_rects_overlap(expanded, other) for other in expanded_rects):
                continue
            placements[obj_idx] = (top, left)
            expanded_rects.append(expanded)
            placed = True
            break

        if not placed:
            return None

    return placements


def _render_expected(
    base_grid: np.ndarray, objects: list[dict[str, object]], placements: list[tuple[int, int]]
) -> np.ndarray:
    out = base_grid.copy()
    for obj, (top, left) in zip(objects, placements):
        color = obj["color"]
        for r, c in obj["ordered_full_cells"]:
            out[top + r, left + c] = color
        for r, c in obj["bends"]:
            out[top + r, left + c] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = int(rng_choice(RNG, [2, 3]))
    colors = list(RNG.choice(COLOR_POOL, size=n_colors, replace=False))
    turn_owner = int(RNG.integers(0, n_colors))

    objects: list[dict[str, object]] = []
    for idx, color in enumerate(colors):
        obj = None
        for _ in range(200):
            candidate = _generate_local_object(RNG, require_turn=idx == turn_owner)
            if candidate is not None:
                obj = candidate
                break
        if obj is None:
            return None
        obj["color"] = int(color)
        objects.append(obj)

    expanded_area = sum((obj["bbox"][0] + 2) * (obj["bbox"][1] + 2) for obj in objects)
    min_side = max(
        12,
        max(obj["bbox"][0] for obj in objects) + 3,
        max(obj["bbox"][1] for obj in objects) + 3,
        int(math.ceil(math.sqrt(expanded_area))) + 1,
    )
    if min_side > GRID_SIDE_MAX:
        return None

    for _ in range(80):
        height = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
        width = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
        placements = _place_objects(RNG, objects, height, width)
        if placements is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for obj, (top, left) in zip(objects, placements):
            color = obj["color"]
            for segment in obj["visible_segments"]:
                for r, c in segment:
                    grid[top + r, left + c] = color
        grid[height - 1, width - 1] = RED

        expected = _render_expected(grid, objects, placements)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        total_bends = sum(len(obj["bends"]) for obj in objects)
        if total_bends == 0:
            continue
        return grid

    return None


def rng_choice(rng: np.random.Generator, values: list[int]) -> int:
    return int(values[int(rng.integers(0, len(values)))])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
