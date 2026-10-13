from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PIPE = 9
LIQUID_COLORS = (1, 2, 3, 4, 6, 7, 8)

# Generator bounds.
MAX_GRID_HEIGHT = 20
MAX_GRID_WIDTH = 30
CONTAINER_COUNT_RANGE = (2, 5)
INNER_WIDTH_RANGE = (2, 4)
CAPACITY_RANGE = (2, 6)
WALL_EXTRA_RANGE = (0, 2)
GAP_RANGE = (1, 4)
MARGIN_RANGE = (1, 3)
GENERATION_ATTEMPTS = 400


def _runs_of_color(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) != color:
            c += 1
            continue
        start = c
        while c + 1 < width and int(row[c + 1]) == color:
            c += 1
        runs.append((start, c))
        c += 1
    return runs


def _wall_height(grid: np.ndarray, bottom: int, col: int) -> int:
    r = bottom
    while r >= 0 and int(grid[r, col]) == WALL:
        r -= 1
    return int(bottom - r)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, int]], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bottom_candidates: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, height - 1):
        runs: list[tuple[int, int]] = []
        for left, right in _runs_of_color(g[r], WALL):
            if right - left + 1 < 3:
                continue
            if int(g[r - 1, left]) != WALL or int(g[r - 1, right]) != WALL:
                continue
            if np.any(g[r - 1, left + 1 : right] == WALL):
                continue
            if int(g[r + 1, left]) == WALL or int(g[r + 1, right]) == WALL:
                continue
            runs.append((left, right))
        if runs:
            bottom_candidates[r] = runs

    assert bottom_candidates, "no container bottoms found"
    bottom = max(bottom_candidates, key=lambda row: len(bottom_candidates[row]))
    runs = sorted(bottom_candidates[bottom])
    max_count = len(runs)
    assert sum(len(v) == max_count for v in bottom_candidates.values()) == 1
    assert len(runs) >= 2, "expected multiple containers"

    inner_width = runs[0][1] - runs[0][0] - 1
    assert inner_width >= 1
    containers: list[dict[str, int]] = []
    for left, right in runs:
        assert right - left - 1 == inner_width
        left_height = _wall_height(g, bottom, left)
        right_height = _wall_height(g, bottom, right)
        capacity = min(left_height, right_height) - 1
        assert capacity >= 1

        inner = slice(left + 1, right)
        liquid_height = 0
        liquid_color = -1
        for r in range(bottom - 1, bottom - capacity - 1, -1):
            row = g[r, inner]
            values = np.unique(row)
            if values.size == 1 and int(values[0]) not in (BG, WALL, PIPE):
                color = int(values[0])
                if liquid_color == -1:
                    liquid_color = color
                assert color == liquid_color
                liquid_height += 1
            else:
                break

        assert liquid_height >= 1, "every container should contain visible liquid"
        empty_top = g[bottom - capacity : bottom - liquid_height, inner]
        assert np.all(empty_top == BG), "liquid must be bottom-aligned and contiguous"

        containers.append(
            {
                "left": int(left),
                "right": int(right),
                "bottom": int(bottom),
                "capacity": int(capacity),
                "liquid_height": int(liquid_height),
                "liquid_color": int(liquid_color),
            }
        )

    colors = [container["liquid_color"] for container in containers]
    assert len(colors) == len(set(colors)), "liquid colors must be distinct"

    pipe_row = bottom + 1
    assert pipe_row < height
    adjacency = [set() for _ in containers]
    for start, end in _runs_of_color(g[pipe_row], PIPE):
        touched = [
            i
            for i, container in enumerate(containers)
            if not (container["right"] < start or container["left"] > end)
        ]
        for i in range(len(touched) - 1):
            a = touched[i]
            b = touched[i + 1]
            adjacency[a].add(b)
            adjacency[b].add(a)

    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(containers)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group: list[int] = []
        while stack:
            node = stack.pop()
            group.append(node)
            for nxt in sorted(adjacency[node]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))

    groups.sort(key=lambda group: group[0])
    return containers, groups


def _paint_liquid(
    out: np.ndarray,
    container: dict[str, int],
    liquid_height: int,
    liquid_color: int,
) -> None:
    left = container["left"] + 1
    right = container["right"]
    bottom = container["bottom"]
    capacity = container["capacity"]
    out[bottom - capacity : bottom, left:right] = BG
    if liquid_height > 0:
        out[bottom - liquid_height : bottom, left:right] = int(liquid_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redistribute liquid across pipe-connected bottom-open containers.

    Input structure:
    - Black background.
    - Several grey U-shaped containers share the same bottom row and the same
      inner width. Their two walls may have different heights.
    - Each container contains a bottom-aligned monochrome liquid block.
    - Maroon horizontal pipe segments on the row below the bottoms connect some
      neighboring containers.

    Rule:
    - Pipe connectivity defines connected groups of containers.
    - For each connected group, sum the liquid heights, distribute that total as
      evenly as possible from left to right, and recolor the whole group using
      the color of the originally tallest container (leftmost on ties).
    - Grey walls and maroon pipes remain unchanged.
    - Isolated containers stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    containers, groups = _parse_scene(g)

    out = g.copy()
    for group in groups:
        if len(group) == 1:
            continue
        total = sum(containers[i]["liquid_height"] for i in group)
        base, extra = divmod(total, len(group))
        leader = max(group, key=lambda i: (containers[i]["liquid_height"], -i))
        color = containers[leader]["liquid_color"]

        for rank, idx in enumerate(group):
            new_height = base + (1 if rank < extra else 0)
            assert new_height <= containers[idx]["capacity"]
            _paint_liquid(out, containers[idx], new_height, color)

    return out


def _sample_edges(count: int) -> list[int]:
    if count == 2:
        return [random.randint(0, 1)]

    mode = random.choice(("all_isolated", "all_connected", "split", "mixed"))
    if mode == "all_isolated":
        return [0] * (count - 1)
    if mode == "all_connected":
        return [1] * (count - 1)
    if mode == "split":
        gap = random.randrange(count - 1)
        return [0 if i == gap else 1 for i in range(count - 1)]

    while True:
        edges = [random.randint(0, 1) for _ in range(count - 1)]
        if 0 in edges and 1 in edges:
            return edges


def _group_slices(edges: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    for i, edge in enumerate(edges, start=0):
        if edge == 0:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, len(edges) + 1))
    return groups


def _valid_totals(capacities: list[int]) -> list[int]:
    count = len(capacities)
    totals: list[int] = []
    for base in range(1, min(capacities) + 1):
        for extra in range(count):
            ok = True
            for i, capacity in enumerate(capacities):
                needed = base + (1 if i < extra else 0)
                if needed > capacity:
                    ok = False
                    break
            if ok:
                totals.append(base * count + extra)
    return sorted(set(totals))


def _sample_positive_heights(capacities: list[int], total: int) -> list[int] | None:
    remaining = total
    heights: list[int] = []
    for i, capacity in enumerate(capacities):
        rest_caps = capacities[i + 1 :]
        slots_left = len(rest_caps)
        low = max(1, remaining - sum(rest_caps))
        high = min(capacity, remaining - slots_left)
        if low > high:
            return None
        value = random.randint(low, high)
        heights.append(int(value))
        remaining -= value
    if remaining != 0:
        return None
    return heights


def _sample_group_heights(capacities: list[int]) -> list[int]:
    totals = _valid_totals(capacities)
    assert totals
    total = random.choice(totals)
    want_tie = len(capacities) > 1 and random.random() < 0.35

    for _ in range(80):
        heights = _sample_positive_heights(capacities, total)
        if heights is None:
            continue
        max_count = heights.count(max(heights))
        if want_tie and max_count >= 2:
            return heights
        if not want_tie and max_count == 1:
            return heights

    fallback = _sample_positive_heights(capacities, total)
    assert fallback is not None
    return fallback


def _render_case(
    shape: tuple[int, int],
    containers: list[dict[str, int]],
    edges: list[int],
) -> np.ndarray:
    height, width = map(int, shape)
    grid = np.full((height, width), BG, dtype=int)

    for container in containers:
        left = container["left"]
        right = container["right"]
        bottom = container["bottom"]
        left_wall = container["left_wall"]
        right_wall = container["right_wall"]
        liquid_height = container["liquid_height"]
        liquid_color = container["liquid_color"]

        grid[bottom - left_wall + 1 : bottom + 1, left] = WALL
        grid[bottom - right_wall + 1 : bottom + 1, right] = WALL
        grid[bottom, left : right + 1] = WALL
        grid[bottom - liquid_height : bottom, left + 1 : right] = liquid_color

    pipe_row = containers[0]["bottom"] + 1
    for i, edge in enumerate(edges):
        if edge:
            start = containers[i]["left"]
            end = containers[i + 1]["right"]
            grid[pipe_row, start : end + 1] = PIPE

    return grid


def _expected_output(
    shape: tuple[int, int],
    containers: list[dict[str, int]],
    edges: list[int],
) -> np.ndarray:
    updated = [dict(container) for container in containers]
    for start, end in _group_slices(edges):
        group = list(range(start, end))
        if len(group) == 1:
            continue
        total = sum(updated[i]["liquid_height"] for i in group)
        base, extra = divmod(total, len(group))
        leader = max(group, key=lambda i: (updated[i]["liquid_height"], -i))
        color = updated[leader]["liquid_color"]
        for rank, idx in enumerate(group):
            updated[idx]["liquid_height"] = base + (1 if rank < extra else 0)
            updated[idx]["liquid_color"] = color
    return _render_case(shape, updated, edges)


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(*CONTAINER_COUNT_RANGE)
    inner_width = random.randint(*INNER_WIDTH_RANGE)
    edges = _sample_edges(count)

    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]

    capacities = [random.randint(*CAPACITY_RANGE) for _ in range(count)]
    wall_pairs: list[tuple[int, int]] = []
    for capacity in capacities:
        extra_left = random.randint(*WALL_EXTRA_RANGE)
        extra_right = random.randint(*WALL_EXTRA_RANGE)
        if extra_left > 0 and extra_right > 0:
            if random.random() < 0.5:
                extra_left = 0
            else:
                extra_right = 0
        left_wall = capacity + 1 + extra_left
        right_wall = capacity + 1 + extra_right
        wall_pairs.append((left_wall, right_wall))

    max_wall = max(max(pair) for pair in wall_pairs)
    bottom = top_margin + max_wall - 1
    grid_height = bottom + 2 + bottom_margin

    outer_width = inner_width + 2
    total_width = left_margin + right_margin + count * outer_width + sum(gaps)
    if grid_height > MAX_GRID_HEIGHT or total_width > MAX_GRID_WIDTH:
        return None

    colors = random.sample(LIQUID_COLORS, count)
    liquid_heights = [0] * count
    for start, end in _group_slices(edges):
        group_caps = capacities[start:end]
        heights = _sample_group_heights(group_caps)
        liquid_heights[start:end] = heights

    containers: list[dict[str, int]] = []
    left = left_margin
    for i in range(count):
        right = left + inner_width + 1
        left_wall, right_wall = wall_pairs[i]
        containers.append(
            {
                "left": int(left),
                "right": int(right),
                "bottom": int(bottom),
                "capacity": int(capacities[i]),
                "left_wall": int(left_wall),
                "right_wall": int(right_wall),
                "liquid_height": int(liquid_heights[i]),
                "liquid_color": int(colors[i]),
            }
        )
        left = right + 1 + (gaps[i] if i < len(gaps) else 0)

    shape = (grid_height, total_width)
    grid = _render_case(shape, containers, edges)
    expected = _expected_output(shape, containers, edges)

    parsed_containers, parsed_groups = _parse_scene(grid)
    if len(parsed_containers) != count:
        return None
    expected_groups = [list(range(start, end)) for start, end in _group_slices(edges)]
    if parsed_groups != expected_groups:
        return None
    for parsed, spec in zip(parsed_containers, containers, strict=False):
        for key in ("left", "right", "bottom", "capacity", "liquid_height", "liquid_color"):
            if parsed[key] != spec[key]:
                return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _sample_case()
        if sample is None:
            continue
        grid, _expected = sample
        return grid
    raise RuntimeError("failed to generate a valid input grid")
