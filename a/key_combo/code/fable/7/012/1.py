from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, the conventional ARC background."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as (N, 2) arrays of (row, column) coordinates."""
    height, width = mask.shape
    unseen = mask.copy().astype(bool)
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(unseen):
        start = (int(start_r), int(start_c))
        if not unseen[start]:
            continue
        unseen[start] = False
        queue = deque([start])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Find the unique solid divider whose two neighboring panels have equal width."""
    height, width = grid.shape
    candidates: list[int] = []
    for c in range(1, width - 1):
        if c != width - c - 1:
            continue
        column = grid[:, c]
        if np.all(column == column[0]) and int(column[0]) != bg:
            candidates.append(c)
    assert len(candidates) == 1, "expected one vertical divider between equal panels"
    divider = candidates[0]
    return grid[:, :divider].copy(), grid[:, divider + 1 :].copy(), divider


def _drop_objects(left: np.ndarray, obstacle_mask: np.ndarray, bg: int) -> np.ndarray:
    """Drop foreground components in the specified lower-object-first order."""
    height, width = left.shape
    assert obstacle_mask.shape == left.shape
    components = _components(left != bg)
    assert components, "expected at least one object in the left panel"

    # Components whose input bottom is lower settle first.  The remaining keys only
    # make independent (non-column-overlapping) ties deterministic.
    components.sort(
        key=lambda cells: (
            -int(cells[:, 0].max()),
            int(cells[:, 0].min()),
            int(cells[:, 1].min()),
        )
    )

    out = np.full_like(left, bg)
    occupied = obstacle_mask.copy().astype(bool)
    for cells in components:
        colors = left[cells[:, 0], cells[:, 1]].copy()
        distance = 0
        while True:
            next_rows = cells[:, 0] + distance + 1
            if np.any(next_rows >= height):
                break
            if np.any(occupied[next_rows, cells[:, 1]]):
                break
            distance += 1

        rows = cells[:, 0] + distance
        assert np.all(out[rows, cells[:, 1]] == bg), "settled objects must not overlap"
        out[rows, cells[:, 1]] = colors
        occupied[rows, cells[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop left-panel objects onto supports defined by invisible right-panel obstacles and return only the left panel.

    Input structure:
    - A solid vertical divider separates two equal-sized panels with a shared background.
    - The left panel contains several separated multi-cell colored objects at different heights.
    - The right panel contains singleton pixels of one obstacle color in row-column correspondence with the left panel.

    Rule:
    - Map every right-panel obstacle to the identical (row, column) coordinate of the left panel, without drawing it.
    - Clear the floating objects, then drop each object straight down as a rigid shape, settling lower objects first.
    - Stop immediately above the bottom edge, an invisible mapped obstacle, or an already settled object, and crop away the right panel and divider.

    Key insight:
    - The right panel is an invisible collision map rather than a picture to overlay; apparently floating final objects are supported by its aligned pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids cannot exceed 30 by 30"
    bg = _background(g)
    left, right, _divider = _split_panels(g, bg)

    right_colors = set(map(int, right[right != bg]))
    assert len(right_colors) == 1, "expected one obstacle color in the right panel"
    obstacle_mask = right != bg
    return _drop_objects(left, obstacle_mask, bg)


# Generator bounds and retry policy.
HEIGHT_RANGE = (13, 18)
PANEL_WIDTH_MAX = 13
GENERATION_ATTEMPTS = 300


def _paint_shape(canvas: np.ndarray, cells: list[Coord], origin: Coord, color: int) -> None:
    r0, c0 = origin
    for r, c in cells:
        canvas[r0 + r, c0 + c] = color


def _translated(cells: list[Coord], origin: Coord) -> set[Coord]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in cells}


def _independent_drop(left: np.ndarray, obstacles: np.ndarray, bg: int) -> np.ndarray:
    """Deliberately wrong model used to verify that settled-object support matters."""
    out = np.full_like(left, bg)
    for cells in _components(left != bg):
        colors = left[cells[:, 0], cells[:, 1]].copy()
        distance = 0
        while True:
            nr = cells[:, 0] + distance + 1
            if np.any(nr >= left.shape[0]) or np.any(obstacles[nr, cells[:, 1]]):
                break
            distance += 1
        out[cells[:, 0] + distance, cells[:, 1]] = colors
    return out


def _candidate() -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.randint(3, 5)
    min_width = {3: 9, 4: 11, 5: 13}[object_count]
    panel_width = random.randint(min_width, PANEL_WIDTH_MAX)
    height = random.randint(*HEIGHT_RANGE)

    palette = list(range(10))
    bg, divider_color, obstacle_color, *object_colors = random.sample(
        palette, 3 + object_count
    )
    left = np.full((height, panel_width), bg, dtype=int)
    right = np.full_like(left, bg)
    expected = np.full_like(left, bg)

    base_width = random.choice((2, 3))
    base_height = random.choice((1, 2))
    base_shape = [(r, c) for r in range(base_height) for c in range(base_width)]
    upper_shapes = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
    ]
    upper_shape = random.choice(upper_shapes)
    upper_h = max(r for r, _c in upper_shape) + 1
    upper_w = max(c for _r, c in upper_shape) + 1

    base_on_left = random.choice((True, False))
    base_c = 1 if base_on_left else panel_width - 1 - base_width
    isolated_c = panel_width - 2 if base_on_left else 1

    base_final = (height - base_height, base_c)
    base_input = (base_final[0] - 1, base_c)
    upper_c = base_c + random.randint(0, base_width - upper_w)
    upper_final = (base_final[0] - upper_h, upper_c)
    upper_lift = random.randint(3, 5)
    upper_input = (upper_final[0] - upper_lift, upper_c)
    if upper_input[0] < 0:
        return None

    # A separate vertical domino is visibly caught in mid-air by a mapped obstacle.
    isolated_shape = [(0, 0), (1, 0)]
    isolated_final = (height - random.randint(6, 8), isolated_c)
    isolated_lift = random.randint(2, min(4, isolated_final[0]))
    isolated_input = (isolated_final[0] - isolated_lift, isolated_c)

    specs: list[tuple[list[Coord], Coord, Coord, int, str]] = [
        (base_shape, base_input, base_final, object_colors[0], "floor"),
        (upper_shape, upper_input, upper_final, object_colors[1], "settled"),
        (isolated_shape, isolated_input, isolated_final, object_colors[2], "obstacle"),
    ]

    # Extra objects vary the count while occupying well-separated one-column lanes.
    forbidden = set(range(max(0, base_c - 1), min(panel_width, base_c + base_width + 1)))
    forbidden |= {c for c in range(panel_width) if abs(c - isolated_c) <= 1}
    lane_candidates = [c for c in range(1, panel_width - 1) if c not in forbidden]
    random.shuffle(lane_candidates)
    lanes: list[int] = []
    if object_count > 3:
        for c in lane_candidates:
            if all(abs(c - other) >= 2 for other in lanes):
                lanes.append(c)
            if len(lanes) == object_count - 3:
                break
    if len(lanes) != object_count - 3:
        return None

    for i, c in enumerate(lanes):
        shape = [(0, 0), (1, 0)] if random.choice((True, False)) else [(0, 0), (1, 0), (2, 0)]
        shape_h = len(shape)
        final_r = height - random.randint(7, 10)
        if final_r < 3 or final_r + shape_h >= height:
            return None
        lift = random.randint(1, min(3, final_r))
        specs.append((shape, (final_r - lift, c), (final_r, c), object_colors[3 + i], "obstacle"))

    input_sets: list[set[Coord]] = []
    final_sets: list[set[Coord]] = []
    for shape, source, destination, color, support in specs:
        src_cells = _translated(shape, source)
        dst_cells = _translated(shape, destination)
        if any(not (0 <= r < height and 0 <= c < panel_width) for r, c in src_cells | dst_cells):
            return None
        if any(src_cells & other for other in input_sets) or any(dst_cells & other for other in final_sets):
            return None
        input_sets.append(src_cells)
        final_sets.append(dst_cells)
        _paint_shape(left, shape, source, color)
        _paint_shape(expected, shape, destination, color)
        if support == "obstacle":
            bottom = max(r for r, _c in dst_cells)
            support_cols = sorted(c for r, c in dst_cells if r == bottom)
            support_c = random.choice(support_cols)
            right[bottom + 1, support_c] = obstacle_color

    # A mapped but irrelevant obstacle rules out treating every marker as visible paint.
    distractor_c = 0 if base_on_left else panel_width - 1
    distractor_r = random.randint(1, height - 2)
    right[distractor_r, distractor_c] = obstacle_color

    # Input objects must remain visibly separate and all obstacle pixels must be legal.
    comps = _components(left != bg)
    if len(comps) != object_count or any(len(cells) < 2 for cells in comps):
        return None
    if np.any((left != bg) & (right != bg)):
        return None

    divider = np.full((height, 1), divider_color, dtype=int)
    grid = np.concatenate((left, divider, right), axis=1)
    try:
        solved = main(grid)
    except (AssertionError, IndexError):
        return None
    if not np.array_equal(solved, expected):
        return None

    # Semantic checks: obstacles and sequential object support must each alter the answer.
    no_obstacles = _drop_objects(left, np.zeros_like(right, dtype=bool), bg)
    independent = _independent_drop(left, right != bg, bg)
    if np.array_equal(no_obstacles, expected) or np.array_equal(independent, expected):
        return None
    if np.any(expected == obstacle_color) or expected.shape != left.shape:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate bilateral invisible-support instance."""
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid invisible-support puzzle")
