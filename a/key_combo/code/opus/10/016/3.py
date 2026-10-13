from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 2
GRAY = 5
ORANGE = 7
CYAN = 8

NW_SE = "nw_se"
NE_SW = "ne_sw"
ORIENTATIONS = (NW_SE, NE_SW)

GENERATE_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (12, 18)
REGION_COUNT_RANGE = (5, 8)
MAX_SIZE = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            labels[r, c] = next_label
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
    return labels, next_label


def _classify_gray_pixels(grid: np.ndarray) -> tuple[dict[tuple[int, int], str], dict[str, int]]:
    h, w = grid.shape
    orientations: dict[tuple[int, int], str] = {}
    totals = {NW_SE: 0, NE_SW: 0}
    for r0, c0 in np.argwhere(grid == GRAY):
        r, c = int(r0), int(c0)
        has_nw_se = (
            (r > 0 and c > 0 and grid[r - 1, c - 1] == GRAY)
            or (r + 1 < h and c + 1 < w and grid[r + 1, c + 1] == GRAY)
        )
        has_ne_sw = (
            (r > 0 and c + 1 < w and grid[r - 1, c + 1] == GRAY)
            or (r + 1 < h and c > 0 and grid[r + 1, c - 1] == GRAY)
        )
        assert has_nw_se ^ has_ne_sw, "each gray pixel must belong to exactly one diagonal orientation"
        orientation = NW_SE if has_nw_se else NE_SW
        orientations[(r, c)] = orientation
        totals[orientation] += 1
    return orientations, totals


def _build_region_graph(labels: np.ndarray, grid: np.ndarray, region_count: int) -> dict[int, set[int]]:
    h, w = grid.shape
    graph = {region_id: set() for region_id in range(1, region_count + 1)}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != WALL:
                continue
            if 0 < r < h - 1 and grid[r - 1, c] != WALL and grid[r + 1, c] != WALL:
                a = int(labels[r - 1, c])
                b = int(labels[r + 1, c])
                if a != b:
                    graph[a].add(b)
                    graph[b].add(a)
            if 0 < c < w - 1 and grid[r, c - 1] != WALL and grid[r, c + 1] != WALL:
                a = int(labels[r, c - 1])
                b = int(labels[r, c + 1])
                if a != b:
                    graph[a].add(b)
                    graph[b].add(a)
    return graph


def _solve_metadata(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    labels, region_count = _label_components(g != WALL)
    assert REGION_COUNT_RANGE[0] <= region_count <= REGION_COUNT_RANGE[1]

    gray_orientations, orientation_totals = _classify_gray_pixels(g)
    assert orientation_totals[NW_SE] > 0 and orientation_totals[NE_SW] > 0
    assert orientation_totals[NW_SE] != orientation_totals[NE_SW]
    majority = NW_SE if orientation_totals[NW_SE] > orientation_totals[NE_SW] else NE_SW

    majority_counts = {region_id: 0 for region_id in range(1, region_count + 1)}
    gray_totals = {region_id: 0 for region_id in range(1, region_count + 1)}
    for (r, c), orientation in gray_orientations.items():
        region_id = int(labels[r, c])
        gray_totals[region_id] += 1
        if orientation == majority:
            majority_counts[region_id] += 1

    max_majority = max(majority_counts.values())
    anchor_regions = [region_id for region_id, count in majority_counts.items() if count == max_majority]
    assert len(anchor_regions) == 1, "anchor region must be unique"
    anchor = anchor_regions[0]

    graph = _build_region_graph(labels, g, region_count)
    parity = {anchor: 0}
    queue = deque([anchor])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            needed = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == needed, "region adjacency must be bipartite"
            else:
                parity[neighbor] = needed
                queue.append(neighbor)
    assert len(parity) == region_count, "all regions must be connected through wall adjacencies"

    return labels, graph, parity, majority, majority_counts, gray_totals, anchor


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    labels, _graph, parity, _majority, _majority_counts, _gray_totals, _anchor = _solve_metadata(grid)
    out = np.array(grid, dtype=int, copy=True)
    for region_id, region_parity in parity.items():
        fill_color = CYAN if region_parity == 0 else ORANGE
        mask = (labels == region_id) & (grid == BG)
        out[mask] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote on diagonal gray-bar orientation to choose the anchor region for alternating wall-region coloring.

    Input structure:
    - Black cells are background, red cells are one-cell-thick walls, and some non-wall
      regions contain gray diagonal bar pixels.
    - The red walls partition the non-wall area into 5-8 4-connected regions whose
      wall-adjacency graph is bipartite.
    - Every gray pixel belongs to exactly one diagonal orientation: NW-SE or NE-SW.

    Rule:
    - Classify each gray pixel by which diagonal axis contains its gray neighbor.
    - Count the two orientations globally and treat the larger one as the majority orientation.
    - Among the wall-separated non-wall regions, choose the unique region containing the
      most majority-orientation gray pixels as the anchor.
    - Fill anchor-region background cells cyan, then alternate orange/cyan across the
      region adjacency graph while keeping walls and gray pixels unchanged.

    Key insight:
    - The global orientation vote does not directly recolor the gray bars; it only decides
      which region becomes the bipartite-coloring anchor.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _segment_lengths(count: int) -> list[int]:
    if count == 0:
        return []
    if count == 2:
        return [2]
    if count == 3:
        return [3]
    if count == 4:
        return [2, 2]
    if count == 5:
        return [3, 2]
    raise ValueError(f"unsupported gray count: {count}")


def _min_width_for_pair(pair: tuple[int, int]) -> int:
    majority_count, minority_count = pair
    if majority_count and minority_count:
        return 4
    return 3 if max(pair) >= 3 else 2


def _segment_cells(orientation: str, length: int, r0: int, c0: int) -> list[tuple[int, int]]:
    if orientation == NW_SE:
        return [(r0 + i, c0 + i) for i in range(length)]
    return [(r0 + i, c0 - i) for i in range(length)]


def _place_region_segments(
    grid: np.ndarray,
    left: int,
    right: int,
    count_pair: tuple[int, int],
    majority_orientation: str,
) -> bool:
    segments: list[tuple[str, int]] = []
    minority_orientation = NE_SW if majority_orientation == NW_SE else NW_SE
    for length in _segment_lengths(count_pair[0]):
        segments.append((majority_orientation, length))
    for length in _segment_lengths(count_pair[1]):
        segments.append((minority_orientation, length))
    segments.sort(key=lambda item: (-item[1], random.random()))

    h = grid.shape[0]
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for orientation, length in segments:
        candidates: list[list[tuple[int, int]]] = []
        for r0 in range(0, h - length + 1):
            if orientation == NW_SE:
                col_range = range(left, right - length + 2)
            else:
                col_range = range(left + length - 1, right + 1)
            for c0 in col_range:
                coords = _segment_cells(orientation, length, r0, c0)
                if any((r, c) in blocked for r, c in coords):
                    continue
                candidates.append(coords)
        random.shuffle(candidates)
        if not candidates:
            return False
        chosen = candidates[0]
        for r, c in chosen:
            occupied.add((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and left <= cc <= right:
                        blocked.add((rr, cc))

    for r, c in occupied:
        grid[r, c] = GRAY
    return True


def _region_intervals(widths: list[int]) -> tuple[list[tuple[int, int]], int]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for idx, width in enumerate(widths):
        intervals.append((cursor, cursor + width - 1))
        cursor += width
        if idx != len(widths) - 1:
            cursor += 1
    return intervals, cursor


def _dihedral_transform(grid: np.ndarray) -> np.ndarray:
    op = random.randrange(8)
    if op == 0:
        return grid
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.fliplr(grid)
    if op == 5:
        return np.flipud(grid)
    if op == 6:
        return grid.T.copy()
    return np.fliplr(np.rot90(grid, 1))


def _choose_plan() -> list[tuple[int, int]]:
    plans = [
        [(4, 0), (2, 3), (3, 0)],
        [(3, 0), (0, 5), (2, 0), (2, 0)],
        [(3, 2), (2, 3), (2, 0)],
    ]
    return random.choice(plans)


def _build_widths(region_count: int, plan_pairs: list[tuple[int, int]], marked_regions: list[int]) -> list[int] | None:
    widths = [1] * region_count
    for region_id, pair in zip(marked_regions, plan_pairs):
        widths[region_id] = _min_width_for_pair(pair)

    wall_count = region_count - 1
    base_total = sum(widths) + wall_count
    if base_total > MAX_SIZE:
        return None

    budget = MAX_SIZE - base_total
    order = list(range(region_count))
    random.shuffle(order)
    for region_id in order:
        cap = 5 if region_id in marked_regions else 3
        while widths[region_id] < cap and budget > 0 and random.random() < 0.55:
            widths[region_id] += 1
            budget -= 1
    return widths


def _plan_totals(plan_pairs: list[tuple[int, int]]) -> tuple[int, int]:
    majority_total = sum(pair[0] for pair in plan_pairs)
    minority_total = sum(pair[1] for pair in plan_pairs)
    return majority_total, minority_total


def _attempt_grid() -> np.ndarray | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    plan_pairs = _choose_plan()
    if len(plan_pairs) > region_count:
        return None

    marked_regions = random.sample(range(region_count), len(plan_pairs))
    widths = _build_widths(region_count, plan_pairs, marked_regions)
    if widths is None:
        return None

    height = random.randint(*GRID_HEIGHT_RANGE)
    intervals, width = _region_intervals(widths)
    if max(height, width) > MAX_SIZE:
        return None

    majority_orientation = random.choice(ORIENTATIONS)
    majority_total, minority_total = _plan_totals(plan_pairs)
    if majority_total <= minority_total:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for left, right in intervals[:-1]:
        grid[:, right + 1] = WALL

    for region_id, pair in zip(marked_regions, plan_pairs):
        left, right = intervals[region_id]
        if not _place_region_segments(grid, left, right, pair, majority_orientation):
            return None

    transformed = _dihedral_transform(grid.copy())
    try:
        labels, _graph, parity, actual_majority, majority_counts, gray_totals, anchor = _solve_metadata(transformed)
    except Exception:
        return None

    if actual_majority not in ORIENTATIONS:
        return None

    anchor_count = majority_counts[anchor]
    if sum(count == anchor_count for count in majority_counts.values()) != 1:
        return None

    non_anchor_totals = [gray_totals[region_id] for region_id in gray_totals if region_id != anchor]
    if not non_anchor_totals or max(non_anchor_totals) < gray_totals[anchor]:
        return None

    if not np.any(transformed == WALL) or not np.any(transformed == GRAY):
        return None

    solved = _solve_grid(transformed)
    if solved.shape != transformed.shape:
        return None
    if not np.array_equal(main(transformed), solved):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
