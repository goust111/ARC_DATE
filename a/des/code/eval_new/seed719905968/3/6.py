from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))
GEN_RETRIES = 600
MIN_SIDE_CHOICES = (2, 3)
REGION_COUNT_RANGE = (5, 9)
MAIN_INNER_H_RANGE = (10, 15)
GRID_W_RANGE = (18, 26)
LEGEND_LENGTH_COUNT_RANGE = (3, 4)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    label = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            labels[r, c] = label
            queue = deque([(r, c)])
            coords = [(r, c)]
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
                        coords.append((nr, nc))
            components.append(coords)

    return labels, components


def _count_wall_run(grid: np.ndarray, wall: int, r: int, c: int, dc: int) -> int:
    h, w = grid.shape
    length = 0
    cc = c + dc
    while 0 <= cc < w and int(grid[r, cc]) == wall:
        length += 1
        cc += dc
    return length


def _read_legend(grid: np.ndarray, bg: int, wall: int) -> tuple[dict[int, int], np.ndarray]:
    color_mask = (grid != bg) & (grid != wall)
    color_coords = np.argwhere(color_mask)
    assert color_coords.size, "expected legend color pixels"

    mapping: dict[int, int] = {}
    for r, c in color_coords:
        left = _count_wall_run(grid, wall, int(r), int(c), -1)
        right = _count_wall_run(grid, wall, int(r), int(c), 1)
        if left > 0 and right == 0:
            length = left
        elif right > 0 and left == 0:
            length = right
        else:
            raise AssertionError("legend color pixel must touch exactly one wall segment")
        mapping[int(length)] = int(grid[int(r), int(c)])

    return mapping, color_mask


def _longest_shared_boundary_lengths(
    grid: np.ndarray,
    labels: np.ndarray,
    region_ids: set[int],
    wall: int,
) -> dict[int, int]:
    h, w = grid.shape
    pair_at: list[list[tuple[int, int] | None]] = [[None] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != wall:
                continue
            adjacent = set()
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    rid = int(labels[nr, nc])
                    if rid in region_ids:
                        adjacent.add(rid)
            if len(adjacent) == 2:
                a, b = sorted(adjacent)
                pair_at[r][c] = (int(a), int(b))

    visited = np.zeros((h, w), dtype=bool)
    longest = {rid: 0 for rid in region_ids}
    for r in range(h):
        for c in range(w):
            pair = pair_at[r][c]
            if pair is None or visited[r, c]:
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            length = 0
            while queue:
                cr, cc = queue.popleft()
                length += 1
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and pair_at[nr][nc] == pair:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            a, b = pair
            longest[a] = max(longest[a], length)
            longest[b] = max(longest[b], length)

    return longest


def _region_analysis(grid: np.ndarray, bg: int, wall: int, color_mask: np.ndarray) -> tuple[np.ndarray, list[int], dict[int, int]]:
    bg_labels, _ = _label_components(grid == bg)
    legend_ids: set[int] = set()

    for r, c in np.argwhere(color_mask):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                rid = int(bg_labels[nr, nc])
                if rid != 0:
                    legend_ids.add(rid)

    region_ids = sorted(
        rid
        for rid in np.unique(bg_labels)
        if rid != 0 and int(rid) not in legend_ids
    )
    assert region_ids, "expected main-map background regions"

    longest = _longest_shared_boundary_lengths(grid, bg_labels, set(region_ids), wall)
    return bg_labels, region_ids, longest


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    values, counts = np.unique(g[g != bg], return_counts=True)
    assert values.size, "expected non-background structure"
    wall = int(values[np.argmax(counts)])

    mapping, color_mask = _read_legend(g, bg, wall)
    labels, region_ids, longest = _region_analysis(g, bg, wall, color_mask)

    out = g.copy()
    for rid in region_ids:
        length = int(longest[rid])
        assert length in mapping, "region longest boundary must be present in legend"
        out[labels == rid] = int(mapping[length])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed map region using the legend color keyed by its longest shared boundary.

    Input structure:
    - bg is the most common color.
    - A single wall color forms one-pixel grey borders that partition the main map into several enclosed regions.
    - A corner legend area is separated from the map by a double wall; each legend entry is a colored pixel adjacent to a wall segment whose visible length is the lookup key.

    Rule:
    - Read the legend as a mapping from wall-segment length to color.
    - Treat bg connected components outside the legend as the map regions.
    - For each region, collect every continuous wall segment that separates it from one neighboring region, find the longest such shared segment, and fill the whole region with the legend color for that length.
    - Keep all wall pixels and the legend itself unchanged.

    Key insight:
    - The deciding feature is not region area or neighbor count, but the length of the single longest shared border segment; the legend turns that geometric measurement into a color.
    """
    return _solve(grid)


def _render_partition(main_inner_h: int, width: int, wall: int, bg: int, target_regions: int, min_side: int) -> np.ndarray | None:
    map_h = main_inner_h + 3
    grid = np.full((map_h, width), bg, dtype=int)
    grid[0, :] = wall
    grid[main_inner_h + 1, :] = wall
    grid[main_inner_h + 2, :] = wall
    grid[:, 0] = wall
    grid[:, width - 1] = wall

    rects = [(1, main_inner_h, 1, width - 2)]

    def valid_orientations(rect: tuple[int, int, int, int]) -> list[str]:
        top, bottom, left, right = rect
        options: list[str] = []
        if bottom - top + 1 >= 2 * min_side + 1:
            options.append("H")
        if right - left + 1 >= 2 * min_side + 1:
            options.append("V")
        return options

    while len(rects) < target_regions:
        splittable = [(idx, rect, valid_orientations(rect)) for idx, rect in enumerate(rects)]
        splittable = [(idx, rect, options) for idx, rect, options in splittable if options]
        if not splittable:
            return None

        idx, rect, options = random.choice(splittable)
        top, bottom, left, right = rect
        orientation = random.choice(options)

        if orientation == "H":
            split = random.randint(top + min_side, bottom - min_side)
            grid[split, left - 1 : right + 2] = wall
            new_rects = [(top, split - 1, left, right), (split + 1, bottom, left, right)]
        else:
            split = random.randint(left + min_side, right - min_side)
            grid[top - 1 : bottom + 2, split] = wall
            new_rects = [(top, bottom, left, split - 1), (top, bottom, split + 1, right)]

        rects[idx : idx + 1] = new_rects

    return grid


def _candidate_lengths(map_grid: np.ndarray, bg: int, wall: int) -> tuple[list[int], dict[int, int]]:
    labels, components = _label_components(map_grid == bg)
    region_ids = {rid for rid in range(1, len(components) + 1)}
    longest = _longest_shared_boundary_lengths(map_grid, labels, region_ids, wall)
    lengths = sorted(set(longest.values()))
    return lengths, longest


def _assemble_grid(map_grid: np.ndarray, bg: int, wall: int, mapping: dict[int, int]) -> np.ndarray:
    map_h, width = map_grid.shape
    legend_lengths = sorted(mapping)
    legend_inner_h = 2 * len(legend_lengths) + 1
    total_h = map_h + legend_inner_h + 1
    assert total_h <= 30

    grid = np.full((total_h, width), wall, dtype=int)
    grid[:map_h, :] = map_grid

    chamber_top = map_h
    chamber_bottom = total_h - 2
    grid[chamber_top : chamber_bottom + 1, 1 : width - 1] = bg
    grid[:, width - 1] = wall
    grid[total_h - 1, :] = wall

    for idx, length in enumerate(legend_lengths):
        r = chamber_top + 1 + 2 * idx
        grid[r, 2 : 2 + length] = wall
        grid[r, 2 + length] = int(mapping[length])

    if random.random() < 0.5:
        grid = np.flipud(grid).copy()

    return grid


def _generate_attempt() -> np.ndarray | None:
    bg, wall = random.sample(ARC_COLORS, 2)
    width = random.randint(*GRID_W_RANGE)
    main_inner_h = random.randint(*MAIN_INNER_H_RANGE)
    target_regions = random.randint(*REGION_COUNT_RANGE)
    min_side = random.choice(MIN_SIDE_CHOICES)

    map_grid = _render_partition(main_inner_h, width, wall, bg, target_regions, min_side)
    if map_grid is None:
        return None

    lengths, longest = _candidate_lengths(map_grid, bg, wall)
    if not (LEGEND_LENGTH_COUNT_RANGE[0] <= len(lengths) <= LEGEND_LENGTH_COUNT_RANGE[1]):
        return None
    if min(lengths, default=0) <= 0:
        return None
    if max(lengths) > width - 5:
        return None

    usable_colors = [c for c in ARC_COLORS if c not in (bg, wall)]
    legend_colors = random.sample(usable_colors, len(lengths))
    mapping = {length: color for length, color in zip(lengths, legend_colors)}

    grid = _assemble_grid(map_grid, bg, wall, mapping)
    if max(grid.shape) > 30:
        return None

    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    if count_map.get(bg, 0) <= count_map.get(wall, 0):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    color_mask = (grid != bg) & (grid != wall)
    labels, region_ids, analyzed = _region_analysis(grid, bg, wall, color_mask)
    if len(region_ids) != target_regions:
        return None
    if sorted(set(analyzed[rid] for rid in region_ids)) != lengths:
        return None
    if any(int(out[labels == rid][0]) != mapping[analyzed[rid]] for rid in region_ids):
        return None
    if len(set(int(out[labels == rid][0]) for rid in region_ids)) < len(lengths):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
