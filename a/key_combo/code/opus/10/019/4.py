from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CONVEX_TO_COLOR = {
    4: BLUE,
    6: RED,
    8: GREEN,
    10: YELLOW,
}
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _append_line(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 != r1 and c0 != c1:
        raise ValueError("segments must be axis-aligned")
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _extract_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray | None], list[bool]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    regions: list[np.ndarray | None] = [None]
    touches_border = [False]
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            cells: list[tuple[int, int]] = []
            border = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            regions.append(np.array(cells, dtype=int))
            touches_border.append(border)
            next_id += 1
    return labels, regions, touches_border


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, cell)) for cell in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        next_cells = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(next_cells) == 1, (cur, next_cells)
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _convex_corner_count(path: list[tuple[int, int]]) -> int:
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        counts[1 if cross > 0 else -1] += 1
    assert counts, "loop must have bends"
    assert abs(counts[1] - counts[-1]) == 4, counts
    return max(counts[1], counts[-1])


def _interior_region_ids(grid: np.ndarray, loop_path: list[tuple[int, int]]) -> tuple[np.ndarray, list[np.ndarray | None], list[bool], set[int]]:
    labels, regions, touches_border = _label_regions(grid == BLACK)
    interior_ids: set[int] = set()
    h, w = grid.shape
    for r, c in loop_path:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK:
                region_id = int(labels[nr, nc])
                if region_id != 0 and not touches_border[region_id]:
                    interior_ids.add(region_id)
    return labels, regions, touches_border, interior_ids


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the interior of each grey closed loop using the color mapped from its convex-corner count.

    Input structure:
    - The grid has a black background and 3-4 disjoint one-pixel-wide grey orthogonal loops.
    - Every loop is 4-connected, closed, away from the border, and each grey pixel has exactly two grey neighbors.
    - Each loop encloses one black interior region, and different loops have different convex-corner counts.

    Rule:
    - Trace every grey loop and classify its bend pixels by turn sign.
    - The majority turn sign marks the convex bends; count them to get C in {4, 6, 8, 10}.
    - Find the unique enclosed black region adjacent to that loop and fill it with the color mapped from C: 4->blue, 6->red, 8->green, 10->yellow.
    - Keep all grey loop pixels unchanged.

    Key insight:
    - The output color of an enclosed region is determined by the geometry of its surrounding boundary, not by the region's own size or shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    loop_components = _extract_components(g == GREY)
    assert loop_components

    for coords in loop_components:
        path = _trace_cycle(coords)
        convex_count = _convex_corner_count(path)
        assert convex_count in CONVEX_TO_COLOR, convex_count
        fill_color = CONVEX_TO_COLOR[convex_count]

        _labels, regions, _touches_border, interior_ids = _interior_region_ids(g, path)
        assert len(interior_ids) == 1, interior_ids
        region = regions[next(iter(interior_ids))]
        assert region is not None and int(region.shape[0]) >= 2
        out[region[:, 0], region[:, 1]] = fill_color

    return out


def _build_loop_path(height: int, width: int, notches: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur_col = 0
    for c0, c1, depth in sorted(notches):
        _append_line(path, 0, cur_col, 0, c0)
        _append_line(path, 0, c0, depth, c0)
        _append_line(path, depth, c0, depth, c1)
        _append_line(path, depth, c1, 0, c1)
        cur_col = c1
    _append_line(path, 0, cur_col, 0, width - 1)
    _append_line(path, 0, width - 1, height - 1, width - 1)
    _append_line(path, height - 1, width - 1, height - 1, 0)
    _append_line(path, height - 1, 0, 0, 0)
    if path[-1] == path[0]:
        path.pop()
    return path


def _loop_patch_from_convex_count(convex_count: int) -> np.ndarray:
    notch_count = (convex_count - 4) // 2
    assert convex_count in CONVEX_TO_COLOR
    assert notch_count >= 0

    if notch_count == 0:
        height = random.randint(4, 7)
        width = random.randint(4, 9)
        path = _build_loop_path(height, width, [])
    else:
        height = random.randint(6, 8)
        max_depth = max(2, height - 4)
        depths = [random.randint(2, max_depth) for _ in range(notch_count)]

        spans = [random.randint(2, 3) for _ in range(notch_count)]
        left_margin = random.randint(2, 3)
        right_margin = random.randint(2, 3)
        gap = 3

        notches: list[tuple[int, int, int]] = []
        c0 = left_margin
        for span, depth in zip(spans, depths):
            c1 = c0 + span
            notches.append((c0, c1, depth))
            c0 = c1 + gap
        width = notches[-1][1] + right_margin + 1
        path = _build_loop_path(height, width, notches)

    patch = np.full((height, width), BLACK, dtype=int)
    for r, c in path:
        patch[r, c] = GREY

    if random.choice([False, True]):
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randint(0, 3))

    loop_components = _extract_components(patch == GREY)
    assert len(loop_components) == 1
    actual_count = _convex_corner_count(_trace_cycle(loop_components[0]))
    assert actual_count == convex_count, (actual_count, convex_count)

    return patch


def _slot_positions(n_loops: int) -> list[tuple[int, int]]:
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return random.sample(slots, n_loops)


def _layout_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    positions = _slot_positions(len(patches))
    assignment = list(zip(random.sample(patches, len(patches)), positions))

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    row_gap = random.randint(2, 3)
    col_gap = random.randint(2, 3)

    row_heights = [0, 0]
    col_widths = [0, 0]
    for patch, pos in assignment:
        extra_h = random.randint(0, 2)
        extra_w = random.randint(0, 2)
        row_heights[pos[0]] = max(row_heights[pos[0]], patch.shape[0] + extra_h)
        col_widths[pos[1]] = max(col_widths[pos[1]], patch.shape[1] + extra_w)

    height = top_margin + row_heights[0] + row_gap + row_heights[1] + bottom_margin
    width = left_margin + col_widths[0] + col_gap + col_widths[1] + right_margin
    if height > 30 or width > 30:
        return None

    row_origins = [top_margin, top_margin + row_heights[0] + row_gap]
    col_origins = [left_margin, left_margin + col_widths[0] + col_gap]

    grid = np.full((height, width), BLACK, dtype=int)
    for patch, (slot_r, slot_c) in assignment:
        free_r = row_heights[slot_r] - patch.shape[0]
        free_c = col_widths[slot_c] - patch.shape[1]
        assert free_r >= 0 and free_c >= 0
        offset_r = random.randint(0, free_r)
        offset_c = random.randint(0, free_c)
        r0 = row_origins[slot_r] + offset_r
        c0 = col_origins[slot_c] + offset_c
        mask = patch == GREY
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = GREY

    return grid


def _loop_stats(grid: np.ndarray) -> list[tuple[int, int]]:
    stats: list[tuple[int, int]] = []
    for coords in _extract_components(grid == GREY):
        path = _trace_cycle(coords)
        convex_count = _convex_corner_count(path)
        _labels, regions, _touches_border, interior_ids = _interior_region_ids(grid, path)
        if len(interior_ids) != 1:
            raise AssertionError(interior_ids)
        region = regions[next(iter(interior_ids))]
        assert region is not None
        stats.append((convex_count, int(region.shape[0])))
    return stats


def _passes_generator_constraints(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, GREY}:
        return False

    grey_coords = np.argwhere(grid == GREY)
    if grey_coords.size == 0:
        return False
    if np.any(grey_coords[:, 0] == 0) or np.any(grey_coords[:, 0] == grid.shape[0] - 1):
        return False
    if np.any(grey_coords[:, 1] == 0) or np.any(grey_coords[:, 1] == grid.shape[1] - 1):
        return False

    stats = _loop_stats(grid)
    if not (3 <= len(stats) <= 4):
        return False

    counts = [item[0] for item in stats]
    if len(set(counts)) != len(counts):
        return False
    if any(count not in CONVEX_TO_COLOR for count in counts):
        return False

    areas_in_count_order = [area for _count, area in sorted(stats)]
    if min(areas_in_count_order) < 2:
        return False
    if areas_in_count_order == sorted(areas_in_count_order):
        return False
    if areas_in_count_order == sorted(areas_in_count_order, reverse=True):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_loops = random.choice([3, 4, 4])
        chosen_counts = random.sample(sorted(CONVEX_TO_COLOR), n_loops)
        try:
            patches = [_loop_patch_from_convex_count(count) for count in chosen_counts]
        except AssertionError:
            continue
        grid = _layout_patches(patches)
        if grid is None:
            continue
        if _passes_generator_constraints(grid):
            return grid
    raise RuntimeError("failed to generate a valid loop-coloring input")
