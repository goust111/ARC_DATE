from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 9
MARKER_COLORS = (1, 2, 3, 4)

MIN_REGIONS = 4
MAX_REGIONS = 6
MIN_REGION_AREA = 15
MAX_REGION_AREA = 40
MIN_SPLIT_AREA = 12
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    n = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            n += 1
            labels[r, c] = n
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = n
                        q.append((nr, nc))
    return labels, n


def _chebyshev_depth(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    h, w = padded.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    outside = np.argwhere(~padded)
    for r, c in outside:
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1:
                dist[nr, nc] = nd
                q.append((nr, nc))

    depth = dist[1:-1, 1:-1] - 1
    depth[~mask] = -1
    return depth


def _build_region_graph(labels: np.ndarray, wall_mask: np.ndarray) -> dict[int, set[int]]:
    n_regions = int(labels.max())
    graph = {region_id: set() for region_id in range(1, n_regions + 1)}
    h, w = labels.shape

    for r, c in np.argwhere(wall_mask):
        rr = int(r)
        cc = int(c)
        if 0 < cc < w - 1:
            left = int(labels[rr, cc - 1])
            right = int(labels[rr, cc + 1])
            if left > 0 and right > 0 and left != right:
                graph[left].add(right)
                graph[right].add(left)
        if 0 < rr < h - 1:
            up = int(labels[rr - 1, cc])
            down = int(labels[rr + 1, cc])
            if up > 0 and down > 0 and up != down:
                graph[up].add(down)
                graph[down].add(up)
    return graph


def _bipartite_phases(labels: np.ndarray, wall_mask: np.ndarray) -> dict[int, int]:
    h, _ = labels.shape
    start = int(labels[h - 1, 0])
    assert start > 0, "lower-left cell must belong to a region"

    graph = _build_region_graph(labels, wall_mask)
    phases = {start: 0}
    q = deque([start])
    while q:
        region = q.popleft()
        for neighbor in graph[region]:
            expected = 1 - phases[region]
            if neighbor in phases:
                assert phases[neighbor] == expected, "region adjacency graph must be bipartite"
            else:
                phases[neighbor] = expected
                q.append(neighbor)

    expected_regions = set(range(1, int(labels.max()) + 1))
    assert set(phases) == expected_regions, "all regions must be connected through walls"
    return phases


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated regions with concentric palettes whose outer-layer phase alternates across neighboring regions.

    Input structure:
    - A black background is partitioned by one-cell-thick maroon walls into several 4-connected regions.
    - Each region contains one or more marker pixels chosen from blue, red, green, and yellow.
    - A marker's Chebyshev depth inside its region determines its position in that region's palette.

    Rule:
    - Keep every wall pixel unchanged and identify the non-wall regions plus their wall-adjacency graph.
    - For each region, sort marker colors by marker depth; if no marker lies on depth 0, prepend black to the palette.
    - Two wall-adjacent regions must use opposite phase offsets, anchored so the region containing the lower-left cell has phase 0.
    - Repaint each region cell with palette[(cell_depth + phase) mod palette_length], covering the original markers.

    Key insight:
    - The markers determine which colors cycle with depth, but the neighborhood graph determines whether a region starts that cycle in phase 0 or phase 1.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_mask = g == WALL
    free_mask = ~wall_mask
    labels, n_regions = _label_components(free_mask)
    assert n_regions >= 2

    phases = _bipartite_phases(labels, wall_mask)
    out = g.copy()

    for region_id in range(1, n_regions + 1):
        region_mask = labels == region_id
        depth = _chebyshev_depth(region_mask)
        marker_coords = np.argwhere(region_mask & (g != BG))
        assert marker_coords.size > 0, "each region must contain at least one marker"

        depth_to_color: dict[int, int] = {}
        for r, c in marker_coords:
            rr = int(r)
            cc = int(c)
            d = int(depth[rr, cc])
            color = int(g[rr, cc])
            if d in depth_to_color:
                assert depth_to_color[d] == color, "multiple marker colors at the same depth are ambiguous"
            else:
                depth_to_color[d] = color

        palette: list[int] = []
        if 0 not in depth_to_color:
            palette.append(BG)
        for d in sorted(depth_to_color):
            palette.append(depth_to_color[d])
        phase = phases[region_id]

        region_depths = depth[region_mask]
        out[region_mask] = np.array(
            [palette[(int(d) + phase) % len(palette)] for d in region_depths],
            dtype=int,
        )

    return out


def _vertical_path(mask: np.ndarray, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 < 4:
        return None

    top_cols = [c for c in range(c0, c1 + 1) if mask[r0, c]]
    bottom_cols = [c for c in range(c0, c1 + 1) if mask[r1, c]]
    if r0 == 0:
        top_cols = [c for c in top_cols if c not in (0, w - 1)]
    if r1 == h - 1:
        bottom_cols = [c for c in bottom_cols if c not in (0, w - 1)]
    interior_top = [c for c in top_cols if c0 < c < c1]
    interior_bottom = [c for c in bottom_cols if c0 < c < c1]
    if interior_top:
        top_cols = interior_top
    if interior_bottom:
        bottom_cols = interior_bottom
    if not top_cols or not bottom_cols:
        return None

    for _ in range(30):
        start_c = rng.choice(top_cols)
        end_c = rng.choice(bottom_cols)
        bend_r = rng.randint(r0 + 1, r1 - 1)
        path = [(r, start_c) for r in range(r0, bend_r + 1)]
        step = 1 if end_c >= start_c else -1
        path.extend((bend_r, c) for c in range(start_c + step, end_c + step, step))
        path.extend((r, end_c) for r in range(bend_r + 1, r1 + 1))
        if len(set(path)) != len(path):
            continue
        if all(mask[r, c] for r, c in path):
            return path
    return None


def _horizontal_path(mask: np.ndarray, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    r0, c0, r1, c1 = _bbox(mask)
    if c1 - c0 < 4:
        return None

    left_rows = [r for r in range(r0, r1 + 1) if mask[r, c0]]
    right_rows = [r for r in range(r0, r1 + 1) if mask[r, c1]]
    if c0 == 0:
        left_rows = [r for r in left_rows if r not in (0, h - 1)]
    if c1 == w - 1:
        right_rows = [r for r in right_rows if r not in (0, h - 1)]
    interior_left = [r for r in left_rows if r0 < r < r1]
    interior_right = [r for r in right_rows if r0 < r < r1]
    if interior_left:
        left_rows = interior_left
    if interior_right:
        right_rows = interior_right
    if not left_rows or not right_rows:
        return None

    for _ in range(30):
        start_r = rng.choice(left_rows)
        end_r = rng.choice(right_rows)
        bend_c = rng.randint(c0 + 1, c1 - 1)
        path = [(start_r, c) for c in range(c0, bend_c + 1)]
        step = 1 if end_r >= start_r else -1
        path.extend((r, bend_c) for r in range(start_r + step, end_r + step, step))
        path.extend((end_r, c) for c in range(bend_c + 1, c1 + 1))
        if len(set(path)) != len(path):
            continue
        if all(mask[r, c] for r, c in path):
            return path
    return None


def _region_areas(labels: np.ndarray, n_regions: int) -> list[int]:
    return [int(np.count_nonzero(labels == region_id)) for region_id in range(1, n_regions + 1)]


def _attempt_split(grid: np.ndarray, labels: np.ndarray, n_regions: int, target: int, rng: random.Random) -> np.ndarray | None:
    region_ids = list(range(1, n_regions + 1))
    weights = [np.count_nonzero(labels == region_id) for region_id in region_ids]
    chosen_region = rng.choices(region_ids, weights=weights, k=1)[0]
    region_mask = labels == chosen_region
    area = int(np.count_nonzero(region_mask))
    if area < 2 * MIN_SPLIT_AREA + 1:
        return None

    path_builders = [_vertical_path, _horizontal_path]
    rng.shuffle(path_builders)
    for build_path in path_builders:
        path = build_path(region_mask, rng)
        if path is None:
            continue
        candidate = grid.copy()
        for r, c in path:
            candidate[r, c] = WALL
        new_labels, new_count = _label_components(candidate != WALL)
        if new_count != n_regions + 1:
            continue
        areas = _region_areas(new_labels, new_count)
        if min(areas) < MIN_SPLIT_AREA:
            continue
        remaining = target - new_count
        if remaining == 0 and (min(areas) < MIN_REGION_AREA or max(areas) > MAX_REGION_AREA):
            continue
        return candidate
    return None


def _choose_marker_cells(
    region_mask: np.ndarray,
    depth: np.ndarray,
    chosen_depths: list[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    picked: list[tuple[int, int]] = []
    for d in chosen_depths:
        candidates = [tuple(map(int, rc)) for rc in np.argwhere(region_mask & (depth == d))]
        rng.shuffle(candidates)
        separated = [
            cell for cell in candidates
            if all(max(abs(cell[0] - pr), abs(cell[1] - pc)) > 1 for pr, pc in picked)
        ]
        if separated:
            picked.append(separated[0])
        elif candidates:
            picked.append(candidates[0])
        else:
            raise AssertionError("chosen depth has no available cell")
    return picked


def _decorate_regions(grid: np.ndarray, labels: np.ndarray, n_regions: int, rng: random.Random) -> np.ndarray | None:
    out = grid.copy()
    region_depths: dict[int, np.ndarray] = {}
    max_depths: dict[int, int] = {}
    eligible_without_zero: list[int] = []

    for region_id in range(1, n_regions + 1):
        region_mask = labels == region_id
        depth = _chebyshev_depth(region_mask)
        max_depth = int(depth[region_mask].max())
        region_depths[region_id] = depth
        max_depths[region_id] = max_depth
        if max_depth >= 1:
            eligible_without_zero.append(region_id)

    if not eligible_without_zero:
        return None

    force_base_region = rng.choice(eligible_without_zero)
    force_zero_region = rng.choice(list(range(1, n_regions + 1)))

    for region_id in range(1, n_regions + 1):
        region_mask = labels == region_id
        depth = region_depths[region_id]
        available = [d for d in range(0, min(3, max_depths[region_id]) + 1) if np.any(region_mask & (depth == d))]
        assert available

        if region_id == force_base_region:
            nonzero = [d for d in available if d > 0]
            count = rng.randint(1, min(3, len(nonzero)))
            chosen_depths = sorted(rng.sample(nonzero, count))
        else:
            include_zero = 0 in available and (region_id == force_zero_region or rng.random() < 0.6)
            pool = available.copy()
            chosen_depths: list[int]
            if include_zero:
                pool.remove(0)
                max_count = min(3, len(available))
                count = rng.randint(1, max_count)
                extras = rng.sample(pool, count - 1) if count > 1 else []
                chosen_depths = sorted([0, *extras])
            else:
                nonzero = [d for d in available if d > 0]
                if nonzero:
                    count = rng.randint(1, min(3, len(nonzero)))
                    chosen_depths = sorted(rng.sample(nonzero, count))
                else:
                    chosen_depths = [0]

        marker_colors = rng.sample(list(MARKER_COLORS), len(chosen_depths))
        marker_cells = _choose_marker_cells(region_mask, depth, chosen_depths, rng)
        for (r, c), color in zip(marker_cells, marker_colors):
            out[r, c] = color

    return out


def _apply_symmetry(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    op = rng.randrange(8)
    if op == 0:
        return grid
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.flipud(grid)
    if op == 5:
        return np.fliplr(grid)
    if op == 6:
        return np.transpose(grid)
    return np.fliplr(np.transpose(grid))


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _grid_from_bands(
    row_heights: list[int],
    col_widths: list[int],
    omit_vertical: tuple[int, int] | None = None,
    omit_horizontal: tuple[int, int] | None = None,
) -> np.ndarray:
    h = sum(row_heights) + len(row_heights) - 1
    w = sum(col_widths) + len(col_widths) - 1
    grid = np.full((h, w), BG, dtype=int)

    row_starts = []
    cur = 0
    for height in row_heights:
        row_starts.append(cur)
        cur += height + 1

    col_starts = []
    cur = 0
    for width in col_widths:
        col_starts.append(cur)
        cur += width + 1

    h_walls = []
    cur = row_heights[0]
    for idx in range(len(row_heights) - 1):
        h_walls.append(cur + idx)
        grid[cur + idx, :] = WALL
        cur += row_heights[idx + 1]

    v_walls = []
    cur = col_widths[0]
    for idx in range(len(col_widths) - 1):
        v_walls.append(cur + idx)
        grid[:, cur + idx] = WALL
        cur += col_widths[idx + 1]

    if omit_vertical is not None:
        row_idx, boundary_idx = omit_vertical
        wall_col = v_walls[boundary_idx]
        r0 = row_starts[row_idx]
        r1 = r0 + row_heights[row_idx] - 1
        grid[r0 : r1 + 1, wall_col] = BG

    if omit_horizontal is not None:
        boundary_idx, col_idx = omit_horizontal
        wall_row = h_walls[boundary_idx]
        c0 = col_starts[col_idx]
        c1 = c0 + col_widths[col_idx] - 1
        grid[wall_row, c0 : c1 + 1] = BG

    return grid


def _generate_partition(rng: random.Random) -> np.ndarray | None:
    target = rng.choices([4, 5, 6], weights=[2, 4, 4], k=1)[0]
    grid = None

    for _ in range(80):
        if target == 4:
            row_heights = [rng.randint(4, 6), rng.randint(4, 6)]
            col_widths = [rng.randint(4, 6), rng.randint(4, 6)]
            candidate = _grid_from_bands(row_heights, col_widths)
        elif target == 6:
            row_heights = [rng.randint(4, 6), rng.randint(4, 6)]
            col_widths = [rng.randint(4, 5), rng.randint(4, 5), rng.randint(4, 5)]
            candidate = _grid_from_bands(row_heights, col_widths)
        else:
            row_heights = [rng.randint(4, 5), rng.randint(4, 5)]
            col_widths = [rng.randint(3, 5), rng.randint(3, 5), rng.randint(3, 5)]
            if rng.random() < 0.5:
                omit_vertical = (rng.randrange(2), rng.randrange(2))
                candidate = _grid_from_bands(row_heights, col_widths, omit_vertical=omit_vertical)
            else:
                omit_horizontal = (0, rng.randrange(3))
                candidate = _grid_from_bands(row_heights, col_widths, omit_horizontal=omit_horizontal)

        labels, n_regions = _label_components(candidate != WALL)
        if n_regions != target:
            continue
        areas = _region_areas(labels, n_regions)
        if min(areas) < MIN_REGION_AREA or max(areas) > MAX_REGION_AREA:
            continue
        try:
            _bipartite_phases(labels, candidate == WALL)
        except AssertionError:
            continue
        grid = candidate
        break

    if grid is None:
        return None

    labels, n_regions = _label_components(grid != WALL)
    if n_regions != target:
        return None

    areas = _region_areas(labels, n_regions)
    if min(areas) < MIN_REGION_AREA or max(areas) > MAX_REGION_AREA:
        return None

    if int(grid[-1, 0]) == WALL:
        return None

    try:
        _bipartite_phases(labels, grid == WALL)
    except AssertionError:
        return None

    decorated = _decorate_regions(grid, labels, n_regions, rng)
    if decorated is None:
        return None

    transformed = _apply_symmetry(decorated, rng)
    if int(transformed[-1, 0]) == WALL:
        return None
    return transformed


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_partition(rng)
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
