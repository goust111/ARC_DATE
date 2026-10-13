import numpy as np
from typing import Dict, List, Tuple


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _connected_components(
    grid: np.ndarray, bg: int, row_mask: np.ndarray
) -> List[List[Tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not row_mask[r, c] or grid[r, c] == bg or visited[r, c]:
                continue

            comp_color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in _DIRS:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(nr, nc, h, w):
                        continue
                    if visited[nr, nc]:
                        continue
                    if not row_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != comp_color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_legend_templates(
    grid: np.ndarray, bg: int
) -> Tuple[Dict[int, np.ndarray], int]:
    h, w = grid.shape
    candidates = []
    if h >= 3:
        candidates.append((0, np.arange(0, 3)))
    if h >= 6:
        candidates.append((h - 3, np.arange(h - 3, h)))

    for start_row, row_indices in candidates:
        row_mask = np.zeros((h, w), dtype=bool)
        row_mask[row_indices, :] = True
        comps = _connected_components(grid, bg, row_mask)

        patterns: Dict[int, np.ndarray] = {}
        for comp in comps:
            color = int(grid[comp[0][0], comp[0][1]])
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)

            if max_r - min_r >= 3 or max_c - min_c >= 3:
                continue

            tpl = np.zeros((3, 3), dtype=np.uint8)
            for rr, cc in comp:
                rr0, cc0 = rr - min_r, cc - min_c
                tpl[rr0, cc0] = 1
            if tpl.sum() == 0:
                continue

            patterns[color] = tpl

        if patterns:
            return patterns, start_row

    return {}, h - 3


def _find_seed_pixels(
    grid: np.ndarray, bg: int, templates: Dict[int, np.ndarray], legend_row: int
) -> Tuple[List[Tuple[int, int, int, int, int, int, int]], List[int]]:
    h, w = grid.shape
    legend_rows = {legend_row, legend_row + 1, legend_row + 2}
    template_colors = set(templates.keys())

    seeds = []
    marker_colors = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in template_colors:
                continue
            if r in legend_rows:
                continue

            neigh = []
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if not _in_bounds(nr, nc, h, w):
                    continue
                if int(grid[nr, nc]) == bg:
                    continue
                neigh.append((nr, nc, int(grid[nr, nc])))

            if len(neigh) != 1:
                continue
            mr, mc, marker_color = neigh[0]
            if mr in legend_rows:
                continue
            dr, dc = mr - r, mc - c
            if abs(dr) + abs(dc) != 1:
                continue

            seeds.append((r, c, color, dr, dc, mr, mc))
            marker_colors.append(marker_color)

    return seeds, marker_colors


def _paint_pattern(
    out: np.ndarray, seed_r: int, seed_c: int, color: int, dr: int, dc: int, tpl: np.ndarray, bg: int
) -> None:
    h, w = out.shape
    top_left_r = seed_r + 2 * dr - 1
    top_left_c = seed_c + 2 * dc - 1

    for i in range(3):
        for j in range(3):
            if not tpl[i, j]:
                continue
            rr, cc = top_left_r + i, top_left_c + j
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == bg:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored 3x3 sprites from seeds using a direction marker and legend template matching.
    Input structure:
    - A black background.
    - A 3-row legend strip at the top or bottom with one-color 3x3 templates.
    - Single-pixel seeds in main area sharing a legend color, each with exactly one adjacent gray marker pixel indicating growth direction.
    Rule:
    - Match each seed color to its legend template of the same color.
    - Place that template in the main area by moving two cells from seed along the marker direction so the template center is at that point.
    - Scan seeds in row-major order and paint template pixels only on currently empty cells; occupied cells remain unchanged.
    - Remove marker pixels in the output while keeping all seeds.
    Key insight:
    - The solution requires three inferences together: color-to-template lookup from the legend, direction from marker adjacency, and deterministic scan-order conflict resolution.
    """
    grid = np.array(grid, dtype=np.int64, copy=False)
    bg = _most_common_color(grid)
    templates, legend_row = _extract_legend_templates(grid, bg)
    if not templates:
        return grid.copy()

    seeds, _ = _find_seed_pixels(grid, bg, templates, legend_row)
    if not seeds:
        return grid.copy()

    seeds.sort(key=lambda s: (s[0], s[1]))
    out = grid.copy()

    for r, c, _, _, _, mr, mc in seeds:
        out[mr, mc] = bg

    for r, c, color, dr, dc, _, _ in seeds:
        tpl = templates.get(color)
        if tpl is None:
            continue
        _paint_pattern(out, r, c, color, dr, dc, tpl, bg)

    return out


def _growth_cells(
    seed_r: int, seed_c: int, dr: int, dc: int, tpl: np.ndarray
) -> List[Tuple[int, int]]:
    top_left_r = seed_r + 2 * dr - 1
    top_left_c = seed_c + 2 * dc - 1
    cells: List[Tuple[int, int]] = []
    locs = np.argwhere(tpl == 1)
    for i, j in locs:
        cells.append((top_left_r + int(i), top_left_c + int(j)))
    return cells


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = 0
    marker = 8

    templates_pool = [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.uint8),
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[0, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
    ]

    h = int(rng.integers(14, 29))
    legend_top = bool(rng.integers(0, 2))
    if h < 6:
        h = 6
    if h >= 6:
        legend_row = 0 if legend_top else h - 3

    n_shapes = int(rng.integers(2, 5))
    shape_idxs = rng.choice(len(templates_pool), size=n_shapes, replace=False)
    shape_templates = [templates_pool[i] for i in shape_idxs]

    color_pool = [1, 2, 3, 4, 5, 6, 7, 9]
    shape_colors = list(rng.choice(color_pool, size=n_shapes, replace=False))

    legend_w = n_shapes * 3 + (n_shapes - 1)
    w = int(rng.integers(max(legend_w + 2, 12), 31))
    left = int(rng.integers(0, w - legend_w + 1))

    grid = np.full((h, w), bg, dtype=np.int64)

    for idx, color in enumerate(shape_colors):
        tpl = shape_templates[idx]
        base_r = legend_row
        base_c = left + idx * 4
        for i, j in np.argwhere(tpl == 1):
            grid[base_r + int(i), base_c + int(j)] = color

    if legend_top:
        main_top, main_bottom = 3, h - 1
    else:
        main_top, main_bottom = 0, h - 4

    seed_count = int(rng.integers(4, 9))
    seeds = 0
    placements = []
    expected_growth = set()

    def in_main_area(r: int, c: int) -> bool:
        return main_top <= r <= main_bottom and 0 <= c < w

    def neighbor_cells_clear(seed_r: int, seed_c: int, marker_r: int, marker_c: int) -> bool:
        for dr, dc in _DIRS:
            nr, nc = seed_r + dr, seed_c + dc
            if not _in_bounds(nr, nc, h, w):
                continue
            if nr == marker_r and nc == marker_c:
                continue
            if grid[nr, nc] != bg:
                return False
        return True

    while seeds < seed_count:
        color = int(rng.choice(shape_colors))
        tpl = shape_templates[shape_colors.index(color)]
        dir_idx = int(rng.integers(0, 4))
        dr, dc = _DIRS[dir_idx]
        want_overlap = len(expected_growth) > 0 and bool(rng.integers(0, 2))
        ok = False

        if want_overlap and expected_growth:
            for _ in range(200):
                anchor = next(iter(rng.choice(list(expected_growth), size=1)))  # type: ignore[arg-type]
                locs = np.argwhere(tpl == 1)
                if len(locs) == 0:
                    break
                li = locs[int(rng.integers(0, len(locs)))]
                tl_r, tl_c = int(anchor[0] - li[0]), int(anchor[1] - li[1])
                r = tl_r - 2 * dr + 1
                c = tl_c - 2 * dc + 1
                marker_r, marker_c = r + dr, c + dc

                if not in_main_area(r, c) or not in_main_area(marker_r, marker_c):
                    continue
                if not in_main_area(c=marker_c, r=marker_r):
                    continue
                if not in_main_area(r, c):
                    continue
                if grid[r, c] != bg or grid[marker_r, marker_c] != bg:
                    continue
                if not neighbor_cells_clear(r, c, marker_r, marker_c):
                    continue

                growth = _growth_cells(r, c, dr, dc, tpl)
                if not all(0 <= gr < h and 0 <= gc < w for gr, gc in growth):
                    continue
                if not any((main_top <= gr <= main_bottom) for gr, gc in growth):
                    continue
                if not all(main_top <= gr <= main_bottom for gr, gc in growth):
                    continue
                if not all(0 <= gc < w for gr, gc in growth):
                    continue

                # require true overlap with previously expected growth region
                if not any(cell in expected_growth for cell in growth):
                    continue

                grid[r, c] = color
                grid[marker_r, marker_c] = marker
                expected_growth.update(growth)
                placements.append((r, c, color, dr, dc))
                seeds += 1
                ok = True
                break

        if not ok:
            for _ in range(250):
                dir_idx = int(rng.integers(0, 4))
                dr, dc = _DIRS[dir_idx]
                r_min, r_max = main_top, main_bottom
                c_min, c_max = 0, w - 1

                if dr == -1:
                    r_min += 3
                elif dr == 1:
                    r_max -= 3
                if dc == -1:
                    c_min += 3
                elif dc == 1:
                    c_max -= 3

                if r_min > r_max or c_min > c_max:
                    continue

                r = int(rng.integers(r_min, r_max + 1))
                c = int(rng.integers(c_min, c_max + 1))
                marker_r, marker_c = r + dr, c + dc

                if not in_main_area(marker_r, marker_c):
                    continue
                if not in_main_area(r, c):
                    continue
                if not in_main_area(marker_r, marker_c):
                    continue
                if grid[r, c] != bg or grid[marker_r, marker_c] != bg:
                    continue
                if not neighbor_cells_clear(r, c, marker_r, marker_c):
                    continue

                growth = _growth_cells(r, c, dr, dc, tpl)
                if not all(0 <= gr < h and 0 <= gc < w for gr, gc in growth):
                    continue
                if not all(main_top <= gr <= main_bottom for gr, gc in growth):
                    continue

                grid[r, c] = color
                grid[marker_r, marker_c] = marker
                expected_growth.update(growth)
                placements.append((r, c, color, dr, dc))
                seeds += 1
                ok = True
                break

        if not ok:
            break

    return np.array(grid, dtype=np.int64)
