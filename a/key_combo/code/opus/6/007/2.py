from __future__ import annotations

from pathlib import Path
import itertools
import random
import sys

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if _TOOLBOX.exists():
    toolbox_str = str(_TOOLBOX)
    if toolbox_str not in sys.path:
        sys.path.insert(0, toolbox_str)

from arc_utils import components_mask, neighbors4


BG = 0
RED = 2
GRAY = 5
ORANGE = 7
WALL = 9
CENTER_COLORS = (RED, GRAY)

GEN_ATTEMPTS = 400
GRID_BOUNDS = (25, 29)
COUNT_VALUES = (1, 2, 3, 4, 5)

DIAMOND_OFFSETS = {
    radius: [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) <= radius]
    for radius in range(6)
}


def _trace_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, cc) for cc in range(c0, c1 + step, step)]
        else:
            assert c0 == c1
            step = 1 if r1 >= r0 else -1
            segment = [(rr, c0) for rr in range(r0, r1 + step, step)]
        if cells:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _build_full_vertical(top: int, bottom: int, left: int, right: int, *, cross_row: int | None = None, cross_col: int | None = None) -> list[tuple[int, int]]:
    row_lo = top + 1
    row_hi = bottom - 1
    col_lo = left + 4
    col_hi = right - 4
    if cross_row is not None and cross_col is not None:
        c_cross = int(cross_col)
        c_top = max(col_lo, min(col_hi, c_cross + random.randint(-2, 2)))
        c_bot = max(col_lo, min(col_hi, c_cross + random.randint(-2, 2)))
        r_a = random.randint(top + 3, cross_row - 2)
        r_b = random.randint(cross_row + 2, bottom - 3)
        pts = [
            (row_lo, c_top),
            (r_a, c_top),
            (r_a, c_cross),
            (cross_row, c_cross),
            (r_b, c_cross),
            (r_b, c_bot),
            (row_hi, c_bot),
        ]
        return _trace_polyline(pts)

    c0 = random.randint(col_lo, col_hi)
    c1 = max(col_lo, min(col_hi, c0 + random.randint(-3, 3)))
    c2 = max(col_lo, min(col_hi, c1 + random.randint(-3, 3)))
    r_a = random.randint(top + 3, top + (bottom - top) // 2 - 1)
    r_b = random.randint(r_a + 2, bottom - 3)
    pts = [
        (row_lo, c0),
        (r_a, c0),
        (r_a, c1),
        (r_b, c1),
        (r_b, c2),
        (row_hi, c2),
    ]
    return _trace_polyline(pts)


def _build_full_horizontal(top: int, bottom: int, left: int, right: int, *, cross_row: int | None = None, cross_col: int | None = None) -> list[tuple[int, int]]:
    col_lo = left + 1
    col_hi = right - 1
    row_lo = top + 4
    row_hi = bottom - 4
    if cross_row is not None and cross_col is not None:
        r_cross = int(cross_row)
        r_left = max(row_lo, min(row_hi, r_cross + random.randint(-2, 2)))
        r_right = max(row_lo, min(row_hi, r_cross + random.randint(-2, 2)))
        c_a = random.randint(left + 3, cross_col - 2)
        c_b = random.randint(cross_col + 2, right - 3)
        pts = [
            (r_left, col_lo),
            (r_left, c_a),
            (r_cross, c_a),
            (r_cross, cross_col),
            (r_cross, c_b),
            (r_right, c_b),
            (r_right, col_hi),
        ]
        return _trace_polyline(pts)

    r0 = random.randint(row_lo, row_hi)
    r1 = max(row_lo, min(row_hi, r0 + random.randint(-3, 3)))
    r2 = max(row_lo, min(row_hi, r1 + random.randint(-3, 3)))
    c_a = random.randint(left + 3, left + (right - left) // 2 - 1)
    c_b = random.randint(c_a + 2, right - 3)
    pts = [
        (r0, col_lo),
        (r0, c_a),
        (r1, c_a),
        (r1, c_b),
        (r2, c_b),
        (r2, col_hi),
    ]
    return _trace_polyline(pts)


def _build_partial_horizontal(branch_row: int, wall_col: int, left: int, right: int, side: str) -> list[tuple[int, int]]:
    if side == "left":
        return _trace_polyline([(branch_row, left + 1), (branch_row, wall_col)])
    return _trace_polyline([(branch_row, wall_col), (branch_row, right - 1)])


def _build_partial_vertical(branch_col: int, wall_row: int, top: int, bottom: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return _trace_polyline([(top + 1, branch_col), (wall_row, branch_col)])
    return _trace_polyline([(wall_row, branch_col), (bottom - 1, branch_col)])


def _paint_wall_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = WALL


def _interior_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    regions: list[np.ndarray] = []
    for coords in components_mask(grid != WALL, connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        regions.append(coords.astype(int))
    regions.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return regions


def _region_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _center_candidates(region_mask: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    coords = np.argwhere(region_mask)
    candidates: dict[int, list[tuple[int, int]]] = {}
    h, w = region_mask.shape
    for radius in COUNT_VALUES:
        good: list[tuple[int, int]] = []
        for r, c in coords:
            ok = True
            for dr, dc in DIAMOND_OFFSETS[radius]:
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < h and 0 <= cc < w and region_mask[rr, cc]):
                    ok = False
                    break
            if ok:
                good.append((int(r), int(c)))
        if good:
            candidates[radius] = good
    return candidates


def _assign_distinct_counts(candidate_maps: list[dict[int, list[tuple[int, int]]]]) -> tuple[int, ...] | None:
    perms = list(itertools.permutations(COUNT_VALUES, len(candidate_maps)))
    random.shuffle(perms)
    for perm in perms:
        if all(perm[idx] in candidate_maps[idx] for idx in range(len(candidate_maps))):
            return tuple(int(v) for v in perm)
    return None


def _near_wall_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    near: list[tuple[int, int]] = []
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == WALL:
                near.append((r, c))
                break
    return near


def _pick_markers(grid: np.ndarray, region_mask: np.ndarray, center: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(region_mask)]
    coords = [rc for rc in coords if rc != center]
    far = [rc for rc in coords if abs(rc[0] - center[0]) + abs(rc[1] - center[1]) > count + 1]
    pool = far if len(far) >= count else coords
    if len(pool) < count:
        return None

    chosen: list[tuple[int, int]] = []
    near_wall = _near_wall_cells(grid, pool)
    if near_wall and random.random() < 0.7:
        picked = random.choice(near_wall)
        chosen.append(picked)
        pool = [rc for rc in pool if rc != picked]

    random.shuffle(pool)
    for rc in pool:
        if len(chosen) == count:
            break
        if all(abs(rc[0] - rr) + abs(rc[1] - cc) >= 2 for rr, cc in chosen):
            chosen.append(rc)
    if len(chosen) < count:
        remaining = [rc for rc in pool if rc not in chosen]
        need = count - len(chosen)
        if len(remaining) < need:
            return None
        chosen.extend(random.sample(remaining, need))
    return chosen


def _reachable_marker_count(grid: np.ndarray, center: tuple[int, int]) -> int:
    h, w = grid.shape
    seen = {center}
    stack = [center]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            value = int(grid[nr, nc])
            if value in (BG, ORANGE):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return sum(int(grid[r, c]) == ORANGE for r, c in seen)


def _render_from_counts(grid: np.ndarray, counts: dict[tuple[int, int], int], *, alternate: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == ORANGE] = BG
    h, w = out.shape
    centers = {tuple(map(int, rc)) for rc in np.argwhere(np.isin(g, CENTER_COLORS))}
    for center in sorted(counts):
        cr, cc = center
        center_color = int(g[cr, cc])
        odd_color = center_color
        even_color = GRAY if center_color == RED else RED
        if not alternate:
            even_color = odd_color
        for radius in range(1, int(counts[center]) + 1):
            paint = odd_color if radius % 2 == 1 else even_color
            for dr in range(-radius, radius + 1):
                dc = radius - abs(dr)
                for sign in (-1, 1):
                    rr = cr + dr
                    cc2 = cc + sign * dc
                    if not (0 <= rr < h and 0 <= cc2 < w):
                        continue
                    if (rr, cc2) in centers:
                        continue
                    if int(out[rr, cc2]) == WALL:
                        continue
                    out[rr, cc2] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the wall-reachable orange markers for each center and turn that count into concentric diamond outlines.

    Input structure:
    - The background is black, and maroon wall segments partition the interior into 2 to 4 enclosed floor regions.
    - Each enclosed region contains exactly one single-pixel center, colored red or gray.
    - Orange singleton marker pixels are scattered through the regions; their meaning depends only on which center can reach them without crossing walls.

    Rule:
    - For each center, run a 4-connected flood from that center, but only move through black or orange cells; walls and other centers block the flood.
    - Count the reachable orange markers in that flood region, erase all orange markers, and use the count k as the number of concentric Manhattan-diamond outline layers to draw around the center.
    - Draw only the outline cells at distances 1, 2, ..., k. A red center paints odd radii red and even radii gray; a gray center swaps that parity. The center pixel itself stays unchanged.
    - Keep the walls unchanged, and skip any diamond cell that would fall outside the grid or on a wall.

    Key insight:
    - The orange dots are not counted globally or by proximity: the walls define which dots belong to each center, and that reachable count alone determines how many alternating-color diamond outlines appear.
    """
    g = np.asarray(grid, dtype=int)
    counts: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(np.isin(g, CENTER_COLORS)):
        center = (int(r), int(c))
        counts[center] = _reachable_marker_count(g, center)
    return _render_from_counts(g, counts, alternate=True)


def _make_partitioned_grid() -> np.ndarray | None:
    h = random.randint(*GRID_BOUNDS)
    w = random.randint(*GRID_BOUNDS)
    grid = np.full((h, w), BG, dtype=int)

    top = left = 1
    bottom = h - 2
    right = w - 2
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL

    target_regions = random.choice((2, 3, 4))
    if target_regions == 2:
        if random.random() < 0.5:
            _paint_wall_path(grid, _build_full_vertical(top, bottom, left, right))
        else:
            _paint_wall_path(grid, _build_full_horizontal(top, bottom, left, right))
    elif target_regions == 3:
        if random.random() < 0.5:
            vpath = _build_full_vertical(top, bottom, left, right)
            _paint_wall_path(grid, vpath)
            row_to_cols: dict[int, list[int]] = {}
            for r, c in vpath:
                row_to_cols.setdefault(int(r), []).append(int(c))
            branch_rows = [r for r, cols in row_to_cols.items() if len(set(cols)) == 1 and top + 5 <= r <= bottom - 5]
            if not branch_rows:
                return None
            branch_row = random.choice(branch_rows)
            wall_col = row_to_cols[branch_row][0]
            if wall_col - left < 7 and right - wall_col < 7:
                return None
            side = "left" if wall_col - left >= 7 and (right - wall_col < 7 or random.random() < 0.5) else "right"
            _paint_wall_path(grid, _build_partial_horizontal(branch_row, wall_col, left, right, side))
        else:
            hpath = _build_full_horizontal(top, bottom, left, right)
            _paint_wall_path(grid, hpath)
            col_to_rows: dict[int, list[int]] = {}
            for r, c in hpath:
                col_to_rows.setdefault(int(c), []).append(int(r))
            branch_cols = [c for c, rows in col_to_rows.items() if len(set(rows)) == 1 and left + 5 <= c <= right - 5]
            if not branch_cols:
                return None
            branch_col = random.choice(branch_cols)
            wall_row = col_to_rows[branch_col][0]
            if wall_row - top < 7 and bottom - wall_row < 7:
                return None
            side = "top" if wall_row - top >= 7 and (bottom - wall_row < 7 or random.random() < 0.5) else "bottom"
            _paint_wall_path(grid, _build_partial_vertical(branch_col, wall_row, top, bottom, side))
    else:
        cross_row = random.randint(top + 7, bottom - 7)
        cross_col = random.randint(left + 7, right - 7)
        _paint_wall_path(grid, _build_full_vertical(top, bottom, left, right, cross_row=cross_row, cross_col=cross_col))
        _paint_wall_path(grid, _build_full_horizontal(top, bottom, left, right, cross_row=cross_row, cross_col=cross_col))

    regions = _interior_regions(grid)
    if len(regions) != target_regions:
        return None
    return grid


def _place_centers_and_markers(base_grid: np.ndarray) -> np.ndarray | None:
    grid = base_grid.copy()
    regions = _interior_regions(grid)
    candidate_maps = [_center_candidates(_region_mask(grid.shape, coords)) for coords in regions]
    if any(not cmap for cmap in candidate_maps):
        return None

    counts = _assign_distinct_counts(candidate_maps)
    if counts is None:
        return None

    center_colors = [RED, GRAY]
    while len(center_colors) < len(regions):
        center_colors.append(random.choice(CENTER_COLORS))
    random.shuffle(center_colors)

    centers: list[tuple[int, int]] = []
    markers_by_region: list[list[tuple[int, int]]] = []
    for idx, region in enumerate(regions):
        region_mask = _region_mask(grid.shape, region)
        count = int(counts[idx])
        center = random.choice(candidate_maps[idx][count])
        markers = _pick_markers(grid, region_mask, center, count)
        if markers is None:
            return None
        centers.append(center)
        markers_by_region.append(markers)

    for center, color in zip(centers, center_colors):
        grid[center] = color
    for marker_cells in markers_by_region:
        for r, c in marker_cells:
            grid[r, c] = ORANGE
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    regions = _interior_regions(grid)
    if not (2 <= len(regions) <= 4):
        return False

    centers = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(grid, CENTER_COLORS))]
    if len(centers) != len(regions):
        return False
    if {int(grid[r, c]) for r, c in centers} != {RED, GRAY} and len(regions) > 1:
        return False

    counts: dict[tuple[int, int], int] = {}
    region_marker_counts: list[int] = []
    for region in regions:
        region_set = {tuple(map(int, rc)) for rc in region}
        region_centers = [center for center in centers if center in region_set]
        if len(region_centers) != 1:
            return False
        center = region_centers[0]
        marker_count = sum(int(grid[r, c]) == ORANGE for r, c in region_set)
        if marker_count not in COUNT_VALUES:
            return False
        region_marker_counts.append(marker_count)
        if marker_count != _reachable_marker_count(grid, center):
            return False
        for dr, dc in DIAMOND_OFFSETS[marker_count]:
            rr = center[0] + dr
            cc = center[1] + dc
            if (rr, cc) not in region_set:
                return False
        counts[center] = marker_count

    if len(set(region_marker_counts)) != len(region_marker_counts):
        return False

    expected = _render_from_counts(grid, counts, alternate=True)
    if np.array_equal(expected, grid):
        return False
    if np.array_equal(expected, _render_from_counts(grid, {center: int(np.sum(grid == ORANGE)) for center in counts}, alternate=True)):
        return False
    if np.array_equal(expected, _render_from_counts(grid, counts, alternate=False)):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base = _make_partitioned_grid()
    if base is None:
        return None
    grid = _place_centers_and_markers(base)
    if grid is None:
        return None
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 2")
