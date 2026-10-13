from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

OBSTACLE_COLORS = (4, 6, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

CELL_SIZE_MIN = 4
CELL_SIZE_MAX = 6
MAX_GENERATION_ATTEMPTS = 400

OBSTACLE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
)


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    if not separators:
        return [(0, length - 1)]

    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _fill_cross(out_cell: np.ndarray, src_cell: np.ndarray, tower: tuple[int, int]) -> None:
    obstacle_mask = (src_cell != BLACK) & (src_cell != ORANGE)
    tr, tc = tower
    out_cell[tr, tc] = RED
    h, w = src_cell.shape
    for dr, dc in DIR4:
        rr, cc = tr + dr, tc + dc
        while 0 <= rr < h and 0 <= cc < w and not obstacle_mask[rr, cc]:
            out_cell[rr, cc] = RED
            rr += dr
            cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separator-defined subregion according to how many orange towers it contains.

    Input structure:
    - The background is black.
    - Full grey rows and columns partition the grid into rectangular subregions.
    - Inside each subregion there may be orange single-pixel towers and small 4-connected obstacle shapes in other colors.

    Rule:
    - Process each subregion independently by counting its orange towers.
    - With 1 tower, overwrite that tower with red and extend red in the four cardinal directions until an obstacle or the subregion boundary stops each ray.
    - With 2 towers, fill the black cells inside their minimal bounding rectangle with blue and overwrite the towers with blue.
    - With 3 or more towers, fill every black cell in the subregion with green and overwrite the towers with green; obstacles and grey separators always stay unchanged.

    Key insight:
    - The grey full-length lines first define independent table cells, then the tower count inside each cell selects one of several unrelated geometric fill rules.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    row_spans = _spans(h, grey_rows)
    col_spans = _spans(w, grey_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            towers = np.argwhere(cell == ORANGE)
            n_towers = int(len(towers))
            if n_towers == 0:
                continue

            out_cell = out[r0 : r1 + 1, c0 : c1 + 1]
            if n_towers == 1:
                _fill_cross(out_cell, cell, tuple(map(int, towers[0])))
            elif n_towers == 2:
                rs = towers[:, 0]
                cs = towers[:, 1]
                rmin, rmax = int(rs.min()), int(rs.max())
                cmin, cmax = int(cs.min()), int(cs.max())
                patch = cell[rmin : rmax + 1, cmin : cmax + 1]
                mask = (patch == BLACK) | (patch == ORANGE)
                out_cell[rmin : rmax + 1, cmin : cmax + 1][mask] = BLUE
            else:
                mask = (cell == BLACK) | (cell == ORANGE)
                out_cell[mask] = GREEN

    return out


def _region_spans_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    start = 0
    for i, size in enumerate(sizes):
        spans.append((start, start + size - 1))
        start += size
        if i != len(sizes) - 1:
            separators.append(start)
            start += 1
    return spans, separators


def _can_place_shape(
    shape_cells: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cell_set = set(shape_cells)
    for r, c in shape_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
    for r, c in shape_cells:
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in occupied and nbr not in cell_set:
                return False
    return True


def _place_obstacles(cell: np.ndarray, branch: int) -> bool:
    h, w = cell.shape
    if branch >= 3:
        obstacle_count = random.choices([0, 1], weights=[3, 2])[0]
    else:
        obstacle_count = random.choices([0, 1, 2], weights=[2, 3, 2])[0]
    if obstacle_count == 0:
        return True

    color = random.choice(OBSTACLE_COLORS)
    occupied: set[tuple[int, int]] = set()
    for _ in range(obstacle_count):
        placed = False
        for _ in range(60):
            shape = random.choice(OBSTACLE_SHAPES)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            if max_r >= h or max_c >= w:
                continue
            top = random.randint(0, h - max_r - 1)
            left = random.randint(0, w - max_c - 1)
            cells = tuple((top + dr, left + dc) for dr, dc in shape)
            if not _can_place_shape(cells, occupied, h, w):
                continue
            for r, c in cells:
                cell[r, c] = color
                occupied.add((r, c))
            placed = True
            break
        if not placed:
            return False
    return True


def _cross_reach(cell: np.ndarray, r: int, c: int) -> list[int]:
    blocked = cell != BLACK
    h, w = cell.shape
    lengths: list[int] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        steps = 0
        while 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc]:
            steps += 1
            rr += dr
            cc += dc
        lengths.append(steps)
    return lengths


def _non_adjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                return False
    return True


def _choose_towers(cell: np.ndarray, branch: int) -> list[tuple[int, int]] | None:
    free = [tuple(map(int, pos)) for pos in np.argwhere(cell == BLACK)]
    if branch == 0:
        return []

    if branch == 1:
        candidates = []
        for r, c in free:
            reach = _cross_reach(cell, r, c)
            if sum(reach) >= 2 and sum(length > 0 for length in reach) >= 2:
                candidates.append((r, c))
        return [random.choice(candidates)] if candidates else None

    if branch == 2:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i, (r1, c1) in enumerate(free):
            for r2, c2 in free[i + 1 :]:
                if abs(r1 - r2) + abs(c1 - c2) <= 1:
                    continue
                rmin, rmax = sorted((r1, r2))
                cmin, cmax = sorted((c1, c2))
                patch = cell[rmin : rmax + 1, cmin : cmax + 1]
                if int(np.count_nonzero(patch == BLACK)) < 3:
                    continue
                candidates.append(((r1, c1), (r2, c2)))
        if not candidates:
            return None
        pair = random.choice(candidates)
        return [pair[0], pair[1]]

    count_choices = [3]
    if len(free) >= 5:
        count_choices.append(4)
    random.shuffle(count_choices)
    for tower_count in count_choices:
        if len(free) <= tower_count:
            continue
        shuffled = free[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        for pos in shuffled:
            if all(abs(pos[0] - r) + abs(pos[1] - c) > 1 for r, c in chosen):
                chosen.append(pos)
                if len(chosen) == tower_count:
                    if len(free) > tower_count and _non_adjacent(chosen):
                        return chosen
                    break
    return None


def _make_region(h: int, w: int, branch: int) -> np.ndarray | None:
    for _ in range(120):
        cell = np.full((h, w), BLACK, dtype=int)
        if not _place_obstacles(cell, branch):
            continue
        towers = _choose_towers(cell, branch)
        if towers is None:
            continue
        for r, c in towers:
            cell[r, c] = ORANGE
        return cell
    return None


def _branch_sequence(n_regions: int) -> list[int]:
    sequence: list[int] = []
    base = [0, 1, 2, 3]
    while len(sequence) < n_regions:
        batch = base[:]
        random.shuffle(batch)
        sequence.extend(batch)
    return sequence[:n_regions]


def _gen_attempt() -> np.ndarray | None:
    for _ in range(50):
        n_rows = random.randint(1, 4)
        n_cols = random.randint(1, 4)
        if n_rows * n_cols >= 4:
            break
    else:
        return None

    row_sizes = [random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(n_rows)]
    col_sizes = [random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(n_cols)]
    row_spans, sep_rows = _region_spans_from_sizes(row_sizes)
    col_spans, sep_cols = _region_spans_from_sizes(col_sizes)

    h = row_spans[-1][1] + 1
    w = col_spans[-1][1] + 1
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for r in sep_rows:
        grid[r, :] = GREY
    for c in sep_cols:
        grid[:, c] = GREY

    branches = _branch_sequence(n_rows * n_cols)
    random.shuffle(branches)
    branch_iter = iter(branches)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            branch = next(branch_iter)
            region = _make_region(r1 - r0 + 1, c1 - c0 + 1, branch)
            if region is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = region

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out == RED):
        return None
    if not np.any(out == BLUE):
        return None
    if not np.any(out == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
