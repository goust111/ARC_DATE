import random
from collections import deque

import numpy as np


BG = 0
BOTTOM_COLOR = 1
TOP_COLOR = 2
LEFT_COLOR = 3
RIGHT_COLOR = 4
SEP_COLOR = 5
SPRITE_COLOR = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_ORDER = ["up", "down", "left", "right"]


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    vals = vals[vals >= 0]
    if vals.size == 0:
        return BG
    counts = np.bincount(vals)
    return int(np.argmax(counts))


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid != BG)
    if pts.size == 0:
        raise ValueError("grid has no foreground pixels")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _intervals(start: int, end: int, separators: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = start
    for pos in sorted(separators):
        if cur <= pos - 1:
            out.append((cur, pos - 1))
        cur = pos + 1
    if cur <= end:
        out.append((cur, end))
    return out


def _component_neighbors(cells: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cell_set = set(cells)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        cur = []
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cell_set:
                cur.append(nxt)
        neighbors[(r, c)] = cur
    return neighbors


def _graph_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> int:
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        node, dist = q.popleft()
        for nxt in neighbors[node]:
            if nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    raise ValueError("component is disconnected")


def _endpoint_direction(endpoint: tuple[int, int], neighbor: tuple[int, int]) -> str:
    dr = endpoint[0] - neighbor[0]
    dc = endpoint[1] - neighbor[1]
    if dr == -1:
        return "up"
    if dr == 1:
        return "down"
    if dc == -1:
        return "left"
    return "right"


def _gap_to_side(cell: tuple[int, int], direction: str, region_h: int, region_w: int) -> int:
    r, c = cell
    if direction == "up":
        return r
    if direction == "down":
        return region_h - 1 - r
    if direction == "left":
        return c
    return region_w - 1 - c


def _infer_direction(cells: list[tuple[int, int]], region_shape: tuple[int, int]) -> str:
    region_h, region_w = region_shape
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}

    if len(rows) == 1:
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        left_gap = min_c
        right_gap = region_w - 1 - max_c
        if left_gap != right_gap:
            return "left" if left_gap < right_gap else "right"
        center_c = (min_c + max_c) / 2.0
        return "left" if center_c < (region_w - 1) / 2.0 else "right"

    if len(cols) == 1:
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        top_gap = min_r
        bottom_gap = region_h - 1 - max_r
        if top_gap != bottom_gap:
            return "up" if top_gap < bottom_gap else "down"
        center_r = (min_r + max_r) / 2.0
        return "up" if center_r < (region_h - 1) / 2.0 else "down"

    neighbors = _component_neighbors(cells)
    endpoints = [cell for cell in cells if len(neighbors[cell]) == 1]
    turns = []
    for cell, nbrs in neighbors.items():
        if len(nbrs) != 2:
            continue
        a, b = nbrs
        if not (a[0] == b[0] or a[1] == b[1]):
            turns.append(cell)

    if not turns or len(endpoints) < 2:
        raise ValueError("sprite shape is not directional")

    turn = turns[0]
    options = []
    for endpoint in endpoints:
        dist = _graph_distance(turn, endpoint, neighbors)
        direction = _endpoint_direction(endpoint, neighbors[endpoint][0])
        gap = _gap_to_side(endpoint, direction, region_h, region_w)
        options.append((dist, gap, direction))

    best_dist = min(dist for dist, _gap, _dir in options)
    best = [item for item in options if item[0] == best_dist]
    if len(best) == 1:
        return best[0][2]

    best_gap = min(gap for _dist, gap, _dir in best)
    for _dist, gap, direction in best:
        if gap == best_gap:
            return direction
    raise AssertionError("unreachable")


def _paint_band(
    out: np.ndarray,
    rr0: int,
    rr1: int,
    cc0: int,
    cc1: int,
    direction: str,
    depth: int,
    color: int,
) -> None:
    if direction == "up":
        band = out[rr0 : min(rr1 + 1, rr0 + depth), cc0 : cc1 + 1]
    elif direction == "down":
        band = out[max(rr0, rr1 - depth + 1) : rr1 + 1, cc0 : cc1 + 1]
    elif direction == "left":
        band = out[rr0 : rr1 + 1, cc0 : min(cc1 + 1, cc0 + depth)]
    else:
        band = out[rr0 : rr1 + 1, max(cc0, cc1 - depth + 1) : cc1 + 1]
    band[band == BG] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partitioned rectangle from the side chosen by its orange pointer sprite.

    The sprite direction selects one of the outer frame sides; the sprite size is the
    fill depth. The frame, grey separators, and orange sprite remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, r1, c0, c1 = _frame_bbox(g)
    top_color = _mode(g[r0, c0 + 1 : c1])
    bottom_color = _mode(g[r1, c0 + 1 : c1])
    left_color = _mode(g[r0 + 1 : r1, c0])
    right_color = _mode(g[r0 + 1 : r1, c1])
    fill_colors = {
        "up": top_color,
        "down": bottom_color,
        "left": left_color,
        "right": right_color,
    }

    row_seps = [r for r in range(r0 + 1, r1) if np.all(g[r, c0 + 1 : c1] == SEP_COLOR)]
    col_seps = [c for c in range(c0 + 1, c1) if np.all(g[r0 + 1 : r1, c] == SEP_COLOR)]
    row_regions = _intervals(r0 + 1, r1 - 1, row_seps)
    col_regions = _intervals(c0 + 1, c1 - 1, col_seps)

    for rr0, rr1 in row_regions:
        for cc0, cc1 in col_regions:
            region = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            sprite_cells = [tuple(map(int, rc)) for rc in np.argwhere(region == SPRITE_COLOR)]
            if not sprite_cells:
                continue
            direction = _infer_direction(sprite_cells, region.shape)
            depth = len(sprite_cells)
            _paint_band(out, rr0, rr1, cc0, cc1, direction, depth, fill_colors[direction])

    return out


L_SHAPES = {
    "right": [
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ],
    "left": [
        [(0, 1), (1, 1), (2, 1), (0, 0)],
        [(0, 1), (1, 1), (2, 1), (2, 0)],
    ],
    "up": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 2), (1, 0), (1, 1), (1, 2)],
    ],
    "down": [
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
    ],
}

PARTITION_CHOICES = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
REGION_SIZE_RANGE = (8, 11)
OUTER_MARGIN_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _balanced_sequence(values: list[str], n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        block = values[:]
        random.shuffle(block)
        out.extend(block)
    return out[:n]


def _translate(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _shape_size(cells: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _line_shape(direction: str, length: int) -> list[tuple[int, int]]:
    if direction in ("left", "right"):
        return [(0, c) for c in range(length)]
    return [(r, 0) for r in range(length)]


def _candidate_placements(
    base_cells: list[tuple[int, int]],
    region_h: int,
    region_w: int,
    intended_direction: str,
) -> list[list[tuple[int, int]]]:
    shape_h, shape_w = _shape_size(base_cells)
    if region_h - shape_h < 2 or region_w - shape_w < 2:
        return []

    candidates = []
    for dr in range(1, region_h - shape_h):
        for dc in range(1, region_w - shape_w):
            cells = _translate(base_cells, dr, dc)
            if _infer_direction(cells, (region_h, region_w)) == intended_direction:
                candidates.append(cells)
    return candidates


def _pick_sprite(direction: str, shape_kind: str, region_h: int, region_w: int) -> list[tuple[int, int]] | None:
    bases: list[list[tuple[int, int]]] = []
    if shape_kind == "line":
        lengths = [2, 3, 4]
        random.shuffle(lengths)
        for length in lengths:
            bases.append(_line_shape(direction, length))
    else:
        variants = L_SHAPES[direction][:]
        random.shuffle(variants)
        bases.extend(variants)

    for base in bases:
        candidates = _candidate_placements(base, region_h, region_w, direction)
        if candidates:
            return random.choice(candidates)
    return None


def _draw_frame_and_separators(
    grid: np.ndarray,
    frame_r0: int,
    frame_c0: int,
    row_sizes: list[int],
    col_sizes: list[int],
) -> list[tuple[int, int, int, int]]:
    interior_h = sum(row_sizes) + len(row_sizes) - 1
    interior_w = sum(col_sizes) + len(col_sizes) - 1
    frame_r1 = frame_r0 + interior_h + 1
    frame_c1 = frame_c0 + interior_w + 1

    grid[frame_r0, frame_c0 + 1 : frame_c1] = TOP_COLOR
    grid[frame_r1, frame_c0 + 1 : frame_c1] = BOTTOM_COLOR
    grid[frame_r0 + 1 : frame_r1, frame_c0] = LEFT_COLOR
    grid[frame_r0 + 1 : frame_r1, frame_c1] = RIGHT_COLOR

    row_regions = []
    cur_r = frame_r0 + 1
    for idx, size in enumerate(row_sizes):
        row_regions.append((cur_r, cur_r + size - 1))
        cur_r += size
        if idx != len(row_sizes) - 1:
            grid[cur_r, frame_c0 + 1 : frame_c1] = SEP_COLOR
            cur_r += 1

    col_regions = []
    cur_c = frame_c0 + 1
    for idx, size in enumerate(col_sizes):
        col_regions.append((cur_c, cur_c + size - 1))
        cur_c += size
        if idx != len(col_sizes) - 1:
            grid[frame_r0 + 1 : frame_r1, cur_c] = SEP_COLOR
            cur_c += 1

    regions = []
    for rr0, rr1 in row_regions:
        for cc0, cc1 in col_regions:
            regions.append((rr0, rr1, cc0, cc1))
    return regions


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(PARTITION_CHOICES)
    row_sizes = [random.randint(*REGION_SIZE_RANGE) for _ in range(n_rows)]
    col_sizes = [random.randint(*REGION_SIZE_RANGE) for _ in range(n_cols)]

    interior_h = sum(row_sizes) + n_rows - 1
    interior_w = sum(col_sizes) + n_cols - 1
    margin_top = random.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    margin_left = random.randint(*OUTER_MARGIN_RANGE)
    margin_right = random.randint(*OUTER_MARGIN_RANGE)

    grid_h = interior_h + 2 + margin_top + margin_bottom
    grid_w = interior_w + 2 + margin_left + margin_right
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_r0 = margin_top
    frame_c0 = margin_left
    regions = _draw_frame_and_separators(grid, frame_r0, frame_c0, row_sizes, col_sizes)

    total_regions = len(regions)
    directions = _balanced_sequence(DIRECTION_ORDER[:], total_regions)
    shape_kinds = _balanced_sequence(["line", "el"], total_regions)
    random.shuffle(regions)

    for idx, (rr0, rr1, cc0, cc1) in enumerate(regions):
        direction = directions[idx]
        shape_kind = shape_kinds[idx]
        region_h = rr1 - rr0 + 1
        region_w = cc1 - cc0 + 1
        sprite = _pick_sprite(direction, shape_kind, region_h, region_w)
        if sprite is None:
            return None
        for r, c in sprite:
            grid[rr0 + r, cc0 + c] = SPRITE_COLOR

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
