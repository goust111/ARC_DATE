from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
PLACE_RETRIES = 180
OBSTACLE_RETRIES = 40


def _in_bounds(height: int, width: int, r: int, c: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in zip(*np.nonzero(mask)):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        queue = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if _in_bounds(height, width, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        comps.append(np.array(pts, dtype=int))

    return comps


def _parse_arrow(grid: np.ndarray, coords: np.ndarray) -> dict[str, object] | None:
    values = grid[coords[:, 0], coords[:, 1]]
    colors, counts = np.unique(values, return_counts=True)
    if len(colors) != 2:
        return None

    tip_color_indices = np.flatnonzero(counts == 1)
    if len(tip_color_indices) != 1:
        return None

    tip_color = int(colors[int(tip_color_indices[0])])
    body_color = next(int(color) for color in colors if int(color) != tip_color)

    body_coords = coords[values == body_color]
    if body_coords.shape[0] < 2:
        return None

    tip_coord = tuple(map(int, coords[values == tip_color][0]))
    rows = body_coords[:, 0]
    cols = body_coords[:, 1]

    if np.all(rows == rows[0]):
        row = int(rows[0])
        col_min = int(cols.min())
        col_max = int(cols.max())
        if len(np.unique(cols)) != body_coords.shape[0]:
            return None
        if body_coords.shape[0] != col_max - col_min + 1:
            return None
        if tip_coord == (row, col_min - 1):
            direction = (0, -1)
        elif tip_coord == (row, col_max + 1):
            direction = (0, 1)
        else:
            return None
    elif np.all(cols == cols[0]):
        col = int(cols[0])
        row_min = int(rows.min())
        row_max = int(rows.max())
        if len(np.unique(rows)) != body_coords.shape[0]:
            return None
        if body_coords.shape[0] != row_max - row_min + 1:
            return None
        if tip_coord == (row_min - 1, col):
            direction = (-1, 0)
        elif tip_coord == (row_max + 1, col):
            direction = (1, 0)
        else:
            return None
    else:
        return None

    return {
        "coords": [tuple(map(int, pt)) for pt in coords],
        "tip": tip_coord,
        "tip_color": tip_color,
        "body_color": body_color,
        "direction": direction,
    }


def _ray_cells(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    dr, dc = map(int, direction)
    r = int(tip[0] + dr)
    c = int(tip[1] + dc)
    cells: list[tuple[int, int]] = []
    while _in_bounds(height, width, r, c) and int(grid[r, c]) == BACKGROUND:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows: list[dict[str, object]] = []
    arrow_mask = np.zeros_like(g, dtype=bool)
    for coords in _components(g != BACKGROUND):
        arrow = _parse_arrow(g, coords)
        if arrow is None:
            continue
        arrows.append(arrow)
        arrow_mask[coords[:, 0], coords[:, 1]] = True

    out = g.copy()
    out[arrow_mask] = BACKGROUND

    ray_colors: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        tip = arrow["tip"]
        direction = arrow["direction"]
        tip_color = int(arrow["tip_color"])
        for cell in _ray_cells(g, tip, direction):
            previous = ray_colors.get(cell)
            if previous is not None:
                assert previous == tip_color, "conflicting ray colors on the same cell"
            ray_colors[cell] = tip_color

    for (r, c), color in ray_colors.items():
        out[r, c] = color
    return out


def _arrow_body_cells(
    tip: tuple[int, int],
    direction: tuple[int, int],
    body_len: int,
) -> list[tuple[int, int]]:
    tr, tc = map(int, tip)
    dr, dc = map(int, direction)
    return [(tr - dr * k, tc - dc * k) for k in range(1, body_len + 1)]


def _forward_cells(
    tip: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    tr, tc = map(int, tip)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    r = tr + dr
    c = tc + dc
    while _in_bounds(height, width, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _can_place_component(
    cells: list[tuple[int, int]],
    occupied: np.ndarray,
    reserved_rays: np.ndarray,
) -> bool:
    height, width = map(int, occupied.shape)
    cell_set = set(cells)
    if not cell_set:
        return False

    for r, c in cell_set:
        if not _in_bounds(height, width, r, c):
            return False
        if occupied[r, c] or reserved_rays[r, c]:
            return False

    for r, c in cell_set:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(height, width, nr, nc) and occupied[nr, nc] and (nr, nc) not in cell_set:
                return False

    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _mark(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _sample_blocking_rectangle(
    block_cell: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    br, bc = map(int, block_cell)
    dr, dc = map(int, direction)

    for _ in range(OBSTACLE_RETRIES):
        if dr == 0:
            rect_h = random.randint(1, min(3, height))
            top_lo = max(0, br - rect_h + 1)
            top_hi = min(br, height - rect_h)
            if top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)

            if dc == 1:
                rect_w = random.randint(1, min(4, width - bc))
                left = bc
            else:
                rect_w = random.randint(1, min(4, bc + 1))
                left = bc - rect_w + 1

            return [(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)]

        rect_w = random.randint(1, min(3, width))
        left_lo = max(0, bc - rect_w + 1)
        left_hi = min(bc, width - rect_w)
        if left_lo > left_hi:
            continue
        left = random.randint(left_lo, left_hi)

        if dr == 1:
            rect_h = random.randint(1, min(4, height - br))
            top = br
        else:
            rect_h = random.randint(1, min(4, br + 1))
            top = br - rect_h + 1

        return [(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)]

    return None


def _place_arrow(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved_rays: np.ndarray,
    body_color: int,
    tip_color: int,
    branch: str,
    obstacle_colors: list[int],
) -> bool:
    height, width = map(int, grid.shape)

    for _ in range(PLACE_RETRIES):
        direction = random.choice(DIRS4)
        body_len = random.randint(2, 5)
        tip = (random.randint(0, height - 1), random.randint(0, width - 1))
        body = _arrow_body_cells(tip, direction, body_len)
        arrow_cells = [tip] + body
        if not _can_place_component(arrow_cells, occupied, reserved_rays):
            continue

        forward = _forward_cells(tip, direction, height, width)
        if branch == "boundary":
            if len(forward) < 1:
                continue
            if any(occupied[r, c] or reserved_rays[r, c] for r, c in forward):
                continue

            _paint_cells(grid, body, body_color)
            _paint_cells(grid, [tip], tip_color)
            _mark(occupied, arrow_cells)
            _mark(reserved_rays, forward)
            return True

        if len(forward) < 2 or not obstacle_colors:
            continue

        max_black = len(forward) - 1
        black_len = random.randint(1, max_black)
        ray_cells = forward[:black_len]
        if any(occupied[r, c] or reserved_rays[r, c] for r, c in ray_cells):
            continue

        block_cell = forward[black_len]
        obstacle_cells = _sample_blocking_rectangle(block_cell, direction, height, width)
        if obstacle_cells is None:
            continue
        if not _can_place_component(obstacle_cells, occupied, reserved_rays):
            continue

        obstacle_color = random.choice(obstacle_colors)
        _paint_cells(grid, body, body_color)
        _paint_cells(grid, [tip], tip_color)
        _paint_cells(grid, obstacle_cells, obstacle_color)
        _mark(occupied, arrow_cells)
        _mark(occupied, obstacle_cells)
        _mark(reserved_rays, ray_cells)
        return True

    return False


def _place_extra_obstacle(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved_rays: np.ndarray,
    obstacle_colors: list[int],
) -> bool:
    if not obstacle_colors:
        return False

    height, width = map(int, grid.shape)
    for _ in range(PLACE_RETRIES):
        rect_h = random.randint(1, min(4, height))
        rect_w = random.randint(1, min(4, width))
        top = random.randint(0, height - rect_h)
        left = random.randint(0, width - rect_w)
        cells = [(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)]
        if not _can_place_component(cells, occupied, reserved_rays):
            continue

        _paint_cells(grid, cells, random.choice(obstacle_colors))
        _mark(occupied, cells)
        return True

    return False


def _arrow_count_for_size(height: int, width: int) -> int:
    area = height * width
    if area < 120:
        return random.choice([1, 2])
    if area < 180:
        return random.choice([1, 2, 3])
    return random.choice([2, 3, 4])


def _blocked_arrow_count(arrow_count: int) -> int:
    scenario = random.choices(
        ["boundary_only", "mixed", "mostly_blocked"],
        weights=[2, 4, 2],
        k=1,
    )[0]
    if scenario == "boundary_only":
        return 0
    if scenario == "mixed":
        return random.randint(1, arrow_count if arrow_count == 1 else arrow_count - 1)
    return random.randint(max(1, arrow_count // 2), arrow_count)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 18)
    width = random.randint(10, 18)
    arrow_count = _arrow_count_for_size(height, width)

    if 2 * arrow_count > 8:
        return None

    arrow_colors = random.sample(list(range(1, 10)), 2 * arrow_count)
    obstacle_colors = [color for color in range(1, 10) if color not in arrow_colors]
    if not obstacle_colors:
        return None

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    reserved_rays = np.zeros((height, width), dtype=bool)

    blocked_count = _blocked_arrow_count(arrow_count)
    branches = ["blocked"] * blocked_count + ["boundary"] * (arrow_count - blocked_count)
    random.shuffle(branches)

    color_pairs = list(zip(arrow_colors[::2], arrow_colors[1::2]))
    random.shuffle(color_pairs)

    for (body_color, tip_color), branch in zip(color_pairs, branches):
        if not _place_arrow(
            grid,
            occupied,
            reserved_rays,
            body_color,
            tip_color,
            branch,
            obstacle_colors,
        ):
            return None

    extra_obstacles = random.randint(0, 2)
    if blocked_count == 0 and random.random() < 0.6:
        extra_obstacles = max(extra_obstacles, 1)
    for _ in range(extra_obstacles):
        _place_extra_obstacle(grid, occupied, reserved_rays, obstacle_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    parsed_arrow_count = 0
    for coords in _components(grid != BACKGROUND):
        if _parse_arrow(grid, coords) is not None:
            parsed_arrow_count += 1
    if parsed_arrow_count != arrow_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
