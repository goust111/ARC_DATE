from __future__ import annotations

import random

import numpy as np


GREY = 5
ARC_COLORS = tuple(range(10))
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (14, 22)
TOWER_COUNT_RANGE = (2, 4)
TOWER_HEIGHT_RANGE = (2, 6)
BASE_TOWER_HEIGHT_RANGE = (4, 6)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _tower_cells(tower: dict[str, int]) -> list[tuple[int, int]]:
    return [
        (int(tower["top"] + k), int(tower["col"]))
        for k in range(int(tower["height"]))
    ]


def _find_towers(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    towers: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (int(bg), GREY):
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "tower color must exist"

        cols = np.unique(coords[:, 1])
        assert cols.size == 1, "each tower must occupy one column"
        rows = sorted(int(r) for r in coords[:, 0].tolist())
        assert len(rows) >= 2, "tower height must be at least two"
        assert rows == list(range(rows[0], rows[0] + len(rows))), "tower pixels must be contiguous"

        top = int(rows[0])
        height = int(len(rows))
        towers.append(
            {
                "color": color,
                "top": top,
                "bottom": int(top + height - 1),
                "height": height,
                "col": int(cols[0]),
            }
        )

    towers.sort(key=lambda tower: (int(tower["col"]), int(tower["top"])))
    assert towers, "expected at least one signal tower"
    return towers


def _cast_ray(
    out: np.ndarray,
    bg: int,
    obstacle_mask: np.ndarray,
    color: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    length: int,
) -> None:
    height, width = out.shape
    for step in range(1, int(length) + 1):
        rr = int(start_r + dr * step)
        cc = int(start_c + dc * step)
        if not (0 <= rr < height and 0 <= cc < width):
            break
        if bool(obstacle_mask[rr, cc]):
            out[rr, cc] = int(color)
            break
        if int(out[rr, cc]) != int(bg):
            break
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find monochrome vertical towers and emit height-limited rays from them.

    Obstacles are original grey pixels: a ray stops before an obstacle but recolors
    the obstacle cell itself. Later towers may recolor the same original obstacle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    towers = _find_towers(g, bg)
    obstacle_mask = g == GREY
    out = g.copy()

    for tower in towers:
        color = int(tower["color"])
        col = int(tower["col"])
        top = int(tower["top"])
        bottom = int(tower["bottom"])
        height = int(tower["height"])

        _cast_ray(out, bg, obstacle_mask, color, top, col, -1, 0, height)
        _cast_ray(out, bg, obstacle_mask, color, bottom, col, 1, 0, height)
        for row in range(top, bottom + 1):
            _cast_ray(out, bg, obstacle_mask, color, row, col, 0, -1, height)
            _cast_ray(out, bg, obstacle_mask, color, row, col, 0, 1, height)

    return out


def _can_place_object(
    cells: list[tuple[int, int]],
    occupied: np.ndarray,
) -> bool:
    height, width = occupied.shape
    for row, col in cells:
        if not (0 <= row < height and 0 <= col < width):
            return False
        for rr in range(max(0, row - 1), min(height, row + 2)):
            for cc in range(max(0, col - 1), min(width, col + 2)):
                if bool(occupied[rr, cc]):
                    return False
    return True


def _paint_cells(
    grid: np.ndarray,
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> None:
    for row, col in cells:
        grid[row, col] = int(color)
        occupied[row, col] = True


def _overlap_rows(top_a: int, height_a: int, top_b: int, height_b: int) -> list[int]:
    start = max(int(top_a), int(top_b))
    stop = min(int(top_a + height_a), int(top_b + height_b))
    return list(range(start, stop))


def _make_tower(color: int, top: int, height: int, col: int) -> dict[str, int]:
    return {
        "color": int(color),
        "top": int(top),
        "bottom": int(top + height - 1),
        "height": int(height),
        "col": int(col),
    }


def _choose_base_pair(shape: tuple[int, int], colors: list[int]) -> tuple[list[dict[str, int]], tuple[int, int]]:
    grid_h, grid_w = map(int, shape)
    for _ in range(300):
        h1 = random.randint(*BASE_TOWER_HEIGHT_RANGE)
        h2 = random.randint(*BASE_TOWER_HEIGHT_RANGE)
        max_dist = min(int(h1), int(h2), 6, grid_w - 5)
        if max_dist < 4:
            continue
        dist = random.randint(4, max_dist)

        left_col_min = 2
        left_col_max = grid_w - dist - 3
        if left_col_max < left_col_min:
            continue
        col1 = random.randint(left_col_min, left_col_max)
        col2 = int(col1 + dist)

        top1 = random.randint(1, grid_h - h1 - 2)
        top2_choices = [
            cand
            for cand in range(1, grid_h - h2 - 1)
            if len(_overlap_rows(top1, h1, cand, h2)) >= 2
        ]
        if not top2_choices:
            continue
        top2 = random.choice(top2_choices)
        overlap = _overlap_rows(top1, h1, top2, h2)

        left_dists = [
            dist_from_left
            for dist_from_left in range(2, dist - 1)
            if dist_from_left <= h1 and dist - dist_from_left <= h2
        ]
        if not left_dists:
            continue

        tower_a = _make_tower(colors[0], top1, h1, col1)
        tower_b = _make_tower(colors[1], top2, h2, col2)
        obstacle = (random.choice(overlap), int(col1 + random.choice(left_dists)))
        return [tower_a, tower_b], obstacle

    raise RuntimeError("failed to place base tower pair")


def _place_extra_tower(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    used_cols: set[int],
) -> dict[str, int] | None:
    grid_h, grid_w = map(int, grid.shape)
    for _ in range(250):
        height = random.randint(*TOWER_HEIGHT_RANGE)
        col = random.randint(1, grid_w - 2)
        if col in used_cols:
            continue
        top = random.randint(1, grid_h - height - 2)
        tower = _make_tower(color, top, height, col)
        cells = _tower_cells(tower)
        if _can_place_object(cells, occupied):
            return tower
    return None


def _path_is_bg(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    distance: int,
) -> bool:
    for step in range(1, int(distance) + 1):
        rr = int(start_r + dr * step)
        cc = int(start_c + dc * step)
        if int(grid[rr, cc]) != _get_bg(grid):
            return False
    return True


def _try_vertical_obstacle(
    grid: np.ndarray,
    occupied: np.ndarray,
    tower: dict[str, int],
) -> bool:
    grid_h = int(grid.shape[0])
    col = int(tower["col"])
    choices: list[tuple[int, int]] = []
    for start_row, dr in (
        (int(tower["top"]), -1),
        (int(tower["bottom"]), 1),
    ):
        limit = int(start_row) if dr == -1 else int(grid_h - 1 - start_row)
        max_dist = min(int(tower["height"]), limit)
        if max_dist < 2:
            continue
        dists = list(range(2, max_dist + 1))
        random.shuffle(dists)
        for dist in dists:
            target = (int(start_row + dr * dist), col)
            if not _can_place_object([target], occupied):
                continue
            if _path_is_bg(grid, start_row, col, dr, 0, dist):
                choices.append(target)
                break

    if not choices:
        return False

    cell = random.choice(choices)
    _paint_cells(grid, occupied, [cell], GREY)
    return True


def _ray_reaches_tower_count(
    towers: list[dict[str, int]],
    target_row: int,
    target_col: int,
) -> int:
    hits = 0
    for tower in towers:
        col = int(tower["col"])
        height = int(tower["height"])
        if int(tower["top"]) <= int(target_row) <= int(tower["bottom"]):
            dist = abs(int(target_col) - col)
            if dist >= 2 and dist <= height:
                hits += 1
        if col == int(target_col):
            up_dist = int(tower["top"]) - int(target_row)
            down_dist = int(target_row) - int(tower["bottom"])
            if 2 <= up_dist <= height or 2 <= down_dist <= height:
                hits += 1
    return hits


def _add_extra_obstacles(
    grid: np.ndarray,
    occupied: np.ndarray,
    towers: list[dict[str, int]],
) -> None:
    random_towers = towers[:]
    random.shuffle(random_towers)
    placed = 0
    target = random.randint(0, max(1, len(towers)))

    for tower in random_towers:
        if placed >= target:
            break
        if random.random() < 0.7 and _try_vertical_obstacle(grid, occupied, tower):
            placed += 1

    untouched_target = random.randint(0, 1)
    grid_h, grid_w = map(int, grid.shape)
    for _ in range(200):
        if untouched_target <= 0:
            break
        row = random.randint(1, grid_h - 2)
        col = random.randint(1, grid_w - 2)
        if not _can_place_object([(row, col)], occupied):
            continue
        if _ray_reaches_tower_count(towers, row, col) != 0:
            continue
        _paint_cells(grid, occupied, [(row, col)], GREY)
        untouched_target -= 1


def _generate_attempt() -> np.ndarray:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    bg = random.choice([color for color in ARC_COLORS if color != GREY])
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    tower_count = random.randint(*TOWER_COUNT_RANGE)
    tower_colors = random.sample([color for color in ARC_COLORS if color not in (bg, GREY)], tower_count)

    towers, shared_obstacle = _choose_base_pair(grid.shape, tower_colors[:2])
    for tower in towers:
        _paint_cells(grid, occupied, _tower_cells(tower), int(tower["color"]))
    _paint_cells(grid, occupied, [shared_obstacle], GREY)

    used_cols = {int(tower["col"]) for tower in towers}
    for color in tower_colors[2:]:
        tower = _place_extra_tower(grid, occupied, color, used_cols)
        if tower is None:
            raise RuntimeError("failed to place extra tower")
        towers.append(tower)
        used_cols.add(int(tower["col"]))
        _paint_cells(grid, occupied, _tower_cells(tower), int(tower["color"]))

    _add_extra_obstacles(grid, occupied, towers)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_attempt()
            out = main(grid)
        except Exception:
            continue

        obstacle_mask = grid == GREY
        if not bool(np.any(obstacle_mask)):
            continue
        if not bool(np.any(out != grid)):
            continue
        if not bool(np.any(out[obstacle_mask] != GREY)):
            continue

        bg = _get_bg(grid)
        towers = _find_towers(grid, bg)
        if len(towers) < 2:
            continue
        return grid

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
