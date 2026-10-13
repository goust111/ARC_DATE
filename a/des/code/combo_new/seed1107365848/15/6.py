from __future__ import annotations

import random

import numpy as np


_BG = 0
_WALL = 5
_SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_ATTEMPTS = 400


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != _BG) & (grid != _WALL))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _pair_rays(grid: np.ndarray) -> list[dict[str, object]]:
    seeds = _seed_positions(grid)
    row_map: dict[int, list[tuple[int, int, int]]] = {}
    col_map: dict[int, list[tuple[int, int, int]]] = {}
    for r, c, color in seeds:
        row_map.setdefault(r, []).append((r, c, color))
        col_map.setdefault(c, []).append((r, c, color))

    rays: list[dict[str, object]] = []
    for row, pts in row_map.items():
        if len(pts) != 2:
            continue
        left, right = sorted(pts, key=lambda x: x[1])
        rays.append({"origin": (row, int(left[1])), "head": (row, int(left[1])), "dir": (0, -1), "color": int(left[2]), "axis": "row"})
        rays.append({"origin": (row, int(right[1])), "head": (row, int(right[1])), "dir": (0, 1), "color": int(right[2]), "axis": "row"})

    for col, pts in col_map.items():
        if len(pts) != 2:
            continue
        top, bottom = sorted(pts, key=lambda x: x[0])
        rays.append({"origin": (int(top[0]), col), "head": (int(top[0]), col), "dir": (-1, 0), "color": int(top[2]), "axis": "col"})
        rays.append({"origin": (int(bottom[0]), col), "head": (int(bottom[0]), col), "dir": (1, 0), "color": int(bottom[2]), "axis": "col"})
    return rays


def _simulate_growth(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    active = []
    for stat_idx, ray in enumerate(_pair_rays(g)):
        item = dict(ray)
        item["stat_idx"] = stat_idx
        active.append(item)
    stats = [
        {
            "origin": tuple(ray["origin"]),
            "dir": tuple(ray["dir"]),
            "axis": str(ray["axis"]),
            "color": int(ray["color"]),
            "steps": 0,
            "stop": None,
        }
        for ray in active
    ]

    while active:
        proposals: dict[tuple[int, int], list[int]] = {}

        for local_idx, ray in enumerate(active):
            head_r, head_c = ray["head"]
            dr, dc = ray["dir"]
            next_r = int(head_r + dr)
            next_c = int(head_c + dc)
            stat = stats[int(ray["stat_idx"])] if "stat_idx" in ray else None
            if stat is None:
                raise AssertionError("ray missing stat index")

            if not (0 <= next_r < g.shape[0] and 0 <= next_c < g.shape[1]):
                stat["stop"] = "boundary"
                continue
            if int(out[next_r, next_c]) != _BG:
                stat["stop"] = "wall" if int(out[next_r, next_c]) == _WALL else "other"
                continue
            proposals.setdefault((next_r, next_c), []).append(local_idx)

        if not proposals:
            break

        next_active: list[dict[str, object]] = []
        for cell, local_ids in proposals.items():
            if len(local_ids) > 1:
                for local_idx in local_ids:
                    ray = active[local_idx]
                    stats[int(ray["stat_idx"])]["stop"] = "collision"
                continue
            local_idx = local_ids[0]
            ray = active[local_idx]
            out[cell] = int(ray["color"])
            ray["head"] = cell
            stats[int(ray["stat_idx"])]["steps"] = int(stats[int(ray["stat_idx"])]["steps"]) + 1
            next_active.append(ray)
        active = next_active

    if collect_stats:
        return out, stats
    return out, []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow paired seed pixels into outward row and column strips until they hit a blocker.

    Input structure:
    - The background is black.
    - Non-black, non-grey pixels are singleton colored seeds scattered on the grid.
    - Some rows or columns contain exactly two seeds, while other rows and columns contain at most one seed.
    - Grey singleton pixels are fixed wall cells that block growth.

    Rule:
    - Find every row with exactly two seeds and every column with exactly two seeds.
    - In a paired row, the left seed grows left and the right seed grows right; in a paired column, the top seed grows up and the bottom seed grows down.
    - Grow all such rays one cell at a time, keeping the original seed cells, and stop a ray when its next cell is a wall, another non-background cell, or the grid boundary.
    - A seed that belongs to both a row pair and a column pair emits both rays.

    Key insight:
    - The trigger is not color or proximity but exact row/column pairing, and each valid pair grows away from its partner rather than toward it.
    """
    out, _stats = _simulate_growth(np.asarray(grid, dtype=int), collect_stats=False)
    return out


def _analyze_scene(grid: np.ndarray) -> dict[str, int]:
    seeds = _seed_positions(grid)
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c, _color in seeds:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    out, stats = _simulate_growth(grid, collect_stats=True)
    dual = 0
    unpaired = 0
    for r, c, _color in seeds:
        row_paired = row_counts[r] == 2
        col_paired = col_counts[c] == 2
        dual += int(row_paired and col_paired)
        unpaired += int((not row_paired) and (not col_paired))

    features = {
        "seed_count": len(seeds),
        "row_pairs": sum(1 for cnt in row_counts.values() if cnt == 2),
        "col_pairs": sum(1 for cnt in col_counts.values() if cnt == 2),
        "dual_count": dual,
        "unpaired_count": unpaired,
        "ray_count": len(stats),
        "wall_stops": sum(1 for s in stats if s["stop"] == "wall"),
        "other_stops": sum(1 for s in stats if s["stop"] == "other"),
        "boundary_stops": sum(1 for s in stats if s["stop"] == "boundary"),
        "collision_stops": sum(1 for s in stats if s["stop"] == "collision"),
        "changed": int(np.count_nonzero(out != grid)),
    }
    return features


def _blank_grid(height: int, width: int) -> np.ndarray:
    return np.full((height, width), _BG, dtype=int)


def _available_seed_colors(count: int) -> list[int]:
    return random.sample(_SEED_COLORS, count)


def _put_seed(grid: np.ndarray, row: int, col: int, color: int) -> None:
    if int(grid[row, col]) != _BG:
        raise AssertionError("seed placement collision")
    grid[row, col] = int(color)


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = start
    dr, dc = direction
    cells = []
    row += dr
    col += dc
    while 0 <= row < shape[0] and 0 <= col < shape[1]:
        cells.append((int(row), int(col)))
        row += dr
        col += dc
    return cells


def _place_wall_on_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    min_step: int = 2,
    max_step: int | None = None,
) -> bool:
    cells = _ray_cells(start, direction, grid.shape)
    if max_step is None:
        usable = cells[min_step - 1 :]
    else:
        usable = cells[min_step - 1 : max_step]
    usable = [cell for cell in usable if int(grid[cell]) == _BG]
    if not usable:
        return False
    row, col = random.choice(usable)
    grid[row, col] = _WALL
    return True


def _scatter_extra_walls(grid: np.ndarray, count: int, reserved: set[tuple[int, int]]) -> None:
    bg_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if int(grid[r, c]) == _BG and (r, c) not in reserved]
    random.shuffle(bg_cells)
    for row, col in bg_cells[:count]:
        grid[row, col] = _WALL


def _add_unpaired_seeds(
    grid: np.ndarray,
    color_pool: list[int],
    count: int,
    used_rows: set[int],
    used_cols: set[int],
) -> int:
    if count <= 0:
        return 0
    placed = 0
    choices = [(r, c) for r in range(1, grid.shape[0] - 1) for c in range(1, grid.shape[1] - 1) if int(grid[r, c]) == _BG]
    random.shuffle(choices)
    for row, col in choices:
        if row in used_rows or col in used_cols:
            continue
        _put_seed(grid, row, col, color_pool[placed])
        used_rows.add(row)
        used_cols.add(col)
        placed += 1
        if placed == count:
            return placed
    return placed


def _build_row_only() -> np.ndarray | None:
    height = random.randint(9, 18)
    width = random.randint(12, 24)
    pair_count = random.randint(2, 3)
    extra_count = random.randint(1, 2)
    color_pool = _available_seed_colors(2 * pair_count + extra_count)
    grid = _blank_grid(height, width)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    color_idx = 0
    placed_wall = False
    reserved: set[tuple[int, int]] = set()

    for _ in range(pair_count):
        success = False
        for _attempt in range(80):
            row = random.randint(1, height - 2)
            left = random.randint(2, width - 6)
            right = random.randint(left + 2, width - 3)
            if row in used_rows or left in used_cols or right in used_cols:
                continue
            success = True
            break
        if not success:
            return None

        _put_seed(grid, row, left, color_pool[color_idx])
        _put_seed(grid, row, right, color_pool[color_idx + 1])
        color_idx += 2
        used_rows.add(row)
        used_cols.update({left, right})
        reserved.update({(row, left), (row, right)})

        if random.random() < 0.8 and _place_wall_on_ray(grid, (row, left), (0, -1)):
            placed_wall = True
        if random.random() < 0.8 and _place_wall_on_ray(grid, (row, right), (0, 1)):
            placed_wall = True

    if _add_unpaired_seeds(grid, color_pool[color_idx:], extra_count, used_rows, used_cols) != extra_count:
        return None
    if not placed_wall:
        pair_rays = _pair_rays(grid)
        random.shuffle(pair_rays)
        for ray in pair_rays:
            if _place_wall_on_ray(grid, tuple(ray["origin"]), tuple(ray["dir"])):
                break
    _scatter_extra_walls(grid, random.randint(0, 2), reserved)
    return grid


def _build_col_only() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(9, 18)
    pair_count = random.randint(2, 3)
    extra_count = random.randint(1, 2)
    color_pool = _available_seed_colors(2 * pair_count + extra_count)
    grid = _blank_grid(height, width)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    color_idx = 0
    placed_wall = False
    reserved: set[tuple[int, int]] = set()

    for _ in range(pair_count):
        success = False
        for _attempt in range(80):
            col = random.randint(1, width - 2)
            top = random.randint(2, height - 6)
            bottom = random.randint(top + 2, height - 3)
            if col in used_cols or top in used_rows or bottom in used_rows:
                continue
            success = True
            break
        if not success:
            return None

        _put_seed(grid, top, col, color_pool[color_idx])
        _put_seed(grid, bottom, col, color_pool[color_idx + 1])
        color_idx += 2
        used_cols.add(col)
        used_rows.update({top, bottom})
        reserved.update({(top, col), (bottom, col)})

        if random.random() < 0.8 and _place_wall_on_ray(grid, (top, col), (-1, 0)):
            placed_wall = True
        if random.random() < 0.8 and _place_wall_on_ray(grid, (bottom, col), (1, 0)):
            placed_wall = True

    if _add_unpaired_seeds(grid, color_pool[color_idx:], extra_count, used_rows, used_cols) != extra_count:
        return None
    if not placed_wall:
        pair_rays = _pair_rays(grid)
        random.shuffle(pair_rays)
        for ray in pair_rays:
            if _place_wall_on_ray(grid, tuple(ray["origin"]), tuple(ray["dir"])):
                break
    _scatter_extra_walls(grid, random.randint(0, 2), reserved)
    return grid


def _build_rectangle() -> np.ndarray | None:
    height = random.randint(10, 20)
    width = random.randint(10, 20)
    extra_count = random.randint(0, 2)
    color_pool = _available_seed_colors(4 + extra_count)
    grid = _blank_grid(height, width)

    for _attempt in range(80):
        top = random.randint(2, height - 5)
        bottom = random.randint(top + 2, height - 3)
        left = random.randint(2, width - 5)
        right = random.randint(left + 2, width - 3)
        if bottom < height - 1 and right < width - 1:
            break
    else:
        return None

    corners = [(top, left), (top, right), (bottom, left), (bottom, right)]
    for color, (row, col) in zip(color_pool[:4], corners, strict=False):
        _put_seed(grid, row, col, color)

    used_rows = {top, bottom}
    used_cols = {left, right}
    _add_unpaired_seeds(grid, color_pool[4:], extra_count, used_rows, used_cols)

    rays = _pair_rays(grid)
    random.shuffle(rays)
    wall_hits = 0
    for ray in rays:
        if random.random() < 0.55 and _place_wall_on_ray(grid, tuple(ray["origin"]), tuple(ray["dir"])):
            wall_hits += 1
    if wall_hits == 0:
        for ray in rays:
            if _place_wall_on_ray(grid, tuple(ray["origin"]), tuple(ray["dir"])):
                break
    return grid


def _build_cross(horizontal_first: bool) -> np.ndarray | None:
    height = random.randint(15, 22)
    width = random.randint(15, 22)
    extra_count = random.randint(0, 1)
    color_pool = _available_seed_colors(4 + extra_count)
    grid = _blank_grid(height, width)

    success = False
    for _attempt in range(120):
        row = random.randint(6, height - 5)
        right = random.randint(4, width - 8)
        if horizontal_first:
            max_dh = min(5, width - 3 - right)
            if max_dh < 2:
                continue
            dh = random.randint(2, max_dh)
            max_dv = min(6, row - 2)
            if max_dv < dh + 1:
                continue
            dv = random.randint(dh + 1, max_dv)
        else:
            max_dh = min(6, width - 3 - right)
            if max_dh < 3:
                continue
            dh = random.randint(3, max_dh)
            max_dv = min(dh - 1, row - 2, 5)
            if max_dv < 2:
                continue
            dv = random.randint(2, max_dv)

        cross_col = right + dh
        bottom = row - dv
        if bottom < 3:
            continue
        top = random.randint(2, bottom - 1)
        left = random.randint(2, right - 2)
        success = True
        break
    if not success:
        return None

    seeds = [
        (row, left),
        (row, right),
        (top, cross_col),
        (bottom, cross_col),
    ]
    for color, (seed_row, seed_col) in zip(color_pool[:4], seeds, strict=False):
        _put_seed(grid, seed_row, seed_col, color)

    used_rows = {row, top, bottom}
    used_cols = {left, right, cross_col}
    _add_unpaired_seeds(grid, color_pool[4:], extra_count, used_rows, used_cols)

    # Keep the crossing ray clear up to the intended intersection.
    if horizontal_first:
        _place_wall_on_ray(grid, (row, left), (0, -1))
        _place_wall_on_ray(grid, (top, cross_col), (-1, 0))
        _place_wall_on_ray(grid, (row, right), (0, 1), min_step=dh + 2)
    else:
        _place_wall_on_ray(grid, (row, left), (0, -1))
        _place_wall_on_ray(grid, (top, cross_col), (-1, 0))
        _place_wall_on_ray(grid, (bottom, cross_col), (1, 0), min_step=dv + 2)
    return grid


def _valid_row_only(feat: dict[str, int]) -> bool:
    return (
        feat["row_pairs"] >= 2
        and feat["col_pairs"] == 0
        and feat["unpaired_count"] >= 1
        and feat["wall_stops"] >= 1
        and feat["boundary_stops"] >= 1
        and feat["changed"] >= 6
        and feat["collision_stops"] == 0
    )


def _valid_col_only(feat: dict[str, int]) -> bool:
    return (
        feat["col_pairs"] >= 2
        and feat["row_pairs"] == 0
        and feat["unpaired_count"] >= 1
        and feat["wall_stops"] >= 1
        and feat["boundary_stops"] >= 1
        and feat["changed"] >= 6
        and feat["collision_stops"] == 0
    )


def _valid_rectangle(feat: dict[str, int]) -> bool:
    return (
        feat["row_pairs"] == 2
        and feat["col_pairs"] == 2
        and feat["dual_count"] == 4
        and feat["wall_stops"] >= 1
        and feat["changed"] >= 8
        and feat["collision_stops"] == 0
    )


def _valid_cross(feat: dict[str, int]) -> bool:
    return (
        feat["row_pairs"] == 1
        and feat["col_pairs"] == 1
        and feat["dual_count"] == 0
        and feat["other_stops"] >= 1
        and feat["wall_stops"] >= 1
        and feat["changed"] >= 6
        and feat["collision_stops"] == 0
    )


_BUILDERS: tuple[tuple[object, object], ...] = (
    (_build_row_only, _valid_row_only),
    (_build_col_only, _valid_col_only),
    (_build_rectangle, _valid_rectangle),
    (lambda: _build_cross(horizontal_first=True), _valid_cross),
    (lambda: _build_cross(horizontal_first=False), _valid_cross),
)


def generate_input() -> np.ndarray:
    for _attempt in range(_GEN_ATTEMPTS):
        builder, validator = random.choice(_BUILDERS)
        grid = builder()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        features = _analyze_scene(grid)
        if validator(features):
            return grid
    raise RuntimeError("failed to generate a valid seed-growth puzzle input")
