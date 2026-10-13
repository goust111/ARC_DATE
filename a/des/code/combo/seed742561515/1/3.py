from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 8
TOWER_COLOR = 5
RECT_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)

DIRS = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _plus_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _cross_cell_set(centers: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for center in centers:
        cells.update(_plus_cells(center))
    return cells


def _detect_towers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    gray_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == TOWER_COLOR)}
    centers: list[tuple[int, int]] = []
    owner_count: dict[tuple[int, int], int] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != TOWER_COLOR:
                continue

            cells = _plus_cells((r, c))
            if any(int(g[rr, cc]) != TOWER_COLOR for rr, cc in cells):
                continue

            diagonals = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
            if any(int(g[rr, cc]) == TOWER_COLOR for rr, cc in diagonals):
                continue

            centers.append((r, c))
            for cell in cells:
                owner_count[cell] = owner_count.get(cell, 0) + 1

    assert centers, "expected at least one gray signal tower"
    assert set(owner_count) == gray_cells, "gray pixels must decompose into 3x3 cross towers"
    assert all(count == 1 for count in owner_count.values()), "cross towers must not overlap"
    return sorted(centers)


def _ray_path(
    shape: tuple[int, int],
    cross_cells: set[tuple[int, int]],
    center: tuple[int, int],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    r = center[0] + 2 * dr
    c = center[1] + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        if (r, c) in cross_cells:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _tower_stats(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    centers = _detect_towers(g)
    cross_cells = _cross_cell_set(centers)
    stats: dict[tuple[int, int], dict[str, object]] = {}

    for center in centers:
        ray_paths: dict[str, list[tuple[int, int]]] = {}
        seen_colors: list[int] = []
        for name, (dr, dc) in DIRS:
            path = _ray_path(g.shape, cross_cells, center, dr, dc)
            ray_paths[name] = path
            for r, c in path:
                value = int(g[r, c])
                if value not in (BG_COLOR, TOWER_COLOR):
                    seen_colors.append(value)

        counts = Counter(seen_colors)
        if not counts:
            branch = "none"
            paint_color = None
        elif len(counts) == 1:
            branch = "single"
            paint_color = next(iter(counts))
        else:
            branch = "multi"
            max_count = max(counts.values())
            modes = [color for color, count in counts.items() if count == max_count]
            assert len(modes) == 1, "multi-color towers must have a unique most frequent color"
            paint_color = modes[0]

        stats[center] = {
            "branch": branch,
            "paint_color": paint_color,
            "counts": counts,
            "paths": ray_paths,
        }

    return stats


def _color_components_are_rectangles(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG_COLOR, TOWER_COLOR) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
                return False
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray signal tower using the colors collected by its four outward rays.

    Input structure
    - Background is azure.
    - Gray objects are 3x3 cross towers made of 5 pixels.
    - Colored objects are solid rectangles that do not overlap towers.

    Rule
    - From each arm endpoint of every tower, cast a ray outward until the grid edge or another tower.
    - Collect all non-background, non-gray pixels crossed by the four rays.
    - If exactly one color is collected, recolor the whole tower to that color.
    - If two or more colors are collected, recolor only the center to the unique most frequent color.
    - If no colored pixel is crossed, leave the tower unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    stats = _tower_stats(g)
    out = g.copy()

    for center, info in stats.items():
        branch = info["branch"]
        paint_color = info["paint_color"]
        if branch == "none":
            continue
        assert paint_color is not None
        if branch == "single":
            for r, c in _plus_cells(center):
                out[r, c] = int(paint_color)
        else:
            assert branch == "multi"
            out[center] = int(paint_color)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
CENTER_MARGIN = 4
MIN_CENTER_GAP = 4
BLOCK_PAIR_PROB = 0.4
BLOCK_PAIR_SEPARATION = 6
MAX_BAR_LENGTH = 5
LAYOUT_RETRIES = 120
PLACEMENT_RETRIES = 80
MAX_GENERATE_ATTEMPTS = 400


def _expand_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _sample_spaced_values(
    candidates: list[int],
    count: int,
    min_gap: int,
    forbidden: tuple[int, ...] = (),
) -> list[int] | None:
    if count == 0:
        return []

    for _ in range(LAYOUT_RETRIES):
        pool = candidates[:]
        random.shuffle(pool)
        chosen: list[int] = []
        for value in pool:
            if any(abs(value - bad) < min_gap for bad in forbidden):
                continue
            if any(abs(value - other) < min_gap for other in chosen):
                continue
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None


def _centers_valid(centers: list[tuple[int, int]]) -> bool:
    if len(set(centers)) != len(centers):
        return False
    for i, (r0, c0) in enumerate(centers):
        for r1, c1 in centers[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < MIN_CENTER_GAP:
                return False
    cells = _cross_cell_set(centers)
    return len(cells) == 5 * len(centers)


def _max_tower_count(length: int) -> int:
    available = length - 2 * CENTER_MARGIN
    if available <= 0:
        return 0
    return 1 + (available - 1) // MIN_CENTER_GAP


def _place_centers(
    h: int,
    w: int,
    tower_count: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], set[str]]] | None:
    row_values = list(range(CENTER_MARGIN, h - CENTER_MARGIN))
    col_values = list(range(CENTER_MARGIN, w - CENTER_MARGIN))

    for _ in range(LAYOUT_RETRIES):
        disabled: dict[tuple[int, int], set[str]] = {}
        want_pair = tower_count >= 2 and random.random() < BLOCK_PAIR_PROB

        if not want_pair:
            rows = _sample_spaced_values(row_values, tower_count, MIN_CENTER_GAP)
            cols = _sample_spaced_values(col_values, tower_count, MIN_CENTER_GAP)
            if rows is None or cols is None:
                continue
            random.shuffle(cols)
            centers = list(zip(rows, cols))
            if _centers_valid(centers):
                return sorted(centers), disabled
            continue

        horizontal_pair = random.random() < 0.5
        if horizontal_pair:
            shared_row = random.choice(row_values)
            pair = None
            for _ in range(PLACEMENT_RETRIES):
                left_col, right_col = sorted(random.sample(col_values, 2))
                if right_col - left_col >= BLOCK_PAIR_SEPARATION:
                    pair = ((shared_row, left_col), (shared_row, right_col))
                    break
            if pair is None:
                continue

            left_center, right_center = pair
            disabled[left_center] = {"right"}
            disabled[right_center] = {"left"}

            extra_rows = _sample_spaced_values(
                row_values,
                tower_count - 2,
                MIN_CENTER_GAP,
                forbidden=(shared_row,),
            )
            extra_cols = _sample_spaced_values(
                col_values,
                tower_count - 2,
                MIN_CENTER_GAP,
                forbidden=(left_center[1], right_center[1]),
            )
            if extra_rows is None or extra_cols is None:
                continue
            random.shuffle(extra_cols)
            centers = [left_center, right_center] + list(zip(extra_rows, extra_cols))
        else:
            shared_col = random.choice(col_values)
            pair = None
            for _ in range(PLACEMENT_RETRIES):
                top_row, bottom_row = sorted(random.sample(row_values, 2))
                if bottom_row - top_row >= BLOCK_PAIR_SEPARATION:
                    pair = ((top_row, shared_col), (bottom_row, shared_col))
                    break
            if pair is None:
                continue

            top_center, bottom_center = pair
            disabled[top_center] = {"down"}
            disabled[bottom_center] = {"up"}

            extra_rows = _sample_spaced_values(
                row_values,
                tower_count - 2,
                MIN_CENTER_GAP,
                forbidden=(top_center[0], bottom_center[0]),
            )
            extra_cols = _sample_spaced_values(
                col_values,
                tower_count - 2,
                MIN_CENTER_GAP,
                forbidden=(shared_col,),
            )
            if extra_rows is None or extra_cols is None:
                continue
            random.shuffle(extra_cols)
            centers = [top_center, bottom_center] + list(zip(extra_rows, extra_cols))

        if _centers_valid(centers):
            return sorted(centers), disabled

    return None


def _path_capacity(path: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> int:
    best = 0
    run = 0
    for cell in path:
        if cell in reserved:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def _paint_rectangle(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    path: list[tuple[int, int]],
    orientation: str,
    color: int,
    length: int,
) -> bool:
    h, w = map(int, grid.shape)
    candidates: list[tuple[tuple[int, int, int, int], set[tuple[int, int]]]] = []

    for start in range(0, len(path) - length + 1):
        segment = path[start : start + length]
        if any(cell in reserved for cell in segment):
            continue

        rows = [r for r, _ in segment]
        cols = [c for _, c in segment]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        boxes: list[tuple[int, int, int, int]] = []
        boxes.append((r0, c0, r1, c1))
        if orientation == "horizontal":
            if r0 - 1 >= 0:
                boxes.append((r0 - 1, c0, r1, c1))
            if r1 + 1 < h:
                boxes.append((r0, c0, r1 + 1, c1))
        else:
            if c0 - 1 >= 0:
                boxes.append((r0, c0 - 1, r1, c1))
            if c1 + 1 < w:
                boxes.append((r0, c0, r1, c1 + 1))

        for box in boxes:
            br0, bc0, br1, bc1 = box
            cells = {
                (rr, cc)
                for rr in range(br0, br1 + 1)
                for cc in range(bc0, bc1 + 1)
            }
            if any(cell in reserved for cell in cells):
                continue
            candidates.append((box, cells))

    if not candidates:
        return False

    box, cells = random.choice(candidates)
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    reserved.update(_expand_cells(cells, grid.shape, radius=1))
    return True


def _sample_branch_types(tower_count: int) -> list[str]:
    base = ["none", "single", "multi"]
    if tower_count >= 3:
        types = base + [random.choice(base) for _ in range(tower_count - 3)]
    else:
        types = random.sample(base, tower_count)
    random.shuffle(types)
    return types


def _try_place_single(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    paths: dict[str, list[tuple[int, int]]],
    usable_dirs: list[str],
    palette: tuple[int, ...],
) -> bool:
    color = random.choice(palette)
    dir_count = random.randint(1, min(3, len(usable_dirs)))
    chosen_dirs = random.sample(usable_dirs, dir_count)

    for name in chosen_dirs:
        path = paths[name]
        capacity = _path_capacity(path, reserved)
        if capacity <= 0:
            return False
        length = random.randint(1, min(MAX_BAR_LENGTH, capacity))
        orientation = "horizontal" if name in ("left", "right") else "vertical"
        if not _paint_rectangle(grid, reserved, path, orientation, color, length):
            return False
    return True


def _try_place_multi(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    paths: dict[str, list[tuple[int, int]]],
    usable_dirs: list[str],
    palette: tuple[int, ...],
) -> bool:
    if len(usable_dirs) < 2 or len(palette) < 2:
        return False

    color_count = 2
    if len(usable_dirs) >= 3 and len(palette) >= 3 and random.random() < 0.35:
        color_count = 3

    chosen_colors = random.sample(palette, color_count)
    chosen_dirs = random.sample(usable_dirs, color_count)

    dominant_dir = chosen_dirs[0]
    dominant_capacity = _path_capacity(paths[dominant_dir], reserved)
    if dominant_capacity < 2:
        return False
    dominant_length = random.randint(2, min(MAX_BAR_LENGTH, dominant_capacity))

    plan: list[tuple[str, int, int]] = [(dominant_dir, chosen_colors[0], dominant_length)]
    for name, color in zip(chosen_dirs[1:], chosen_colors[1:]):
        capacity = _path_capacity(paths[name], reserved)
        high = min(MAX_BAR_LENGTH, capacity, dominant_length - 1)
        if high < 1:
            return False
        plan.append((name, color, random.randint(1, high)))

    for name, color, length in plan:
        orientation = "horizontal" if name in ("left", "right") else "vertical"
        if not _paint_rectangle(grid, reserved, paths[name], orientation, color, length):
            return False
    return True


def _place_noise_rectangles(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
    palette: tuple[int, ...],
) -> None:
    h, w = map(int, grid.shape)
    for _ in range(random.randint(0, 2)):
        for _ in range(PLACEMENT_RETRIES):
            rect_h = random.randint(1, 3)
            rect_w = random.randint(1, 4)
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            cells = {
                (rr, cc)
                for rr in range(r0, r0 + rect_h)
                for cc in range(c0, c0 + rect_w)
            }
            if cells & reserved:
                continue
            if cells & ray_cells:
                continue
            color = random.choice(palette)
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            reserved.update(_expand_cells(cells, grid.shape, radius=1))
            break


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    max_towers = min(5, _max_tower_count(h), _max_tower_count(w))
    if max_towers < 2:
        return None
    tower_count = random.randint(2, max_towers)

    placement = _place_centers(h, w, tower_count)
    if placement is None:
        return None
    centers, disabled_dirs = placement

    grid = np.full((h, w), BG_COLOR, dtype=int)
    cross_cells = _cross_cell_set(centers)
    for r, c in cross_cells:
        grid[r, c] = TOWER_COLOR

    reserved = _expand_cells(cross_cells, grid.shape, radius=1)
    all_paths: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = {}
    ray_cells: set[tuple[int, int]] = set()
    for center in centers:
        per_tower: dict[str, list[tuple[int, int]]] = {}
        for name, (dr, dc) in DIRS:
            path = _ray_path(grid.shape, cross_cells, center, dr, dc)
            per_tower[name] = path
            ray_cells.update(path)
        all_paths[center] = per_tower

    palette = tuple(random.sample(RECT_COLORS, random.randint(3, 5)))
    branch_types = _sample_branch_types(tower_count)
    expected = {center: branch for center, branch in zip(centers, branch_types)}

    work_order = sorted(
        centers,
        key=lambda center: {"multi": 0, "single": 1, "none": 2}[expected[center]],
    )

    for center in work_order:
        branch = expected[center]
        paths = all_paths[center]
        blocked = disabled_dirs.get(center, set())
        usable_dirs = [
            name
            for name, _delta in DIRS
            if name not in blocked and _path_capacity(paths[name], reserved) > 0
        ]

        if branch == "none":
            continue
        if branch == "single":
            if not usable_dirs or not _try_place_single(grid, reserved, paths, usable_dirs, palette):
                return None
        else:
            assert branch == "multi"
            if not _try_place_multi(grid, reserved, paths, usable_dirs, palette):
                return None

    _place_noise_rectangles(grid, reserved, ray_cells, palette)

    try:
        stats = _tower_stats(grid)
        if set(stats) != set(expected):
            return None
        if any(str(stats[center]["branch"]) != expected[center] for center in expected):
            return None
        if not _color_components_are_rectangles(grid):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
