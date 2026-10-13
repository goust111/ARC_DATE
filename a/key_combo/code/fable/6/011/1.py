from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import bbox_coords, find_components, get_bg


CoordArray = np.ndarray

GRID_MIN = 14
GRID_MAX = 24
GEN_TRIES = 400
PLACE_TRIES = 120


def _normalize_coords(coords: set[tuple[int, int]]) -> CoordArray:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    arr[:, 0] -= int(r0)
    arr[:, 1] -= int(c0)
    return arr


def _random_shape() -> CoordArray:
    target_area = random.randint(3, 6)
    cells = {(0, 0)}
    frontier = {(0, 0)}
    while len(cells) < target_area:
        base_r, base_c = random.choice(sorted(frontier))
        nbrs = [(base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)]
        random.shuffle(nbrs)
        added = False
        for rr, cc in nbrs:
            if -3 <= rr <= 3 and -3 <= cc <= 3 and (rr, cc) not in cells:
                cells.add((rr, cc))
                frontier.add((rr, cc))
                added = True
                break
        if not added:
            frontier = set(cells)
    return _normalize_coords(cells)


def _shape_dims(shape: CoordArray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _place_shape(anchor: tuple[int, int], shape: CoordArray) -> CoordArray:
    r0, c0 = map(int, anchor)
    return shape + np.array([r0, c0], dtype=int)


def _paint_cells(grid: np.ndarray, coords: CoordArray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _classify_singleton_color(coords: list[tuple[int, int]]) -> tuple[str, int]:
    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r, c in coords:
        row_groups.setdefault(int(r), []).append(int(c))
        col_groups.setdefault(int(c), []).append(int(r))

    if row_groups and all(len(cols) == 2 for cols in row_groups.values()):
        sums = {cols[0] + cols[1] for cols in row_groups.values()}
        if len(sums) == 1:
            total = next(iter(sums))
            if total % 2 == 0:
                return "h", total // 2

    if col_groups and all(len(rows) == 2 for rows in col_groups.values()):
        sums = {rows[0] + rows[1] for rows in col_groups.values()}
        if len(sums) == 1:
            total = next(iter(sums))
            if total % 2 == 0:
                return "v", total // 2

    raise ValueError("singleton color does not define a valid hidden wall family")


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, set[int], set[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    singletons = [comp for comp in comps if int(comp["area"]) == 1]
    objects = [comp for comp in comps if int(comp["area"]) > 1]

    by_color: dict[int, list[tuple[int, int]]] = {}
    for comp in singletons:
        color = int(comp["color"])
        r, c = map(int, comp["coords"][0])
        by_color.setdefault(color, []).append((r, c))

    wall_r: int | None = None
    wall_c: int | None = None
    horizontal_colors: set[int] = set()
    vertical_colors: set[int] = set()

    for color, coords in by_color.items():
        orientation, axis = _classify_singleton_color(coords)
        if orientation == "h":
            horizontal_colors.add(color)
            if wall_c is None:
                wall_c = axis
            else:
                assert wall_c == axis
        else:
            vertical_colors.add(color)
            if wall_r is None:
                wall_r = axis
            else:
                assert wall_r == axis

    assert wall_r is not None and wall_c is not None
    return bg, int(wall_r), int(wall_c), horizontal_colors, vertical_colors, objects


def _slide_objects(
    shape: tuple[int, int],
    objects: list[dict],
    *,
    axis: str,
    wall_index: int,
    direction: int,
) -> list[dict]:
    h, w = map(int, shape)
    occupancy = np.zeros((h, w), dtype=bool)
    result: list[dict] = []

    if axis == "h" and direction > 0:
        ordered = sorted(objects, key=lambda obj: int(obj["bbox"][3]), reverse=True)
    elif axis == "h":
        ordered = sorted(objects, key=lambda obj: int(obj["bbox"][1]))
    elif direction > 0:
        ordered = sorted(objects, key=lambda obj: int(obj["bbox"][2]), reverse=True)
    else:
        ordered = sorted(objects, key=lambda obj: int(obj["bbox"][0]))

    delta = np.array([0, direction], dtype=int) if axis == "h" else np.array([direction, 0], dtype=int)

    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int)
        shift = 0
        while True:
            moved = coords + delta * (shift + 1)
            if moved[:, 0].min() < 0 or moved[:, 1].min() < 0 or moved[:, 0].max() >= h or moved[:, 1].max() >= w:
                break
            if axis == "h":
                if np.any(moved[:, 1] == int(wall_index)):
                    break
            else:
                if np.any(moved[:, 0] == int(wall_index)):
                    break
            if occupancy[moved[:, 0], moved[:, 1]].any():
                break
            shift += 1

        final_coords = coords + delta * shift
        occupancy[final_coords[:, 0], final_coords[:, 1]] = True
        result.append(
            {
                "color": int(obj["color"]),
                "coords": final_coords,
                "bbox": bbox_coords(final_coords),
            }
        )
    return result


def _solve_from_objects(
    shape: tuple[int, int],
    bg: int,
    wall_r: int,
    wall_c: int,
    horizontal_colors: set[int],
    vertical_colors: set[int],
    objects: list[dict],
) -> np.ndarray:
    allowed_colors = set(horizontal_colors) | set(vertical_colors)
    assert all(int(obj["color"]) in allowed_colors for obj in objects)

    horizontal_objects = [obj for obj in objects if int(obj["color"]) in horizontal_colors]
    vertical_objects = [obj for obj in objects if int(obj["color"]) in vertical_colors]

    left = [obj for obj in horizontal_objects if int(obj["bbox"][3]) < wall_c]
    right = [obj for obj in horizontal_objects if int(obj["bbox"][1]) > wall_c]
    top = [obj for obj in vertical_objects if int(obj["bbox"][2]) < wall_r]
    bottom = [obj for obj in vertical_objects if int(obj["bbox"][0]) > wall_r]

    moved = []
    moved.extend(_slide_objects(shape, left, axis="h", wall_index=wall_c, direction=1))
    moved.extend(_slide_objects(shape, right, axis="h", wall_index=wall_c, direction=-1))
    moved.extend(_slide_objects(shape, top, axis="v", wall_index=wall_r, direction=1))
    moved.extend(_slide_objects(shape, bottom, axis="v", wall_index=wall_r, direction=-1))

    out = np.full(shape, int(bg), dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for obj in moved:
        coords = np.asarray(obj["coords"], dtype=int)
        assert not occupied[coords[:, 0], coords[:, 1]].any()
        occupied[coords[:, 0], coords[:, 1]] = True
        _paint_cells(out, coords, int(obj["color"]))
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden horizontal and vertical wall lines from symmetric singleton pairs,
    erase those singleton guides, and slide each monochrome object toward the wall family
    indicated by its color until it docks against that unseen wall.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, wall_r, wall_c, horizontal_colors, vertical_colors, objects = _extract_scene(g)
    return _solve_from_objects(g.shape, bg, wall_r, wall_c, horizontal_colors, vertical_colors, objects)


def _cells_free(occupied: set[tuple[int, int]], coords: CoordArray) -> bool:
    return all((int(r), int(c)) not in occupied for r, c in coords)


def _same_color_isolated(grid: np.ndarray, bg: int, color: int, cell: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = map(int, cell)
    if int(grid[r, c]) != int(bg):
        return False
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
            return False
    return True


def _try_add_horizontal_pairs(grid: np.ndarray, bg: int, color: int, wall_c: int, used_rows: set[int]) -> bool:
    h, w = grid.shape
    pair_count = random.randint(1, 3)
    for _ in range(pair_count):
        placed = False
        for _ in range(PLACE_TRIES):
            row = random.randrange(h)
            dist_max = min(wall_c, w - 1 - wall_c)
            dist = random.randint(1, dist_max)
            left = (row, wall_c - dist)
            right = (row, wall_c + dist)
            if row in used_rows:
                continue
            if not (_same_color_isolated(grid, bg, color, left) and _same_color_isolated(grid, bg, color, right)):
                continue
            grid[left] = int(color)
            grid[right] = int(color)
            used_rows.add(row)
            placed = True
            break
        if not placed:
            return False
    return True


def _try_add_vertical_pairs(grid: np.ndarray, bg: int, color: int, wall_r: int, used_cols: set[int]) -> bool:
    h, w = grid.shape
    pair_count = random.randint(1, 3)
    for _ in range(pair_count):
        placed = False
        for _ in range(PLACE_TRIES):
            col = random.randrange(w)
            dist_max = min(wall_r, h - 1 - wall_r)
            dist = random.randint(1, dist_max)
            top = (wall_r - dist, col)
            bottom = (wall_r + dist, col)
            if col in used_cols:
                continue
            if not (_same_color_isolated(grid, bg, color, top) and _same_color_isolated(grid, bg, color, bottom)):
                continue
            grid[top] = int(color)
            grid[bottom] = int(color)
            used_cols.add(col)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randint(0, 9)

    wall_r = random.randint(4, h - 5)
    wall_c = random.randint(4, w - 5)

    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    h_color_count = random.randint(1, 3)
    v_color_count = random.randint(1, 3)
    horizontal_colors = set(palette[:h_color_count])
    vertical_colors = set(palette[h_color_count : h_color_count + v_color_count])

    object_specs: list[dict] = []
    final_cells: set[tuple[int, int]] = set()
    input_cells: set[tuple[int, int]] = set()

    def add_object(axis: str, side: str, color: int) -> bool:
        nonlocal final_cells, input_cells
        shape = _random_shape()
        shape_h, shape_w = _shape_dims(shape)
        for _ in range(PLACE_TRIES):
            if axis == "h":
                final_c = wall_c - shape_w if side == "left" else wall_c + 1
                if side == "left":
                    max_shift = final_c
                    input_c = final_c - random.randint(0, max_shift)
                else:
                    max_shift = w - (final_c + shape_w)
                    input_c = final_c + random.randint(0, max_shift)
                row_starts = []
                if wall_r - shape_h >= 0:
                    row_starts.extend(range(0, wall_r - shape_h + 1))
                if wall_r + 1 <= h - shape_h:
                    row_starts.extend(range(wall_r + 1, h - shape_h + 1))
                if not row_starts:
                    continue
                input_r = random.choice(row_starts)
                final_r = input_r
            else:
                final_r = wall_r - shape_h if side == "top" else wall_r + 1
                if side == "top":
                    max_shift = final_r
                    input_r = final_r - random.randint(0, max_shift)
                else:
                    max_shift = h - (final_r + shape_h)
                    input_r = final_r + random.randint(0, max_shift)
                col_starts = []
                if wall_c - shape_w >= 0:
                    col_starts.extend(range(0, wall_c - shape_w + 1))
                if wall_c + 1 <= w - shape_w:
                    col_starts.extend(range(wall_c + 1, w - shape_w + 1))
                if not col_starts:
                    continue
                input_c = random.choice(col_starts)
                final_c = input_c

            input_coords = _place_shape((input_r, input_c), shape)
            final_coords = _place_shape((final_r, final_c), shape)

            if not _cells_free(final_cells, final_coords) or not _cells_free(input_cells, input_coords):
                continue
            final_cells |= {tuple(map(int, rc)) for rc in final_coords}
            input_cells |= {tuple(map(int, rc)) for rc in input_coords}
            object_specs.append(
                {
                    "color": int(color),
                    "coords": input_coords,
                    "bbox": bbox_coords(input_coords),
                    "axis": axis,
                    "side": side,
                }
            )
            return True
        return False

    for color in sorted(horizontal_colors):
        side = random.choice(["left", "right"])
        if not add_object("h", side, color):
            raise ValueError("failed to place horizontal object")
        if random.random() < 0.45:
            if not add_object("h", side, color):
                raise ValueError("failed to place extra horizontal object")

    for color in sorted(vertical_colors):
        side = random.choice(["top", "bottom"])
        if not add_object("v", side, color):
            raise ValueError("failed to place vertical object")
        if random.random() < 0.45:
            if not add_object("v", side, color):
                raise ValueError("failed to place extra vertical object")

    expected = _solve_from_objects((h, w), bg, wall_r, wall_c, horizontal_colors, vertical_colors, object_specs)

    grid = np.full((h, w), int(bg), dtype=int)
    for obj in object_specs:
        _paint_cells(grid, np.asarray(obj["coords"], dtype=int), int(obj["color"]))

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for color in sorted(horizontal_colors):
        if not _try_add_horizontal_pairs(grid, bg, color, wall_c, used_rows):
            raise ValueError("failed to place horizontal singleton pairs")
    for color in sorted(vertical_colors):
        if not _try_add_vertical_pairs(grid, bg, color, wall_r, used_cols):
            raise ValueError("failed to place vertical singleton pairs")

    produced = main(grid)
    if not np.array_equal(produced, expected):
        raise ValueError("generated case does not round-trip through solver")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        try:
            grid, _expected = _build_case()
            return grid
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid hidden-wall dispatch instance")
