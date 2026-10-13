from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_objects(grid: np.ndarray) -> list[dict]:
    """Minimal 4-connected monochrome component finder following arc_utils conventions."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_coords(arr)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "coord_set": set(coords),
                    "bbox": bbox,
                    "area": int(arr.shape[0]),
                    "center_col": (bbox[1] + bbox[3]) // 2,
                }
            )
    return objects


def _adjacent_to_ray(coord_set: set[tuple[int, int]], ray_cells: set[tuple[int, int]]) -> bool:
    for r, c in coord_set:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in ray_cells:
                return True
    return False


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _find_objects(g)

    projected = g.copy()
    centers = [obj["center_col"] for obj in objects]
    for center in centers:
        projected[h - 1, center] = YELLOW

    ray_cells_by_source: list[set[tuple[int, int]]] = []
    for center in centers:
        source_cells: set[tuple[int, int]] = set()
        for dc in (-1, 1):
            r = h - 2
            c = center + dc
            while _in_bounds((h, w), r, c):
                value = int(projected[r, c])
                if value not in (BG, YELLOW):
                    break
                projected[r, c] = YELLOW
                source_cells.add((r, c))
                r -= 1
                c += dc
        ray_cells_by_source.append(source_cells)

    hit_sources: list[set[int]] = []
    touched: list[bool] = []
    for obj in objects:
        sources = {
            idx
            for idx, ray_cells in enumerate(ray_cells_by_source)
            if _adjacent_to_ray(obj["coord_set"], ray_cells)
        }
        hit_sources.append(sources)
        touched.append(bool(sources))

    return {
        "objects": objects,
        "projected": projected,
        "ray_cells_by_source": ray_cells_by_source,
        "touched": touched,
        "hit_sources": hit_sources,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Project each object's horizontal center to the bottom row, emit diagonal yellow rays, and recolor hit objects.

    Input structure:
    - Black (0) background fills the grid, and the last row is empty.
    - Three or four foreground objects appear in the upper part of the grid.
    - Each object is a single-color 4-connected shape with its own bounding box.

    Rule:
    - For every foreground object, compute the horizontal center column of its bounding box and place a yellow (4) marker on the bottom row in that column.
    - From each bottom marker, extend one ray diagonally up-left and one diagonally up-right; each ray paints only black cells yellow and stops before the first non-black, non-yellow cell or at the grid edge.
    - If any object has at least one pixel whose 4-neighborhood contains a yellow ray cell, recolor that entire object to yellow; untouched objects keep their original color.

    Key insight:
    - The bottom markers are determined by object-centered projection, and those projected marker positions uniquely determine which diagonal paths can remotely touch other objects.
    """
    analysis = _analyze_scene(np.asarray(input_grid, dtype=int))
    out = analysis["projected"].copy()
    for obj, touched in zip(analysis["objects"], analysis["touched"]):
        if touched:
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = YELLOW
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_connected_shape() -> np.ndarray:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w))
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue
        norm = _normalize_cells(cells)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        height = max(rows) + 1
        width = max(cols) + 1
        if height < 2 or width < 2:
            continue
        if 0 not in cols or (width - 1) not in cols:
            continue
        if 0 not in rows or (height - 1) not in rows:
            continue
        return np.array(sorted(norm), dtype=int)
    raise RuntimeError("failed to sample a connected shape")


def _target_shape_variant(direction: int) -> np.ndarray:
    patterns = [
        {(0, 0), (-1, 0), (-1, 1)},
        {(0, 0), (-1, 0), (-1, 1), (0, 1)},
        {(0, 0), (-1, 0), (-1, 1), (-2, 0)},
        {(0, 0), (-1, 0), (-1, 1), (-2, 0), (-2, 1)},
        {(0, 0), (-1, 0), (-1, 1), (-2, 0), (0, 1)},
    ]
    cells = random.choice(patterns)
    if direction == -1:
        cells = {(r, -c) for r, c in cells}
    return np.array(sorted(cells), dtype=int)


def _absolute_coords(local_coords: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + int(r), left + int(c)) for r, c in local_coords}


def _bbox_from_set(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap_or_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _build_grid(
    shape: tuple[int, int],
    placed: list[dict],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for obj in placed:
        for r, c in obj["cells"]:
            grid[r, c] = obj["color"]
    return grid


def _place_random_object(
    h: int,
    w: int,
    placed: list[dict],
    color: int,
    upper_band_last: int,
) -> dict | None:
    for _ in range(150):
        local = _random_connected_shape()
        local_h = int(local[:, 0].max()) + 1
        local_w = int(local[:, 1].max()) + 1
        if local_h > upper_band_last:
            continue
        top = random.randint(1, upper_band_last - local_h + 1)
        left = random.randint(1, w - local_w - 1)
        cells = _absolute_coords(local, top, left)
        bbox = _bbox_from_set(cells)
        if any(_boxes_overlap_or_touch(bbox, other["bbox"]) for other in placed):
            continue
        return {"cells": cells, "bbox": bbox, "color": color}
    return None


def _validate_generated_scene(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(g[h - 1, :] == BG):
        return False

    analysis = _analyze_scene(g)
    objects = analysis["objects"]
    if len(objects) not in (3, 4):
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    if any(not (3 <= obj["area"] <= 6) for obj in objects):
        return False
    if any(obj["bbox"][2] >= h - 1 for obj in objects):
        return False

    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects):
            if i >= j:
                continue
            if _boxes_overlap_or_touch(obj_i["bbox"], obj_j["bbox"], pad=0):
                return False
        for j, obj_j in enumerate(objects):
            if i == j:
                continue
            if obj_j["bbox"][1] <= obj_i["center_col"] <= obj_j["bbox"][3]:
                return False

    if sum(len(cells) for cells in analysis["ray_cells_by_source"]) == 0:
        return False

    touched = analysis["touched"]
    if not any(touched) or all(touched):
        return False

    remote_hits = [
        any(source != idx for source in sources)
        for idx, sources in enumerate(analysis["hit_sources"])
    ]
    if not any(remote_hits):
        return False
    if any(sources == {idx} for idx, sources in enumerate(analysis["hit_sources"])):
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if int(np.count_nonzero(out == YELLOW)) <= int(np.count_nonzero(g == YELLOW)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    upper_band_last = min(9, h - 5)
    if upper_band_last < 5:
        return None

    n_objects = random.randint(3, 4)
    colors = random.sample(PALETTE, n_objects)

    direction = random.choice((-1, 1))
    source_local = _random_connected_shape()
    source_h = int(source_local[:, 0].max()) + 1
    source_w = int(source_local[:, 1].max()) + 1
    center_offset = (source_w - 1) // 2

    hit_row_max = min(upper_band_last - 1, 8)
    if hit_row_max < 2:
        return None
    hit_row = random.randint(2, hit_row_max)
    ray_len = h - 1 - hit_row

    left_min = 1
    left_max = w - source_w - 1
    if direction == 1:
        left_max = min(left_max, w - 3 - center_offset - ray_len)
    else:
        left_min = max(left_min, 2 - center_offset + ray_len)
    if left_min > left_max:
        return None

    source_top_min = max(3, hit_row + 3)
    source_top_max = upper_band_last - source_h + 1
    if source_top_min > source_top_max:
        return None

    source_left = random.randint(left_min, left_max)
    source_top = random.randint(source_top_min, source_top_max)
    source_cells = _absolute_coords(source_local, source_top, source_left)
    source_bbox = _bbox_from_set(source_cells)
    source_center = (source_bbox[1] + source_bbox[3]) // 2
    ray_col = source_center + direction * ray_len

    target_local = _target_shape_variant(direction)
    target_cols = [int(c) for _, c in target_local]
    target_rows = [int(r) for r, _ in target_local]
    target_left = ray_col + direction - min(target_cols)
    target_top = hit_row - max(target_rows)
    target_cells = _absolute_coords(target_local, target_top, target_left)
    target_bbox = _bbox_from_set(target_cells)
    if (
        target_bbox[0] < 1
        or target_bbox[1] < 1
        or target_bbox[2] > upper_band_last
        or target_bbox[3] > w - 2
    ):
        return None
    if _boxes_overlap_or_touch(source_bbox, target_bbox):
        return None

    placed = [
        {"cells": source_cells, "bbox": source_bbox, "color": colors[0]},
        {"cells": target_cells, "bbox": target_bbox, "color": colors[1]},
    ]

    for color in colors[2:]:
        obj = _place_random_object(h, w, placed, color, upper_band_last)
        if obj is None:
            return None
        placed.append(obj)

    random.shuffle(placed)
    grid = _build_grid((h, w), placed)
    if not _validate_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
