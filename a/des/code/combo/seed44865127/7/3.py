from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
OBJECT_COUNT_RANGE = (3, 6)
MAX_OBJECT_AREA = 8
MAX_OBJECT_SPAN = 7
MAX_GENERATION_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 500
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return (
        1 + max(r for r, _ in cells),
        1 + max(c for _, c in cells),
    )


def _shape_perimeter(cells: set[tuple[int, int]]) -> int:
    perimeter = 0
    for r, c in cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in cells:
                perimeter += 1
    return perimeter


def _is_rectangle(cells: set[tuple[int, int]]) -> bool:
    h, w = _shape_dims(cells)
    return len(cells) == h * w


def _rotate90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize_cells({(c, -r) for r, c in cells})


def _mirror(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize_cells({(r, -c) for r, c in cells})


def _random_transform(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(random.randint(0, 3)):
        out = _rotate90(out)
    if random.random() < 0.5:
        out = _mirror(out)
    return _normalize_cells(out)


def _make_rectangle() -> set[tuple[int, int]]:
    h = random.randint(1, 4)
    w = random.randint(1, 5)
    return {(r, c) for r in range(h) for c in range(w)}


def _make_line() -> set[tuple[int, int]]:
    length = random.randint(2, MAX_OBJECT_SPAN)
    return {(0, c) for c in range(length)}


def _make_l_shape() -> set[tuple[int, int]]:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    return {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}


def _make_polyomino() -> set[tuple[int, int]]:
    area = random.randint(4, MAX_OBJECT_AREA)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _sample_shape() -> set[tuple[int, int]] | None:
    kind = random.choice(("rectangle", "line", "ell", "polyomino"))
    if kind == "rectangle":
        cells = _make_rectangle()
    elif kind == "line":
        cells = _make_line()
    elif kind == "ell":
        cells = _make_l_shape()
    else:
        cells = _make_polyomino()
    cells = _random_transform(cells)
    h, w = _shape_dims(cells)
    if max(h, w) > MAX_OBJECT_SPAN or len(cells) > MAX_OBJECT_AREA:
        return None
    return cells


def _has_perimeter_area_contrast(objects: list[dict]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            area_i = int(objects[i]["area"])
            area_j = int(objects[j]["area"])
            perimeter_i = int(objects[i]["perimeter"])
            perimeter_j = int(objects[j]["perimeter"])
            if area_i == area_j and perimeter_i != perimeter_j:
                return True
            if (area_i - area_j) * (perimeter_i - perimeter_j) < 0:
                return True
    return False


def _sample_object_specs(count: int) -> list[dict] | None:
    objects: list[dict] = []
    used_perimeters: set[int] = set()
    for _ in range(MAX_SHAPE_ATTEMPTS):
        if len(objects) == count:
            break
        cells = _sample_shape()
        if cells is None:
            continue
        perimeter = _shape_perimeter(cells)
        if perimeter in used_perimeters:
            continue
        obj = {
            "cells": cells,
            "area": len(cells),
            "perimeter": perimeter,
            "dims": _shape_dims(cells),
            "is_rectangle": _is_rectangle(cells),
        }
        objects.append(obj)
        used_perimeters.add(perimeter)
    if len(objects) != count:
        return None
    if all(bool(obj["is_rectangle"]) for obj in objects):
        return None
    if not _has_perimeter_area_contrast(objects):
        return None
    return objects


def _can_place(blocked: np.ndarray, cells: set[tuple[int, int]], top: int, left: int) -> bool:
    for r, c in cells:
        if bool(blocked[top + r, left + c]):
            return False
    return True


def _paint_shape(grid: np.ndarray, cells: set[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]], top: int, left: int) -> None:
    height, width = blocked.shape
    for r, c in cells:
        gr = top + r
        gc = left + c
        for rr in range(max(0, gr - 1), min(height, gr + 2)):
            for cc in range(max(0, gc - 1), min(width, gc + 2)):
                blocked[rr, cc] = True


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            objects.append({"color": color, "coords": arr, "bbox": bbox})
    return objects


def _grid_perimeter(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    perimeter = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < height and 0 <= nc < width) or int(g[nr, nc]) == bg:
                perimeter += 1
    return perimeter


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object with a horizontal bar whose length equals the
    object's 4-neighbor perimeter, then alternate the sorted bars between the
    top and bottom halves of the output canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _find_objects(g, bg)
    assert objects, "expected at least one foreground object"

    for obj in objects:
        obj["perimeter"] = _grid_perimeter(g, obj["coords"], bg)

    objects.sort(key=lambda obj: (int(obj["perimeter"]), obj["bbox"][0], obj["bbox"][1]))
    count = len(objects)
    out = np.full((2 * count, max(int(obj["perimeter"]) for obj in objects)), bg, dtype=int)

    top_row = 0
    bottom_row = 2 * count - 1
    for rank, obj in enumerate(objects, start=1):
        row = top_row if rank % 2 == 1 else bottom_row
        out[row, : int(obj["perimeter"])] = int(obj["color"])
        if rank % 2 == 1:
            top_row += 1
        else:
            bottom_row -= 1

    return out


def _expected_output(bg: int, objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: (int(obj["perimeter"]), obj["bbox"][0], obj["bbox"][1]))
    count = len(ordered)
    out = np.full((2 * count, max(int(obj["perimeter"]) for obj in ordered)), bg, dtype=int)
    top_row = 0
    bottom_row = 2 * count - 1
    for rank, obj in enumerate(ordered, start=1):
        row = top_row if rank % 2 == 1 else bottom_row
        out[row, : int(obj["perimeter"])] = int(obj["color"])
        if rank % 2 == 1:
            top_row += 1
        else:
            bottom_row -= 1
    return out


def _place_objects(height: int, width: int, bg: int, objects: list[dict], colors: list[int]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    placed: list[dict | None] = [None] * len(objects)

    order = sorted(range(len(objects)), key=lambda idx: (objects[idx]["dims"][0] * objects[idx]["dims"][1], objects[idx]["area"]), reverse=True)
    for idx in order:
        obj = objects[idx]
        obj_h, obj_w = obj["dims"]
        if height - obj_h <= 1 or width - obj_w <= 1:
            return None
        candidates = [(top, left) for top in range(1, height - obj_h) for left in range(1, width - obj_w)]
        random.shuffle(candidates)
        for top, left in candidates:
            if not _can_place(blocked, obj["cells"], top, left):
                continue
            _paint_shape(grid, obj["cells"], top, left, colors[idx])
            _mark_blocked(blocked, obj["cells"], top, left)
            placed[idx] = {
                "color": colors[idx],
                "perimeter": obj["perimeter"],
                "bbox": (top, left, top + obj_h - 1, left + obj_w - 1),
            }
            break
        if placed[idx] is None:
            return None

    return grid, [obj for obj in placed if obj is not None]


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    objects = _sample_object_specs(count)
    if objects is None:
        return None

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], count)
    min_h = max(GRID_HEIGHT_RANGE[0], max(int(obj["dims"][0]) for obj in objects) + 2)
    min_w = max(GRID_WIDTH_RANGE[0], max(int(obj["dims"][1]) for obj in objects) + 2)
    height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_w, GRID_WIDTH_RANGE[1])

    placed_result = _place_objects(height, width, bg, objects, colors)
    if placed_result is None:
        return None
    grid, placed_objects = placed_result

    if _get_bg(grid) != bg:
        return None
    if len(_find_objects(grid, bg)) != count:
        return None
    if len({int(obj["perimeter"]) for obj in placed_objects}) != count:
        return None

    expected = _expected_output(bg, placed_objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
