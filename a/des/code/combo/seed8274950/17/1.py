from __future__ import annotations

from collections import Counter
import random

import numpy as np


CORE_COLOR = 9
THREE_SIDE_COLOR = 2
EDGE_COLOR = 7
LEAF_COLOR = 4

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (13, 24)
GRID_W_RANGE = (13, 24)
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 120
WALK_ATTEMPTS = 200
SAFE_INPUT_COLORS = [0, 1, 3, 5, 6, 8]


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int((border == bg).sum()) * 2 <= int(border.size):
        values, counts = np.unique(grid, return_counts=True)
        bg = int(values[np.argmax(counts)])
    return bg


def _same_color_degree(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    color = int(grid[r, c])
    degree = 0
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
            degree += 1
    return degree


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor foreground cells by their same-object 4-neighbor degree.

    Each non-background pixel is recolored according to how many 4-neighbors
    share its color, which is equivalent to the local connectivity within a
    monochrome 4-connected object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == bg:
                continue
            degree = _same_color_degree(g, r, c)
            if degree == 4:
                out[r, c] = CORE_COLOR
            elif degree == 3:
                out[r, c] = THREE_SIDE_COLOR
            elif degree == 2:
                out[r, c] = EDGE_COLOR
            elif degree == 1:
                out[r, c] = LEAF_COLOR
    return out


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(int(r - min_r), int(c - min_c)) for r, c in coords}


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _shape_variants(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    mask = _mask_from_coords(_normalize(coords))
    variants: list[set[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            pts = {
                (int(r), int(c))
                for r, c in np.argwhere(variant)
            }
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                variants.append(pts)
    return variants


def _rectangle_shape() -> set[tuple[int, int]]:
    h = random.randint(3, 8)
    w = random.randint(3, 8)
    return {(r, c) for r in range(h) for c in range(w)}


def _l_shape() -> set[tuple[int, int]]:
    thickness = random.randint(3, 4)
    h = random.randint(thickness + 2, thickness + 6)
    w = random.randint(thickness + 2, thickness + 6)
    coords: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(thickness):
            coords.add((r, c))
    for r in range(thickness):
        for c in range(w):
            coords.add((r, c))
    return coords


def _plus_shape() -> set[tuple[int, int]]:
    thickness = random.randint(3, 5)
    arm_r = random.randint(1, 3)
    arm_c = random.randint(1, 3)
    h = thickness + 2 * arm_r
    w = thickness + 2 * arm_c
    coords: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(thickness):
            coords.add((r, arm_c + c))
    for r in range(thickness):
        for c in range(w):
            coords.add((arm_r + r, c))
    return coords


def _t_shape() -> set[tuple[int, int]]:
    bar_len = random.choice([3, 5, 7])
    stem_len = random.randint(2, 5)
    mid = bar_len // 2
    coords = {(0, c) for c in range(bar_len)}
    for r in range(1, stem_len + 1):
        coords.add((r, mid))
    return coords


def _random_walk_path(min_len: int = 4, max_len: int = 9, max_span: int = 8) -> set[tuple[int, int]] | None:
    for _ in range(WALK_ATTEMPTS):
        target = random.randint(min_len, max_len)
        path = [(0, 0)]
        cells = {(0, 0)}
        min_r = max_r = 0
        min_c = max_c = 0

        while len(path) < target:
            r, c = path[-1]
            directions = DIR4[:]
            random.shuffle(directions)
            candidates: list[tuple[int, int, int, int, int, int]] = []
            for dr, dc in directions:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) in cells:
                    continue
                new_min_r = min(min_r, nr)
                new_max_r = max(max_r, nr)
                new_min_c = min(min_c, nc)
                new_max_c = max(max_c, nc)
                if new_max_r - new_min_r + 1 > max_span or new_max_c - new_min_c + 1 > max_span:
                    continue
                adjacent = sum((nr + adr, nc + adc) in cells for adr, adc in DIR4)
                if adjacent != 1:
                    continue
                candidates.append((nr, nc, new_min_r, new_max_r, new_min_c, new_max_c))
            if not candidates:
                break
            nr, nc, min_r, max_r, min_c, max_c = random.choice(candidates)
            path.append((nr, nc))
            cells.add((nr, nc))

        if len(cells) >= min_len:
            return _normalize(cells)
    return None


def _thick_shape() -> set[tuple[int, int]]:
    maker = random.choice([_rectangle_shape, _l_shape, _plus_shape])
    return _normalize(maker())


def _thin_shape() -> set[tuple[int, int]] | None:
    if random.random() < 0.5:
        return _normalize(_t_shape())
    return _random_walk_path()


def _extra_shape() -> set[tuple[int, int]] | None:
    roll = random.random()
    if roll < 0.4:
        return _thick_shape()
    if roll < 0.8:
        return _thin_shape()
    return {(0, 0)}


def _shape_dims(coords: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _can_place(
    placed: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in placed:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_shape(
    shape: set[tuple[int, int]],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    variants = _shape_variants(shape)
    random.shuffle(variants)
    for _ in range(PLACEMENT_TRIES):
        variant = random.choice(variants)
        shape_h, shape_w = _shape_dims(variant)
        if shape_h > h - 2 or shape_w > w - 2:
            continue
        top = random.randint(1, h - shape_h - 1)
        left = random.randint(1, w - shape_w - 1)
        placed = {(int(top + r), int(left + c)) for r, c in variant}
        if _can_place(placed, occupied, h, w):
            return placed
    return None


def _degrees_present(grid: np.ndarray, bg: int) -> set[int]:
    seen: set[int] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == bg:
                continue
            seen.add(_same_color_degree(grid, r, c))
    return seen


def generate_input() -> np.ndarray:
    """
    Generate separated monochrome foreground objects with varied local degrees.

    Every sample includes:
    - one thick object that contributes degree 2/3/4 cells,
    - one thin object that contributes degree 1 cells,
    - one isolated singleton for degree 0,
    plus optional extra objects for variety.
    """
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        bg = random.choice(SAFE_INPUT_COLORS)
        fg_colors = [color for color in SAFE_INPUT_COLORS if color != bg]

        shapes: list[set[tuple[int, int]]] = [_thick_shape()]

        thin = _thin_shape()
        if thin is None:
            continue
        shapes.append(thin)
        shapes.append({(0, 0)})

        max_extra = 2 if min(h, w) >= 16 else 1
        for _extra in range(random.randint(0, max_extra)):
            extra = _extra_shape()
            if extra is None:
                break
            shapes.append(_normalize(extra))

        if len(shapes) > len(fg_colors):
            continue

        grid = np.full((h, w), bg, dtype=int)
        occupied: set[tuple[int, int]] = set()
        placed_objects: list[set[tuple[int, int]]] = []

        ordered_shapes = sorted(shapes, key=len, reverse=True)
        success = True
        for shape in ordered_shapes:
            placed = _place_shape(shape, h, w, occupied)
            if placed is None:
                success = False
                break
            occupied.update(placed)
            placed_objects.append(placed)
        if not success:
            continue

        random.shuffle(fg_colors)
        for color, cells in zip(fg_colors, placed_objects):
            for r, c in cells:
                grid[r, c] = int(color)

        if int((grid == bg).sum()) * 2 <= int(grid.size):
            continue
        if _degrees_present(grid, bg) != {0, 1, 2, 3, 4}:
            continue
        if len(np.unique(grid)) < 4:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC heatmap input")
