import random
from collections import defaultdict

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 400


SHAPES = [
    {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)},
    {(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)},
    {(1, -1), (1, 0), (1, 1), (0, 0), (-1, 0)},
    {(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)},
    {(-1, 1), (0, 1), (1, 1), (0, 0), (0, -1)},
    {(-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)},
    {(-1, 1), (-1, 0), (0, 0), (1, 0), (1, -1)},
    {(-1, -1), (-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)},
    {(-1, 1), (-1, 0), (0, 0), (1, 1), (1, 0), (1, -1)},
    {(-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)},
]


def _components4(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [r0 for r0, _ in cells]
            cs = [c0 for _, c0 in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return comps


def _normalize_shape(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _hv_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    cells = [(r1, c1)]
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1 + step_c, c2 + step_c, step_c):
        cells.append((r1, c))
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1 + step_r, r2 + step_r, step_r):
        if (r, c2) != cells[-1]:
            cells.append((r, c2))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in _components4(g):
        groups[_normalize_shape(comp["cells"])].append(comp)

    paired = []
    singles = []
    for comps in groups.values():
        if len(comps) == 2:
            paired.append(comps)
        else:
            singles.extend(comps)

    for comp in singles:
        for r, c in comp["cells"]:
            out[r, c] = BG

    for comp_a, comp_b in paired:
        center_a = _bbox_center(comp_a["bbox"])
        center_b = _bbox_center(comp_b["bbox"])
        for r, c in _hv_path(center_a, center_b):
            out[r, c] = MAROON
    return out


def _shape_bbox(shape: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return min(rs), min(cs), max(rs), max(cs)


def _place_shape(shape: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in shape}


def _inside(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _expanded_bbox(cells: set[tuple[int, int]], pad: int = 1) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs) - pad, min(cs) - pad, max(rs) + pad, max(cs) + pad


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _valid_new_object(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    placed_cells: set[tuple[int, int]],
    placed_boxes: list[tuple[int, int, int, int]],
) -> bool:
    if not _inside(cells, h, w):
        return False
    if cells & placed_cells:
        return False
    box = _expanded_bbox(cells, pad=1)
    if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
        return False
    return all(not _bbox_overlap(box, other) for other in placed_boxes)


def _path_clear(
    path: list[tuple[int, int]],
    own_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_paths: list[set[tuple[int, int]]],
) -> bool:
    path_set = set(path)
    if path_set & (occupied - own_cells):
        return False
    for other in reserved_paths:
        if path_set & other:
            return False
    return True


def _centers_for_branch(
    branch: str,
    shape: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = _shape_bbox(shape)
    min_r = 1 - r0
    max_r = h - 2 - r1
    min_c = 1 - c0
    max_c = w - 2 - c1
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(120):
        if branch == "h":
            row = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c - 5)
            right = random.randint(left + 5, max_c)
            return (row, left), (row, right)
        if branch == "v":
            col = random.randint(min_c, max_c)
            top = random.randint(min_r, max_r - 5)
            bottom = random.randint(top + 5, max_r)
            return (top, col), (bottom, col)
        row1 = random.randint(min_r, max_r - 4)
        row2 = random.randint(row1 + 4, max_r)
        col1 = random.randint(min_c, max_c - 4)
        col2 = random.randint(col1 + 4, max_c)
        if random.random() < 0.5:
            return (row1, col1), (row2, col2)
        return (row1, col2), (row2, col1)
    return None


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _choose_counts() -> tuple[int, int]:
    num_pairs = random.choice([1, 2, 2, 3])
    num_singles = random.choice([0, 1, 1, 2])
    if num_pairs + num_singles > 5:
        num_singles = 5 - num_pairs
    return num_pairs, num_singles


def _verify_instance(grid: np.ndarray) -> bool:
    comps = _components4(grid)
    if not comps:
        return False
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in comps:
        bbox = comp["bbox"]
        if (bbox[0] + bbox[2]) % 2 != 0 or (bbox[1] + bbox[3]) % 2 != 0:
            return False
        groups[_normalize_shape(comp["cells"])].append(comp)
    if not all(len(v) in (1, 2) for v in groups.values()):
        return False
    if not any(len(v) == 2 for v in groups.values()):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.full((h, w), BG, dtype=int)

    num_pairs, num_singles = _choose_counts()
    total_shapes = num_pairs + num_singles
    if total_shapes > len(SHAPES):
        return None

    shapes = random.sample(SHAPES, total_shapes)
    colors = random.sample(PALETTE, num_pairs * 2 + num_singles)
    branches = [random.choice(["h", "v", "l"]) for _ in range(num_pairs)]

    placed_cells: set[tuple[int, int]] = set()
    placed_boxes: list[tuple[int, int, int, int]] = []
    reserved_paths: list[set[tuple[int, int]]] = []
    color_idx = 0

    for pair_idx in range(num_pairs):
        shape = shapes[pair_idx]
        branch = branches[pair_idx]
        success = False
        for _ in range(180):
            centers = _centers_for_branch(branch, shape, h, w)
            if centers is None:
                return None
            center_a, center_b = centers
            cells_a = _place_shape(shape, center_a)
            cells_b = _place_shape(shape, center_b)
            if not _valid_new_object(cells_a, h, w, placed_cells, placed_boxes):
                continue
            if not _valid_new_object(cells_b, h, w, placed_cells | cells_a, placed_boxes + [_expanded_bbox(cells_a)]):
                continue
            path = _hv_path(center_a, center_b)
            own = cells_a | cells_b
            if not _path_clear(path, own, placed_cells | own, reserved_paths):
                continue

            _paint(grid, cells_a, colors[color_idx])
            _paint(grid, cells_b, colors[color_idx + 1])
            color_idx += 2
            placed_cells |= own
            placed_boxes.append(_expanded_bbox(cells_a))
            placed_boxes.append(_expanded_bbox(cells_b))
            reserved_paths.append(set(path) - own)
            success = True
            break
        if not success:
            return None

    for single_idx in range(num_singles):
        shape = shapes[num_pairs + single_idx]
        success = False
        r0, c0, r1, c1 = _shape_bbox(shape)
        min_r = 1 - r0
        max_r = h - 2 - r1
        min_c = 1 - c0
        max_c = w - 2 - c1
        for _ in range(160):
            center = (random.randint(min_r, max_r), random.randint(min_c, max_c))
            cells = _place_shape(shape, center)
            if not _valid_new_object(cells, h, w, placed_cells, placed_boxes):
                continue
            if any(cells & path for path in reserved_paths):
                continue
            _paint(grid, cells, colors[color_idx])
            color_idx += 1
            placed_cells |= cells
            placed_boxes.append(_expanded_bbox(cells))
            success = True
            break
        if not success:
            return None

    if not _verify_instance(grid):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
