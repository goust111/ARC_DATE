from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
FG_COLORS = [1, 2, 3, 4, 6, 8, 9]
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (5, 7)
HOLE_COUNT_RANGE = (3, 5)
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACE_RETRIES = 200

HOLE_PATTERNS = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_cells(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _interior_holes(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    for pts in _components(~mask):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    return int(Counter(map(int, border.tolist())).most_common(1)[0][0])


def _object_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            mask = _mask_from_cells(cells)
            objects.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "holes": _interior_holes(mask),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _object_components(g, bg)
    hole_objects = [obj for obj in objects if obj["holes"]]

    if len(hole_objects) <= 1:
        return g.copy()

    hole_objects.sort(key=lambda obj: obj["area"], reverse=True)
    colors = [int(obj["color"]) for obj in hole_objects]

    out = g.copy()
    for i, obj in enumerate(hole_objects):
        new_color = colors[(i + 1) % len(colors)]
        for r, c in obj["cells"]:
            out[r, c] = new_color
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _add_protrusions(cells: set[tuple[int, int]], base_h: int, base_w: int) -> None:
    for _ in range(random.randint(0, 4)):
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if base_w < 3:
                continue
            length = random.randint(1, base_w - 2)
            start = random.randint(1, base_w - length - 1)
            row = -1 if side == "top" else base_h
            new_cells = {(row, c) for c in range(start, start + length)}
        else:
            if base_h < 3:
                continue
            length = random.randint(1, base_h - 2)
            start = random.randint(1, base_h - length - 1)
            col = -1 if side == "left" else base_w
            new_cells = {(r, col) for r in range(start, start + length)}
        if not (cells & new_cells):
            cells |= new_cells


def _carve_open_notches(cells: set[tuple[int, int]], base_h: int, base_w: int) -> None:
    for _ in range(random.randint(0, 2)):
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if base_w < 4:
                continue
            length = random.randint(1, min(2, base_w - 2))
            start = random.randint(1, base_w - length - 1)
            row = 0 if side == "top" else base_h - 1
            for c in range(start, start + length):
                cells.discard((row, c))
        else:
            if base_h < 4:
                continue
            length = random.randint(1, min(2, base_h - 2))
            start = random.randint(1, base_h - length - 1)
            col = 0 if side == "left" else base_w - 1
            for r in range(start, start + length):
                cells.discard((r, col))


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    max_r = max(r for r, _c in norm)
    max_c = max(c for _r, c in norm)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _shape_ok(mask: np.ndarray, require_hole: bool) -> bool:
    if mask.sum() < 6:
        return False
    h, w = mask.shape
    if h > 10 or w > 10:
        return False
    if max(h, w) > 3 * max(1, min(h, w)):
        return False
    if len(_components(mask)) != 1:
        return False
    holes = _interior_holes(mask)
    if require_hole:
        return len(holes) == 1 and 1 <= len(holes[0]) <= 3
    return len(holes) == 0


def _make_hole_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        pattern = random.choice(HOLE_PATTERNS)
        pat_h = max(r for r, _c in pattern) + 1
        pat_w = max(c for _r, c in pattern) + 1
        base_h = random.randint(max(4, pat_h + 2), 8)
        base_w = random.randint(max(4, pat_w + 2), 8)

        hole_r = random.randint(1, base_h - pat_h - 1)
        hole_c = random.randint(1, base_w - pat_w - 1)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}
        for dr, dc in pattern:
            cells.discard((hole_r + dr, hole_c + dc))

        _add_protrusions(cells, base_h, base_w)
        mask = _cells_to_mask(cells)
        if _shape_ok(mask, require_hole=True):
            return mask
    return None


def _make_solid_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        base_h = random.randint(3, 8)
        base_w = random.randint(3, 8)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}
        _add_protrusions(cells, base_h, base_w)
        _carve_open_notches(cells, base_h, base_w)
        mask = _cells_to_mask(cells)
        if _shape_ok(mask, require_hole=False):
            return mask
    return None


def _place_objects(objects: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for obj in sorted(objects, key=lambda item: int(item["mask"].sum()), reverse=True):
        mask = obj["mask"]
        mh, mw = mask.shape
        positions: list[tuple[int, int]] = []
        for top in range(1, h - mh):
            for left in range(1, w - mw):
                r0 = top - 1
                r1 = top + mh + 1
                c0 = left - 1
                c1 = left + mw + 1
                if blocked[r0:r1, c0:c1].any():
                    continue
                positions.append((top, left))
        if not positions:
            return None
        top, left = random.choice(positions)
        grid[top : top + mh, left : left + mw][mask] = int(obj["color"])
        blocked[top - 1 : top + mh + 1, left - 1 : left + mw + 1] = True
    return grid


def _touching_other_objects(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            here = int(grid[r, c])
            if here == BG:
                continue
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    other = int(grid[nr, nc])
                    if other != BG and other != here:
                        return True
    return False


def _valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if not np.all(grid[0, :] == BG) or not np.all(grid[-1, :] == BG):
        return False
    if not np.all(grid[:, 0] == BG) or not np.all(grid[:, -1] == BG):
        return False
    if _touching_other_objects(grid):
        return False

    objects = _object_components(grid, BG)
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False
    colors = [int(obj["color"]) for obj in objects]
    if len(set(colors)) != len(colors):
        return False
    if any(color not in FG_COLORS for color in colors):
        return False

    hole_objects = [obj for obj in objects if obj["holes"]]
    solid_objects = [obj for obj in objects if not obj["holes"]]
    if not (HOLE_COUNT_RANGE[0] <= len(hole_objects) <= HOLE_COUNT_RANGE[1]):
        return False
    if not solid_objects:
        return False
    if any(len(obj["holes"]) != 1 or not (1 <= len(obj["holes"][0]) <= 3) for obj in hole_objects):
        return False
    if any(obj["holes"] for obj in solid_objects):
        return False

    areas = sorted((int(obj["area"]) for obj in hole_objects), reverse=True)
    if any(areas[i] - areas[i + 1] < 3 for i in range(len(areas) - 1)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    max_holes = min(HOLE_COUNT_RANGE[1], object_count - 1)
    hole_count = random.randint(HOLE_COUNT_RANGE[0], max_holes)

    colors = random.sample(FG_COLORS, object_count)
    hole_shapes: list[np.ndarray] = []
    hole_areas: list[int] = []
    for _ in range(hole_count):
        mask = None
        for _ in range(SHAPE_RETRIES):
            candidate = _make_hole_shape()
            if candidate is None:
                continue
            area = int(candidate.sum())
            if all(abs(area - prev) >= 3 for prev in hole_areas):
                mask = candidate
                hole_areas.append(area)
                break
        if mask is None:
            return None
        hole_shapes.append(mask)

    solid_shapes: list[np.ndarray] = []
    for _ in range(object_count - hole_count):
        mask = _make_solid_shape()
        if mask is None:
            return None
        solid_shapes.append(mask)

    object_defs = []
    for color, mask in zip(colors[:hole_count], hole_shapes, strict=False):
        object_defs.append({"color": color, "mask": mask})
    for color, mask in zip(colors[hole_count:], solid_shapes, strict=False):
        object_defs.append({"color": color, "mask": mask})
    random.shuffle(object_defs)

    for _ in range(PLACE_RETRIES):
        grid = _place_objects(object_defs, h, w)
        if grid is not None and _valid_input(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
