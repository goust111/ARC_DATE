import random
from typing import List, Sequence, Tuple

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
TRANSFORMS = ("id", "h", "v", "hv")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _full_line_indices(grid: np.ndarray, axis: int) -> List[int]:
    if axis == 0:
        return [
            i
            for i in range(grid.shape[0])
            if len(set(map(int, grid[i, :]))) == 1 and int(grid[i, 0]) != BG
        ]
    return [
        i
        for i in range(grid.shape[1])
        if len(set(map(int, grid[:, i]))) == 1 and int(grid[0, i]) != BG
    ]


def _group_consecutive(indices: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    for idx in indices:
        if not groups or idx != groups[-1][-1] + 1:
            groups.append([int(idx)])
        else:
            groups[-1].append(int(idx))
    return groups


def _panel_intervals(size: int, separator_groups: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start = 0
    for group in separator_groups:
        end = int(group[0])
        intervals.append((start, end))
        start = int(group[-1]) + 1
    intervals.append((start, size))
    return intervals


def _extract_object(panel: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    coords = np.argwhere(panel != BG)
    assert coords.size > 0, "each panel must contain an object"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max()) + 1
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max()) + 1
    return (r0, r1, c0, c1), panel[r0:r1, c0:c1].copy()


def _apply_transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "id":
        return arr.copy()
    if kind == "h":
        return np.fliplr(arr)
    if kind == "v":
        return np.flipud(arr)
    if kind == "hv":
        return np.flipud(np.fliplr(arr))
    raise ValueError(f"unknown transform: {kind}")


def _target_transform(panel_r: int, panel_c: int) -> str:
    dr = panel_r - 1
    dc = panel_c - 1
    if dr == 0 and dc == 0:
        return "id"
    if dr == 0:
        return "h"
    if dc == 0:
        return "v"
    return "hv"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_groups = _group_consecutive(_full_line_indices(g, axis=0))
    col_groups = _group_consecutive(_full_line_indices(g, axis=1))
    assert len(row_groups) == 2 and len(col_groups) == 2, "expected a 3x3 panel grid"

    row_intervals = _panel_intervals(g.shape[0], row_groups)
    col_intervals = _panel_intervals(g.shape[1], col_groups)
    assert len(row_intervals) == 3 and len(col_intervals) == 3

    center_r0, center_r1 = row_intervals[1]
    center_c0, center_c1 = col_intervals[1]
    center_panel = g[center_r0:center_r1, center_c0:center_c1]
    center_bbox, reference = _extract_object(center_panel)

    out = g.copy()
    for pr, (r0, r1) in enumerate(row_intervals):
        for pc, (c0, c1) in enumerate(col_intervals):
            panel = np.full((r1 - r0, c1 - c0), BG, dtype=int)
            rr0, rr1, cc0, cc1 = center_bbox
            transformed = _apply_transform(reference, _target_transform(pr, pc))
            panel[rr0:rr1, cc0:cc1] = transformed
            out[r0:r1, c0:c1] = panel

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    target_area = random.randint(max(3, (h * w) // 3), max(3, h * w - 1))
    start = random.choice(cells)
    chosen = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(chosen) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _mask_distinct_under_required_transforms(mask: np.ndarray) -> bool:
    variants = [_apply_transform(mask.astype(int), kind) for kind in TRANSFORMS]
    return len({variant.tobytes() for variant in variants}) == 4


def _generate_reference_object(panel_h: int, panel_w: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    for _ in range(200):
        obj_h = random.randint(2, panel_h - 2)
        obj_w = random.randint(2, panel_w - 2)
        mask = _random_connected_mask(obj_h, obj_w)
        if not _connected(mask):
            continue
        if not _mask_distinct_under_required_transforms(mask):
            continue
        color = random.choice(OBJECT_COLORS)
        obj = np.where(mask, color, BG).astype(int)
        top = random.randint(1, panel_h - obj_h - 1)
        left = random.randint(1, panel_w - obj_w - 1)
        return obj, (top, left)
    raise RuntimeError("failed to generate a sufficiently asymmetric object")


def _place_object(panel_h: int, panel_w: int, obj: np.ndarray, top: int, left: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    h, w = obj.shape
    panel[top : top + h, left : left + w] = obj
    return panel


def _build_grid(panel_h: int, panel_w: int, center_obj: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    sep = 1
    total_h = 3 * panel_h + 2 * sep
    total_w = 3 * panel_w + 2 * sep
    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[panel_h, :] = SEPARATOR_COLOR
    grid[2 * panel_h + 1, :] = SEPARATOR_COLOR
    grid[:, panel_w] = SEPARATOR_COLOR
    grid[:, 2 * panel_w + 1] = SEPARATOR_COLOR

    top, left = offset
    for pr in range(3):
        for pc in range(3):
            r0 = pr * (panel_h + sep)
            c0 = pc * (panel_w + sep)
            desired = _apply_transform(center_obj, _target_transform(pr, pc))
            if pr == 1 and pc == 1:
                shown = desired
            else:
                shown = _apply_transform(center_obj, random.choice(TRANSFORMS))
            panel = _place_object(panel_h, panel_w, shown, top, left)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        panel_h = random.randint(5, 9)
        panel_w = random.randint(5, 9)
        obj, offset = _generate_reference_object(panel_h, panel_w)
        grid = _build_grid(panel_h, panel_w, obj, offset)
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
