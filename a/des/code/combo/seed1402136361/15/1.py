from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

# Generator bounds.
GRID_HEIGHT_RANGE = (16, 26)
GRID_WIDTH_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (3, 5)
TRUNK_LENGTH_RANGE = (4, 8)
BRANCH_COUNT_MAX = 3
BRANCH_LENGTH_CAP = 4
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            coords: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append({"color": color, "coords": coords})

    return comps


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox_mask(cells: set[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    arr = np.array(sorted(cells), dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    r1 = int(arr[:, 0].max())
    c1 = int(arr[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask, r0, c0


def _segment_summary(cells: set[tuple[int, int]]) -> dict[str, object]:
    mask, r0, c0 = _bbox_mask(cells)
    h, w = mask.shape
    segments: list[list[tuple[int, int]]] = []
    max_h = 0
    max_v = 0

    for r in range(h):
        c = 0
        while c < w:
            if not mask[r, c]:
                c += 1
                continue
            start = c
            while c < w and mask[r, c]:
                c += 1
            seg = [(r0 + r, c0 + cc) for cc in range(start, c)]
            max_h = max(max_h, len(seg))
            segments.append(seg)

    for c in range(w):
        r = 0
        while r < h:
            if not mask[r, c]:
                r += 1
                continue
            start = r
            while r < h and mask[r, c]:
                r += 1
            seg = [(r0 + rr, c0 + c) for rr in range(start, r)]
            max_v = max(max_v, len(seg))
            segments.append(seg)

    max_len = max(max_h, max_v)
    keep: set[tuple[int, int]] = set()
    for seg in segments:
        if len(seg) == max_len:
            keep.update(seg)

    return {"keep": keep, "max_h": max_h, "max_v": max_v, "max_len": max_len}


def _mode_of_cells(cells: set[tuple[int, int]]) -> str:
    summary = _segment_summary(cells)
    max_h = int(summary["max_h"])
    max_v = int(summary["max_v"])
    if max_h > max_v:
        return "horizontal"
    if max_v > max_h:
        return "vertical"
    return "tie"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the longest maximal horizontal or vertical line segments in each object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    out = np.full(g.shape, bg, dtype=int)

    for comp in _components(g, bg):
        color = int(comp["color"])
        cells = {tuple(map(int, coord)) for coord in comp["coords"]}
        keep = _segment_summary(cells)["keep"]
        for r, c in keep:
            out[r, c] = color

    return out


def _horizontal_candidate() -> set[tuple[int, int]]:
    length = random.randint(*TRUNK_LENGTH_RANGE)
    cells = {(0, c) for c in range(length)}
    branch_count = random.randint(1, min(BRANCH_COUNT_MAX, length))
    anchors = random.sample(range(length), branch_count)
    for anchor in anchors:
        direction = random.choice((-1, 1))
        branch_len = random.randint(1, min(BRANCH_LENGTH_CAP, length - 1))
        for step in range(1, branch_len + 1):
            cells.add((direction * step, anchor))
    return _normalize_cells(cells)


def _vertical_candidate() -> set[tuple[int, int]]:
    length = random.randint(*TRUNK_LENGTH_RANGE)
    cells = {(r, 0) for r in range(length)}
    branch_count = random.randint(1, min(BRANCH_COUNT_MAX, length))
    anchors = random.sample(range(length), branch_count)
    for anchor in anchors:
        direction = random.choice((-1, 1))
        branch_len = random.randint(1, min(BRANCH_LENGTH_CAP, length - 1))
        for step in range(1, branch_len + 1):
            cells.add((anchor, direction * step))
    return _normalize_cells(cells)


def _tie_candidate() -> set[tuple[int, int]]:
    length = random.randint(*TRUNK_LENGTH_RANGE)
    cross_col = random.randrange(length)
    up = random.randint(0, length - 1)
    down = length - 1 - up

    cells = {(0, c) for c in range(length)}
    for r in range(-up, down + 1):
        cells.add((r, cross_col))

    spur_len = random.randint(1, min(BRANCH_LENGTH_CAP, length - 2))
    if random.random() < 0.5:
        anchor_col = random.choice([c for c in range(length) if c != cross_col])
        direction = random.choice((-1, 1))
        for step in range(1, spur_len + 1):
            cells.add((direction * step, anchor_col))
    else:
        anchor_row = random.choice([r for r in range(-up, down + 1) if r != 0])
        direction = random.choice((-1, 1))
        for step in range(1, spur_len + 1):
            cells.add((anchor_row, cross_col + direction * step))

    return _normalize_cells(cells)


def _build_object(mode: str) -> set[tuple[int, int]]:
    builders = {
        "horizontal": _horizontal_candidate,
        "vertical": _vertical_candidate,
        "tie": _tie_candidate,
    }
    for _ in range(SHAPE_ATTEMPTS):
        cells = builders[mode]()
        summary = _segment_summary(cells)
        if _mode_of_cells(cells) != mode:
            continue
        if len(summary["keep"]) >= len(cells):
            continue
        return cells
    raise RuntimeError(f"failed to build a {mode} object")


def _bbox_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return max(rows) + 1, max(cols) + 1


def _can_place(occupied: np.ndarray, cells: set[tuple[int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        rr = top + r
        cc = left + c
        if occupied[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_objects(
    height: int,
    width: int,
    objects: list[dict[str, object]],
    bg: int,
) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    objects = sorted(objects, key=lambda obj: len(obj["cells"]), reverse=True)
    for obj in objects:
        cells = obj["cells"]
        color = int(obj["color"])
        box_h, box_w = _bbox_size(cells)
        positions = [
            (top, left)
            for top in range(height - box_h + 1)
            for left in range(width - box_w + 1)
        ]
        random.shuffle(positions)

        placed = False
        for top, left in positions[:PLACEMENT_ATTEMPTS]:
            if not _can_place(occupied, cells, top, left):
                continue
            for r, c in cells:
                rr = top + r
                cc = left + c
                grid[rr, cc] = color
                occupied[rr, cc] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _mode_list(count: int) -> list[str]:
    base = ["horizontal", "vertical", "tie"]
    if count <= 3:
        random.shuffle(base)
        return base[:count]
    extra = [random.choice(base) for _ in range(count - 3)]
    modes = base + extra
    random.shuffle(modes)
    return modes


def _valid_input(grid: np.ndarray, modes: list[str]) -> bool:
    bg = _background_color(grid)
    comps = _components(grid, bg)
    if len(comps) != len(modes):
        return False

    seen_modes = []
    for comp in comps:
        cells = {tuple(map(int, coord)) for coord in comp["coords"]}
        summary = _segment_summary(cells)
        if len(summary["keep"]) >= len(cells):
            return False
        seen_modes.append(_mode_of_cells(cells))

    if sorted(seen_modes) != sorted(modes):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out != bg) >= np.count_nonzero(grid != bg):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], obj_count)
    modes = _mode_list(obj_count)

    objects = []
    for color, mode in zip(colors, modes):
        objects.append({"color": color, "mode": mode, "cells": _build_object(mode)})

    grid = _place_objects(height, width, objects, bg)
    if grid is None:
        return None
    if not _valid_input(grid, modes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
