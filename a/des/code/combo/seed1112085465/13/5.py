from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 14
GRID_MAX = 28
PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 4
BOX_MIN = 3
BOX_MAX = 7
SHAPE_ATTEMPTS = 200
PAIR_ATTEMPTS = 200
SCENE_ATTEMPTS = 400
PLACEMENT_TRIES = 300
RECT_GAP = 1

Cell = tuple[int, int]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells: set[Cell]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[Cell]) -> tuple[set[Cell], int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    shifted = {(r - r0, c - c0) for r, c in cells}
    return shifted, r1 - r0 + 1, c1 - c0 + 1


def _is_connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _preserves_full_bbox(cells: set[Cell], height: int, width: int) -> bool:
    return (
        any(r == 0 for r, _ in cells)
        and any(c == 0 for _, c in cells)
        and any(r == height - 1 for r, _ in cells)
        and any(c == width - 1 for _, c in cells)
    )


def _component_info(coords: list[Cell]) -> dict:
    cells = set(coords)
    r0, c0, r1, c1 = _bbox(cells)
    local = {(r - r0, c - c0) for r, c in cells}
    return {
        "coords": coords,
        "cells": cells,
        "local": local,
        "area": len(coords),
        "r0": r0,
        "c0": c0,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
    }


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Cell] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(_component_info(coords))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the pixels missing from each incomplete copy by comparing it with its same-colored master.

    Input structure:
    - Background is black.
    - Every non-background color appears in exactly two 4-connected monochromatic objects.
    - For each color pair, the larger object is the master and the smaller one is a copy.
    - After aligning the two objects by bbox top-left, the copy's pixels form a subset of the master's pixels.

    Rule:
    - Leave each master unchanged.
    - For each copy, align its bbox top-left with the master's bbox top-left.
    - Wherever the master has a pixel and the copy does not, paint yellow at the corresponding copy position.
    - Keep the copy's original colored pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == YELLOW), "yellow is reserved for output markers"

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors, "expected at least one foreground color"

    out = g.copy()
    for color in colors:
        components = _color_components(g, color)
        assert len(components) == 2, f"color {color} must appear in exactly two objects"

        components.sort(key=lambda comp: comp["area"])
        copy_comp, master_comp = components
        assert copy_comp["area"] < master_comp["area"], "copy must be smaller than master"

        copy_local = copy_comp["local"]
        master_local = master_comp["local"]
        assert copy_local <= master_local, "copy shape must be a subset of the aligned master"

        for dr, dc in master_local - copy_local:
            rr = int(copy_comp["r0"] + dr)
            cc = int(copy_comp["c0"] + dc)
            assert _in_bounds(g.shape, rr, cc), "missing-position marker must stay in bounds"
            assert int(out[rr, cc]) == BG, "missing-position marker must land on background"
            out[rr, cc] = YELLOW

    return out


def _grow_connected_shape(height: int, width: int, target_area: int) -> set[Cell] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < height and 0 <= nc < width
    }

    while len(cells) < target_area:
        if not frontier:
            return None
        choices = list(frontier)
        weights = []
        for r, c in choices:
            degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            weights.append(degree * degree + 1)
        chosen = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(chosen)
        cells.add(chosen)
        for nr, nc in _neighbors4(*chosen):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _make_master_shape() -> tuple[set[Cell], int, int] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(BOX_MIN, BOX_MAX)
        box_w = random.randint(BOX_MIN, BOX_MAX)
        area = box_h * box_w
        min_area = max(5, (area + 3) // 4)
        max_area = min(area, max(min_area, int(round(area * 0.72))))
        target_area = random.randint(min_area, max_area)
        grown = _grow_connected_shape(box_h, box_w, target_area)
        if grown is None:
            continue
        cells, height, width = _normalize(grown)
        if height < 2 or width < 2 or len(cells) < 5:
            continue
        return cells, height, width
    return None


def _make_same_bbox_copy(master: set[Cell], height: int, width: int) -> set[Cell] | None:
    copy = set(master)
    limit = max(1, min(len(master) // 3, len(master) - 2))
    target_removed = random.randint(1, limit)
    removed = 0

    while removed < target_removed:
        candidates = []
        for cell in copy:
            trial = copy - {cell}
            if len(trial) < 2:
                continue
            if not _preserves_full_bbox(trial, height, width):
                continue
            if not _is_connected(trial):
                continue
            candidates.append(cell)
        if not candidates:
            break
        weights = [
            3 if 0 < r < height - 1 and 0 < c < width - 1 else 1
            for r, c in candidates
        ]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        copy.remove(chosen)
        removed += 1

    if removed == 0 or copy == master:
        return None
    return copy


def _make_shrunk_copy(master: set[Cell], height: int, width: int) -> set[Cell] | None:
    candidates: list[set[Cell]] = []
    min_size = max(2, len(master) // 3)
    for max_r in range(height):
        for max_c in range(width):
            if max_r == height - 1 and max_c == width - 1:
                continue
            trial = {(r, c) for r, c in master if r <= max_r and c <= max_c}
            if len(trial) < min_size or len(trial) >= len(master):
                continue
            if not any(r == 0 for r, _ in trial):
                continue
            if not any(c == 0 for _, c in trial):
                continue
            if not _is_connected(trial):
                continue
            candidates.append(trial)
    if not candidates:
        return None
    return set(random.choice(candidates))


def _make_pair(use_same_bbox: bool) -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        made = _make_master_shape()
        if made is None:
            continue
        master, height, width = made
        if use_same_bbox:
            copy = _make_same_bbox_copy(master, height, width)
        else:
            copy = _make_shrunk_copy(master, height, width)
        if copy is None:
            continue
        return {
            "master": master,
            "copy": copy,
            "h": height,
            "w": width,
            "same_bbox": use_same_bbox,
        }
    return None


def _branch_schedule(count: int) -> list[bool]:
    if count == 1:
        return [bool(random.getrandbits(1))]
    extra_same = random.randint(0, 1) if count % 2 else 0
    same_count = count // 2 + extra_same
    flags = [True] * same_count + [False] * (count - same_count)
    random.shuffle(flags)
    return flags


def _rect_clear(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    for rr, cc, hh, ww in rects:
        separated = (
            r0 + h - 1 + RECT_GAP < rr
            or rr + hh - 1 + RECT_GAP < r0
            or c0 + w - 1 + RECT_GAP < cc
            or cc + ww - 1 + RECT_GAP < c0
        )
        if not separated:
            return False
    return True


def _place_rect(
    rects: list[tuple[int, int, int, int]], height: int, width: int, h: int, w: int
) -> tuple[int, int] | None:
    if h > height or w > width:
        return None
    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        if _rect_clear(rects, r0, c0, h, w):
            return r0, c0
    return None


def _paint_shape(grid: np.ndarray, cells: set[Cell], anchor: tuple[int, int], color: int) -> None:
    r0, c0 = anchor
    for dr, dc in cells:
        grid[r0 + dr, c0 + dc] = color


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["copy_anchor"]
        for dr, dc in spec["master"] - spec["copy"]:
            out[r0 + dr, c0 + dc] = YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    min_side = GRID_MIN + 2 * pair_count
    height = random.randint(min_side, GRID_MAX)
    width = random.randint(min_side, GRID_MAX)
    colors = random.sample(INPUT_COLORS, pair_count)
    branches = _branch_schedule(pair_count)

    specs: list[dict] = []
    for color, use_same_bbox in zip(colors, branches):
        spec = _make_pair(use_same_bbox)
        if spec is None:
            return None
        spec["color"] = color
        specs.append(spec)

    specs.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)

    rects: list[tuple[int, int, int, int]] = []
    for spec in specs:
        master_anchor = _place_rect(rects, height, width, spec["h"], spec["w"])
        if master_anchor is None:
            return None
        rects.append((master_anchor[0], master_anchor[1], spec["h"], spec["w"]))

        copy_anchor = _place_rect(rects, height, width, spec["h"], spec["w"])
        if copy_anchor is None:
            return None
        rects.append((copy_anchor[0], copy_anchor[1], spec["h"], spec["w"]))

        spec["master_anchor"] = master_anchor
        spec["copy_anchor"] = copy_anchor

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _paint_shape(grid, spec["master"], spec["master_anchor"], spec["color"])
        _paint_shape(grid, spec["copy"], spec["copy_anchor"], spec["color"])

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
