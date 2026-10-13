from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_legend_colors(last_row: np.ndarray, bg: int) -> list[int]:
    row = np.asarray(last_row, dtype=int)
    best: tuple[int, int] | None = None
    start: int | None = None

    for idx, value in enumerate(row):
        if int(value) != bg and start is None:
            start = idx
        elif int(value) == bg and start is not None:
            run = (start, idx)
            if best is None or (run[1] - run[0]) > (best[1] - best[0]):
                best = run
            start = None
    if start is not None:
        run = (start, len(row))
        if best is None or (run[1] - run[0]) > (best[1] - best[0]):
            best = run

    assert best is not None, "expected a non-background legend strip on the last row"
    run_mask = np.zeros(len(row), dtype=bool)
    run_mask[best[0] : best[1]] = True
    assert not np.any((row != bg) & ~run_mask), "expected a single contiguous legend strip"

    colors = [int(v) for v in row[best[0] : best[1]]]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors


def _bbox_area(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty color group"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(maxs[0] - mins[0] + 1) * int(maxs[1] - mins[1] + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color pixels, rank each color by its bbox area,
    and recolor groups using the bottom legend strip from left to right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    scene = g[:-1]
    bg = _most_common_color(scene)
    legend_colors = _find_legend_colors(g[-1], bg)

    group_colors = [int(c) for c in np.unique(scene) if int(c) != bg]
    assert len(group_colors) == len(legend_colors), "legend length must match color-group count"

    areas = {color: _bbox_area(scene == color) for color in group_colors}
    assert len(set(areas.values())) == len(areas), "bbox areas must be unique"

    ranked = sorted(group_colors, key=lambda color: areas[color])
    out = g.copy()
    top = out[:-1, :]
    for old_color, new_color in zip(ranked, legend_colors):
        top[scene == old_color] = int(new_color)
    return out


COLOR_IDS = tuple(range(10))
GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (14, 20)
GROUP_COUNT_RANGE = (3, 4)
BOX_HEIGHT_RANGE = (2, 6)
BOX_WIDTH_RANGE = (2, 6)
BOX_PLACEMENT_RETRIES = 150
GENERATION_RETRIES = 400


def _sample_colors(group_count: int) -> tuple[int, list[int], list[int]]:
    bg = random.choice(COLOR_IDS)
    non_bg = [color for color in COLOR_IDS if color != bg]
    chosen = random.sample(non_bg, 2 * group_count)
    input_colors = chosen[:group_count]
    legend_colors = chosen[group_count:]
    random.shuffle(input_colors)
    random.shuffle(legend_colors)
    return int(bg), [int(c) for c in input_colors], [int(c) for c in legend_colors]


def _choose_boxes(top_height: int, width: int, group_count: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((top_height, width), dtype=bool)
    used_areas: set[int] = set()
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(group_count):
        placed = False
        for _ in range(BOX_PLACEMENT_RETRIES):
            h = random.randint(BOX_HEIGHT_RANGE[0], min(BOX_HEIGHT_RANGE[1], top_height))
            w = random.randint(BOX_WIDTH_RANGE[0], min(BOX_WIDTH_RANGE[1], width))
            area = h * w
            if area in used_areas:
                continue

            r0 = random.randint(0, top_height - h)
            c0 = random.randint(0, width - w)
            rr0 = max(0, r0 - 1)
            rr1 = min(top_height, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            occupied[r0 : r0 + h, c0 : c0 + w] = True
            used_areas.add(area)
            boxes.append((r0, c0, h, w))
            placed = True
            break
        if not placed:
            return None

    return boxes


def _add_rect(cells: set[tuple[int, int]], r0: int, c0: int, h: int, w: int) -> None:
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            cells.add((rr, cc))


def _build_group_cells(height: int, width: int) -> list[tuple[int, int]]:
    mode = random.choice(("dots", "chips", "mixed"))
    cells: set[tuple[int, int]] = {
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    }

    if mode in ("chips", "mixed"):
        patch_count = random.randint(1, 4)
        for _ in range(patch_count):
            patch_h = random.randint(1, min(2, height))
            patch_w = random.randint(1, min(2, width))
            edge = random.choice(("top", "bottom", "left", "right", "any"))
            if edge == "top":
                r0 = 0
                c0 = random.randint(0, width - patch_w)
            elif edge == "bottom":
                r0 = height - patch_h
                c0 = random.randint(0, width - patch_w)
            elif edge == "left":
                r0 = random.randint(0, height - patch_h)
                c0 = 0
            elif edge == "right":
                r0 = random.randint(0, height - patch_h)
                c0 = width - patch_w
            else:
                r0 = random.randint(0, height - patch_h)
                c0 = random.randint(0, width - patch_w)
            _add_rect(cells, r0, c0, patch_h, patch_w)

    extra_points = random.randint(1, max(2, min(6, height * width // 2)))
    for _ in range(extra_points):
        r = random.randrange(height)
        c = random.randrange(width)
        cells.add((r, c))
        if mode != "dots" and random.random() < 0.4:
            nr = min(height - 1, max(0, r + random.choice((-1, 0, 1))))
            nc = min(width - 1, max(0, c + random.choice((-1, 0, 1))))
            cells.add((nr, nc))

    return sorted(cells)


def _expected_output(
    grid: np.ndarray,
    groups: list[dict[str, object]],
    legend_start: int,
    legend_colors: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[-1, legend_start : legend_start + len(legend_colors)] = np.array(legend_colors, dtype=int)

    ranked = sorted(groups, key=lambda group: int(group["area"]))
    for group, new_color in zip(ranked, legend_colors):
        for rr, cc in group["cells"]:
            out[int(rr), int(cc)] = int(new_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(max(GRID_WIDTH_RANGE[0], group_count + 2), GRID_WIDTH_RANGE[1])
    top_height = height - 2
    if top_height < BOX_HEIGHT_RANGE[0]:
        return None

    boxes = _choose_boxes(top_height, width, group_count)
    if boxes is None:
        return None

    bg, input_colors, legend_colors = _sample_colors(group_count)
    grid = np.full((height, width), bg, dtype=int)
    groups: list[dict[str, object]] = []

    for color, (r0, c0, box_h, box_w) in zip(input_colors, boxes):
        local_cells = _build_group_cells(box_h, box_w)
        global_cells: list[tuple[int, int]] = []
        for dr, dc in local_cells:
            rr = r0 + dr
            cc = c0 + dc
            grid[rr, cc] = int(color)
            global_cells.append((rr, cc))
        groups.append(
            {
                "color": int(color),
                "area": int(box_h * box_w),
                "cells": global_cells,
            }
        )

    legend_start = random.randint(0, width - group_count)
    grid[-1, legend_start : legend_start + group_count] = np.array(legend_colors, dtype=int)

    if _most_common_color(grid[:-1]) != bg or _most_common_color(grid) != bg:
        return None

    expected = _expected_output(grid, groups, legend_start, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
