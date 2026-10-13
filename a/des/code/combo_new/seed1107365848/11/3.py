from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_MIN, GRID_H_MAX = 12, 28
GRID_W_MIN, GRID_W_MAX = 12, 28
REGION_COUNT_MIN, REGION_COUNT_MAX = 1, 4
OPEN_RECT_COUNT_MIN, OPEN_RECT_COUNT_MAX = 0, 2
SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX = 0, 2
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
EDGE_MARGIN = 1
BRANCH_LENGTH_MIN, BRANCH_LENGTH_MAX = 1, 4

Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _set_cell(cells: dict[Coord, int], r: int, c: int, color: int) -> bool:
    prev = cells.get((r, c))
    if prev is not None and prev != color:
        return False
    cells[(r, c)] = int(color)
    return True


def _normalize_cells(cells: dict[Coord, int]) -> tuple[dict[Coord, int], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    shift_r = -min(rows)
    shift_c = -min(cols)
    shifted = {(r + shift_r, c + shift_c): int(color) for (r, c), color in cells.items()}
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1
    return shifted, int(height), int(width)


def _draw_closed_rectangle(
    inner_h: int,
    inner_w: int,
    side_colors: dict[str, int],
) -> dict[Coord, int]:
    cells: dict[Coord, int] = {}
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    for c in range(outer_w):
        cells[(0, c)] = int(side_colors["top"])
        cells[(outer_h - 1, c)] = int(side_colors["bottom"])
    for r in range(1, outer_h - 1):
        cells[(r, 0)] = int(side_colors["left"])
        cells[(r, outer_w - 1)] = int(side_colors["right"])
    return cells


def _add_branch(
    rng: random.Random,
    cells: dict[Coord, int],
    outer_h: int,
    outer_w: int,
    side_colors: dict[str, int],
) -> None:
    side = rng.choice(["top", "bottom", "left", "right"])
    length = rng.randint(BRANCH_LENGTH_MIN, BRANCH_LENGTH_MAX)

    if side == "top":
        anchor_c = rng.randrange(outer_w)
        new_cells = [(-step, anchor_c) for step in range(1, length + 1)]
        color = int(side_colors["top"])
    elif side == "bottom":
        anchor_c = rng.randrange(outer_w)
        new_cells = [(outer_h - 1 + step, anchor_c) for step in range(1, length + 1)]
        color = int(side_colors["bottom"])
    elif side == "left":
        anchor_r = rng.randrange(1, outer_h - 1)
        new_cells = [(anchor_r, -step) for step in range(1, length + 1)]
        color = int(side_colors["left"])
    else:
        anchor_r = rng.randrange(1, outer_h - 1)
        new_cells = [(anchor_r, outer_w - 1 + step) for step in range(1, length + 1)]
        color = int(side_colors["right"])

    if any(coord in cells for coord in new_cells):
        return
    for r, c in new_cells:
        cells[(r, c)] = color


def _make_region_object(rng: random.Random, branch_type: str) -> dict[str, object]:
    while True:
        inner_h = rng.randint(1, 5)
        inner_w = rng.randint(1, 6)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if branch_type != "tie" or outer_h != outer_w:
            break

    if branch_type == "unique":
        minority_color, majority_color = rng.sample(COLOR_POOL, 2)
        minority_side = rng.choice(["top", "bottom", "left", "right"])
        side_colors = {side: int(majority_color) for side in ("top", "bottom", "left", "right")}
        side_colors[minority_side] = int(minority_color)
    else:
        tie_style = rng.choice(["2-2", "1-1-2"])
        shorter_is_horizontal = outer_w < outer_h
        if tie_style == "2-2":
            short_color, long_color = rng.sample(COLOR_POOL, 2)
            if shorter_is_horizontal:
                side_colors = {
                    "top": int(short_color),
                    "bottom": int(short_color),
                    "left": int(long_color),
                    "right": int(long_color),
                }
            else:
                side_colors = {
                    "top": int(long_color),
                    "bottom": int(long_color),
                    "left": int(short_color),
                    "right": int(short_color),
                }
        else:
            fill_color, rival_color, majority_color = rng.sample(COLOR_POOL, 3)
            side_colors = {side: int(majority_color) for side in ("top", "bottom", "left", "right")}
            if shorter_is_horizontal:
                side_colors[rng.choice(["top", "bottom"])] = int(fill_color)
                side_colors[rng.choice(["left", "right"])] = int(rival_color)
            else:
                side_colors[rng.choice(["left", "right"])] = int(fill_color)
                side_colors[rng.choice(["top", "bottom"])] = int(rival_color)

    cells = _draw_closed_rectangle(inner_h, inner_w, side_colors)
    for _ in range(rng.choice([0, 1, 1, 2])):
        _add_branch(rng, cells, outer_h, outer_w, side_colors)
    cells, height, width = _normalize_cells(cells)

    return {
        "cells": cells,
        "height": height,
        "width": width,
        "kind": "region",
        "branch_type": branch_type,
    }


def _make_open_rectangle(rng: random.Random) -> dict[str, object]:
    inner_h = rng.randint(1, 4)
    inner_w = rng.randint(1, 5)
    colors = rng.sample(COLOR_POOL, 4)
    side_colors = {
        "top": int(colors[0]),
        "bottom": int(colors[1]),
        "left": int(colors[2]),
        "right": int(colors[3]),
    }
    cells = _draw_closed_rectangle(inner_h, inner_w, side_colors)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap_side = rng.choice(["top", "bottom", "left", "right"])
    if gap_side == "top":
        del cells[(0, rng.randrange(1, outer_w - 1))]
    elif gap_side == "bottom":
        del cells[(outer_h - 1, rng.randrange(1, outer_w - 1))]
    elif gap_side == "left":
        del cells[(rng.randrange(1, outer_h - 1), 0)]
    else:
        del cells[(rng.randrange(1, outer_h - 1), outer_w - 1)]

    cells, height, width = _normalize_cells(cells)
    return {"cells": cells, "height": height, "width": width, "kind": "open"}


def _make_segment_object(rng: random.Random) -> dict[str, object]:
    color = int(rng.choice(COLOR_POOL))
    orientation = rng.choice(["h", "v"])
    length = rng.randint(2, 6)
    cells: dict[Coord, int] = {}

    if orientation == "h":
        for c in range(length):
            cells[(0, c)] = color
        if rng.random() < 0.6:
            anchor_c = rng.randrange(length)
            direction = rng.choice([-1, 1])
            spur_len = rng.randint(1, 3)
            for step in range(1, spur_len + 1):
                cells[(direction * step, anchor_c)] = color
    else:
        for r in range(length):
            cells[(r, 0)] = color
        if rng.random() < 0.6:
            anchor_r = rng.randrange(length)
            direction = rng.choice([-1, 1])
            spur_len = rng.randint(1, 3)
            for step in range(1, spur_len + 1):
                cells[(anchor_r, direction * step)] = color

    cells, height, width = _normalize_cells(cells)
    return {"cells": cells, "height": height, "width": width, "kind": "segment"}


def _can_place(abs_cells: list[tuple[int, int, int]], occupied: np.ndarray) -> bool:
    height, width = map(int, occupied.shape)
    for r, c, _ in abs_cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(height - 1, r + 1)
        c1 = min(width - 1, c + 1)
        if bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any()):
            return False
    return True


def _place_objects(rng: random.Random, height: int, width: int, objects: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = sorted(objects, key=lambda obj: int(obj["height"]) * int(obj["width"]), reverse=True)
    for obj in order:
        obj_h = int(obj["height"])
        obj_w = int(obj["width"])
        cells = obj["cells"]
        row_min = EDGE_MARGIN
        col_min = EDGE_MARGIN
        row_max = height - obj_h - EDGE_MARGIN
        col_max = width - obj_w - EDGE_MARGIN
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(row_min, row_max)
            left = rng.randint(col_min, col_max)
            abs_cells = [(top + r, left + c, int(color)) for (r, c), color in cells.items()]
            if not _can_place(abs_cells, occupied):
                continue
            for rr, cc, color in abs_cells:
                grid[rr, cc] = color
                occupied[rr, cc] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _region_fill_color(side_specs: list[tuple[str, int, int]]) -> tuple[int, str]:
    counts = Counter(color for _, color, _ in side_specs)
    min_count = min(counts.values())
    candidates = {color for color, count in counts.items() if count == min_count}

    if len(candidates) == 1:
        return next(iter(candidates)), "unique"

    best: tuple[int, int] | None = None
    best_color = -1
    for index, (_, color, length) in enumerate(side_specs):
        if color not in candidates:
            continue
        key = (int(length), int(index))
        if best is None or key < best:
            best = key
            best_color = int(color)
    return best_color, "tie"


def _detect_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    regions: list[dict[str, object]] = []

    for coords in _components(g == BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1:
            continue

        inner_h = r1 - r0 + 1
        inner_w = c1 - c0 + 1
        if coords.shape[0] != inner_h * inner_w:
            continue

        top = r0 - 1
        left = c0 - 1
        bottom = r1 + 1
        right = c1 + 1
        if top < 0 or left < 0 or bottom >= height or right >= width:
            continue

        if bool((g[top, left : right + 1] == BLACK).any()):
            continue
        if bool((g[bottom, left : right + 1] == BLACK).any()):
            continue
        if bool((g[top : bottom + 1, left] == BLACK).any()):
            continue
        if bool((g[top : bottom + 1, right] == BLACK).any()):
            continue

        top_vals = g[top, c0 : c1 + 1]
        bottom_vals = g[bottom, c0 : c1 + 1]
        left_vals = g[r0 : r1 + 1, left]
        right_vals = g[r0 : r1 + 1, right]
        if len(set(map(int, top_vals))) != 1:
            continue
        if len(set(map(int, bottom_vals))) != 1:
            continue
        if len(set(map(int, left_vals))) != 1:
            continue
        if len(set(map(int, right_vals))) != 1:
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        side_specs = [
            ("top", int(top_vals[0]), outer_w),
            ("left", int(left_vals[0]), outer_h),
            ("bottom", int(bottom_vals[0]), outer_w),
            ("right", int(right_vals[0]), outer_h),
        ]
        fill_color, branch_type = _region_fill_color(side_specs)
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "fill_color": int(fill_color),
                "branch_type": branch_type,
            }
        )

    regions.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed rectangular region with the minority color on its boundary.

    Input structure:
    - The background is black.
    - Foreground pixels form one-pixel-thick horizontal and vertical colored line segments.
    - Some segment cycles create fully enclosed rectangular black regions, while other segments are dangling or belong to open pseudo-rectangles.
    - Each valid region is bounded by four complete sides; each side is monochrome, though different sides may use different colors.

    Rule:
    - Find every black connected component that is a solid rectangle and is surrounded on all four sides by a complete one-pixel-thick colored border.
    - For each such region, read the colors used on its four sides and count how many sides each color occupies.
    - Fill the region interior with the color whose side-count is minimal; if several colors tie for the minimum, choose the one that occupies the shortest side.
    - Leave all boundary lines and all non-enclosing dangling segments unchanged.

    Key insight:
    - The target color is not tied to a fixed palette: the solver must first detect which black rectangles are truly closed, then perform a minority-color decision on each region's boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _detect_regions(g):
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(region["fill_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_H_MIN, GRID_H_MAX)
    width = rng.randint(GRID_W_MIN, GRID_W_MAX)
    region_count = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)

    branch_types = [rng.choice(["unique", "tie"]) for _ in range(region_count)]
    if region_count >= 2:
        branch_types[0] = "unique"
        branch_types[1] = "tie"
        rng.shuffle(branch_types)

    objects = [_make_region_object(rng, branch_type) for branch_type in branch_types]
    open_count = rng.randint(OPEN_RECT_COUNT_MIN, OPEN_RECT_COUNT_MAX)
    segment_count = rng.randint(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX)
    for _ in range(open_count):
        objects.append(_make_open_rectangle(rng))
    for _ in range(segment_count):
        objects.append(_make_segment_object(rng))

    grid = _place_objects(rng, height, width, objects)
    if grid is None:
        return None

    regions = _detect_regions(grid)
    detected_types = Counter(region["branch_type"] for region in regions)
    expected_types = Counter(branch_types)
    if len(regions) != region_count or detected_types != expected_types:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
