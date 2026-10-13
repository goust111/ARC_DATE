import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_cells(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _dims(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _reflect_in_bbox(cells):
    h, w = _dims(cells)
    return sorted((r, w - 1 - c) for r, c in cells), h, w


def _translate(cells, dr: int, dc: int):
    return sorted((r + dr, c + dc) for r, c in cells)


def _components_of_color(grid: np.ndarray, color: int, col_limit: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(min(col_limit, w)):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < col_limit and not visited[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            comps.append(sorted(cells))
    return comps


def _axis_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(cols) == 1, "expected exactly one full-height red axis"
    return cols[0]


def _left_objects(grid: np.ndarray, axis: int):
    colors = sorted(
        {
            int(v)
            for v in np.unique(grid[:, :axis])
            if int(v) not in {BLACK, RED}
        }
    )
    objects = []
    for color in colors:
        comps = _components_of_color(grid, color, axis)
        for cells in comps:
            r0, c0, r1, c1 = _bbox(cells)
            objects.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return objects


def _markers(grid: np.ndarray, axis: int):
    markers = []
    h, w = grid.shape
    for r in range(h):
        for c in range(axis + 1, w):
            color = int(grid[r, c])
            if color in {BLACK, RED}:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    assert int(grid[nr, nc]) in {BLACK, RED}, "marker pixels must be isolated"
            markers.append({"row": r, "col": c, "color": color})
    return markers


def _bbox_center(top: int, left: int, height: int, width: int):
    return top + (height - 1) / 2.0, left + (width - 1) / 2.0


def _distance_to_center(marker, center_r: float, center_c: float) -> float:
    return abs(marker["row"] - center_r) + abs(marker["col"] - center_c)


def _greedy_marker_assignment(reflected_infos, markers):
    remaining = list(markers)
    assignments = []
    for info in reflected_infos:
        center_r, center_c = _bbox_center(info["top"], info["left"], info["height"], info["width"])
        best_idx = min(
            range(len(remaining)),
            key=lambda i: (
                _distance_to_center(remaining[i], center_r, center_c),
                remaining[i]["row"],
                remaining[i]["col"],
                remaining[i]["color"],
            ),
        )
        marker = remaining.pop(best_idx)
        assignments.append(marker)
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the left sprites across the red axis, then restack and recolor those reflections by nearest right-side markers.

    Input structure:
    - The grid has a black background and a single full-height red vertical axis.
    - The left side contains 3-4 distinct-color 4-connected sprites with pairwise different areas.
    - The right side contains the same number of isolated colored marker pixels.

    Rule:
    - Mirror each left sprite horizontally, but do not keep its original position.
    - Sort the sprites by area from small to large, then stack their mirrored bboxes top-to-bottom with no vertical gaps and with each bbox starting at axis column + 2.
    - For each stacked mirrored bbox in that area order, greedily take the remaining marker pixel with minimum Manhattan distance to the bbox center and paint the whole mirrored sprite with that marker color.
    - Keep the original left sprites and the red axis; any marker covered by a painted mirrored sprite disappears, while uncovered markers remain.

    Key insight:
    - Shape, placement, and color come from three independent cues: horizontal reflection gives the mask, area ranking gives the destination order, and nearest remaining markers supply the paint colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    axis = _axis_col(g)
    objects = _left_objects(g, axis)
    markers = _markers(g, axis)
    assert len(objects) == len(markers) and len(objects) >= 1

    ordered = sorted(
        objects,
        key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1], obj["color"]),
    )

    reflected_infos = []
    top = 0
    left = axis + 2
    for obj in ordered:
        r0, c0, _r1, _c1 = obj["bbox"]
        normalized = sorted((r - r0, c - c0) for r, c in obj["cells"])
        reflected_shape, height, width = _reflect_in_bbox(normalized)
        placed = _translate(reflected_shape, top, left)
        assert top + height <= g.shape[0], "stacked reflections must fit vertically"
        assert left + width <= g.shape[1], "stacked reflections must fit horizontally"
        reflected_infos.append(
            {
                "placed": placed,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
            }
        )
        top += height

    assigned_markers = _greedy_marker_assignment(reflected_infos, markers)
    out = g.copy()
    for info, marker in zip(reflected_infos, assigned_markers):
        for r, c in info["placed"]:
            out[r, c] = marker["color"]
    return out


def _random_shape(area: int):
    for _ in range(400):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    if -4 <= nr <= 4 and -4 <= nc <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            candidate = random.choice(frontier)
            cells.add(candidate)
        if len(cells) != area:
            continue
        shape = _normalize_cells(cells)
        height, width = _dims(shape)
        if not (2 <= height <= 4 and 2 <= width <= 4):
            continue
        if len(shape) == height * width:
            continue
        return shape
    return None


def _separated(bbox_a, bbox_b, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_left_objects(shapes, height: int, axis: int):
    placements = []
    for shape in random.sample(shapes, k=len(shapes)):
        sh, sw = _dims(shape)
        max_col = axis - sw - 1
        if max_col < 0:
            return None
        for _ in range(250):
            top = random.randint(0, height - sh)
            left = random.randint(0, max_col)
            bbox = (top, left, top + sh - 1, left + sw - 1)
            if all(_separated(bbox, placed["bbox"], gap=1) for placed in placements):
                placements.append(
                    {
                        "shape": shape,
                        "bbox": bbox,
                        "top": top,
                        "left": left,
                        "area": len(shape),
                    }
                )
                break
        else:
            return None
    return placements


def _choose_marker_cell(info, want_uncovered: bool, chosen_cells):
    shape_cells = list(info["placed"])
    shape_set = set(shape_cells)
    bbox_cells = [
        (r, c)
        for r in range(info["top"], info["top"] + info["height"])
        for c in range(info["left"], info["left"] + info["width"])
    ]
    empty_cells = [cell for cell in bbox_cells if cell not in shape_set]
    primary = empty_cells if want_uncovered else shape_cells
    secondary = shape_cells if want_uncovered else empty_cells
    center_r, center_c = _bbox_center(info["top"], info["left"], info["height"], info["width"])

    for pool in (primary, secondary, bbox_cells):
        shuffled = list(pool)
        random.shuffle(shuffled)
        shuffled.sort(key=lambda cell: abs(cell[0] - center_r) + abs(cell[1] - center_c))
        for cell in shuffled:
            if all(abs(cell[0] - r) + abs(cell[1] - c) != 1 for r, c in chosen_cells):
                return cell
    return None


def _gen_attempt():
    count = random.choice([3, 4])
    areas = random.sample([4, 5, 6, 7, 8], count)
    shapes = []
    for area in areas:
        shape = _random_shape(area)
        if shape is None:
            return None
        shapes.append(shape)

    stack_height = sum(_dims(shape)[0] for shape in shapes)
    max_width = max(_dims(shape)[1] for shape in shapes)
    min_height = max(14, stack_height)
    max_height = min(18, stack_height + 4)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    axis = random.randint(8, 11)
    min_width = max(18, axis + 2 + max_width + 2)
    max_width_total = min(24, axis + 2 + max_width + 8)
    if min_width > max_width_total:
        return None
    width = random.randint(min_width, max_width_total)

    left_placements = _place_left_objects(shapes, height, axis)
    if left_placements is None or len(left_placements) != count:
        return None

    ordered_left = sorted(left_placements, key=lambda item: item["area"])
    reflected_infos = []
    top = 0
    right_left = axis + 2
    for placed in ordered_left:
        reflected_shape, sh, sw = _reflect_in_bbox(placed["shape"])
        painted = _translate(reflected_shape, top, right_left)
        reflected_infos.append(
            {
                "placed": painted,
                "top": top,
                "left": right_left,
                "height": sh,
                "width": sw,
                "area": placed["area"],
            }
        )
        top += sh

    if top > height:
        return None

    colors = random.sample(PALETTE, 2 * count)
    left_colors = colors[:count]
    marker_colors = colors[count:]
    random.shuffle(left_colors)

    grid = np.zeros((height, width), dtype=int)
    grid[:, axis] = RED

    for placed, color in zip(left_placements, left_colors):
        for r, c in _translate(placed["shape"], placed["top"], placed["left"]):
            grid[r, c] = color

    uncovered_count = random.randint(1, count - 1)
    uncovered_indices = set(random.sample(range(count), uncovered_count))
    chosen_marker_cells = []
    markers = []
    for idx, (info, color) in enumerate(zip(reflected_infos, marker_colors)):
        cell = _choose_marker_cell(info, idx in uncovered_indices, chosen_marker_cells)
        if cell is None:
            return None
        chosen_marker_cells.append(cell)
        markers.append({"row": cell[0], "col": cell[1], "color": color})
        grid[cell[0], cell[1]] = color

    if len({marker["row"] for marker in markers}) != count:
        return None

    expected = grid.copy()
    assigned = _greedy_marker_assignment(reflected_infos, markers)
    for info, marker in zip(reflected_infos, assigned):
        for r, c in info["placed"]:
            expected[r, c] = marker["color"]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
