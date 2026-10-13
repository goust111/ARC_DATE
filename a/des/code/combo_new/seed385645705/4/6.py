import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
TARGET_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 500

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _all_bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand_cells(
    cells: set[tuple[int, int]], h: int, w: int, radius: int = 1
) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for nr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for nc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                expanded.add((nr, nc))
    return expanded


def _arrow_cells(center: tuple[int, int], step: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    dr, dc = step
    if dr == 0:
        return {(r, c), (r - 1, c), (r + 1, c), (r, c + dc)}
    return {(r, c), (r, c - 1), (r, c + 1), (r + dr, c)}


def _parse_arrow(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    assert len(cells) == 4, "each arrow must be a 4-cell T-shape"
    degrees = {
        cell: sum((nr, nc) in cells for nr, nc in _neighbors4(*cell))
        for cell in cells
    }
    centers = [cell for cell, degree in degrees.items() if degree == 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(centers) == 1 and len(leaves) == 3, "invalid arrow component"

    center = centers[0]
    neighbors = [cell for cell in _neighbors4(*center) if cell in cells]
    same_row = [cell for cell in neighbors if cell[0] == center[0]]
    same_col = [cell for cell in neighbors if cell[1] == center[1]]
    if len(same_col) == 2 and len(same_row) == 1:
        tip = same_row[0]
    elif len(same_row) == 2 and len(same_col) == 1:
        tip = same_col[0]
    else:
        raise AssertionError("invalid arrow geometry")
    step = (tip[0] - center[0], tip[1] - center[1])
    back = (center[0] - step[0], center[1] - step[1])
    assert back not in cells, "arrow tip must point away from the tail"
    return center, tip, step


def _ray_target_color(
    grid: np.ndarray, tip: tuple[int, int], step: tuple[int, int]
) -> int:
    h, w = grid.shape
    r, c = tip
    dr, dc = step
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        value = int(grid[r, c])
        if value not in (BLACK, GREY, MAROON):
            return value
    raise AssertionError("ray did not hit a colored target pixel")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each maroon arrow away from its grey frame to fetch a color and fill that frame's interior.

    Input structure:
    - The background is black.
    - One or more grey one-cell-thick rectangular frames enclose black interiors.
    - Each frame has exactly one maroon arrow attached just outside one side, and
      scattered elsewhere are isolated single colored pixels that are neither black,
      grey, nor maroon.

    Rule:
    - For each maroon arrow, identify the grey frame cell directly behind the arrow
      center to determine which frame it belongs to.
    - Start from the arrow tip and march outward along the arrow direction until the
      first non-black, non-grey, non-maroon pixel is found.
    - Fill the black interior of the associated grey frame with that found color, then
      erase the arrow back to black.

    Key insight:
    - The arrow carries two linked signals at once: its attachment point picks the
      owning frame, and its facing direction selects the color source by ray-casting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_components = _connected_components(g == GREY)
    assert grey_components, "expected at least one grey frame"

    frames: list[dict[str, object]] = []
    frame_by_cell: dict[tuple[int, int], int] = {}
    for comp in grey_components:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"
        expected = _frame_cells(bbox)
        assert comp == expected, "grey component must be a complete rectangular frame"
        assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "frame interior must start black"
        frame_id = len(frames)
        frames.append({"bbox": bbox, "arrow": None})
        for cell in comp:
            frame_by_cell[cell] = frame_id

    arrow_components = _connected_components(g == MAROON)
    assert len(arrow_components) == len(frames), "expected one arrow per frame"
    assigned_frames: set[int] = set()

    for comp in arrow_components:
        center, tip, step = _parse_arrow(comp)
        anchor = (center[0] - step[0], center[1] - step[1])
        frame_id = frame_by_cell.get(anchor)
        assert frame_id is not None, "arrow must point away from a frame side"
        assert frame_id not in assigned_frames, "frame must have exactly one arrow"
        assigned_frames.add(frame_id)
        frames[frame_id]["arrow"] = {
            "cells": comp,
            "tip": tip,
            "step": step,
        }

    out = g.copy()
    for frame in frames:
        arrow = frame["arrow"]
        assert arrow is not None, "missing arrow assignment"
        tip = arrow["tip"]
        step = arrow["step"]
        fill_color = _ray_target_color(g, tip, step)
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    return out


def _region_candidate(
    h: int,
    w: int,
    direction: str,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    step = DIRS[direction]
    for _ in range(250):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        if h - frame_h - 2 < 1 or w - frame_w - 2 < 1:
            return None

        r0 = random.randint(1, h - frame_h - 2)
        c0 = random.randint(1, w - frame_w - 2)
        bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        r0, c0, r1, c1 = bbox

        if direction == "up":
            anchor = random.randint(c0 + 1, c1 - 1)
            center = (r0 - 1, anchor)
            max_dist = r0 - 2
        elif direction == "down":
            anchor = random.randint(c0 + 1, c1 - 1)
            center = (r1 + 1, anchor)
            max_dist = (h - 1) - (r1 + 2)
        elif direction == "left":
            anchor = random.randint(r0 + 1, r1 - 1)
            center = (anchor, c0 - 1)
            max_dist = c0 - 2
        else:
            anchor = random.randint(r0 + 1, r1 - 1)
            center = (anchor, c1 + 1)
            max_dist = (w - 1) - (c1 + 2)

        if max_dist < 2:
            continue

        distance = random.randint(2, max_dist)
        tip = (center[0] + step[0], center[1] + step[1])
        target = (tip[0] + step[0] * distance, tip[1] + step[1] * distance)
        path = {
            (tip[0] + step[0] * k, tip[1] + step[1] * k)
            for k in range(1, distance)
        }
        frame = _frame_cells(bbox)
        arrow = _arrow_cells(center, step)
        occupied = frame | arrow | {target}
        region = _all_bbox_cells(bbox) | arrow | path | {target}
        footprint = _expand_cells(region, h, w, radius=1)

        all_cells = region | footprint
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            continue
        if footprint & blocked:
            continue
        return {
            "bbox": bbox,
            "frame": frame,
            "arrow": arrow,
            "tip": tip,
            "step": step,
            "path": path,
            "target": target,
            "footprint": footprint,
            "occupied": occupied,
        }
    return None


def _shape_range(n_regions: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_regions == 2:
        return (18, 24), (18, 24)
    if n_regions == 3:
        return (22, 28), (22, 28)
    return (26, 30), (26, 30)


def _expected_output(grid: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        color = region["color"]
        r0, c0, r1, c1 = region["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        for r, c in region["arrow"]:
            out[r, c] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(2, 4)
    h_range, w_range = _shape_range(n_regions)
    h = random.randint(*h_range)
    w = random.randint(*w_range)

    directions = random.sample(list(DIRS.keys()), n_regions)
    colors = random.sample(TARGET_COLORS, n_regions)

    blocked: set[tuple[int, int]] = set()
    regions: list[dict[str, object]] = []
    for direction, color in zip(directions, colors):
        region = _region_candidate(h, w, direction, blocked)
        if region is None:
            return None
        region["direction"] = direction
        region["color"] = color
        blocked |= region["footprint"]
        regions.append(region)

    grid = np.full((h, w), BLACK, dtype=int)
    for region in regions:
        for r, c in region["frame"]:
            grid[r, c] = GREY
        for r, c in region["arrow"]:
            grid[r, c] = MAROON
        tr, tc = region["target"]
        grid[tr, tc] = region["color"]

    expected = _expected_output(grid, regions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
