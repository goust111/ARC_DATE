from __future__ import annotations

import random
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_003_5", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


BG = Color.BLACK
TAIL = Color.GREY
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
POS_SLOPE_DIRS = [(-1, -1), (1, 1)]
NEG_SLOPE_DIRS = [(-1, 1), (1, -1)]
GEN_ATTEMPTS = 400
INTERSECTION_MARGIN = 6
MIN_FRAME_SIDE = 4
MAX_MARKER_DISTANCE = 4
MIN_MARKER_DISTANCE = 2


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = map(int, start)
    dr, dc = map(int, direction)
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _frame_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r0 = min(int(a[0]), int(b[0]))
    r1 = max(int(a[0]), int(b[0]))
    c0 = min(int(a[1]), int(b[1]))
    c1 = max(int(a[1]), int(b[1]))
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _tail_marker_from_component(comp: dict, grid: np.ndarray) -> dict | None:
    coords = [tuple(map(int, rc)) for rc in comp["coords"]]
    colors = {coord: int(grid[coord]) for coord in coords}
    marker_cells = [coord for coord, color in colors.items() if color != TAIL]
    grey_cells = {coord for coord, color in colors.items() if color == TAIL}
    if len(coords) != 3 or len(marker_cells) != 1 or len(grey_cells) != 2:
        return None

    marker = marker_cells[0]
    marker_color = colors[marker]
    if marker_color == BG:
        return None

    for dr, dc in DIAG_DIRS:
        expected = {(marker[0] - dr, marker[1] - dc), (marker[0] - 2 * dr, marker[1] - 2 * dc)}
        if grey_cells == expected:
            return {
                "color": marker_color,
                "marker": marker,
                "direction": (dr, dc),
                "tail": tuple(sorted(grey_cells)),
            }
    return None


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    tailed: list[dict] = []
    source_singletons: list[dict] = []

    for comp in components:
        parsed = _tail_marker_from_component(comp, g)
        if parsed is not None:
            tailed.append(parsed)
            continue

        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        if len(coords) == 1:
            r, c = coords[0]
            color = int(g[r, c])
            if color not in (bg, TAIL):
                source_singletons.append({"color": color, "pos": (r, c)})
                continue
        raise AssertionError("unexpected component structure")

    assert len(tailed) == 4, f"expected four tailed markers, got {len(tailed)}"
    assert len(source_singletons) == 1, f"expected one color source, got {len(source_singletons)}"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for item in tailed:
        by_color[int(item["color"])].append(item)

    assert len(by_color) == 2, f"expected two tailed colors, got {sorted(by_color)}"
    assert all(len(items) == 2 for items in by_color.values()), "each tailed color must appear twice"

    source = source_singletons[0]
    assert int(source["color"]) not in by_color, "source must use a third color"

    intersections: dict[int, tuple[int, int]] = {}
    for color, items in by_color.items():
        ray_a = set(_ray_cells(g.shape, items[0]["marker"], items[0]["direction"]))
        ray_b = set(_ray_cells(g.shape, items[1]["marker"], items[1]["direction"]))
        hits = sorted(ray_a & ray_b)
        assert len(hits) == 1, f"color {color} should have exactly one ray intersection"
        intersections[int(color)] = hits[0]

    corners = list(intersections.values())
    assert len(corners) == 2
    assert corners[0] != corners[1], "the two generated corners must differ"
    assert corners[0][0] != corners[1][0] and corners[0][1] != corners[1][1], "rectangle must be non-degenerate"

    return {
        "bg": bg,
        "tailed": tailed,
        "source": source,
        "intersections": intersections,
        "corners": corners,
    }


def _draw_frame_over(grid: np.ndarray, corners: list[tuple[int, int]], color: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    (r1, c1), (r2, c2) = corners
    r0, r1 = sorted((int(r1), int(r2)))
    c0, c1 = sorted((int(c1), int(c2)))
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Turn two same-color diagonal ray intersections into the corners of a new hollow frame.

    Input structure:
    - A black background contains four multicolor 8-connected components, each made of one colored marker pixel and a two-cell gray diagonal tail.
    - The four markers use exactly two non-gray colors, with one tailed pair per color, plus one isolated singleton pixel of a third color.
    - Within each same-color pair, the two rays implied by the tails intersect at exactly one in-bounds cell.

    Rule:
    - For each tailed marker, cast a diagonal ray from the marker in the direction opposite to its gray tail.
    - Pair the markers by color and compute the unique intersection cell of each pair of rays.
    - Treat the two intersections as opposite corners of an axis-aligned one-pixel-thick hollow rectangle, draw that frame in the singleton pixel's color, and leave all other pixels unchanged.

    Key insight:
    - The visible markers never draw the result directly: tails define rays, ray intersections define corners, and only those generated corners determine the final rectangular frame while the singleton contributes color only.
    """
    scene = _extract_scene(np.asarray(input_grid, dtype=int))
    return _draw_frame_over(np.asarray(input_grid, dtype=int), scene["corners"], scene["source"]["color"])


def _group_cells(groups: list[set[tuple[int, int]]]) -> bool:
    for i, group_a in enumerate(groups):
        for j, group_b in enumerate(groups):
            if i >= j:
                continue
            for ra, ca in group_a:
                for rb, cb in group_b:
                    if max(abs(ra - rb), abs(ca - cb)) <= 1:
                        return False
    return True


def _sample_source_cell(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    frame: set[tuple[int, int]],
    corners: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    H, W = shape
    (r1, c1), (r2, c2) = corners
    r0, r1 = sorted((int(r1), int(r2)))
    c0, c1 = sorted((int(c1), int(c2)))

    prefer_inside = random.random() < 0.5
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            cell = (r, c)
            if cell in forbidden or cell in frame:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in forbidden):
                continue
            inside = r0 < r < r1 and c0 < c < c1
            outside = r < r0 or r > r1 or c < c0 or c > c1
            if (prefer_inside and inside) or ((not prefer_inside) and outside):
                candidates.append(cell)
    if not candidates:
        for r in range(H):
            for c in range(W):
                cell = (r, c)
                if cell in forbidden or cell in frame:
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in forbidden):
                    continue
                candidates.append(cell)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_marker_component(
    corner: tuple[int, int], color: int, direction: tuple[int, int], distance: int
) -> dict | None:
    dr, dc = direction
    marker = (corner[0] - distance * dr, corner[1] - distance * dc)
    tail1 = (marker[0] - dr, marker[1] - dc)
    tail2 = (marker[0] - 2 * dr, marker[1] - 2 * dc)
    cells = [marker, tail1, tail2]
    return {"color": int(color), "marker": marker, "tail": [tail1, tail2], "cells": cells, "direction": direction}


def _generated_scene_valid(grid: np.ndarray) -> None:
    scene = _extract_scene(grid)
    corners = scene["corners"]
    frame_color = int(scene["source"]["color"])
    frame = _frame_cells(corners[0], corners[1])
    occupied = {
        tuple(map(int, rc))
        for rc in np.argwhere(np.asarray(grid, dtype=int) != int(scene["bg"]))
    }
    marker_cells = set()
    for item in scene["tailed"]:
        marker_cells.add(item["marker"])
        marker_cells.update(item["tail"])

    assert int(scene["source"]["pos"][0]) >= 0
    assert scene["source"]["pos"] not in frame, "source should not lie on the frame perimeter"
    assert frame.isdisjoint(marker_cells), "frame should not overwrite existing tailed markers"

    out = main(grid)
    expected = np.asarray(grid, dtype=int).copy()
    for r, c in frame:
        expected[r, c] = frame_color
    assert np.array_equal(out, expected), "main output does not match independently rendered frame"
    assert np.any(out != grid), "generated sample should not be degenerate"
    assert len(set(map(int, np.unique(grid))) - {int(scene["bg"])}) == 4, "expected exactly four foreground colors"
    assert len(occupied) == 13, "expected 4*(marker+2 tails)+1 source foreground cells"


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 30)
    W = random.randint(18, 30)
    grid = np.full((H, W), BG, dtype=int)

    usable_colors = [color for color in Color.NOT_BLACK if color != TAIL]
    pair_a, pair_b, frame_color = random.sample(usable_colors, 3)

    r_low, r_high = INTERSECTION_MARGIN, H - INTERSECTION_MARGIN - 1
    c_low, c_high = INTERSECTION_MARGIN, W - INTERSECTION_MARGIN - 1
    if r_low >= r_high or c_low >= c_high:
        return None

    corner_a = (random.randint(r_low, r_high), random.randint(c_low, c_high))
    for _ in range(60):
        corner_b = (random.randint(r_low, r_high), random.randint(c_low, c_high))
        if abs(corner_a[0] - corner_b[0]) >= MIN_FRAME_SIDE and abs(corner_a[1] - corner_b[1]) >= MIN_FRAME_SIDE:
            break
    else:
        return None

    components: list[dict] = []
    for corner, color in ((corner_a, pair_a), (corner_b, pair_b)):
        directions = [random.choice(POS_SLOPE_DIRS), random.choice(NEG_SLOPE_DIRS)]
        random.shuffle(directions)
        for direction in directions:
            distance = random.randint(MIN_MARKER_DISTANCE, MAX_MARKER_DISTANCE)
            component = _build_marker_component(corner, color, direction, distance)
            assert component is not None
            if not all(in_bounds((H, W), r, c) for r, c in component["cells"]):
                return None
            components.append(component)

    flat_cells = [tuple(cell) for comp in components for cell in comp["cells"]]
    if len(set(flat_cells)) != len(flat_cells):
        return None

    groups = [set(comp["cells"]) for comp in components]
    if not _group_cells(groups):
        return None

    frame = _frame_cells(corner_a, corner_b)
    occupied = set(flat_cells)
    if not frame.isdisjoint(occupied):
        return None

    source_pos = _sample_source_cell((H, W), occupied, frame, (corner_a, corner_b))
    if source_pos is None:
        return None

    groups.append({source_pos})
    if not _group_cells(groups):
        return None

    for comp in components:
        marker_r, marker_c = comp["marker"]
        grid[marker_r, marker_c] = int(comp["color"])
        for tail_r, tail_c in comp["tail"]:
            grid[tail_r, tail_c] = TAIL

    grid[source_pos] = int(frame_color)

    try:
        _generated_scene_valid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-level-frame ARC input")


if __name__ == "__main__":
    module_path = Path(__file__)
    mod = SourceFileLoader("_self_check_arc_task_5", str(module_path)).load_module()
    for _ in range(20):
        sample = mod.generate_input()
        mod._generated_scene_valid(sample)
