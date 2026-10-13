from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
BLUE = 1
PALETTE = [2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (11, 18)
GRID_W_RANGE = (11, 18)
OBJECT_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _symmetry_kind(mask: np.ndarray) -> str:
    lr = np.array_equal(mask, np.fliplr(mask))
    ud = np.array_equal(mask, np.flipud(mask))
    if lr and not ud:
        return "h_only"
    if ud and not lr:
        return "v_only"
    if not lr and not ud:
        return "asym"
    raise AssertionError("shape must not be symmetric in both directions")


def _build_shape_catalog() -> dict[str, list[np.ndarray]]:
    horizontal = [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    vertical = [mask.T.copy() for mask in horizontal]
    asymmetric = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ]
    catalog = {"h_only": horizontal, "v_only": vertical, "asym": asymmetric}
    for kind, masks in catalog.items():
        for mask in masks:
            assert _symmetry_kind(mask) == kind
            assert 3 <= int(mask.sum()) <= 7
    return catalog


SHAPES = _build_shape_catalog()


def _trace_bounce_path(shape: tuple[int, int], start: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    path = [(r, c)]
    dc = 1
    bounces = 0

    while True:
        nr = r - 1
        nc = c + dc
        if nc < 0 or nc >= w:
            dc *= -1
            bounces += 1
            nc = c + dc
        if nr < 0 or nr >= h:
            break
        r, c = nr, nc
        path.append((r, c))

    return path, bounces


def _wall_mask(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    wall = np.zeros(shape, dtype=bool)
    for r, c in path:
        wall[int(r), int(c)] = True
    return wall


def _label_regions(wall: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = wall.shape
    labels = np.full((h, w), -1, dtype=int)
    next_label = 0

    for r in range(h):
        for c in range(w):
            if wall[r, c] or labels[r, c] != -1:
                continue
            labels[r, c] = next_label
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            next_label += 1

    return labels, next_label


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _mirror_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = map(int, bbox)
    if orientation == "vertical":
        return r0 + r1 - coords[:, 0], coords[:, 1]
    if orientation == "horizontal":
        return coords[:, 0], c0 + c1 - coords[:, 1]
    raise AssertionError(f"unknown orientation: {orientation}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bounced blue diagonal as a temporary wall that chooses whether each object flips vertically or horizontally.

    Input structure:
    - A black background contains exactly one blue start pixel and several separate monochrome foreground objects in other colors.
    - Each object is a 4-connected irregular mask whose bounding box does not touch the grid border.
    - The blue pixel defines a bounced up-right diagonal path that avoids all foreground pixels.

    Rule:
    - Starting from the unique blue pixel, repeatedly move one row upward and one column horizontally, initially up-right.
    - Whenever the next horizontal step would leave the grid, reverse only the horizontal direction and continue; mark every visited path cell blue.
    - Treat the full blue path as a 4-connectivity wall. Objects in the region containing the top-left corner flip vertically inside their own bbox; objects in the region containing the bottom-right corner flip horizontally inside their own bbox.
    - Erase the entire blue path, including the original start pixel, so only the mirrored foreground objects remain on black.

    Key insight:
    - The blue pixels are not output content. Their bounced geometry partitions the canvas, and that partition decides which local mirror each object uses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    start_cells = np.argwhere(g == BLUE)
    assert start_cells.shape == (1, 2), "expected exactly one blue start pixel"
    start = tuple(map(int, start_cells[0]))

    path, _bounces = _trace_bounce_path(g.shape, start)
    wall = _wall_mask(g.shape, path)
    labels, region_count = _label_regions(wall)
    assert region_count == 2, "path must split the grid into exactly two regions"
    region_a = int(labels[0, 0])
    region_b = int(labels[-1, -1])
    assert region_a >= 0 and region_b >= 0 and region_a != region_b

    colors = sorted(set(map(int, np.unique(g))) - {bg, BLUE})
    components = find_components(g, bg=bg, colors=colors, connectivity=4, monochromatic=True)

    out = np.full_like(g, bg)
    for comp in components:
        coords = comp["coords"]
        bbox = tuple(map(int, comp["bbox"]))
        color = int(comp["color"])
        object_regions = labels[coords[:, 0], coords[:, 1]]
        region = int(object_regions[0])
        assert region in (region_a, region_b)
        assert np.all(object_regions == region), "object must stay inside one partition"

        orientation = "vertical" if region == region_a else "horizontal"
        rr, cc = _mirror_coords(coords, bbox, orientation)
        assert np.all(out[rr, cc] == bg), "mirrored objects must not overlap"
        out[rr, cc] = color

    return out


def _candidate_start_columns(h: int, w: int) -> list[tuple[int, list[tuple[int, int]], np.ndarray, int, int]]:
    candidates = []
    for c in range(1, w - 1):
        path, bounces = _trace_bounce_path((h, w), (h - 1, c))
        if bounces < 1:
            continue
        wall = _wall_mask((h, w), path)
        if wall[0, 0] or wall[h - 1, w - 1]:
            continue
        labels, n_regions = _label_regions(wall)
        if n_regions != 2:
            continue
        region_a = int(labels[0, 0])
        region_b = int(labels[h - 1, w - 1])
        if region_a < 0 or region_b < 0 or region_a == region_b:
            continue
        candidates.append((c, path, labels, region_a, region_b))
    return candidates


def _placement_options(
    labels: np.ndarray,
    region: int,
    shape: np.ndarray,
    reserved: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = labels.shape
    sh, sw = shape.shape
    opts = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if np.any(labels[r0 : r0 + sh, c0 : c0 + sw] != region):
                continue
            if reserved[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            opts.append((r0, c0))
    return opts


def _stamp_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> dict:
    sh, sw = shape.shape
    rows, cols = np.nonzero(shape)
    grid[top + rows, left + cols] = color
    reserved[top - 1 : top + sh + 1, left - 1 : left + sw + 1] = True
    return {
        "color": int(color),
        "bbox": (top, left, top + sh - 1, left + sw - 1),
        "center_c": float(left + (sw - 1) / 2),
        "shape": shape.copy(),
    }


def _make_object_plan(n_objects: int) -> list[tuple[str, str]]:
    n_a = random.randint(1, n_objects - 1)
    n_b = n_objects - n_a
    plan = [("A", "h_only"), ("B", "v_only")]

    for _ in range(n_a - 1):
        plan.append(("A", random.choice(["h_only", "asym", "asym"])))
    for _ in range(n_b - 1):
        plan.append(("B", random.choice(["v_only", "asym", "asym"])))

    random.shuffle(plan)
    return plan


def _cross_half_counterexample(placed: list[dict], width: int, region_a: int, region_b: int) -> bool:
    mid = (width - 1) / 2
    for item in placed:
        if item["region_id"] == region_a and item["center_c"] > mid:
            return True
        if item["region_id"] == region_b and item["center_c"] < mid:
            return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        assert grid.shape == out.shape
        assert not np.any(out == BLUE)
        assert not np.array_equal(grid, out)
        assert np.count_nonzero(grid == BLUE) == 1

        start = tuple(map(int, np.argwhere(grid == BLUE)[0]))
        path, bounces = _trace_bounce_path(grid.shape, start)
        assert bounces >= 1
        wall = _wall_mask(grid.shape, path)
        labels, n_regions = _label_regions(wall)
        assert n_regions == 2
        region_a = int(labels[0, 0])
        region_b = int(labels[-1, -1])
        assert region_a != region_b

        colors = sorted(set(map(int, np.unique(grid))) - {BG, BLUE})
        comps = find_components(grid, bg=BG, colors=colors, connectivity=4, monochromatic=True)
        assert OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]

        seen_regions = set()
        changed_a = False
        changed_b = False
        cross_half = False
        mid = (grid.shape[1] - 1) / 2

        for comp in comps:
            coords = comp["coords"]
            bbox = tuple(map(int, comp["bbox"]))
            color = int(comp["color"])
            region = int(labels[coords[0, 0], coords[0, 1]])
            assert region in (region_a, region_b)
            assert np.all(labels[coords[:, 0], coords[:, 1]] == region)
            r0, c0, r1, c1 = bbox
            assert np.all(labels[r0 : r1 + 1, c0 : c1 + 1] == region)
            seen_regions.add(region)

            patch = _component_mask(coords, bbox)
            if region == region_a:
                changed_a = changed_a or not np.array_equal(patch, np.flipud(patch))
                cross_half = cross_half or ((c0 + c1) / 2 > mid)
                expected_patch = np.flipud(patch)
            else:
                changed_b = changed_b or not np.array_equal(patch, np.fliplr(patch))
                cross_half = cross_half or ((c0 + c1) / 2 < mid)
                expected_patch = np.fliplr(patch)

            out_patch = out[r0 : r1 + 1, c0 : c1 + 1] == color
            assert np.array_equal(out_patch, expected_patch)

        assert seen_regions == {region_a, region_b}
        assert changed_a and changed_b
        assert cross_half
        return True
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    start_candidates = _candidate_start_columns(h, w)
    if not start_candidates:
        return None

    start_c, path, labels, region_a, region_b = random.choice(start_candidates)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    plan = _make_object_plan(n_objects)
    colors = random.sample(PALETTE, n_objects)

    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, start_c] = BLUE
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for (region_name, shape_kind), color in zip(plan, colors):
        region_id = region_a if region_name == "A" else region_b
        shape = random.choice(SHAPES[shape_kind]).copy()
        options = _placement_options(labels, region_id, shape, reserved)
        if not options:
            return None

        random.shuffle(options)
        choice = options[0]
        if region_name == "A":
            right_half = [opt for opt in options if opt[1] + (shape.shape[1] - 1) / 2 > (w - 1) / 2]
            if right_half and not _cross_half_counterexample(placed, w, region_a, region_b):
                choice = random.choice(right_half)
        else:
            left_half = [opt for opt in options if opt[1] + (shape.shape[1] - 1) / 2 < (w - 1) / 2]
            if left_half and not _cross_half_counterexample(placed, w, region_a, region_b):
                choice = random.choice(left_half)

        top, left = choice
        item = _stamp_object(grid, reserved, shape, color, top, left)
        item["region_id"] = region_id
        placed.append(item)

    if not _cross_half_counterexample(placed, w, region_a, region_b):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bounced-partition mirror puzzle")
