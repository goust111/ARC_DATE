from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 5
RECT_COLORS = (1, 2, 3, 4)
RECT_COUNT_CHOICES = (3, 4)
GRID_SIZE_RANGE = (16, 23)
RECT_SIZE_RANGE = (4, 8)
CORE_SIZE_CHOICES = (1, 1, 1, 2)
MAJOR_CORNERS = ("nw", "ne", "sw", "se")
MINOR_EXTENT_CHOICES = (0, 0, 1, 1, 2)
GRID_MARGIN = 1
GEN_RETRIES = 400


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    cols = set(map(int, np.unique(grid)))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def _rect_cells(rect: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = rect["bbox"]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _compose_from_rects(shape: tuple[int, int], rects: list[dict], order: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        rect = rects[idx]
        rr, cc = _rect_cells(rect)
        out[rr, cc] = rect["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the overlapped rectangles and repaint them in marker-count order.

    Input structure:
    - The background is black, with 3 or 4 colored axis-aligned solid rectangles and sparse grey marker pixels.
    - Each non-grey foreground color belongs to one rectangle; overlaps may currently show any one of the overlapping colors.
    - The visible non-overlapped border fragments are sufficient to recover each rectangle's full bounding box.

    Rule:
    - For every non-black, non-grey color, infer its rectangle as the bounding box of that color's visible pixels.
    - Count grey marker pixels inside each inferred bounding box.
    - Create a black canvas of the same size and fill the rectangles from the smallest marker count to the largest, letting later rectangles overwrite earlier ones.
    - Do not draw the grey marker pixels in the output.

    Key insight:
    - The overlap priority is not encoded by position or color id; it must be recovered from hidden marker counts inside the reconstructed rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rects: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in colors_present(g, exclude={BG, MARKER}):
        b = bbox(g == color)
        assert b is not None, f"missing visible pixels for color {color}"
        r0, c0, r1, c1 = b
        marker_count = int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == MARKER))
        rects.append((marker_count, int(color), b))

    counts = [count for count, _color, _bbox in rects]
    assert len(rects) >= 3, "expected at least three rectangles"
    assert len(counts) == len(set(counts)), "marker counts must be unique"

    out = np.full_like(g, BG)
    for _count, color, (r0, c0, r1, c1) in sorted(rects):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _sample_minor(total_extra: int) -> int:
    max_minor = min(2, total_extra - 2)
    return min(random.choice(MINOR_EXTENT_CHOICES), max_minor)


def _sample_extents(corner: str, core_h: int, core_w: int) -> tuple[int, int, int, int]:
    h = random.randint(*RECT_SIZE_RANGE)
    w = random.randint(*RECT_SIZE_RANGE)
    extra_h = h - core_h
    extra_w = w - core_w
    if extra_h < 2 or extra_w < 2:
        raise ValueError("core too large for rectangle size")

    minor_vertical = _sample_minor(extra_h)
    minor_horizontal = _sample_minor(extra_w)

    if corner == "nw":
        up, down = extra_h - minor_vertical, minor_vertical
        left, right = extra_w - minor_horizontal, minor_horizontal
    elif corner == "ne":
        up, down = extra_h - minor_vertical, minor_vertical
        left, right = minor_horizontal, extra_w - minor_horizontal
    elif corner == "sw":
        up, down = minor_vertical, extra_h - minor_vertical
        left, right = extra_w - minor_horizontal, minor_horizontal
    elif corner == "se":
        up, down = minor_vertical, extra_h - minor_vertical
        left, right = minor_horizontal, extra_w - minor_horizontal
    else:
        raise ValueError(f"unknown corner {corner}")
    return up, down, left, right


def _coverage_grid(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    cov = np.zeros(shape, dtype=int)
    for rect in rects:
        rr, cc = _rect_cells(rect)
        cov[rr, cc] += 1
    return cov


def _edge_anchor_ok(rect: dict, cov: np.ndarray) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    top = np.any(cov[r0, c0 : c1 + 1] == 1)
    bottom = np.any(cov[r1, c0 : c1 + 1] == 1)
    left = np.any(cov[r0 : r1 + 1, c0] == 1)
    right = np.any(cov[r0 : r1 + 1, c1] == 1)
    return bool(top and bottom and left and right)


def _safe_marker_cells(rect: dict, cov: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    coords: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if cov[r, c] == 1:
                coords.append((r, c))
    return coords


def _hidden_output(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    order = [idx for idx, _rect in sorted(enumerate(rects), key=lambda item: item[1]["count"])]
    return _compose_from_rects(shape, rects, order)


def _assign_marker_counts(rects: list[dict], cov: np.ndarray) -> bool:
    safe_cells = [_safe_marker_cells(rect, cov) for rect in rects]
    capacities = [len(cells) for cells in safe_cells]
    if min(capacities) < 1:
        return False

    for _ in range(40):
        counts = random.sample((1, 2, 3, 4, 5), len(rects))
        shuffled = counts[:]
        random.shuffle(shuffled)
        if all(cap >= count for cap, count in zip(capacities, shuffled)):
            for rect, cells, count in zip(rects, safe_cells, shuffled):
                rect["count"] = count
                rect["markers"] = random.sample(cells, count)
            return True
    return False


def _build_rects(shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = shape
    core_h = random.choice(CORE_SIZE_CHOICES)
    core_w = random.choice(CORE_SIZE_CHOICES)
    corners = random.sample(MAJOR_CORNERS, len(colors))

    rects: list[dict] = []
    max_up = max_down = max_left = max_right = 0
    for color, corner in zip(colors, corners):
        up, down, left, right = _sample_extents(corner, core_h, core_w)
        rects.append(
            {
                "color": color,
                "corner": corner,
                "extents": (up, down, left, right),
            }
        )
        max_up = max(max_up, up)
        max_down = max(max_down, down)
        max_left = max(max_left, left)
        max_right = max(max_right, right)

    min_core_r0 = GRID_MARGIN + max_up
    max_core_r0 = h - GRID_MARGIN - core_h - max_down
    min_core_c0 = GRID_MARGIN + max_left
    max_core_c0 = w - GRID_MARGIN - core_w - max_right
    if min_core_r0 > max_core_r0 or min_core_c0 > max_core_c0:
        return None

    core_r0 = random.randint(min_core_r0, max_core_r0)
    core_c0 = random.randint(min_core_c0, max_core_c0)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    seen_bboxes = set()
    for rect in rects:
        up, down, left, right = rect["extents"]
        box = (core_r0 - up, core_c0 - left, core_r1 + down, core_c1 + right)
        if box in seen_bboxes:
            return None
        seen_bboxes.add(box)
        rect["bbox"] = box

    for i in range(len(rects)):
        box_i = rects[i]["bbox"]
        area_i = (box_i[2] - box_i[0] + 1) * (box_i[3] - box_i[1] + 1)
        for j in range(i + 1, len(rects)):
            box_j = rects[j]["bbox"]
            area_j = (box_j[2] - box_j[0] + 1) * (box_j[3] - box_j[1] + 1)
            inter = _intersection_area(box_i, box_j)
            if inter <= 0:
                return None
            if inter > int(0.70 * min(area_i, area_j)):
                return None

    cov = _coverage_grid(shape, rects)
    if not _assign_marker_counts(rects, cov):
        return None
    return rects


def _visible_bboxes_match(grid: np.ndarray, rects: list[dict]) -> bool:
    for rect in rects:
        if bbox(grid == rect["color"]) != rect["bbox"]:
            return False
    return True


def _render_input(shape: tuple[int, int], rects: list[dict]) -> np.ndarray | None:
    target_order = [idx for idx, _rect in sorted(enumerate(rects), key=lambda item: item[1]["count"])]
    for _ in range(60):
        display_order = list(range(len(rects)))
        random.shuffle(display_order)
        if display_order == target_order:
            continue

        grid = _compose_from_rects(shape, rects, display_order)
        for rect in rects:
            for r, c in rect["markers"]:
                grid[r, c] = MARKER
        if _visible_bboxes_match(grid, rects):
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIZE_RANGE),
        random.randint(*GRID_SIZE_RANGE),
    )
    n_rects = random.choice(RECT_COUNT_CHOICES)
    colors = random.sample(RECT_COLORS, n_rects)

    rects = _build_rects(shape, colors)
    if rects is None:
        return None

    inp = _render_input(shape, rects)
    if inp is None:
        return None
    expected = _hidden_output(shape, rects)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-rectangle input")
