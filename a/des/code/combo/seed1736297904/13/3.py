from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIAG_TL_BR = "tl_br"
DIAG_TR_BL = "tr_bl"
DIAGONALS = (DIAG_TL_BR, DIAG_TR_BL)

BG_COLOR = 0
GRID_MIN = 12
GRID_MAX = 28
OBJECT_COUNT_RANGE = (2, 5)
RECT_SIDE_MAX = 12
PLACEMENT_MARGIN = 1
MARKER_GAP = 1
GEN_RETRIES = 400
RECT_RETRIES = 180
COLOR_POOL = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _l_cells_from_bbox(
    r0: int, c0: int, r1: int, c1: int, diagonal: str
) -> set[tuple[int, int]]:
    if diagonal == DIAG_TL_BR:
        return {
            (r0, c0),
            (r0, c0 + 1),
            (r0 + 1, c0),
            (r1, c1),
            (r1, c1 - 1),
            (r1 - 1, c1),
        }
    if diagonal == DIAG_TR_BL:
        return {
            (r0, c1),
            (r0, c1 - 1),
            (r0 + 1, c1),
            (r1, c0),
            (r1, c0 + 1),
            (r1 - 1, c0),
        }
    raise ValueError(f"unknown diagonal: {diagonal}")


def _validate_color_object(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    assert points, "each object must contain at least one pixel"
    r0, c0, r1, c1 = _bbox(points)
    assert r1 > r0 and c1 > c0, "L-pair bounding boxes must be at least 2x2"

    expected = [
        _l_cells_from_bbox(r0, c0, r1, c1, DIAG_TL_BR),
        _l_cells_from_bbox(r0, c0, r1, c1, DIAG_TR_BL),
    ]
    assert any(points == option for option in expected), (
        "each color must form exactly two opposite-corner L markers"
    )
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each color appears as two opposite-corner L markers. The full rectangle
    between those two corners must be filled, processing larger rectangles
    first and restoring the original L markers at the end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    objects: list[tuple[int, tuple[int, int, int, int], int]] = []

    for color in colors:
        points = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        r0, c0, r1, c1 = _validate_color_object(points)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        objects.append((area, (r0, c0, r1, c1), color))

    out = np.full_like(g, bg)
    for _, (r0, c0, r1, c1), color in sorted(
        objects,
        key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2], item[1][3]),
    ):
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    marker_mask = g != bg
    out[marker_mask] = g[marker_mask]
    return out


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _halo(cells: set[tuple[int, int]], h: int, w: int, gap: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _sample_dimensions(rng: random.Random, h: int, w: int) -> tuple[int, int]:
    max_h = min(RECT_SIDE_MAX, h - 2 * PLACEMENT_MARGIN)
    max_w = min(RECT_SIDE_MAX, w - 2 * PLACEMENT_MARGIN)
    if max_h < 3 or max_w < 3:
        raise ValueError("grid too small for the configured placement margin")

    mode = rng.choices(
        ["regular", "flat_h", "flat_w"],
        weights=[0.7, 0.15, 0.15],
        k=1,
    )[0]
    if mode == "regular":
        rect_h = rng.randint(3, max_h)
        rect_w = rng.randint(3, max_w)
        return rect_h, rect_w
    if mode == "flat_h":
        if max_w >= 4:
            return 2, rng.randint(4, max_w)
        return rng.randint(3, max_h), rng.randint(3, max_w)
    if max_h >= 4:
        return rng.randint(4, max_h), 2
    return rng.randint(3, max_h), rng.randint(3, max_w)


def _sample_rect(
    rng: random.Random,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    used_areas: set[int],
) -> tuple[tuple[int, int, int, int], str, set[tuple[int, int]]] | None:
    for _ in range(RECT_RETRIES):
        rect_h, rect_w = _sample_dimensions(rng, h, w)
        r0 = rng.randint(PLACEMENT_MARGIN, h - rect_h - PLACEMENT_MARGIN)
        c0 = rng.randint(PLACEMENT_MARGIN, w - rect_w - PLACEMENT_MARGIN)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

        diagonal = rng.choice(DIAGONALS)
        cells = _l_cells_from_bbox(*rect, diagonal)
        area = _rect_area(rect)
        if area in used_areas:
            continue
        if area <= len(cells):
            continue
        if any(cell in forbidden for cell in cells):
            continue
        return rect, diagonal, cells
    return None


def _overlap_profile(objects: list[dict[str, object]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for i, obj_a in enumerate(objects):
        rect_a = obj_a["rect"]
        for obj_b in objects[i + 1 :]:
            rect_b = obj_b["rect"]
            counts[_intersection_area(rect_a, rect_b)] += 1
    return counts


def _grid_shape_for_count(rng: random.Random, count: int) -> tuple[int, int]:
    min_side = max(GRID_MIN, 3 * count + 5)
    max_side = min(GRID_MAX, 4 * count + 10)
    return rng.randint(min_side, max_side), rng.randint(min_side, max_side)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*OBJECT_COUNT_RANGE)
    h, w = _grid_shape_for_count(rng, count)
    colors = rng.sample(COLOR_POOL, count)
    overlap_mode = rng.choices(
        ["overlap", "disjoint", "mixed"],
        weights=[0.45, 0.25, 0.30],
        k=1,
    )[0]
    require_both_diagonals = count >= 2 and rng.random() < 0.5

    objects: list[dict[str, object]] = []
    forbidden: set[tuple[int, int]] = set()
    used_areas: set[int] = set()

    for color in colors:
        placed = _sample_rect(rng, h, w, forbidden, used_areas)
        if placed is None:
            return None
        rect, diagonal, cells = placed
        objects.append(
            {
                "color": color,
                "rect": rect,
                "diagonal": diagonal,
                "cells": cells,
            }
        )
        used_areas.add(_rect_area(rect))
        forbidden |= _halo(cells, h, w, MARKER_GAP)

    if require_both_diagonals and len({obj["diagonal"] for obj in objects}) < 2:
        return None

    profile = _overlap_profile(objects)
    max_overlap = max(profile, default=0)
    if overlap_mode == "overlap" and max_overlap < 2:
        return None
    if overlap_mode == "disjoint" and max_overlap > 0:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cells"]:
            grid[r, c] = color

    changed = int(np.count_nonzero(main(grid) != grid))
    if changed < count:
        return None

    non_bg = int(np.count_nonzero(grid != BG_COLOR))
    if non_bg >= grid.size // 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
