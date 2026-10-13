from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
ORANGE = 7
GREY = 5
PALETTE = np.array([2, 4, 8, 9], dtype=int)

FRAME_COLOR_POOL = [1, 3, 6, 8]
GRID_HEIGHT_RANGE = (18, 25)
GRID_WIDTH_RANGE = (18, 25)
FRAME_COUNT_CHOICES = [2, 2, 3, 3, 4]
FRAME_HEIGHT_RANGE = (5, 9)
FRAME_WIDTH_RANGE = (5, 10)
DISTRACTOR_COUNT_RANGE = (3, 8)
GENERATION_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _box_border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _box_interior_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _box_all_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _point_in_strict_interior(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = []

    for color in _colors_present(g, exclude={bg}):
        color_mask = g == color
        box = _bbox(color_mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _rect_border_mask(g.shape, box)
        if not np.array_equal(color_mask, border):
            continue

        markers = [
            (rr, cc)
            for rr, cc in np.argwhere(g == ORANGE)
            if _point_in_strict_interior(box, int(rr), int(cc))
        ]
        assert len(markers) == 1, "each frame must contain exactly one orange marker"
        marker = tuple(map(int, markers[0]))

        frames.append(
            {
                "color": int(color),
                "box": box,
                "border": border,
                "border_cells": _box_border_cells(box),
                "interior_cells": _box_interior_cells(box),
                "marker": marker,
            }
        )

    frames.sort(key=lambda frame: frame["color"])
    assert 2 <= len(frames) <= 4, "expected 2 to 4 rectangular frames"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Color each frame interior with marker-centered Manhattan rings, resolving overlaps by the nearest frame marker.

    Input structure:
    - A black background contains 2 to 4 complete one-pixel-thick hollow rectangular frames in distinct colors.
    - Each frame has exactly one orange marker strictly inside it, and some frame interiors overlap.
    - Grey singleton pixels may appear outside every frame interior as distractors.

    Rule:
    - Keep every frame border pixel unchanged.
    - For each non-border pixel that lies inside one or more frame interiors, compute its Manhattan distance to each containing frame's orange marker.
    - Use the smallest such distance to index the cyclic palette red, yellow, teal, maroon; if distances tie, prefer the smaller frame color id.
    - Recolor marker pixels as distance 0, and leave all pixels outside every frame interior unchanged.

    Key insight:
    - Each rectangle is both an object and the domain boundary of its own distance field, so overlapping interiors create a bounded multi-source Voronoi competition instead of a full-grid ring pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    out = g.copy()

    border_union = np.zeros_like(g, dtype=bool)
    for frame in frames:
        border_union |= frame["border"]

    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if border_union[r, c]:
                continue

            contenders = []
            for frame in frames:
                if _point_in_strict_interior(frame["box"], r, c):
                    mr, mc = frame["marker"]
                    dist = abs(r - mr) + abs(c - mc)
                    contenders.append((int(dist), int(frame["color"])))

            if not contenders:
                continue

            best_dist, _best_color = min(contenders)
            out[r, c] = int(PALETTE[best_dist % len(PALETTE)])

    return out


def _overlap_box(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    r0 = max(ar0 + 1, br0 + 1)
    c0 = max(ac0 + 1, bc0 + 1)
    r1 = min(ar1 - 1, br1 - 1)
    c1 = min(ac1 - 1, bc1 - 1)
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _make_overlapping_pair(
    H: int, W: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]] | None:
    outer_h = random.randint(max(7, FRAME_HEIGHT_RANGE[0] + 2), FRAME_HEIGHT_RANGE[1])
    outer_w = random.randint(max(7, FRAME_WIDTH_RANGE[0] + 2), FRAME_WIDTH_RANGE[1])
    inner_h = random.randint(FRAME_HEIGHT_RANGE[0], outer_h - 2)
    inner_w = random.randint(FRAME_WIDTH_RANGE[0], outer_w - 2)

    outer_r0 = random.randint(0, H - outer_h)
    outer_c0 = random.randint(0, W - outer_w)
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)

    inner_r0_min = outer_r0 + 1
    inner_r0_max = outer[2] - inner_h
    inner_c0_min = outer_c0 + 1
    inner_c0_max = outer[3] - inner_w
    if inner_r0_min > inner_r0_max or inner_c0_min > inner_c0_max:
        return None

    inner_r0 = random.randint(inner_r0_min, inner_r0_max)
    inner_c0 = random.randint(inner_c0_min, inner_c0_max)
    inner = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)

    marker = (
        random.randint(inner[0] + 1, inner[2] - 1),
        random.randint(inner[1] + 1, inner[3] - 1),
    )
    return outer, inner, marker


def _random_box(H: int, W: int) -> tuple[int, int, int, int]:
    h = random.randint(*FRAME_HEIGHT_RANGE)
    w = random.randint(*FRAME_WIDTH_RANGE)
    r0 = random.randint(0, H - h)
    c0 = random.randint(0, W - w)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _overlap_cells(rects: list[dict]) -> set[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for rect in rects:
        counts.update(rect["interior_cells"])
    return {cell for cell, n in counts.items() if n >= 2}


def _isolated_distractor_cells(
    H: int,
    W: int,
    forbidden: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = []
    for r in range(H):
        for c in range(W):
            if (r, c) in forbidden or (r, c) in occupied:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(H, r + 2)):
                for cc in range(max(0, c - 1), min(W, c + 2)):
                    if (rr, cc) in occupied:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _place_distractors(
    grid: np.ndarray,
    count: int,
    forbidden: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for _ in range(count):
        candidates = _isolated_distractor_cells(H, W, forbidden, occupied)
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = GREY
        occupied.add((r, c))
    return True


def _build_candidate() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    frame_colors = random.sample(FRAME_COLOR_POOL, frame_count)

    pair = _make_overlapping_pair(H, W)
    if pair is None:
        return None

    shared_marker = pair[2]
    boxes = [pair[0], pair[1]]
    occupied_space = _box_all_cells(pair[0]) | _box_all_cells(pair[1])
    for _ in range(frame_count - 2):
        added = False
        for _try in range(80):
            box = _random_box(H, W)
            if any(_box_border_cells(box) & _box_border_cells(existing) for existing in boxes):
                continue
            if _box_all_cells(box) & occupied_space:
                continue
            boxes.append(box)
            occupied_space |= _box_all_cells(box)
            added = True
            break
        if not added:
            return None

    rects = []
    for color, box in zip(frame_colors, boxes):
        rects.append(
            {
                "color": int(color),
                "box": box,
                "border_cells": _box_border_cells(box),
                "interior_cells": _box_interior_cells(box),
            }
        )

    if not _overlap_cells(rects):
        return None

    rects[0]["marker"] = shared_marker
    rects[1]["marker"] = shared_marker
    for rect in rects[2:]:
        rect["marker"] = random.choice(sorted(rect["interior_cells"]))

    grid = np.full((H, W), BG, dtype=int)
    occupied = set()
    interior_union = set()
    for rect in rects:
        interior_union |= rect["interior_cells"]
        r0, c0, r1, c1 = rect["box"]
        color = rect["color"]
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        occupied |= rect["border_cells"]

    for rect in rects:
        mr, mc = rect["marker"]
        grid[mr, mc] = ORANGE
        occupied.add((mr, mc))

    forbidden = interior_union | occupied
    need = random.randint(*DISTRACTOR_COUNT_RANGE)
    if not _place_distractors(grid, need, forbidden, occupied):
        return None

    if _get_bg(grid) != BG:
        return None

    try:
        frames = _find_frames(grid)
        out = main(grid)
    except Exception:
        return None

    if len(frames) != frame_count:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    overlap_nonborder = _overlap_cells(rects) - set().union(*(rect["border_cells"] for rect in rects))
    if not overlap_nonborder:
        return None
    if not any(out[r, c] != grid[r, c] for r, c in overlap_nonborder):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
