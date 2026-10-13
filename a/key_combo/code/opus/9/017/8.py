from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 5

GRID_MIN_MARGIN = 1
GRID_MAX_MARGIN = 4
FRAME_MIN_SIZE = 14
FRAME_MAX_SIZE = 19
OBJECT_COUNT_BAG: list[int] = []
GEN_RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_mask = grid == FRAME
    coords = np.argwhere(frame_mask)
    assert coords.size > 0, "missing gray frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame too small"
    border = _rect_border_mask(grid.shape, box)
    assert np.array_equal(frame_mask, border), "gray pixels must form exactly one rectangular border"
    return box


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    box = _find_frame(g)
    r0, c0, r1, c1 = box

    inside_mask = np.zeros_like(g, dtype=bool)
    inside_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    inside_components = _components((g == RED) & inside_mask)
    assert inside_components, "no red objects inside frame"

    objects = []
    for idx, comp in enumerate(inside_components):
        bbox = _bbox_from_cells(comp)
        objects.append(
            {
                "id": idx,
                "cells": tuple(sorted(comp)),
                "bbox": bbox,
                "area": len(comp),
            }
        )

    outside_mask = g == RED
    outside_mask[r0 + 1 : r1, c0 + 1 : c1] = False
    distractors = _components(outside_mask)

    return {"grid": g, "box": box, "objects": objects, "distractors": distractors}


def _selected_ids(objects: list[dict]) -> set[int]:
    ordered = sorted(objects, key=lambda obj: obj["area"])
    areas = [obj["area"] for obj in ordered]
    assert len(set(areas)) == len(areas), "inside-object areas must be unique"
    start = 0 if len(ordered) % 2 == 1 else 1
    return {ordered[i]["id"] for i in range(start, len(ordered), 2)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor alternating area-ranked red objects inside the gray frame, with the starting phase set by the object-count parity.

    Input structure:
    - Black background.
    - One one-cell-thick gray rectangular frame.
    - Strictly inside the frame: 5 to 7 red 4-connected objects whose pixel areas are all distinct.
    - Outside the frame: additional red distractor pixels or tiny red pieces.

    Rule:
    - Detect the gray frame and ignore everything outside its interior when collecting target objects.
    - Extract only the red 4-connected objects strictly inside the frame.
    - Sort those interior objects by area from small to large and let n be their count.
    - If n is odd, recolor sorted indices 0, 2, 4, ... blue; if n is even, recolor sorted indices 1, 3, 5, ... blue.
    - Leave the unselected red objects, the gray frame, the outside distractors, and the background unchanged.

    Key insight:
    - The frame isolates the only relevant objects, and the alternating selection does not start from a fixed position: counting the interior objects reveals whether the area order starts selecting at index 0 or index 1.
    """
    parsed = _parse_grid(grid)
    out = parsed["grid"].copy()
    chosen = _selected_ids(parsed["objects"])
    for obj in parsed["objects"]:
        if obj["id"] in chosen:
            for r, c in obj["cells"]:
                out[r, c] = BLUE
    return out


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(cells: tuple[tuple[int, int], ...], rotation: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotation % 4):
        pts = [(-c, r) for r, c in pts]
    return _normalize_shape(tuple(pts))


SHAPE_LIBRARY = {
    2: [((0, 0), (0, 1))],
    3: [((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (0, 2))],
    4: [
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ],
    5: [
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2)),
    ],
    7: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (3, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1)),
    ],
    8: [
        ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1)),
    ],
}


def _random_shape(area: int) -> tuple[tuple[int, int], ...]:
    base = random.choice(SHAPE_LIBRARY[area])
    return _transform_shape(base, rotation=random.randrange(4), flip=bool(random.randrange(2)))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _choose_object_count() -> int:
    global OBJECT_COUNT_BAG
    if not OBJECT_COUNT_BAG:
        OBJECT_COUNT_BAG = [5, 6]
        random.shuffle(OBJECT_COUNT_BAG)
    return OBJECT_COUNT_BAG.pop()


def _selected_area_set(areas: tuple[int, ...]) -> set[int]:
    start = 0 if len(areas) % 2 == 1 else 1
    return {areas[i] for i in range(start, len(areas), 2)}


def _area_subset_ok(areas: tuple[int, ...]) -> bool:
    selected = _selected_area_set(areas)
    return selected != {a for a in areas if a % 2 == 0} and selected != {a for a in areas if a % 2 == 1}


VALID_AREA_SUBSETS = {
    n: [combo for combo in itertools.combinations(range(2, 9), n) if _area_subset_ok(combo)]
    for n in (5, 6)
}


def _choose_areas(n: int) -> list[int]:
    return list(random.choice(VALID_AREA_SUBSETS[n]))


def _place_inside_objects(grid: np.ndarray, box: tuple[int, int, int, int], areas: list[int]) -> bool:
    r0, c0, r1, c1 = box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    blocked = np.zeros((inner_h, inner_w), dtype=bool)

    for area in sorted(areas, reverse=True):
        shape = _random_shape(area)
        sh, sw = _shape_dims(shape)
        placements = []
        for base_r in range(inner_h - sh + 1):
            for base_c in range(inner_w - sw + 1):
                coords = [(base_r + dr, base_c + dc) for dr, dc in shape]
                if all(not blocked[rr, cc] for rr, cc in coords):
                    placements.append(coords)
        if not placements:
            return False
        coords = random.choice(placements)
        for rr, cc in coords:
            grid[r0 + 1 + rr, c0 + 1 + cc] = RED
        for rr, cc in coords:
            rr0 = max(0, rr - 1)
            rr1 = min(inner_h, rr + 2)
            cc0 = max(0, cc - 1)
            cc1 = min(inner_w, cc + 2)
            blocked[rr0:rr1, cc0:cc1] = True
    return True


def _place_outside_distractors(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    count = random.randint(3, 6)
    for _ in range(count):
        area = 1 if random.random() < 0.55 else 2
        placements = []
        if area == 1:
            for r in range(h):
                for c in range(w):
                    if not blocked[r, c]:
                        placements.append([(r, c)])
        else:
            for r in range(h):
                for c in range(w):
                    for dr, dc in ((0, 1), (1, 0)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not blocked[r, c] and not blocked[nr, nc]:
                            placements.append([(r, c), (nr, nc)])
        if not placements:
            return False
        cells = random.choice(placements)
        for r, c in cells:
            grid[r, c] = RED
        for r, c in cells:
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            blocked[rr0:rr1, cc0:cc1] = True
    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    out = parsed["grid"].copy()
    chosen = _selected_ids(parsed["objects"])
    for obj in parsed["objects"]:
        color = BLUE if obj["id"] in chosen else RED
        for r, c in obj["cells"]:
            out[r, c] = color
    return out


def _shortcut_free(parsed: dict) -> bool:
    objects = parsed["objects"]
    n = len(objects)
    if n not in (5, 6):
        return False

    by_area = sorted(objects, key=lambda obj: obj["area"])
    area_selected = _selected_ids(objects)
    by_left = sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["area"]))
    by_top = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["area"]))
    start = 0 if n % 2 == 1 else 1
    left_selected = {by_left[i]["id"] for i in range(start, n, 2)}
    top_selected = {by_top[i]["id"] for i in range(start, n, 2)}
    if area_selected == left_selected or area_selected == top_selected:
        return False

    areas = tuple(obj["area"] for obj in by_area)
    if not _area_subset_ok(areas):
        return False

    if [obj["id"] for obj in by_area] == [obj["id"] for obj in by_left]:
        return False
    if [obj["id"] for obj in by_area] == [obj["id"] for obj in by_top]:
        return False
    return True


def _sample_semantics_ok(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_grid(grid)
    except AssertionError:
        return False

    objects = parsed["objects"]
    if len(objects) not in (5, 6):
        return False

    areas = [obj["area"] for obj in objects]
    if sorted(areas) != sorted(set(areas)):
        return False
    if any(area < 2 or area > 8 for area in areas):
        return False

    if len(parsed["distractors"]) < 2:
        return False
    if any(len(comp) > 2 for comp in parsed["distractors"]):
        return False

    box = parsed["box"]
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 12 or c1 - c0 + 1 < 12:
        return False

    if not _shortcut_free(parsed):
        return False

    expected = _expected_output(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _attempt() -> np.ndarray | None:
    frame_h = random.randint(FRAME_MIN_SIZE, FRAME_MAX_SIZE)
    frame_w = random.randint(FRAME_MIN_SIZE, FRAME_MAX_SIZE)
    top = random.randint(GRID_MIN_MARGIN, GRID_MAX_MARGIN)
    bottom = random.randint(GRID_MIN_MARGIN, GRID_MAX_MARGIN)
    left = random.randint(GRID_MIN_MARGIN, GRID_MAX_MARGIN)
    right = random.randint(GRID_MIN_MARGIN, GRID_MAX_MARGIN)

    h = frame_h + top + bottom
    w = frame_w + left + right
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    box = (top, left, top + frame_h - 1, left + frame_w - 1)
    border = _rect_border_mask(grid.shape, box)
    grid[border] = FRAME

    areas = _choose_areas(_choose_object_count())
    if not _place_inside_objects(grid, box, areas):
        return None
    if not _place_outside_distractors(grid, box):
        return None
    if not _sample_semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
