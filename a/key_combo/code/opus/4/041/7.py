from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
IN_COLOR = Color.RED
OUT_COLOR = Color.PINK
AREA_THRESHOLD = 4
GEN_MAX_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
OBJECT_COUNT_RANGE = (5, 7)
PAIR_COUNT_PROB = 0.4
PLACEMENT_ATTEMPTS = 200

_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_CONTAINER_TEMPLATES = [
    {
        "outer": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        "inners": [((0, 2),)],
    },
    {
        "outer": ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
        "inners": [((0, 2),), ((0, 2), (1, 2))],
    },
]


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_cells(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], rot: int, flip: bool
) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(int(rot) % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_cells(pts)


def _translate_cells(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r) + int(dr), int(c) + int(dc)) for r, c in cells)


def _bbox_from_cells(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
        and outer != inner
    )


def _can_place(grid: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BG:
            return False
        for dr, dc in _NEIGHBORS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == IN_COLOR and (rr, cc) not in cell_set:
                return False
    return True


def _paint(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _random_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    if int(area) == 1:
        return ((0, 0),)
    max_side = 4
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(random.choice(frontier))
        shape = _normalize_cells(tuple(cells))
        bbox = _bbox_from_cells(shape)
        if bbox[2] + 1 <= max_side and bbox[3] + 1 <= max_side:
            rot = random.randrange(4)
            flip = random.random() < 0.5
            return _transform_cells(shape, rot=rot, flip=flip)
    return tuple((0, c) for c in range(int(area)))


def _place_pair(
    grid: np.ndarray,
    objects: list[dict],
    blocked_boxes: list[tuple[int, int, int, int]],
    pair_id: int,
) -> bool:
    template = random.choice(_CONTAINER_TEMPLATES)
    rot = random.randrange(4)
    flip = random.random() < 0.5
    outer_rel = _transform_cells(template["outer"], rot=rot, flip=flip)
    inner_rel = _transform_cells(random.choice(template["inners"]), rot=rot, flip=flip)
    pair_rel = outer_rel + inner_rel
    outer_rel_bbox = _bbox_from_cells(outer_rel)
    pair_h = outer_rel_bbox[2] + 1
    pair_w = outer_rel_bbox[3] + 1
    H, W = grid.shape

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(0, H - pair_h)
        left = random.randint(0, W - pair_w)
        outer = _translate_cells(outer_rel, top, left)
        inner = _translate_cells(inner_rel, top, left)
        outer_bbox = _bbox_from_cells(outer)
        if any(_boxes_intersect(_expand_bbox(outer_bbox, 1), b) for b in blocked_boxes):
            continue
        if not _can_place(grid, _translate_cells(pair_rel, top, left)):
            continue
        _paint(grid, outer, IN_COLOR)
        _paint(grid, inner, IN_COLOR)
        objects.append({"kind": "outer", "pair_id": int(pair_id), "cells": outer, "bbox": outer_bbox, "area": len(outer)})
        objects.append(
            {
                "kind": "inner",
                "pair_id": int(pair_id),
                "cells": inner,
                "bbox": _bbox_from_cells(inner),
                "area": len(inner),
            }
        )
        blocked_boxes.append(_expand_bbox(outer_bbox, 1))
        return True
    return False


def _place_free_object(
    grid: np.ndarray,
    area: int,
    objects: list[dict],
    blocked_boxes: list[tuple[int, int, int, int]],
) -> bool:
    H, W = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _random_polyomino(int(area))
        bbox = _bbox_from_cells(shape)
        h = bbox[2] + 1
        w = bbox[3] + 1
        if h > H or w > W:
            continue
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        cells = _translate_cells(shape, top, left)
        bbox_abs = _bbox_from_cells(cells)
        if any(_boxes_intersect(bbox_abs, blocked) for blocked in blocked_boxes):
            continue
        if any(_bbox_strict_contains(bbox_abs, obj["bbox"]) or _bbox_strict_contains(obj["bbox"], bbox_abs) for obj in objects):
            continue
        if not _can_place(grid, cells):
            continue
        _paint(grid, cells, IN_COLOR)
        objects.append({"kind": "free", "pair_id": None, "cells": cells, "bbox": bbox_abs, "area": len(cells)})
        return True
    return False


def _validate_layout(grid: np.ndarray, objects: list[dict], pair_count: int) -> bool:
    if not np.any(grid == IN_COLOR):
        return False
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[IN_COLOR],
        return_format="dict",
    )
    if not (OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]):
        return False
    designated = {(obj["kind"], obj["pair_id"], obj["bbox"]) for obj in objects if obj["kind"] != "free"}
    if len(designated) != pair_count * 2:
        return False
    for obj in objects:
        if obj["kind"] != "inner":
            continue
        outers = [other for other in objects if other["kind"] == "outer" and other["pair_id"] == obj["pair_id"]]
        if len(outers) != 1 or not _bbox_strict_contains(outers[0]["bbox"], obj["bbox"]):
            return False
    for i, obj in enumerate(objects):
        for j, other in enumerate(objects):
            if i >= j:
                continue
            same_pair = obj["pair_id"] is not None and obj["pair_id"] == other["pair_id"]
            allowed = same_pair and {obj["kind"], other["kind"]} == {"outer", "inner"}
            if not allowed and (
                _bbox_strict_contains(obj["bbox"], other["bbox"]) or _bbox_strict_contains(other["bbox"], obj["bbox"])
            ):
                return False

    out = main(grid)
    if not np.any(out == OUT_COLOR):
        return False
    if not np.any(out == IN_COLOR):
        return False
    erased = int(np.count_nonzero((grid == IN_COLOR) & (out == BG)))
    if erased < pair_count:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor large red objects to pink, then erase contained objects that became different-colored from their containers.

    Input structure:
    - The grid has a black background and several disconnected red 4-connected foreground objects.
    - Object areas range from tiny single pixels up to medium shapes of a few cells.
    - Some small objects have bounding boxes fully inside the bounding boxes of larger red objects.

    Rule:
    - Find every red 4-connected object and measure its area.
    - Recolor each object of area at least four from red to pink; smaller red objects stay red.
    - After recoloring, erase any foreground object whose bounding box lies fully inside the bounding box of another foreground object of the other color.
    - If multiple differently colored containers contain the same object, one is enough to erase it; erase targets in increasing area order without repeating the test after deletions.

    Key insight:
    - The decisive containment pairs are initially safe because every object starts red; the area-threshold recoloring creates the color mismatch that makes the inner objects removable.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[IN_COLOR],
        return_format="dict",
    )
    recolored: list[dict] = []
    for obj in objects:
        new_color = OUT_COLOR if int(obj["area"]) >= AREA_THRESHOLD else IN_COLOR
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
        recolored.append(
            {
                "coords": coords,
                "bbox": obj["bbox"],
                "area": int(obj["area"]),
                "color": int(new_color),
            }
        )

    to_erase: list[int] = []
    for idx, obj in enumerate(recolored):
        for jdx, other in enumerate(recolored):
            if idx == jdx or int(obj["color"]) == int(other["color"]):
                continue
            if _bbox_strict_contains(other["bbox"], obj["bbox"]):
                to_erase.append(int(idx))
                break

    for idx in sorted(set(to_erase), key=lambda k: (recolored[k]["area"], recolored[k]["bbox"])):
        coords = recolored[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        H = random.randint(*GRID_HEIGHT_RANGE)
        W = random.randint(*GRID_WIDTH_RANGE)
        grid = np.full((H, W), BG, dtype=int)
        target_objects = random.randint(*OBJECT_COUNT_RANGE)
        pair_count = 2 if target_objects >= 6 and random.random() < PAIR_COUNT_PROB else 1

        objects: list[dict] = []
        blocked_boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for pair_id in range(pair_count):
            if not _place_pair(grid, objects, blocked_boxes, pair_id):
                ok = False
                break
        if not ok:
            continue

        remaining = target_objects - 2 * pair_count
        free_areas: list[int] = []
        if remaining >= 1:
            free_areas.append(random.randint(1, 3))
        if remaining >= 2:
            free_areas.append(random.randint(4, 6))
        while len(free_areas) < remaining:
            free_areas.append(random.randint(1, 6))
        random.shuffle(free_areas)

        for area in free_areas:
            if not _place_free_object(grid, area, objects, blocked_boxes):
                ok = False
                break
        if not ok:
            continue

        if _validate_layout(grid, objects, pair_count):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
