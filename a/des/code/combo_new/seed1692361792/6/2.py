from __future__ import annotations

import math
from collections import deque

import numpy as np


BG_COLOR = 8
MAX_GENERATION_ATTEMPTS = 400
PAIR_MASK_ATTEMPTS = 200
PAIR_COUNT_RANGE = (1, 3)
RECT_DIMS = [(h, w) for h in range(3, 6) for w in range(3, 7) if 9 <= h * w <= 20]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_to_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.argwhere(mask)}


def _is_connected(mask: np.ndarray) -> bool:
    cells = _mask_to_cells(mask)
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _grow_connected_mask(h: int, w: int, area: int, rng: np.random.Generator) -> np.ndarray:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pick = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(pick)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _apply_pair_transform(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot_k))
    return np.fliplr(out) if flip_lr else out


def _build_object(mask: np.ndarray, color: int) -> dict:
    mask = _crop_mask(mask)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    rel_cells = {tuple(map(int, coord)) for coord in coords}
    return {
        "color": int(color),
        "mask": mask,
        "cells": rel_cells,
        "area": int(mask.sum()),
        "height": int(r1 - r0 + 1),
        "width": int(c1 - c0 + 1),
    }


def _complements_to_rectangle(obj_a: dict, obj_b: dict) -> bool:
    cells_a = obj_a["cells"]
    cells_b = obj_b["cells"]
    ha, wa = int(obj_a["height"]), int(obj_a["width"])
    hb, wb = int(obj_b["height"]), int(obj_b["width"])
    area_sum = int(obj_a["area"]) + int(obj_b["area"])

    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted_b = {(r + dr, c + dc) for r, c in cells_b}
            if cells_a & shifted_b:
                continue
            top = min(0, dr)
            left = min(0, dc)
            bottom = max(ha - 1, dr + hb - 1)
            right = max(wa - 1, dc + wb - 1)
            bbox_area = (bottom - top + 1) * (right - left + 1)
            if bbox_area == area_sum:
                return True
    return False


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG_COLOR:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rel_cells = {(rr - r0, cc - c0) for rr, cc in coords}
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in rel_cells:
                mask[int(rr), int(cc)] = True
            objects.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "cells": rel_cells,
                    "area": len(coords),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "mask": mask,
                }
            )

    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors))
    assert len(objects) % 2 == 0 and objects
    return objects


def _find_unique_pairs(objects: list[dict]) -> list[tuple[int, int]]:
    n = len(objects)
    edges = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _complements_to_rectangle(objects[i], objects[j]):
                edges[i].append(j)
                edges[j].append(i)

    solutions: list[list[tuple[int, int]]] = []

    def search(unpaired: set[int], current: list[tuple[int, int]]) -> None:
        if len(solutions) >= 2:
            return
        if not unpaired:
            solutions.append(sorted(current))
            return
        pivot = min(unpaired, key=lambda idx: len([j for j in edges[idx] if j in unpaired]))
        choices = sorted(j for j in edges[pivot] if j in unpaired)
        for partner in choices:
            current.append((pivot, partner))
            search(unpaired - {pivot, partner}, current)
            current.pop()

    search(set(range(n)), [])
    assert len(solutions) == 1

    for i, j in solutions[0]:
        assert objects[i]["area"] != objects[j]["area"]
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments and repaint the smaller one with its larger partner's color.

    Input structure:
    - The background is azure.
    - The non-background cells form an even number of 4-connected monochrome objects, each in a unique color.
    - The objects can be partitioned into pairs such that translating the two shapes, without rotation or reflection, makes each pair fill a solid rectangle exactly.

    Rule:
    - Extract every monochrome object and compare shapes pairwise by translation only.
    - Two objects are partners when some relative offset makes their occupied cells disjoint and turns their union into a full rectangle.
    - Within each partner pair, compare areas and repaint every cell of the smaller object to the larger object's color.
    - Leave the larger object, all other pairs, and the azure background unchanged.

    Key insight:
    - The hard part is inferring an implied rectangle from two scattered fragments; once the geometric pairing is known, recoloring is a separate area comparison.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)
    pairs = _find_unique_pairs(objects)

    for i, j in pairs:
        obj_a = objects[i]
        obj_b = objects[j]
        larger = obj_a if int(obj_a["area"]) > int(obj_b["area"]) else obj_b
        smaller = obj_b if larger is obj_a else obj_a
        coords = smaller["coords"]
        out[coords[:, 0], coords[:, 1]] = int(larger["color"])

    return out


def _generate_pair_masks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_MASK_ATTEMPTS):
        rect_h, rect_w = RECT_DIMS[int(rng.integers(0, len(RECT_DIMS)))]
        total_area = rect_h * rect_w
        split = int(rng.integers(2, total_area - 1))
        if split * 2 == total_area:
            continue

        mask_a = _grow_connected_mask(rect_h, rect_w, split, rng)
        mask_b = ~mask_a
        if not _is_connected(mask_b):
            continue

        rot_k = int(rng.integers(0, 4))
        flip_lr = bool(rng.integers(0, 2))
        mask_a = _crop_mask(_apply_pair_transform(mask_a, rot_k, flip_lr))
        mask_b = _crop_mask(_apply_pair_transform(mask_b, rot_k, flip_lr))

        if min(mask_a.shape) < 2 or min(mask_b.shape) < 2:
            continue
        if bool(mask_a.all()) or bool(mask_b.all()):
            continue
        if int(mask_a.sum()) == int(mask_b.sum()):
            continue
        return mask_a, mask_b
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_objects(
    grid: np.ndarray, objects: list[dict], rng: np.random.Generator
) -> list[dict] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: int(objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1]),
        reverse=True,
    )

    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        ph, pw = mask.shape
        blocked = _dilate8(occupied)
        candidates = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                continue
            chosen = (int(r0), int(c0))
            break
        if chosen is None:
            return None

        r0, c0 = chosen
        coords = np.argwhere(mask)
        abs_coords = coords + np.array([r0, c0], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = int(obj["color"])
        occupied[abs_coords[:, 0], abs_coords[:, 1]] = True
        placed[idx] = {
            **obj,
            "coords": abs_coords,
        }

    return [obj for obj in placed if obj is not None]


def _expected_output(grid: np.ndarray, objects: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for i, j in pairs:
        obj_a = objects[i]
        obj_b = objects[j]
        larger = obj_a if int(obj_a["area"]) > int(obj_b["area"]) else obj_b
        smaller = obj_b if larger is obj_a else obj_a
        coords = smaller["coords"]
        out[coords[:, 0], coords[:, 1]] = int(larger["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    colors = rng.choice([c for c in range(10) if c != BG_COLOR], size=pair_count * 2, replace=False)

    objects: list[dict] = []
    intended_pairs: list[tuple[int, int]] = []
    color_index = 0
    for _ in range(pair_count):
        pair_masks = _generate_pair_masks(rng)
        if pair_masks is None:
            return None
        mask_a, mask_b = pair_masks
        if bool(rng.integers(0, 2)):
            mask_a, mask_b = mask_b, mask_a
        obj_a = _build_object(mask_a, int(colors[color_index]))
        obj_b = _build_object(mask_b, int(colors[color_index + 1]))
        intended_pairs.append((len(objects), len(objects) + 1))
        objects.extend([obj_a, obj_b])
        color_index += 2

    try:
        graph_pairs = _find_unique_pairs(objects)
    except AssertionError:
        return None
    if sorted(graph_pairs) != sorted(intended_pairs):
        return None

    total_box_area = sum(int(obj["mask"].shape[0] * obj["mask"].shape[1]) for obj in objects)
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)
    side_est = int(math.ceil(math.sqrt(total_box_area * 2.4)))
    row_lo = max(max_h + 2, side_est - 2)
    col_lo = max(max_w + 2, side_est - 2)
    row_hi = min(30, side_est + 6)
    col_hi = min(30, side_est + 8)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    rows = int(rng.integers(row_lo, row_hi + 1))
    cols = int(rng.integers(col_lo, col_hi + 1))
    grid = np.full((rows, cols), BG_COLOR, dtype=int)

    placed = _place_objects(grid, objects, rng)
    if placed is None:
        return None

    expected = _expected_output(grid, placed, intended_pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
