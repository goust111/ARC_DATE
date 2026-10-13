from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
GROUND = 5
BLOCKER = 7
OBJECT_COLORS = (2, 3, 4, 6)
MAX_GEN_ATTEMPTS = 400
MAX_BLOCKER_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _enumerate_shapes() -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {2: [], 3: []}
    for h in (2, 3):
        seen = set()
        for w in (1, 2, 3):
            total = h * w
            for bits in range(1 << total):
                area = bits.bit_count()
                if not (4 <= area <= 6):
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(total):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                if not np.all(mask.any(axis=1)):
                    continue
                if not np.all(mask.any(axis=0)):
                    continue
                if not _is_connected(mask):
                    continue
                key = tuple(tuple(int(v) for v in row) for row in mask)
                if key in seen:
                    continue
                seen.add(key)
                out[h].append(mask.copy())
    return out


SHAPES_BY_HEIGHT = _enumerate_shapes()


def _component_from(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> dict:
    sr, sc = start
    color = int(grid[sr, sc])
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return {
        "color": color,
        "top": r0,
        "bottom": r1,
        "left": c0,
        "right": c1,
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "area": len(cells),
        "mask": mask,
        "cells": cells,
    }


def _parse_objects(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1]):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color in (BACKGROUND, GROUND, BLOCKER):
                continue
            obj = _component_from(grid, (r, c), seen)
            assert obj["left"] == 0, "each object must touch the left wall"
            assert obj["height"] in (2, 3), "each object must span 2-3 rows"
            assert 1 <= obj["width"] <= 3, "each object must fit inside 1-3 columns"
            assert 4 <= obj["area"] <= 6, "each object must have 4-6 pixels"
            assert np.all(obj["mask"].any(axis=1)), "every object row must be occupied"
            objects.append(obj)
    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors)), "object colors must be distinct"
    objects.sort(key=lambda obj: obj["top"])
    return objects


def _sheared_coords(mask: np.ndarray, top: int, width: int) -> list[tuple[int, int]]:
    coords = []
    bbox_w = mask.shape[1]
    for rel_r in range(mask.shape[0]):
        inward = max(0, rel_r - 1)
        start_c = width - bbox_w - inward
        row_cols = np.flatnonzero(mask[rel_r])
        for rel_c in row_cols:
            coords.append((top + rel_r, start_c + int(rel_c)))
    return coords


def _right_aligned_coords(mask: np.ndarray, top: int, width: int) -> list[tuple[int, int]]:
    coords = []
    start_c = width - mask.shape[1]
    for rel_r in range(mask.shape[0]):
        row_cols = np.flatnonzero(mask[rel_r])
        for rel_c in row_cols:
            coords.append((top + rel_r, start_c + int(rel_c)))
    return coords


def _forbidden_mask(shape: tuple[int, int], blockers: list[tuple[int, int]]) -> np.ndarray:
    forbidden = np.zeros(shape, dtype=bool)
    for r, c in blockers:
        forbidden[r, c] = True
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert np.all(g[-1] == GROUND), "bottom row must be the ground row"

    blockers = [tuple(map(int, p)) for p in np.argwhere(g == BLOCKER)]
    assert 2 <= len(blockers) <= 3, "expected 2-3 blocker pixels"
    for br, bc in blockers:
        assert br < h - 1, "blockers must sit above the ground row"
    for (r1, c1), (r2, c2) in combinations(blockers, 2):
        assert max(abs(r1 - r2), abs(c1 - c2)) > 1, "blockers must be isolated"

    objects = _parse_objects(g)
    assert 3 <= len(objects) <= 4, "expected 3-4 moving objects"

    forbidden = _forbidden_mask(g.shape, blockers)
    for obj in objects:
        obj["transformed"] = _sheared_coords(obj["mask"], obj["top"], w)
        obj["unsheared"] = _right_aligned_coords(obj["mask"], obj["top"], w)
        for r, c in obj["transformed"]:
            assert 0 <= r < h - 1 and 0 <= c < w, "sheared object must stay in bounds"
        obj["safe_area"] = sum(1 for r, c in obj["transformed"] if not forbidden[r, c])
        obj["unsheared_safe_area"] = sum(1 for r, c in obj["unsheared"] if not forbidden[r, c])
    return objects, blockers, forbidden


def _render_stacked(
    shape: tuple[int, int], objects: list[dict], blockers: list[tuple[int, int]]
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BACKGROUND, dtype=int)
    out[-1, :] = GROUND
    for r, c in blockers:
        out[r, c] = BLOCKER

    current_bottom = h - 2
    for obj in sorted(objects, key=lambda item: (item["safe_area"], item["top"])):
        new_top = current_bottom - obj["height"] + 1
        assert new_top >= 0, "stacked objects must fit above the ground"
        for r, c in _sheared_coords(obj["mask"], new_top, w):
            assert out[r, c] == BACKGROUND, "stacked objects must not overlap blockers or each other"
            out[r, c] = obj["color"]
        current_bottom = new_top - 1
    return out


def _stack_with_order(
    shape: tuple[int, int], objects: list[dict], blockers: list[tuple[int, int]], order: list[int]
) -> np.ndarray:
    h, w = shape
    by_id = {obj["id"]: obj for obj in objects}
    out = np.full((h, w), BACKGROUND, dtype=int)
    out[-1, :] = GROUND
    for r, c in blockers:
        out[r, c] = BLOCKER
    current_bottom = h - 2
    for obj_id in order:
        obj = by_id[obj_id]
        new_top = current_bottom - obj["height"] + 1
        if new_top < 0:
            return out
        for r, c in _sheared_coords(obj["mask"], new_top, w):
            if out[r, c] != BACKGROUND:
                return out
            out[r, c] = obj["color"]
        current_bottom = new_top - 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move the left-wall objects to the right by a row-wise shear, rank them by post-shear safe area against blocker halos, then restack them above the ground.

    Input structure:
    - A black background contains a full grey ground row on the bottom, 3-4 distinct monochrome 4-connected objects attached to the left wall, and 2-3 orange singleton blockers in the right half.
    - Each object occupies 2-3 rows, 1-3 columns, and 4-6 pixels total; the blockers are isolated from each other and from the ground.

    Rule:
    - For each left-wall object, ignore its source position and compute its right-wall sheared placement: the first two object rows stay right-aligned, and each lower row shifts one extra cell left.
    - Mark every orange blocker and its full 8-neighborhood as forbidden, then count for each sheared object how many of its pixels lie outside the forbidden cells; this count is its safe area.
    - Clear all original object pixels, sort the objects by safe area ascending with original top row as the tie-breaker, and stack the sheared shapes consecutively upward from just above the ground while keeping the grey ground row and orange blockers unchanged.

    Key insight:
    - The ranking does not come from the original left-wall shapes: the shear changes where each object lands, the blocker halos change which of those new pixels remain safe, and only those post-shear safe counts determine the final stack order.
    """
    objects, blockers, _forbidden = _analyze_grid(np.asarray(grid, dtype=int))
    for idx, obj in enumerate(objects):
        obj["id"] = idx
    return _render_stacked(grid.shape, objects, blockers)


def _sample_shape(rng: random.Random, height: int) -> np.ndarray:
    return SHAPES_BY_HEIGHT[height][rng.randrange(len(SHAPES_BY_HEIGHT[height]))].copy()


def _build_layout(rng: random.Random) -> tuple[int, int, list[int], list[np.ndarray], list[int], list[int]] | None:
    n_objects = rng.choice([3, 4])
    heights = [rng.choice([2, 3]) for _ in range(n_objects)]
    shapes = [_sample_shape(rng, h) for h in heights]
    top_margin = rng.randint(1, 3)
    gaps = [rng.randint(1, 2) for _ in range(n_objects - 1)]
    bottom_clearance = rng.randint(2, 4)
    h = top_margin + sum(heights) + sum(gaps) + bottom_clearance + 1
    if h > 22:
        return None
    w = rng.randint(17, 20)

    tops = []
    current = top_margin
    for idx, height in enumerate(heights):
        tops.append(current)
        current += height
        if idx < len(gaps):
            current += gaps[idx]
    if current > h - 1 - bottom_clearance:
        return None
    return h, w, tops, shapes, heights, rng.sample(list(OBJECT_COLORS), n_objects)


def _candidate_blockers(
    h: int, w: int, objects: list[dict], total_stack_height: int
) -> list[tuple[int, int]]:
    min_top = min(obj["top"] for obj in objects)
    max_bottom = max(obj["bottom"] for obj in objects)
    max_row = min(max_bottom + 1, h - total_stack_height - 4)
    cols = range(max(w // 2, w - 7), w - 1)
    out = []
    for r in range(max(1, min_top - 1), max_row + 1):
        for c in cols:
            halo = {(r, c)}
            halo.update((nr, nc) for nr, nc in _neighbors8(r, c) if _in_bounds((h, w), nr, nc))
            if any(any(cell == hc for hc in halo) for obj in objects for cell in obj["transformed"]):
                out.append((r, c))
    return out


def _order_from(values: list[int], tops: list[int]) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: (values[idx], tops[idx]))


def _semantic_checks(grid: np.ndarray) -> bool:
    objects, blockers, forbidden = _analyze_grid(grid)
    for idx, obj in enumerate(objects):
        obj["id"] = idx

    safe = [obj["safe_area"] for obj in objects]
    areas = [obj["area"] for obj in objects]
    tops = [obj["top"] for obj in objects]
    unsheared = [obj["unsheared_safe_area"] for obj in objects]

    if len(set(safe)) != len(safe):
        return False
    if not any(s < a for s, a in zip(safe, areas)):
        return False

    safe_order = _order_from(safe, tops)
    top_order = list(range(len(objects)))
    area_order = _order_from(areas, tops)
    unsheared_order = _order_from(unsheared, tops)
    if safe_order == top_order:
        return False
    if safe_order == area_order:
        return False
    if safe_order == unsheared_order:
        return False

    if not any(obj["safe_area"] != obj["unsheared_safe_area"] for obj in objects):
        return False

    for br, bc in blockers:
        halo = {(br, bc)}
        halo.update((nr, nc) for nr, nc in _neighbors8(br, bc) if _in_bounds(grid.shape, nr, nc))
        if not any(any(cell == hc for hc in halo) for obj in objects for cell in obj["transformed"]):
            return False

    out = _render_stacked(grid.shape, objects, blockers)
    if np.array_equal(out, grid):
        return False

    alt_top = _stack_with_order(grid.shape, objects, blockers, top_order)
    alt_area = _stack_with_order(grid.shape, objects, blockers, area_order)
    alt_unsheared = _stack_with_order(grid.shape, objects, blockers, unsheared_order)
    if np.array_equal(out, alt_top):
        return False
    if np.array_equal(out, alt_area):
        return False
    if np.array_equal(out, alt_unsheared):
        return False

    expected = main(grid)
    if not np.array_equal(out, expected):
        return False

    _ = forbidden
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _build_layout(rng)
    if layout is None:
        return None
    h, w, tops, shapes, _heights, colors = layout

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[-1, :] = GROUND
    objects = []
    for idx, (top, mask, color) in enumerate(zip(tops, shapes, colors)):
        grid[top : top + mask.shape[0], : mask.shape[1]][mask] = color
        objects.append(
            {
                "id": idx,
                "top": top,
                "bottom": top + mask.shape[0] - 1,
                "height": mask.shape[0],
                "width": mask.shape[1],
                "area": int(mask.sum()),
                "mask": mask.copy(),
                "color": color,
                "transformed": _sheared_coords(mask, top, w),
            }
        )

    total_stack_height = sum(obj["height"] for obj in objects)
    candidates = _candidate_blockers(h, w, objects, total_stack_height)
    if len(candidates) < 6:
        return None

    for _ in range(MAX_BLOCKER_ATTEMPTS):
        blocker_count = rng.choice([2, 3])
        blockers = rng.sample(candidates, blocker_count)
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for (r1, c1), (r2, c2) in combinations(blockers, 2)):
            continue
        test = grid.copy()
        for r, c in blockers:
            if test[r, c] != BACKGROUND:
                break
            test[r, c] = BLOCKER
        else:
            if _semantic_checks(test):
                return test
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
