from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2

MAX_GENERATION_ATTEMPTS = 500
PADDING = 1

BASE_SHAPES = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    H, W = mask.shape
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    comps = _components4(grid != BG)
    objects = []
    for idx, coords in enumerate(comps, start=1):
        rr = coords[:, 0]
        cc = coords[:, 1]
        marker_count = int(np.count_nonzero(grid[rr, cc] == RED))
        objects.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(coords.shape[0]),
                "marker_count": marker_count,
                "touch_right": bool(np.any(cc == W - 1)),
            }
        )
    return objects


def _tile_with_mirror(grid: np.ndarray) -> np.ndarray:
    return np.concatenate([grid, np.fliplr(grid)], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the input horizontally, tile it side by side, then erase merged components whose total marker count exceeds the threshold.

    Input structure:
    - A black background contains 3-4 separated 4-connected foreground objects built from blue cells with 0-2 embedded red marker cells.
    - Some objects touch the original grid's right border, while others stay strictly inside the grid.

    Rule:
    - Reflect the whole grid left-to-right to make a mirror copy, then concatenate `[original | mirror]`.
    - In the tiled grid, treat blue and red together as foreground and find all 4-connected components.
    - Count the red marker cells inside each tiled component.
    - If a tiled component contains more than 2 red cells, erase the entire component to black.
    - Otherwise keep the component and recolor all of its red cells to blue.

    Key insight:
    - Only objects that touch the original right border merge with their own mirror across the tiling seam, so their post-merge marker totals can differ from their original totals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    tiled = _tile_with_mirror(g)
    out = tiled.copy()
    for coords in _components4(tiled != BG):
        rr = coords[:, 0]
        cc = coords[:, 1]
        red_total = int(np.count_nonzero(tiled[rr, cc] == RED))
        if red_total > 2:
            out[rr, cc] = BG
        else:
            out[rr, cc] = BLUE
    return out


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            if not any(v.shape == candidate.shape and np.array_equal(v, candidate) for v in variants):
                variants.append(candidate.copy())
    return variants


ALL_SHAPES: tuple[np.ndarray, ...] = tuple(
    candidate
    for base in BASE_SHAPES
    for candidate in _unique_transforms(base)
)


def _choose_shape(marker_count: int, *, touch_right: bool) -> np.ndarray:
    candidates = [
        shape
        for shape in ALL_SHAPES
        if int(shape.sum()) > marker_count and (not touch_right or shape.shape[1] >= 2)
    ]
    return random.choice(candidates).copy()


def _shape_patch(mask: np.ndarray, marker_count: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    marker_cells = set(random.sample(coords, marker_count))
    for r, c in coords:
        patch[r, c] = RED if (r, c) in marker_cells else BLUE
    return patch


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    H, W = occupied.shape
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for dr, dc in np.argwhere(mask):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - PADDING)
        r2 = min(H, r + PADDING + 1)
        c1 = max(0, c - PADDING)
        c2 = min(W, c + PADDING + 1)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(patch != BG)
    for dr, dc in coords:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(patch[dr, dc])
        occupied[rr, cc] = True


def _role_specs() -> list[dict]:
    specs = [
        {"name": "edge_removed", "touch_right": True, "marker_count": 2},
        {"name": "edge_survivor", "touch_right": True, "marker_count": 1},
        {"name": "interior_two", "touch_right": False, "marker_count": 2},
    ]
    if random.random() < 0.55:
        specs.append(
            {
                "name": "extra",
                "touch_right": random.choice([True, False]),
                "marker_count": random.choice([0, 1, 2]),
            }
        )
    return specs


def _build_attempt() -> np.ndarray | None:
    specs = _role_specs()
    H = random.randint(7, 8) if len(specs) == 4 else random.randint(6, 8)
    W = random.randint(6, 7)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placed_specs = []

    planned = []
    for spec in specs:
        shape = _choose_shape(spec["marker_count"], touch_right=spec["touch_right"])
        planned.append({**spec, "shape": shape, "area": int(shape.sum())})
    planned.sort(key=lambda item: (item["touch_right"], item["area"]), reverse=True)

    for spec in planned:
        shape = spec["shape"]
        h, w = shape.shape
        if spec["touch_right"]:
            cols = [W - w]
        else:
            if W - w <= 0:
                return None
            cols = list(range(0, W - w))
        rows = list(range(0, H - h + 1))
        candidates = [(r0, c0) for r0 in rows for c0 in cols]
        random.shuffle(candidates)

        patch = _shape_patch(shape, spec["marker_count"])
        placed = False
        for r0, c0 in candidates:
            if _can_place(occupied, shape, int(r0), int(c0)):
                _place_patch(grid, occupied, patch, int(r0), int(c0))
                placed_specs.append({**spec, "patch": patch, "r0": int(r0), "c0": int(c0)})
                placed = True
                break
        if not placed:
            return None

    return grid


def _semantic_expected(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    H, W = grid.shape
    objects = _extract_objects(grid)
    assert 3 <= len(objects) <= 4
    assert set(np.unique(grid)).issubset({BG, BLUE, RED})

    right_counts = [obj["marker_count"] for obj in objects if obj["touch_right"]]
    interior_counts = [obj["marker_count"] for obj in objects if not obj["touch_right"]]
    assert 2 in right_counts
    assert 1 in right_counts
    assert 2 in interior_counts

    for obj in objects:
        assert 3 <= obj["area"] <= 6
        assert obj["marker_count"] <= 2

    tiled = _tile_with_mirror(grid)
    id_left = np.zeros((H, W), dtype=int)
    for obj in objects:
        rr = obj["coords"][:, 0]
        cc = obj["coords"][:, 1]
        id_left[rr, cc] = obj["id"]
    id_grid = np.concatenate([id_left, np.fliplr(id_left)], axis=1)

    tiled_components = _components4(tiled != BG)
    edge_count = sum(obj["touch_right"] for obj in objects)
    expected_component_count = 2 * (len(objects) - edge_count) + edge_count
    assert len(tiled_components) == expected_component_count

    component_hits = Counter()
    expected = tiled.copy()
    removed = 0
    seam_survivors = 0
    edge_removed = 0
    interior_two_survivors = 0

    objects_by_id = {obj["id"]: obj for obj in objects}
    for coords in tiled_components:
        rr = coords[:, 0]
        cc = coords[:, 1]
        source_ids = {int(v) for v in id_grid[rr, cc] if int(v) != 0}
        assert len(source_ids) == 1
        obj_id = next(iter(source_ids))
        obj = objects_by_id[obj_id]
        component_hits[obj_id] += 1

        red_total = int(np.count_nonzero(tiled[rr, cc] == RED))
        expected_red = obj["marker_count"] * (2 if obj["touch_right"] else 1)
        assert red_total == expected_red

        spans_seam = bool(np.any(cc < W) and np.any(cc >= W))
        if red_total > 2:
            expected[rr, cc] = BG
            removed += 1
            if obj["touch_right"]:
                edge_removed += 1
        else:
            expected[rr, cc] = BLUE
            if spans_seam:
                seam_survivors += 1
            if (not obj["touch_right"]) and obj["marker_count"] == 2:
                interior_two_survivors += 1

    for obj in objects:
        assert component_hits[obj["id"]] == (1 if obj["touch_right"] else 2)

    out = main(grid)
    assert np.array_equal(out, expected)
    assert removed >= 1
    assert edge_removed >= 1
    assert seam_survivors >= 1
    assert interior_two_survivors >= 2

    return expected, {
        "objects": objects,
        "tiled_components": tiled_components,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is None:
            continue
        try:
            _semantic_expected(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
