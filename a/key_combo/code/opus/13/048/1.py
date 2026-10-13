from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
BLACK = 0

PRIORITY_BY_KIND = {"H": 1, "V": 2, "D": 3}
FILL_BY_PRIORITY = {1: BLUE, 2: GREEN, 3: YELLOW}

H_TEMPLATES = [
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=int),
    np.array([[0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 0, 0]], dtype=int),
    np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=int),
]

D_TEMPLATES = [
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=int),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _anti_diag_reflect(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(mask)).T


def _classify_mask(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    diagonal = False
    if mask.shape[0] == mask.shape[1]:
        diagonal = np.array_equal(mask, mask.T) or np.array_equal(mask, _anti_diag_reflect(mask))
    kinds = [horizontal, vertical, diagonal]
    if sum(kinds) != 1:
        raise ValueError("non-noise object must have exactly one symmetry category")
    if horizontal:
        return "H"
    if vertical:
        return "V"
    return "D"


def _distance_map(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rr = np.arange(shape[0], dtype=int)[:, None, None]
    cc = np.arange(shape[1], dtype=int)[None, :, None]
    cr = coords[:, 0][None, None, :]
    cc2 = coords[:, 1][None, None, :]
    return (np.abs(rr - cr) + np.abs(cc - cc2)).min(axis=2)


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    objects = []
    noise = []
    for coords in _components(grid == RED):
        if len(coords) == 1:
            noise.append((int(coords[0, 0]), int(coords[0, 1])))
            continue
        kind = _classify_mask(_tight_mask(coords))
        objects.append(
            {
                "coords": coords,
                "kind": kind,
                "priority": PRIORITY_BY_KIND[kind],
                "bbox": _bbox(coords),
            }
        )
    return objects, noise


def _solve(grid: np.ndarray, *, ignore_priority: bool = False, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    objects, noise = _parse_objects(g)
    if not objects:
        raise ValueError("expected at least one non-noise red object")

    h, w = g.shape
    out = np.zeros_like(g)
    object_mask = np.zeros_like(g, dtype=bool)
    noise_mask = np.zeros_like(g, dtype=bool)
    objects_by_priority: dict[int, list[int]] = defaultdict(list)
    distance_maps = []
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        object_mask[coords[:, 0], coords[:, 1]] = True
        objects_by_priority[obj["priority"]].append(idx)
        distance_maps.append(_distance_map(coords, g.shape))
    for r, c in noise:
        noise_mask[r, c] = True
    out[object_mask] = RED

    same_priority_ties = 0
    cross_priority_wins = 0

    for r in range(h):
        for c in range(w):
            if object_mask[r, c] or noise_mask[r, c]:
                continue

            if ignore_priority:
                per_object = [(int(distance_maps[i][r, c]), i) for i in range(len(objects))]
                best_distance = min(dist for dist, _idx in per_object)
                winners = [i for dist, i in per_object if dist == best_distance]
                if len(winners) > 1:
                    out[r, c] = BLACK
                else:
                    out[r, c] = FILL_BY_PRIORITY[objects[winners[0]]["priority"]]
                continue

            per_priority: list[tuple[int, int]] = []
            for priority, idxs in objects_by_priority.items():
                best = min(int(distance_maps[i][r, c]) for i in idxs)
                per_priority.append((best, priority))
            best_distance, chosen_priority = min(per_priority)
            chosen_object_ids = [
                i for i in objects_by_priority[chosen_priority] if int(distance_maps[i][r, c]) == best_distance
            ]
            if len(chosen_object_ids) > 1:
                out[r, c] = BLACK
                same_priority_ties += 1
                continue

            if any(priority > chosen_priority and dist == best_distance for dist, priority in per_priority):
                cross_priority_wins += 1
            out[r, c] = FILL_BY_PRIORITY[chosen_priority]

    meta = {
        "objects": objects,
        "noise": noise,
        "same_priority_ties": same_priority_ties,
        "cross_priority_wins": cross_priority_wins,
    }
    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the background by nearest red seed objects, with symmetry type deciding expansion priority.

    Input structure:
    - A black background contains several red 4-connected foreground objects and some isolated red singleton noise pixels.
    - Every non-singleton red object has exactly one reflection-symmetry category: horizontal, vertical, or diagonal.
    - All meaningful objects share the same red color; only their shapes distinguish their roles.

    Rule:
    - Split red pixels into 4-connected components and erase singleton components as noise.
    - Classify each remaining object by its tight-bbox reflection symmetry: horizontal objects have highest priority, vertical objects next, diagonal objects lowest.
    - For every non-object cell, compare its Manhattan distance to the nearest object of each priority group; the smallest distance wins, and equal distances across priorities are resolved in favor of the higher-priority group.
    - Fill cells won by horizontal / vertical / diagonal priorities with blue / green / yellow respectively, but if two objects of the same winning priority are equally near then leave that cell black as a separator line.
    - Keep all original non-noise red object pixels red.

    Key insight:
    - Symmetry does not recolor the objects directly; instead it gates which family of seeds wins contested nearest-fill territory, while same-priority ties stay black.
    """
    return _solve(grid)


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = mask.copy()
    for _ in range(rng.randrange(4)):
        transformed = np.rot90(transformed)
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    if rng.random() < 0.5:
        transformed = np.flipud(transformed)
    transformed = transformed.astype(int)
    if _classify_mask(transformed.astype(bool)) not in {"H", "V", "D"}:
        raise ValueError("template transform produced invalid symmetry")
    return transformed


def _sample_mask(kind: str, rng: random.Random) -> np.ndarray:
    if kind == "H":
        mask = rng.choice(H_TEMPLATES).copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        return mask
    if kind == "V":
        mask = rng.choice(H_TEMPLATES).T.copy()
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    base = rng.choice(D_TEMPLATES).copy()
    if rng.random() < 0.5:
        base = np.fliplr(base)
    if rng.random() < 0.5:
        base = np.flipud(base)
    return base


def _slot_bounds(length: int, size: int, side: str) -> tuple[int, int]:
    max_start = length - size
    if max_start <= 0:
        return 0, 0
    if side == "top":
        lo = 1
        hi = max(1, length // 2 - size)
    elif side == "bottom":
        lo = min(max_start, max(length // 2, 1))
        hi = max_start - 1
    elif side == "left":
        lo = 1
        hi = max(1, length // 2 - size)
    elif side == "right":
        lo = min(max_start, max(length // 2, 1))
        hi = max_start - 1
    else:
        raise ValueError(side)
    if hi < lo:
        lo, hi = 0, max_start
    return int(lo), int(hi)


def _sample_top_left(grid_shape: tuple[int, int], mask_shape: tuple[int, int], slot: str, rng: random.Random):
    h, w = grid_shape
    mh, mw = mask_shape
    vertical, horizontal = slot.split("_")
    r0_lo, r0_hi = _slot_bounds(h, mh, vertical)
    c0_lo, c0_hi = _slot_bounds(w, mw, horizontal)
    r0 = rng.randint(r0_lo, r0_hi)
    c0 = rng.randint(c0_lo, c0_hi)
    return r0, c0


def _expand_bbox(bbox: tuple[int, int, int, int], gap: int, shape: tuple[int, int]):
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - gap),
        max(0, c0 - gap),
        min(shape[0] - 1, r1 + gap),
        min(shape[1] - 1, c1 + gap),
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    placed = coords + np.array([top, left], dtype=int)
    grid[placed[:, 0], placed[:, 1]] = color
    return placed


def _apply_grid_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = grid.copy()
    choice = rng.randrange(4)
    if choice == 1:
        transformed = np.fliplr(transformed)
    elif choice == 2:
        transformed = np.flipud(transformed)
    elif choice == 3:
        transformed = np.rot90(transformed, 2)
    return transformed.copy()


def _validate_generated(grid: np.ndarray) -> bool:
    out, meta = _solve(grid, return_meta=True)
    out_no_priority = _solve(grid, ignore_priority=True)
    kinds = {obj["kind"] for obj in meta["objects"]}
    if kinds != {"H", "V", "D"}:
        return False
    if len(meta["noise"]) < 1:
        return False
    if meta["same_priority_ties"] < 1:
        return False
    if meta["cross_priority_wins"] < 1:
        return False
    if np.array_equal(out, out_no_priority):
        return False
    if not all(int((out == color).sum()) > 0 for color in (BLUE, GREEN, YELLOW)):
        return False
    for r, c in meta["noise"]:
        if int(out[r, c]) != BLACK:
            return False
    if int((out == RED).sum()) != sum(len(obj["coords"]) for obj in meta["objects"]):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    base_layout = [("top_left", "H"), ("top_right", "H"), ("bottom_left", "V"), ("bottom_right", "D")]

    for _attempt in range(400):
        h = rng.randint(14, 18)
        w = rng.randint(14, 18)
        grid = np.zeros((h, w), dtype=int)
        layout = list(base_layout)
        if rng.random() < 0.5:
            layout[0], layout[1] = layout[1], layout[0]
        if rng.random() < 0.5:
            layout[2], layout[3] = layout[3], layout[2]

        placed_bboxes: list[tuple[int, int, int, int]] = []
        red_cells: set[tuple[int, int]] = set()
        success = True

        for slot, kind in layout:
            mask = _sample_mask(kind, rng)
            for _ in range(50):
                top, left = _sample_top_left(grid.shape, mask.shape, slot, rng)
                coords = np.argwhere(mask) + np.array([top, left], dtype=int)
                bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
                expanded = _expand_bbox(bbox, gap=1, shape=grid.shape)
                if any(_bboxes_overlap(expanded, other) for other in placed_bboxes):
                    continue
                if any((int(r), int(c)) in red_cells for r, c in coords):
                    continue
                _paste_mask(grid, mask, top, left, RED)
                placed_bboxes.append(bbox)
                red_cells.update((int(r), int(c)) for r, c in coords)
                break
            else:
                success = False
                break

        if not success:
            continue

        noise_target = rng.randint(1, 4)
        available = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BLACK]
        rng.shuffle(available)
        noise_added = 0
        for r, c in available:
            if noise_added >= noise_target:
                break
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in red_cells):
                continue
            grid[r, c] = RED
            red_cells.add((r, c))
            noise_added += 1

        if noise_added < 1:
            continue

        grid = _apply_grid_transform(grid, rng)
        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
