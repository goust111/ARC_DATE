from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
LOCK_COLOR = 5
BG_CHOICES = (0, 1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
LOCK_SIZE_RANGE = (6, 8)
SHAPE_AREA_MIN = 4
SHAPE_AREA_MAX = 9
OBJECT_PADDING = 1
GEN_RETRIES = 400
SHAPE_RETRIES = 120
LAYOUT_RETRIES = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in cells))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _mask_from_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    _, h, w = _normalize(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _has_internal_hole(shape: tuple[tuple[int, int], ...]) -> bool:
    mask = _mask_from_shape(shape)
    bg = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if r not in (0, mask.shape[0] - 1) and c not in (0, mask.shape[1] - 1):
                continue
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in DIR4:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bg[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((int(nr), int(nc)))
    return bool(np.any(bg & ~seen))


def _apply_transform(shape: tuple[tuple[int, int], ...], transform_id: int) -> tuple[tuple[int, int], ...]:
    mask = _mask_from_shape(shape)
    if transform_id >= 4:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, transform_id % 4)
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    return _normalize(coords)[0]


def _orientation_set(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    return {_apply_transform(shape, transform_id) for transform_id in range(8)}


def _extract_lock_info(
    grid: np.ndarray,
    color: int,
    cells: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(cells)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if set(map(int, np.unique(sub))) - {int(bg), int(color)}:
        return None
    missing = sub == bg
    if not missing.any():
        return None
    if missing[0, :].any() or missing[-1, :].any() or missing[:, 0].any() or missing[:, -1].any():
        return None
    missing_comps = _components(missing)
    if len(missing_comps) != 1:
        return None
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "missing": _normalize(missing_comps[0])[0],
    }


def _detect_locks(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        comps = _components(grid == color)
        if not comps:
            continue
        infos: list[dict[str, object]] = []
        dims = set()
        ok = True
        for comp in comps:
            info = _extract_lock_info(grid, color, comp, bg)
            if info is None:
                ok = False
                break
            r0, c0, r1, c1 = info["bbox"]
            dims.add((int(r1) - int(r0) + 1, int(c1) - int(c0) + 1))
            infos.append(info)
        if ok and infos and len(dims) == 1:
            candidates.append((int(color), infos))
    assert len(candidates) == 1, f"expected one lock color, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    lock_color, locks = _detect_locks(g, bg)

    keys: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), int(lock_color)}):
        for comp in _components(g == color):
            shape = _normalize(comp)[0]
            keys.append(
                {
                    "color": int(color),
                    "cells": comp,
                    "orientations": _orientation_set(shape),
                }
            )

    assert len(keys) == len(locks) and keys, "locks and keys must pair one-to-one"

    out = g.copy()
    for key in keys:
        for r, c in key["cells"]:
            out[int(r), int(c)] = bg

    unused = set(range(len(keys)))
    for lock in locks:
        missing = lock["missing"]
        matches = [idx for idx in unused if missing in keys[idx]["orientations"]]
        assert len(matches) == 1, "each lock must match exactly one remaining key"
        idx = matches[0]
        unused.remove(idx)
        r0, c0, r1, c1 = lock["bbox"]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(keys[idx]["color"])

    assert not unused, "every key must be used"
    return out


def _random_connected_shape(inner_h: int, inner_w: int) -> tuple[tuple[int, int], ...] | None:
    max_cells = min(SHAPE_AREA_MAX, inner_h * inner_w - 1)
    min_cells = min(SHAPE_AREA_MIN, max_cells)
    if min_cells > max_cells:
        return None

    for _ in range(SHAPE_RETRIES):
        target = random.randint(min_cells, max_cells)
        cells = {(
            random.randrange(inner_h),
            random.randrange(inner_w),
        )}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            weights: list[int] = []
            seen = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        touch = sum((nr + rr, nc + cc) in cells for rr, cc in DIR4)
                        frontier.append((nr, nc))
                        weights.append(1 + touch)
            if not frontier:
                break
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != target:
            continue
        shape, h, w = _normalize(tuple(cells))
        if h < 2 or w < 2 or len(shape) == h * w:
            continue
        if _has_internal_hole(shape):
            continue
        if len(_orientation_set(shape)) != 8:
            continue
        return shape
    return None


def _find_location(occupied: np.ndarray, h: int, w: int, padding: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        rr0 = max(0, r - padding)
        cc0 = max(0, c - padding)
        rr1 = min(H, r + h + padding)
        cc1 = min(W, c + w + padding)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        return int(r), int(c)
    return None


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    lock_h = random.randint(*LOCK_SIZE_RANGE)
    lock_w = random.randint(*LOCK_SIZE_RANGE)
    inner_h = lock_h - 2
    inner_w = lock_w - 2
    if inner_h < 2 or inner_w < 2:
        return None

    base_shapes: list[tuple[tuple[int, int], ...]] = []
    used_shape_classes: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(n_pairs):
        shape = None
        for _ in range(SHAPE_RETRIES):
            candidate = _random_connected_shape(inner_h, inner_w)
            if candidate is None:
                continue
            shape_class = min(_orientation_set(candidate))
            if shape_class in used_shape_classes:
                continue
            used_shape_classes.add(shape_class)
            shape = candidate
            break
        if shape is None:
            return None
        base_shapes.append(shape)

    bg = random.choice(BG_CHOICES)
    usable_colors = [color for color in range(10) if color not in (bg, LOCK_COLOR)]
    if len(usable_colors) < n_pairs:
        return None
    key_colors = random.sample(usable_colors, n_pairs)
    transforms = [random.randrange(8) for _ in range(n_pairs)]
    key_shapes = [_apply_transform(shape, transform_id) for shape, transform_id in zip(base_shapes, transforms, strict=False)]

    total_lock_area = n_pairs * lock_h * lock_w
    min_area = 2 * total_lock_area + 12
    max_key_h = max(_normalize(shape)[1] for shape in key_shapes)
    max_key_w = max(_normalize(shape)[2] for shape in key_shapes)
    dims = [
        (H, W)
        for H in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)
        for W in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)
        if H * W >= min_area and H >= max(lock_h, max_key_h) + 2 and W >= max(lock_w, max_key_w) + 2
    ]
    if not dims:
        return None

    locks = [{"hole": shape} for shape in base_shapes]
    keys = [{"shape": shape, "color": color} for shape, color in zip(key_shapes, key_colors, strict=False)]

    placed = False
    for _ in range(LAYOUT_RETRIES):
        H, W = random.choice(dims)
        occupied = np.zeros((H, W), dtype=bool)
        ok = True

        placeables = [("lock", idx, lock_h, lock_w) for idx in range(n_pairs)]
        for idx, shape in enumerate(key_shapes):
            _, sh, sw = _normalize(shape)
            placeables.append(("key", idx, sh, sw))
        placeables.sort(key=lambda item: item[2] * item[3], reverse=True)

        for kind, idx, h, w in placeables:
            pos = _find_location(occupied, h, w, OBJECT_PADDING)
            if pos is None:
                ok = False
                break
            r0, c0 = pos
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            if kind == "lock":
                locks[idx]["pos"] = pos
            else:
                keys[idx]["pos"] = pos
        if ok:
            placed = True
            break
    if not placed:
        return None

    grid = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)

    for idx, lock in enumerate(locks):
        top, left = lock["pos"]
        grid[top : top + lock_h, left : left + lock_w] = LOCK_COLOR
        for r, c in lock["hole"]:
            grid[top + 1 + int(r), left + 1 + int(c)] = bg
        expected[top : top + lock_h, left : left + lock_w] = keys[idx]["color"]

    for key in keys:
        top, left = key["pos"]
        for r, c in key["shape"]:
            grid[top + int(r), left + int(c)] = key["color"]

    if _most_common_color(grid) != bg:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC input")
