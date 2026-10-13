from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

OBJECT_COUNT_RANGE = (2, 5)
GRID_SIDE_RANGE = (16, 30)
OBJECT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components_by_color(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append((color, coords))

    return comps


def _evaluate_fold(
    pts: set[tuple[int, int]],
    shape: tuple[int, int],
    kind: str,
    twice_axis: int,
    source_positive: bool,
):
    h, w = shape
    source: list[tuple[int, int]] = []
    target: list[tuple[int, int]] = []

    for r, c in pts:
        coord = c if kind == "vertical" else r
        if 2 * coord < twice_axis:
            positive = False
        elif 2 * coord > twice_axis:
            positive = True
        else:
            continue

        if positive == source_positive:
            source.append((r, c))
        else:
            target.append((r, c))

    if not source or not target:
        return None

    mirrored: set[tuple[int, int]] = set()
    for r, c in source:
        rr = r
        cc = c
        if kind == "vertical":
            cc = twice_axis - c
        else:
            rr = twice_axis - r
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        mirrored.add((rr, cc))

    target_set = set(target)
    if not target_set.issubset(mirrored):
        return None

    paired = mirrored & target_set
    filled = mirrored - target_set
    if not paired or not filled:
        return None

    return {
        "kind": kind,
        "twice_axis": int(twice_axis),
        "source_positive": bool(source_positive),
        "source": source,
        "score": (len(paired), -len(filled), -abs(len(source) - len(target))),
    }


def _best_fold_for_component(coords: list[tuple[int, int]]):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    local = {(r - r0, c - c0) for r, c in coords}
    shape = (r1 - r0 + 1, c1 - c0 + 1)

    candidates = []
    for twice_axis in range(0, 2 * shape[1] - 1):
        for source_positive in (False, True):
            cand = _evaluate_fold(local, shape, "vertical", twice_axis, source_positive)
            if cand is not None:
                candidates.append(cand)
    for twice_axis in range(0, 2 * shape[0] - 1):
        for source_positive in (False, True):
            cand = _evaluate_fold(local, shape, "horizontal", twice_axis, source_positive)
            if cand is not None:
                candidates.append(cand)

    assert candidates, "expected a unique best reflection axis"
    candidates.sort(key=lambda cand: cand["score"], reverse=True)
    best = candidates[0]
    assert sum(cand["score"] == best["score"] for cand in candidates) == 1, "reflection axis must be unique"
    return best, r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each near-symmetric foreground object across its hidden mirror axis and mark already-mirrored pixels yellow.

    Input structure:
    - The background is black.
    - The grid contains several 4-connected monochrome foreground objects separated by background.
    - Each object is organized around one implicit horizontal or vertical reflection axis, with pixels on both sides and some extra pixels only on one side.

    Rule:
    - For each foreground object, infer the unique best horizontal or vertical reflection axis.
    - Choose the fuller side of that axis and reflect all of its pixels onto the opposite side while keeping the original pixels in place.
    - If a reflected position already contains the object's color, recolor that destination pixel yellow.
    - If a reflected position is background, paint it with the object's original color.

    Key insight:
    - The objects are not perfectly symmetric, so the axis must be inferred from the strongest partial mirror match; yellow marks the part that was already mirrored while unchanged-color additions show the missing repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = g.copy()
    for color, coords in _components_by_color(g):
        fold, r0, c0 = _best_fold_for_component(coords)
        for lr, lc in fold["source"]:
            rr = lr
            cc = lc
            if fold["kind"] == "vertical":
                cc = fold["twice_axis"] - lc
            else:
                rr = fold["twice_axis"] - lr

            gr = r0 + rr
            gc = c0 + cc
            current = int(out[gr, gc])
            assert current in (BLACK, color), "objects must stay separated during folding"
            out[gr, gc] = YELLOW if current == color else color

    return out


def _grow_region(shape: tuple[int, int], target_size: int, seed: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    cells = {seed}
    frontier = set(_neighbors4(seed[0], seed[1], h, w))

    while len(cells) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in cells:
                frontier.add(nb)

    return cells


def _expand_region(
    base: set[tuple[int, int]],
    shape: tuple[int, int],
    extra_count: int,
) -> set[tuple[int, int]] | None:
    h, w = shape
    cells = set(base)

    while extra_count > 0:
        frontier = set()
        for r, c in cells:
            for nb in _neighbors4(r, c, h, w):
                if nb not in cells:
                    frontier.add(nb)
        if not frontier:
            return None

        cells.add(random.choice(tuple(frontier)))
        extra_count -= 1

    return cells


def _render_vertical_object(
    core: set[tuple[int, int]],
    source_half: set[tuple[int, int]],
    axis_mode: str,
) -> np.ndarray:
    height = max(r for r, _c in source_half) + 1
    half_w = max(c for _r, c in source_half) + 1
    width = 2 * half_w + (1 if axis_mode == "cell" else 0)
    obj = np.zeros((height, width), dtype=int)

    for r, c in source_half:
        obj[r, c] = 1
    for r, c in core:
        obj[r, width - 1 - c] = 1

    if axis_mode == "cell":
        axis_col = half_w
        for r, c in core:
            if c == half_w - 1:
                obj[r, axis_col] = 1

    return obj


def _generate_object_mask() -> np.ndarray | None:
    for _ in range(OBJECT_ATTEMPTS):
        height = random.randint(5, 9)
        half_w = random.randint(2, 4)
        core_max = min(height * half_w - 1, height + half_w)
        core_min = max(3, height // 2)
        if core_min > core_max:
            continue

        seed = (random.randrange(height), half_w - 1)
        core_size = random.randint(core_min, core_max)
        core = _grow_region((height, half_w), core_size, seed)
        if core is None:
            continue

        max_extra = min(height * half_w - len(core), half_w + 2)
        if max_extra < 1:
            continue

        source_half = _expand_region(core, (height, half_w), random.randint(1, max_extra))
        if source_half is None:
            continue

        axis_mode = random.choice(("gap", "cell"))
        obj = _render_vertical_object(core, source_half, axis_mode)
        if random.random() < 0.5:
            obj = np.fliplr(obj)
        if random.random() < 0.5:
            obj = obj.T

        try:
            _best_fold_for_component(list(map(tuple, np.argwhere(obj > 0))))
        except AssertionError:
            continue
        return obj

    return None


def _try_place_objects(
    shape: tuple[int, int],
    objects: list[np.ndarray],
    colors: list[int],
) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda idx: objects[idx].shape[0] * objects[idx].shape[1], reverse=True)

    for idx in order:
        obj = objects[idx]
        oh, ow = obj.shape
        color = colors[idx]
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - oh)
            c0 = random.randint(0, w - ow)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + oh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + ow + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            view = grid[r0:r0 + oh, c0:c0 + ow]
            view[obj > 0] = color
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _generate_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    objects: list[np.ndarray] = []
    for _ in range(object_count):
        obj = _generate_object_mask()
        if obj is None:
            return None
        objects.append(obj)

    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    if any(obj.shape[0] > height or obj.shape[1] > width for obj in objects):
        return None

    colors = random.sample(PALETTE, object_count)
    grid = _try_place_objects((height, width), objects, colors)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if solved.shape != grid.shape or np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-symmetry input")
