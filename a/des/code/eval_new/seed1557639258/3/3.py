from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


AZURE = 8
WHITE = 0
MAROON = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OBJECT_PALETTE = [1, 2, 3, 4, 5, 6, 7]


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform(cells: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rot):
        pts = [(-c, r) for r, c in pts]
    return _normalize(pts)


def _build_shape_variants() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    base_shapes = {
        2: [((0, 0), (0, 1))],
        3: [
            ((0, 0), (0, 1), (0, 2)),
            ((0, 0), (1, 0), (1, 1)),
        ],
        4: [
            ((0, 0), (0, 1), (0, 2), (0, 3)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (1, 0), (2, 0), (2, 1)),
            ((0, 1), (1, 0), (1, 1), (1, 2)),
        ],
        5: [
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
            ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
            ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
            ((0, 1), (1, 1), (1, 2), (2, 0), (2, 1)),
        ],
    }
    variants: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size, shapes in base_shapes.items():
        seen: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes:
            for flip in (False, True):
                for rot in range(4):
                    seen.add(_transform(shape, rot, flip))
        variants[size] = sorted(seen)
    return variants


SHAPE_VARIANTS = _build_shape_variants()


def _checkerboard(h: int, w: int, phase: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.where((rows + cols + phase) % 2 == 0, AZURE, WHITE).astype(int)


def _find_background_colors(grid: np.ndarray) -> tuple[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert vals.size >= 2, "expected two checkerboard background colors"
    return int(vals[order[0]]), int(vals[order[1]])


def _find_objects(grid: np.ndarray, bg_colors: tuple[int, int]) -> tuple[list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_mask = np.isin(g, bg_colors)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if bool(bg_mask[r, c]) or bool(seen[r, c]):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            objects.append({"color": color, "cells": cells})

    return objects, bg_mask


def _diffusion_marks(
    objects: list[dict[str, object]],
    h: int,
    w: int,
    background_mask: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    marks: dict[tuple[int, int], set[int]] = defaultdict(set)
    for obj in objects:
        color = int(obj["color"])
        cells = list(obj["cells"])
        radius = len(cells)
        covered: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                for step in range(1, radius + 1):
                    rr = r + dr * step
                    cc = c + dc * step
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    if bool(background_mask[rr, cc]):
                        covered.add((int(rr), int(cc)))
        for cell in covered:
            marks[cell].add(color)
    return marks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a checkerboard background using straight diffusion waves emitted by small colored objects.

    Input structure:
    - The background is a two-color checkerboard made from azure and white cells.
    - Two to four small monochrome 4-connected objects sit on top of that checkerboard, each with a distinct color.
    - The objects themselves are the only non-checkerboard regions in the input.

    Rule:
    - For each colored object, compute its area N and extend rays of length N upward, downward, leftward, and rightward from every pixel of that object.
    - Only background cells are paintable: each reached checkerboard cell is marked with the source object's color, while object pixels remain unchanged.
    - If a background cell is marked by exactly one object, recolor it to that object's color; if it is marked by two or more different objects, recolor it maroon; untouched checkerboard cells stay as they were.

    Key insight:
    - The diffusion distance is not fixed: each object's own pixel count determines how far its four-direction wave reaches, and overlap between different waves collapses to one shared maroon color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = _find_background_colors(g)
    objects, bg_mask = _find_objects(g, bg_colors)

    out = g.copy()
    marks = _diffusion_marks(objects, h, w, bg_mask)
    for (r, c), colors in marks.items():
        if len(colors) >= 2:
            out[r, c] = MAROON
        else:
            out[r, c] = next(iter(colors))
    return out


def _can_place(
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    return all(
        0 <= r < h and 0 <= c < w and (r, c) not in blocked and (r, c) not in forbidden
        for r, c in cells
    )


def _reserve(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> None:
    for r, c in cells:
        occupied.add((r, c))
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))


def _shape_cells_at(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _try_place_random(
    size: int,
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    variants = SHAPE_VARIANTS[size][:]
    random.shuffle(variants)
    for _ in range(120):
        shape = random.choice(variants)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        top = random.randint(0, h - shape_h)
        left = random.randint(0, w - shape_w)
        cells = _shape_cells_at(shape, top, left)
        if _can_place(cells, blocked, h, w, forbidden):
            return cells
    return None


def _target_anchor(target: tuple[int, int], direction: str, distance: int) -> tuple[int, int]:
    r, c = target
    if direction == "left":
        return r, c - distance
    if direction == "right":
        return r, c + distance
    if direction == "up":
        return r - distance, c
    return r + distance, c


def _try_place_targeted(
    size: int,
    direction: str,
    target: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    variants = SHAPE_VARIANTS[size][:]
    random.shuffle(variants)
    distances = list(range(2, size + 1)) + [1]
    seen_dist = []
    for d in distances:
        if d not in seen_dist:
            seen_dist.append(d)
    random.shuffle(seen_dist)

    for shape in variants:
        anchors = list(shape)
        random.shuffle(anchors)
        for distance in seen_dist:
            ar, ac = _target_anchor(target, direction, distance)
            for sr, sc in anchors:
                top = ar - sr
                left = ac - sc
                cells = _shape_cells_at(shape, top, left)
                if _can_place(cells, blocked, h, w, forbidden):
                    return cells
    return None


def _scene_marks(objects: list[dict[str, object]], h: int, w: int) -> dict[tuple[int, int], set[int]]:
    occupied = {cell for obj in objects for cell in obj["cells"]}
    bg_mask = np.ones((h, w), dtype=bool)
    for r, c in occupied:
        bg_mask[r, c] = False
    return _diffusion_marks(objects, h, w, bg_mask)


def _paint_objects(grid: np.ndarray, objects: list[dict[str, object]]) -> None:
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cells"]:
            grid[r, c] = color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    phase = random.randint(0, 1)

    num_objects = random.randint(2, 4)
    if num_objects == 2:
        mode = random.choice(["none", "none", "none", "pair"])
    else:
        mode = random.choice(["none", "none", "none", "pair", "triple"])

    colors = random.sample(OBJECT_PALETTE, num_objects)
    sizes = [random.randint(2, 5) for _ in range(num_objects)]
    if len(set(sizes)) == 1:
        idx = random.randrange(num_objects)
        sizes[idx] = random.choice([size for size in range(2, 6) if size != sizes[idx]])

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []
    target: tuple[int, int] | None = None

    needed = 0
    if mode == "pair":
        needed = 2
    elif mode == "triple":
        needed = 3

    if needed:
        margin_r = 5
        margin_c = 5
        target = (
            random.randint(margin_r, h - margin_r - 1),
            random.randint(margin_c, w - margin_c - 1),
        )
        directions = random.sample(["left", "right", "up", "down"], needed)
        forbidden = {target}
        for i in range(needed):
            cells = _try_place_targeted(sizes[i], directions[i], target, blocked, forbidden, h, w)
            if cells is None:
                return None
            obj = {"color": colors[i], "cells": cells}
            objects.append(obj)
            _reserve(cells, occupied, blocked, h, w)

    for i in range(needed, num_objects):
        forbidden = {target} if target is not None else set()
        cells = _try_place_random(sizes[i], blocked, forbidden, h, w)
        if cells is None:
            return None
        obj = {"color": colors[i], "cells": cells}
        objects.append(obj)
        _reserve(cells, occupied, blocked, h, w)

    marks = _scene_marks(objects, h, w)
    overlap_cells = [cell for cell, cols in marks.items() if len(cols) >= 2]
    single_cells = [cell for cell, cols in marks.items() if len(cols) == 1]
    if not single_cells:
        return None
    if mode == "none" and overlap_cells:
        return None
    if mode == "pair":
        if target is None or len(marks.get(target, set())) < 2:
            return None
    if mode == "triple":
        if target is None or len(marks.get(target, set())) < 3:
            return None

    grid = _checkerboard(h, w, phase)
    _paint_objects(grid, objects)
    if len(_find_objects(grid, (AZURE, WHITE))[0]) != num_objects:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard diffusion puzzle")
