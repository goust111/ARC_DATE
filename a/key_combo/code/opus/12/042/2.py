from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


Color = int
Coord = tuple[int, int]

BG = 0
ORANGE = 7
TEAL = 8
INPUT_COLORS = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


SHAPES_BY_AREA: dict[int, tuple[tuple[Coord, ...], ...]] = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    5: (
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ),
}


GRID_MIN = 15
GRID_MAX = 18
OBJECT_COUNTS = (3, 4)
AREAS = (2, 3, 4, 5)
ANCHOR_MIN_GAP = 2
BODY_MIN_GAP = 3
MAX_GENERATION_ATTEMPTS = 500
BODY_HIT_PROB = 0.7


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords.sort()
            anchor = coords[0]
            area = len(coords)
            ray_cells: list[Coord] = []
            for dr, dc in DIRS4:
                for step in range(1, area + 1):
                    nr = anchor[0] + dr * step
                    nc = anchor[1] + dc * step
                    if in_bounds((h, w), nr, nc):
                        ray_cells.append((nr, nc))

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "body": set(coords),
                    "anchor": anchor,
                    "area": area,
                    "rays": ray_cells,
                }
            )

    objects.sort(key=lambda obj: (obj["area"], obj["anchor"][0], obj["anchor"][1]))
    for idx, obj in enumerate(objects):
        obj["id"] = idx
    return objects


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == BG, "expected black background"
    objects = _find_objects(g)
    assert 2 <= len(objects) <= 5, "expected 2-5 foreground objects"

    anchors = [obj["anchor"] for obj in objects]
    assert len({obj["color"] for obj in objects}) == len(objects), "object colors must be unique"
    assert len({a[0] for a in anchors}) == len(anchors), "anchors must have distinct rows"
    assert len({a[1] for a in anchors}) == len(anchors), "anchors must have distinct columns"

    out = g.copy()
    body_owner: dict[Coord, int] = {}
    anchor_cells = {obj["anchor"] for obj in objects}
    for obj in objects:
        for cell in obj["coords"]:
            body_owner[cell] = obj["id"]

    orange_cells: set[Coord] = set()
    teal_cells: set[Coord] = set()
    ray_sources: dict[Coord, set[int]] = {}

    for obj in objects:
        obj_id = obj["id"]
        for cell in obj["rays"]:
            owner = body_owner.get(cell)
            if owner is not None:
                if owner == obj_id:
                    continue
                if cell in anchor_cells:
                    continue
                out[cell] = TEAL
                teal_cells.add(cell)
                continue

            if cell in orange_cells:
                continue

            prev = ray_sources.get(cell)
            if prev is None:
                ray_sources[cell] = {obj_id}
                out[cell] = obj["color"]
            elif obj_id not in prev:
                prev.add(obj_id)
                out[cell] = ORANGE
                orange_cells.add(cell)

    plain_ray_cells = {
        cell
        for cell in ray_sources
        if cell not in orange_cells and cell not in body_owner
    }
    meta = {
        "objects": objects,
        "orange_cells": orange_cells,
        "teal_cells": teal_cells,
        "plain_ray_cells": plain_ray_cells,
        "ray_sources": ray_sources,
    }
    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend area-length rays from each object anchor and recolor ray crossings or body hits specially.

    Input structure:
    - The grid is mostly black background with 3-5 small monochrome 4-connected objects.
    - Each object has a unique visible color and an anchor at its topmost, then leftmost, body pixel.
    - Different objects have anchors on distinct rows and distinct columns.

    Rule:
    - Compute each object's area A, then emit four 1-pixel rays of length A from its anchor.
    - Draw rays in ascending area order; later single-source rays replace earlier single-source rays.
    - If rays from different objects pass through the same non-body cell, freeze that cell to orange.
    - If a ray from one object passes through another object's non-anchor body cell, recolor that body cell teal.
    - Original body pixels otherwise stay in place and anchors always keep their own object color.

    Key insight:
    - Area is not a count to report directly; it is converted into ray length, and only after that does drawing order determine whether a cell is an ordinary ray, a multi-source crossing, or a body-hit recolor.
    """
    return _solve(grid, return_meta=False)


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int]:
    choices = [
        combo
        for combo in combinations(range(lo, hi + 1), count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    return list(random.choice(choices))


def _shape_extent(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return (
        max(r for r, _ in shape),
        max(c for _, c in shape),
    )


def _place_shape(anchor: Coord, shape: tuple[Coord, ...]) -> list[Coord]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in shape]


def _well_separated(existing: list[list[Coord]], candidate: list[Coord]) -> bool:
    for coords in existing:
        for r1, c1 in coords:
            for r2, c2 in candidate:
                if max(abs(r1 - r2), abs(c1 - c2)) < BODY_MIN_GAP:
                    return False
    return True


def _build_grid(
    h: int,
    w: int,
    anchors: list[Coord],
    colors: list[int],
    areas: list[int],
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    placed: list[list[Coord]] = []
    for anchor, color, area in zip(anchors, colors, areas):
        shapes = list(SHAPES_BY_AREA[area])
        random.shuffle(shapes)
        placed_one = False
        for shape in shapes:
            er, ec = _shape_extent(shape)
            if anchor[0] + er >= h or anchor[1] + ec >= w:
                continue
            coords = _place_shape(anchor, shape)
            if not _well_separated(placed, coords):
                continue
            for r, c in coords:
                grid[r, c] = color
            placed.append(coords)
            placed_one = True
            break
        if not placed_one:
            return None
    return grid


def _shape_by_cells(area: int, cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    for shape in SHAPES_BY_AREA[area]:
        if shape == cells:
            return shape
    raise ValueError("shape not found in catalog")


BODY_HIT_VARIANTS = (
    (
        ((0, 0), 3, _shape_by_cells(3, ((0, 0), (0, 1), (0, 2)))),
        ((3, 2), 5, _shape_by_cells(5, ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)))),
        ((1, 5), 4, _shape_by_cells(4, ((0, 0), (0, 1), (1, 0), (1, 1)))),
    ),
    (
        ((0, 0), 3, _shape_by_cells(3, ((0, 0), (1, 0), (2, 0)))),
        ((2, 3), 5, _shape_by_cells(5, ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)))),
        ((5, 1), 4, _shape_by_cells(4, ((0, 0), (0, 1), (1, 0), (1, 1)))),
    ),
)


def _build_body_hit_grid(h: int, w: int, n: int, colors: list[int]) -> np.ndarray | None:
    variant = random.choice(BODY_HIT_VARIANTS)
    base_r = random.randint(1, h - 9)
    base_c = random.randint(1, w - 9)

    anchors: list[Coord] = []
    areas: list[int] = []
    chosen_shapes: list[tuple[Coord, ...]] = []
    for (dr, dc), area, shape in variant:
        anchors.append((base_r + dr, base_c + dc))
        areas.append(area)
        chosen_shapes.append(shape)

    grid = np.zeros((h, w), dtype=int)
    placed: list[list[Coord]] = []
    for anchor, color, shape in zip(anchors, colors[:3], chosen_shapes):
        coords = _place_shape(anchor, shape)
        if not _well_separated(placed, coords):
            return None
        for r, c in coords:
            if not in_bounds((h, w), r, c):
                return None
            grid[r, c] = color
        placed.append(coords)

    if n == 4:
        extra_area = 2
        extra_color = colors[3]
        for _ in range(100):
            anchor = (
                random.randint(1, h - 3),
                random.randint(1, w - 3),
            )
            if anchor[0] in {r for r, _ in anchors} or anchor[1] in {c for _, c in anchors}:
                continue
            shapes = list(SHAPES_BY_AREA[extra_area])
            random.shuffle(shapes)
            placed_one = False
            for shape in shapes:
                er, ec = _shape_extent(shape)
                if anchor[0] + er >= h or anchor[1] + ec >= w:
                    continue
                coords = _place_shape(anchor, shape)
                if not _well_separated(placed, coords):
                    continue
                for r, c in coords:
                    grid[r, c] = extra_color
                placed.append(coords)
                anchors.append(anchor)
                areas.append(extra_area)
                placed_one = True
                break
            if placed_one:
                break
        else:
            return None

    return grid


def _valid_crossing(meta: dict) -> bool:
    return len(meta["orange_cells"]) >= 1


def _reasonable_density(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    changed = int(np.count_nonzero(inp != out))
    if changed < 8:
        return False
    if changed > int(inp.size * 0.33):
        return False
    if len(meta["plain_ray_cells"]) < 6:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n = random.choice(OBJECT_COUNTS)
    colors = random.sample(list(INPUT_COLORS), n)
    require_body_hit = random.random() < BODY_HIT_PROB
    if require_body_hit:
        grid = _build_body_hit_grid(h, w, n, colors)
        if grid is None:
            return None
        areas = [obj["area"] for obj in _find_objects(grid)]
    else:
        areas = random.sample(list(AREAS), n)

        row_hi = h - 3
        col_hi = w - 3
        if row_hi < 3 or col_hi < 3:
            return None

        rows = _sample_spaced_values(1, row_hi, n, ANCHOR_MIN_GAP)
        cols = _sample_spaced_values(1, col_hi, n, ANCHOR_MIN_GAP)
        random.shuffle(cols)
        anchors = list(zip(rows, cols))
        random.shuffle(areas)

        grid = _build_grid(h, w, anchors, colors, areas)
        if grid is None:
            return None

    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return None

    objects = meta["objects"]
    if len(objects) != n:
        return None
    if sorted(obj["area"] for obj in objects) != sorted(areas):
        return None
    if not _valid_crossing(meta):
        return None
    if require_body_hit and len(meta["teal_cells"]) == 0:
        return None
    if not _reasonable_density(grid, out, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
