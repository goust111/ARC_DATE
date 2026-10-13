from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
OBJECT_COUNT_OPTIONS = (2, 3)
PATH_LENGTH_POOL = (4, 5, 6, 7, 8)
ODD_RING_AREAS = (3, 5, 7, 9)
EVEN_RING_AREAS = (2, 4, 6, 8, 10, 12)
AREA_TO_INNER_DIMS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    5: [(1, 5), (5, 1)],
    6: [(2, 3), (3, 2)],
    7: [(1, 7), (7, 1)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
}

GRID_MIN = 18
GRID_MAX = 28
BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_ATTEMPTS = 300
PATH_BUILD_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _ordered_path(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}
    degrees = {}
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        degrees[(r, c)] = int(deg)
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "expected a one-pixel-wide path with exactly two endpoints"
    assert all(deg <= 2 for deg in degrees.values()), "path must not branch"

    r0, c0, _r1, _c1 = bbox_coords(np.array(list(pts), dtype=int))
    start = min(endpoints, key=lambda pt: (abs(pt[0] - r0) + abs(pt[1] - c0), pt[0], pt[1]))

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must have a unique continuation"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(pts), "path traversal must visit every path pixel exactly once"
    return ordered


def _enclosed_background_area(grid: np.ndarray, ring_coords: np.ndarray) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox_coords(ring_coords)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)

    local_h = rr1 - rr0 + 1
    local_w = cc1 - cc0 + 1
    blocked = np.zeros((local_h, local_w), dtype=bool)
    for r, c in ring_coords:
        blocked[int(r) - rr0, int(c) - cc0] = True

    exterior = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(lr: int, lc: int) -> None:
        if 0 <= lr < local_h and 0 <= lc < local_w and not blocked[lr, lc] and not exterior[lr, lc]:
            exterior[lr, lc] = True
            q.append((lr, lc))

    for lr in range(local_h):
        push(lr, 0)
        push(lr, local_w - 1)
    for lc in range(local_w):
        push(0, lc)
        push(local_h - 1, lc)

    while q:
        lr, lc = q.popleft()
        for nlr, nlc in neighbors4(lr, lc):
            push(nlr, nlc)

    area = 0
    for lr in range(local_h):
        for lc in range(local_w):
            gr = rr0 + lr
            gc = cc0 + lc
            if not blocked[lr, lc] and not exterior[lr, lc] and int(grid[gr, gc]) == BLACK:
                area += 1
    return int(area)


def _transform_coords(coords: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    if flip:
        arr[:, 1] *= -1
    for _ in range(rot % 4):
        arr = np.column_stack((-arr[:, 1], arr[:, 0]))
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr.astype(int)


def _build_path(length: int) -> np.ndarray:
    def extend(path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        current = path[-1]
        dirs = DIRS4[:]
        random.shuffle(dirs)
        if len(path) >= 2 and random.random() < 0.65:
            prev = path[-2]
            prev_dir = (current[0] - prev[0], current[1] - prev[1])
            dirs.sort(key=lambda step: step == prev_dir)
        for dr, dc in dirs:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in occupied:
                continue
            adjacent = [(nr, nc) for nr, nc in neighbors4(*nxt) if (nr, nc) in occupied]
            if adjacent != [current]:
                continue
            path.append(nxt)
            occupied.add(nxt)
            if extend(path, occupied):
                return True
            occupied.remove(nxt)
            path.pop()
        return False

    for _ in range(PATH_BUILD_ATTEMPTS):
        path = [(0, 0)]
        occupied = {(0, 0)}
        if not extend(path, occupied):
            continue
        coords = np.array(path, dtype=int)
        coords = _transform_coords(coords, random.randrange(4), random.random() < 0.5)
        ordered = _ordered_path(coords)
        d0 = ordered[0][0] + ordered[0][1]
        d1 = ordered[-1][0] + ordered[-1][1]
        if d0 == d1:
            continue
        return coords
    raise RuntimeError("failed to build a valid path")


def _make_ring(area: int) -> np.ndarray:
    inner_h, inner_w = random.choice(AREA_TO_INNER_DIMS[area])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    coords = []
    for r in range(outer_h):
        for c in range(outer_w):
            if r in (0, outer_h - 1) or c in (0, outer_w - 1):
                coords.append((r, c))
    return np.array(coords, dtype=int)


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    return (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _paint_object(
    grid: np.ndarray,
    local_coords: np.ndarray,
    top: int,
    left: int,
    color: int | None,
    ordered_path: list[tuple[int, int]] | None,
    path_colors: list[int] | None,
) -> None:
    if color is not None:
        for r, c in local_coords:
            grid[top + int(r), left + int(c)] = int(color)
        return
    assert ordered_path is not None and path_colors is not None
    for (r, c), value in zip(ordered_path, path_colors):
        grid[top + int(r), left + int(c)] = int(value)


def _sample_ring_areas(count: int) -> list[int]:
    odd_count = 1 if count == 2 else random.choice((1, 2))
    even_count = count - odd_count
    picked = random.sample(ODD_RING_AREAS, odd_count) + random.sample(EVEN_RING_AREAS, even_count)
    random.shuffle(picked)
    return picked


def _generate_objects(count: int) -> list[dict]:
    path_lengths = random.sample(PATH_LENGTH_POOL, count)
    ring_areas = _sample_ring_areas(count)

    objects: list[dict] = []
    for area in ring_areas:
        coords = _make_ring(area)
        r0, c0, r1, c1 = bbox_coords(coords)
        objects.append(
            {
                "kind": "ring",
                "coords": coords,
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "bbox": (0, 0, r1, c1),
            }
        )

    for length in path_lengths:
        coords = _build_path(length)
        ordered = _ordered_path(coords)
        colors = random.sample(PATH_COLORS, length)
        r0, c0, r1, c1 = bbox_coords(coords)
        objects.append(
            {
                "kind": "path",
                "coords": coords,
                "ordered": ordered,
                "path_colors": colors,
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "bbox": (0, 0, r1, c1),
            }
        )
    random.shuffle(objects)
    return objects


def _place_objects(height: int, width: int, objects: list[dict]) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for obj in sorted(objects, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        obj_h, obj_w = obj["shape"]
        if obj_h + 2 * BORDER > height or obj_w + 2 * BORDER > width:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(BORDER, height - obj_h - BORDER)
            left = random.randint(BORDER, width - obj_w - BORDER)
            box = (top, left, top + obj_h - 1, left + obj_w - 1)
            if all(_boxes_far_enough(box, other, PLACEMENT_PADDING) for other in placed_boxes):
                placed_boxes.append(box)
                if obj["kind"] == "ring":
                    _paint_object(grid, obj["coords"], top, left, GRAY, None, None)
                else:
                    _paint_object(
                        grid,
                        obj["coords"],
                        top,
                        left,
                        None,
                        obj["ordered"],
                        obj["path_colors"],
                    )
                placed = True
                break
        if not placed:
            return None
    return grid


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    rings = []
    for coords in _components_mask(grid == GRAY):
        rings.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": _enclosed_background_area(grid, coords),
            }
        )

    paths = []
    path_mask = (grid != BLACK) & (grid != GRAY)
    for coords in _components_mask(path_mask):
        ordered = _ordered_path(coords)
        paths.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "ordered": ordered,
                "length": len(ordered),
            }
        )
    return rings, paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the gray control loops and reverse only the ranked paths whose paired loop encloses an odd number of black cells.

    Input structure:
    - A black background contains 2-3 disjoint gray one-pixel-wide closed loops and the same number of disjoint multicolor one-pixel-wide 4-connected paths.
    - Each path has exactly two endpoints, no branches, distinct pixel colors along the path, and a unique length among the paths.
    - Each gray loop encloses a unique number of black cells, and the loops do not touch the paths or each other.

    Rule:
    - Count the number of enclosed black cells inside each gray loop.
    - Rank loops by enclosed area from small to large, and rank paths by path length from short to long.
    - Pair equal ranks; for each pair, traverse the path from the endpoint closer to the path bbox's top-left corner and reverse its color sequence iff the paired loop's enclosed area is odd.
    - Leave even-paired paths unchanged, erase every gray loop to black, and keep all path geometry fixed.

    Key insight:
    - The gray loops are not output objects; they are detached control signals whose enclosed-area parity decides whether the equally ranked path keeps or reverses its color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == BLACK, "expected black background"

    rings, paths = _analyze_grid(g)
    assert len(rings) == len(paths) and 2 <= len(paths) <= 3, "expected equally many loops and paths"

    rings.sort(key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]))
    paths.sort(key=lambda item: (item["length"], item["bbox"][0], item["bbox"][1]))

    out = g.copy()
    out[out == GRAY] = BLACK
    for ring, path in zip(rings, paths):
        if ring["area"] % 2 == 0:
            continue
        colors = [int(g[r, c]) for r, c in path["ordered"]]
        for (r, c), value in zip(path["ordered"], reversed(colors)):
            out[r, c] = int(value)
    return out


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        rings, paths = _analyze_grid(grid)
        if len(rings) != len(paths) or len(paths) not in OBJECT_COUNT_OPTIONS:
            return False
        areas = [ring["area"] for ring in rings]
        lengths = [path["length"] for path in paths]
        if len(set(areas)) != len(areas) or len(set(lengths)) != len(lengths):
            return False
        if not any(area % 2 == 0 for area in areas):
            return False
        if not any(area % 2 == 1 for area in areas):
            return False
        for path in paths:
            colors = [int(grid[r, c]) for r, c in path["ordered"]]
            if len(set(colors)) != len(colors):
                return False
        out = main(grid)
        if out.shape != grid.shape:
            return False
        if np.any(out == GRAY):
            return False
        if np.array_equal(out, grid):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(OBJECT_COUNT_OPTIONS)
    objects = _generate_objects(count)

    max_h = max(obj["shape"][0] for obj in objects) + 2 * BORDER
    max_w = max(obj["shape"][1] for obj in objects) + 2 * BORDER
    min_side = max(GRID_MIN, 20 + 2 * (count - 2))
    height = random.randint(max(min_side, max_h), GRID_MAX)
    width = random.randint(max(min_side, max_w), GRID_MAX)

    grid = _place_objects(height, width, objects)
    if grid is None or not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except RuntimeError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
