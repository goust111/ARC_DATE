from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOLE_SHAPES = {
    1: [((0, 0),)],
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
}
HOLE_PARTITIONS = {
    1: [(1,)],
    2: [(2,), (1, 1)],
    3: [(3,), (2, 1), (1, 1, 1)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_color_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, GREY}):
        for coords in _mask_components(grid == color):
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return comps


def _hole_mask_and_area(local_grid: np.ndarray) -> tuple[np.ndarray, int]:
    black_mask = local_grid == BLACK
    hole_mask = np.zeros_like(black_mask, dtype=bool)
    area = 0
    h, w = local_grid.shape
    for comp in _mask_components(black_mask):
        touches_border = any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)
        if touches_border:
            continue
        for r, c in comp:
            hole_mask[r, c] = True
        area += len(comp)
    return hole_mask, area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored object into outside black cells by a geodesic range equal to its enclosed hole area.

    Input structure:
    - Black is the background, grey cells are inert blockers, and each other color forms one 4-connected object.
    - Every object may contain one or more enclosed black holes inside its own bounding box.
    - Objects are separated so their outside expansion zones do not meet.

    Rule:
    - For each colored object, count the black cells inside its bbox that are not 4-connected to the bbox boundary; call this count k.
    - Starting from black cells 4-adjacent to the object's outer boundary, flood only through black cells for at most k steps.
    - Recolor every reached outside black cell with the object's color, while leaving grey blockers, other objects, and the object's internal holes unchanged.

    Key insight:
    - The hidden fuel is not the object size but the total area of its enclosed black holes, and the expansion follows black-cell path distance around blockers instead of a simple square dilation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _find_color_components(g):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        local = g[r0 : r1 + 1, c0 : c1 + 1]
        local_holes, steps = _hole_mask_and_area(local)
        if steps <= 0:
            continue

        hole_mask = np.zeros(g.shape, dtype=bool)
        hole_mask[r0 : r1 + 1, c0 : c1 + 1] = local_holes

        boundary_black: deque[tuple[int, int, int]] = deque()
        seen = np.zeros(g.shape, dtype=bool)
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if g[nr, nc] != BLACK or hole_mask[nr, nc] or seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                boundary_black.append((nr, nc, 1))

        while boundary_black:
            r, c, dist = boundary_black.popleft()
            out[r, c] = color
            if dist >= steps:
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if seen[nr, nc] or g[nr, nc] != BLACK or hole_mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                boundary_black.append((nr, nc, dist + 1))

    return out


def _quadrant_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (0, 0, mid_r - 1, mid_c - 1),
        (0, mid_c, mid_r - 1, w - 1),
        (mid_r, 0, h - 1, mid_c - 1),
        (mid_r, mid_c, h - 1, w - 1),
    ]


def _quadrant_area(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bounds
    return r1 - r0 + 1, c1 - c0 + 1


def _choose_hole_parts(k: int) -> tuple[int, ...]:
    choices = list(HOLE_PARTITIONS[k])
    weights = [1, 2, 2][: len(choices)]
    return random.choices(choices, weights=weights, k=1)[0]


def _can_place_hole(
    occupied: np.ndarray,
    top: int,
    left: int,
    shape: tuple[tuple[int, int], ...],
) -> bool:
    h, w = occupied.shape
    cells = [(top + dr, left + dc) for dr, dc in shape]
    if not all(0 <= r < h and 0 <= c < w for r, c in cells):
        return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _place_holes(obj_h: int, obj_w: int, k: int) -> np.ndarray | None:
    interior_h = obj_h - 2
    interior_w = obj_w - 2
    if interior_h <= 0 or interior_w <= 0:
        return None

    for _ in range(60):
        hole_mask = np.zeros((obj_h, obj_w), dtype=bool)
        occupied = np.zeros((interior_h, interior_w), dtype=bool)
        ok = True
        for area in _choose_hole_parts(k):
            shape = random.choice(HOLE_SHAPES[area])
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            placed = False
            for _attempt in range(80):
                top = random.randint(0, interior_h - shape_h)
                left = random.randint(0, interior_w - shape_w)
                if not _can_place_hole(occupied, top, left, shape):
                    continue
                for dr, dc in shape:
                    occupied[top + dr, left + dc] = True
                    hole_mask[1 + top + dr, 1 + left + dc] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and int(hole_mask.sum()) == k:
            return hole_mask
    return None


def _paint_object(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: int,
    k: int,
) -> dict | None:
    qr0, qc0, qr1, qc1 = bounds
    qh, qw = _quadrant_area(bounds)
    max_h = min(7, qh - 2 * (k + 1))
    max_w = min(7, qw - 2 * (k + 1))
    if max_h < 5 or max_w < 5:
        return None

    for _ in range(80):
        obj_h = random.randint(5, max_h)
        obj_w = random.randint(5, max_w)
        holes = _place_holes(obj_h, obj_w, k)
        if holes is None:
            continue

        top = random.randint(qr0 + k + 1, qr1 - obj_h - k)
        left = random.randint(qc0 + k + 1, qc1 - obj_w - k)
        grid[top : top + obj_h, left : left + obj_w] = color
        grid[top : top + obj_h, left : left + obj_w][holes] = BLACK
        return {
            "color": color,
            "bbox": (top, left, top + obj_h - 1, left + obj_w - 1),
            "bounds": bounds,
            "k": k,
        }
    return None


def _free_cells_for_rect(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    pad: int = 0,
) -> bool:
    h, w = grid.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BLACK)


def _side_obstacle_candidates(info: dict) -> list[list[tuple[int, int]]]:
    qr0, qc0, qr1, qc1 = info["bounds"]
    r0, c0, r1, c1 = info["bbox"]
    candidates: list[list[tuple[int, int]]] = []

    if r0 - qr0 >= 3:
        row = r0 - 2
        cells = [(row, c) for c in range(max(qc0, c0 - 1), min(qc1, c1 + 1) + 1)]
        if len(cells) >= 2:
            candidates.append(cells)
    if qr1 - r1 >= 3:
        row = r1 + 2
        cells = [(row, c) for c in range(max(qc0, c0 - 1), min(qc1, c1 + 1) + 1)]
        if len(cells) >= 2:
            candidates.append(cells)
    if c0 - qc0 >= 3:
        col = c0 - 2
        cells = [(r, col) for r in range(max(qr0, r0 - 1), min(qr1, r1 + 1) + 1)]
        if len(cells) >= 2:
            candidates.append(cells)
    if qc1 - c1 >= 3:
        col = c1 + 2
        cells = [(r, col) for r in range(max(qr0, r0 - 1), min(qr1, r1 + 1) + 1)]
        if len(cells) >= 2:
            candidates.append(cells)

    random.shuffle(candidates)
    return candidates


def _place_grey_obstacles(grid: np.ndarray, info: dict) -> int:
    placed = 0
    for cells in _side_obstacle_candidates(info):
        if _free_cells_for_rect(
            grid,
            min(r for r, _ in cells),
            min(c for _, c in cells),
            max(r for r, _ in cells),
            max(c for _, c in cells),
        ):
            for r, c in cells:
                grid[r, c] = GREY
            placed += len(cells)
            break

    if random.random() < 0.45:
        qr0, qc0, qr1, qc1 = info["bounds"]
        for _ in range(40):
            vertical = random.random() < 0.5
            length = random.randint(2, 3)
            if vertical:
                r0 = random.randint(qr0, qr1 - length + 1)
                c0 = random.randint(qc0, qc1)
                r1, c1 = r0 + length - 1, c0
            else:
                r0 = random.randint(qr0, qr1)
                c0 = random.randint(qc0, qc1 - length + 1)
                r1, c1 = r0, c0 + length - 1
            if not _free_cells_for_rect(grid, r0, c0, r1, c1, pad=1):
                continue
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    grid[r, c] = GREY
            placed += (r1 - r0 + 1) * (c1 - c0 + 1)
            break

    return placed


def _new_cells_by_color(inp: np.ndarray, out: np.ndarray) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for color in set(map(int, np.unique(out))) - {BLACK, GREY}:
        masks[color] = (inp == BLACK) & (out == color)
    return masks


def _ranges_are_separate(masks: dict[int, np.ndarray], out: np.ndarray) -> bool:
    colors = list(masks)
    for i, color_a in enumerate(colors):
        body_a = out == color_a
        new_a = masks[color_a]
        if not new_a.any():
            return False
        for color_b in colors[i + 1 :]:
            body_b = out == color_b
            for r, c in zip(*np.where(body_a | new_a)):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if _in_bounds(out.shape, nr, nc) and body_b[nr, nc]:
                        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(26, 30)
    w = random.randint(26, 30)
    grid = np.zeros((h, w), dtype=int)
    quadrants = _quadrant_bounds(h, w)
    n_objects = random.randint(2, 4)
    chosen = random.sample(range(4), n_objects)
    colors = random.sample(PALETTE, n_objects)

    high_fuel_idx = random.randrange(n_objects)
    objects = []
    grey_total = 0
    for idx, quad_idx in enumerate(chosen):
        k = random.choice((2, 3)) if idx == high_fuel_idx else random.choice((1, 2, 3))
        info = _paint_object(grid, quadrants[quad_idx], colors[idx], k)
        if info is None:
            return None
        objects.append(info)

    random.shuffle(objects)
    for info in objects:
        grey_total += _place_grey_obstacles(grid, info)

    if grey_total == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    new_by_color = _new_cells_by_color(grid, out)
    if len(new_by_color) != n_objects:
        return None
    if not _ranges_are_separate(new_by_color, out):
        return None

    hole_areas = []
    for info in objects:
        r0, c0, r1, c1 = info["bbox"]
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        _holes, area = _hole_mask_and_area(local)
        hole_areas.append(area)
    if max(hole_areas) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
