from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _frame_info(center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int, int]:
    r, c = center
    h, w = shape
    d = min(r, c, h - 1 - r, w - 1 - c)
    return d, r - d, c - d, r + d, c + d


def _find_cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLUE:
                continue
            if all(int(g[r + dr, c + dc]) == BLUE for dr, dc in DIRS4):
                centers.append((r, c))
    return centers


def _component_size(grid: np.ndarray, start: tuple[int, int]) -> int:
    if int(grid[start]) != BLUE:
        return 0
    h, w = grid.shape
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLUE and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen)


def _boxes_separated(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = box1
    rr0, cc0, rr1, cc1 = box2
    return r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0


def _distances_to_edges(center: tuple[int, int], shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    r, c = center
    h, w = shape
    return {
        (-1, 0): r,
        (1, 0): h - 1 - r,
        (0, -1): c,
        (0, 1): w - 1 - c,
    }


def _extension_cells(center: tuple[int, int], direction: tuple[int, int], final_step: int) -> list[tuple[int, int]]:
    r, c = center
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(2, final_step + 1)]


def _try_add_blue(
    grid: np.ndarray,
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    intended_centers: set[tuple[int, int]],
) -> bool:
    if not cells:
        return False
    h, w = grid.shape
    unique = sorted(set(cells))
    for r, c in unique:
        if not (0 <= r < h and 0 <= c < w):
            return False

    backup = [int(grid[r, c]) for r, c in unique]
    for r, c in unique:
        grid[r, c] = BLUE

    ok = set(_find_cross_centers(grid)) == intended_centers
    if not ok:
        for (r, c), value in zip(unique, backup):
            grid[r, c] = value
    return ok


def _box_mask(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for center in centers:
        _d, r0, c0, r1, c1 = _frame_info(center, shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _inside_any_box(r: int, c: int, boxes: list[tuple[int, int, int, int]], pad: int = 0) -> bool:
    for r0, c0, r1, c1 in boxes:
        if r0 - pad <= r <= r1 + pad and c0 - pad <= c <= c1 + pad:
            return True
    return False


def _can_place_shape(grid: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells and int(grid[rr, cc]) == BLUE:
                    return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect blue '+' centers, draw center-scaled green square frames, and erase non-cross blue pixels inside each frame.

    Input structure:
    - The grid is a black background with blue foreground pixels forming several 4-connected shapes.
    - Some blue pixels are '+' centers: the pixel itself and its four orthogonal neighbors are all blue.
    - Each valid center lies away from the border so its orthogonal neighbors exist inside the canvas.

    Rule:
    - Find every blue pixel whose up, down, left, and right neighbors are also blue.
    - For each center (r, c), compute d = min(r, c, H-1-r, W-1-c) and draw a one-pixel green square border
      with corners (r-d, c-d) and (r+d, c+d), overwriting any previous color on the border cells.
    - Then, inside that square's strict interior, erase every blue pixel to black except the five pixels of any
      detected '+' cross itself; pixels outside all frames remain unchanged.

    Key insight:
    - The '+' motif is not the final object but the anchor that simultaneously determines where a frame is drawn
      and which interior blue pixels are protected from the frame-driven erasure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    centers = _find_cross_centers(g)
    protected = set()
    for center in centers:
        protected |= _cross_cells(center)

    for center in centers:
        _d, r0, c0, r1, c1 = _frame_info(center, g.shape)
        out[r0, c0 : c1 + 1] = GREEN
        out[r1, c0 : c1 + 1] = GREEN
        out[r0 : r1 + 1, c0] = GREEN
        out[r0 : r1 + 1, c1] = GREEN

    for center in centers:
        _d, r0, c0, r1, c1 = _frame_info(center, g.shape)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == BLUE and (r, c) not in protected:
                    out[r, c] = BG

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
CENTER_COUNT_CHOICES = [1, 2, 3]
CENTER_COUNT_WEIGHTS = [2, 4, 2]
MAX_FRAME_RADIUS = 4
GEN_ATTEMPTS = 600

OUTSIDE_SHAPES = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
]


def _sample_centers(h: int, w: int) -> list[tuple[int, int]] | None:
    k = random.choices(CENTER_COUNT_CHOICES, weights=CENTER_COUNT_WEIGHTS, k=1)[0]
    max_d = min(MAX_FRAME_RADIUS, min(h, w) // 2 - 1)
    if max_d < 3:
        return None

    all_candidates = []
    deep_candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            d = min(r, c, h - 1 - r, w - 1 - c)
            if 1 <= d <= max_d:
                all_candidates.append((r, c))
            if 3 <= d <= max_d:
                deep_candidates.append((r, c))

    if not deep_candidates:
        return None

    for _ in range(200):
        first = random.choice(deep_candidates)
        centers = [first]
        boxes = [(_frame_info(first, (h, w))[1:])]
        pool = all_candidates[:]
        random.shuffle(pool)

        for cand in pool:
            if cand == first:
                continue
            box = _frame_info(cand, (h, w))[1:]
            if abs(cand[0] - first[0]) + abs(cand[1] - first[1]) < 6 and len(centers) == 1:
                pass
            if any(abs(cand[0] - rr) + abs(cand[1] - cc) < 6 for rr, cc in centers):
                continue
            if any(not _boxes_separated(box, other, pad=1) for other in boxes):
                continue
            centers.append(cand)
            boxes.append(box)
            if len(centers) == k:
                break

        if len(centers) == k:
            return centers
    return None


def _add_detached_interior_clutter(
    grid: np.ndarray,
    center: tuple[int, int],
    intended_centers: set[tuple[int, int]],
) -> bool:
    d, r0, c0, r1, c1 = _frame_info(center, grid.shape)
    if d < 2:
        return False

    protected = _cross_cells(center)
    candidates = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in protected:
                continue
            if any((r + dr, c + dc) in protected for dr, dc in DIRS4):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    if not candidates:
        return False

    for cell in candidates:
        if _try_add_blue(grid, [cell], intended_centers):
            return True
    return False


def _add_embedded_extensions(
    grid: np.ndarray,
    center: tuple[int, int],
    intended_centers: set[tuple[int, int]],
) -> tuple[bool, bool]:
    d, _r0, _c0, _r1, _c1 = _frame_info(center, grid.shape)
    distances = _distances_to_edges(center, grid.shape)
    dirs = list(DIRS4)
    random.shuffle(dirs)

    added_inside = False
    added_outside = False

    inside_dirs = [direction for direction in dirs if min(d - 1, distances[direction]) >= 2]
    if inside_dirs:
        direction = random.choice(inside_dirs)
        final_step = random.randint(2, min(d - 1, distances[direction]))
        added_inside = _try_add_blue(grid, _extension_cells(center, direction, final_step), intended_centers)

    across_dirs = [direction for direction in dirs if distances[direction] >= d + 1]
    if across_dirs:
        direction = random.choice(across_dirs)
        max_step = distances[direction]
        final_step = random.randint(d + 1, min(max_step, d + 3))
        added_outside = _try_add_blue(grid, _extension_cells(center, direction, final_step), intended_centers)

    return added_inside, added_outside


def _place_outside_shape(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    intended_centers: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    templates = OUTSIDE_SHAPES[:]
    random.shuffle(templates)

    for template in templates:
        max_r = max(r for r, _c in template)
        max_c = max(c for _r, c in template)
        positions = [(r0, c0) for r0 in range(h - max_r) for c0 in range(w - max_c)]
        random.shuffle(positions)
        for r0, c0 in positions:
            cells = {(r0 + r, c0 + c) for r, c in template}
            if any(_inside_any_box(r, c, boxes, pad=1) for r, c in cells):
                continue
            if not _can_place_shape(grid, cells, pad=1):
                continue
            if _try_add_blue(grid, list(cells), intended_centers):
                return True
    return False


def _validate_semantics(grid: np.ndarray, centers: list[tuple[int, int]]) -> bool:
    if set(_find_cross_centers(grid)) != set(centers):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    boxes = []
    protected = set()
    for center in centers:
        d, r0, c0, r1, c1 = _frame_info(center, grid.shape)
        boxes.append((r0, c0, r1, c1))
        protected |= _cross_cells(center)
        if not np.all(out[r0, c0 : c1 + 1] == GREEN):
            return False
        if not np.all(out[r1, c0 : c1 + 1] == GREEN):
            return False
        if not np.all(out[r0 : r1 + 1, c0] == GREEN):
            return False
        if not np.all(out[r0 : r1 + 1, c1] == GREEN):
            return False

    any_erased = False
    for r0, c0, r1, c1 in boxes:
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == BLUE and (r, c) not in protected:
                    if int(out[r, c]) != BG:
                        return False
                    any_erased = True

    any_outside_blue = False
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLUE:
                continue
            if _inside_any_box(r, c, boxes):
                continue
            any_outside_blue = True
            if int(out[r, c]) != BLUE:
                return False

    if not any_erased or not any_outside_blue:
        return False
    if not np.any(out == GREEN):
        return False

    component_sizes = [_component_size(grid, center) for center in centers]
    if max(component_sizes, default=0) <= 5:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    centers = _sample_centers(h, w)
    if not centers:
        return None

    grid = np.zeros((h, w), dtype=int)
    for center in centers:
        for r, c in _cross_cells(center):
            grid[r, c] = BLUE

    intended_centers = set(centers)
    if set(_find_cross_centers(grid)) != intended_centers:
        return None

    boxes = [(_frame_info(center, grid.shape)[1:]) for center in centers]

    embedded_flags = [False] * len(centers)
    if len(centers) == 1:
        embedded_flags[0] = random.random() < 0.65
    else:
        order = list(range(len(centers)))
        random.shuffle(order)
        embedded_flags[order[0]] = True
        embedded_flags[order[1]] = False
        for idx in order[2:]:
            embedded_flags[idx] = random.random() < 0.5

    have_erased_material = False
    have_outside_from_cross = False
    for center, embedded in zip(centers, embedded_flags):
        if embedded:
            added_inside, added_outside = _add_embedded_extensions(grid, center, intended_centers)
            have_erased_material |= added_inside
            have_outside_from_cross |= added_outside

        if random.random() < 0.85 or not have_erased_material:
            have_erased_material |= _add_detached_interior_clutter(grid, center, intended_centers)

    if not have_erased_material:
        return None

    outside_shape_count = random.randint(1, 3)
    placed_outside = 0
    for _ in range(outside_shape_count):
        if _place_outside_shape(grid, boxes, intended_centers):
            placed_outside += 1

    if placed_outside == 0:
        return None

    if not have_outside_from_cross and random.random() < 0.5:
        for center in centers:
            _added_inside, added_outside = _add_embedded_extensions(grid, center, intended_centers)
            have_outside_from_cross |= added_outside
            if have_outside_from_cross:
                break

    if not _validate_semantics(grid, centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
