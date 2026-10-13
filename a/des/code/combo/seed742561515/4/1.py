from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
CONTENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIR_STEPS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# Empty-corner mapping, interpreted clockwise:
# top-left -> up, top-right -> right, bottom-right -> down, bottom-left -> left.
TRIANGLE_PATTERNS: dict[str, np.ndarray] = {
    "up": np.array([[0, 1], [1, 1]], dtype=bool),
    "right": np.array([[1, 0], [1, 1]], dtype=bool),
    "down": np.array([[1, 1], [1, 0]], dtype=bool),
    "left": np.array([[1, 1], [0, 1]], dtype=bool),
}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_grey_rectangle(
    grid: np.ndarray,
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return False
    if coords.shape[0] != 2 * height + 2 * width - 4:
        return False

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(window[border] == GREY):
        return False
    if np.any(window[~border] == GREY):
        return False

    comp_mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        comp_mask[int(r - r0), int(c - c0)] = True
    return bool(np.array_equal(comp_mask, border))


def _triangle_candidates(inner: np.ndarray, color: int) -> list[tuple[int, int, str]]:
    mask = inner == int(color)
    height, width = inner.shape
    candidates: list[tuple[int, int, str]] = []
    for r in range(height - 1):
        for c in range(width - 1):
            block = mask[r : r + 2, c : c + 2]
            for direction, pattern in TRIANGLE_PATTERNS.items():
                if np.array_equal(block, pattern):
                    candidates.append((r, c, direction))
    return candidates


def _triangle_direction(inner: np.ndarray, color: int) -> str:
    candidates = _triangle_candidates(inner, color)
    assert len(candidates) == 1, "expected exactly one indicator triangle per rectangle"
    return candidates[0][2]


def _ray_cells(
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    if direction == "up":
        rr, cc = r0 - 1, mid_c
    elif direction == "down":
        rr, cc = r1 + 1, mid_c
    elif direction == "left":
        rr, cc = mid_r, c0 - 1
    else:
        assert direction == "right"
        rr, cc = mid_r, c1 + 1

    dr, dc = DIR_STEPS[direction]
    cells: list[tuple[int, int]] = []
    for _ in range(int(length)):
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((int(rr), int(cc)))
        else:
            break
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    From each grey rectangular enclosure, emit an outside ray in the triangle's direction.

    Rule:
    - Count all non-black pixels inside each grey rectangle.
    - Find the unique 2x2 three-cell L-shaped triangle of that same color.
    - Use its missing-corner orientation to choose one of four ray directions.
    - Start from the midpoint just outside the corresponding rectangle side.
    - Draw a same-colored ray whose length equals the interior non-black count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    rectangles: list[tuple[int, int, int, int]] = []
    for coords in _components4(g == GREY):
        bbox = _bbox(coords)
        if _is_grey_rectangle(g, coords, bbox):
            rectangles.append(bbox)

    assert rectangles, "expected at least one grey rectangle"

    for bbox in rectangles:
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colored = inner[inner != BLACK]
        assert colored.size >= 3, "each rectangle needs a triangle"
        colors = np.unique(colored)
        assert colors.size == 1, "all non-black pixels inside one rectangle must share a color"
        color = int(colors[0])
        assert color not in (BLACK, GREY)

        direction = _triangle_direction(inner, color)
        ray_len = int(colored.size)
        for rr, cc in _ray_cells(bbox, direction, ray_len, g.shape):
            out[rr, cc] = color

    return out


GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
OUTER_SIZE_OPTIONS = (5, 7, 9)
MAX_RECTANGLES = 4
PLACEMENT_ATTEMPTS = 160
GEN_RETRIES = 300
TRUNCATE_PROB = 0.4


def _triangle_cells(direction: str, top: int, left: int) -> list[tuple[int, int]]:
    pattern = TRIANGLE_PATTERNS[direction]
    cells: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if pattern[dr, dc]:
                cells.append((top + dr, left + dc))
    return cells


def _chebyshev_far_enough(
    cell: tuple[int, int],
    occupied: list[tuple[int, int]],
) -> bool:
    r, c = cell
    return all(max(abs(r - orow), abs(c - ocol)) > 1 for orow, ocol in occupied)


def _make_colored_cells(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]] | None:
    tri_r = random.randint(0, inner_h - 2)
    tri_c = random.randint(0, inner_w - 2)
    triangle = _triangle_cells(direction, tri_r, tri_c)

    candidates = [
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if (r, c) not in triangle and _chebyshev_far_enough((r, c), triangle)
    ]
    random.shuffle(candidates)

    safe_extras: list[tuple[int, int]] = []
    for cell in candidates:
        if _chebyshev_far_enough(cell, triangle + safe_extras):
            safe_extras.append(cell)

    max_extra = min(5, len(safe_extras))
    extra_count = random.randint(0, max_extra)
    return triangle + safe_extras[:extra_count]


def _rand_from_range(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _sample_position(
    grid_h: int,
    grid_w: int,
    obj_h: int,
    obj_w: int,
    direction: str,
    ray_len: int,
    truncated: bool,
) -> tuple[int, int] | None:
    r_min, r_max = 1, grid_h - obj_h - 1
    c_min, c_max = 1, grid_w - obj_w - 1
    if r_min > r_max or c_min > c_max:
        return None

    if direction == "up":
        r_lo, r_hi = (1, min(r_max, ray_len - 1)) if truncated else (max(1, ray_len), r_max)
        c_lo, c_hi = c_min, c_max
    elif direction == "down":
        if truncated:
            r_lo, r_hi = max(1, grid_h - obj_h - ray_len + 1), r_max
        else:
            r_lo, r_hi = 1, min(r_max, grid_h - obj_h - ray_len)
        c_lo, c_hi = c_min, c_max
    elif direction == "left":
        c_lo, c_hi = (1, min(c_max, ray_len - 1)) if truncated else (max(1, ray_len), c_max)
        r_lo, r_hi = r_min, r_max
    else:
        assert direction == "right"
        if truncated:
            c_lo, c_hi = max(1, grid_w - obj_w - ray_len + 1), c_max
        else:
            c_lo, c_hi = 1, min(c_max, grid_w - obj_w - ray_len)
        r_lo, r_hi = r_min, r_max

    r0 = _rand_from_range(int(r_lo), int(r_hi))
    c0 = _rand_from_range(int(c_lo), int(c_hi))
    if r0 is None or c0 is None:
        return None
    return int(r0), int(c0)


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _build_object(
    grid_h: int,
    grid_w: int,
    reserved: np.ndarray,
    color: int,
) -> dict | None:
    size_choices_h = [s for s in OUTER_SIZE_OPTIONS if s <= grid_h - 2]
    size_choices_w = [s for s in OUTER_SIZE_OPTIONS if s <= grid_w - 2]
    if not size_choices_h or not size_choices_w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        obj_h = random.choice(size_choices_h)
        obj_w = random.choice(size_choices_w)
        inner_h = obj_h - 2
        inner_w = obj_w - 2
        direction = random.choice(list(DIR_STEPS))
        colored_cells = _make_colored_cells(inner_h, inner_w, direction)
        if colored_cells is None:
            continue

        ray_len = len(colored_cells)
        truncated = random.random() < TRUNCATE_PROB
        placement = _sample_position(
            grid_h,
            grid_w,
            obj_h,
            obj_w,
            direction,
            ray_len,
            truncated,
        )
        if placement is None:
            continue
        r0, c0 = placement
        bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
        ray_cells = _ray_cells(bbox, direction, ray_len, (grid_h, grid_w))
        if not ray_cells:
            continue

        available = len(ray_cells)
        if truncated and not (available < ray_len):
            continue
        if not truncated and available != ray_len:
            continue

        rect_cells = {
            (rr, cc)
            for rr in range(r0, r0 + obj_h)
            for cc in range(c0, c0 + obj_w)
        }
        reserve_cells = rect_cells | set(ray_cells)
        padded = _expand_cells(reserve_cells, (grid_h, grid_w))
        if any(reserved[rr, cc] for rr, cc in padded):
            continue

        abs_colored = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in colored_cells]
        return {
            "bbox": bbox,
            "color": int(color),
            "direction": direction,
            "colored_cells": abs_colored,
            "ray_cells": ray_cells,
            "reserve_cells": reserve_cells,
            "padded_cells": padded,
        }
    return None


def _draw_input(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    for rr, cc in obj["colored_cells"]:
        grid[rr, cc] = obj["color"]


def _draw_expected(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        for rr, cc in obj["ray_cells"]:
            out[rr, cc] = obj["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    max_rectangles = min(
        MAX_RECTANGLES,
        4 if min(grid_h, grid_w) >= 20 else 3 if min(grid_h, grid_w) >= 16 else 2,
    )
    n_rectangles = random.randint(1, max_rectangles)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    colors = random.sample(CONTENT_COLORS, n_rectangles)
    objects: list[dict] = []

    for color in colors:
        obj = _build_object(grid_h, grid_w, reserved, color)
        if obj is None:
            return None
        _draw_input(grid, obj)
        for rr, cc in obj["padded_cells"]:
            reserved[rr, cc] = True
        objects.append(obj)

    expected = _draw_expected(grid, objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
