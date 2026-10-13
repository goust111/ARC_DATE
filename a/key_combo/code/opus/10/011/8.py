from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

OBJECT_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
INNER_DIM_RANGE = range(1, 9)
MAX_RING_PIXELS = 26
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _enclosed_area(component: np.ndarray) -> int:
    r0 = int(component[:, 0].min())
    c0 = int(component[:, 1].min())
    r1 = int(component[:, 0].max())
    c1 = int(component[:, 1].max())

    wall = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in component:
        wall[int(r - r0 + 1), int(c - c0 + 1)] = True

    seen = np.zeros_like(wall, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < wall.shape[0] and 0 <= nc < wall.shape[1]:
                if wall[nr, nc] or seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = (~wall) & (~seen)
    return int(enclosed.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort closed loops by enclosed area and compress them into horizontal color stripes.

    Input structure:
    - The grid uses black (0) as background.
    - Each non-background 4-connected component is a one-cell-thick closed loop in a unique color.
    - Loops are separated from one another and each loop encloses one background region.

    Rule:
    - For every loop, count its own pixels and also count the area of the background region it encloses.
    - Sort the loops by enclosed area from small to large, breaking ties by the loop's topmost row.
    - Build an output whose height is the number of loops and whose width is the largest loop pixel count.
      Each sorted loop becomes one row-colored stripe whose width equals that loop's pixel count.

    Key insight:
    - The same object is measured in two different ways: enclosed area controls vertical order, while
      loop pixel count controls stripe width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = []
    for comp in _monochrome_components(g):
        color = int(g[tuple(comp[0])])
        objects.append(
            {
                "color": color,
                "pixels": int(len(comp)),
                "area": _enclosed_area(comp),
                "top": int(comp[:, 0].min()),
            }
        )

    assert objects, "expected at least one loop"
    objects.sort(key=lambda item: (item["area"], item["top"]))

    width = max(int(item["pixels"]) for item in objects)
    out = np.zeros((len(objects), width), dtype=int)
    for r, item in enumerate(objects):
        out[r, : int(item["pixels"])] = int(item["color"])
    return out


def _rectangle_loop_patch(inner_h: int, inner_w: int, color: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _candidate_shapes() -> list[dict[str, int]]:
    shapes = []
    for inner_h in INNER_DIM_RANGE:
        for inner_w in INNER_DIM_RANGE:
            pixels = 2 * (inner_h + inner_w) + 4
            if pixels > MAX_RING_PIXELS:
                continue
            shapes.append(
                {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                    "area": inner_h * inner_w,
                    "pixels": pixels,
                }
            )
    return shapes


CANDIDATE_SHAPES = _candidate_shapes()


def _sample_shapes(count: int) -> list[dict[str, int]] | None:
    for _ in range(400):
        shapes = random.sample(CANDIDATE_SHAPES, count)
        areas = [shape["area"] for shape in shapes]
        pixels = [shape["pixels"] for shape in shapes]
        if len(set(areas)) != count or len(set(pixels)) != count:
            continue
        area_order = sorted(range(count), key=lambda i: areas[i])
        pixel_order = sorted(range(count), key=lambda i: pixels[i])
        if area_order == pixel_order:
            continue
        return [dict(shape) for shape in shapes]
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + outer_h + 1)
    c1 = min(w, left + outer_w + 1)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + outer_h + 1)
    c1 = min(w, left + outer_w + 1)
    blocked[r0:r1, c0:c1] = True


def _place_shapes(height: int, width: int, shapes: list[dict[str, int]]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(shapes)),
        key=lambda i: shapes[i]["outer_h"] * shapes[i]["outer_w"],
        reverse=True,
    )

    for idx in order:
        shape = shapes[idx]
        outer_h = int(shape["outer_h"])
        outer_w = int(shape["outer_w"])
        rows = list(range(1, height - outer_h))
        cols = list(range(1, width - outer_w))
        if not rows or not cols:
            return None

        candidates = [(r, c) for r in rows for c in cols]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if not _can_place(blocked, top, left, outer_h, outer_w):
                continue
            patch = _rectangle_loop_patch(
                int(shape["inner_h"]),
                int(shape["inner_w"]),
                int(shape["color"]),
            )
            grid[top : top + outer_h, left : left + outer_w] = patch
            _mark_blocked(blocked, top, left, outer_h, outer_w)
            shape["top"] = top
            shape["left"] = left
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_output(shapes: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(shapes, key=lambda shape: (shape["area"], shape["top"]))
    width = max(int(shape["pixels"]) for shape in ordered)
    out = np.zeros((len(ordered), width), dtype=int)
    for r, shape in enumerate(ordered):
        out[r, : int(shape["pixels"])] = int(shape["color"])
    return out


def _valid_training_signal(shapes: list[dict[str, int]]) -> bool:
    count = len(shapes)
    area_order = sorted(range(count), key=lambda i: (shapes[i]["area"], shapes[i]["top"]))
    top_order = sorted(range(count), key=lambda i: shapes[i]["top"])
    if area_order == top_order:
        return False
    if len({int(shape["top"]) for shape in shapes}) != count:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    shapes = _sample_shapes(count)
    if shapes is None:
        return None

    colors = random.sample(PALETTE, count)
    for shape, color in zip(shapes, colors):
        shape["color"] = color

    min_height = max(GRID_HEIGHT_RANGE[0], max(int(shape["outer_h"]) for shape in shapes) + 2)
    min_width = max(
        GRID_WIDTH_RANGE[0],
        max(int(shape["outer_w"]) for shape in shapes) + 2,
        max(int(shape["pixels"]) for shape in shapes),
    )
    if min_height > GRID_HEIGHT_RANGE[1] or min_width > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    grid = _place_shapes(height, width, shapes)
    if grid is None or not _valid_training_signal(shapes):
        return None

    expected = _expected_output(shapes)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
