from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 8
MAX_AREA_BY_COUNT = {2: 42, 3: 32, 4: 24}

GRID_H_MIN = 12
GRID_H_MAX = 28
GRID_W_MIN = 12
GRID_W_MAX = 30

PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor noise-obscured rectangles by ranking their recovered areas and applying the bottom strip colors.

    Input structure:
    - The background is black.
    - Above the last row sit several non-overlapping solid rectangles of distinct colors.
    - Some rectangle pixels are replaced by gray noise, but each rectangle still has visible pixels of its own color.
    - The bottom row contains one-pixel color markers read from left to right, with black gaps allowed between them.

    Rule:
    - Ignore the bottom row and, for each non-black, non-gray color above it, recover that rectangle's full bounding box from the visible pixels of that color.
    - Treat any gray pixels inside the recovered box as covered rectangle pixels and compute the rectangle's true area from the full box size.
    - Read the bottom-row marker colors from left to right and sort the recovered rectangles from smallest area to largest area.
    - Repaint each full rectangle box with the correspondingly ranked marker color, and erase the entire bottom row to black.

    Key insight:
    - The visible colored remnants identify each hidden rectangle's full extent, while the bottom strip is not a legend by color identity but a rank-ordered recoloring list tied to rectangle area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2, "expected at least one content row and one strip row"

    source = g[:-1]
    strip_colors = [int(v) for v in g[-1] if int(v) != BG]
    assert strip_colors, "expected at least one bottom-strip color"

    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(source):
        color = int(color)
        if color in (BG, GREY):
            continue
        coords = np.argwhere(source == color)
        bbox = _bbox(coords)
        rectangles.append((_area(bbox), bbox))

    assert len(rectangles) == len(strip_colors), "strip length must match rectangle count"
    areas = [area for area, _bbox_ in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must have a unique order"

    out = g.copy()
    out[-1, :] = BG
    for (_, bbox), new_color in zip(sorted(rectangles, key=lambda item: item[0]), strip_colors):
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(new_color)

    return out


def _sample_shapes(rng: np.random.Generator, rect_count: int) -> list[tuple[int, int]] | None:
    max_area = MAX_AREA_BY_COUNT[rect_count]
    candidates = [
        (h, w)
        for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)
        for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)
        if h * w <= max_area
    ]
    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for idx in rng.permutation(len(candidates)):
        shape = candidates[int(idx)]
        if shape[0] * shape[1] in used_areas:
            continue
        chosen.append(shape)
        used_areas.add(shape[0] * shape[1])
        if len(chosen) == rect_count:
            return chosen
    return None


def _choose_grid_shape(
    rng: np.random.Generator, shapes: list[tuple[int, int]]
) -> tuple[int, int] | None:
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    total_area = sum(h * w for h, w in shapes)
    rect_count = len(shapes)

    min_h = max(GRID_H_MIN, max_h + 4)
    min_w = max(GRID_W_MIN, max_w + 4, 2 * rect_count - 1)
    if min_h > GRID_H_MAX or min_w > GRID_W_MAX:
        return None

    for _ in range(80):
        height = int(rng.integers(min_h, GRID_H_MAX + 1))
        width = int(rng.integers(min_w, GRID_W_MAX + 1))
        if (height - 1) * width >= total_area * 3:
            return height, width
    return None


def _place_rectangles(
    rng: np.random.Generator, work_h: int, work_w: int, shapes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((work_h, work_w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []

    for rect_h, rect_w in sorted(shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
        if work_h - rect_h <= 1 or work_w - rect_w <= 1:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(1, work_h - rect_h))
            c0 = int(rng.integers(1, work_w - rect_w))
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placements.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    return placements


def _paint_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _scatter_noise(
    rng: np.random.Generator, grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> None:
    r0, c0, r1, c1 = bbox
    reserved = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in reserved
    ]
    max_noise = min(len(candidates), max(1, _area(bbox) // 3))
    noise_count = int(rng.integers(1, max_noise + 1))
    for idx in np.atleast_1d(rng.choice(len(candidates), size=noise_count, replace=False)):
        r, c = candidates[int(idx)]
        grid[r, c] = GREY


def _spaced_columns(
    rng: np.random.Generator, width: int, count: int
) -> list[int] | None:
    if width < 2 * count - 1:
        return None
    base_positions = np.sort(rng.choice(width - count + 1, size=count, replace=False))
    return [int(base + offset) for offset, base in enumerate(base_positions)]


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[int, int, int, int]],
    strip_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    out[-1, :] = BG
    for bbox, new_color in zip(sorted(placements, key=_area), strip_colors):
        _paint_rectangle(out, bbox, new_color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = int(rng.integers(RECT_COUNT_RANGE[0], RECT_COUNT_RANGE[1] + 1))
    shapes = _sample_shapes(rng, rect_count)
    if shapes is None:
        return None

    grid_shape = _choose_grid_shape(rng, shapes)
    if grid_shape is None:
        return None
    height, width = grid_shape

    placements = _place_rectangles(rng, height - 1, width, shapes)
    if placements is None:
        return None

    colors = [int(v) for v in rng.choice(PALETTE, size=2 * rect_count, replace=False)]
    rect_colors = colors[:rect_count]
    strip_colors = colors[rect_count:]

    grid = np.full((height, width), BG, dtype=int)
    for bbox, color in zip(placements, rect_colors):
        _paint_rectangle(grid, bbox, color)
        _scatter_noise(rng, grid, bbox)

    strip_cols = _spaced_columns(rng, width, rect_count)
    if strip_cols is None:
        return None
    for col, color in zip(strip_cols, strip_colors):
        grid[-1, col] = color

    expected = _expected_output(grid, placements, strip_colors)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
