from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location
import random


Rect = tuple[int, int, int, int, int, int]  # (area, color, r0, c0, r1, c1)


def _solid_rectangles(g: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for color in np.unique(g):
        for coords in components_mask(g == int(color), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = int(coords.shape[0])
            if h > 1 and w > 1 and area == h * w:
                rects.append((area, int(color), r0, c0, r1, c1))
    return rects


def _period_values(
    g: np.ndarray, blocked: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    values: dict[tuple[int, int], int] = {}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            key = (r % row_period, c % col_period)
            value = int(g[r, c])
            if key in values and values[key] != value:
                return None
            values[key] = value
    return values


def _reconstruct_rect(
    values: dict[tuple[int, int], int], row_period: int, col_period: int, rect: Rect
) -> np.ndarray:
    _, _, r0, c0, r1, c1 = rect
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            key = (r % row_period, c % col_period)
            assert key in values, "period residue must be visible outside the rectangles"
            out[rr, cc] = values[key]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeating background pattern hidden under the largest solid rectangle.

    Input structure:
    - The grid is covered by a small row/column periodic pattern.
    - Several non-pattern colors form solid rectangular foreground objects that
      overwrite the pattern.
    - The requested output region is the largest of those solid rectangles.

    Rule:
    - Find every 4-connected monochromatic object whose bounding box is a filled
      rectangle with both dimensions greater than one.
    - Treat those rectangles as masks, infer the smallest repeating tile that is
      consistent with all remaining visible pixels, and repaint the largest
      rectangle's coordinates with that tile.

    Key insight: The rectangle colors are occluders; the visible pixels outside
    them are enough to reconstruct the original periodic background underneath.
    """
    g = np.asarray(grid, dtype=int)
    rects = _solid_rectangles(g)
    assert rects, "expected at least one solid rectangular occluder"

    max_area = max(area for area, *_ in rects)
    targets = [rect for rect in rects if rect[0] == max_area]
    assert len(targets) == 1, "largest rectangle must be unique"
    target = targets[0]

    blocked = np.zeros(g.shape, dtype=bool)
    for _, _, r0, c0, r1, c1 in rects:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    h, w = g.shape
    period_candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            values = _period_values(g, blocked, row_period, col_period)
            if values is not None:
                period_candidates.append((row_period, col_period, values))
    assert period_candidates, "visible pixels must define a periodic pattern"

    min_period_area = min(rp * cp for rp, cp, _ in period_candidates)
    outputs = [
        _reconstruct_rect(values, row_period, col_period, target)
        for row_period, col_period, values in period_candidates
        if row_period * col_period == min_period_area
    ]
    assert outputs, "expected at least one minimal period"
    first = outputs[0]
    assert all(np.array_equal(first, out) for out in outputs), "minimal periods must agree"
    return first

# --------------- generator (att2) ---------------

# Tile palette and period choices.
_GEN_COLOR_RANGE = range(10)
_GEN_TILE_SIZES = [3, 4]
_GEN_TILE_COLOR_COUNTS = [3, 4]
_GEN_STEP_CHOICES = {3: [1, 2], 4: [1, 3]}
# Canvas dimensions from the observed task family.
_GEN_HEIGHT_RANGE = (19, 23)
_GEN_WIDTH_RANGE = (19, 22)
# Occluder rectangle count, size, and coverage constraints.
_GEN_RECT_COUNTS = [2, 3]
_GEN_RECT_SIDE_RANGE = range(3, 13)
_GEN_AREA_RANGE = (9, 100)
_GEN_TARGET_AREA_RANGE = (36, 100)
_GEN_COVERAGE_RANGE = (0.18, 0.33)
# Rejection sampling budget.
_GEN_ATTEMPT_LIMIT = 300

_GEN_DIMS_BY_AREA = {
    h * w: [(hh, ww) for hh in _GEN_RECT_SIDE_RANGE for ww in _GEN_RECT_SIDE_RANGE if hh * ww == h * w]
    for h in _GEN_RECT_SIDE_RANGE
    for w in _GEN_RECT_SIDE_RANGE
}


def _gen_tile() -> np.ndarray:
    size = random.choice(_GEN_TILE_SIZES)
    n_colors = random.choice(_GEN_TILE_COLOR_COUNTS) if size == 3 else 4
    colors = random.sample(_GEN_COLOR_RANGE, n_colors)
    step_choices = _GEN_STEP_CHOICES[n_colors]
    row_step = random.choice(step_choices)
    col_step = random.choice(step_choices)
    offset = random.randrange(n_colors)
    tile = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            tile[r, c] = colors[(offset + row_step * r + col_step * c) % n_colors]
    return tile


def _gen_periodic_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].astype(int, copy=True)


def _gen_choose_areas(height: int, width: int, count: int) -> tuple[int, ...] | None:
    min_total = int(np.ceil(_GEN_COVERAGE_RANGE[0] * height * width))
    max_total = int(np.floor(_GEN_COVERAGE_RANGE[1] * height * width))
    area_min, area_max = _GEN_AREA_RANGE
    target_min, target_max = _GEN_TARGET_AREA_RANGE
    small_areas = sorted(a for a in _GEN_DIMS_BY_AREA if area_min <= a <= area_max)
    target_areas = [a for a in small_areas if target_min <= a <= target_max]
    possible: dict[int, list[tuple[int, ...]]] = {}
    for total in range(min_total, max_total + 1):
        tuples: list[tuple[int, ...]] = []
        for target in target_areas:
            if count == 2:
                small = total - target
                if small in _GEN_DIMS_BY_AREA and area_min <= small < target:
                    tuples.append((target, small))
            else:
                rem = total - target
                for small1 in small_areas:
                    small2 = rem - small1
                    if (
                        area_min <= small1 < target
                        and area_min <= small2 < target
                        and small2 in _GEN_DIMS_BY_AREA
                    ):
                        tuples.append((target, small1, small2))
        if tuples:
            possible[total] = tuples
    if not possible:
        return None
    total = random.choice(list(possible))
    return random.choice(possible[total])


def _gen_place_rectangles(shape: tuple[int, int], dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros(shape, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in dims:
        patch = np.ones((rect_h, rect_w), dtype=int)
        try:
            row, col = random_free_location(occupied, patch, transparent=0)
        except ValueError:
            return None
        paste(occupied, patch, row, col)
        placed.append((row, col, rect_h, rect_w))
    return placed


def _gen_all_residues_visible(occupied: np.ndarray, row_period: int, col_period: int) -> bool:
    seen = np.zeros((row_period, col_period), dtype=bool)
    for residue_r in range(row_period):
        for residue_c in range(col_period):
            seen[residue_r, residue_c] = np.any(
                occupied[residue_r::row_period, residue_c::col_period] == 0
            )
    return bool(seen.all())


def _gen_attempt() -> np.ndarray | None:
    tile = _gen_tile()
    height = random.randint(*_GEN_HEIGHT_RANGE)
    width = random.randint(*_GEN_WIDTH_RANGE)
    base = _gen_periodic_grid(tile, height, width)

    count = random.choice(_GEN_RECT_COUNTS)
    areas = _gen_choose_areas(height, width, count)
    if areas is None:
        return None

    dims = [random.choice(_GEN_DIMS_BY_AREA[area]) for area in areas]
    placed = _gen_place_rectangles((height, width), dims)
    if placed is None:
        return None

    occupied = np.zeros((height, width), dtype=int)
    for row, col, rect_h, rect_w in placed:
        occupied[row : row + rect_h, col : col + rect_w] = 1
    if not _gen_all_residues_visible(occupied, tile.shape[0], tile.shape[1]):
        return None

    target_row, target_col, target_h, target_w = placed[0]
    expected = base[target_row : target_row + target_h, target_col : target_col + target_w].copy()
    tile_colors = set(map(int, tile.ravel()))
    occluder_colors = random.sample([c for c in _GEN_COLOR_RANGE if c not in tile_colors], count)

    grid = base.copy()
    for color, (row, col, rect_h, rect_w) in zip(occluder_colors, placed):
        patch = np.full((rect_h, rect_w), color, dtype=int)
        paste(grid, patch, row, col)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_ATTEMPT_LIMIT} attempts")
