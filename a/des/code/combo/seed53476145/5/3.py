from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 1
MARKER_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
REGION_COUNT_RANGE = (1, 4)
OUTER_HEIGHT_RANGE = (5, 9)
OUTER_WIDTH_RANGE = (5, 9)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    return cells


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    if gr == r0:
        return "top"
    if gr == r1:
        return "bottom"
    if gc == c0:
        return "left"
    if gc == c1:
        return "right"
    raise AssertionError("gap must lie on the rectangle boundary")


def _extract_regions(grid: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    wall_mask = grid == WALL
    regions: list[tuple[int, int, int, int, str]] = []
    for comp in _components(wall_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "wall rectangles must enclose an interior"

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        perimeter_set = set(perimeter)
        comp_set = set(comp)
        assert comp_set <= perimeter_set, "wall component must lie on a rectangle perimeter"

        missing = [cell for cell in perimeter if int(grid[cell]) != WALL]
        assert len(missing) == 1, "each wall must have exactly one single-cell gap"
        gap = missing[0]
        assert int(grid[gap]) == BG, "the wall gap must be background"
        assert gap[0] not in (r0, r1) or gap[1] not in (c0, c1), "gap must not be on a corner"
        assert len(comp) == len(perimeter) - 1, "rectangle perimeter must be complete except for the gap"
        assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == WALL), "walls must be one-pixel thick"

        regions.append((r0, c0, r1, c1, _gap_side(gap, r0, c0, r1, c1)))

    regions.sort()
    assert regions, "expected at least one blue walled region"
    return regions


def _compress_region(
    source: np.ndarray,
    target: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
) -> None:
    interior = source[r0 + 1 : r1, c0 + 1 : c1]
    target[r0 + 1 : r1, c0 + 1 : c1] = BG

    if side in ("top", "bottom"):
        for local_c in range(interior.shape[1]):
            markers = [
                (local_r, int(interior[local_r, local_c]))
                for local_r in range(interior.shape[0])
                if int(interior[local_r, local_c]) not in (BG, WALL)
            ]
            if side == "top":
                markers.sort(key=lambda item: item[0])
                for depth, (_row, color) in enumerate(markers):
                    target[r0 + 1 + depth, c0 + 1 + local_c] = color
            else:
                markers.sort(key=lambda item: item[0], reverse=True)
                for depth, (_row, color) in enumerate(markers):
                    target[r1 - 1 - depth, c0 + 1 + local_c] = color
    else:
        for local_r in range(interior.shape[0]):
            markers = [
                (local_c, int(interior[local_r, local_c]))
                for local_c in range(interior.shape[1])
                if int(interior[local_r, local_c]) not in (BG, WALL)
            ]
            if side == "left":
                markers.sort(key=lambda item: item[0])
                for depth, (_col, color) in enumerate(markers):
                    target[r0 + 1 + local_r, c0 + 1 + depth] = color
            else:
                markers.sort(key=lambda item: item[0], reverse=True)
                for depth, (_col, color) in enumerate(markers):
                    target[r0 + 1 + local_r, c1 - 1 - depth] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each region's colored interior pixels toward the unique wall gap.

    The gap side defines the gravity direction:
    - top gap: stack upward inside each column
    - bottom gap: stack downward inside each column
    - left gap: stack leftward inside each row
    - right gap: stack rightward inside each row
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1, side in _extract_regions(g):
        _compress_region(g, out, r0, c0, r1, c1, side)
    return out


def _draw_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    side: str,
) -> tuple[int, int, int, int, str]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    for r, c in _perimeter_cells(r0, c0, r1, c1):
        grid[r, c] = WALL

    if side == "top":
        gap = (r0, random.randint(c0 + 1, c1 - 1))
    elif side == "bottom":
        gap = (r1, random.randint(c0 + 1, c1 - 1))
    elif side == "left":
        gap = (random.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c1)
    grid[gap] = BG

    return (r0, c0, r1, c1, side)


def _sample_marker_positions(inner_h: int, inner_w: int, side: str) -> list[tuple[int, int, int]]:
    if side in ("top", "bottom"):
        depth_count = inner_h
        axis_values = sorted(random.sample(range(inner_w), random.randint(1, min(inner_w, 4))))
        stack_axis = random.choice(axis_values) if depth_count >= 2 and random.random() < 0.5 else None
        counts = []
        for axis in axis_values:
            if axis == stack_axis:
                counts.append(random.randint(2, min(3, depth_count)))
            else:
                counts.append(1 if random.random() < 0.65 else min(2, depth_count))
        while sum(counts) > len(MARKER_COLORS):
            shrinkable = [idx for idx, count in enumerate(counts) if count > 1]
            if not shrinkable:
                break
            counts[random.choice(shrinkable)] -= 1
        total = sum(counts)
        assert total <= len(MARKER_COLORS)
        colors = random.sample(MARKER_COLORS, total)
        markers: list[tuple[int, int, int]] = []
        color_idx = 0
        for axis, count in zip(axis_values, counts, strict=False):
            depths = random.sample(range(depth_count), count)
            for depth in depths:
                markers.append((depth, axis, colors[color_idx]))
                color_idx += 1
        return markers

    depth_count = inner_w
    axis_values = sorted(random.sample(range(inner_h), random.randint(1, min(inner_h, 4))))
    stack_axis = random.choice(axis_values) if depth_count >= 2 and random.random() < 0.5 else None
    counts = []
    for axis in axis_values:
        if axis == stack_axis:
            counts.append(random.randint(2, min(3, depth_count)))
        else:
            counts.append(1 if random.random() < 0.65 else min(2, depth_count))
    while sum(counts) > len(MARKER_COLORS):
        shrinkable = [idx for idx, count in enumerate(counts) if count > 1]
        if not shrinkable:
            break
        counts[random.choice(shrinkable)] -= 1
    total = sum(counts)
    assert total <= len(MARKER_COLORS)
    colors = random.sample(MARKER_COLORS, total)
    markers = []
    color_idx = 0
    for axis, count in zip(axis_values, counts, strict=False):
        depths = random.sample(range(depth_count), count)
        for depth in depths:
            markers.append((axis, depth, colors[color_idx]))
            color_idx += 1
    return markers


def _place_regions(grid: np.ndarray, region_count: int) -> list[tuple[int, int, int, int, str]] | None:
    h, w = grid.shape
    specs: list[tuple[int, int, int, int, str]] = []
    occupied = np.zeros_like(grid, dtype=bool)

    for _ in range(region_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            outer_h = random.randint(OUTER_HEIGHT_RANGE[0], min(OUTER_HEIGHT_RANGE[1], h - 2))
            outer_w = random.randint(OUTER_WIDTH_RANGE[0], min(OUTER_WIDTH_RANGE[1], w - 2))
            max_r0 = h - outer_h - 1
            max_c0 = w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            side = random.choice(SIDES)
            spec = _draw_region(grid, r0, c0, outer_h, outer_w, side)
            specs.append(spec)
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return specs


def _paint_markers(grid: np.ndarray, spec: tuple[int, int, int, int, str]) -> None:
    r0, c0, r1, c1, side = spec
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    for local_r, local_c, color in _sample_marker_positions(inner_h, inner_w, side):
        grid[r0 + 1 + local_r, c0 + 1 + local_c] = color


def _expected_from_specs(
    grid: np.ndarray, specs: list[tuple[int, int, int, int, str]]
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, side in sorted(specs):
        _compress_region(grid, out, r0, c0, r1, c1, side)
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    max_regions = min(REGION_COUNT_RANGE[1], max(1, (h * w) // 60))
    region_count = random.randint(REGION_COUNT_RANGE[0], max_regions)
    specs = _place_regions(grid, region_count)
    if specs is None:
        return None

    for spec in specs:
        _paint_markers(grid, spec)

    extracted = _extract_regions(grid)
    if extracted != sorted(specs):
        return None

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
