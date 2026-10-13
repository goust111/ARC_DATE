from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
SEPARATOR = 4
NEUTRAL = 5
MARKER_COLORS = (BLUE, RED, GREEN)

REGION_SIZE_RANGE = (4, 7)  # sampled with randint(low, high): 4..6
ACTIVE_COUNT_RANGE = (4, 7)  # 4..6 active regions
MAX_ATTEMPTS = 400
REGION_ATTEMPTS = 80

SHAPE_TEMPLATES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid, dtype=int).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands)
    return bands


def _region_marker_mask(region: np.ndarray, bg: int, separator: int) -> np.ndarray:
    return (region != bg) & (region != separator) & (region != NEUTRAL)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate exactly-two-marker regions by repainting their neutral component with the shared marker color, then erase all markers.

    Input structure
    - The background is black and one color forms full separator rows and full separator columns.
    - The separator lines partition the grid into a 3x3 table of rectangular regions.
    - Each region contains one 4-connected grey neutral component and either one or two single-pixel colored markers from red, blue, or green.
    Rule
    - Find the full separator rows and columns, then inspect each partitioned region independently.
    - Count the non-background, non-separator, non-grey marker pixels in the region.
    - If a region has exactly two markers, those two markers share one color; recolor the whole grey component to that color.
    - If a region has exactly one marker, keep the grey component grey.
    - In every region erase all marker pixels to background, while keeping separator lines and background unchanged.
    Key insight
    - The same marker pixels have a dual role: their count decides whether the region activates, and their shared color supplies the repaint for the neutral component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _get_bg(g)
    foreground = _colors_present(g, exclude={bg})
    separator_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in foreground:
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if sep_rows and sep_cols:
            separator_candidates.append((color, sep_rows, sep_cols))
    assert len(separator_candidates) == 1, "expected exactly one separator color"

    separator, sep_rows, sep_cols = separator_candidates[0]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 partition"

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = g[r0:r1, c0:c1]
            grey_components = _components(region == NEUTRAL)
            assert len(grey_components) == 1, "each region must contain one grey component"

            marker_mask = _region_marker_mask(region, bg, separator)
            marker_coords = np.argwhere(marker_mask)
            marker_count = int(marker_coords.shape[0])
            assert marker_count in (1, 2), "each region must contain one or two markers"

            marker_colors = {int(region[r, c]) for r, c in marker_coords}
            assert len(marker_colors) == 1, "markers within a region must share one visible color"
            marker_color = next(iter(marker_colors))

            for rr, cc in marker_coords:
                out[r0 + int(rr), c0 + int(cc)] = bg

            if marker_count == 2:
                for rr, cc in grey_components[0]:
                    out[r0 + rr, c0 + cc] = marker_color

    return out


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _transform_shape(shape: tuple[tuple[int, int], ...], rotation: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = [(r, -c) if flip else (r, c) for r, c in shape]
    for _ in range(rotation):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(tuple(pts))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {
        _transform_shape(shape, rotation=rotation, flip=flip)
        for rotation in range(4)
        for flip in (False, True)
    }
    return sorted(variants)


ALL_SHAPES = [variant for template in SHAPE_TEMPLATES for variant in _shape_variants(template)]


def _boundary_candidates(cells: set[tuple[int, int]], region_h: int, region_w: int) -> list[tuple[int, int]]:
    candidates = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < region_h and 0 <= nc < region_w and (nr, nc) not in cells:
                candidates.add((nr, nc))
    return sorted(candidates)


def _pick_marker_positions(candidates: list[tuple[int, int]], marker_count: int) -> list[tuple[int, int]] | None:
    if len(candidates) < marker_count:
        return None
    if marker_count == 1:
        return [candidates[int(np.random.randint(len(candidates)))]]

    good_pairs = []
    fallback_pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            fallback_pairs.append((a, b))
            if dist >= 2:
                good_pairs.append((a, b))
    pairs = good_pairs if good_pairs else fallback_pairs
    if not pairs:
        return None
    a, b = pairs[int(np.random.randint(len(pairs)))]
    return [a, b]


def _sample_region_shape(region_h: int, region_w: int, marker_count: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    choices = ALL_SHAPES[:]
    np.random.shuffle(choices)
    for shape in choices:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        if max_r >= region_h or max_c >= region_w:
            continue
        offset_r = int(np.random.randint(0, region_h - max_r))
        offset_c = int(np.random.randint(0, region_w - max_c))
        cells = {(offset_r + r, offset_c + c) for r, c in shape}
        candidates = _boundary_candidates(cells, region_h, region_w)
        markers = _pick_marker_positions(candidates, marker_count)
        if markers is not None:
            return cells, markers
    return None


def _row_starts(sizes: np.ndarray) -> list[int]:
    starts = [0]
    for size in sizes[:-1]:
        starts.append(starts[-1] + int(size) + 1)
    return starts


def _assign_active_colors(active_indices: list[int]) -> dict[int, int]:
    chosen_count = int(np.random.randint(2, len(MARKER_COLORS) + 1))
    palette = list(np.random.choice(np.array(MARKER_COLORS, dtype=int), size=chosen_count, replace=False))
    np.random.shuffle(active_indices)

    colors: dict[int, int] = {}
    for idx, color in zip(active_indices, palette):
        colors[idx] = int(color)
    for idx in active_indices[len(palette) :]:
        colors[idx] = int(palette[int(np.random.randint(len(palette)))])
    return colors


def _build_example() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    region_heights = np.random.randint(*REGION_SIZE_RANGE, size=3)
    region_widths = np.random.randint(*REGION_SIZE_RANGE, size=3)
    grid_h = int(region_heights.sum() + 2)
    grid_w = int(region_widths.sum() + 2)
    if max(grid_h, grid_w) > 30:
        return None

    input_grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = input_grid.copy()
    row_starts = _row_starts(region_heights)
    col_starts = _row_starts(region_widths)

    sep_rows = [row_starts[1] - 1, row_starts[2] - 1]
    sep_cols = [col_starts[1] - 1, col_starts[2] - 1]
    input_grid[sep_rows, :] = SEPARATOR
    input_grid[:, sep_cols] = SEPARATOR
    expected[sep_rows, :] = SEPARATOR
    expected[:, sep_cols] = SEPARATOR

    active_count = int(np.random.randint(*ACTIVE_COUNT_RANGE))
    region_indices = list(range(9))
    active_indices = list(np.random.choice(np.array(region_indices, dtype=int), size=active_count, replace=False))
    active_set = set(active_indices)
    inactive_count = 9 - active_count
    assert active_count >= 4 and inactive_count >= 3

    active_colors = _assign_active_colors(active_indices[:])
    metadata: list[dict] = []

    for idx in region_indices:
        rr, cc = divmod(idx, 3)
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        region_h = int(region_heights[rr])
        region_w = int(region_widths[cc])

        marker_count = 2 if idx in active_set else 1
        marker_color = active_colors[idx] if idx in active_set else int(MARKER_COLORS[int(np.random.randint(len(MARKER_COLORS)))])

        region_result = None
        for _ in range(REGION_ATTEMPTS):
            region_result = _sample_region_shape(region_h, region_w, marker_count)
            if region_result is not None:
                break
        if region_result is None:
            return None

        cells, marker_positions = region_result
        for r, c in cells:
            input_grid[r0 + r, c0 + c] = NEUTRAL
            expected[r0 + r, c0 + c] = marker_color if idx in active_set else NEUTRAL
        for r, c in marker_positions:
            input_grid[r0 + r, c0 + c] = marker_color

        metadata.append(
            {
                "index": idx,
                "bbox": (r0, c0, region_h, region_w),
                "cells": cells,
                "marker_positions": marker_positions,
                "marker_count": marker_count,
                "marker_color": marker_color,
                "active": idx in active_set,
            }
        )

    return input_grid, expected, metadata


def _validate_generated_example(input_grid: np.ndarray, expected: np.ndarray, metadata: list[dict]) -> bool:
    h, w = input_grid.shape
    if input_grid.ndim != 2 or expected.ndim != 2 or input_grid.shape != expected.shape:
        return False
    if max(h, w) > 30:
        return False

    sep_rows = [r for r in range(h) if np.all(input_grid[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(w) if np.all(input_grid[:, c] == SEPARATOR)]
    if len(sep_rows) != 2 or len(sep_cols) != 2:
        return False

    active_colors = set()
    active_count = 0
    inactive_count = 0
    for region in metadata:
        r0, c0, region_h, region_w = region["bbox"]
        region_view = input_grid[r0 : r0 + region_h, c0 : c0 + region_w]
        cells = set(region["cells"])
        markers = list(region["marker_positions"])
        marker_color = int(region["marker_color"])
        marker_count = int(region["marker_count"])

        if marker_count != len(markers):
            return False
        if len(_components(region_view == NEUTRAL)) != 1:
            return False
        if not (3 <= len(cells) <= 6):
            return False
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1):
            return False

        for r, c in cells:
            if input_grid[r0 + r, c0 + c] != NEUTRAL:
                return False
        for r, c in markers:
            if input_grid[r0 + r, c0 + c] != marker_color:
                return False
            if not any((nr, nc) in cells for nr, nc in _neighbors4(r, c)):
                return False

        if marker_count == 2:
            active_count += 1
            active_colors.add(marker_color)
            for r, c in cells:
                if expected[r0 + r, c0 + c] != marker_color:
                    return False
        else:
            inactive_count += 1
            for r, c in cells:
                if expected[r0 + r, c0 + c] != NEUTRAL:
                    return False
        for r, c in markers:
            if expected[r0 + r, c0 + c] != BG:
                return False

    if active_count < 4 or inactive_count < 3 or len(active_colors) < 2:
        return False

    try:
        actual = main(input_grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        input_grid, expected, metadata = built
        if _validate_generated_example(input_grid, expected, metadata):
            return input_grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
