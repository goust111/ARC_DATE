from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

MARKER_COLORS = (RED, GREEN, YELLOW, PINK)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _layer_distance(r: int, c: int, h: int, w: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _ring_cells(h: int, w: int, dist: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if _layer_distance(r, c, h, w) == dist:
                cells.append((r, c))
    return cells


def _sorted_marker_palette(inner: np.ndarray) -> list[int]:
    markers: list[tuple[int, int]] = []
    h, w = inner.shape
    for r, c in np.argwhere(inner != BLACK):
        d = _layer_distance(int(r), int(c), h, w)
        markers.append((d, int(inner[r, c])))
    markers.sort()
    distances = [d for d, _color in markers]
    assert len(distances) == len(set(distances)), "marker distances must be unique"
    return [ORANGE] + [color for _d, color in markers]


def _paint_bands(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    palette = _sorted_marker_palette(inner)
    out = np.zeros((h, w), dtype=int)
    m = len(palette)
    for r in range(h):
        for c in range(w):
            d = _layer_distance(r, c, h, w)
            out[r, c] = palette[d % m]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the corner-marked frame and refill its interior with concentric cyclic color bands.

    Input structure:
    - A black background contains one non-black rectangular cluster.
    - Four orange corner pixels mark the corners of a rectangle, while blue pixels form a one-cell border along that rectangle's edges between the corners.
    - Inside the border is a black rectangular area containing 2-4 isolated marker pixels in distinct colors and at distinct Chebyshev depths from the interior boundary.

    Rule:
    - Find the four orange corner pixels and use them to define the rectangle that encloses the cluster.
    - Remove the outer one-cell border by shrinking that rectangle inward by one cell on each side; this cropped subgrid is the output size.
    - Build a compact palette whose first color is orange, then append the interior marker colors ordered by increasing Chebyshev distance from the cropped subgrid's boundary.
    - Recolor every cell of the cropped subgrid with palette[distance_to_boundary mod palette_length], overriding both the original markers and the black background.

    Key insight:
    - The corner color plays two roles at once: it locates the crop rectangle and also becomes the first color of the cyclic band palette, so boundary layer 0 is always orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    corner_coords = np.argwhere(g == ORANGE)
    assert len(corner_coords) == 4, "expected exactly four orange corner markers"
    bbox = _bbox(g == ORANGE)
    assert bbox is not None
    r0, c0, r1, c1 = bbox

    expected_corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    actual_corners = {tuple(map(int, rc)) for rc in corner_coords}
    assert actual_corners == expected_corners, "orange pixels must occupy the bbox corners"
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must contain a non-empty interior"

    inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    return _paint_bands(inner)


def _min_size_for_markers(marker_count: int) -> int:
    # Need enough depth to show two full palette cycles: max_d >= 2*palette_len - 1.
    needed_depth = 2 * (marker_count + 1) - 1
    return 2 * needed_depth + 1


def _sample_distances(depth_max: int, marker_count: int) -> list[int]:
    if marker_count == 2:
        patterns = (
            (0, 2),
            (1, 3),
            (0, depth_max),
            (max(0, depth_max - 2), depth_max),
        )
    elif marker_count == 3:
        patterns = (
            (0, 2, 4),
            (1, 3, 5),
            (0, 1, depth_max),
            (0, max(1, depth_max // 2), depth_max),
        )
    else:
        patterns = (
            (0, 2, 4, 6),
            (1, 3, 5, 7),
            (0, 1, 3, depth_max),
            (0, depth_max - 3, depth_max - 1, depth_max),
        )

    valid = [tuple(d for d in pattern if 0 <= d <= depth_max) for pattern in patterns]
    valid = [pattern for pattern in valid if len(pattern) == marker_count and len(set(pattern)) == marker_count]
    valid = [pattern for pattern in valid if max(pattern) <= depth_max]
    if valid and random.random() < 0.7:
        return list(random.choice(valid))

    return sorted(random.sample(range(depth_max + 1), marker_count))


def _place_markers(inner_h: int, inner_w: int, distances: Iterable[int], colors: Iterable[int]) -> np.ndarray | None:
    inner = np.zeros((inner_h, inner_w), dtype=int)
    used: set[tuple[int, int]] = set()
    for dist, color in zip(distances, colors, strict=True):
        candidates = _ring_cells(inner_h, inner_w, int(dist))
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            if (r, c) in used:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                continue
            inner[r, c] = int(color)
            used.add((r, c))
            placed = True
            break
        if not placed:
            return None
    return inner


def _build_grid() -> np.ndarray | None:
    marker_count = random.randint(2, 4)
    min_dim = _min_size_for_markers(marker_count)
    inner_h = random.randint(min_dim, min(min_dim + 3, 20))
    inner_w = random.randint(min_dim, min(min_dim + 4, 20))
    depth_max = min(inner_h, inner_w) // 2
    if _layer_distance(inner_h // 2, inner_w // 2, inner_h, inner_w) < 2 * (marker_count + 1) - 1:
        return None

    distances = _sample_distances(depth_max, marker_count)
    colors = random.sample(MARKER_COLORS, marker_count)
    inner = _place_markers(inner_h, inner_w, distances, colors)
    if inner is None:
        return None

    cluster = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    cluster[0, 0] = ORANGE
    cluster[0, -1] = ORANGE
    cluster[-1, 0] = ORANGE
    cluster[-1, -1] = ORANGE
    cluster[0, 1:-1] = BLUE
    cluster[-1, 1:-1] = BLUE
    cluster[1:-1, 0] = BLUE
    cluster[1:-1, -1] = BLUE
    cluster[1:-1, 1:-1] = inner

    pad_top = random.randint(0, 4)
    pad_bottom = random.randint(0, 4)
    pad_left = random.randint(0, 4)
    pad_right = random.randint(0, 4)
    grid_h = cluster.shape[0] + pad_top + pad_bottom
    grid_w = cluster.shape[1] + pad_left + pad_right
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[pad_top : pad_top + cluster.shape[0], pad_left : pad_left + cluster.shape[1]] = cluster

    solved = main(grid)
    if solved.shape != inner.shape:
        return None
    if not np.array_equal(solved, _paint_bands(inner)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
