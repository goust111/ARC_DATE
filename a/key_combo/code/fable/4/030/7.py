from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _chebyshev_distance_to_bbox(
    r: int, c: int, bbox: tuple[int, int, int, int]
) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(max(dr, dc))


def _paint_layers(
    out: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    ordered_layers: list[tuple[int, int]],
) -> None:
    top, left, bottom, right = inner_bbox
    for i, (color, thickness) in enumerate(ordered_layers):
        if i == len(ordered_layers) - 1:
            out[top : bottom + 1, left : right + 1] = int(color)
            return

        t = int(thickness)
        height = bottom - top + 1
        width = right - left + 1
        assert height > 2 * t and width > 2 * t, "non-final layer must leave an inner rectangle"

        out[top : top + t, left : right + 1] = color
        out[bottom - t + 1 : bottom + 1, left : right + 1] = color
        out[top : bottom + 1, left : left + t] = color
        out[top : bottom + 1, right - t + 1 : right + 1] = color

        top += t
        left += t
        bottom -= t
        right -= t


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deposit the outside color dots into the frame interior as distance-ordered rectangular bands.

    Input structure:
    - The grid has a dominant background, one single-color hollow rectangular frame, and several
      outside dot colors whose pixels are all single-cell scattered dots.
    - The frame interior is empty in the input.
    - Each dot color has a unique count and a unique minimum Chebyshev distance to the frame.

    Rule:
    - Identify the hollow rectangular frame and erase every non-frame dot.
    - For each dot color, measure the minimum Chebyshev distance from any of its dots to the frame.
    - Sort colors from nearest to farthest, then fill the frame interior from outside to inside:
      every non-final color paints a rectangular ring whose thickness equals that color's dot count,
      and the final color fills the remaining center.

    Key insight:
    - Layer order comes from distance to the frame, while layer thickness comes from dot count;
      those two rankings are independent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_candidates = []
    for color in _colors_present(g, exclude={bg}):
        box = _frame_bbox(g == color)
        if box is not None:
            frame_candidates.append((int(color), box))
    assert len(frame_candidates) == 1, "expected exactly one hollow rectangular frame"

    frame_color, frame_bbox = frame_candidates[0]
    r0, c0, r1, c1 = frame_bbox
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "expected an empty frame interior"

    layer_specs: list[tuple[int, int, int]] = []
    for color in _colors_present(g, exclude={bg, frame_color}):
        coords = np.argwhere(g == color)
        assert len(coords) >= 1, "each dot color must appear at least once"
        distances = []
        for rr, cc in coords:
            r = int(rr)
            c = int(cc)
            assert not (r0 <= r <= r1 and c0 <= c <= c1), "dot colors must lie outside the frame"
            distances.append(_chebyshev_distance_to_bbox(r, c, frame_bbox))
        min_distance = min(distances)
        assert min_distance >= 1, "outside dots must have positive distance to the frame"
        layer_specs.append((min_distance, len(coords), int(color)))

    assert layer_specs, "expected at least one outside dot color"
    distances = [distance for distance, _count, _color in layer_specs]
    counts = [count for _distance, count, _color in layer_specs]
    assert len(set(distances)) == len(distances), "expected unique minimum distances"
    assert len(set(counts)) == len(counts), "expected unique dot counts"

    ordered_layers = [
        (color, count)
        for distance, count, color in sorted(layer_specs, key=lambda item: item[0])
    ]

    out = np.full_like(g, bg)
    out[g == frame_color] = frame_color
    _paint_layers(out, (r0 + 1, c0 + 1, r1 - 1, c1 - 1), ordered_layers)
    return out


PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


def _ring_cells(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], distance: int
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    d = int(distance)
    cells = []

    top = r0 - d
    bottom = r1 + d
    left = c0 - d
    right = c1 + d

    for c in range(left, right + 1):
        if 0 <= top < h and 0 <= c < w:
            cells.append((top, c))
        if 0 <= bottom < h and 0 <= c < w and bottom != top:
            cells.append((bottom, c))
    for r in range(top + 1, bottom):
        if 0 <= r < h and 0 <= left < w:
            cells.append((r, left))
        if 0 <= r < h and 0 <= right < w and right != left:
            cells.append((r, right))
    random.shuffle(cells)
    return cells


def _sample_non_monotone_counts(n_colors: int) -> list[int]:
    count_pool = list(range(1, 6))
    for _ in range(200):
        counts = random.sample(count_pool, n_colors)
        if counts != sorted(counts) and counts != sorted(counts, reverse=True):
            return counts
    raise RuntimeError("failed to sample non-monotone counts")


def _build_expected(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    ordered_layers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    _paint_layers(out, (r0 + 1, c0 + 1, r1 - 1, c1 - 1), ordered_layers)
    return out


def _attempt_generate() -> np.ndarray | None:
    bg = 0
    n_colors = random.choice((3, 4))
    all_colors = random.sample(PALETTE, n_colors + 1)
    frame_color = all_colors[0]
    dot_colors = all_colors[1:]

    sorted_distances = sorted(random.sample(range(1, 5), n_colors))
    distance_order_colors = dot_colors[:]
    random.shuffle(distance_order_colors)
    counts_by_distance = _sample_non_monotone_counts(n_colors)
    ordered_layers = list(zip(distance_order_colors, counts_by_distance))
    last_thickness = counts_by_distance[-1]

    center_h = random.randint(1, max(1, 2 * last_thickness))
    center_w = random.randint(1, max(1, 2 * last_thickness))
    outer_sum = sum(counts_by_distance[:-1])
    inner_h = center_h + 2 * outer_sum
    inner_w = center_w + 2 * outer_sum
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    max_distance = max(sorted_distances)
    min_pad = max_distance + 1
    if frame_h + 2 * min_pad > 30 or frame_w + 2 * min_pad > 30:
        return None

    top_pad = random.randint(min_pad, 30 - frame_h - min_pad)
    left_pad = random.randint(min_pad, 30 - frame_w - min_pad)
    bottom_pad = random.randint(min_pad, 30 - frame_h - top_pad)
    right_pad = random.randint(min_pad, 30 - frame_w - left_pad)

    height = frame_h + top_pad + bottom_pad
    width = frame_w + left_pad + right_pad
    if height > 30 or width > 30:
        return None

    r0 = top_pad
    c0 = left_pad
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    frame_bbox = (r0, c0, r1, c1)

    grid = np.full((height, width), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    distance_by_color = dict(zip(distance_order_colors, sorted_distances))
    count_by_color = dict(zip(distance_order_colors, counts_by_distance))

    for color in dot_colors:
        distance = distance_by_color[color]
        count = count_by_color[color]
        cells = _ring_cells(grid.shape, frame_bbox, distance)
        if len(cells) < count:
            return None
        for r, c in cells[:count]:
            grid[r, c] = color

    expected = _build_expected(grid.shape, bg, frame_color, frame_bbox, ordered_layers)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
