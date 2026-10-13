import random

import numpy as np


BG = 0
GEN_RETRIES = 500
LAYOUT_RETRIES = 120
RECT_TRIES = 100
RECT_COUNT_CHOICES = (2, 3, 4)
CONTENT_H_RANGE = (10, 22)
CONTENT_W_RANGE = (10, 22)
CORE_H_RANGE = (1, 3)
CORE_W_RANGE = (1, 3)
PARTIAL_OVERLAP_PROB = 0.5
ARC_COLORS = tuple(range(1, 10))


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _paint_rectangles(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    order = sorted(range(len(rects)), key=lambda idx: _rect_area(rects[idx]), reverse=True)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]
        canvas[r0 : r1 + 1, c0 : c1 + 1] = int(colors[idx])
    return canvas


def _coverage_counts(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def _recolor_by_counts(counts: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    out = np.zeros_like(counts, dtype=int)
    mask = counts > 0
    if np.any(mask):
        legend = np.array(legend_colors, dtype=int)
        out[mask] = legend[counts[mask] - 1]
    return out


def _contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _has_partial_pair(rects: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if not _contains(rects[i], rects[j]) and not _contains(rects[j], rects[i]):
                return True
    return False


def _all_depths_present(counts: np.ndarray) -> bool:
    max_depth = int(counts.max())
    return max_depth >= 2 and all(np.any(counts == depth) for depth in range(1, max_depth + 1))


def _recoverable_bboxes(
    content: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> bool:
    for rect, color in zip(rects, colors):
        pts = np.argwhere(content == int(color))
        if len(pts) == 0:
            return False
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != rect:
            return False
        if np.any(content[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1] == BG):
            return False
    return True


def _attach_legend(content: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    h, w = content.shape
    grid = np.zeros((h, w + 1), dtype=int)
    grid[:, :w] = content
    start = random.randint(0, h - len(legend_colors))
    for i, color in enumerate(legend_colors):
        grid[start + i, w] = int(color)
    return grid


def _sample_core_rectangles(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    want_partial = count >= 2 and random.random() < PARTIAL_OVERLAP_PROB
    for _ in range(LAYOUT_RETRIES):
        max_core_h = min(CORE_H_RANGE[1], h - 4)
        max_core_w = min(CORE_W_RANGE[1], w - 4)
        if max_core_h < CORE_H_RANGE[0] or max_core_w < CORE_W_RANGE[0]:
            return None

        core_h = random.randint(CORE_H_RANGE[0], max_core_h)
        core_w = random.randint(CORE_W_RANGE[0], max_core_w)
        core_r0 = random.randint(2, h - core_h - 3)
        core_c0 = random.randint(2, w - core_w - 3)
        core_r1 = core_r0 + core_h - 1
        core_c1 = core_c0 + core_w - 1

        rects: list[tuple[int, int, int, int]] = []
        used_rects: set[tuple[int, int, int, int]] = set()
        used_areas: set[int] = set()

        for _rect_idx in range(count):
            placed = False
            for _ in range(RECT_TRIES):
                top = random.randint(0, core_r0)
                bottom = random.randint(0, h - 1 - core_r1)
                left = random.randint(0, core_c0)
                right = random.randint(0, w - 1 - core_c1)
                r0 = core_r0 - top
                c0 = core_c0 - left
                r1 = core_r1 + bottom
                c1 = core_c1 + right
                if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
                    continue
                rect = (r0, c0, r1, c1)
                area = _rect_area(rect)
                if rect in used_rects or area in used_areas:
                    continue
                rects.append(rect)
                used_rects.add(rect)
                used_areas.add(area)
                placed = True
                break
            if not placed:
                break

        if len(rects) != count:
            continue
        if want_partial and not _has_partial_pair(rects):
            continue
        return rects
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor overlapping solid rectangles by their coverage depth using the right-edge legend.

    Input structure:
    - Background is black.
    - All columns except the last contain several differently colored filled axis-aligned rectangles that may overlap.
    - The last column is a vertical color legend: reading its non-black cells from top to bottom gives the output color for depth 1, depth 2, and so on.

    Rule:
    - Remove the legend column from consideration.
    - For each remaining non-black color, collect all visible pixels of that color and reconstruct its original rectangle as the tight bounding box of that exposed fragment set.
    - Count, for every content cell, how many reconstructed rectangles cover that position.
    - Replace each covered cell by the legend color indexed by that overlap count; keep black background unchanged.
    - Return only the content region, so the output is one column narrower than the input.

    Key insight:
    - The visible color only tells which rectangle is topmost there; the real task is to rebuild every full rectangle first, then remap cells from geometry-based overlap depth through the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a legend column"

    content = g[:, :-1]
    legend = [int(v) for v in g[:, -1] if int(v) != BG]
    assert legend, "legend column must contain at least one non-background color"

    rects: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(content)):
        if color == BG:
            continue
        pts = np.argwhere(content == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bbox = content[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(bbox != BG), "each rectangle's exposed pixels must reveal a solid bbox"
        rects.append((int(r0), int(c0), int(r1), int(c1)))

    counts = _coverage_counts(content.shape, rects)
    max_depth = int(counts.max())
    assert max_depth <= len(legend), "legend is shorter than the required overlap depth"
    return _recolor_by_counts(counts, legend)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*CONTENT_H_RANGE)
    w = random.randint(*CONTENT_W_RANGE)
    rect_count = random.choice(RECT_COUNT_CHOICES)
    rects = _sample_core_rectangles(h, w, rect_count)
    if rects is None:
        return None

    colors = random.sample(list(ARC_COLORS), 2 * rect_count)
    rect_colors = [int(c) for c in colors[:rect_count]]
    legend_colors = [int(c) for c in colors[rect_count:]]

    content = _paint_rectangles((h, w), rects, rect_colors)
    counts = _coverage_counts((h, w), rects)
    if int(counts.max()) != rect_count:
        return None
    if not _all_depths_present(counts):
        return None
    if not _recoverable_bboxes(content, rects, rect_colors):
        return None

    grid = _attach_legend(content, legend_colors)
    expected = _recolor_by_counts(counts, legend_colors)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangles input")
