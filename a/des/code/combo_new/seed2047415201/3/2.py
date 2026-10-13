import random

import numpy as np


AZURE = 8
COLOR_COUNT = 10

GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
RECT_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = (2, 10)
SAMPLE_COUNT_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 300
MAX_RECT_ATTEMPTS = 200
MAX_SAMPLE_ATTEMPTS = 200


def _colors_by_first_seen(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    seen = {int(bg)}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color not in seen:
                seen.add(color)
                colors.append(color)
    return colors


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand sparse colored samples into layered solid rectangles.

    Input structure:
    - The background is Azure.
    - Each non-Azure color appears as sparse pixels inside one continuous row
      band and one continuous column band.
    - Different colors may have overlapping row and column bands, but the sparse
      input pixels themselves are visible color samples.

    Rule:
    - For every foreground color, find the min and max row and column occupied
      by that color; these extremes define its rectangle.
    - Fill the full rectangle for each color, drawing larger-area rectangles
      before smaller-area rectangles.
    - Smaller rectangles are drawn later and overwrite larger rectangles in
      overlapping cells.
    - Cells outside all recovered rectangles remain Azure.

    Key insight:
    - The scattered pixels are samples of hidden rectangles; their color and
      extreme positions encode each rectangle, and area gives the visible
      layering order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = AZURE
    assert np.any(g == bg), "expected Azure background"

    rects: list[tuple[int, int, int, int, int, int]] = []
    for color in _colors_by_first_seen(g, bg):
        coords = np.argwhere(g == int(color))
        assert coords.size > 0, "foreground color must have at least one sample"
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rects.append((int(color), int(r0), int(r1), int(c0), int(c1), int(area)))

    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if rect_a[5] == rect_b[5]:
                box_a = (rect_a[1], rect_a[2], rect_a[3], rect_a[4])
                box_b = (rect_b[1], rect_b[2], rect_b[3], rect_b[4])
                assert not _rectangles_overlap(box_a, box_b), (
                    "overlapping rectangles must have different areas"
                )

    out = np.full(g.shape, bg, dtype=int)
    for color, r0, r1, c0, c1, _area_value in sorted(rects, key=lambda item: -item[5]):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _random_rect(height: int, width: int) -> tuple[int, int, int, int] | None:
    max_h = min(RECT_SIDE_RANGE[1], height)
    max_w = min(RECT_SIDE_RANGE[1], width)
    if max_h < RECT_SIDE_RANGE[0] or max_w < RECT_SIDE_RANGE[0]:
        return None

    rh = random.randint(RECT_SIDE_RANGE[0], max_h)
    rw = random.randint(RECT_SIDE_RANGE[0], max_w)
    r0 = random.randint(0, height - rh)
    c0 = random.randint(0, width - rw)
    return int(r0), int(r0 + rh - 1), int(c0), int(c0 + rw - 1)


def _overlap_pair(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(MAX_RECT_ATTEMPTS):
        large = _random_rect(height, width)
        if large is None:
            return None
        large_area = _area(large)

        small_h_max = min(7, height)
        small_w_max = min(7, width)
        if small_h_max < 2 or small_w_max < 2:
            return None
        sh = random.randint(2, small_h_max)
        sw = random.randint(2, small_w_max)
        if sh * sw >= large_area:
            continue

        lr0, lr1, lc0, lc1 = large
        r_min = max(0, lr0 - sh + 1)
        r_max = min(height - sh, lr1)
        c_min = max(0, lc0 - sw + 1)
        c_max = min(width - sw, lc1)
        if r_min > r_max or c_min > c_max:
            continue
        sr0 = random.randint(r_min, r_max)
        sc0 = random.randint(c_min, c_max)
        small = (int(sr0), int(sr0 + sh - 1), int(sc0), int(sc0 + sw - 1))
        if _rectangles_overlap(large, small):
            return [large, small]
    return None


def _sample_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]] | None:
    r0, r1, c0, c1 = rect
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    area = len(cells)
    if area <= 1:
        return None

    min_count = min(SAMPLE_COUNT_RANGE[0], area - 1)
    max_count = min(SAMPLE_COUNT_RANGE[1], area - 1)
    if min_count > max_count:
        return None

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        samples = {
            (r0, random.randint(c0, c1)),
            (r1, random.randint(c0, c1)),
            (random.randint(r0, r1), c0),
            (random.randint(r0, r1), c1),
        }
        target_count = random.randint(min_count, max_count)
        random.shuffle(cells)
        for cell in cells:
            if len(samples) >= target_count:
                break
            samples.add(cell)

        rows = [r for r, _ in samples]
        cols = [c for _, c in samples]
        if (
            len(samples) < area
            and min(rows) == r0
            and max(rows) == r1
            and min(cols) == c0
            and max(cols) == c1
        ):
            return samples
    return None


def _render_expected(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    out = np.full(shape, AZURE, dtype=int)
    ordered = sorted(zip(rects, colors), key=lambda item: -_area(item[0]))
    for rect, color in ordered:
        r0, r1, c0, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _valid_rect_set(rects: list[tuple[int, int, int, int]]) -> bool:
    areas = [_area(rect) for rect in rects]
    if len(set(areas)) != len(areas):
        return False

    has_overlap = False
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            if _rectangles_overlap(a, b):
                has_overlap = True
    return has_overlap


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    rects = _overlap_pair(height, width)
    if rects is None:
        return None

    used_areas = {_area(rect) for rect in rects}
    for _ in range(rect_count - len(rects)):
        placed = False
        for _attempt in range(MAX_RECT_ATTEMPTS):
            rect = _random_rect(height, width)
            if rect is None:
                return None
            area = _area(rect)
            if area in used_areas:
                continue
            rects.append(rect)
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return None

    if not _valid_rect_set(rects):
        return None

    colors = random.sample([c for c in range(COLOR_COUNT) if c != AZURE], len(rects))
    grid = np.full((height, width), AZURE, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for rect, color in zip(rects, colors):
        samples = _sample_cells(rect)
        if samples is None:
            return None

        # Keep the sparse input unambiguous: one visible sample color per cell.
        if any(cell in occupied for cell in samples):
            return None
        for r, c in samples:
            grid[r, c] = int(color)
            occupied.add((int(r), int(c)))

    expected = _render_expected(grid.shape, rects, colors)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-rectangle input")
