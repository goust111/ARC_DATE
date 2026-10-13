from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
COLOR_POOL = np.arange(1, 10)
COUNT_POOL = np.arange(2, 7)

GRID_H_RANGE = (9, 26)
GRID_W_RANGE = (9, 26)
N_COLOR_RANGE = (2, 5)

BBOX_RETRIES = 250
POINT_RETRIES = 250
GENERATION_RETRIES = 600
OVERLAP_PROB = 0.5
MIN_FORCED_OVERLAP_AREA = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each color's scattered pixels into its count-ordered bounding rectangle.

    Input structure:
    - Black background with several foreground colors drawn as scattered single pixels.
    - Each foreground color appears at least twice and at most six times.
    - Foreground pixel counts are distinct across colors.
    - The same-color pixels span both rows and columns, so each color has a non-degenerate axis-aligned bounding box.

    Rule:
    - For every non-black color, compute the minimal axis-aligned bounding box of all pixels of that color.
    - Sort the colors by their original pixel count from fewest to most.
    - Starting from a black grid of the same shape, draw each color's solid bounding rectangle in that order.
    - Later, higher-count rectangles overwrite earlier rectangles wherever their bounding boxes overlap.

    Key insight:
    - The sparse same-color points are rectangle corners and samples, while the visible pixel count gives the layer order for overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        pts = np.argwhere(g == color)
        count = int(pts.shape[0])
        assert 2 <= count <= 6, "each foreground color must occur 2..6 times"
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        assert r0 < r1 and c0 < c1, "each color must have a non-degenerate bbox"
        objects.append((count, color, r0, c0, r1, c1))

    assert objects, "expected at least one foreground color"
    counts = [count for count, *_ in objects]
    assert len(set(counts)) == len(counts), "foreground color counts must be distinct"

    out = np.zeros_like(g)
    for _count, color, r0, c0, r1, c1 in sorted(objects, key=lambda item: item[0]):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


# --------------- generator ---------------


Rect = tuple[int, int, int, int]


def _overlap_area(a: Rect, b: Rect) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _too_close(a: Rect, b: Rect, pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _any_overlap(rects: list[Rect]) -> bool:
    return any(_overlap_area(a, b) > 0 for a, b in combinations(rects, 2))


def _sample_bbox(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
    compact: bool,
) -> Rect | None:
    max_h = min(height, 7 if compact else 12)
    max_w = min(width, 7 if compact else 12)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(BBOX_RETRIES):
        bh = int(rng.integers(2, max_h + 1))
        bw = int(rng.integers(2, max_w + 1))
        if bh * bw < count + 3:
            continue
        r0 = int(rng.integers(0, height - bh + 1))
        c0 = int(rng.integers(0, width - bw + 1))
        return (r0, c0, r0 + bh - 1, c0 + bw - 1)
    return None


def _make_disjoint_bboxes(
    rng: np.random.Generator,
    height: int,
    width: int,
    counts: list[int],
) -> list[Rect] | None:
    rects: list[Rect | None] = [None] * len(counts)
    placed: list[Rect] = []
    order = sorted(range(len(counts)), key=lambda i: -counts[i])

    for idx in order:
        for _ in range(BBOX_RETRIES):
            rect = _sample_bbox(rng, height, width, counts[idx], compact=True)
            if rect is not None and all(not _too_close(rect, other, pad=1) for other in placed):
                rects[idx] = rect
                placed.append(rect)
                break
        else:
            return None

    return [rect for rect in rects if rect is not None]


def _make_overlapping_bboxes(
    rng: np.random.Generator,
    height: int,
    width: int,
    counts: list[int],
) -> list[Rect] | None:
    rects: list[Rect] = []
    first = _sample_bbox(rng, height, width, counts[0], compact=False)
    if first is None:
        return None
    rects.append(first)

    second = None
    for _ in range(BBOX_RETRIES):
        candidate = _sample_bbox(rng, height, width, counts[1], compact=False)
        if candidate is not None and _overlap_area(first, candidate) >= MIN_FORCED_OVERLAP_AREA:
            second = candidate
            break
    if second is None:
        return None
    rects.append(second)

    for count in counts[2:]:
        rect = _sample_bbox(rng, height, width, count, compact=False)
        if rect is None:
            return None
        rects.append(rect)

    if not _any_overlap(rects):
        return None
    return rects


def _points_for_bbox(
    rng: np.random.Generator,
    rect: Rect,
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = rect
    diagonals = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
    ]
    rng.shuffle(diagonals)

    cells = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occupied
    ]
    if len(cells) < count:
        return None

    for anchors in diagonals:
        if any(cell in occupied for cell in anchors):
            continue
        extras = [cell for cell in cells if cell not in anchors]
        need = count - len(anchors)
        if need > len(extras):
            continue
        for _ in range(POINT_RETRIES):
            if need:
                picked = rng.choice(len(extras), size=need, replace=False)
                pts = anchors + [extras[int(i)] for i in picked]
            else:
                pts = list(anchors)
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            if min(rows) == r0 and max(rows) == r1 and min(cols) == c0 and max(cols) == c1:
                return pts

    for _ in range(POINT_RETRIES):
        picked = rng.choice(len(cells), size=count, replace=False)
        pts = [cells[int(i)] for i in picked]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        if min(rows) == r0 and max(rows) == r1 and min(cols) == c0 and max(cols) == c1:
            return pts

    return None


def _choose_grid_shape(
    rng: np.random.Generator,
    n_colors: int,
    want_overlap: bool,
) -> tuple[int, int]:
    h_min, h_max = GRID_H_RANGE
    w_min, w_max = GRID_W_RANGE
    if not want_overlap:
        h_min = max(h_min, 12 + n_colors)
        w_min = max(w_min, 12 + n_colors)
    height = int(rng.integers(h_min, h_max + 1))
    width = int(rng.integers(w_min, w_max + 1))
    return height, width


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_colors = int(rng.integers(N_COLOR_RANGE[0], N_COLOR_RANGE[1] + 1))
    want_overlap = bool(rng.random() < OVERLAP_PROB)
    height, width = _choose_grid_shape(rng, n_colors, want_overlap)

    counts = rng.choice(COUNT_POOL, size=n_colors, replace=False).astype(int).tolist()
    rng.shuffle(counts)
    colors = rng.choice(COLOR_POOL, size=n_colors, replace=False).astype(int).tolist()

    if want_overlap:
        rects = _make_overlapping_bboxes(rng, height, width, counts)
    else:
        rects = _make_disjoint_bboxes(rng, height, width, counts)
    if rects is None or len(rects) != n_colors:
        return None
    if want_overlap != _any_overlap(rects):
        return None

    grid = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for idx in rng.permutation(n_colors):
        idx = int(idx)
        pts = _points_for_bbox(rng, rects[idx], counts[idx], occupied)
        if pts is None:
            return None
        for r, c in pts:
            occupied.add((r, c))
            grid[r, c] = colors[idx]

    color_counts = Counter(map(int, grid[grid != BG].reshape(-1)))
    if sorted(color_counts.values()) != sorted(counts):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if not np.any(out == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
