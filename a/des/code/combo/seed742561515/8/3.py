import numpy as np
from collections import Counter


BG = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()

GEN_RETRIES = 400
RECT_RETRIES = 300
PAIR_RETRIES = 300

GRID_MIN = 12
GRID_MAX = 22
RECT_COUNT_VALUES = np.array([2, 3, 4, 5], dtype=int)
RECT_COUNT_WEIGHTS = np.array([3.0, 4.0, 2.0, 1.0], dtype=float)
OVERLAP_PROB = 0.55

MIN_SIDE = 3
MAX_SIDE = 9
MAX_OVERLAP_FRACTION = 0.45
MAX_OUTPUT_DENSITY = 0.36
MIN_VISIBLE_PER_COLOR = 5


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _border_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _draw_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _contains_rect(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return bool(or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1)


def _contains_cell(rect: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    r, c = cell
    return bool(r0 <= r <= r1 and c0 <= c <= c1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ih = min(ar1, br1) - max(ar0, br0) + 1
    iw = min(ac1, bc1) - max(ac0, bc0) + 1
    if ih <= 0 or iw <= 0:
        return 0
    return int(ih * iw)


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _diag_pairs(rect: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    return [((r0, c0), (r1, c1)), ((r0, c1), (r1, c0))]


def _feasible_diagonals(
    rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    options = []
    for pair in _diag_pairs(rect):
        if all(not _contains_cell(other, cell) for cell in pair for other in others):
            options.append(pair)
    return options


def _render_rectangles(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    order = sorted(
        range(len(rects)),
        key=lambda idx: (-_rect_area(rects[idx]), rects[idx][0], rects[idx][1], rects[idx][2], rects[idx][3]),
    )
    for idx in order:
        _draw_border(out, rects[idx], int(colors[idx]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = Counter(map(int, g.reshape(-1)))
    colors = [color for color, count in counts.items() if color != BG and count > 0]
    if not colors:
        return np.full_like(g, BG)
    assert all(counts[color] == 2 for color in colors), "each non-background color must appear exactly twice"

    rects: list[tuple[int, int, int, int]] = []
    rect_colors: list[int] = []
    for color in colors:
        pts = np.argwhere(g == color)
        assert pts.shape == (2, 2), "each color must mark exactly two cells"
        r0 = int(min(pts[0, 0], pts[1, 0]))
        c0 = int(min(pts[0, 1], pts[1, 1]))
        r1 = int(max(pts[0, 0], pts[1, 0]))
        c1 = int(max(pts[0, 1], pts[1, 1]))
        rects.append((r0, c0, r1, c1))
        rect_colors.append(int(color))

    return _render_rectangles(g.shape, rects, rect_colors)


def _random_rect(height: int, width: int) -> tuple[int, int, int, int]:
    max_h = min(MAX_SIDE, height)
    max_w = min(MAX_SIDE, width)
    rect_h = int(RNG.integers(MIN_SIDE, max_h + 1))
    rect_w = int(RNG.integers(MIN_SIDE, max_w + 1))
    top = int(RNG.integers(0, height - rect_h + 1))
    left = int(RNG.integers(0, width - rect_w + 1))
    return top, left, top + rect_h - 1, left + rect_w - 1


def _sample_separate_rect(
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
    used_areas: set[int],
) -> tuple[int, int, int, int] | None:
    for _ in range(RECT_RETRIES):
        rect = _random_rect(height, width)
        if _rect_area(rect) in used_areas:
            continue
        if any(_expanded_overlap(rect, other, pad=1) for other in existing):
            continue
        return rect
    return None


def _pattern_candidates(
    rect: tuple[int, int, int, int],
    rect_h: int,
    rect_w: int,
    height: int,
    width: int,
    pattern: str,
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    candidates: list[tuple[int, int, int, int]] = []

    if pattern == "dr":
        for top in range(r0 + 1, r1):
            bottom = top + rect_h - 1
            if bottom <= r1 or bottom >= height:
                continue
            for left in range(c0 + 1, c1):
                right = left + rect_w - 1
                if right <= c1 or right >= width:
                    continue
                candidates.append((top, left, bottom, right))
    elif pattern == "dl":
        for top in range(r0 + 1, r1):
            bottom = top + rect_h - 1
            if bottom <= r1 or bottom >= height:
                continue
            for right in range(c0 + 1, c1):
                left = right - rect_w + 1
                if left >= c0 or left < 0:
                    continue
                candidates.append((top, left, bottom, right))
    elif pattern == "ur":
        for bottom in range(r0 + 1, r1):
            top = bottom - rect_h + 1
            if top >= r0 or top < 0:
                continue
            for left in range(c0 + 1, c1):
                right = left + rect_w - 1
                if right <= c1 or right >= width:
                    continue
                candidates.append((top, left, bottom, right))
    else:
        for bottom in range(r0 + 1, r1):
            top = bottom - rect_h + 1
            if top >= r0 or top < 0:
                continue
            for right in range(c0 + 1, c1):
                left = right - rect_w + 1
                if left >= c0 or left < 0:
                    continue
                candidates.append((top, left, bottom, right))

    return candidates


def _sample_overlap_pair(
    height: int, width: int, used_areas: set[int]
) -> list[tuple[int, int, int, int]] | None:
    patterns = ("dr", "dl", "ur", "ul")
    for _ in range(PAIR_RETRIES):
        rect1 = _random_rect(height, width)
        area1 = _rect_area(rect1)
        if area1 in used_areas:
            continue

        rect_h = int(RNG.integers(MIN_SIDE, min(MAX_SIDE, height) + 1))
        rect_w = int(RNG.integers(MIN_SIDE, min(MAX_SIDE, width) + 1))
        pattern = str(RNG.choice(patterns))
        candidates = _pattern_candidates(rect1, rect_h, rect_w, height, width, pattern)
        if not candidates:
            continue

        rect2 = candidates[int(RNG.integers(0, len(candidates)))]
        area2 = _rect_area(rect2)
        if area2 in used_areas or area2 == area1:
            continue
        if _contains_rect(rect1, rect2) or _contains_rect(rect2, rect1):
            continue

        overlap = _intersection_area(rect1, rect2)
        if overlap == 0:
            continue
        if overlap / min(area1, area2) > MAX_OVERLAP_FRACTION:
            continue
        if not (_border_cells(rect1) & _border_cells(rect2)):
            continue

        pair = [rect1, rect2]
        if not _feasible_diagonals(rect1, [rect2]):
            continue
        if not _feasible_diagonals(rect2, [rect1]):
            continue
        return pair
    return None


def _validate_layout(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    marker_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    colors: list[int],
) -> bool:
    marker_cells = [cell for pair in marker_pairs for cell in pair]
    if len(set(marker_cells)) != len(marker_cells):
        return False

    border_hits: Counter[tuple[int, int]] = Counter()
    for rect in rects:
        for cell in _border_cells(rect):
            border_hits[cell] += 1
    if border_hits and max(border_hits.values()) > 2:
        return False

    out = _render_rectangles(shape, rects, colors)
    density = int(np.count_nonzero(out != BG)) / float(out.size)
    if density > MAX_OUTPUT_DENSITY:
        return False
    for color in colors:
        if int(np.count_nonzero(out == color)) < MIN_VISIBLE_PER_COLOR:
            return False
    return True


def generate_input() -> np.ndarray:
    weights = RECT_COUNT_WEIGHTS / RECT_COUNT_WEIGHTS.sum()
    for _ in range(GEN_RETRIES):
        height = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
        width = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
        n_rects = int(RNG.choice(RECT_COUNT_VALUES, p=weights))
        want_overlap = bool(RNG.random() < OVERLAP_PROB)

        rects: list[tuple[int, int, int, int]] = []
        used_areas: set[int] = set()

        if want_overlap:
            pair = _sample_overlap_pair(height, width, used_areas)
            if pair is None:
                continue
            rects.extend(pair)
            used_areas.update(_rect_area(rect) for rect in pair)

        while len(rects) < n_rects:
            rect = _sample_separate_rect(height, width, rects, used_areas)
            if rect is None:
                break
            rects.append(rect)
            used_areas.add(_rect_area(rect))
        if len(rects) != n_rects:
            continue

        marker_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for idx, rect in enumerate(rects):
            others = [other for j, other in enumerate(rects) if j != idx]
            options = _feasible_diagonals(rect, others)
            if not options:
                marker_pairs = []
                break
            marker_pairs.append(options[int(RNG.integers(0, len(options)))])
        if len(marker_pairs) != n_rects:
            continue

        colors = [int(color) for color in RNG.choice(COLOR_POOL, size=n_rects, replace=False)]
        if not _validate_layout((height, width), rects, marker_pairs, colors):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for color, pair in zip(colors, marker_pairs):
            for r, c in pair:
                grid[r, c] = color

        produced = main(grid)
        if not isinstance(produced, np.ndarray) or produced.shape != grid.shape:
            continue
        return grid

    raise RuntimeError("failed to generate a valid rectangle-marker puzzle input")
