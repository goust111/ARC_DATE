from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _largest_solid_rect(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    height, width = g.shape
    candidates: list[tuple[int, int, int, int, int, int]] = []

    for color in colors_present(g, exclude={bg}):
        mask = g == color
        for top in range(height):
            full_cols = np.ones(width, dtype=bool)
            for bottom in range(top, height):
                full_cols &= mask[bottom]
                col = 0
                while col < width:
                    while col < width and not full_cols[col]:
                        col += 1
                    if col == width:
                        break

                    left = col
                    while col < width and full_cols[col]:
                        col += 1
                    right = col - 1

                    rect_h = bottom - top + 1
                    rect_w = right - left + 1
                    if rect_h >= 2 and rect_w >= 2:
                        area = rect_h * rect_w
                        candidates.append((area, color, top, left, bottom, right))

    assert candidates, "expected at least one solid non-background rectangle"
    max_area = max(area for area, *_rest in candidates)
    winners = [candidate for candidate in candidates if candidate[0] == max_area]
    assert len(winners) == 1, "largest solid rectangle must be unique"

    _area, color, top, left, bottom, right = winners[0]
    return color, top, left, bottom, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the unique largest solid monochrome rectangle and erase the surrounding noise.

    Input structure
    - The grid has a black background with scattered non-background noise pixels.
    - One non-background color contains a unique large solid axis-aligned rectangle,
      possibly with extra same-color protrusions outside the rectangle.
    Rule
    - Find the largest-area solid rectangle whose cells all share one non-background
      color and whose height and width are both at least two.
    - Return a blank background grid with only that rectangle redrawn in its
      original position and color.
    Key insight
    - The retained color is identified by the large filled rectangular support,
      not by its color id or by total color frequency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK

    color, top, left, bottom, right = _largest_solid_rect(g, bg)
    assert np.all(g[top : bottom + 1, left : right + 1] == color)

    out = np.full_like(g, bg)
    out[top : bottom + 1, left : right + 1] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_GRID_SIZE = 16
_GEN_BACKGROUND = 0
_GEN_COLORS = list(range(1, 10))

# Target rectangle geometry.
_GEN_RECT_HEIGHTS = (2, 3, 4)
_GEN_RECT_WIDTHS = (3, 4, 5, 6, 7)
_GEN_RECT_AREAS = range(9, 15)
_GEN_RECT_EDGE_MARGIN = 2
_GEN_RECT_SIZES = [(h, w) for h in _GEN_RECT_HEIGHTS for w in _GEN_RECT_WIDTHS if h * w in _GEN_RECT_AREAS]

# Pixel-count budgets.
_GEN_TARGET_EXTRA_RANGE = (7, 13)
_GEN_TOTAL_NONZERO_RANGE = (72, 102)
_GEN_DISTRACTOR_COLOR_COUNT = 8
_GEN_DISTRACTOR_PIXEL_RANGE = (5, 14)

# Placement behavior.
_GEN_DOMINO_PROBABILITY = 0.35
_GEN_MAX_ATTEMPTS = 300
_GEN_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_bounded_counts(n: int, total: int, lo: int, hi: int) -> list[int] | None:
    if total < n * lo or total > n * hi:
        return None

    counts: list[int] = []
    remaining = total
    for i in range(n):
        left = n - i - 1
        v_min = max(lo, remaining - left * hi)
        v_max = min(hi, remaining - left * lo)
        value = random.randint(v_min, v_max)
        counts.append(value)
        remaining -= value

    random.shuffle(counts)
    return counts


def _gen_near_rect(r: int, c: int, top: int, left: int, bottom: int, right: int) -> bool:
    if top <= r <= bottom and left <= c <= right:
        return True
    if top <= r <= bottom and (c == left - 1 or c == right + 1):
        return True
    if left <= c <= right and (r == top - 1 or r == bottom + 1):
        return True
    return False


def _gen_has_selected_neighbor(selected: set[tuple[int, int]], r: int, c: int) -> bool:
    return any((r + dr, c + dc) in selected for dr, dc in _GEN_DIRS4)


def _gen_place_target_extras(
    grid: np.ndarray,
    color: int,
    count: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> bool:
    cells = [
        (r, c)
        for r in range(_GEN_GRID_SIZE)
        for c in range(_GEN_GRID_SIZE)
        if grid[r, c] == _GEN_BACKGROUND and not _gen_near_rect(r, c, top, left, bottom, right)
    ]
    random.shuffle(cells)

    selected: set[tuple[int, int]] = set()
    for r, c in cells:
        if not _gen_has_selected_neighbor(selected, r, c):
            selected.add((r, c))
            if len(selected) == count:
                break

    if len(selected) != count:
        return False

    for r, c in selected:
        grid[r, c] = color
    return True


def _gen_would_make_2x2(grid: np.ndarray, color: int, cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                if r0 < 0 or c0 < 0 or r0 + 1 >= _GEN_GRID_SIZE or c0 + 1 >= _GEN_GRID_SIZE:
                    continue
                ok = True
                for rr in (r0, r0 + 1):
                    for cc in (c0, c0 + 1):
                        if (rr, cc) not in cell_set and grid[rr, cc] != color:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
    return False


def _gen_single_candidates(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(_GEN_GRID_SIZE)
        for c in range(_GEN_GRID_SIZE)
        if grid[r, c] == _GEN_BACKGROUND and not _gen_would_make_2x2(grid, color, ((r, c),))
    ]


def _gen_domino_candidates(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(_GEN_GRID_SIZE):
        for c in range(_GEN_GRID_SIZE):
            if grid[r, c] != _GEN_BACKGROUND:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if rr < _GEN_GRID_SIZE and cc < _GEN_GRID_SIZE and grid[rr, cc] == _GEN_BACKGROUND:
                    cells = ((r, c), (rr, cc))
                    if not _gen_would_make_2x2(grid, color, cells):
                        candidates.append(cells)
    return candidates


def _gen_place_distractor_color(grid: np.ndarray, color: int, count: int) -> bool:
    remaining = count
    while remaining:
        use_domino = remaining >= 2 and random.random() < _GEN_DOMINO_PROBABILITY
        if use_domino:
            dominoes = _gen_domino_candidates(grid, color)
            if dominoes:
                for r, c in random.choice(dominoes):
                    grid[r, c] = color
                remaining -= 2
                continue

        singles = _gen_single_candidates(grid, color)
        if not singles:
            return False
        r, c = random.choice(singles)
        grid[r, c] = color
        remaining -= 1

    return True


def _gen_target_rect_plan() -> tuple[int, int, int, int, int, int]:
    target_color = random.randint(_GEN_COLORS[0], _GEN_COLORS[-1])
    h, w = random.choice(_GEN_RECT_SIZES)
    top = random.randint(_GEN_RECT_EDGE_MARGIN, _GEN_GRID_SIZE - h - _GEN_RECT_EDGE_MARGIN)
    left = random.randint(_GEN_RECT_EDGE_MARGIN, _GEN_GRID_SIZE - w - _GEN_RECT_EDGE_MARGIN)
    bottom = top + h - 1
    right = left + w - 1
    area = h * w
    return target_color, top, left, bottom, right, area


def _gen_count_plan(area: int) -> tuple[int, list[int]] | None:
    extra_count = random.randint(*_GEN_TARGET_EXTRA_RANGE)
    target_total = random.randint(*_GEN_TOTAL_NONZERO_RANGE)
    distractor_total = target_total - area - extra_count
    distractor_counts = _gen_bounded_counts(
        _GEN_DISTRACTOR_COLOR_COUNT, distractor_total, *_GEN_DISTRACTOR_PIXEL_RANGE
    )
    if distractor_counts is None:
        return None
    return extra_count, distractor_counts


def _gen_attempt() -> np.ndarray | None:
    target_color, top, left, bottom, right, area = _gen_target_rect_plan()
    count_plan = _gen_count_plan(area)
    if count_plan is None:
        return None

    extra_count, distractor_counts = count_plan
    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    out[top : bottom + 1, left : right + 1] = target_color
    grid = out.copy()

    if not _gen_place_target_extras(grid, target_color, extra_count, top, left, bottom, right):
        return None

    distractor_colors = [color for color in _GEN_COLORS if color != target_color]
    random.shuffle(distractor_colors)
    for color, count in zip(distractor_colors, distractor_counts):
        if not _gen_place_distractor_color(grid, color, count):
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
