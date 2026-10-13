from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

RANK_COLORS = (BLUE, RED, GREEN)
GRID_MIN = 13
GRID_MAX = 20
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 8
MAX_GENERATION_ATTEMPTS = 800

_AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
    for _w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
        _AREA_TO_DIMS.setdefault(_h * _w, []).append((_h, _w))


def _corners(rect: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = rect
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    rr = min(a[2], b[2]) - max(a[0], b[0]) + 1
    cc = min(a[3], b[3]) - max(a[1], b[1]) + 1
    if rr <= 0 or cc <= 0:
        return 0
    return rr * cc


def _candidate_rectangles(points: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    candidates = []
    for r0, r1 in itertools.combinations(rows, 2):
        for c0, c1 in itertools.combinations(cols, 2):
            rect = (r0, c0, r1, c1)
            if set(_corners(rect)) <= points:
                candidates.append(rect)
    return sorted(candidates)


def _extract_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    points = {tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)}
    assert len(points) == 12, "expected exactly three four-corner marker sets"
    assert np.count_nonzero(grid) == len(points), "input may only contain yellow corner markers"

    candidates = _candidate_rectangles(points)
    assert candidates, "corner markers must define at least one rectangle"

    cand_points = {rect: frozenset(_corners(rect)) for rect in candidates}
    containing: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {pt: [] for pt in points}
    for rect in candidates:
        for pt in cand_points[rect]:
            containing[pt].append(rect)

    solutions: set[tuple[tuple[int, int, int, int], ...]] = set()

    def search(remaining: frozenset[tuple[int, int]], chosen: tuple[tuple[int, int, int, int], ...]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.add(tuple(sorted(chosen)))
            return

        pivot = min(
            remaining,
            key=lambda pt: sum(1 for rect in containing[pt] if cand_points[rect] <= remaining),
        )
        options = [rect for rect in containing[pivot] if cand_points[rect] <= remaining]
        for rect in options:
            search(remaining - cand_points[rect], chosen + (rect,))

    search(frozenset(points), ())
    assert len(solutions) == 1, "yellow markers must admit one unique rectangle grouping"

    rects = list(next(iter(solutions)))
    areas = [_area(rect) for rect in rects]
    assert len(rects) == 3, "expected three rectangles"
    assert len(set(areas)) == 3, "rectangle areas must be distinct"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer three corner-marked rectangles, rank them by area, then color and occlude by that same ranking.

    Input structure:
    - The grid is a black background containing only yellow corner markers.
    - Exactly three groups of four yellow pixels exist; each group shares two rows and two columns.
    - Each four-corner group defines one axis-aligned rectangle whose interior is otherwise black in the input.

    Rule:
    - Recover the unique partition of the twelve yellow pixels into three four-corner rectangles.
    - Compute each rectangle's full area including its border cells, then rank the rectangles by area.
    - Map smallest, middle, and largest areas to blue, red, and green respectively, and fill rectangles in descending area order so smaller rectangles overwrite larger ones where they overlap.

    Key insight:
    - The same area ordering drives both semantics at once: it assigns each rectangle's fill color and also determines the front-to-back painting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    rects = _extract_rectangles(g)
    by_area = sorted(rects, key=_area)
    color_of = {rect: color for rect, color in zip(by_area, RANK_COLORS)}

    out = np.full(g.shape, BLACK, dtype=int)
    for rect in sorted(rects, key=_area, reverse=True):
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = color_of[rect]
    return out


def _sample_dims() -> list[tuple[int, int]]:
    areas = random.sample(list(_AREA_TO_DIMS), 3)
    dims = [random.choice(_AREA_TO_DIMS[area]) for area in areas]
    random.shuffle(dims)
    return dims


def _make_rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _corners_are_disjoint(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    mine = set(_corners(rect))
    return all(mine.isdisjoint(_corners(other)) for other in others)


def _moderate_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    overlap = _overlap_area(a, b)
    if overlap == 0:
        return False
    return overlap <= max(6, min(_area(a), _area(b)) // 2)


def _placement_candidates(
    height: int,
    width: int,
    h: int,
    w: int,
    existing: list[tuple[int, int, int, int]],
    *,
    overlap_with: tuple[int, int, int, int] | None = None,
    disjoint_from: tuple[tuple[int, int, int, int], ...] = (),
) -> list[tuple[int, int, int, int]]:
    candidates = []
    for r0 in range(height - h + 1):
        for c0 in range(width - w + 1):
            rect = _make_rect(r0, c0, h, w)
            if not _corners_are_disjoint(rect, existing):
                continue
            if overlap_with is not None and not _moderate_overlap(rect, overlap_with):
                continue
            if any(_overlap_area(rect, other) > 0 for other in disjoint_from):
                continue
            candidates.append(rect)
    random.shuffle(candidates)
    return candidates


def _render_input(height: int, width: int, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for rect in rects:
        rr, cc = zip(*_corners(rect))
        grid[rr, cc] = YELLOW
    return grid


def _pair_count(rects: list[tuple[int, int, int, int]]) -> int:
    count = 0
    for a, b in itertools.combinations(rects, 2):
        if _overlap_area(a, b) > 0:
            count += 1
    return count


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    dims = _sample_dims()
    target_pair_count = 1 if random.random() < 0.6 else 2

    first_h, first_w = dims[0]
    first = _make_rect(
        random.randint(0, height - first_h),
        random.randint(0, width - first_w),
        first_h,
        first_w,
    )

    second_h, second_w = dims[1]
    second_choices = _placement_candidates(height, width, second_h, second_w, [first], overlap_with=first)
    if not second_choices:
        return None
    second = random.choice(second_choices)

    third_h, third_w = dims[2]
    if target_pair_count == 1:
        third_choices = _placement_candidates(
            height,
            width,
            third_h,
            third_w,
            [first, second],
            disjoint_from=(first, second),
        )
    else:
        anchor, block = random.choice(((first, second), (second, first)))
        third_choices = _placement_candidates(
            height,
            width,
            third_h,
            third_w,
            [first, second],
            overlap_with=anchor,
            disjoint_from=(block,),
        )
    if not third_choices:
        return None
    third = random.choice(third_choices)

    rects = [first, second, third]
    if _pair_count(rects) != target_pair_count:
        return None

    grid = _render_input(height, width, rects)
    try:
        recovered = _extract_rectangles(grid)
        expected = np.full((height, width), BLACK, dtype=int)
        color_of = {rect: color for rect, color in zip(sorted(rects, key=_area), RANK_COLORS)}
        for rect in sorted(rects, key=_area, reverse=True):
            r0, c0, r1, c1 = rect
            expected[r0 : r1 + 1, c0 : c1 + 1] = color_of[rect]
        if sorted(recovered) != sorted(rects):
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-ranked rectangle input")
