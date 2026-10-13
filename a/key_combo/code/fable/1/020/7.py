from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive
Point = tuple[int, int]

BLACK = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW

MAX_ATTEMPTS = 300
MAX_UNION_FRACTION = 0.75

_TEMPLATE_LIBRARY: list[dict[str, object]] = [
    {
        "name": "overlap_chain_3a",
        "kind": "overlap",
        "weight": 3,
        "rects": [(1, 0, 3, 4), (0, 2, 3, 3), (1, 1, 2, 2)],
    },
    {
        "name": "overlap_chain_3b",
        "kind": "overlap",
        "weight": 3,
        "rects": [(1, 0, 3, 4), (0, 2, 2, 4), (0, 1, 3, 3)],
    },
    {
        "name": "nested_overlap_4a",
        "kind": "nested",
        "weight": 2,
        "rects": [(0, 0, 3, 5), (1, 3, 2, 4), (0, 2, 4, 6), (2, 1, 4, 5)],
    },
    {
        "name": "overlap_stack_4a",
        "kind": "overlap",
        "weight": 2,
        "rects": [(2, 1, 3, 4), (3, 0, 5, 3), (1, 2, 4, 4), (0, 0, 6, 4)],
    },
    {
        "name": "overlap_stack_4b",
        "kind": "overlap",
        "weight": 2,
        "rects": [(1, 0, 5, 4), (2, 0, 3, 2), (1, 1, 4, 5), (0, 1, 3, 3)],
    },
    {
        "name": "dense_overlap_5",
        "kind": "dense",
        "weight": 1,
        "rects": [(1, 0, 4, 4), (1, 2, 3, 5), (2, 0, 3, 1), (0, 0, 5, 3), (0, 2, 4, 6)],
    },
    {
        "name": "nested_overlap_4b",
        "kind": "nested",
        "weight": 2,
        "rects": [(2, 1, 4, 6), (1, 4, 3, 5), (0, 0, 5, 6), (0, 2, 2, 3)],
    },
]


def _rect_corners(rect: Rect) -> tuple[Point, Point, Point, Point]:
    r0, c0, r1, c1 = rect
    return (r0, c0), (r0, c1), (r1, c0), (r1, c1)


def _render_input(shape: tuple[int, int], points: list[Point]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    if points:
        rr, cc = zip(*points)
        out[np.array(rr), np.array(cc)] = YELLOW
    return out


def _cover_counts(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _render_output(shape: tuple[int, int], points: list[Point], rects: list[Rect]) -> np.ndarray:
    cover = _cover_counts(shape, rects)
    out = np.full(shape, BLACK, dtype=int)
    out[cover % 2 == 1] = RED
    if points:
        rr, cc = zip(*points)
        out[np.array(rr), np.array(cc)] = YELLOW
    return out


def _candidate_rectangles(points: list[Point]) -> list[Rect]:
    row_to_cols: dict[int, set[int]] = {}
    for r, c in points:
        row_to_cols.setdefault(int(r), set()).add(int(c))

    candidates: list[Rect] = []
    for r0, r1 in combinations(sorted(row_to_cols), 2):
        common_cols = sorted(row_to_cols[r0] & row_to_cols[r1])
        for c0, c1 in combinations(common_cols, 2):
            candidates.append((int(r0), int(c0), int(r1), int(c1)))
    return candidates


def _unique_rectangle_partition(points: list[Point]) -> tuple[list[Rect], list[Rect], list[list[int]]]:
    pts = sorted({(int(r), int(c)) for r, c in points})
    assert len(pts) % 4 == 0, "yellow point count must be a multiple of four"
    assert len(pts) >= 4, "expected at least one rectangle"

    point_index = {point: idx for idx, point in enumerate(pts)}
    candidates = _candidate_rectangles(pts)
    candidate_masks: list[int] = []
    point_to_candidates: list[list[int]] = [[] for _ in pts]
    for cand_idx, rect in enumerate(candidates):
        mask = 0
        for point in _rect_corners(rect):
            mask |= 1 << point_index[point]
        candidate_masks.append(mask)
        for point in _rect_corners(rect):
            point_to_candidates[point_index[point]].append(cand_idx)

    memo: dict[int, tuple[int, list[int] | None]] = {}

    def search(remaining: int) -> tuple[int, list[int] | None]:
        if remaining == 0:
            return 1, []
        cached = memo.get(remaining)
        if cached is not None:
            return cached

        active_points = [idx for idx in range(len(pts)) if (remaining >> idx) & 1]
        best_options: list[int] | None = None
        for point_idx in active_points:
            options = [
                cand_idx
                for cand_idx in point_to_candidates[point_idx]
                if candidate_masks[cand_idx] & remaining == candidate_masks[cand_idx]
            ]
            if not options:
                memo[remaining] = (0, None)
                return memo[remaining]
            if best_options is None or len(options) < len(best_options):
                best_options = options

        assert best_options is not None
        solution_count = 0
        witness: list[int] | None = None
        for cand_idx in best_options:
            sub_count, sub_solution = search(remaining ^ candidate_masks[cand_idx])
            if sub_count == 0:
                continue
            if witness is None:
                witness = [cand_idx] + ([] if sub_solution is None else sub_solution)
            solution_count += sub_count
            if solution_count > 1:
                memo[remaining] = (solution_count, witness)
                return memo[remaining]

        memo[remaining] = (solution_count, witness)
        return memo[remaining]

    total_count, witness = search((1 << len(pts)) - 1)
    assert total_count == 1, f"expected a unique rectangle partition, found {total_count}"
    assert witness is not None
    solution = sorted((candidates[idx] for idx in witness), key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))
    assert len(solution) == len(pts) // 4
    return solution, candidates, point_to_candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group all yellow dots into the unique set of axis-aligned corner rectangles and parity-fill their combined coverage.

    Input structure:
    - The background is black.
    - The only foreground pixels are yellow singleton corner markers, with total count 4n.
    - Those yellow dots admit exactly one global partition into axis-aligned rectangles, although extra row/column alignments may suggest false rectangles.

    Rule:
    - Recover the unique partition of yellow dots into rectangle corner quartets.
    - For each recovered rectangle, treat its whole closed area, including the boundary rows and columns, as covered once.
    - Paint cells covered an odd number of times red and leave cells covered an even number of times black.
    - Restore every original yellow corner marker on top of that parity fill.

    Key insight:
    - The dots are not independent local clues: they must be grouped globally into rectangle corners, and overlapping rectangles cancel by parity in their shared region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BLACK, "expected a black background"
    assert colors_present(g, exclude={bg}) == [YELLOW], "expected only yellow corner dots on black"

    points = [tuple(map(int, coord)) for coord in np.argwhere(g == YELLOW)]
    rects, _candidates, _point_to_candidates = _unique_rectangle_partition(points)
    return _render_output(g.shape, points, rects)


def _rectangles_overlap(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return min(ar1, br1) >= max(ar0, br0) and min(ac1, bc1) >= max(ac0, bc0)


def _rectangles_nested(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar0 < br0 < br1 < ar1 and ac0 < bc0 < bc1 < ac1
    ) or (
        br0 < ar0 < ar1 < br1 and bc0 < ac0 < ac1 < bc1
    )


def _sample_template() -> list[Rect]:
    weights = [int(template["weight"]) for template in _TEMPLATE_LIBRARY]
    chosen = random.choices(_TEMPLATE_LIBRARY, weights=weights, k=1)[0]
    return [tuple(rect) for rect in chosen["rects"]]  # type: ignore[arg-type]


def _transform_template(rects: list[Rect]) -> list[Rect]:
    work = list(rects)
    if random.random() < 0.5:
        work = [(c0, r0, c1, r1) for r0, c0, r1, c1 in work]

    row_count = 1 + max(max(r0, r1) for r0, _c0, r1, _c1 in work)
    col_count = 1 + max(max(c0, c1) for _r0, c0, _r1, c1 in work)

    if random.random() < 0.5:
        work = [(row_count - 1 - r1, c0, row_count - 1 - r0, c1) for r0, c0, r1, c1 in work]
    if random.random() < 0.5:
        work = [(r0, col_count - 1 - c1, r1, col_count - 1 - c0) for r0, c0, r1, c1 in work]

    return work


def _sample_axis_positions(count: int) -> tuple[list[int], int]:
    while True:
        start = random.randint(1, 2)
        positions = [start]
        for _ in range(1, int(count)):
            positions.append(positions[-1] + random.randint(2, 4))
        end_margin = random.randint(1, 2)
        size = positions[-1] + end_margin + 1
        if size <= 30:
            return positions, size


def _instantiate_template(rects: list[Rect]) -> tuple[tuple[int, int], list[Point], list[Rect]]:
    transformed = _transform_template(rects)
    row_count = 1 + max(max(r0, r1) for r0, _c0, r1, _c1 in transformed)
    col_count = 1 + max(max(c0, c1) for _r0, c0, _r1, c1 in transformed)

    row_positions, height = _sample_axis_positions(row_count)
    col_positions, width = _sample_axis_positions(col_count)

    actual_rects = [
        (row_positions[r0], col_positions[c0], row_positions[r1], col_positions[c1])
        for r0, c0, r1, c1 in transformed
    ]
    points = sorted({point for rect in actual_rects for point in _rect_corners(rect)})
    return (height, width), points, sorted(actual_rects)


def _semantic_checks(shape: tuple[int, int], points: list[Point], seed_rects: list[Rect]) -> bool:
    if len(points) != 4 * len(seed_rects):
        return False
    if max(shape) > 30:
        return False

    try:
        solved_rects, candidates, point_to_candidates = _unique_rectangle_partition(points)
    except AssertionError:
        return False

    if solved_rects != sorted(seed_rects):
        return False
    if len(candidates) <= len(solved_rects):
        return False

    ambiguous_points = sum(len(options) > 1 for options in point_to_candidates)
    if ambiguous_points < 4:
        return False

    if not any(_rectangles_overlap(a, b) for a, b in combinations(solved_rects, 2)):
        return False

    cover = _cover_counts(shape, solved_rects)
    yellow_mask = np.zeros(shape, dtype=bool)
    if points:
        rr, cc = zip(*points)
        yellow_mask[np.array(rr), np.array(cc)] = True

    hidden_even = (cover > 0) & (cover % 2 == 0) & (~yellow_mask)
    visible_red = (cover % 2 == 1) & (~yellow_mask)
    if int(hidden_even.sum()) < 4:
        return False
    if int(visible_red.sum()) < 8:
        return False
    if int((cover > 0).sum()) > int(np.prod(shape) * MAX_UNION_FRACTION):
        return False
    if int(cover.max()) < 2:
        return False

    inp = _render_input(shape, points)
    expected = _render_output(shape, points, solved_rects)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rects = _sample_template()
    shape, points, actual_rects = _instantiate_template(rects)
    if not _semantic_checks(shape, points, actual_rects):
        return None
    return _render_input(shape, points)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-parity input")
