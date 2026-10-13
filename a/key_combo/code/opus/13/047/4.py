from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RANK_COLORS = (RED, BLUE, GREEN)
MAX_GENERATION_ATTEMPTS = 400


def _find_rectangles(yellow_cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rows = sorted({r for r, _ in yellow_cells})
    cols = sorted({c for _, c in yellow_cells})
    rectangles = set()
    for r0, r1 in combinations(rows, 2):
        if r1 - r0 < 2:
            continue
        for c0, c1 in combinations(cols, 2):
            if c1 - c0 < 2:
                continue
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if corners.issubset(yellow_cells):
                rectangles.add((r0, c0, r1, c1))
    return sorted(rectangles)


def _interior_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 - 1) * (c1 - c0 - 1)


def _inside_interior(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 < r < r1 and c0 < c < c1


def _interior_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    }


def _ranked_rectangles(rectangles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ranked = sorted(rectangles, key=_interior_area, reverse=True)
    areas = [_interior_area(rect) for rect in ranked]
    assert len(set(areas)) == len(areas), "rectangle interior areas must be distinct"
    return ranked


def _color_by_rectangle(rectangles: list[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], int]:
    ranked = _ranked_rectangles(rectangles)
    return {rect: RANK_COLORS[idx] for idx, rect in enumerate(ranked)}


def _transform_from_parts(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    grey_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    rect_to_color = _color_by_rectangle(rectangles)

    for rect in _ranked_rectangles(rectangles):
        color = rect_to_color[rect]
        for r, c in grey_cells:
            if _inside_interior(rect, r, c):
                out[r, c] = color

    for r0, c0, r1, c1 in rectangles:
        out[r0, c0] = YELLOW
        out[r0, c1] = YELLOW
        out[r1, c0] = YELLOW
        out[r1, c1] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color grey mask cells inside yellow-corner rectangles, with area rank controlling both color assignment and reverse overlap priority.

    Input structure:
    - The grid is a black background containing 2 or 3 rectangles implied only by yellow corner markers.
    - Each rectangle is defined by four yellow pixels sharing two rows and two columns, and its open interior may contain grey mask pixels.
    - Rectangle interiors can overlap, and the same grey pixel may lie inside multiple rectangles.

    Rule:
    - Detect every rectangle whose four yellow corners are present.
    - Compute each rectangle's pure interior area `(height-2) x (width-2)` and rank rectangles from largest to smallest.
    - Assign colors by that ranking: largest -> red, next -> blue, smallest -> green.
    - For each rectangle, recolor only the grey pixels that lie in its open interior.
    - If a grey pixel belongs to multiple rectangles, keep the color of the smallest such rectangle; preserve all yellow corner markers.

    Key insight:
    - The same area ordering is used twice in opposite directions: larger rectangles get earlier colors, but smaller rectangles win when overlapping grey mask cells compete.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)}
    rectangles = _find_rectangles(yellow_cells)
    assert 2 <= len(rectangles) <= 3, "expected 2 or 3 yellow-corner rectangles"
    _ranked_rectangles(rectangles)

    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    return _transform_from_parts(g.shape, rectangles, grey_cells)


def _membership_zones(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    zones: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            subset = tuple(
                idx for idx, rect in enumerate(rectangles) if _inside_interior(rect, r, c)
            )
            if subset:
                zones.setdefault(subset, []).append((r, c))
    return zones


def _winner_for_subset(subset: tuple[int, ...], areas: list[int]) -> int:
    return min(subset, key=lambda idx: areas[idx])


def _subset_of_cell(
    cell: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> tuple[int, ...]:
    r, c = cell
    return tuple(idx for idx, rect in enumerate(rectangles) if _inside_interior(rect, r, c))


def _shuffled_permutations(rng: np.random.Generator, values: list[int]) -> list[tuple[int, ...]]:
    perms = list(permutations(values))
    rng.shuffle(perms)
    return perms


def _sample_rectangles(
    rng: np.random.Generator, shape: tuple[int, int], rect_count: int
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    max_core_h = 2 if rect_count == 3 else 3
    max_core_w = 2 if rect_count == 3 else 3

    for _ in range(80):
        core_h = int(rng.integers(1, max_core_h + 1))
        core_w = int(rng.integers(1, max_core_w + 1))
        max_top = h - rect_count - core_h
        max_left = w - rect_count - core_w
        if max_top < rect_count or max_left < rect_count:
            return None

        core_top = int(rng.integers(rect_count, max_top + 1))
        core_left = int(rng.integers(rect_count, max_left + 1))
        core_bottom = core_top + core_h - 1
        core_right = core_left + core_w - 1

        top_values = list(map(int, rng.choice(np.arange(1, core_top + 1), size=rect_count, replace=False)))
        bottom_values = list(
            map(int, rng.choice(np.arange(core_bottom, h - 1), size=rect_count, replace=False))
        )
        left_values = list(
            map(int, rng.choice(np.arange(1, core_left + 1), size=rect_count, replace=False))
        )
        right_values = list(
            map(int, rng.choice(np.arange(core_right, w - 1), size=rect_count, replace=False))
        )

        for tops in _shuffled_permutations(rng, top_values):
            for bottoms in _shuffled_permutations(rng, bottom_values):
                for lefts in _shuffled_permutations(rng, left_values):
                    for rights in _shuffled_permutations(rng, right_values):
                        rectangles = [
                            (tops[i] - 1, lefts[i] - 1, bottoms[i] + 1, rights[i] + 1)
                            for i in range(rect_count)
                        ]
                        if _rectangles_are_valid(shape, rectangles):
                            return rectangles
    return None


def _rectangles_are_valid(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> bool:
    h, w = shape
    rect_count = len(rectangles)
    areas = [_interior_area(rect) for rect in rectangles]
    if min(areas) < 4 or len(set(areas)) != rect_count:
        return False

    row_marks = []
    col_marks = []
    yellow_cells = set()
    for r0, c0, r1, c1 in rectangles:
        if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
            return False
        if r1 - r0 < 2 or c1 - c0 < 2:
            return False
        row_marks.extend([r0, r1])
        col_marks.extend([c0, c1])
        yellow_cells.update({(r0, c0), (r0, c1), (r1, c0), (r1, c1)})

    if len(set(row_marks)) != 2 * rect_count or len(set(col_marks)) != 2 * rect_count:
        return False
    if set(_find_rectangles(yellow_cells)) != set(rectangles):
        return False

    zones = _membership_zones(shape, rectangles)
    if not any(len(subset) >= 2 for subset in zones):
        return False

    for idx in range(rect_count):
        if not any(idx in subset for subset in zones):
            return False

    areas = [_interior_area(rect) for rect in rectangles]
    for idx in range(rect_count):
        if not any(_winner_for_subset(subset, areas) == idx for subset in zones):
            return False

    return True


def _choose_cell(
    rng: np.random.Generator,
    cells: list[tuple[int, int]],
    grey_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    unused = [cell for cell in cells if cell not in grey_cells]
    if unused:
        return unused[int(rng.integers(0, len(unused)))]
    return None


def _sample_grey_cells(
    rng: np.random.Generator,
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
) -> set[tuple[int, int]] | None:
    rect_count = len(rectangles)
    areas = [_interior_area(rect) for rect in rectangles]
    zones = _membership_zones(shape, rectangles)
    grey_cells: set[tuple[int, int]] = set()

    def counts() -> list[int]:
        out = [0] * rect_count
        for cell in grey_cells:
            for idx in _subset_of_cell(cell, rectangles):
                out[idx] += 1
        return out

    overlap_subsets = [
        subset
        for subset, cells in zones.items()
        if len(subset) >= 2 and cells and max(areas[i] for i in subset) > min(areas[i] for i in subset)
    ]
    if not overlap_subsets:
        return None

    overlap_subsets.sort(key=lambda subset: (-len(subset), -len(zones[subset]), subset))
    conflict_cell = _choose_cell(rng, zones[overlap_subsets[0]], grey_cells)
    if conflict_cell is None:
        return None
    grey_cells.add(conflict_cell)

    for idx in sorted(range(rect_count), key=lambda i: areas[i], reverse=True):
        winner_subsets = [
            subset for subset, cells in zones.items() if cells and _winner_for_subset(subset, areas) == idx
        ]
        winner_subsets.sort(key=lambda subset: (len(subset), -len(zones[subset]), subset))
        chosen = None
        for subset in winner_subsets:
            chosen = _choose_cell(rng, zones[subset], grey_cells)
            if chosen is not None:
                break
        if chosen is None:
            if _winner_for_subset(_subset_of_cell(conflict_cell, rectangles), areas) == idx:
                chosen = conflict_cell
            else:
                return None
        grey_cells.add(chosen)

    safety = 0
    while min(counts()) < 3 and safety < 100:
        current = counts()
        need = {idx for idx, count in enumerate(current) if count < 3}
        candidates = []
        for subset, cells in zones.items():
            if not any(idx in need for idx in subset):
                continue
            if any(current[idx] >= 8 for idx in subset):
                continue
            cell = _choose_cell(rng, cells, grey_cells)
            if cell is None:
                continue
            needed_hits = sum(idx in need for idx in subset)
            extra_hits = len(subset) - needed_hits
            score = (needed_hits, -extra_hits, -len(subset), len(cells))
            candidates.append((score, cell))
        if not candidates:
            return None
        best_score = max(score for score, _ in candidates)
        best_cells = [cell for score, cell in candidates if score == best_score]
        grey_cells.add(best_cells[int(rng.integers(0, len(best_cells)))])
        safety += 1

    current = counts()
    if min(current) < 3 or max(current) > 8:
        return None

    extra_steps = int(rng.integers(0, 4))
    for _ in range(extra_steps):
        current = counts()
        candidates = []
        for subset, cells in zones.items():
            if any(current[idx] >= 8 for idx in subset):
                continue
            cell = _choose_cell(rng, cells, grey_cells)
            if cell is None:
                continue
            score = (-len(subset), len(cells))
            candidates.append((score, cell))
        if not candidates:
            break
        top_score = max(score for score, _ in candidates)
        top_cells = [cell for score, cell in candidates if score == top_score]
        grey_cells.add(top_cells[int(rng.integers(0, len(top_cells)))])

    final_counts = counts()
    if not all(3 <= count <= 8 for count in final_counts):
        return None
    return grey_cells


def _render_input(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    grey_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in grey_cells:
        grid[r, c] = GREY
    for r0, c0, r1, c1 in rectangles:
        grid[r0, c0] = YELLOW
        grid[r0, c1] = YELLOW
        grid[r1, c0] = YELLOW
        grid[r1, c1] = YELLOW
    return grid


def _validate_semantics(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    grey_cells: set[tuple[int, int]],
) -> bool:
    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    if set(_find_rectangles(yellow_cells)) != set(rectangles):
        return False

    counts = []
    for rect in rectangles:
        count = sum(_inside_interior(rect, r, c) for r, c in grey_cells)
        counts.append(count)
    if not all(3 <= count <= 8 for count in counts):
        return False

    areas = [_interior_area(rect) for rect in rectangles]
    if len(set(areas)) != len(areas):
        return False

    output = _transform_from_parts(grid.shape, rectangles, grey_cells)
    if not np.array_equal(main(grid), output):
        return False

    rect_to_color = _color_by_rectangle(rectangles)
    for rect, color in rect_to_color.items():
        if not np.any(output == color):
            return False

    conflict_cells = [
        cell for cell in grey_cells if len(_subset_of_cell(cell, rectangles)) >= 2
    ]
    if not conflict_cells:
        return False
    for cell in conflict_cells:
        subset = _subset_of_cell(cell, rectangles)
        winning_idx = _winner_for_subset(subset, areas)
        winning_rect = rectangles[winning_idx]
        if output[cell] != rect_to_color[winning_rect]:
            return False

    if np.count_nonzero(output == YELLOW) != 4 * len(rectangles):
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = 3 if rng.random() < 0.7 else 2
    row_low = 14 if rect_count == 3 else 12
    col_low = 14 if rect_count == 3 else 12
    shape = (
        int(rng.integers(row_low, 19)),
        int(rng.integers(col_low, 19)),
    )

    rectangles = _sample_rectangles(rng, shape, rect_count)
    if rectangles is None:
        return None

    grey_cells = _sample_grey_cells(rng, shape, rectangles)
    if grey_cells is None:
        return None

    grid = _render_input(shape, rectangles, grey_cells)
    if _validate_semantics(grid, rectangles, grey_cells):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/047/4")
