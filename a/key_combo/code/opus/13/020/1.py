from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SOURCE_COLORS = (BLUE, RED, YELLOW, ORANGE)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _guide_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid == GREEN, axis=1))
    assert len(rows) == 1, "expected exactly one horizontal guide row"
    row = int(rows[0])
    values = set(map(int, np.unique(grid[row])))
    assert values <= {BLACK, GREEN}, "guide row must contain only green cells and black gaps"
    assert GREEN in values and BLACK in values, "guide row must have both green segments and gaps"
    return row


def _rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for coords in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "grey rectangles must have non-empty interiors"
        assert int(coords.shape[0]) == height * width, "grey objects must be filled rectangles"
        rects.append((r0, c0, r1, c1))
    rects.sort(key=lambda rect: (rect[1], rect[0]))
    assert 2 <= len(rects) <= 4, "expected 2-4 routing rectangles"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each routed grey rectangle interior using the color of its valid projected marker pixel.

    Input structure:
    - The grid has black background, 2-4 solid grey axis-aligned rectangles, and one horizontal green guide row spanning the full width with black gaps.
    - Additional isolated source pixels use colors from {blue, red, yellow, orange} and sit above or below the guide.
    - Rectangles do not overlap, and the guide row stays separate from the rectangles.

    Rule:
    - For every colored source pixel, project vertically to the guide cell in the same column.
    - If that guide cell is green, route the source color to the unique grey rectangle whose column span contains that column; if the guide cell is a black gap, ignore the source pixel.
    - For each rectangle with exactly one valid routed source pixel, recolor only its interior to that source color while keeping the one-cell grey border unchanged; all other cells remain as in the input.

    Key insight:
    - The green guide is both a validity filter and a column-based router: gaps erase markers, and surviving columns decide which rectangle receives the marker's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    guide_row = _guide_row(g)
    rects = _rectangles(g)
    out = g.copy()
    routed_colors: list[list[int]] = [[] for _ in rects]

    source_mask = ~np.isin(g, (BLACK, GREEN, GREY))
    for r_raw, c_raw in np.argwhere(source_mask):
        r = int(r_raw)
        c = int(c_raw)
        assert r != guide_row, "source pixels cannot be on the guide row"
        if int(g[guide_row, c]) != GREEN:
            continue
        matches = [idx for idx, (_r0, c0, _r1, c1) in enumerate(rects) if c0 <= c <= c1]
        assert len(matches) <= 1, "projection column cannot belong to multiple rectangles"
        if matches:
            routed_colors[matches[0]].append(int(g[r, c]))

    for (r0, c0, r1, c1), colors in zip(rects, routed_colors):
        if len(colors) == 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = colors[0]

    return out


def _spans_separated(existing: list[tuple[int, int]], c0: int, c1: int) -> bool:
    return all(c1 + 1 < ec0 or ec1 + 1 < c0 for ec0, ec1 in existing)


def _place_rectangles(
    height: int,
    width: int,
    guide_row: int,
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        sides = [random.choice((-1, 1)) for _ in range(count)]
        if count >= 2 and len(set(sides)) == 1 and random.random() < 0.8:
            sides[random.randrange(count)] *= -1

        intervals: list[tuple[int, int]] = []
        placed: list[tuple[int, int, int, int]] = []
        order = list(range(count))
        random.shuffle(order)
        ok = True

        for idx in order:
            rect_h = random.randint(3, 5)
            rect_w = random.randint(3, 6)
            candidate_cols = list(range(width - rect_w + 1))
            random.shuffle(candidate_cols)
            placed_here = False
            for c0 in candidate_cols:
                c1 = c0 + rect_w - 1
                if not _spans_separated(intervals, c0, c1):
                    continue
                if sides[idx] < 0:
                    max_r0 = guide_row - rect_h - 1
                    if max_r0 < 0:
                        continue
                    r0 = random.randint(0, max_r0)
                else:
                    min_r0 = guide_row + 2
                    if min_r0 > height - rect_h:
                        continue
                    r0 = random.randint(min_r0, height - rect_h)
                placed.append((r0, c0, r0 + rect_h - 1, c1))
                intervals.append((c0, c1))
                placed_here = True
                break
            if not placed_here:
                ok = False
                break

        if ok:
            placed.sort(key=lambda rect: (rect[1], rect[0]))
            return placed
    return None


def _choose_gap_columns(width: int, rects: list[tuple[int, int, int, int]]) -> list[int] | None:
    for _ in range(200):
        gap_count = random.randint(1, 3)
        gap_cols: set[int] = set()
        made = 0
        for _ in range(gap_count):
            gap_w = random.randint(1, min(3, width - 2))
            starts = list(range(1, width - gap_w))
            random.shuffle(starts)
            for start in starts:
                cells = set(range(start, start + gap_w))
                if any((cell in gap_cols) or (cell - 1 in gap_cols) or (cell + 1 in gap_cols) for cell in cells):
                    continue
                gap_cols.update(cells)
                made += 1
                break
        if made != gap_count:
            continue
        if all(any(c not in gap_cols for c in range(c0, c1 + 1)) for _r0, c0, _r1, c1 in rects):
            return sorted(gap_cols)
    return None


def _can_place_source(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    r0 = max(0, r - 1)
    r1 = min(grid.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(grid.shape[1], c + 2)
    return not np.any(grid[r0:r1, c0:c1] != BLACK)


def _place_source(grid: np.ndarray, guide_row: int, side: int, col: int, color: int) -> bool:
    if side < 0:
        rows = list(range(0, guide_row - 1))
    else:
        rows = list(range(guide_row + 2, grid.shape[0]))
    random.shuffle(rows)
    for r in rows:
        if _can_place_source(grid, r, col):
            grid[r, col] = color
            return True
    return False


def _expected_output(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    fill_by_rect: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for idx, color in fill_by_rect.items():
        r0, c0, r1, c1 = rects[idx]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(color)
    return out


def _build_input() -> np.ndarray | None:
    height = random.randint(16, 22)
    width = random.randint(18, 26)
    guide_row = random.randint(6, height - 7)
    rect_count = random.randint(2, 4)

    rects = _place_rectangles(height, width, guide_row, rect_count)
    if rects is None:
        return None

    gap_cols = _choose_gap_columns(width, rects)
    if gap_cols is None:
        return None
    gap_set = set(gap_cols)

    grid = np.zeros((height, width), dtype=int)
    grid[guide_row, :] = GREEN
    if gap_cols:
        grid[guide_row, gap_cols] = BLACK
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    eligible = [idx for idx, (_r0, c0, _r1, c1) in enumerate(rects) if any(c not in gap_set for c in range(c0, c1 + 1))]
    if not eligible:
        return None

    if len(eligible) == 1:
        active_count = 1
    elif random.random() < 0.7:
        active_count = random.randint(1, len(eligible) - 1)
    else:
        active_count = random.randint(1, len(eligible))
    active_rects = random.sample(eligible, active_count)

    fill_by_rect: dict[int, int] = {}
    sides_used: list[int] = []
    for rect_idx in active_rects:
        _r0, c0, _r1, c1 = rects[rect_idx]
        valid_cols = [c for c in range(c0, c1 + 1) if c not in gap_set]
        color = random.choice(SOURCE_COLORS)
        side = random.choice((-1, 1))
        if not _place_source(grid, guide_row, side, random.choice(valid_cols), color):
            return None
        fill_by_rect[rect_idx] = color
        sides_used.append(side)

    gap_cols_inside_rects = [
        c
        for _r0, c0, _r1, c1 in rects
        for c in range(c0, c1 + 1)
        if c in gap_set
    ]
    invalid_pool = gap_cols_inside_rects if gap_cols_inside_rects else gap_cols
    missing_sides = [side for side in (-1, 1) if side not in sides_used]
    invalid_count = random.randint(max(1, len(missing_sides)), max(2, len(missing_sides)) + 2)

    for i in range(invalid_count):
        side = missing_sides[i] if i < len(missing_sides) else random.choice((-1, 1))
        if not _place_source(grid, guide_row, side, random.choice(invalid_pool), random.choice(SOURCE_COLORS)):
            return None

    above_sources = int(np.count_nonzero(np.isin(grid[:guide_row, :], SOURCE_COLORS)))
    below_sources = int(np.count_nonzero(np.isin(grid[guide_row + 1 :, :], SOURCE_COLORS)))
    if above_sources == 0 or below_sources == 0:
        return None

    expected = _expected_output(grid, rects, fill_by_rect)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
