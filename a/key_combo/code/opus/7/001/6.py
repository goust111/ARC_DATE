import random
from typing import NamedTuple

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

RECT_COLORS = (BLUE, GREEN, ORANGE)
GRID_MIN = 12
GRID_MAX = 15
MAX_ATTEMPTS = 500
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Rect(NamedTuple):
    r0: int
    c0: int
    r1: int
    c1: int
    color: int


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_hollow_rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for color in RECT_COLORS:
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
                raise AssertionError("rectangles must stay away from the border")
            if r1 - r0 < 2 or c1 - c0 < 2:
                raise AssertionError("rectangle must have a strict interior")
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                raise AssertionError("component border is not a rectangle outline")
            if not np.all(sub[~border] == BG):
                raise AssertionError("rectangle interior must be empty")
            if len(comp) != int(border.sum()):
                raise AssertionError("rectangle outline has extra pixels")
            rects.append(Rect(r0, c0, r1, c1, color))
    rects.sort(key=lambda rect: (rect.r0, rect.c0, rect.color))
    return rects


def _on_outline(rect: Rect, r: int, c: int) -> bool:
    return rect.r0 <= r <= rect.r1 and rect.c0 <= c <= rect.c1 and (r in (rect.r0, rect.r1) or c in (rect.c0, rect.c1))


def _strictly_inside(rect: Rect, r: int, c: int) -> bool:
    return rect.r0 < r < rect.r1 and rect.c0 < c < rect.c1


def _fill_color(rects: list[Rect], r: int, c: int) -> int | None:
    if any(_on_outline(rect, r, c) for rect in rects):
        return None
    inside = [rect.color for rect in rects if _strictly_inside(rect, r, c)]
    if len(inside) > 1:
        raise AssertionError("rectangles may not overlap on an interior point")
    return inside[0] if inside else RED


def _rows_and_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    cols = [c for c in range(grid.shape[1]) if int(grid[0, c]) == YELLOW]
    rows = [r for r in range(grid.shape[0]) if int(grid[r, -1]) == YELLOW]
    return sorted(rows), sorted(cols)


def _expected_output(grid: np.ndarray, rows: list[int], cols: list[int], rects: list[Rect]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in rows:
        for c in cols:
            color = _fill_color(rects, r, c)
            if color is not None:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the marker-row/marker-column Cartesian product, inheriting each hollow rectangle's color only at interior crossings.

    Input structure:
    - A black background contains 2 to 4 yellow markers on the top row and 2 to 4 yellow markers on the rightmost column.
    - Inside the border there are 1 to 3 non-touching hollow rectangles with outline colors chosen from blue, green, and orange.
    - Rectangle interiors stay black, and the rectangles do not touch the grid boundary.

    Rule:
    - Read the selected columns from the top-row yellow markers and the selected rows from the right-column yellow markers.
    - Form every Cartesian-product crossing between those rows and columns.
    - If a crossing lies strictly inside a hollow rectangle, recolor that cell with the rectangle's outline color; if it lies outside every rectangle, recolor it red.
    - If a crossing ever landed on a rectangle outline, the outline would stay unchanged; all other cells remain as they were.

    Key insight:
    - The changed positions come from the edge-marker Cartesian product, but the color at each changed position is determined by geometric containment inside the hollow rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = _rows_and_cols(g)
    assert 2 <= len(rows) <= 4 and 2 <= len(cols) <= 4, "expected 2 to 4 markers on each edge"
    rects = _find_hollow_rectangles(g)
    assert 1 <= len(rects) <= 3, "expected 1 to 3 hollow rectangles"
    return _expected_output(g, rows, cols, rects)


def _boxes_are_separated(a: Rect, b: Rect) -> bool:
    return a.r1 + 1 < b.r0 or b.r1 + 1 < a.r0 or a.c1 + 1 < b.c0 or b.c1 + 1 < a.c0


def _candidate_boxes_for_seed(
    h: int,
    w: int,
    rows: list[int],
    cols: list[int],
    seed_r: int,
    seed_c: int,
) -> list[tuple[int, int, int, int]]:
    row_pairs = []
    for r0 in range(1, seed_r):
        if r0 in rows:
            continue
        for r1 in range(seed_r + 1, h - 1):
            if r1 in rows:
                continue
            if 4 <= (r1 - r0 + 1) <= 8:
                row_pairs.append((r0, r1))
    col_pairs = []
    for c0 in range(1, seed_c):
        if c0 in cols:
            continue
        for c1 in range(seed_c + 1, w - 1):
            if c1 in cols:
                continue
            if 4 <= (c1 - c0 + 1) <= 8:
                col_pairs.append((c0, c1))
    boxes = [(r0, c0, r1, c1) for r0, r1 in row_pairs for c0, c1 in col_pairs]
    random.shuffle(boxes)
    return boxes


def _count_inside_points(rect: Rect, rows: list[int], cols: list[int]) -> int:
    return sum(1 for r in rows for c in cols if _strictly_inside(rect, r, c))


def _sample_rectangles(h: int, w: int, rows: list[int], cols: list[int]) -> list[Rect] | None:
    colors = list(RECT_COLORS)
    random.shuffle(colors)
    want = random.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
    seeds = [(r, c) for r in rows for c in cols]
    random.shuffle(seeds)
    rects: list[Rect] = []
    active_needed = 1 if want == 1 else random.choice([1, 2])
    for color in colors[:want]:
        placed = False
        seed_order = seeds[:]
        random.shuffle(seed_order)
        for seed_r, seed_c in seed_order:
            for r0, c0, r1, c1 in _candidate_boxes_for_seed(h, w, rows, cols, seed_r, seed_c):
                rect = Rect(r0, c0, r1, c1, color)
                if all(_boxes_are_separated(rect, other) for other in rects):
                    rects.append(rect)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    active_rects = sum(_count_inside_points(rect, rows, cols) > 0 for rect in rects)
    if active_rects < active_needed:
        return None
    return rects


def _draw_rectangles(grid: np.ndarray, rects: list[Rect]) -> None:
    for rect in rects:
        grid[rect.r0, rect.c0 : rect.c1 + 1] = rect.color
        grid[rect.r1, rect.c0 : rect.c1 + 1] = rect.color
        grid[rect.r0 : rect.r1 + 1, rect.c0] = rect.color
        grid[rect.r0 : rect.r1 + 1, rect.c1] = rect.color


def _marker_positions(h: int, w: int) -> tuple[list[int], list[int]]:
    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)
    rows = sorted(random.sample(list(range(2, h - 2)), row_count))
    cols = sorted(random.sample(list(range(2, w - 2)), col_count))
    return rows, cols


def _build_grid(h: int, w: int, rows: list[int], cols: list[int], rects: list[Rect]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, cols] = YELLOW
    grid[rows, -1] = YELLOW
    _draw_rectangles(grid, rects)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX
    rows, cols = _rows_and_cols(g)
    assert 2 <= len(rows) <= 4
    assert 2 <= len(cols) <= 4
    assert all(r != 0 for r in rows)
    assert all(c != w - 1 for c in cols)
    rects = _find_hollow_rectangles(g)
    assert 1 <= len(rects) <= 3
    assert len({rect.color for rect in rects}) == len(rects)
    for i, rect in enumerate(rects):
        assert rect.r0 >= 1 and rect.c0 >= 1 and rect.r1 <= h - 2 and rect.c1 <= w - 2
        for other in rects[i + 1 :]:
            assert _boxes_are_separated(rect, other)
    crossings = [(r, c) for r in rows for c in cols]
    outline_hits = [pos for pos in crossings if any(_on_outline(rect, pos[0], pos[1]) for rect in rects)]
    inside_hits = [pos for pos in crossings if any(_strictly_inside(rect, pos[0], pos[1]) for rect in rects)]
    outside_hits = [
        pos
        for pos in crossings
        if not any(_on_outline(rect, pos[0], pos[1]) or _strictly_inside(rect, pos[0], pos[1]) for rect in rects)
    ]
    assert not outline_hits
    assert len(inside_hits) >= 2
    assert len(outside_hits) >= 2
    out = _expected_output(g, rows, cols, rects)
    diff = {(r, c) for r in range(h) for c in range(w) if int(g[r, c]) != int(out[r, c])}
    assert diff == set(inside_hits + outside_hits)
    fill_colors = {int(out[r, c]) for r, c in crossings}
    assert RED in fill_colors and len(fill_colors) >= 2
    assert np.array_equal(main(g), out)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    rows, cols = _marker_positions(h, w)
    rects = _sample_rectangles(h, w, rows, cols)
    if rects is None:
        return None
    grid = _build_grid(h, w, rows, cols, rects)
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
