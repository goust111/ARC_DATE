from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rect_perimeter(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int], *, strict: bool) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    if strict:
        return r0 < r < r1 and c0 < c < c1
    return r0 <= r <= r1 and c0 <= c <= c1


def _infer_rectangles_from_points(points: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rows_to_cols: dict[int, set[int]] = defaultdict(set)
    for r, c in points:
        rows_to_cols[int(r)].add(int(c))

    rects = set()
    rows = sorted(rows_to_cols)
    for i, r0 in enumerate(rows):
        cols0 = rows_to_cols[r0]
        for r1 in rows[i + 1 :]:
            common = sorted(cols0 & rows_to_cols[r1])
            for j, c0 in enumerate(common):
                for c1 in common[j + 1 :]:
                    rects.add((r0, c0, r1, c1))
    return sorted(rects)


def _infer_virtual_walls(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]]:
    rectangles: list[tuple[int, int, int, int]] = []
    walls: set[tuple[int, int]] = set()
    for color in np.unique(grid):
        color = int(color)
        if color in (Color.BLACK, Color.BLUE):
            continue
        points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(points) < 4:
            continue
        for rect in _infer_rectangles_from_points(points):
            rectangles.append(rect)
            walls.update(_rect_perimeter(rect))
    return rectangles, walls


def _flood_with_virtual_walls(
    grid: np.ndarray, seed: tuple[int, int], walls: set[tuple[int, int]]
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = grid.copy()
    seen = {seed}
    q: deque[tuple[int, int]] = deque([seed])
    filled = set()

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if (rr, cc) in seen or not in_bounds(grid.shape, rr, cc):
                continue
            seen.add((rr, cc))
            if grid[rr, cc] != Color.BLACK or (rr, cc) in walls:
                continue
            out[rr, cc] = Color.BLUE
            filled.add((rr, cc))
            q.append((rr, cc))
    return out, filled


def _naive_black_flood(grid: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {seed}
    q: deque[tuple[int, int]] = deque([seed])
    filled = set()
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if (rr, cc) in seen or not in_bounds(grid.shape, rr, cc):
                continue
            seen.add((rr, cc))
            if grid[rr, cc] != Color.BLACK:
                continue
            filled.add((rr, cc))
            q.append((rr, cc))
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill from the blue seed through black cells, but treat inferred rectangle perimeters as invisible walls.

    Input structure:
    - The background is black.
    - Non-blue foreground pixels are colored corner markers; any four same-color pixels at the corners of one axis-aligned rectangle define an invisible rectangular wall.
    - Some same-color groups are incomplete or malformed and do not define any wall, and exactly one blue seed pixel is present.

    Rule:
    - Infer every axis-aligned rectangle whose four corners are marked by pixels of the same non-blue color.
    - Treat the full perimeter of each inferred rectangle, including the black cells between the corner markers, as blocked for 4-neighbor movement.
    - Starting from the unique blue seed, recolor every reachable black cell blue; keep corner markers, invisible-wall cells, and unreachable black cells unchanged.

    Key insight:
    - The blocking structure is not drawn explicitly; it must be reconstructed from four-corner color agreements, while incomplete same-color corner groups are only decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seeds = np.argwhere(g == Color.BLUE)
    assert seeds.shape == (1, 2), "expected exactly one blue seed"
    seed = tuple(map(int, seeds[0]))

    _, walls = _infer_virtual_walls(g)
    out, _ = _flood_with_virtual_walls(g, seed, walls)
    return out


def _bbox_conflicts(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_outer_rect(h: int, w: int, *, min_size: int, max_size: int) -> tuple[int, int, int, int]:
    for _ in range(200):
        rect_h = random.randint(min_size, min(max_size, h - 2))
        rect_w = random.randint(min_size, min(max_size, w - 2))
        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(1, w - rect_w - 1)
        return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    raise RuntimeError("failed to sample outer rectangle")


def _sample_disjoint_rect(
    h: int,
    w: int,
    existing: list[tuple[int, int, int, int]],
    *,
    min_size: int,
    max_size: int,
    gap: int,
) -> tuple[int, int, int, int] | None:
    for _ in range(300):
        rect_h = random.randint(min_size, min(max_size, h - 2))
        rect_w = random.randint(min_size, min(max_size, w - 2))
        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(1, w - rect_w - 1)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if all(not _bbox_conflicts(rect, other, gap) for other in existing):
            return rect
    return None


def _sample_inner_rect(outer: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = outer
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 9 or outer_w < 9:
        return None
    for _ in range(300):
        top = random.randint(2, outer_h - 6)
        bottom = random.randint(2, outer_h - top - 4)
        left = random.randint(2, outer_w - 6)
        right = random.randint(2, outer_w - left - 4)
        inner = (r0 + top, c0 + left, r1 - bottom, c1 - right)
        if inner[2] - inner[0] + 1 >= 4 and inner[3] - inner[1] + 1 >= 4:
            return inner
    return None


def _candidate_cells(
    h: int,
    w: int,
    walls: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    predicate,
) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in walls or cell in occupied:
                continue
            if predicate(cell):
                cells.append(cell)
    return cells


def _make_invalid_pattern() -> list[tuple[int, int]]:
    style = random.choice(["three_corners", "diag3", "tee4"])
    if style == "three_corners":
        dh = random.randint(2, 4)
        dw = random.randint(2, 4)
        corners = [(0, 0), (0, dw - 1), (dh - 1, 0), (dh - 1, dw - 1)]
        missing = random.randrange(4)
        pattern = [corners[i] for i in range(4) if i != missing]
    elif style == "diag3":
        length = random.randint(3, 4)
        pattern = [(i, i) for i in range(length)]
    else:
        pattern = [(0, 1), (1, 0), (1, 1), (1, 2)]
    assert not _infer_rectangles_from_points(pattern)
    return pattern


def _place_invalid_group(
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    pattern: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    max_r = max(r for r, _ in pattern)
    max_c = max(c for _, c in pattern)
    for _ in range(300):
        r0 = random.randint(0, h - max_r - 1)
        c0 = random.randint(0, w - max_c - 1)
        cells = [(r0 + dr, c0 + dc) for dr, dc in pattern]
        if all(cell not in forbidden for cell in cells):
            return cells
    return None


def _render_grid(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int, int]],
    invalid_groups: list[tuple[int, list[tuple[int, int]]]],
    seed: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), Color.BLACK, dtype=int)
    for color, points in invalid_groups:
        for r, c in points:
            grid[r, c] = color
    for r0, c0, r1, c1, color in rectangles:
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[r, c] = color
    grid[seed] = Color.BLUE
    return grid


def _build_oracle_output(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int, int]],
    seed: tuple[int, int],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    walls = set()
    for r0, c0, r1, c1, _color in rectangles:
        walls.update(_rect_perimeter((r0, c0, r1, c1)))
    return _flood_with_virtual_walls(grid, seed, walls)


def _generate_layout():
    scenario = random.choice(["outside", "inside", "nested_gap"])
    if scenario == "nested_gap":
        h = random.randint(16, 28)
        w = random.randint(16, 28)
        outer = _sample_outer_rect(h, w, min_size=9, max_size=14)
        inner = _sample_inner_rect(outer)
        if inner is None:
            return None
        rect_boxes = [outer, inner]
    else:
        h = random.randint(12, 26)
        w = random.randint(12, 26)
        primary = _sample_outer_rect(h, w, min_size=5, max_size=11)
        rect_boxes = [primary]

    extra_target = random.randint(0, 2)
    if scenario == "outside":
        extra_target = max(1, extra_target)
    for _ in range(extra_target):
        rect = _sample_disjoint_rect(h, w, rect_boxes, min_size=4, max_size=8, gap=2)
        if rect is not None:
            rect_boxes.append(rect)

    palette = list(range(2, 10))
    random.shuffle(palette)
    if len(rect_boxes) + 1 > len(palette):
        return None
    rectangles = [
        (rect[0], rect[1], rect[2], rect[3], palette[i]) for i, rect in enumerate(rect_boxes)
    ]

    walls = set()
    occupied = set()
    for r0, c0, r1, c1, _color in rectangles:
        perimeter = _rect_perimeter((r0, c0, r1, c1))
        walls.update(perimeter)
        occupied.update({(r0, c0), (r0, c1), (r1, c0), (r1, c1)})

    invalid_groups: list[tuple[int, list[tuple[int, int]]]] = []
    invalid_count = random.randint(1, min(3, len(palette) - len(rectangles)))
    forbidden = walls | occupied
    for i in range(invalid_count):
        pattern = _make_invalid_pattern()
        cells = _place_invalid_group(h, w, forbidden, pattern)
        if cells is None:
            return None
        color = palette[len(rectangles) + i]
        invalid_groups.append((color, cells))
        forbidden.update(cells)
        occupied.update(cells)

    if scenario == "outside":
        candidates = _candidate_cells(
            h,
            w,
            walls,
            occupied,
            lambda cell: all(not _point_in_rect(cell, rect[:4], strict=False) for rect in rectangles),
        )
    elif scenario == "inside":
        target = rectangles[0]
        candidates = _candidate_cells(
            h,
            w,
            walls,
            occupied,
            lambda cell: _point_in_rect(cell, target[:4], strict=True),
        )
    else:
        outer = rectangles[0]
        inner = rectangles[1]
        candidates = _candidate_cells(
            h,
            w,
            walls,
            occupied,
            lambda cell: _point_in_rect(cell, outer[:4], strict=True)
            and not _point_in_rect(cell, inner[:4], strict=False),
        )
    if len(candidates) < 4:
        return None
    seed = random.choice(candidates)

    grid = _render_grid((h, w), rectangles, invalid_groups, seed)
    expected, filled = _build_oracle_output(grid, rectangles, seed)
    naive = _naive_black_flood(grid, seed)

    if not filled or len(filled) < 5:
        return None
    if len(naive - filled) < 4:
        return None
    if np.count_nonzero(expected == Color.BLACK) < 8:
        return None
    if not any(len(points) == 3 for _color, points in invalid_groups):
        return None
    if sorted(_infer_virtual_walls(grid)[0]) != sorted(rect[:4] for rect in rectangles):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
