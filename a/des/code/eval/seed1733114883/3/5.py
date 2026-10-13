from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
MARK_COLOR = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

GRID_SIZE_RANGE = (15, 25)
GROUP_COUNT_RANGE = (2, 4)
SOURCE_AREA_RANGE = (3, 10)
SOURCE_HEIGHT_RANGE = (2, 5)
SOURCE_WIDTH_RANGE = (2, 5)
SCREEN_EXTRA_RANGE = (0, 2)
SEPARATION_RANGE = (2, 4)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400

DIRECTIONS = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    rr = cr + dr
                    cc2 = cc + dc
                    if 0 <= rr < h and 0 <= cc2 < w and bool(mask[rr, cc2]) and not bool(seen[rr, cc2]):
                        seen[rr, cc2] = True
                        stack.append((rr, cc2))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(random.randint(0, 3)):
        out = {(c, -r) for r, c in out}
        out = _normalize(out)
    if random.random() < 0.5:
        max_c = max(c for _r, c in out)
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.5:
        max_r = max(r for r, _c in out)
        out = {(max_r - r, c) for r, c in out}
    return _normalize(out)


def _is_line_segment(coords: np.ndarray) -> bool:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if rows.size == 1:
        ordered = np.sort(coords[:, 1])
        return int(coords.shape[0]) == int(ordered[-1] - ordered[0] + 1)
    if cols.size == 1:
        ordered = np.sort(coords[:, 0])
        return int(coords.shape[0]) == int(ordered[-1] - ordered[0] + 1)
    return False


def _source_component(coords: np.ndarray) -> bool:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    return rows.size > 1 and cols.size > 1


def _apply_projection(out: np.ndarray, source: np.ndarray, screen: np.ndarray) -> None:
    r0, c0, r1, c1 = _bbox(screen)
    if r0 == r1:
        screen_cols = {int(c) for c in screen[:, 1]}
        for col in np.unique(source[:, 1]):
            col_int = int(col)
            assert c0 <= col_int <= c1
            assert col_int in screen_cols
            out[r0, col_int] = MARK_COLOR
    else:
        screen_rows = {int(r) for r in screen[:, 0]}
        for row in np.unique(source[:, 0]):
            row_int = int(row)
            assert r0 <= row_int <= r1
            assert row_int in screen_rows
            out[row_int, c0] = MARK_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each colored source shape and its matching same-color screen segment, then draw the orthogonal projection onto the screen in yellow.

    Input structure:
    - Black background with 2-4 independent color groups.
    - Each group has one connected 2D source shape and one same-color straight screen segment.
    - The screen is horizontal or vertical, detached from the source, and lies directly above, below, left, or right of it.

    Rule:
    - For every non-background color, identify the straight component as the screen and the 2D component as the source.
    - A horizontal screen receives the set of occupied source columns.
    - A vertical screen receives the set of occupied source rows.
    - Mark projected screen cells yellow while leaving every other pixel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG_COLOR]
    assert 2 <= len(colors) <= 4

    for color in colors:
        comps = _components(g == int(color))
        lines = [comp for comp in comps if _is_line_segment(comp)]
        sources = [comp for comp in comps if _source_component(comp)]
        assert len(comps) == 2
        assert len(lines) == 1
        assert len(sources) == 1

        screen = lines[0]
        source = sources[0]
        sr0, sc0, sr1, sc1 = _bbox(screen)
        or0, oc0, or1, oc1 = _bbox(source)

        if sr0 == sr1:
            assert or1 < sr0 or or0 > sr0
            assert sc0 <= oc0 and oc1 <= sc1
        else:
            assert oc1 < sc0 or oc0 > sc0
            assert sr0 <= or0 and or1 <= sr1

        _apply_projection(out, source, screen)

    return out


def _random_source_shape() -> set[tuple[int, int]]:
    for _ in range(200):
        area = random.randint(*SOURCE_AREA_RANGE)
        choices = [
            (height, width)
            for height in range(SOURCE_HEIGHT_RANGE[0], SOURCE_HEIGHT_RANGE[1] + 1)
            for width in range(SOURCE_WIDTH_RANGE[0], SOURCE_WIDTH_RANGE[1] + 1)
            if height * width >= area
        ]
        height, width = random.choice(choices)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return _transform_shape(_normalize(cells))
    raise RuntimeError("failed to build a non-degenerate source shape")


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _grid_bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _conflicts(occupied: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(occupied[rr, cc]):
                    return True
    return False


def _make_screen(
    direction: str,
    source_cells: set[tuple[int, int]],
    origin_r: int,
    origin_c: int,
    extra_a: int,
    extra_b: int,
    separation: int,
) -> tuple[set[tuple[int, int]], str]:
    src_r0, src_c0, src_r1, src_c1 = _grid_bounds(source_cells)
    abs_r0 = origin_r + src_r0
    abs_c0 = origin_c + src_c0
    abs_r1 = origin_r + src_r1
    abs_c1 = origin_c + src_c1

    if direction == "up":
        row = abs_r0 - separation
        start = abs_c0 - extra_a
        end = abs_c1 + extra_b
        return ({(row, c) for c in range(start, end + 1)}, "horizontal")
    if direction == "down":
        row = abs_r1 + separation
        start = abs_c0 - extra_a
        end = abs_c1 + extra_b
        return ({(row, c) for c in range(start, end + 1)}, "horizontal")
    if direction == "left":
        col = abs_c0 - separation
        start = abs_r0 - extra_a
        end = abs_r1 + extra_b
        return ({(r, col) for r in range(start, end + 1)}, "vertical")
    col = abs_c1 + separation
    start = abs_r0 - extra_a
    end = abs_r1 + extra_b
    return ({(r, col) for r in range(start, end + 1)}, "vertical")


def _place_group(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    direction: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    source_shape = _random_source_shape()
    src_r0, src_c0, src_r1, src_c1 = _grid_bounds(source_shape)
    src_h = src_r1 - src_r0 + 1
    src_w = src_c1 - src_c0 + 1

    for _ in range(PLACEMENT_RETRIES):
        extra_a = random.randint(*SCREEN_EXTRA_RANGE)
        extra_b = random.randint(*SCREEN_EXTRA_RANGE)
        separation = random.randint(*SEPARATION_RANGE)

        if direction in ("up", "down"):
            row_min = separation if direction == "up" else 0
            row_max = h - src_h - (separation if direction == "down" else 0)
            col_min = extra_a
            col_max = w - src_w - extra_b
        else:
            row_min = extra_a
            row_max = h - src_h - extra_b
            col_min = separation if direction == "left" else 0
            col_max = w - src_w - (separation if direction == "right" else 0)

        if row_min > row_max or col_min > col_max:
            continue

        origin_r = random.randint(row_min, row_max)
        origin_c = random.randint(col_min, col_max)
        source_cells = _shift_cells(source_shape, origin_r, origin_c)
        screen_cells, _orientation = _make_screen(
            direction,
            source_shape,
            origin_r,
            origin_c,
            extra_a,
            extra_b,
            separation,
        )

        group_cells = source_cells | screen_cells
        if _conflicts(occupied, group_cells):
            continue

        for r, c in group_cells:
            grid[r, c] = int(color)
            occupied[r, c] = True
        return source_cells, screen_cells

    return None


def _expected_output(
    grid: np.ndarray,
    groups: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for source_cells, screen_cells in groups:
        rows = {r for r, _c in screen_cells}
        cols = {c for _r, c in screen_cells}
        if len(rows) == 1:
            row = next(iter(rows))
            lit_cols = {c for _r, c in source_cells}
            for col in lit_cols:
                assert (row, col) in screen_cells
                out[row, col] = MARK_COLOR
        else:
            col = next(iter(cols))
            lit_rows = {r for r, _c in source_cells}
            for row in lit_rows:
                assert (row, col) in screen_cells
                out[row, col] = MARK_COLOR
    return out


def _attempt_input() -> np.ndarray | None:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    min_size = max(GRID_SIZE_RANGE[0], 11 + 2 * group_count)
    height = random.randint(min_size, GRID_SIZE_RANGE[1])
    width = random.randint(min_size, GRID_SIZE_RANGE[1])

    grid = np.full((height, width), BG_COLOR, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    colors = random.sample(PALETTE, k=group_count)
    directions = random.sample(DIRECTIONS, k=group_count)
    groups: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []

    order = list(range(group_count))
    random.shuffle(order)
    for idx in order:
        placed = _place_group(grid, occupied, int(colors[idx]), directions[idx])
        if placed is None:
            return None
        groups.append(placed)

    expected = _expected_output(grid, groups)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection puzzle input")
