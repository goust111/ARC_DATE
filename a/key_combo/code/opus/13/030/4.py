from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    ORANGE = 7


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _path_cells(rect: dict[str, int]) -> set[tuple[int, int]]:
    r0 = rect["r0"]
    r1 = rect["r1"]
    red_c = rect["red_c"]
    green_c = rect["green_c"]
    row = r1 - 1
    step = 1 if green_c > red_c else -1
    cells = {(r, red_c) for r in range(r0 + 1, r1)}
    for c in range(red_c, green_c + step, step):
        cells.add((row, c))
    return cells


def _split_regions_for_rect(rect: dict[str, int]) -> list[list[tuple[int, int]]]:
    r0 = rect["r0"]
    c0 = rect["c0"]
    r1 = rect["r1"]
    c1 = rect["c1"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    open_mask = np.ones((inner_h, inner_w), dtype=bool)
    for pr, pc in _path_cells(rect):
        open_mask[pr - (r0 + 1), pc - (c0 + 1)] = False

    parts = _components(open_mask)
    regions: list[list[tuple[int, int]]] = []
    for coords in parts:
        regions.append([(r + r0 + 1, c + c0 + 1) for r, c in coords])
    return regions


def _validate_rectangle(g: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 6 and w >= 6, "rectangle interior must be at least 4x4"

    box = g[r0 : r1 + 1, c0 : c1 + 1]
    expected_border_area = 2 * (h + w) - 4
    assert int(np.count_nonzero(box)) == expected_border_area, "rectangle must be hollow"
    assert len(coords) == expected_border_area, "all non-black cells in component must belong to one border"

    assert int(g[r0, c0]) == Color.BLUE and int(g[r0, c1]) == Color.BLUE
    assert int(g[r1, c0]) == Color.BLUE and int(g[r1, c1]) == Color.BLUE

    red_cols = [c for c in range(c0 + 1, c1) if int(g[r0, c]) == Color.RED]
    green_cols = [c for c in range(c0 + 1, c1) if int(g[r1, c]) == Color.GREEN]
    assert len(red_cols) == 1, "top border must contain exactly one red marker"
    assert len(green_cols) == 1, "bottom border must contain exactly one green marker"
    red_c = red_cols[0]
    green_c = green_cols[0]
    assert red_c != green_c, "markers must be in different columns"

    for c in range(c0 + 1, c1):
        if c != red_c:
            assert int(g[r0, c]) == Color.BLUE
        if c != green_c:
            assert int(g[r1, c]) == Color.BLUE
    for r in range(r0 + 1, r1):
        assert int(g[r, c0]) == Color.BLUE
        assert int(g[r, c1]) == Color.BLUE

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert bool(np.all(interior == Color.BLACK)), "rectangle interior must start empty"

    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "red_c": red_c, "green_c": green_c}


def _find_rectangles(g: np.ndarray) -> list[dict[str, int]]:
    fg = g != Color.BLACK
    rects = []
    for coords in _components(fg):
        rects.append(_validate_rectangle(g, coords))
    assert rects, "expected at least one marked rectangular frame"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the two regions created by each marked rectangle's internal L path according to their relative areas.

    Input structure:
    - A black background contains one or more one-pixel-thick blue hollow rectangles.
    - Each rectangle has exactly one red marker on its top border and one green marker on its bottom border, both away from the corners.
    - Rectangle interiors are black and the red/green marker columns differ.

    Rule:
    - For each rectangle, draw a yellow L path inside it: start below the red marker, go straight down to the inner bottom row, then go horizontally on that row to the cell above the green marker.
    - Treat the blue border, the red/green markers, and the yellow path as blockers; they split the rectangle interior into exactly two 4-connected black regions.
    - Fill the larger region grey and the smaller region orange, while keeping the border, markers, and yellow path unchanged.

    Key insight:
    - The red and green markers do not just mark endpoints: they define an internal barrier whose exact L shape determines the sizes of the two enclosed regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == Color.BLACK, "expected black background"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED, Color.GREEN}

    rects = _find_rectangles(g)
    out = g.copy()

    for rect in rects:
        path = _path_cells(rect)
        for r, c in path:
            assert int(out[r, c]) == Color.BLACK, "path must be drawn on empty interior cells"
            out[r, c] = Color.YELLOW

        regions = _split_regions_for_rect(rect)
        assert len(regions) == 2, "L path must split each rectangle into exactly two regions"
        areas = [len(region) for region in regions]
        assert areas[0] != areas[1], "the two regions must have different areas"

        large_idx = 0 if areas[0] > areas[1] else 1
        small_idx = 1 - large_idx
        for r, c in regions[large_idx]:
            out[r, c] = Color.GREY
        for r, c in regions[small_idx]:
            out[r, c] = Color.ORANGE

    return out


GEN_RETRIES = 500
GRID_MIN = 16
GRID_MAX = 24
RECT_COUNT_RANGE = (2, 4)
OUTER_H_RANGE = (6, 9)
OUTER_W_RANGE = (6, 10)
MIN_GAP = 1
MIN_SMALL_REGION = 3
MIN_AREA_DIFF = 2


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + MIN_GAP < pr0 or pr1 + MIN_GAP < r0 or c1 + MIN_GAP < pc0 or pc1 + MIN_GAP < c0):
            return False
    return True


def _regions_from_inner(inner_h: int, inner_w: int, red_inner_c: int, green_inner_c: int):
    open_mask = np.ones((inner_h, inner_w), dtype=bool)
    open_mask[:, red_inner_c] = False
    step = 1 if green_inner_c > red_inner_c else -1
    for c in range(red_inner_c, green_inner_c + step, step):
        open_mask[inner_h - 1, c] = False
    return _components(open_mask)


def _valid_marker_pairs(h: int, w: int, desired_sign: int | None) -> list[tuple[int, int]]:
    inner_h = h - 2
    inner_w = w - 2
    pairs: list[tuple[int, int]] = []
    for red_border_c in range(1, w - 1):
        for green_border_c in range(1, w - 1):
            if red_border_c == green_border_c:
                continue
            sign = 1 if green_border_c > red_border_c else -1
            if desired_sign is not None and sign != desired_sign:
                continue
            regions = _regions_from_inner(inner_h, inner_w, red_border_c - 1, green_border_c - 1)
            if len(regions) != 2:
                continue
            areas = sorted(len(region) for region in regions)
            if areas[0] < MIN_SMALL_REGION or areas[1] - areas[0] < MIN_AREA_DIFF:
                continue
            pairs.append((red_border_c, green_border_c))
    return pairs


def _direction_plan(k: int) -> list[int]:
    plan = []
    if k >= 2:
        plan.extend([-1, 1])
    while len(plan) < k:
        plan.append(random.choice([-1, 1]))
    random.shuffle(plan)
    return plan


def _render_input(shape: tuple[int, int], rects: list[dict[str, int]]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for rect in rects:
        r0 = rect["r0"]
        c0 = rect["c0"]
        r1 = rect["r1"]
        c1 = rect["c1"]
        g[r0, c0 : c1 + 1] = Color.BLUE
        g[r1, c0 : c1 + 1] = Color.BLUE
        g[r0 : r1 + 1, c0] = Color.BLUE
        g[r0 : r1 + 1, c1] = Color.BLUE
        g[r0, rect["red_c"]] = Color.RED
        g[r1, rect["green_c"]] = Color.GREEN
    return g


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    k = random.randint(*RECT_COUNT_RANGE)
    dirs = _direction_plan(k)

    rects: list[dict[str, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for desired_sign in dirs:
        placed = False
        for _ in range(200):
            rh = random.randint(OUTER_H_RANGE[0], min(OUTER_H_RANGE[1], h))
            rw = random.randint(OUTER_W_RANGE[0], min(OUTER_W_RANGE[1], w))
            if rh >= h or rw >= w:
                continue
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if not _boxes_clear(box, boxes):
                continue

            pairs = _valid_marker_pairs(rh, rw, desired_sign)
            if not pairs:
                continue
            red_c_off, green_c_off = random.choice(pairs)
            rects.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + rh - 1,
                    "c1": c0 + rw - 1,
                    "red_c": c0 + red_c_off,
                    "green_c": c0 + green_c_off,
                }
            )
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input((h, w), rects)
    if get_bg(inp) != Color.BLACK:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
