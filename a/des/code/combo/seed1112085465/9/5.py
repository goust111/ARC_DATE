from __future__ import annotations

import random

import numpy as np


AZURE = 8
MAROON = 9
GREY = 5
BLUE = 1
RED = 2
GREEN = 3

MAX_GRID_SIZE = 30
MIN_INTERVALS = 4
MAX_INTERVALS = 10
MAX_MEASURE = 4
MAX_OBJECTS = 4
GEN_ATTEMPTS = 400


def _find_spacing(edge: np.ndarray) -> int:
    marks = np.flatnonzero(np.asarray(edge, dtype=int) == MAROON)
    assert len(marks) >= 2, "need at least two ruler marks"
    diffs = np.diff(marks)
    spacing = int(diffs[0])
    assert spacing > 0 and np.all(diffs == spacing), "ruler marks must be equally spaced"
    return spacing


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []

        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        rows = [rr for rr, _ in coords]
        cols = [cc for _, cc in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey objects must be solid rectangles"
        boxes.append((r0, c0, r1, c1))

    boxes.sort()
    return boxes


def _spread_positions(span: int, count: int) -> list[int]:
    assert 0 <= count <= span
    if count == 0:
        return []
    positions = [((2 * i + 1) * span) // (2 * count) for i in range(count)]
    assert len(set(positions)) == count
    return positions


def _paint_stripes(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    horizontal_count: int,
    vertical_count: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert horizontal_count <= inner_h and vertical_count <= inner_w

    stripe_rows = [r0 + 1 + rr for rr in _spread_positions(inner_h, horizontal_count)]
    stripe_cols = [c0 + 1 + cc for cc in _spread_positions(inner_w, vertical_count)]

    for rr in stripe_rows:
        out[rr, c0 + 1 : c1] = RED
    for cc in stripe_cols:
        out[r0 + 1 : r1, cc] = BLUE
    for rr in stripe_rows:
        for cc in stripe_cols:
            out[rr, cc] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the ruler spacings from the top and left edges, then encode each grey
    rectangle's size-in-intervals as horizontal and vertical interior stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    dy = _find_spacing(g[:, 0])
    dx = _find_spacing(g[0, :])

    out = g.copy()
    for bbox in _component_bboxes(g == GREY):
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        assert height >= 3 and width >= 3
        assert (height - 1) % dy == 0 and (width - 1) % dx == 0
        hy = (height - 1) // dy
        hx = (width - 1) // dx
        _paint_stripes(out, bbox, hy, hx)

    return out


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _candidate_specs(
    row_intervals: int,
    col_intervals: int,
    dy: int,
    dx: int,
) -> list[tuple[tuple[int, int, int, int], int, int]]:
    specs: list[tuple[tuple[int, int, int, int], int, int]] = []
    max_hy = min(MAX_MEASURE, row_intervals - 2)
    max_hx = min(MAX_MEASURE, col_intervals - 2)

    for hy in range(1, max_hy + 1):
        for hx in range(1, max_hx + 1):
            for start_y in range(1, row_intervals - hy):
                for start_x in range(1, col_intervals - hx):
                    r0 = start_y * dy
                    c0 = start_x * dx
                    r1 = (start_y + hy) * dy
                    c1 = (start_x + hx) * dx
                    specs.append(((r0, c0, r1, c1), hy, hx))
    return specs


def _render_input(
    shape: tuple[int, int],
    dy: int,
    dx: int,
    objects: list[tuple[tuple[int, int, int, int], int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), AZURE, dtype=int)
    grid[::dy, 0] = MAROON
    grid[0, ::dx] = MAROON
    for bbox, _hy, _hx in objects:
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _render_expected(
    inp: np.ndarray,
    objects: list[tuple[tuple[int, int, int, int], int, int]],
) -> np.ndarray:
    out = inp.copy()
    for bbox, hy, hx in objects:
        _paint_stripes(out, bbox, hy, hx)
    return out


def _choose_objects(
    row_intervals: int,
    col_intervals: int,
    dy: int,
    dx: int,
) -> list[tuple[tuple[int, int, int, int], int, int]] | None:
    specs = _candidate_specs(row_intervals, col_intervals, dy, dx)
    if not specs:
        return None

    desired = random.randint(1, MAX_OBJECTS)
    random.shuffle(specs)
    chosen: list[tuple[tuple[int, int, int, int], int, int]] = []

    for spec in specs:
        bbox, _hy, _hx = spec
        if all(_boxes_have_gap(bbox, other_bbox) for other_bbox, _, _ in chosen):
            chosen.append(spec)
            if len(chosen) == desired:
                break

    if len(chosen) < desired:
        return None
    return chosen


def _gen_attempt() -> np.ndarray | None:
    dy = random.randint(2, 5)
    dx = random.randint(2, 5)

    max_row_intervals = min(MAX_INTERVALS, (MAX_GRID_SIZE - 1) // dy)
    max_col_intervals = min(MAX_INTERVALS, (MAX_GRID_SIZE - 1) // dx)
    if max_row_intervals < MIN_INTERVALS or max_col_intervals < MIN_INTERVALS:
        return None

    row_intervals = random.randint(MIN_INTERVALS, max_row_intervals)
    col_intervals = random.randint(MIN_INTERVALS, max_col_intervals)
    objects = _choose_objects(row_intervals, col_intervals, dy, dx)
    if objects is None:
        return None

    h = row_intervals * dy + 1
    w = col_intervals * dx + 1
    inp = _render_input((h, w), dy, dx, objects)
    expected = _render_expected(inp, objects)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
