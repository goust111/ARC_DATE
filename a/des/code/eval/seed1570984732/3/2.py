import random

import numpy as np


BG = 8
ARROW = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 9)

PIECE_COUNT_RANGE = (3, 5)
RING_THICKNESS_RANGE = (1, 4)
FILL_CENTER_SIZE_RANGE = (1, 6)
NO_FILL_SPAN_RANGE = (0, 6)
GRID_MARGIN_RANGE = (6, 12)
PLACEMENT_ATTEMPTS = 300
GEN_ATTEMPTS = 500

LEFT_ARROW = ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = coords - np.array([r0, c0], dtype=int)
    order = np.lexsort((shifted[:, 1], shifted[:, 0]))
    return tuple((int(shifted[i, 0]), int(shifted[i, 1])) for i in order)


def _extract_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    rects: list[tuple[int, int, int, int]] = []
    saw_arrow = False

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        if area == h * w and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            rects.append((area, h, w, color))
            continue
        assert _normalize(coords) == LEFT_ARROW, "unexpected non-rectangular object"
        saw_arrow = True

    assert saw_arrow, "missing arrow marker"
    assert len(rects) >= 2, "need at least two colored rectangles"
    areas = [area for area, _h, _w, _color in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    rects.sort(key=lambda item: item[0], reverse=True)
    return rects


def _build_output(rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    _outer_area, outer_h, outer_w, outer_color = rects[0]
    out = np.full((outer_h, outer_w), outer_color, dtype=int)
    inset = 1

    for _area, h, w, color in rects[1:-1]:
        if 2 * inset >= outer_h or 2 * inset >= outer_w:
            break
        out[inset : outer_h - inset, inset : outer_w - inset] = color
        inset += min(h, w)

    if 2 * inset < outer_h and 2 * inset < outer_w:
        out[inset : outer_h - inset, inset : outer_w - inset] = rects[-1][3]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered solid rectangles by area, then rebuild them as nested
    concentric rectangles: the largest rectangle sets the output size, each
    intermediate rectangle contributes one inward layer whose thickness is its
    short side, and the smallest rectangle fills any remaining center.
    """
    return _build_output(_extract_rectangles(np.asarray(grid, dtype=int)))


def _sample_layout(piece_count: int) -> tuple[int, int, list[int]] | None:
    ring_count = piece_count - 2
    thicknesses = [random.randint(*RING_THICKNESS_RANGE) for _ in range(ring_count)]
    total_inset = 1 + sum(thicknesses)
    fill_center = random.choice([False, True])

    if fill_center:
        center_h = random.randint(*FILL_CENTER_SIZE_RANGE)
        center_w = random.randint(*FILL_CENTER_SIZE_RANGE)
    else:
        if random.choice([False, True]):
            center_h = 0
            center_w = random.randint(*NO_FILL_SPAN_RANGE)
        else:
            center_h = random.randint(*NO_FILL_SPAN_RANGE)
            center_w = 0

    outer_h = 2 * total_inset + center_h
    outer_w = 2 * total_inset + center_w
    if not (6 <= outer_h <= 22 and 6 <= outer_w <= 22):
        return None
    return outer_h, outer_w, thicknesses


def _sample_specs() -> list[tuple[int, int]] | None:
    piece_count = random.randint(*PIECE_COUNT_RANGE)
    layout = _sample_layout(piece_count)
    if layout is None:
        return None

    outer_h, outer_w, thicknesses = layout
    specs = [(outer_h, outer_w)]
    prev_area = outer_h * outer_w
    max_side = max(outer_h, outer_w) + 8

    for thickness in thicknesses:
        max_long = min(max_side, (prev_area - 1) // thickness)
        if max_long < thickness:
            return None
        long_side = random.randint(thickness, max_long)
        dims = (thickness, long_side)
        if random.choice([False, True]):
            dims = (dims[1], dims[0])
        specs.append(dims)
        prev_area = dims[0] * dims[1]

    max_area = prev_area - 1
    if max_area < 1:
        return None
    for _ in range(100):
        h = random.randint(1, min(max_side, max_area))
        w = random.randint(1, min(max_side, max_area))
        if h * w < prev_area:
            specs.append((h, w))
            return specs
    return None


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_grid, r0 + h + 1)
    cc1 = min(w_grid, c0 + w + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    h_grid, w_grid = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_grid, r0 + h + 1)
    cc1 = min(w_grid, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _paint_arrow(grid: np.ndarray) -> None:
    r0 = grid.shape[0] - 3
    c0 = grid.shape[1] - 3
    for dr, dc in LEFT_ARROW:
        grid[r0 + dr, c0 + dc] = ARROW


def _scatter_rectangles(rects: list[tuple[int, int, int, int]]) -> np.ndarray | None:
    max_h = max(h for _area, h, _w, _color in rects)
    max_w = max(w for _area, _h, w, _color in rects)
    total_area = sum(area for area, _h, _w, _color in rects)
    base = max(max_h, max_w, int(np.sqrt(total_area * 3)))
    grid_h = random.randint(base + GRID_MARGIN_RANGE[0], base + GRID_MARGIN_RANGE[1])
    grid_w = random.randint(base + GRID_MARGIN_RANGE[0], base + GRID_MARGIN_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    _mark_blocked(blocked, grid_h - 3, grid_w - 3, 3, 3)
    ordered = sorted(rects, key=lambda item: item[0], reverse=True)

    for _area, h, w, color in ordered:
        starts = [(r, c) for r in range(1, grid_h - h) for c in range(1, grid_w - w)]
        random.shuffle(starts)
        placed = False
        for r0, c0 in starts:
            if not _can_place(blocked, r0, c0, h, w):
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = color
            _mark_blocked(blocked, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None

    _paint_arrow(grid)
    return grid


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    areas = [h * w for h, w in specs]
    if len(set(areas)) != len(areas):
        return None
    if any(areas[i] <= areas[i + 1] for i in range(len(areas) - 1)):
        return None

    colors = random.sample(PALETTE, len(specs))
    rects = [(h * w, h, w, color) for (h, w), color in zip(specs, colors)]
    expected = _build_output(rects)
    grid = _scatter_rectangles(rects)
    if grid is None:
        return None

    if _mode_color(grid) != BG:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
