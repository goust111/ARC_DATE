import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


RAW_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
)


def _build_shape_library():
    library = []
    for coords in RAW_SHAPES:
        h = max(r for r, _c in coords) + 1
        w = max(c for _r, c in coords) + 1
        library.append({"coords": coords, "bbox": (h, w), "area": len(coords)})
    return tuple(library)


SHAPE_LIBRARY = _build_shape_library()


def _full_color_separator(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        r
        for r in range(h)
        if int(grid[r, 0]) != BLACK and np.all(grid[r] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(w)
        if int(grid[0, c]) != BLACK and np.all(grid[:, c] == grid[0, c])
    ]
    matches = []
    for r in row_candidates:
        row_color = int(grid[r, 0])
        for c in col_candidates:
            if row_color == int(grid[0, c]) == int(grid[r, c]):
                matches.append((r, c, row_color))
    assert len(matches) == 1, "expected a unique separator cross"
    sep_r, sep_c, sep_color = matches[0]
    assert sep_r == grid.shape[0] - sep_r - 1
    assert sep_c == grid.shape[1] - sep_c - 1
    return sep_r, sep_c, sep_color


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _signature(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    sep_r, sep_c, _sep_color = _full_color_separator(grid)
    return [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]


def _analyze_shapes(grid: np.ndarray) -> tuple[dict[tuple[tuple[int, int], ...], tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    all_shapes: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    per_quadrant: list[set[tuple[tuple[int, int], ...]]] = []

    for quadrant in _quadrants(grid):
        comps = _components(quadrant != BLACK)
        present: set[tuple[tuple[int, int], ...]] = set()
        for cells in comps:
            colors = {int(quadrant[r, c]) for r, c in cells}
            assert len(colors) == 1, "each shape must be monochromatic"
            color = colors.pop()
            sig = _signature(cells)
            area = len(cells)
            previous = all_shapes.get(sig)
            if previous is None:
                all_shapes[sig] = (color, area)
            else:
                assert previous == (color, area), "same shape must keep its color and area"
            present.add(sig)
        per_quadrant.append(present)

    assert 4 <= len(all_shapes) <= 5
    missing: list[tuple[tuple[int, int], ...]] = []
    for present in per_quadrant:
        absent = [sig for sig in all_shapes if sig not in present]
        assert len(absent) == 1, "each quadrant must miss exactly one standard shape"
        missing.append(absent[0])
    assert len(set(missing)) == 4, "the four missing shapes must be distinct"
    return all_shapes, missing


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the shape sets in the four quadrants, recover the missing shape in each
    quadrant, and encode each missing shape as a horizontal bar whose length equals
    its area.
    """
    g = np.asarray(grid, dtype=int)
    all_shapes, missing = _analyze_shapes(g)
    widths = [all_shapes[sig][1] for sig in missing]
    out = np.full((4, max(widths)), BLACK, dtype=int)
    for r, sig in enumerate(missing):
        color, area = all_shapes[sig]
        out[r, :area] = color
    return out


def _slot_boxes(
    qh: int,
    qw: int,
    *,
    sep_top: bool,
    sep_bottom: bool,
    sep_left: bool,
    sep_right: bool,
) -> list[tuple[int, int, int, int]]:
    row_lo = 1 if sep_top else 0
    row_hi = qh - 1 if sep_bottom else qh
    col_lo = 1 if sep_left else 0
    col_hi = qw - 1 if sep_right else qw
    usable_h = row_hi - row_lo
    usable_w = col_hi - col_lo
    slot_h = min(4, (usable_h - 1) // 2)
    slot_w = min(4, (usable_w - 1) // 2)
    assert slot_h >= 3 and slot_w >= 3
    top_r = row_lo
    bottom_r = row_hi - slot_h
    left_c = col_lo
    right_c = col_hi - slot_w
    return [
        (top_r, top_r + slot_h, left_c, left_c + slot_w),
        (top_r, top_r + slot_h, right_c, right_c + slot_w),
        (bottom_r, bottom_r + slot_h, left_c, left_c + slot_w),
        (bottom_r, bottom_r + slot_h, right_c, right_c + slot_w),
    ]


def _paint_shape(
    canvas: np.ndarray,
    coords: tuple[tuple[int, int], ...],
    color: int,
    box: tuple[int, int, int, int],
) -> None:
    r0, r1, c0, c1 = box
    shape_h = max(r for r, _c in coords) + 1
    shape_w = max(c for _r, c in coords) + 1
    start_r = random.randint(r0, r1 - shape_h)
    start_c = random.randint(c0, c1 - shape_w)
    for dr, dc in coords:
        canvas[start_r + dr, start_c + dc] = color


def _make_quadrant(
    qh: int,
    qw: int,
    shapes: list[dict[str, object]],
    *,
    sep_top: bool,
    sep_bottom: bool,
    sep_left: bool,
    sep_right: bool,
) -> np.ndarray:
    quadrant = np.full((qh, qw), BLACK, dtype=int)
    boxes = _slot_boxes(
        qh,
        qw,
        sep_top=sep_top,
        sep_bottom=sep_bottom,
        sep_left=sep_left,
        sep_right=sep_right,
    )
    slot_ids = random.sample(range(4), len(shapes))
    ordered_shapes = random.sample(shapes, len(shapes))
    for shape, slot_id in zip(ordered_shapes, slot_ids):
        _paint_shape(
            quadrant,
            shape["coords"],
            int(shape["color"]),
            boxes[slot_id],
        )
    return quadrant


def _expected_output(missing_indices: list[int], shapes: list[dict[str, object]]) -> np.ndarray:
    missing_shapes = [shapes[idx] for idx in missing_indices]
    width = max(int(shape["area"]) for shape in missing_shapes)
    out = np.full((4, width), BLACK, dtype=int)
    for r, shape in enumerate(missing_shapes):
        out[r, : int(shape["area"])] = int(shape["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_shapes = random.choice((4, 5))
    side_lo = 8 if n_shapes == 4 else 9
    qh = random.randint(side_lo, 11)
    qw = random.randint(side_lo, 11)

    base_shapes = random.sample(list(SHAPE_LIBRARY), n_shapes)
    colors = random.sample(PALETTE, n_shapes)
    shapes = []
    for shape, color in zip(base_shapes, colors):
        shapes.append(
            {
                "coords": shape["coords"],
                "bbox": shape["bbox"],
                "area": shape["area"],
                "color": color,
            }
        )

    missing_indices = random.sample(range(n_shapes), 4)
    quadrant_shapes = []
    for quadrant_idx in range(4):
        missing_idx = missing_indices[quadrant_idx]
        quadrant_shapes.append(
            [shape for idx, shape in enumerate(shapes) if idx != missing_idx]
        )

    quadrants = [
        _make_quadrant(qh, qw, quadrant_shapes[0], sep_top=False, sep_bottom=True, sep_left=False, sep_right=True),
        _make_quadrant(qh, qw, quadrant_shapes[1], sep_top=False, sep_bottom=True, sep_left=True, sep_right=False),
        _make_quadrant(qh, qw, quadrant_shapes[2], sep_top=True, sep_bottom=False, sep_left=False, sep_right=True),
        _make_quadrant(qh, qw, quadrant_shapes[3], sep_top=True, sep_bottom=False, sep_left=True, sep_right=False),
    ]

    grid = np.full((2 * qh + 1, 2 * qw + 1), BLACK, dtype=int)
    grid[qh, :] = GREY
    grid[:, qw] = GREY
    grid[:qh, :qw] = quadrants[0]
    grid[:qh, qw + 1 :] = quadrants[1]
    grid[qh + 1 :, :qw] = quadrants[2]
    grid[qh + 1 :, qw + 1 :] = quadrants[3]

    expected = _expected_output(missing_indices, shapes)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
