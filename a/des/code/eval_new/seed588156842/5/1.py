from __future__ import annotations

import random

import numpy as np


def _is_contiguous(values: list[int]) -> bool:
    if not values:
        return False
    return values == list(range(values[0], values[-1] + 1))


def _infer_rectangle(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"missing pixels for color {color}"

    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    horizontal_bbox = None
    vertical_bbox = None

    if len(rows) == 2:
        cols_by_row = []
        for row in rows:
            row_cols = sorted(int(c) for c in coords[coords[:, 0] == row, 1])
            cols_by_row.append(row_cols)
        if (
            cols_by_row[0] == cols_by_row[1]
            and len(cols_by_row[0]) >= 2
            and _is_contiguous(cols_by_row[0])
        ):
            horizontal_bbox = (rows[0], cols_by_row[0][0], rows[1], cols_by_row[0][-1])

    if len(cols) == 2:
        rows_by_col = []
        for col in cols:
            col_rows = sorted(int(r) for r in coords[coords[:, 1] == col, 0])
            rows_by_col.append(col_rows)
        if (
            rows_by_col[0] == rows_by_col[1]
            and len(rows_by_col[0]) >= 2
            and _is_contiguous(rows_by_col[0])
        ):
            vertical_bbox = (rows_by_col[0][0], cols[0], rows_by_col[0][-1], cols[1])

    assert horizontal_bbox is not None or vertical_bbox is not None, (
        f"color {color} does not form two aligned opposite-edge fragments"
    )
    if horizontal_bbox is not None and vertical_bbox is not None:
        assert horizontal_bbox == vertical_bbox, f"ambiguous rectangle for color {color}"
        return horizontal_bbox
    if horizontal_bbox is not None:
        return horizontal_bbox
    return vertical_bbox


def _render_output(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(
        rectangles,
        key=lambda item: (
            (item[2] - item[0] + 1) * (item[3] - item[1] + 1),
            item[0],
            item[1],
            item[2],
            item[3],
        ),
    )
    for r0, c0, r1, c1, color in ordered:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each rectangle from two same-color border fragments, fill it solid, and overlay the rectangles by increasing area.

    Input structure:
    - Black background with several colored straight-line fragments scattered on the grid.
    - Each non-black color appears exactly twice, as two horizontal segments on opposite rows or two vertical segments on opposite columns.
    - The two same-color fragments have equal length and aligned endpoints, so they are surviving pieces of one axis-aligned rectangle border.

    Rule:
    - For each color, use the fragment rows or columns as one pair of rectangle sides.
    - Use the shared start/end coordinates of the fragments as the other pair of sides, which uniquely recovers the full rectangle.
    - Fill that whole rectangle, including its interior, with the fragment color.
    - Draw all recovered rectangles onto a black canvas of the same size, from smaller area to larger area, letting later rectangles overwrite earlier ones.

    Key insight:
    - Two equal aligned segments of one color are not separate objects; they are opposite border remnants whose combined span already fixes the missing rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        r0, c0, r1, c1 = _infer_rectangle(g, color)
        rectangles.append((r0, c0, r1, c1, int(color)))

    return _render_output(g.shape, rectangles)


def _fragment_cells(rect: tuple[int, int, int, int], orientation: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if orientation == "h":
        return {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
    assert orientation == "v"
    return {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}


def _cells_have_padding(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    height, width = shape
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for rr in range(max(0, r - 1), min(height - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(width - 1, c + 1) + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_rectangle(
    height: int,
    width: int,
    orientation: str,
    used_areas: set[int],
    occupied_fragments: set[tuple[int, int]],
    existing_rects: list[tuple[int, int, int, int]],
    require_overlap: bool,
) -> tuple[int, int, int, int] | None:
    max_rect_h = min(height, 12)
    max_rect_w = min(width, 12)

    min_rect_h = 3 if orientation == "h" else 2
    min_rect_w = 2 if orientation == "h" else 3
    if max_rect_h < min_rect_h or max_rect_w < min_rect_w:
        return None

    for _ in range(300):
        rect_h = random.randint(min_rect_h, max_rect_h)
        rect_w = random.randint(min_rect_w, max_rect_w)
        area = rect_h * rect_w
        if area in used_areas:
            continue

        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

        if require_overlap and not any(_rectangles_overlap(rect, other) for other in existing_rects):
            continue

        cells = _fragment_cells(rect, orientation)
        if not _cells_have_padding(cells, occupied_fragments, (height, width)):
            continue

        return rect
    return None


def _sample_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["h", "v"])]
    if random.random() < 0.7:
        split = random.randint(1, count - 1)
        orientations = ["h"] * split + ["v"] * (count - split)
        random.shuffle(orientations)
        return orientations
    return [random.choice(["h", "v"]) for _ in range(count)]


def _generate_attempt() -> np.ndarray | None:
    rect_count = random.randint(2, 5)
    min_side = max(12, 6 + 2 * rect_count)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    colors = random.sample(list(range(1, 10)), rect_count)
    orientations = _sample_orientations(rect_count)
    want_overlap = rect_count >= 2 and random.random() < 0.65

    rects: list[tuple[int, int, int, int]] = []
    occupied_fragments: set[tuple[int, int]] = set()
    used_areas: set[int] = set()

    for idx, orientation in enumerate(orientations):
        require_overlap = bool(want_overlap and idx == 1)
        rect = _sample_rectangle(
            height,
            width,
            orientation,
            used_areas,
            occupied_fragments,
            rects,
            require_overlap,
        )
        if rect is None:
            return None
        rects.append(rect)
        used_areas.add((rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1))
        occupied_fragments.update(_fragment_cells(rect, orientation))

    overlap_pairs = sum(
        1
        for i in range(len(rects))
        for j in range(i + 1, len(rects))
        if _rectangles_overlap(rects[i], rects[j])
    )
    if want_overlap and overlap_pairs == 0:
        return None

    input_grid = np.zeros((height, width), dtype=int)
    rectangle_specs: list[tuple[int, int, int, int, int]] = []
    for color, orientation, rect in zip(colors, orientations, rects, strict=False):
        for r, c in _fragment_cells(rect, orientation):
            input_grid[r, c] = int(color)
        rectangle_specs.append((rect[0], rect[1], rect[2], rect[3], int(color)))

    expected = _render_output(input_grid.shape, rectangle_specs)

    if int((input_grid != 0).sum()) >= int(0.35 * input_grid.size):
        return None
    if int((expected != 0).sum()) >= int(0.8 * expected.size):
        return None
    if np.array_equal(input_grid, expected):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle after repeated attempts")
