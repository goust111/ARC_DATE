from __future__ import annotations

import numpy as np


GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 200
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _transform_shape(coords: np.ndarray, rotations: int, flip: bool) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    if flip:
        arr[:, 1] *= -1
    for _ in range(rotations % 4):
        arr = np.column_stack([arr[:, 1], -arr[:, 0]])
    return _normalize(arr)


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        area = int(rng.integers(2, 6))
        cells = {(0, 0)}
        while len(cells) < area:
            base = tuple(np.asarray(tuple(cells)[int(rng.integers(len(cells)))], dtype=int))
            dr, dc = DIRS4[int(rng.integers(4))]
            cells.add((int(base[0] + dr), int(base[1] + dc)))
        shape = _normalize(np.asarray(sorted(cells), dtype=int))
        shape = _transform_shape(shape, int(rng.integers(4)), bool(rng.integers(2)))
        if int(shape[:, 0].max()) < 5 and int(shape[:, 1].max()) < 5:
            return shape
    raise RuntimeError("failed to sample a small connected shape")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([dr, dc], dtype=int)


def _inside(coords: np.ndarray, height: int, width: int) -> bool:
    arr = np.asarray(coords, dtype=int)
    return bool(
        np.all((0 <= arr[:, 0]) & (arr[:, 0] < height))
        and np.all((0 <= arr[:, 1]) & (arr[:, 1] < width))
    )


def _expand(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _swept_cells(src: np.ndarray, dst: np.ndarray) -> set[tuple[int, int]]:
    src_arr = np.asarray(src, dtype=int)
    dst_arr = np.asarray(dst, dtype=int)
    dr = int(dst_arr[0, 0] - src_arr[0, 0])
    dc = int(dst_arr[0, 1] - src_arr[0, 1])
    assert np.all(dst_arr - src_arr == np.array([dr, dc], dtype=int))
    assert dr == 0 or dc == 0
    steps = max(abs(dr), abs(dc))
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    out: set[tuple[int, int]] = set()
    for k in range(steps + 1):
        moved = _translate(src_arr, k * step_r, k * step_c)
        out |= _coords_set(moved)
    return out


def _touching(a: np.ndarray, b: np.ndarray) -> bool:
    bset = _coords_set(b)
    for r, c in _coords_set(a):
        if (r + 1, c) in bset or (r - 1, c) in bset or (r, c + 1) in bset or (r, c - 1) in bset:
            return True
    return False


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, np.ndarray | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, np.ndarray | tuple[int, int, int, int]]] = []
    for r, c in np.argwhere(g == int(color)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((int(cr), int(cc)))
            for dr, dc in DIRS4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        coords = np.asarray(pts, dtype=int)
        components.append({"coords": coords, "bbox": _bbox(coords)})
    return components


def _meeting_shift_horizontal(left: np.ndarray, right: np.ndarray) -> int:
    left_front: dict[int, int] = {}
    right_front: dict[int, int] = {}
    for r, c in np.asarray(left, dtype=int):
        row = int(r)
        col = int(c)
        left_front[row] = max(col, left_front.get(row, col))
    for r, c in np.asarray(right, dtype=int):
        row = int(r)
        col = int(c)
        right_front[row] = min(col, right_front.get(row, col))
    shared_rows = sorted(set(left_front) & set(right_front))
    assert shared_rows, "horizontal pairs must share at least one occupied row"
    distances = [int(right_front[row] - left_front[row]) for row in shared_rows]
    assert all(dist > 0 for dist in distances), "horizontal pair order is invalid"
    return min((dist - 1) // 2 for dist in distances)


def _meeting_shift_vertical(top: np.ndarray, bottom: np.ndarray) -> int:
    top_front: dict[int, int] = {}
    bottom_front: dict[int, int] = {}
    for r, c in np.asarray(top, dtype=int):
        row = int(r)
        col = int(c)
        top_front[col] = max(row, top_front.get(col, row))
    for r, c in np.asarray(bottom, dtype=int):
        row = int(r)
        col = int(c)
        bottom_front[col] = min(row, bottom_front.get(col, row))
    shared_cols = sorted(set(top_front) & set(bottom_front))
    assert shared_cols, "vertical pairs must share at least one occupied column"
    distances = [int(bottom_front[col] - top_front[col]) for col in shared_cols]
    assert all(dist > 0 for dist in distances), "vertical pair order is invalid"
    return min((dist - 1) // 2 for dist in distances)


def _collapse_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ra0, ca0, ra1, ca1 = _bbox(a)
    rb0, cb0, rb1, cb1 = _bbox(b)
    row_overlap = max(ra0, rb0) <= min(ra1, rb1)
    col_overlap = max(ca0, cb0) <= min(ca1, cb1)

    if row_overlap and not col_overlap:
        left, right = (a, b) if ca0 <= cb0 else (b, a)
        shift = _meeting_shift_horizontal(left, right)
        moved_left = _translate(left, 0, shift)
        moved_right = _translate(right, 0, -shift)
        assert _coords_set(moved_left).isdisjoint(_coords_set(moved_right))
        assert _touching(moved_left, moved_right), "horizontal pair must stop when the objects touch"
        return moved_left, moved_right

    if col_overlap and not row_overlap:
        top, bottom = (a, b) if ra0 <= rb0 else (b, a)
        shift = _meeting_shift_vertical(top, bottom)
        moved_top = _translate(top, shift, 0)
        moved_bottom = _translate(bottom, -shift, 0)
        assert _coords_set(moved_top).isdisjoint(_coords_set(moved_bottom))
        assert _touching(moved_top, moved_bottom), "vertical pair must stop when the objects touch"
        return moved_top, moved_bottom

    raise AssertionError("each same-color pair must be aligned on exactly one axis")


def _row_touch_offset(shape: np.ndarray) -> int:
    best = 0
    for row in np.unique(np.asarray(shape, dtype=int)[:, 0]):
        cols = shape[shape[:, 0] == int(row), 1]
        best = max(best, int(cols.max() - cols.min() + 1))
    return best


def _col_touch_offset(shape: np.ndarray) -> int:
    best = 0
    for col in np.unique(np.asarray(shape, dtype=int)[:, 1]):
        rows = shape[shape[:, 1] == int(col), 0]
        best = max(best, int(rows.max() - rows.min() + 1))
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-colored object pair toward the middle until the two pieces touch.

    Input structure:
    - The background is Black.
    - The foreground consists of several small 4-connected monochrome objects.
    - Each non-black color appears in exactly two disconnected objects of area 2-5.
    - The two objects of a color are aligned either horizontally or vertically.

    Rule:
    - For each color, find its two 4-connected objects.
    - If their row spans overlap, move the left object rightward and the right object leftward by equal amounts; if their column spans overlap, move the upper object downward and the lower object upward by equal amounts.
    - Both objects stop at the last position where they are still disjoint and now touch by a 4-neighbor edge.
    - Preserve every object's shape and color, and combine the moved results for all colors on the same black background.

    Key insight:
    - The two objects of each color must be treated as a synchronized pair: they move symmetrically toward each other, so the final joined shape sits at the midpoint of the original separation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    bg = 0
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    out = np.zeros_like(g)
    occupied: set[tuple[int, int]] = set()
    for color in colors:
        components = _components_of_color(g, color)
        assert len(components) == 2, "each foreground color must appear in exactly two components"
        for component in components:
            area = int(len(component["coords"]))  # type: ignore[index]
            assert 2 <= area <= 5, "objects must have area between 2 and 5"

        moved_a, moved_b = _collapse_pair(
            np.asarray(components[0]["coords"], dtype=int),  # type: ignore[index]
            np.asarray(components[1]["coords"], dtype=int),  # type: ignore[index]
        )
        assert len(moved_a) + len(moved_b) == int(np.count_nonzero(g == color))

        for coords in (moved_a, moved_b):
            assert _inside(coords, g.shape[0], g.shape[1])
            for cell in _coords_set(coords):
                assert cell not in occupied, "different color pairs must not overlap after moving"
                occupied.add(cell)
                out[cell] = int(color)

    return out


def _place_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    color: int,
    orientation: str,
    input_halo: set[tuple[int, int]],
    output_halo: set[tuple[int, int]],
    sweep_reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    height, width = grid.shape

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        shape = _random_shape(rng)
        shp_h = int(shape[:, 0].max() + 1)
        shp_w = int(shape[:, 1].max() + 1)

        if orientation == "h":
            touch = _row_touch_offset(shape)
            max_shift = min(5, (width - touch - shp_w) // 2)
            if max_shift < 1:
                continue
            shift = int(rng.integers(1, max_shift + 1))
            left_min = shift
            left_max = width - touch - shp_w - shift
            if left_min > left_max or shp_h > height:
                continue
            r0 = int(rng.integers(0, height - shp_h + 1))
            c0 = int(rng.integers(left_min, left_max + 1))

            final_a = _translate(shape, r0, c0)
            final_b = _translate(shape, r0, c0 + touch)
            src_a = _translate(final_a, 0, -shift)
            src_b = _translate(final_b, 0, shift)
        else:
            touch = _col_touch_offset(shape)
            max_shift = min(5, (height - touch - shp_h) // 2)
            if max_shift < 1:
                continue
            shift = int(rng.integers(1, max_shift + 1))
            top_min = shift
            top_max = height - touch - shp_h - shift
            if top_min > top_max or shp_w > width:
                continue
            r0 = int(rng.integers(top_min, top_max + 1))
            c0 = int(rng.integers(0, width - shp_w + 1))

            final_a = _translate(shape, r0, c0)
            final_b = _translate(shape, r0 + touch, c0)
            src_a = _translate(final_a, -shift, 0)
            src_b = _translate(final_b, shift, 0)

        if not (_inside(src_a, height, width) and _inside(src_b, height, width)):
            continue
        if not (_inside(final_a, height, width) and _inside(final_b, height, width)):
            continue

        input_cells = _coords_set(src_a) | _coords_set(src_b)
        final_cells = _coords_set(final_a) | _coords_set(final_b)
        swept_cells = _swept_cells(src_a, final_a) | _swept_cells(src_b, final_b)

        if not input_cells.isdisjoint(input_halo):
            continue
        if not final_cells.isdisjoint(output_halo):
            continue
        if not swept_cells.isdisjoint(sweep_reserved):
            continue

        moved_a, moved_b = _collapse_pair(src_a, src_b)
        if (_coords_set(moved_a) | _coords_set(moved_b)) != final_cells:
            continue

        for r, c in input_cells:
            grid[int(r), int(c)] = int(color)
        for r, c in final_cells:
            expected[int(r), int(c)] = int(color)

        input_halo |= _expand(input_cells, height, width)
        output_halo |= _expand(final_cells, height, width)
        sweep_reserved |= swept_cells
        return True

    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    min_side = 12 + 3 * pair_count
    height = int(rng.integers(min_side, 31))
    width = int(rng.integers(min_side, 31))
    colors = [int(c) for c in rng.choice(np.array(PALETTE, dtype=int), size=pair_count, replace=False)]

    orientations = ["h" if bool(bit) else "v" for bit in rng.integers(0, 2, size=pair_count)]
    if pair_count >= 2 and len(set(orientations)) == 1:
        flip_idx = int(rng.integers(pair_count))
        orientations[flip_idx] = "v" if orientations[flip_idx] == "h" else "h"

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    input_halo: set[tuple[int, int]] = set()
    output_halo: set[tuple[int, int]] = set()
    sweep_reserved: set[tuple[int, int]] = set()

    for color, orientation in zip(colors, orientations):
        ok = _place_pair(
            grid=grid,
            expected=expected,
            color=int(color),
            orientation=orientation,
            input_halo=input_halo,
            output_halo=output_halo,
            sweep_reserved=sweep_reserved,
            rng=rng,
        )
        if not ok:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
