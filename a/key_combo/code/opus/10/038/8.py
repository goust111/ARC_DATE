import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, GREEN, ORANGE)
TABLE_SIZE = 3
GEN_MAX_ATTEMPTS = 400

_GEN_CALL_COUNT = 0
_MARKER_PERMS = list(itertools.permutations(range(TABLE_SIZE)))
_SIZE_PATTERNS = (
    (2, 3, 4),
    (2, 4, 4),
    (3, 3, 4),
)
_BASE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected4(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    h, w = mask.shape
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [tuple(map(int, rc)) for rc in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _shape_variants(arr: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    variants = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(arr) if flip else arr
        for k in range(4):
            rotated = np.rot90(base, k)
            shape = _normalize_coords(np.argwhere(rotated))
            if shape not in seen:
                seen.add(shape)
                variants.append(shape)
    return variants


def _all_shape_variants() -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    variants = []
    for base in _BASE_SHAPES:
        for shape in _shape_variants(base):
            if shape not in seen:
                seen.add(shape)
                variants.append(shape)
    return variants


_SHAPE_VARIANTS = _all_shape_variants()


def _cell_origin(index: int, cell_size: int) -> int:
    return int(index) * (int(cell_size) + 1)


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(map(int, divider_indices)), int(length)]
    runs = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            runs.append((left + 1, right - 1))
    return runs


def _parse_table(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_dividers = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_dividers = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(row_dividers) == TABLE_SIZE - 1
    assert len(col_dividers) == TABLE_SIZE - 1

    row_runs = _non_divider_runs(g.shape[0], row_dividers)
    col_runs = _non_divider_runs(g.shape[1], col_dividers)
    assert len(row_runs) == TABLE_SIZE
    assert len(col_runs) == TABLE_SIZE

    markers: dict[tuple[int, int], int] = {}
    masks: list[tuple[int, int, int, int, int, int]] = []
    empty_cells: list[tuple[int, int]] = []
    mask_shapes = []

    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(region))) - {BLACK}
            assert GREY not in colors

            has_mask = RED in colors
            marker_colors = sorted(colors & set(MARKER_COLORS))
            other_colors = colors - {RED, *MARKER_COLORS}
            assert not other_colors

            if has_mask:
                assert not marker_colors
                mask = region == RED
                assert 2 <= int(mask.sum()) <= 4
                assert _is_connected4(mask)
                masks.append((ri, ci, r0, r1, c0, c1))
                mask_shapes.append(_normalize_coords(np.argwhere(mask)))
            elif marker_colors:
                assert len(marker_colors) == 1
                marker_color = int(marker_colors[0])
                assert int(np.sum(region == marker_color)) == 1
                markers[(ri, ci)] = marker_color
            else:
                empty_cells.append((ri, ci))

    assert len(markers) == 3
    assert len(masks) == 3
    assert len(empty_cells) == 3
    assert len({ri for ri, _ in markers}) == 3
    assert len({ci for _, ci in markers}) == 3
    assert len({ri for ri, *_ in masks}) == 3
    assert len({ci for _, ci, *_ in masks}) == 3
    assert len(set(mask_shapes)) == 3

    return row_runs, col_runs, markers, masks


def _shifted_markers(markers: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    return {((ri + 1) % TABLE_SIZE, ci): color for (ri, ci), color in markers.items()}


def _apply_rule(grid: np.ndarray, markers: dict[tuple[int, int], int], masks):
    shifted = _shifted_markers(markers)
    mask_positions = {(ri, ci) for ri, ci, *_ in masks}
    assert mask_positions == set(shifted)

    out = np.asarray(grid, dtype=int).copy()
    for ri, ci, r0, r1, c0, c1 in masks:
        aligned = [
            color
            for (marker_ri, marker_ci), color in shifted.items()
            if marker_ri == ri or marker_ci == ci
        ]
        assert len(aligned) == 1

        original_aligned = [
            color
            for (marker_ri, marker_ci), color in markers.items()
            if marker_ri == ri or marker_ci == ci
        ]
        assert len(original_aligned) == 2

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == RED] = aligned[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each red mask cell by the marker whose virtual table position is obtained after a one-row cyclic downward shift.

    Input structure:
    - A black background contains a full grey separator table that partitions the grid into a 3x3 array of rectangular cells.
    - Exactly three cells contain singleton markers, with distinct colors from blue, green, and orange, and the three marker cells occupy different table rows and different table columns.
    - Exactly three other cells contain distinct 4-connected red mask shapes of size 2-4 pixels, also occupying different table rows and different table columns.

    Rule:
    - Read the 3x3 table coordinates of the marker cells and the mask cells.
    - Move every marker one table row downward with wraparound: row 0->1, row 1->2, row 2->0, while keeping its table column unchanged.
    - For each mask cell, find the unique shifted marker that shares the same table row or the same table column with that mask cell.
    - Recolor every red pixel in that mask cell to the matched marker color, and leave marker pixels, separators, empty cells, and background unchanged.

    Key insight:
    - The apparent marker-to-mask alignment is not based on the visible marker cells; the correct row/column match only appears after mentally applying the hidden cyclic row shift to the markers.
    """
    g = np.asarray(grid, dtype=int)
    _, _, markers, masks = _parse_table(g)
    return _apply_rule(g, markers, masks)


def _choose_cell_sizes(mode: int) -> tuple[int, int]:
    if mode < 9:
        return 3 + (mode % 3), 3 + ((mode // 3) % 3)
    return random.randint(3, 5), random.randint(3, 5)


def _choose_marker_layout(mode: int):
    marker_cols = _MARKER_PERMS[mode % len(_MARKER_PERMS)] if mode < len(_MARKER_PERMS) else random.choice(_MARKER_PERMS)
    marker_colors = list(MARKER_COLORS)
    if mode < len(_MARKER_PERMS):
        shift = mode % len(marker_colors)
        marker_colors = marker_colors[shift:] + marker_colors[:shift]
    else:
        random.shuffle(marker_colors)
    return marker_cols, marker_colors


def _choose_mask_shapes(cell_h: int, cell_w: int, mode: int) -> list[tuple[tuple[int, int], ...]]:
    fitting = {2: [], 3: [], 4: []}
    for shape in _SHAPE_VARIANTS:
        h, w = _shape_dims(shape)
        if h <= cell_h and w <= cell_w:
            fitting[len(shape)].append(shape)

    pattern = _SIZE_PATTERNS[mode % len(_SIZE_PATTERNS)]
    chosen = []
    used = set()

    for size in pattern:
        options = [shape for shape in fitting[size] if shape not in used]
        if not options:
            break
        shape = random.choice(options)
        chosen.append(shape)
        used.add(shape)

    if len(chosen) < 3:
        pool = [shape for size in (2, 3, 4) for shape in fitting[size] if shape not in used]
        if len(pool) < 3 - len(chosen):
            raise ValueError("not enough fitting mask variants")
        chosen.extend(random.sample(pool, 3 - len(chosen)))

    random.shuffle(chosen)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    return chosen


def _paint_separators(grid: np.ndarray, cell_h: int, cell_w: int) -> None:
    for divider_row in range(1, TABLE_SIZE):
        grid[_cell_origin(divider_row, cell_h) - 1, :] = GREY
    for divider_col in range(1, TABLE_SIZE):
        grid[:, _cell_origin(divider_col, cell_w) - 1] = GREY


def _place_marker(grid: np.ndarray, cell_h: int, cell_w: int, ri: int, ci: int, color: int) -> None:
    r0 = _cell_origin(ri, cell_h)
    c0 = _cell_origin(ci, cell_w)
    rr = r0 + random.randint(0, cell_h - 1)
    cc = c0 + random.randint(0, cell_w - 1)
    grid[rr, cc] = color


def _place_mask(grid: np.ndarray, cell_h: int, cell_w: int, ri: int, ci: int, shape: tuple[tuple[int, int], ...]) -> None:
    r0 = _cell_origin(ri, cell_h)
    c0 = _cell_origin(ci, cell_w)
    shape_h, shape_w = _shape_dims(shape)
    off_r = random.randint(0, cell_h - shape_h)
    off_c = random.randint(0, cell_w - shape_w)
    for dr, dc in shape:
        grid[r0 + off_r + dr, c0 + off_c + dc] = RED


def _wrong_original_row_output(grid: np.ndarray, markers: dict[tuple[int, int], int], masks) -> np.ndarray:
    row_color = {ri: color for (ri, _), color in markers.items()}
    out = np.asarray(grid, dtype=int).copy()
    for ri, _, r0, r1, c0, c1 in masks:
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == RED] = row_color[ri]
    return out


def _gen_attempt(mode: int) -> np.ndarray | None:
    cell_h, cell_w = _choose_cell_sizes(mode)
    height = TABLE_SIZE * cell_h + (TABLE_SIZE - 1)
    width = TABLE_SIZE * cell_w + (TABLE_SIZE - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    _paint_separators(grid, cell_h, cell_w)

    marker_cols, marker_colors = _choose_marker_layout(mode)
    markers: dict[tuple[int, int], int] = {}
    for ri, (ci, color) in enumerate(zip(marker_cols, marker_colors)):
        markers[(ri, ci)] = int(color)
        _place_marker(grid, cell_h, cell_w, ri, ci, int(color))

    shifted = _shifted_markers(markers)
    try:
        shapes = _choose_mask_shapes(cell_h, cell_w, mode)
    except ValueError:
        return None

    mask_items = list(shifted.items())
    random.shuffle(mask_items)
    for ((ri, ci), _color), shape in zip(mask_items, shapes):
        _place_mask(grid, cell_h, cell_w, ri, ci, shape)

    try:
        _, _, parsed_markers, masks = _parse_table(grid)
        expected = _apply_rule(grid, parsed_markers, masks)
        wrong = _wrong_original_row_output(grid, parsed_markers, masks)
        if np.array_equal(wrong, expected):
            return None
        solved = main(grid)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid cyclic-shift separator-table input")
