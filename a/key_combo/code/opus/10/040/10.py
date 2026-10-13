from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

PALETTE = (BLUE, RED, GREEN)
MAX_ATTEMPTS = 400
CELL_SIZE_RANGE = (4, 5)
SHAPE_SIZE_RANGE = (2, 5)
MAIN_ROWS = 3
MAIN_COLS = 3
DICT_ROWS = 1
DICT_COLS = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))
    return runs


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _is_single_component(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, rc)) for rc in coords}
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nb in _neighbors4(*cell):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def _extract_single_shape(block: np.ndarray, expected_color: int | None = None) -> tuple[tuple[tuple[int, int], ...], int]:
    colors = set(map(int, np.unique(block))) - {BLACK}
    assert colors, "expected a foreground shape in every block"
    if expected_color is not None:
        assert colors == {expected_color}, "unexpected block colors"
        color = int(expected_color)
    else:
        assert len(colors) == 1, "expected exactly one non-black color per block"
        color = int(next(iter(colors)))
    coords = np.argwhere(block == color)
    assert 2 <= len(coords) <= 5, "shape must have 2..5 cells"
    assert _is_single_component(coords), "shape must be 4-connected"
    return _normalize_shape(coords), color


def _parse_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full_sep_rows = np.all(grid == ORANGE, axis=1)
    full_sep_cols = np.all(grid == ORANGE, axis=0)
    assert full_sep_rows[0] and full_sep_rows[-1], "expected orange top/bottom border"
    assert full_sep_cols[0] and full_sep_cols[-1], "expected orange left/right border"
    row_bands = _runs([r for r in range(grid.shape[0]) if not full_sep_rows[r]])
    col_bands = _runs([c for c in range(grid.shape[1]) if not full_sep_cols[c]])
    assert len(row_bands) == MAIN_ROWS + DICT_ROWS, "expected 3 main rows plus 1 dictionary row"
    assert len(col_bands) == MAIN_COLS, "expected 3 columns"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all blocks must share one size"
    return row_bands, col_bands


def _all_shapes(max_size: int = 5) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(1, max_size):
        next_shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes[size]:
            cells = set(shape)
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    expanded = tuple(sorted(cells | {(nr, nc)}))
                    coords = np.array(expanded, dtype=int)
                    next_shapes.add(_normalize_shape(coords))
        shapes[size + 1] = next_shapes
    return {size: sorted(group) for size, group in shapes.items()}


ALL_SHAPES = _all_shapes(max_size=SHAPE_SIZE_RANGE[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey shape in the upper lattice block by matching it to the colored shape dictionary in the bottom row.

    Input structure:
    - A black grid is divided by full orange separator rows and columns into a 3x3
      upper main area and a 1x3 bottom dictionary row, with equal-sized cells.
    - Every upper cell contains exactly one 4-connected grey shape on black.
    - Every dictionary cell contains exactly one 4-connected colored shape on black,
      using blue, red, and green once each.

    Rule:
    - Read the orange lattice to locate the four row bands and three column bands.
    - Extract the normalized mask of each colored dictionary shape and map that mask
      to its cell color.
    - For every upper cell, extract the normalized mask of its grey shape and look up
      the exactly matching dictionary mask, allowing translation only.
    - Recolor the grey pixels in that cell to the matched dictionary color; leave the
      orange separators, black background, and dictionary row unchanged.

    Key insight:
    - Color is not determined by position or by coarse size statistics; it is encoded
      indirectly by an explicit bottom-row shape dictionary, so exact mask equality is
      the deciding relation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_bands, col_bands = _parse_bands(g)

    dictionary: dict[tuple[tuple[int, int], ...], int] = {}
    seen_colors: list[int] = []
    dict_r0, dict_r1 = row_bands[-1]
    for c0, c1 in col_bands:
        block = g[dict_r0 : dict_r1 + 1, c0 : c1 + 1]
        shape, color = _extract_single_shape(block, expected_color=None)
        assert color in PALETTE, "dictionary colors must be blue/red/green"
        assert shape not in dictionary, "dictionary shapes must be unique"
        dictionary[shape] = color
        seen_colors.append(color)
    assert set(seen_colors) == set(PALETTE), "dictionary must contain blue, red, and green exactly once"

    out = g.copy()
    counts = Counter()
    for r0, r1 in row_bands[:MAIN_ROWS]:
        for c0, c1 in col_bands:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            shape, _ = _extract_single_shape(block, expected_color=GREY)
            assert shape in dictionary, "every grey shape must match a dictionary template"
            color = dictionary[shape]
            counts[color] += 1
            target = out[r0 : r1 + 1, c0 : c1 + 1]
            target[target == GREY] = color
    assert sorted(counts.values()) == [3, 3, 3], "each dictionary shape must appear three times in the main grid"
    return out


def _shape_pool(cell_size: int) -> list[tuple[tuple[int, int], ...]]:
    pool: list[tuple[tuple[int, int], ...]] = []
    for area in range(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1] + 1):
        for shape in ALL_SHAPES[area]:
            h, w = _shape_bbox(shape)
            if h <= cell_size and w <= cell_size:
                pool.append(shape)
    return pool


def _choose_shape_triplet(cell_size: int) -> list[tuple[tuple[int, int], ...]]:
    pool = _shape_pool(cell_size)
    by_area: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    by_area_bbox: dict[tuple[int, tuple[int, int]], list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for shape in pool:
        area = len(shape)
        bbox = _shape_bbox(shape)
        by_area[area].append(shape)
        by_area_bbox[(area, bbox)].append(shape)

    use_size2_branch = 2 in by_area and random.random() < 0.35
    if use_size2_branch:
        pair_groups = [group for (area, _bbox), group in by_area_bbox.items() if area >= 3 and len(group) >= 2]
        if pair_groups:
            pair = random.sample(random.choice(pair_groups), 2)
            domino = random.choice(by_area[2])
            return [domino, *pair]

    same_area_candidates = [area for area, group in by_area.items() if area >= 3 and len(group) >= 3]
    if same_area_candidates:
        random.shuffle(same_area_candidates)
        for area in same_area_candidates:
            shapes = by_area[area][:]
            random.shuffle(shapes)
            for i in range(len(shapes)):
                for j in range(i + 1, len(shapes)):
                    for k in range(j + 1, len(shapes)):
                        triple = [shapes[i], shapes[j], shapes[k]]
                        bboxes = [_shape_bbox(shape) for shape in triple]
                        if len(set(bboxes)) < 3:
                            return triple

    for _ in range(200):
        triple = random.sample(pool, 3)
        areas = [len(shape) for shape in triple]
        bboxes = [_shape_bbox(shape) for shape in triple]
        if len(set(areas)) < 3 and len(set(bboxes)) < 3:
            return triple
    raise RuntimeError("failed to choose a counterexample-rich shape triplet")


def _offset_choices(shape: tuple[tuple[int, int], ...], cell_size: int) -> list[tuple[int, int]]:
    h, w = _shape_bbox(shape)
    return [(dr, dc) for dr in range(cell_size - h + 1) for dc in range(cell_size - w + 1)]


def _sample_occurrence_offsets(shape: tuple[tuple[int, int], ...], cell_size: int, count: int) -> list[tuple[int, int]]:
    choices = _offset_choices(shape, cell_size)
    if len(choices) == 1:
        return [choices[0]] * count
    picks = [random.choice(choices) for _ in range(count)]
    if len(set(picks)) == 1:
        alt = random.choice([choice for choice in choices if choice != picks[0]])
        picks[-1] = alt
    return picks


def _paint_shape(block: np.ndarray, shape: tuple[tuple[int, int], ...], offset: tuple[int, int], color: int) -> None:
    r_off, c_off = offset
    for r, c in shape:
        block[r_off + r, c_off + c] = color


def _sample_layout() -> list[list[int]]:
    values = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    for _ in range(200):
        random.shuffle(values)
        layout = [values[0:3], values[3:6], values[6:9]]
        rows_ok = all(len(set(row)) >= 2 for row in layout)
        cols_ok = all(len({layout[r][c] for r in range(3)}) >= 2 for c in range(3))
        if rows_ok and cols_ok:
            return layout
    raise RuntimeError("failed to build a non-degenerate 3x3 layout")


def _build_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    row_bands, col_bands = _parse_bands(inp)
    dict_r0, dict_r1 = row_bands[-1]
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for c0, c1 in col_bands:
        block = inp[dict_r0 : dict_r1 + 1, c0 : c1 + 1]
        shape, color = _extract_single_shape(block)
        mapping[shape] = color
    for r0, r1 in row_bands[:MAIN_ROWS]:
        for c0, c1 in col_bands:
            block = inp[r0 : r1 + 1, c0 : c1 + 1]
            shape, _ = _extract_single_shape(block, expected_color=GREY)
            out_block = out[r0 : r1 + 1, c0 : c1 + 1]
            out_block[out_block == GREY] = mapping[shape]
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    row_bands, col_bands = _parse_bands(inp)
    dict_r0, dict_r1 = row_bands[-1]
    dict_shapes = []
    dict_colors = []
    main_shapes = []
    main_offsets: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = defaultdict(list)
    dict_offsets: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}

    for c0, c1 in col_bands:
        block = inp[dict_r0 : dict_r1 + 1, c0 : c1 + 1]
        shape, color = _extract_single_shape(block)
        dict_shapes.append(shape)
        dict_colors.append(color)
        coords = np.argwhere(block == color)
        rr0, cc0 = coords.min(axis=0)
        dict_offsets[shape] = (int(rr0), int(cc0))

    if set(dict_colors) != set(PALETTE) or len(set(dict_shapes)) != 3:
        return False

    shape_ids = {}
    for idx, shape in enumerate(dict_shapes):
        shape_ids[shape] = idx

    layout: list[list[int]] = []
    for r0, r1 in row_bands[:MAIN_ROWS]:
        row_ids: list[int] = []
        for c0, c1 in col_bands:
            block = inp[r0 : r1 + 1, c0 : c1 + 1]
            shape, _ = _extract_single_shape(block, expected_color=GREY)
            main_shapes.append(shape)
            row_ids.append(shape_ids[shape])
            coords = np.argwhere(block == GREY)
            rr0, cc0 = coords.min(axis=0)
            main_offsets[shape].append((int(rr0), int(cc0)))
        layout.append(row_ids)

    if Counter(main_shapes) != Counter({shape: 3 for shape in dict_shapes}):
        return False
    if not all(len(set(row)) >= 2 for row in layout):
        return False
    if not all(len({layout[r][c] for r in range(3)}) >= 2 for c in range(3)):
        return False

    areas = [len(shape) for shape in dict_shapes]
    bboxes = [_shape_bbox(shape) for shape in dict_shapes]
    if len(set(areas)) == 3:
        return False
    if len(set(bboxes)) == 3:
        return False

    varied_offset = False
    for shape in dict_shapes:
        occurrences = main_offsets[shape]
        if len(set(occurrences)) >= 2:
            varied_offset = True
        if dict_offsets[shape] not in set(occurrences) or len(set(occurrences)) >= 2:
            varied_offset = True
    if not varied_offset:
        return False

    return True


def _attempt() -> np.ndarray | None:
    cell_size = random.randint(*CELL_SIZE_RANGE)
    grid_h = (MAIN_ROWS + DICT_ROWS) * cell_size + (MAIN_ROWS + DICT_ROWS + 1)
    grid_w = MAIN_COLS * cell_size + (MAIN_COLS + 1)
    inp = np.full((grid_h, grid_w), ORANGE, dtype=int)

    row_bands = [(1 + r * (cell_size + 1), 1 + r * (cell_size + 1) + cell_size - 1) for r in range(MAIN_ROWS + DICT_ROWS)]
    col_bands = [(1 + c * (cell_size + 1), 1 + c * (cell_size + 1) + cell_size - 1) for c in range(MAIN_COLS)]
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    shapes = _choose_shape_triplet(cell_size)
    random.shuffle(shapes)
    color_by_shape = {shape: color for shape, color in zip(shapes, random.sample(list(PALETTE), len(PALETTE)))}

    for shape, (c0, c1) in zip(shapes, col_bands):
        block = inp[row_bands[-1][0] : row_bands[-1][1] + 1, c0 : c1 + 1]
        dict_choices = _offset_choices(shape, cell_size)
        dict_offset = random.choice(dict_choices)
        _paint_shape(block, shape, dict_offset, color_by_shape[shape])

    layout = _sample_layout()
    occurrence_offsets = {shape: _sample_occurrence_offsets(shape, cell_size, 3) for shape in shapes}
    used_index = Counter()
    for grid_r in range(MAIN_ROWS):
        for grid_c in range(MAIN_COLS):
            shape = shapes[layout[grid_r][grid_c]]
            block = inp[row_bands[grid_r][0] : row_bands[grid_r][1] + 1, col_bands[grid_c][0] : col_bands[grid_c][1] + 1]
            offset = occurrence_offsets[shape][used_index[shape]]
            used_index[shape] += 1
            _paint_shape(block, shape, offset, GREY)

    expected = _build_expected_output(inp)
    if _validate_instance(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
