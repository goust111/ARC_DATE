from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Grid = np.ndarray
Box = tuple[int, int, int, int]
Coord = tuple[int, int]

BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5)
MAX_SIDE = 30
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox_from_coords(coords: list[Coord]) -> Box:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_monochrome_components(grid: Grid, bg: int = BACKGROUND) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return components


def _filled_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _valid_rectangles(grid: Grid) -> list[dict]:
    rectangles = []
    for component in _find_monochrome_components(grid, bg=BACKGROUND):
        area = int(component["area"])
        r0, c0, r1, c1 = component["bbox"]
        if area >= 9 and r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3 and _filled_rectangle(component):
            rectangles.append(component)
    return rectangles


def _group_overlapping_intervals(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], list[list[int]]]:
    groups: list[list[int]] = []
    common_los: list[int] = []
    common_his: list[int] = []
    center_sums: list[float] = []
    counts: list[int] = []

    for idx, lo, hi in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        matches = [
            group_idx
            for group_idx in range(len(groups))
            if max(common_los[group_idx], lo) <= min(common_his[group_idx], hi)
        ]
        assert len(matches) <= 1, "each interval should belong to at most one overlap group"
        if matches:
            group_idx = matches[0]
            groups[group_idx].append(idx)
            common_los[group_idx] = max(common_los[group_idx], lo)
            common_his[group_idx] = min(common_his[group_idx], hi)
            center_sums[group_idx] += (lo + hi) / 2.0
            counts[group_idx] += 1
        else:
            groups.append([idx])
            common_los.append(lo)
            common_his.append(hi)
            center_sums.append((lo + hi) / 2.0)
            counts.append(1)

    order = sorted(range(len(groups)), key=lambda group_idx: center_sums[group_idx] / counts[group_idx])
    rect_to_rank: dict[int, int] = {}
    ordered_groups: list[list[int]] = []
    for rank, group_idx in enumerate(order):
        ordered_groups.append(groups[group_idx])
        for rect_idx in groups[group_idx]:
            rect_to_rank[rect_idx] = rank
    return rect_to_rank, ordered_groups


def _appearance_palette(values: Iterable[int], bg: int = BACKGROUND) -> tuple[int, ...]:
    palette: list[int] = []
    for value in values:
        color = int(value)
        if color != bg and color not in palette:
            palette.append(color)
    return tuple(palette)


def _complete_latin_unique(table: Grid, palette: tuple[int, ...], blank: int = BACKGROUND) -> Grid:
    out = np.asarray(table, dtype=int).copy()
    palette_set = set(map(int, palette))
    solutions: list[Grid] = []

    def domain(r: int, c: int) -> list[int]:
        row_used = {int(value) for value in out[r, :] if int(value) != blank}
        col_used = {int(value) for value in out[:, c] if int(value) != blank}
        return [color for color in palette if color not in row_used and color not in col_used]

    def search() -> None:
        if len(solutions) > 1:
            return

        blanks = np.argwhere(out == blank)
        if len(blanks) == 0:
            assert all({int(value) for value in row} == palette_set for row in out), "invalid Latin row"
            assert all({int(value) for value in col} == palette_set for col in out.T), "invalid Latin column"
            solutions.append(out.copy())
            return

        options: list[tuple[int, int, int, list[int]]] = []
        for r_raw, c_raw in blanks:
            r, c = int(r_raw), int(c_raw)
            choices = domain(r, c)
            if not choices:
                return
            options.append((len(choices), r, c, choices))

        options.sort(key=lambda item: (item[0], item[1], item[2]))
        _, r, c, choices = options[0]
        for color in choices:
            out[r, c] = color
            search()
            out[r, c] = blank

    for row in out:
        seen = [int(value) for value in row if int(value) != blank]
        assert len(seen) == len(set(seen)), "rows may not repeat colors"
    for col in out.T:
        seen = [int(value) for value in col if int(value) != blank]
        assert len(seen) == len(set(seen)), "columns may not repeat colors"

    search()
    assert len(solutions) == 1, "expected a uniquely completable Latin table"
    return solutions[0]


def _row_col_boxes(rectangles: list[dict], row_groups: list[list[int]], col_groups: list[list[int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_boxes: list[tuple[int, int]] = []
    for group in row_groups:
        intervals = {(int(rectangles[idx]["bbox"][0]), int(rectangles[idx]["bbox"][2])) for idx in group}
        assert len(intervals) == 1, "each physical row should share one vertical bbox interval"
        row_boxes.append(next(iter(intervals)))

    col_boxes: list[tuple[int, int]] = []
    for group in col_groups:
        intervals = {(int(rectangles[idx]["bbox"][1]), int(rectangles[idx]["bbox"][3])) for idx in group}
        assert len(intervals) == 1, "each physical column should share one horizontal bbox interval"
        col_boxes.append(next(iter(intervals)))

    return row_boxes, col_boxes


def _compressed_table(rectangles: list[dict]) -> tuple[Grid, tuple[int, ...], list[tuple[int, int]], list[tuple[int, int]]]:
    assert rectangles, "expected at least one valid rectangle"
    rect_to_row, row_groups = _group_overlapping_intervals(
        [(idx, int(rect["bbox"][0]), int(rect["bbox"][2])) for idx, rect in enumerate(rectangles)]
    )
    rect_to_col, col_groups = _group_overlapping_intervals(
        [(idx, int(rect["bbox"][1]), int(rect["bbox"][3])) for idx, rect in enumerate(rectangles)]
    )

    palette = _appearance_palette(rect["color"] for rect in rectangles)
    size = len(palette)
    assert size in (4, 5), "the Latin palette must have size 4 or 5"
    assert len(row_groups) == size and len(col_groups) == size, "row/column counts must match the palette size"

    table = np.full((size, size), BACKGROUND, dtype=int)
    for idx, rect in enumerate(rectangles):
        row = rect_to_row[idx]
        col = rect_to_col[idx]
        assert int(table[row, col]) == BACKGROUND, "each lattice cell should contain at most one rectangle"
        table[row, col] = int(rect["color"])

    row_boxes, col_boxes = _row_col_boxes(rectangles, row_groups, col_groups)
    return table, palette, row_boxes, col_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the scattered rectangle lattice to a color table, complete its Latin-square blanks, then paint the missing physical rectangles.

    Input structure:
    - The background is black, and the meaningful objects are monochrome 4-connected rectangles of area at least 9 that exactly fill their bboxes.
    - Those rectangles occupy a hidden KxK lattice: same-row rectangles share one vertical bbox band, same-column rectangles share one horizontal bbox band.
    - Some lattice positions are absent as black gaps, while smaller colored blobs or pixels may appear as distractors in the spaces between rectangles.

    Rule:
    - Ignore every component that is not a filled rectangle of area at least 9.
    - Group the remaining rectangles by overlapping bbox rows and columns to obtain a partial KxK color table.
    - Treat missing table cells as blanks and uniquely complete the table so every row and every column contains the full foreground palette exactly once.
    - Paint each completed blank back into the original grid using the bbox height of its row and the bbox width of its column, leaving distractors unchanged.

    Key insight:
    - The physical grid must first be filtered and compressed to expose an abstract Latin-square constraint; only after solving that abstract table can the missing color be projected back into the original rectangle lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rectangles = _valid_rectangles(g)
    table, palette, row_boxes, col_boxes = _compressed_table(rectangles)
    solved = _complete_latin_unique(table, palette, blank=BACKGROUND)

    out = g.copy()
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            if int(table[row, col]) != BACKGROUND:
                continue
            color = int(solved[row, col])
            r0, r1 = row_boxes[row]
            c0, c1 = col_boxes[col]
            assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == BACKGROUND), "blank rectangle region must be black"
            out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _gen_latin_square(size: int, palette: tuple[int, ...]) -> Grid:
    row_perm = random.sample(range(size), size)
    col_perm = random.sample(range(size), size)
    color_perm = random.sample(list(palette), size)
    square = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            square[r, c] = color_perm[(row_perm[r] + col_perm[c]) % size]
    return square


def _candidate_blank_sets(size: int, count: int) -> list[list[Coord]]:
    candidates: list[list[Coord]] = []

    all_cells = [(r, c) for r in range(size) for c in range(size)]
    if count == 1:
        return [[cell] for cell in all_cells]

    rows = list(range(size))
    cols = list(range(size))

    # Same-row patterns.
    for r in rows:
        for col_choice in _combinations(cols, count):
            candidates.append([(r, c) for c in col_choice])

    # Same-column patterns.
    for c in cols:
        for row_choice in _combinations(rows, count):
            candidates.append([(r, c) for r in row_choice])

    # Distinct-row / distinct-column patterns.
    for row_choice in _combinations(rows, count):
        for col_choice in _permuted_combinations(cols, count):
            candidates.append([(row_choice[i], col_choice[i]) for i in range(count)])

    # L-shapes for 3 blanks.
    if count == 3:
        for r0 in range(size - 1):
            for c0 in range(size - 1):
                block = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
                for missing in range(4):
                    candidates.append([cell for idx, cell in enumerate(block) if idx != missing])

    deduped = []
    seen = set()
    random.shuffle(candidates)
    for cells in candidates:
        key = tuple(sorted(cells))
        if key not in seen:
            seen.add(key)
            deduped.append(list(key))
    return deduped


def _combinations(items: list[int], count: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def search(start: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            out.append(tuple(chosen))
            return
        for idx in range(start, len(items)):
            chosen.append(items[idx])
            search(idx + 1, chosen)
            chosen.pop()

    search(0, [])
    return out


def _permuted_combinations(items: list[int], count: int) -> list[tuple[int, ...]]:
    chosen = _combinations(items, count)
    out: list[tuple[int, ...]] = []

    def permute(prefix: list[int], rest: list[int]) -> None:
        if not rest:
            out.append(tuple(prefix))
            return
        for idx in range(len(rest)):
            permute(prefix + [rest[idx]], rest[:idx] + rest[idx + 1 :])

    for comb in chosen:
        permute([], list(comb))
    return out


def _sample_blank_positions(full_square: Grid) -> list[Coord]:
    size = int(full_square.shape[0])
    counts = [1, 2, 3]
    random.shuffle(counts)

    for count in counts:
        candidates = _candidate_blank_sets(size, count)
        random.shuffle(candidates)
        for cells in candidates:
            partial = full_square.copy()
            for r, c in cells:
                partial[r, c] = BACKGROUND
            try:
                solved = _complete_latin_unique(partial, _appearance_palette(full_square.ravel()))
            except AssertionError:
                continue
            if np.array_equal(solved, full_square):
                return cells
    raise AssertionError("failed to find uniquely completable blanks")


def _sample_axis_sizes(size: int) -> tuple[list[int], list[int], int, int, int, int, int]:
    for _ in range(200):
        if size == 4:
            row_heights = [random.randint(3, 5) for _ in range(size)]
            col_widths = [random.randint(3, 5) for _ in range(size)]
        else:
            row_heights = [random.randint(3, 4) for _ in range(size)]
            col_widths = [random.randint(3, 4) for _ in range(size)]

        gap = 2
        base_h = sum(row_heights) + gap * (size - 1)
        base_w = sum(col_widths) + gap * (size - 1)
        if base_h > MAX_SIDE or base_w > MAX_SIDE:
            continue

        extra_h = MAX_SIDE - base_h
        extra_w = MAX_SIDE - base_w
        top = random.randint(0, min(3, extra_h))
        bottom = random.randint(0, min(3, extra_h - top))
        left = random.randint(0, min(3, extra_w))
        right = random.randint(0, min(3, extra_w - left))
        total_height = top + base_h + bottom
        total_width = left + base_w + right
        return row_heights, col_widths, gap, top, left, total_height, total_width
    raise AssertionError("failed to sample rectangle geometry")


def _axis_boxes(lengths: list[int], gap: int, start: int) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int]] = []
    pos = start
    for idx, length in enumerate(lengths):
        boxes.append((pos, pos + length - 1))
        pos += length
        if idx < len(lengths) - 1:
            pos += gap
    return boxes


def _shape_orientations(coords: list[Coord]) -> list[list[Coord]]:
    out: list[list[Coord]] = []
    seen = set()

    def normalize(points: list[Coord]) -> list[Coord]:
        min_r = min(r for r, _ in points)
        min_c = min(c for _, c in points)
        return sorted((r - min_r, c - min_c) for r, c in points)

    cur = coords
    for _ in range(4):
        norm = normalize(cur)
        key = tuple(norm)
        if key not in seen:
            seen.add(key)
            out.append(norm)

        flipped = normalize([(r, -c) for r, c in cur])
        key = tuple(flipped)
        if key not in seen:
            seen.add(key)
            out.append(flipped)

        cur = [(-c, r) for r, c in cur]
    return out


def _place_distractor(
    grid: Grid,
    expected: Grid,
    corridor_mask: np.ndarray,
    occupied: set[Coord],
    palette: tuple[int, ...],
    shapes: list[list[Coord]],
) -> bool:
    height, width = grid.shape
    for _ in range(240):
        shape = random.choice(shapes)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        r0 = random.randint(0, height - max_r - 1)
        c0 = random.randint(0, width - max_c - 1)
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]

        if any(not corridor_mask[r, c] for r, c in cells):
            continue
        if any((r, c) in occupied for r, c in cells):
            continue

        cell_set = set(cells)
        if any((nr, nc) in occupied for r, c in cells for nr, nc in _neighbors4(r, c) if (nr, nc) not in cell_set):
            continue

        touched_colors = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width:
                    touched = int(expected[nr, nc])
                    if touched != BACKGROUND:
                        touched_colors.add(touched)

        available = [color for color in palette if color not in touched_colors]
        if not available:
            continue

        color = random.choice(available)
        for r, c in cells:
            grid[r, c] = color
            expected[r, c] = color
            occupied.add((r, c))
        return True

    return False


def _render_rectangles(table: Grid, row_boxes: list[tuple[int, int]], col_boxes: list[tuple[int, int]], shape: tuple[int, int]) -> Grid:
    out = np.full(shape, BACKGROUND, dtype=int)
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            color = int(table[row, col])
            if color == BACKGROUND:
                continue
            r0, r1 = row_boxes[row]
            c0, c1 = col_boxes[col]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _validate_instance(inp: Grid, expected: Grid, full_square: Grid, blank_cells: list[Coord]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > MAX_SIDE or max(expected.shape) > MAX_SIDE:
        return False

    try:
        components = _find_monochrome_components(inp)
        rectangles = _valid_rectangles(inp)
        table, palette, row_boxes, col_boxes = _compressed_table(rectangles)
        solved = _complete_latin_unique(table, palette)
    except AssertionError:
        return False

    rectangle_keys = {
        (int(rect["color"]), tuple(map(int, rect["bbox"])))
        for rect in rectangles
    }
    for component in components:
        key = (int(component["color"]), tuple(map(int, component["bbox"])))
        if key not in rectangle_keys and int(component["area"]) >= 9:
            return False

    partial = full_square.copy()
    for r, c in blank_cells:
        partial[r, c] = BACKGROUND

    if not np.array_equal(table, partial):
        return False
    if not np.array_equal(solved, full_square):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    for row, col in blank_cells:
        r0, r1 = row_boxes[row]
        c0, c1 = col_boxes[col]
        if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != BACKGROUND):
            return False
        if not np.all(expected[r0 : r1 + 1, c0 : c1 + 1] == int(full_square[row, col])):
            return False

    return True


def _gen_attempt() -> Grid | None:
    size = random.choice((4, 5))
    palette = tuple(random.sample(PALETTE, size))
    full_square = _gen_latin_square(size, palette)
    blank_cells = _sample_blank_positions(full_square)

    row_heights, col_widths, gap, top, left, height, width = _sample_axis_sizes(size)
    row_boxes = _axis_boxes(row_heights, gap, top)
    col_boxes = _axis_boxes(col_widths, gap, left)

    partial = full_square.copy()
    for r, c in blank_cells:
        partial[r, c] = BACKGROUND

    expected = _render_rectangles(full_square, row_boxes, col_boxes, (height, width))
    inp = _render_rectangles(partial, row_boxes, col_boxes, (height, width))

    lattice_r0 = row_boxes[0][0]
    lattice_r1 = row_boxes[-1][1]
    lattice_c0 = col_boxes[0][0]
    lattice_c1 = col_boxes[-1][1]
    corridor_mask = np.zeros_like(inp, dtype=bool)
    corridor_mask[lattice_r0 : lattice_r1 + 1, lattice_c0 : lattice_c1 + 1] = True
    corridor_mask[expected != BACKGROUND] = False

    occupied: set[Coord] = set()
    filled_shapes = _shape_orientations([(0, 0), (0, 1), (1, 0), (1, 1)])
    filled_shapes += _shape_orientations([(0, 0), (0, 1), (0, 2), (0, 3)])
    irregular_shapes = _shape_orientations([(0, 0), (1, 0), (1, 1)])
    irregular_shapes += _shape_orientations([(0, 0), (0, 1), (1, 1), (2, 1)])
    single_shapes = [[(0, 0)]]

    plan = [
        filled_shapes,
        irregular_shapes,
        single_shapes,
    ]
    extras = random.randint(0, 2)
    for _ in range(extras):
        plan.append(random.choice((filled_shapes, irregular_shapes, single_shapes)))

    random.shuffle(plan)
    for shapes in plan:
        if not _place_distractor(inp, expected, corridor_mask, occupied, palette, shapes):
            return None

    if not _validate_instance(inp, expected, full_square, blank_cells):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-latin instance")
