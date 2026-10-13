import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATION_ATTEMPTS = 400
MAX_INTERSECTION_COUNT = 4


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _line_cells(segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for start, end in segments:
        cells.update(_segment_cells(start, end))
    return cells


def _degree_in_cells(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIR4)


def _analyze_lines(
    lines: list[dict[str, object]]
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int]], list[int]] | None:
    cell_sets: list[set[tuple[int, int]]] = []
    occupancy: dict[tuple[int, int], list[int]] = {}

    for lid, line in enumerate(lines):
        cells = _line_cells(line["segments"])  # type: ignore[index]
        endpoint_count = 0
        for cell in cells:
            degree = _degree_in_cells(cell, cells)
            if degree == 1:
                endpoint_count += 1
            elif degree != 2:
                return None
        if endpoint_count != 2:
            return None

        cell_sets.append(cells)
        for cell in cells:
            occupancy.setdefault(cell, []).append(lid)

    yellow_cells: list[tuple[int, int]] = []
    yellow_by_line = [set() for _ in lines]
    for cell, lids in occupancy.items():
        if len(lids) == 1:
            continue
        if len(lids) != 2:
            return None

        yellow_cells.append(cell)
        for lid in lids:
            if _degree_in_cells(cell, cell_sets[lid]) != 2:
                return None
            yellow_by_line[lid].add(cell)

    yellow_set = set(yellow_cells)
    for r, c in yellow_set:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in yellow_set:
                return None

    counts = [len(points) for points in yellow_by_line]
    if not yellow_cells:
        return None
    if any(count == 0 for count in counts):
        return None
    if max(counts) > MAX_INTERSECTION_COUNT:
        return None
    if len(set(counts)) < 2:
        return None

    return cell_sets, yellow_cells, counts


def _make_hvh_line(height: int, width: int, row_a: int, row_b: int, col_mid: int) -> dict[str, object]:
    start_side = random.choice(("left", "right"))
    end_side = random.choice(("left", "right"))
    start_col = 0 if start_side == "left" else width - 1
    end_col = 0 if end_side == "left" else width - 1
    return {
        "segments": [
            ((row_a, start_col), (row_a, col_mid)),
            ((row_a, col_mid), (row_b, col_mid)),
            ((row_b, col_mid), (row_b, end_col)),
        ]
    }


def _make_vhv_line(height: int, width: int, row_mid: int, col_a: int, col_b: int) -> dict[str, object]:
    start_side = random.choice(("top", "bottom"))
    end_side = random.choice(("top", "bottom"))
    start_row = 0 if start_side == "top" else height - 1
    end_row = 0 if end_side == "top" else height - 1
    return {
        "segments": [
            ((start_row, col_a), (row_mid, col_a)),
            ((row_mid, col_a), (row_mid, col_b)),
            ((row_mid, col_b), (end_row, col_b)),
        ]
    }


def _render_expected(
    height: int,
    width: int,
    cell_sets: list[set[tuple[int, int]]],
    yellow_cells: list[tuple[int, int]],
    counts: list[int],
    line_colors: list[int],
    band_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height + 1, width), dtype=int)
    for lid, cells in enumerate(cell_sets):
        for r, c in cells:
            grid[r, c] = line_colors[lid]
    for r, c in yellow_cells:
        grid[r, c] = YELLOW

    band_span = 2 * len(band_colors) - 1
    start_col = random.randint(0, width - band_span)
    for idx, color in enumerate(band_colors):
        grid[height, start_col + 2 * idx] = color

    expected = grid.copy()
    yellow_set = set(yellow_cells)
    for lid, cells in enumerate(cell_sets):
        new_color = band_colors[counts[lid] - 1]
        for r, c in cells:
            if (r, c) not in yellow_set:
                expected[r, c] = new_color
    return grid, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orthogonal polyline by looking up its yellow-crossing count in the bottom color band.

    Input structure:
    - Black is the background.
    - Above the bottom row, several one-pixel-thick colored polylines run horizontally and vertically.
    - Each polyline uses its own color; whenever exactly two polylines share one cell, that cell is yellow.
    - The bottom row is a left-to-right color band of isolated non-black pixels separated by black gaps.

    Rule:
    - Recover each polyline as its own colored pixels plus the yellow crossings whose four neighbors split into two same-color pairs.
    - Count how many yellow crossings belong to each polyline.
    - Use that count K as a 1-based index into the bottom color band.
    - Recolor every non-yellow pixel of that polyline to the K-th band color.
    - Leave yellow crossings and the bottom color band unchanged.

    Key insight:
    - Yellow is not a separate object; it is a shared crossing pixel whose two incident colors reveal which two polylines pass through it, and the number of such shared pixels selects the replacement color from the bottom lookup strip.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 3, "expected a drawable area plus a bottom color band"

    draw = g[:-1]
    band_row = g[-1]
    band_colors = [int(value) for value in band_row if int(value) != BLACK]
    assert band_colors, "expected at least one band color on the bottom row"

    line_colors = sorted(int(v) for v in np.unique(draw) if int(v) not in (BLACK, YELLOW))
    assert line_colors, "expected at least one polyline color"

    yellow_positions = np.argwhere(draw == YELLOW)
    crossings_by_color = {color: set() for color in line_colors}

    for r, c in yellow_positions:
        neighbor_colors = []
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < h - 1 and 0 <= cc < w, "yellow crossings must stay inside the drawing area"
            value = int(draw[rr, cc])
            assert value != BLACK and value != YELLOW, "yellow crossings must connect to colored ports"
            neighbor_colors.append(value)

        counts = Counter(neighbor_colors)
        assert len(counts) == 2 and set(counts.values()) == {2}, "each yellow must join exactly two polylines"
        for color in counts:
            crossings_by_color[color].add((int(r), int(c)))

    out = g.copy()
    for color in line_colors:
        crossing_count = len(crossings_by_color[color])
        assert 1 <= crossing_count <= len(band_colors), "every polyline needs a valid 1-based band lookup"
        out[:-1][draw == color] = band_colors[crossing_count - 1]

    return out


def _gen_attempt() -> np.ndarray | None:
    line_count = random.randint(3, 4)
    hvh_count = random.randint(1, line_count - 1)
    vhv_count = line_count - hvh_count

    needed_rows = 2 * hvh_count + vhv_count
    needed_cols = hvh_count + 2 * vhv_count

    draw_h = max(12, 3 * needed_rows + random.randint(0, 2))
    width = max(12, 3 * needed_cols + random.randint(0, 2))
    if draw_h > 29 or width > 30:
        return None

    row_slots = list(range(1, draw_h - 1, 3))
    col_slots = list(range(1, width - 1, 3))
    if len(row_slots) < needed_rows or len(col_slots) < needed_cols:
        return None

    rows = random.sample(row_slots, needed_rows)
    cols = random.sample(col_slots, needed_cols)
    random.shuffle(rows)
    random.shuffle(cols)

    lines: list[dict[str, object]] = []
    for _ in range(hvh_count):
        row_a = rows.pop()
        row_b = rows.pop()
        col_mid = cols.pop()
        lines.append(_make_hvh_line(draw_h, width, row_a, row_b, col_mid))
    for _ in range(vhv_count):
        row_mid = rows.pop()
        col_a = cols.pop()
        col_b = cols.pop()
        lines.append(_make_vhv_line(draw_h, width, row_mid, col_a, col_b))
    random.shuffle(lines)

    analyzed = _analyze_lines(lines)
    if analyzed is None:
        return None
    cell_sets, yellow_cells, counts = analyzed

    line_colors = random.sample(list(PALETTE), line_count)
    remaining = [color for color in PALETTE if color not in line_colors]
    band_colors = random.sample(remaining, max(counts))

    grid, expected = _render_expected(
        draw_h,
        width,
        cell_sets,
        yellow_cells,
        counts,
        line_colors,
        band_colors,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-polyline task")
