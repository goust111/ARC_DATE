from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
CELL_SIZE = 3
STEP = CELL_SIZE + 1


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _is_queen_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (dr == 0 or dc == 0 or abs(dr) == abs(dc)) and (dr != 0 or dc != 0)


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    assert _is_queen_aligned(a, b), "target cell must lie on an 8-way ray"
    return _sign(b[0] - a[0]), _sign(b[1] - a[1])


def _line_cells_between(
    a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = _direction(a, b)
    out = []
    r, c = a[0] + dr, a[1] + dc
    while (r, c) != b:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _cell_intervals(line_positions: list[int], total: int) -> list[tuple[int, int]] | None:
    if not line_positions:
        return None
    intervals: list[tuple[int, int]] = []
    prev = -1
    for pos in sorted(line_positions) + [int(total)]:
        start = prev + 1
        end = pos - 1
        if start <= end:
            if end - start + 1 != CELL_SIZE:
                return None
            intervals.append((start, end))
        prev = pos
    if not intervals:
        return None
    return intervals


def _find_separator_grid(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    candidates = []
    for color in sorted(map(int, np.unique(g))):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        row_intervals = _cell_intervals(rows, h)
        col_intervals = _cell_intervals(cols, w)
        if row_intervals is None or col_intervals is None:
            continue

        line_mask = np.zeros_like(g, dtype=bool)
        line_mask[rows, :] = True
        line_mask[:, cols] = True
        if np.any((g == color) & ~line_mask):
            continue
        candidates.append((color, row_intervals, col_intervals))

    assert len(candidates) == 1, "expected one regular 3x3-cell separator grid"
    return candidates[0]


def _inner_background(
    g: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> int:
    vals: list[int] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            vals.extend(map(int, g[r0 : r1 + 1, c0 : c1 + 1].ravel()))
    assert vals, "expected at least one cell interior"
    return Counter(vals).most_common(1)[0][0]


def _parse_grid(g: np.ndarray) -> tuple[list[dict], list[int]]:
    sep_color, row_intervals, col_intervals = _find_separator_grid(g)
    bg = _inner_background(g, row_intervals, col_intervals)
    assert bg != sep_color, "cell background must differ from separator color"

    markers: list[dict] = []
    cell_to_index: dict[tuple[int, int], int] = {}
    seen_colors: set[int] = set()

    for cr, (r0, r1) in enumerate(row_intervals):
        for cc, (c0, c1) in enumerate(col_intervals):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere(patch != bg)
            assert coords.shape[0] <= 1, "each cell may contain at most one marker"
            if coords.shape[0] == 0:
                continue

            pr, pc = map(int, coords[0])
            offset = (pr - 1, pc - 1)
            assert offset != (0, 0), "marker cannot sit in the cell center"
            assert offset[0] in (-1, 0, 1) and offset[1] in (-1, 0, 1)

            color = int(patch[pr, pc])
            assert color not in (bg, sep_color), "marker color must be visible and non-separator"
            assert color not in seen_colors, "marker colors must be unique"
            seen_colors.add(color)

            index = len(markers)
            cell = (cr, cc)
            cell_to_index[cell] = index
            markers.append(
                {
                    "cell": cell,
                    "offset": offset,
                    "color": color,
                    "abs": (r0 + pr, c0 + pc),
                }
            )

    assert len(markers) >= 2, "expected at least two markers in the pointer ring"

    cell_rows = len(row_intervals)
    cell_cols = len(col_intervals)
    targets: list[int] = []
    for marker in markers:
        dr, dc = marker["offset"]
        cr, cc = marker["cell"]
        r, c = cr + dr, cc + dc
        target = None
        while 0 <= r < cell_rows and 0 <= c < cell_cols:
            if (r, c) in cell_to_index:
                target = cell_to_index[(r, c)]
                break
            r += dr
            c += dc
        assert target is not None, "every marker must point to a visible target marker"
        targets.append(target)

    indegree = [0] * len(markers)
    for target in targets:
        indegree[target] += 1
    assert all(deg == 1 for deg in indegree), "pointer graph must have one predecessor per marker"

    seen: set[int] = set()
    cur = 0
    for _ in range(len(markers)):
        assert cur not in seen, "pointer graph closed before using all markers"
        seen.add(cur)
        cur = targets[cur]
    assert cur == 0 and len(seen) == len(markers), "pointers must form one closed ring"
    return markers, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle the singleton marker colors forward along the hidden 8-way pointer ring.

    Input structure:
    - A one-pixel separator grid divides the canvas into a regular table of 3 by 3 cell interiors.
    - Some cells contain exactly one colored marker pixel, never at the center of its own 3 by 3 cell;
      all other cells are empty background.
    - Marker colors are all distinct, and the marker offset from its cell center is one of the eight
      compass directions.

    Rule:
    - Interpret each marker's offset inside its cell as an 8-way direction in cell-table coordinates.
    - From that marker's cell, scan along that direction until the first occupied cell; that marker
      points to the marker in that first occupied cell.
    - The pointers are guaranteed to form a single directed cycle. Keep all marker positions and
      offsets fixed, but recolor every target marker with the original color of the marker pointing
      to it.

    Key insight:
    - The visible pixels are not arrows; their off-center subcell positions encode an invisible
      pointer cycle, and the color cycle follows those pointers rather than spatial order or color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers, targets = _parse_grid(g)
    out = g.copy()
    for src, dst in enumerate(targets):
        r, c = markers[dst]["abs"]
        out[r, c] = markers[src]["color"]
    return out


# --------------- generator ---------------

TEMPLATES: dict[int, tuple[tuple[int, int], ...]] = {
    4: ((0, 0), (0, 3), (1, 4), (1, 0)),
    5: ((0, 1), (0, 3), (1, 4), (3, 4), (3, 1)),
    6: ((0, 1), (0, 3), (1, 4), (3, 4), (4, 3), (4, 1)),
    7: ((0, 1), (0, 4), (1, 5), (3, 5), (4, 4), (4, 2), (4, 1)),
    8: ((0, 1), (0, 4), (1, 5), (4, 5), (5, 4), (5, 1), (4, 0), (1, 0)),
}


def _bounds(points: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in points) + 1, max(c for _r, c in points) + 1


def _rotate_point(r: int, c: int, h: int, w: int, turns: int) -> tuple[int, int]:
    if turns == 0:
        return r, c
    if turns == 1:
        return c, h - 1 - r
    if turns == 2:
        return h - 1 - r, w - 1 - c
    return w - 1 - c, r


def _transform_points(
    points: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    h, w = _bounds(points)
    transformed = list(points)
    if random.choice([False, True]):
        transformed = [(r, w - 1 - c) for r, c in transformed]

    turns = random.randrange(4)
    rotated = [_rotate_point(r, c, h, w, turns) for r, c in transformed]
    if turns % 2:
        h, w = w, h

    min_r = min(r for r, _c in rotated)
    min_c = min(c for _r, c in rotated)
    normalized = tuple((r - min_r, c - min_c) for r, c in rotated)
    h, w = _bounds(normalized)
    return normalized, h, w


def _clear_pointer_cycle(points: tuple[tuple[int, int], ...]) -> bool:
    occupied = set(points)
    has_cardinal = False
    has_diagonal = False
    has_jump = False

    for i, cell in enumerate(points):
        target = points[(i + 1) % len(points)]
        if not _is_queen_aligned(cell, target):
            return False

        dr, dc = _direction(cell, target)
        has_cardinal |= dr == 0 or dc == 0
        has_diagonal |= dr != 0 and dc != 0
        has_jump |= max(abs(target[0] - cell[0]), abs(target[1] - cell[1])) > 1

        for between in _line_cells_between(cell, target):
            if between in occupied:
                return False

    return has_cardinal and has_diagonal and has_jump


def _draw_separator_grid(cell_rows: int, cell_cols: int, sep_color: int) -> np.ndarray:
    h = cell_rows * STEP + 1
    w = cell_cols * STEP + 1
    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(0, h, STEP):
        grid[r, :] = sep_color
    for c in range(0, w, STEP):
        grid[:, c] = sep_color
    return grid


def _marker_abs(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    cr, cc = cell
    dr, dc = direction
    return cr * STEP + 2 + dr, cc * STEP + 2 + dc


def _render_case(
    cell_rows: int,
    cell_cols: int,
    cycle_cells: tuple[tuple[int, int], ...],
    colors: list[int],
    sep_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = _draw_separator_grid(cell_rows, cell_cols, sep_color)
    out = inp.copy()
    abs_positions: list[tuple[int, int]] = []

    for i, cell in enumerate(cycle_cells):
        target = cycle_cells[(i + 1) % len(cycle_cells)]
        direction = _direction(cell, target)
        r, c = _marker_abs(cell, direction)
        inp[r, c] = int(colors[i])
        out[r, c] = int(colors[i])
        abs_positions.append((r, c))

    for i, color in enumerate(colors):
        target_pos = abs_positions[(i + 1) % len(abs_positions)]
        out[target_pos] = int(color)
    return inp, out


def _make_cycle_cells() -> tuple[int, int, tuple[tuple[int, int], ...]]:
    count = random.randint(4, 8)
    base = TEMPLATES[count]
    points, h, w = _transform_points(base)
    assert _clear_pointer_cycle(points)

    cell_rows = random.randint(max(3, h), 7)
    cell_cols = random.randint(max(3, w), 7)
    off_r = random.randint(0, cell_rows - h)
    off_c = random.randint(0, cell_cols - w)
    placed = tuple((r + off_r, c + off_c) for r, c in points)
    assert _clear_pointer_cycle(placed)
    return cell_rows, cell_cols, placed


def _validate_generated(inp: np.ndarray, expected: np.ndarray, cycle_cells: tuple[tuple[int, int], ...]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if expected.ndim != 2 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if not _clear_pointer_cycle(cycle_cells):
        return False

    try:
        markers, targets = _parse_grid(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    parsed_cells = [marker["cell"] for marker in markers]
    parsed_target_cells = [parsed_cells[target] for target in targets]
    planned_targets = [cycle_cells[(i + 1) % len(cycle_cells)] for i in range(len(cycle_cells))]
    if set(parsed_cells) != set(cycle_cells):
        return False
    if set(zip(parsed_cells, parsed_target_cells)) != set(zip(cycle_cells, planned_targets)):
        return False

    directions = [_direction(cell, target) for cell, target in zip(cycle_cells, planned_targets)]
    has_cardinal = any(dr == 0 or dc == 0 for dr, dc in directions)
    has_diagonal = any(dr != 0 and dc != 0 for dr, dc in directions)
    has_jump = any(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
        for a, b in zip(cycle_cells, planned_targets)
    )
    return has_cardinal and has_diagonal and has_jump


def generate_input() -> np.ndarray:
    for _ in range(300):
        cell_rows, cell_cols, cycle_cells = _make_cycle_cells()
        sep_color = random.choice(COLORS)
        marker_palette = [color for color in COLORS if color != sep_color]
        colors = random.sample(marker_palette, len(cycle_cells))
        random.shuffle(colors)

        inp, expected = _render_case(cell_rows, cell_cols, cycle_cells, colors, sep_color)
        if _validate_generated(inp, expected, cycle_cells):
            return inp
    raise RuntimeError("failed to generate a valid pointer-cycle grid")
