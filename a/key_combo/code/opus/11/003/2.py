import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
DIVIDER = 5
PROTOTYPE_COLOR = 3
MARKER_COLORS = (1, 2, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_dividers(grid: np.ndarray):
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == DIVIDER)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(rows) == 2 and len(cols) == 2, "expected a 3x3 divider grid"
    assert rows[0] + 1 < rows[1] and cols[0] + 1 < cols[1], "divider cells must be non-empty"
    return rows, cols


def _cell_spans(dividers: list[int], limit: int):
    starts = [0, dividers[0] + 1, dividers[1] + 1]
    ends = [dividers[0], dividers[1], limit]
    return list(zip(starts, ends))


def _cell_bounds(row_spans, col_spans, cell_r: int, cell_c: int):
    r0, r1 = row_spans[cell_r]
    c0, c1 = col_spans[cell_c]
    return r0, r1, c0, c1


def _extract_prototype(grid: np.ndarray, row_spans, col_spans):
    r0, r1, c0, c1 = _cell_bounds(row_spans, col_spans, 0, 0)
    patch = grid[r0:r1, c0:c1]
    coords = np.argwhere(patch == PROTOTYPE_COLOR)
    assert 3 <= len(coords) <= 5, "prototype must contain 3-5 pixels"
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    bbox_h = int(max_r - min_r + 1)
    bbox_w = int(max_c - min_c + 1)
    assert bbox_h >= 2 and bbox_w >= 2, "prototype bbox must be at least 2x2"
    rel = {(int(r - min_r), int(c - min_c)) for r, c in coords}
    assert _is_connected(rel), "prototype must be 4-connected"
    return rel, bbox_h, bbox_w


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _cell_marker_info(grid: np.ndarray, row_spans, col_spans, cell_r: int, cell_c: int):
    r0, r1, c0, c1 = _cell_bounds(row_spans, col_spans, cell_r, cell_c)
    patch = grid[r0:r1, c0:c1]
    coords = np.argwhere((patch != BG) & (patch != DIVIDER))
    if len(coords) == 0:
        return 0, None
    colors = {int(patch[r, c]) for r, c in coords}
    assert len(colors) == 1, "all markers inside one cell must have the same color"
    color = next(iter(colors))
    assert color in MARKER_COLORS, "marker color must be blue, red, or yellow"
    return int(len(coords)), color


def _copy_offsets(kind: str, count: int, bbox_h: int, bbox_w: int):
    if kind == "same_row":
        return [(k * bbox_h, 0) for k in range(count)]
    if kind == "same_col":
        return [(0, k * bbox_w) for k in range(count)]
    return [(k, k) for k in range(count)]


def _render_stamps(
    out: np.ndarray,
    row_spans,
    col_spans,
    cell_r: int,
    cell_c: int,
    rel_shape: set[tuple[int, int]],
    bbox_h: int,
    bbox_w: int,
    color: int,
    count: int,
):
    r0, r1, c0, c1 = _cell_bounds(row_spans, col_spans, cell_r, cell_c)
    out[r0:r1, c0:c1] = BG
    if count == 0:
        return

    if cell_r == 0:
        kind = "same_row"
    elif cell_c == 0:
        kind = "same_col"
    else:
        kind = "diag"

    for off_r, off_c in _copy_offsets(kind, count, bbox_h, bbox_w):
        for dr, dc in rel_shape:
            rr = r0 + off_r + dr
            cc = c0 + off_c + dc
            if r0 <= rr < r1 and c0 <= cc < c1:
                out[rr, cc] = color


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rows, cols = _find_dividers(g)
    row_spans = _cell_spans(rows, g.shape[0])
    col_spans = _cell_spans(cols, g.shape[1])
    rel_shape, bbox_h, bbox_w = _extract_prototype(g, row_spans, col_spans)

    out = g.copy()
    for cell_r in range(3):
        for cell_c in range(3):
            if cell_r == 0 and cell_c == 0:
                continue
            count, color = _cell_marker_info(g, row_spans, col_spans, cell_r, cell_c)
            _render_stamps(out, row_spans, col_spans, cell_r, cell_c, rel_shape, bbox_h, bbox_w, color or BG, count)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marked divider-grid cell with repeated colored copies of the top-left prototype sprite.

    Input structure:
    - A black background grid is split into a 3x3 layout by two full grey horizontal lines and two full grey vertical lines.
    - The top-left cell contains one irregular green 4-connected prototype sprite with 3-5 pixels.
    - Other cells are either empty or contain 1-3 isolated single-pixel markers of one shared color chosen from blue, red, or yellow.

    Rule:
    - Use the grey lines to recover the nine cell interiors, and normalize the green prototype by its own bounding box.
    - In each non-prototype cell, count marker pixels N and read their shared color C; empty cells stay empty.
    - Stamp the prototype N times in color C starting from the cell's inner top-left corner: cells in the prototype's row stack vertically, cells in its column stack horizontally, and all remaining cells shift one step down-right per copy. Pixels that fall outside the cell are clipped.

    Key insight:
    - The divider grid is both the counting boundary and the coordinate frame that encodes stamping direction through relative cell position to the top-left prototype cell.
    """
    return _transform(grid)


def _random_connected_shape(max_h: int, max_w: int, rng: random.Random):
    for _ in range(200):
        bbox_h = rng.randint(2, max_h)
        bbox_w = rng.randint(2, max_w)
        area = bbox_h * bbox_w
        if area < 3:
            continue
        target = rng.randint(3, min(5, area - 1 if area > 3 else 3))
        cells = {(rng.randrange(bbox_h), rng.randrange(bbox_w))}
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) != target or not _is_connected(cells):
            continue
        if min(rows) != 0 or max(rows) != bbox_h - 1 or min(cols) != 0 or max(cols) != bbox_w - 1:
            continue
        if len(cells) == area:
            continue
        return cells, bbox_h, bbox_w
    raise RuntimeError("failed to build prototype shape")


def _non_adjacent_marker_positions(height: int, width: int, count: int, rng: random.Random):
    points = [(r, c) for r in range(height) for c in range(width)]
    choices = []
    for combo in combinations(points, count):
        ok = True
        for i in range(count):
            for j in range(i + 1, count):
                if max(abs(combo[i][0] - combo[j][0]), abs(combo[i][1] - combo[j][1])) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            choices.append(combo)
    assert choices, "every cell must support the requested marker count"
    return list(rng.choice(choices))


def _place_dividers(grid: np.ndarray, row_sizes: list[int], col_sizes: list[int]):
    row_divs = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    col_divs = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    for r in row_divs:
        grid[r, :] = DIVIDER
    for c in col_divs:
        grid[:, c] = DIVIDER
    return row_divs, col_divs


def _branch_of(cell: tuple[int, int]):
    if cell[0] == 0:
        return "same_row"
    if cell[1] == 0:
        return "same_col"
    return "diag"


def _expected_cell_patch(shape, bbox_h, bbox_w, cell_h, cell_w, branch, color, count):
    patch = np.zeros((cell_h, cell_w), dtype=int)
    for off_r, off_c in _copy_offsets(branch, count, bbox_h, bbox_w):
        for dr, dc in shape:
            rr = off_r + dr
            cc = off_c + dc
            if 0 <= rr < cell_h and 0 <= cc < cell_w:
                patch[rr, cc] = color
    return patch


def _validate_semantics(grid: np.ndarray, out: np.ndarray):
    rows, cols = _find_dividers(grid)
    row_spans = _cell_spans(rows, grid.shape[0])
    col_spans = _cell_spans(cols, grid.shape[1])
    shape, bbox_h, bbox_w = _extract_prototype(grid, row_spans, col_spans)

    assert np.array_equal(out[rows[0]], grid[rows[0]])
    assert np.array_equal(out[rows[1]], grid[rows[1]])
    assert np.array_equal(out[:, cols[0]], grid[:, cols[0]])
    assert np.array_equal(out[:, cols[1]], grid[:, cols[1]])

    proto_r0, proto_r1, proto_c0, proto_c1 = _cell_bounds(row_spans, col_spans, 0, 0)
    assert np.array_equal(out[proto_r0:proto_r1, proto_c0:proto_c1], grid[proto_r0:proto_r1, proto_c0:proto_c1])

    branch_seen = set()
    marker_cells = 0
    empty_cells = 0
    clipped_cells = 0
    distinct_nonempty = set()
    count_one_seen = False
    for cell_r in range(3):
        for cell_c in range(3):
            if cell_r == 0 and cell_c == 0:
                continue
            r0, r1, c0, c1 = _cell_bounds(row_spans, col_spans, cell_r, cell_c)
            count, color = _cell_marker_info(grid, row_spans, col_spans, cell_r, cell_c)
            branch = _branch_of((cell_r, cell_c))
            expected = _expected_cell_patch(shape, bbox_h, bbox_w, r1 - r0, c1 - c0, branch, color or BG, count)
            actual = out[r0:r1, c0:c1]
            assert np.array_equal(actual, expected), "output patch does not match semantic rule"
            if count == 0:
                empty_cells += 1
                continue
            marker_cells += 1
            branch_seen.add(branch)
            count_one_seen |= count == 1
            distinct_nonempty.add(tuple(tuple(int(v) for v in row) for row in actual))
            if int(np.count_nonzero(actual)) < count * len(shape):
                clipped_cells += 1

    assert marker_cells >= 4, "need enough marked cells to show the rule"
    assert empty_cells >= 1, "need at least one empty cell"
    assert branch_seen == {"same_row", "same_col", "diag"}, "all direction branches must appear"
    assert count_one_seen, "need a count-1 example"
    assert clipped_cells >= 1, "need at least one clipping case"
    assert len(distinct_nonempty) >= 3, "non-empty output patches must not collapse to one pattern"


def _gen_attempt(rng: random.Random):
    row_sizes = [rng.randint(3, 4) for _ in range(3)]
    col_sizes = [rng.randint(3, 4) for _ in range(3)]
    height = sum(row_sizes) + 2
    width = sum(col_sizes) + 2
    grid = np.zeros((height, width), dtype=int)
    row_divs, col_divs = _place_dividers(grid, row_sizes, col_sizes)
    row_spans = _cell_spans(row_divs, height)
    col_spans = _cell_spans(col_divs, width)

    shape, bbox_h, bbox_w = _random_connected_shape(row_sizes[0], col_sizes[0], rng)
    proto_r0, proto_r1, proto_c0, proto_c1 = _cell_bounds(row_spans, col_spans, 0, 0)
    proto_off_r = rng.randint(0, (proto_r1 - proto_r0) - bbox_h)
    proto_off_c = rng.randint(0, (proto_c1 - proto_c0) - bbox_w)
    for dr, dc in shape:
        grid[proto_r0 + proto_off_r + dr, proto_c0 + proto_off_c + dc] = PROTOTYPE_COLOR

    same_row_cells = [(0, 1), (0, 2)]
    same_col_cells = [(1, 0), (2, 0)]
    diag_cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
    chosen = {
        rng.choice(same_row_cells): (3, None),
        rng.choice(same_col_cells): (2, None),
        rng.choice(diag_cells): (rng.choice((2, 3)), None),
    }

    remaining = [cell for cell in [(r, c) for r in range(3) for c in range(3)] if cell != (0, 0) and cell not in chosen]
    rng.shuffle(remaining)
    total_marked = rng.randint(4, 6)
    extras = remaining[: total_marked - len(chosen)]
    chosen[extras[0]] = (1, None)
    for cell in extras[1:]:
        chosen[cell] = (rng.randint(1, 3), None)

    color_pool = list(MARKER_COLORS)
    rng.shuffle(color_pool)
    mandatory_cells = list(chosen.keys())[:3]
    for idx, cell in enumerate(mandatory_cells):
        count, _ = chosen[cell]
        chosen[cell] = (count, color_pool[idx])
    for cell in list(chosen.keys())[3:]:
        count, _ = chosen[cell]
        chosen[cell] = (count, rng.choice(MARKER_COLORS))

    for (cell_r, cell_c), (count, color) in chosen.items():
        r0, r1, c0, c1 = _cell_bounds(row_spans, col_spans, cell_r, cell_c)
        spots = _non_adjacent_marker_positions(r1 - r0, c1 - c0, count, rng)
        for dr, dc in spots:
            grid[r0 + dr, c0 + dc] = color

    out = _transform(grid)
    _validate_semantics(grid, out)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
