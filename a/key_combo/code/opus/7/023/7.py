from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SNAKE_COLORS = (1, 2, 3)
LOCATOR_COLORS = (4, 7)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid))


def _find_locator_pair(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    colors = [
        color for color in _colors_present(g)
        if color not in {BG, *SNAKE_COLORS}
    ]

    candidates: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
    for pair in itertools.combinations(colors, 2):
        mask = np.isin(g, pair)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) == 0 or len(cols) == 0:
            continue
        if int(mask.sum()) != len(rows) * len(cols):
            continue
        if set(int(v) for v in np.unique(g[mask])) != set(pair):
            continue
        candidates.append((pair, rows, cols))

    assert len(candidates) == 1, "expected a unique two-color locator pair"
    return candidates[0]


def _trace_snake(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    pts = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(g, SNAKE_COLORS))]
    assert pts, "expected a snake"

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [pt for pt in pts if deg[pt] == 1]
    assert len(ends) == 2, "snake must have exactly two endpoints"
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    while True:
        order.append(cur)
        nbrs = [pt for pt in _neighbors4(*cur) if pt in pset and pt != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "snake must be a simple path"
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts), "snake traversal must visit every pixel"
    return order


def _segment_inside_rect(
    order: list[tuple[int, int]],
    rect: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = rect
    inside = [r0 <= r <= r1 and c0 <= c <= c1 for r, c in order]
    if not any(inside):
        return None

    first = inside.index(True)
    last = len(inside) - 1 - inside[::-1].index(True)
    if not all(inside[first:last + 1]):
        return None
    return first, last


def _turn_count(coords: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(coords) - 1):
        dr1 = coords[i][0] - coords[i - 1][0]
        dc1 = coords[i][1] - coords[i - 1][1]
        dr2 = coords[i + 1][0] - coords[i][0]
        dc2 = coords[i + 1][1] - coords[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """Read the snake segment inside the locator-defined window as a single column.

    Input structure:
    - Black background surrounds one 4-connected snake using colors 1/2/3 and one disjoint two-color locator using colors 4 and 7.
    - The locator pixels occupy every cell of a unique row-set/column-set cross product.
    - The snake has exactly two endpoints and enters the locator window at most once.

    Rule:
    - Find the unique non-snake color pair whose union exactly fills its active rows x active columns cross product.
    - Turn that pair's active rows and columns into a window by taking the bounding rectangle from min row/col to max row/col.
    - Trace the snake from the top-left-most endpoint and keep the consecutive snake pixels that lie inside that window.
    - Output those snake colors in traversal order as a Kx1 column.

    Key insight:
    The yellow-orange product is only a locator: it defines where to crop the snake traversal, and the answer is the snake's ordered subpath rather than the product region itself.
    """
    g = np.asarray(grid, dtype=int)

    _pair, rows, cols = _find_locator_pair(g)
    rect = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    order = _trace_snake(g)
    interval = _segment_inside_rect(order, rect)
    assert interval is not None, "window must contain a contiguous snake segment"
    start, end = interval
    segment = order[start:end + 1]
    assert len(segment) >= 1

    out = np.array([int(g[r, c]) for r, c in segment], dtype=int)
    return out.reshape((-1, 1))


def _can_step(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = nxt
    h, w = shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if nxt in used:
        return False
    for nb in _neighbors4(r, c):
        if nb in used and nb != cur:
            return False
    return True


def _build_snake_path(height: int, width: int, target_len: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        start = (random.randint(1, height - 2), random.randint(1, width - 2))
        path = [start]
        used = {start}

        def dfs() -> bool:
            if len(path) == target_len:
                return True

            cur = path[-1]
            candidates = [nb for nb in _neighbors4(*cur) if _can_step(nb, cur, used, (height, width))]
            random.shuffle(candidates)
            candidates.sort(
                key=lambda pt: (
                    abs(pt[0] - height // 2) + abs(pt[1] - width // 2),
                    random.random(),
                )
            )

            for nxt in candidates:
                path.append(nxt)
                used.add(nxt)
                if len(path) < target_len:
                    future = [nb for nb in _neighbors4(*nxt) if _can_step(nb, nxt, used, (height, width))]
                    if not future and len(path) != target_len:
                        used.remove(nxt)
                        path.pop()
                        continue
                if dfs():
                    return True
                used.remove(nxt)
                path.pop()
            return False

        if dfs():
            if path[-1] < path[0]:
                path.reverse()
            turns = _turn_count(path)
            if 6 <= turns <= 16:
                return path
    return None


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _random_partition(total: int, parts: int, min_len: int, max_len: int) -> list[int] | None:
    if parts * min_len > total or parts * max_len < total:
        return None
    out = [min_len] * parts
    remaining = total - parts * min_len
    while remaining > 0:
        idxs = [i for i, val in enumerate(out) if val < max_len]
        if not idxs:
            return None
        idx = random.choice(idxs)
        out[idx] += 1
        remaining -= 1
    random.shuffle(out)
    return out


def _assign_snake_colors(
    path: list[tuple[int, int]],
    segment_range: tuple[int, int],
    rect: tuple[int, int, int, int],
) -> list[int] | None:
    path_index = {coord: idx for idx, coord in enumerate(path)}
    seg_start, seg_end = segment_range
    segment_coords = path[seg_start:seg_end + 1]
    row_major_idxs = [path_index[coord] for coord in sorted(segment_coords)]

    for _ in range(200):
        palette = random.sample(list(SNAKE_COLORS), random.randint(2, 3))
        run_count = random.randint(max(3, len(palette)), min(8, len(path)))
        lengths = _random_partition(len(path), run_count, 1, 5)
        if lengths is None:
            continue

        run_colors: list[int] = []
        available = palette[:]
        random.shuffle(available)
        for i in range(run_count):
            choices = palette[:]
            if i > 0:
                choices = [c for c in choices if c != run_colors[-1]]
            if available:
                forced = [c for c in choices if c in available]
                if forced:
                    color = random.choice(forced)
                    available.remove(color)
                else:
                    color = random.choice(choices)
            else:
                color = random.choice(choices)
            run_colors.append(color)

        colors: list[int] = []
        for length, color in zip(lengths, run_colors):
            colors.extend([color] * length)

        seg_colors = colors[seg_start:seg_end + 1]
        if len(set(seg_colors)) < 2:
            continue
        if seg_colors == list(reversed(seg_colors)):
            continue
        row_major_colors = [colors[idx] for idx in row_major_idxs]
        if seg_colors == row_major_colors:
            continue

        r0, c0, r1, c1 = rect
        if not any(
            (r, c) != segment_coords[0] and (r0 <= r <= r1 and c0 <= c <= c1)
            for r, c in segment_coords[1:]
        ):
            continue
        return colors
    return None


def _locator_options(
    rect: tuple[int, int, int, int],
    snake_set: set[tuple[int, int]],
    segment: list[tuple[int, int]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    r0, c0, r1, c1 = rect
    row_values = list(range(r0, r1 + 1))
    col_values = list(range(c0, c1 + 1))

    row_opts: list[tuple[int, ...]] = []
    for size in range(2, min(4, len(row_values)) + 1):
        for rows in itertools.combinations(row_values, size):
            if rows[0] == r0 and rows[-1] == r1:
                row_opts.append(rows)

    col_opts: list[tuple[int, ...]] = []
    for size in range(2, min(4, len(col_values)) + 1):
        for cols in itertools.combinations(col_values, size):
            if cols[0] == c0 and cols[-1] == c1:
                col_opts.append(cols)

    segment_set = set(segment)
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for rows in row_opts:
        row_set = set(rows)
        for cols in col_opts:
            col_set = set(cols)
            if len(rows) * len(cols) < 4:
                continue
            if any((r, c) in snake_set for r in rows for c in cols):
                continue
            if len(rows) == len(row_values) and len(cols) == len(col_values):
                continue
            if not any((r not in row_set) or (c not in col_set) for r, c in segment_set):
                continue
            out.append((rows, cols))
    return out


def _choose_window_and_locator(
    path: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...], tuple[int, int]] | None:
    height, width = shape
    snake_set = set(path)
    path_index = {coord: idx for idx, coord in enumerate(path)}
    seg_len_max = min(12, len(path) - 1)

    for _ in range(500):
        base_len = random.randint(4, seg_len_max)
        start = random.randint(0, len(path) - base_len)
        end = start + base_len - 1
        br0, bc0, br1, bc1 = _bbox(path[start:end + 1])

        top = random.randint(0, min(2, br0))
        left = random.randint(0, min(2, bc0))
        bottom = random.randint(0, min(2, height - 1 - br1))
        right = random.randint(0, min(2, width - 1 - bc1))
        rect = (br0 - top, bc0 - left, br1 + bottom, bc1 + right)

        rect_h = rect[2] - rect[0] + 1
        rect_w = rect[3] - rect[1] + 1
        if rect_h < 2 or rect_w < 2 or rect_h > 8 or rect_w > 8:
            continue

        interval = _segment_inside_rect(path, rect)
        if interval is None:
            continue
        seg_start, seg_end = interval
        if seg_end - seg_start + 1 < 4:
            continue
        if seg_start == 0 and seg_end == len(path) - 1:
            continue

        segment = path[seg_start:seg_end + 1]
        if _turn_count(segment) < 1:
            continue
        row_major_indices = [path_index[coord] for coord in sorted(segment)]
        segment_indices = list(range(seg_start, seg_end + 1))
        if segment_indices == row_major_indices or segment_indices == row_major_indices[::-1]:
            continue

        options = _locator_options(rect, snake_set, segment)
        if not options:
            continue

        weighted: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for rows, cols in options:
            row_gap = len(rows) < rect_h
            col_gap = len(cols) < rect_w
            if row_gap and col_gap:
                weighted.extend([(rows, cols)] * 4)
            elif row_gap or col_gap:
                weighted.extend([(rows, cols)] * 2)
        if not weighted:
            weighted = options
        rows, cols = random.choice(weighted)
        return rect, rows, cols, (seg_start, seg_end)

    return None


def _paint_locator(
    grid: np.ndarray,
    rows: tuple[int, ...],
    cols: tuple[int, ...],
) -> None:
    cells = [(r, c) for r in rows for c in cols]
    for r, c in cells:
        grid[r, c] = random.choice(LOCATOR_COLORS)
    values = [int(grid[r, c]) for r, c in cells]
    if len(set(values)) == 1:
        r, c = random.choice(cells)
        grid[r, c] = LOCATOR_COLORS[0] if values[0] == LOCATOR_COLORS[1] else LOCATOR_COLORS[1]


def _expected_output(
    path: list[tuple[int, int]],
    colors: list[int],
    segment_range: tuple[int, int],
) -> np.ndarray:
    start, end = segment_range
    return np.array(colors[start:end + 1], dtype=int).reshape((-1, 1))


def _validate_instance(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    segment_range: tuple[int, int],
    expected: np.ndarray,
) -> bool:
    try:
        _pair, got_rows, got_cols = _find_locator_pair(grid)
        got_rect = (int(got_rows.min()), int(got_cols.min()), int(got_rows.max()), int(got_cols.max()))
        if got_rect != rect:
            return False
        if tuple(map(int, got_rows)) != rows or tuple(map(int, got_cols)) != cols:
            return False

        order = _trace_snake(grid)
        interval = _segment_inside_rect(order, rect)
        if interval != segment_range:
            return False
        segment = order[segment_range[0]:segment_range[1] + 1]
        if len(segment) < 4 or _turn_count(segment) < 1:
            return False

        row_set = set(rows)
        col_set = set(cols)
        if not any((r not in row_set) or (c not in col_set) for r, c in segment):
            return False

        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
    except AssertionError:
        return False
    return True


def _generate_input_once() -> np.ndarray | None:
    height = random.randint(13, 19)
    width = random.randint(13, 19)
    target_len = random.randint(20, 28)

    path = _build_snake_path(height, width, target_len)
    if path is None:
        return None

    chosen = _choose_window_and_locator(path, (height, width))
    if chosen is None:
        return None
    rect, rows, cols, segment_range = chosen

    colors = _assign_snake_colors(path, segment_range, rect)
    if colors is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color
    _paint_locator(grid, rows, cols)

    expected = _expected_output(path, colors, segment_range)
    if expected.shape[0] > 30:
        return None
    if not _validate_instance(grid, rect, rows, cols, segment_range, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_input_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
