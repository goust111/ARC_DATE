from __future__ import annotations

import random

import numpy as np

Coord = tuple[int, int]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dominant_color(values: np.ndarray) -> tuple[int, int]:
    uniq, counts = np.unique(values, return_counts=True)
    idx = int(np.argmax(counts))
    return int(uniq[idx]), int(counts[idx])


def _detect_dye_lines(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    h, w = grid.shape
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}

    for r in range(h):
        row = grid[r, :]
        if np.any(row == bg):
            continue
        color, count = _dominant_color(row)
        if color != bg and count * 2 > w:
            rows[int(r)] = color

    for c in range(w):
        col = grid[:, c]
        if np.any(col == bg):
            continue
        color, count = _dominant_color(col)
        if color != bg and count * 2 > h:
            cols[int(c)] = color

    assert rows or cols, "expected at least one full dye row or column"
    return rows, cols


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _color_component(grid: np.ndarray, color: int) -> list[Coord]:
    cells = {tuple(map(int, pt)) for pt in np.argwhere(grid == color)}
    assert cells, f"expected cells for color {color}"

    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    assert seen == cells, f"color {color} must form one connected path"
    return list(cells)


def _ordered_path(cells: list[Coord]) -> list[Coord]:
    cell_set = set(cells)
    adjacency: dict[Coord, list[Coord]] = {}
    for r, c in cell_set:
        adjacency[(r, c)] = [nxt for nxt in _neighbors4(r, c) if nxt in cell_set]

    degrees = {cell: len(neighbors) for cell, neighbors in adjacency.items()}
    assert all(deg <= 2 for deg in degrees.values()), "path must be one-pixel-wide without branches"

    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    left_endpoints = [cell for cell in endpoints if cell[1] == 0]
    top_endpoints = [cell for cell in endpoints if cell[0] == 0]
    if left_endpoints:
        assert len(left_endpoints) == 1, "expected a unique left-edge endpoint"
        start = left_endpoints[0]
    else:
        assert len(top_endpoints) == 1, "expected a unique top-edge endpoint when no left endpoint exists"
        start = top_endpoints[0]

    ordered = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxts = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(cell_set), "path traversal must visit every cell exactly once"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each edge-to-edge path in segments using the most recently crossed dye line.

    Input structure:
    - One color is the dominant background.
    - Several different-colored one-pixel-wide orthogonal paths connect the grid edges.
    - Some complete rows or complete columns act as dye lines; each dye line has a unique non-background color that is different from every path color.
    - Where a path crosses a dye line, the visible cell keeps the path color because the path is drawn on top.

    Rule:
    - Identify the starting endpoint of each path: use the left-edge endpoint when it exists, otherwise use the top-edge endpoint.
    - Walk along the path from that start, keeping the original path color until a dye line is crossed.
    - A crossed horizontal dye row changes the path color only after the crossing cell, and only when the path moves vertically through that row.
    - A crossed vertical dye column changes the path color only after the crossing cell, and only when the path moves horizontally through that column.
    - Dye lines themselves stay fixed; only the path pixels are recolored segment by segment.

    Key insight:
    - The full rows and columns are not ordinary foreground objects: they are hidden control lines that must be inferred first, then applied in the exact order each path encounters them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    dye_rows, dye_cols = _detect_dye_lines(g, bg)
    dye_colors = set(dye_rows.values()) | set(dye_cols.values())

    out = g.copy()
    path_colors = sorted(int(v) for v in np.unique(g) if int(v) not in dye_colors and int(v) != bg)
    assert path_colors, "expected at least one visible path color"

    for color in path_colors:
        ordered = _ordered_path(_color_component(g, color))
        current_color = int(color)
        for i, (r, c) in enumerate(ordered):
            out[r, c] = current_color

            vertical_motion = (
                (i > 0 and ordered[i - 1][0] != r)
                or (i + 1 < len(ordered) and ordered[i + 1][0] != r)
            )
            horizontal_motion = (
                (i > 0 and ordered[i - 1][1] != c)
                or (i + 1 < len(ordered) and ordered[i + 1][1] != c)
            )

            crossed: list[int] = []
            if vertical_motion and r in dye_rows:
                crossed.append(dye_rows[r])
            if horizontal_motion and c in dye_cols:
                crossed.append(dye_cols[c])

            assert len(crossed) <= 1, "simultaneous dye-line updates are ambiguous"
            if crossed:
                current_color = crossed[0]

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _sample_spaced(count: int, low: int, high: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    if high - low + 1 < count:
        return None

    for _ in range(120):
        pts = sorted(random.sample(range(low, high + 1), count))
        if all(b - a >= gap for a, b in zip(pts, pts[1:])):
            return pts
    return None


def _pick(zone: tuple[int, int]) -> int:
    lo, hi = zone
    assert lo <= hi
    return random.randint(lo, hi)


def _row_sequence(zones: list[tuple[int, int]], start_top: bool) -> list[int]:
    assert len(zones) >= 2
    if start_top:
        if len(zones) == 2:
            pattern = [1]
        else:
            mid = random.randint(1, len(zones) - 2)
            pattern = random.choice([[len(zones) - 1], [len(zones) - 1, mid], [mid, len(zones) - 1]])
    else:
        if len(zones) == 2:
            pattern = random.choice([[0, 1], [1, 0]])
        else:
            mid = random.randint(1, len(zones) - 2)
            pattern = random.choice(
                [[0, len(zones) - 1], [0, len(zones) - 1, mid], [mid, 0, len(zones) - 1], [len(zones) - 1, 0, mid]]
            )
    return [_pick(zones[idx]) for idx in pattern]


def _extend_path(cells: list[Coord], end: Coord) -> None:
    r0, c0 = cells[-1]
    r1, c1 = end
    assert r0 == r1 or c0 == c1, "segments must be orthogonal"
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r0, c))
        return

    step = 1 if r1 > r0 else -1
    for r in range(r0 + step, r1 + step, step):
        cells.append((r, c0))


def _draw_path(waypoints: list[Coord]) -> list[Coord]:
    cells = [waypoints[0]]
    for end in waypoints[1:]:
        _extend_path(cells, end)
    return cells


def _horizontal_scene() -> np.ndarray | None:
    n_lines = random.choice([2, 3, 3, 4])
    if n_lines == 2:
        n_paths = 2
    elif n_lines == 3:
        n_paths = random.choice([2, 3])
    else:
        n_paths = random.choice([2, 3])

    group_sizes = _positive_composition(n_lines, n_paths)
    start_top = random.random() < 0.5

    line_groups: list[list[int]] = []
    current = random.randint(2, 4)
    for idx, size in enumerate(group_sizes):
        rows = [current]
        for _ in range(size - 1):
            current += random.randint(2, 4)
            rows.append(current)
        line_groups.append(rows)
        if idx + 1 < len(group_sizes):
            current += random.randint(3, 5)

    last_line = line_groups[-1][-1]
    min_h = max(12, last_line + 3)
    max_h = 22
    if min_h > max_h:
        return None
    h = random.randint(min_h, max_h)
    w = random.randint(14, 24)

    total_colors = 1 + n_lines + n_paths
    if total_colors > 10:
        return None
    palette = random.sample(list(range(10)), total_colors)
    bg = palette[0]
    line_colors = palette[1 : 1 + n_lines]
    path_colors = palette[1 + n_lines :]

    grid = np.full((h, w), bg, dtype=int)
    occupied: set[Coord] = set()

    line_idx = 0
    for rows in line_groups:
        for r in rows:
            color = line_colors[line_idx]
            line_idx += 1
            grid[r, :] = color

    for path_idx, group_rows in enumerate(line_groups):
        prev_last = line_groups[path_idx - 1][-1] if path_idx > 0 else None
        next_first = line_groups[path_idx + 1][0] if path_idx + 1 < len(line_groups) else None

        upper = 0 if path_idx == 0 and start_top else (1 if prev_last is None else prev_last + 2)
        lower = (h - 2) if next_first is None else next_first - 2
        first_line = group_rows[0]
        last_line = group_rows[-1]
        if not (upper <= first_line - 1 and last_line + 1 <= lower):
            return None

        zones = [(upper, first_line - 1)]
        zones.extend((a + 1, b - 1) for a, b in zip(group_rows, group_rows[1:]))
        zones.append((last_line + 1, lower))
        if any(lo > hi for lo, hi in zones):
            return None

        is_top_path = path_idx == 0 and start_top
        rows = _row_sequence(zones, start_top=is_top_path)

        if is_top_path:
            turns = len(rows)
            cols = _sample_spaced(turns, 1, w - 3, 2)
            if cols is None:
                return None
            waypoints = [(0, cols[0]), (rows[0], cols[0])]
            for i in range(1, len(rows)):
                waypoints.append((rows[i - 1], cols[i]))
                waypoints.append((rows[i], cols[i]))
            waypoints.append((rows[-1], w - 1))
        else:
            turns = len(rows) - 1
            cols = _sample_spaced(turns, 2, w - 3, 2)
            if cols is None:
                return None
            waypoints = [(rows[0], 0)]
            for i, col in enumerate(cols):
                waypoints.append((rows[i], col))
                waypoints.append((rows[i + 1], col))
            waypoints.append((rows[-1], w - 1))

        cells = _draw_path(waypoints)
        if len(cells) != len(set(cells)):
            return None
        if any(cell in occupied for cell in cells):
            return None
        occupied.update(cells)

        path_color = path_colors[path_idx]
        for r, c in cells:
            grid[r, c] = path_color

    return grid


def _path_crossings(grid: np.ndarray, color: int, dye_rows: dict[int, int], dye_cols: dict[int, int]) -> int:
    ordered = _ordered_path(_color_component(grid, color))
    count = 0
    for i, (r, c) in enumerate(ordered):
        vertical_motion = (
            (i > 0 and ordered[i - 1][0] != r)
            or (i + 1 < len(ordered) and ordered[i + 1][0] != r)
        )
        horizontal_motion = (
            (i > 0 and ordered[i - 1][1] != c)
            or (i + 1 < len(ordered) and ordered[i + 1][1] != c)
        )
        if vertical_motion and r in dye_rows:
            count += 1
        if horizontal_motion and c in dye_cols:
            count += 1
    return count


def _generate_attempt() -> np.ndarray | None:
    grid = _horizontal_scene()
    if grid is None:
        return None

    if random.random() < 0.5:
        grid = grid.T.copy()

    bg = _background_color(grid)
    dye_rows, dye_cols = _detect_dye_lines(grid, bg)
    if bool(dye_rows) == bool(dye_cols):
        return None

    dye_colors = set(dye_rows.values()) | set(dye_cols.values())
    path_colors = [int(v) for v in np.unique(grid) if int(v) not in dye_colors and int(v) != bg]
    if len(path_colors) < 2:
        return None

    if any(_path_crossings(grid, color, dye_rows, dye_cols) == 0 for color in path_colors):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None

    detected_lines = len(dye_rows) + len(dye_cols)
    expected_lines = len(dye_colors)
    if detected_lines != expected_lines:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
