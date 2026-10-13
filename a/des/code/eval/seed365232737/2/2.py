from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

PATH_COLORS = (GREY, GREEN, RED)

GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (15, 23)
LEGEND_GAP_CHOICES = (1, 2)
LEGEND_SIDES = ("top", "bottom", "left", "right")
GENERATION_RETRIES = 400
WALK_RETRIES = 80
WALK_MIN_VERTICES = 7
WALK_MAX_VERTICES = 13
MIN_AXIS_POINTS = 4


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _cross(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    return int(d1[0] * d2[1] - d1[1] * d2[0])


def _left_normal(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[1], d[0])


def _right_normal(d: tuple[int, int]) -> tuple[int, int]:
    return (d[1], -d[0])


def _legend_colors(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    legend_mask = (g != BLACK) & (~np.isin(g, PATH_COLORS))
    coords = np.argwhere(legend_mask)
    assert coords.size > 0, "legend must exist"

    h, w = g.shape
    assert any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords), "legend must touch a grid edge"

    colors, counts = np.unique(g[legend_mask], return_counts=True)
    assert len(colors) == 3, "legend must use exactly three colors"
    separator_candidates = colors[counts == counts.max()]
    assert len(separator_candidates) == 1 and int(counts.max()) >= 2, "legend needs one repeated separator color"
    separator = int(separator_candidates[0])

    left_turn_colors: set[int] = set()
    right_turn_colors: set[int] = set()
    for r in range(h):
        for c in range(w - 1):
            if not legend_mask[r, c] or not legend_mask[r, c + 1]:
                continue
            a = int(g[r, c])
            b = int(g[r, c + 1])
            if a != separator and b == separator:
                left_turn_colors.add(a)
            if a == separator and b != separator:
                right_turn_colors.add(b)

    assert len(left_turn_colors) == 1, "left-turn legend entry must be unique"
    assert len(right_turn_colors) == 1, "right-turn legend entry must be unique"
    return int(next(iter(left_turn_colors))), int(next(iter(right_turn_colors)))


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    path_mask = np.isin(g, PATH_COLORS)

    start_pts = np.argwhere(g == GREEN)
    end_pts = np.argwhere(g == RED)
    assert start_pts.shape[0] == 1, "expected exactly one green start"
    assert end_pts.shape[0] == 1, "expected exactly one red end"
    start = tuple(map(int, start_pts[0]))
    end = tuple(map(int, end_pts[0]))

    for r, c in np.argwhere(path_mask):
        deg = sum(bool(path_mask[rr, cc]) for rr, cc in _neighbors4(int(r), int(c), h, w))
        if (int(r), int(c)) in {start, end}:
            assert deg == 1, "each endpoint must have degree 1"
        else:
            assert int(g[int(r), int(c)]) == GREY, "only endpoints may be non-grey on the path"
            assert deg == 2, "path interior must have degree 2"

    visited = {start}
    ordered = [start]
    prev: tuple[int, int] | None = None
    curr = start
    while curr != end:
        next_cells = [
            (rr, cc)
            for rr, cc in _neighbors4(curr[0], curr[1], h, w)
            if path_mask[rr, cc] and (rr, cc) != prev
        ]
        assert len(next_cells) == 1, "path must be a single non-branching chain"
        nxt = next_cells[0]
        assert nxt not in visited, "path may not loop back on itself"
        ordered.append(nxt)
        visited.add(nxt)
        prev, curr = curr, nxt

    assert len(visited) == int(path_mask.sum()), "all path cells must lie on the traced chain"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the grey path from green to red and grow colored spikes at each turn.

    At a left turn, grow a length-2 spike on the right-hand side of the incoming
    direction, using the color read from the legend entry shaped like [color, separator].
    At a right turn, grow a length-2 spike on the left-hand side of the incoming
    direction, using the color read from the legend entry shaped like [separator, color].
    Spikes stop early at non-black cells or grid boundaries.
    """
    g = np.asarray(grid, dtype=int)
    left_color, right_color = _legend_colors(g)
    path = _trace_path(g)

    out = g.copy()
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt = path[i + 1]
        incoming = (curr[0] - prev[0], curr[1] - prev[1])
        outgoing = (nxt[0] - curr[0], nxt[1] - curr[1])
        if incoming == outgoing:
            continue

        turn = _cross(incoming, outgoing)
        assert turn in (-1, 1), "turns must be 90 degrees"
        if turn > 0:
            spike_color = left_color
            spike_dir = _right_normal(incoming)
        else:
            spike_color = right_color
            spike_dir = _left_normal(incoming)

        for step in (1, 2):
            rr = curr[0] + spike_dir[0] * step
            cc = curr[1] + spike_dir[1] * step
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                break
            if int(out[rr, cc]) != BLACK:
                break
            out[rr, cc] = spike_color

    return out


def _turn_signs(points: list[tuple[int, int]]) -> list[int]:
    signs: list[int] = []
    for i in range(1, len(points) - 1):
        d1 = (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        d2 = (points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        if d1 != d2:
            signs.append(_cross(d1, d2))
    return signs


def _axis_lines(lo: int, hi: int) -> list[list[int]]:
    options: list[list[int]] = []
    for start in (lo, lo + 1):
        if start > hi:
            continue
        vals = list(range(start, hi + 1, 2))
        if len(vals) >= MIN_AXIS_POINTS:
            options.append(vals)
    return options


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _extend_walk(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    target_len: int,
    rows: int,
    cols: int,
) -> bool:
    if len(path) == target_len:
        return True

    curr = path[-1]
    prev_dir = None
    if len(path) >= 2:
        prev = path[-2]
        prev_dir = (curr[0] - prev[0], curr[1] - prev[1])

    options: list[tuple[int, float, tuple[int, int]]] = []
    for nxt in _coarse_neighbors(curr, rows, cols):
        if nxt in used:
            continue
        step = (nxt[0] - curr[0], nxt[1] - curr[1])
        turn_bonus = 1 if prev_dir is not None and step != prev_dir else 0
        future = sum(1 for nb in _coarse_neighbors(nxt, rows, cols) if nb not in used)
        options.append((turn_bonus, future, nxt))

    random.shuffle(options)
    options.sort(key=lambda item: (item[0], item[1], random.random()), reverse=True)

    for _turn_bonus, _future, nxt in options:
        used.add(nxt)
        path.append(nxt)
        if _extend_walk(path, used, target_len, rows, cols):
            return True
        path.pop()
        used.remove(nxt)
    return False


def _build_coarse_walk(rows: int, cols: int, target_len: int) -> list[tuple[int, int]] | None:
    for _ in range(WALK_RETRIES):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}
        if not _extend_walk(path, used, target_len, rows, cols):
            continue
        if random.random() < 0.5:
            path.reverse()
        signs = _turn_signs(path)
        if len(signs) < 2:
            continue
        if not any(sign > 0 for sign in signs):
            continue
        if not any(sign < 0 for sign in signs):
            continue
        return path
    return None


def _expand_walk(points: list[tuple[int, int]], row_coords: list[int], col_coords: list[int]) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    for i, (rr_idx, cc_idx) in enumerate(points):
        rr = row_coords[rr_idx]
        cc = col_coords[cc_idx]
        if i == 0:
            expanded.append((rr, cc))
            continue
        prev_r, prev_c = expanded[-1]
        dr = int(np.sign(rr - prev_r))
        dc = int(np.sign(cc - prev_c))
        cur_r, cur_c = prev_r, prev_c
        while (cur_r, cur_c) != (rr, cc):
            cur_r += dr
            cur_c += dc
            expanded.append((cur_r, cur_c))
    return expanded


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for i, (r, c) in enumerate(path):
        if i == 0:
            grid[r, c] = GREEN
        elif i == len(path) - 1:
            grid[r, c] = RED
        else:
            grid[r, c] = GREY


def _place_legend(grid: np.ndarray) -> tuple[str, int]:
    h, w = grid.shape
    gap = random.choice(LEGEND_GAP_CHOICES)
    pattern = [MAROON, ORANGE] + [BLACK] * gap + [ORANGE, AZURE]
    span = len(pattern)
    side = random.choice(LEGEND_SIDES)

    if side == "top":
        c0 = random.randint(0, w - span)
        grid[0, c0 : c0 + span] = pattern
    elif side == "bottom":
        c0 = random.randint(0, w - span)
        grid[h - 1, c0 : c0 + span] = pattern
    elif side == "left":
        r0 = random.randint(1, h - 2)
        grid[r0, 0:span] = pattern
    else:
        r0 = random.randint(1, h - 2)
        grid[r0, w - span : w] = pattern

    return side, span


def _path_bounds(h: int, w: int, legend_side: str, legend_span: int) -> tuple[int, int, int, int] | None:
    r0, r1 = 2, h - 3
    c0, c1 = 2, w - 3

    if legend_side == "top":
        r0 = max(r0, 2)
    elif legend_side == "bottom":
        r1 = min(r1, h - 3)
    elif legend_side == "left":
        c0 = max(c0, legend_span + 1)
    else:
        c1 = min(c1, w - legend_span - 2)

    if r1 - r0 < 6 or c1 - c0 < 6:
        return None
    return r0, r1, c0, c1


def _added_spike_colors(inp: np.ndarray, out: np.ndarray) -> set[int]:
    added = (inp == BLACK) & (out != BLACK)
    return {int(v) for v in np.unique(out[added])}


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        grid = np.full((h, w), BLACK, dtype=int)

        legend_side, legend_span = _place_legend(grid)
        bounds = _path_bounds(h, w, legend_side, legend_span)
        if bounds is None:
            continue
        r0, r1, c0, c1 = bounds

        row_options = _axis_lines(r0, r1)
        col_options = _axis_lines(c0, c1)
        if not row_options or not col_options:
            continue
        row_coords = random.choice(row_options)
        col_coords = random.choice(col_options)

        max_vertices = min(WALK_MAX_VERTICES, len(row_coords) * len(col_coords) - 1)
        if max_vertices < WALK_MIN_VERTICES:
            continue
        target_len = random.randint(WALK_MIN_VERTICES, max_vertices)
        coarse_walk = _build_coarse_walk(len(row_coords), len(col_coords), target_len)
        if coarse_walk is None:
            continue

        path = _expand_walk(coarse_walk, row_coords, col_coords)
        _paint_path(grid, path)

        try:
            out = main(grid)
        except AssertionError:
            continue

        new_colors = _added_spike_colors(grid, out)
        if not ({MAROON, AZURE} <= new_colors):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
