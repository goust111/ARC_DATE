from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _endpoint_squares(g: np.ndarray, bg: int) -> dict[int, list[tuple[int, int, int, int]]]:
    h, w = g.shape
    by_color: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color == bg:
                continue
            block = g[r : r + 2, c : c + 2]
            if np.all(block == color):
                by_color[color].append((r, c, r + 1, c + 1))
    return dict(by_color)


def _square_cells(squares: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in squares:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    return cells


def _start_first(squares: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(squares, key=lambda b: (b[1], b[0]))


def _outside_connection(
    g: np.ndarray,
    square: tuple[int, int, int, int],
    color: int,
    conflict: int,
) -> tuple[int, int]:
    h, w = g.shape
    r0, c0, r1, c1 = square
    inside = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    found: set[tuple[int, int]] = set()
    for r, c in inside:
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in inside:
                continue
            if int(g[rr, cc]) in (color, conflict):
                found.add((rr, cc))
    assert len(found) == 1, f"endpoint square must touch one path cell, got {found}"
    return next(iter(found))


def _conflict_accepts_color(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for (dr1, dc1), (dr2, dc2) in (((1, 0), (-1, 0)), ((0, 1), (0, -1))):
        r1, c1 = r + dr1, c + dc1
        r2, c2 = r + dr2, c + dc2
        if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
            continue
        if int(g[r1, c1]) == color and int(g[r2, c2]) == color:
            return True
    return False


def _allowed_neighbor(
    g: np.ndarray,
    cell: tuple[int, int],
    color: int,
    conflict: int,
    blocked: set[tuple[int, int]],
) -> bool:
    r, c = cell
    h, w = g.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in blocked:
        return False
    value = int(g[r, c])
    if value == color:
        return True
    if value == conflict:
        return _conflict_accepts_color(g, r, c, color)
    return False


def _trace_path(
    g: np.ndarray,
    color: int,
    squares: list[tuple[int, int, int, int]],
    conflict: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    ordered_squares = _start_first(squares)
    start_square, end_square = ordered_squares
    blocked = _square_cells(ordered_squares)
    start = _outside_connection(g, start_square, color, conflict)
    end = _outside_connection(g, end_square, color, conflict)

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(g.size + 5):
        if cur == end:
            return path, start_square

        if int(g[cur]) == conflict and prev is not None:
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            nxt = (cur[0] + dr, cur[1] + dc)
            assert _allowed_neighbor(g, nxt, color, conflict, blocked), "crossing must continue straight"
        else:
            candidates = []
            for dr, dc in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if prev is not None and nxt == prev:
                    continue
                if _allowed_neighbor(g, nxt, color, conflict, blocked):
                    candidates.append(nxt)
            assert len(candidates) == 1, f"path color {color} has ambiguous continuation at {cur}: {candidates}"
            nxt = candidates[0]

        prev, cur = cur, nxt
        path.append(cur)

    raise AssertionError("path trace did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor woven path crossings according to alternating over-under order.

    Input structure:
    - The dominant color is background.
    - Each thread color has exactly two solid 2x2 endpoint markers and a one-pixel-wide orthogonal path between them.
    - Spatial crossings between two differently colored paths are shown as one shared conflict color instead of either thread color.

    Rule:
    - Use the matching 2x2 endpoint markers to identify each thread and trace its path from the left endpoint, or from the top endpoint when both markers share a column.
    - For each conflict pixel, find the two traced paths that pass through it.
    - Along every path independently, its first conflict is over, its second is under, and this alternates thereafter.
    - A crossing is recolored with the unique over path color; when both paths claim over, the path whose start marker appears earlier in reading order is shown.

    Key insight:
    - The gray pixels are not separate objects; they are shared path coordinates whose visible color is determined by each thread's own alternating crossing count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    endpoint_by_color = {
        color: squares
        for color, squares in _endpoint_squares(g, bg).items()
        if len(squares) == 2
    }
    assert len(endpoint_by_color) >= 2, "expected at least two path colors"

    path_colors = set(endpoint_by_color)
    non_bg = set(map(int, np.unique(g))) - {bg}
    conflict_candidates = list(non_bg - path_colors)
    assert len(conflict_candidates) == 1, f"expected one conflict color, got {conflict_candidates}"
    conflict = int(conflict_candidates[0])

    traces: dict[int, list[tuple[int, int]]] = {}
    start_markers: dict[int, tuple[int, int, int, int]] = {}
    for color, squares in endpoint_by_color.items():
        trace, start_square = _trace_path(g, int(color), squares, conflict)
        traces[int(color)] = trace
        start_markers[int(color)] = start_square

    marker_order = {
        color: i
        for i, color in enumerate(
            sorted(start_markers, key=lambda col: (start_markers[col][0], start_markers[col][1]))
        )
    }

    crossing_paths: dict[tuple[int, int], list[int]] = defaultdict(list)
    crossing_index: dict[tuple[int, tuple[int, int]], int] = {}
    for color, trace in traces.items():
        count = 0
        for cell in trace:
            if int(g[cell]) != conflict:
                continue
            crossing_paths[cell].append(color)
            crossing_index[(color, cell)] = count
            count += 1

    all_conflicts = {tuple(map(int, cell)) for cell in np.argwhere(g == conflict)}
    assert set(crossing_paths) == all_conflicts, "every conflict pixel must lie on traced paths"

    out = g.copy()
    for cell, colors in crossing_paths.items():
        assert len(colors) == 2, f"crossing {cell} must belong to two paths"
        over = [color for color in colors if crossing_index[(color, cell)] % 2 == 0]
        if len(over) == 1:
            chosen = over[0]
        else:
            chosen = min(colors if not over else over, key=lambda color: marker_order[color])
        out[cell] = int(chosen)

    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    assert ac == bc
    step = 1 if br >= ar else -1
    return [(r, ac) for r in range(ar, br + step, step)]


def _polyline(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(vertices, vertices[1:]):
        part = _segment(a, b)
        if cells:
            part = part[1:]
        cells.extend(part)
    return cells


def _paint_square(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _spaced_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    for _ in range(200):
        picked = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _render_from_paths(
    shape: tuple[int, int],
    bg: int,
    conflict: int,
    paths: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, int(bg), dtype=int)

    for path in paths:
        color = int(path["color"])
        for box in path["squares"]:
            _paint_square(inp, box, color)
        for r, c in path["order"]:
            inp[int(r), int(c)] = color

    cell_to_paths: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, path in enumerate(paths):
        for cell in path["order"]:
            cell_to_paths[cell].append(i)

    crossings = {cell: ids for cell, ids in cell_to_paths.items() if len(ids) == 2}
    if len(crossings) < 2:
        raise ValueError("not enough crossings")
    if any(len(ids) != 2 for ids in crossings.values()):
        raise ValueError("ambiguous multi-path crossing")

    for cell in crossings:
        inp[cell] = int(conflict)

    out = inp.copy()
    marker_order = {
        i: rank
        for rank, i in enumerate(
            sorted(range(len(paths)), key=lambda j: (paths[j]["start_square"][0], paths[j]["start_square"][1]))
        )
    }
    cross_rank: dict[tuple[int, tuple[int, int]], int] = {}
    for i, path in enumerate(paths):
        k = 0
        for cell in path["order"]:
            if cell in crossings:
                cross_rank[(i, cell)] = k
                k += 1

    has_unique = False
    has_tie = False
    for cell, ids in crossings.items():
        over = [i for i in ids if cross_rank[(i, cell)] % 2 == 0]
        if len(over) == 1:
            chosen = over[0]
            has_unique = True
        else:
            chosen = min(ids if not over else over, key=lambda i: marker_order[i])
            if len(over) == 2:
                has_tie = True
        out[cell] = int(paths[chosen]["color"])

    if not (has_unique and has_tie):
        raise ValueError("sample should show both unique-over and tie cases")
    return inp, out


def _many_vertical() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_vertical = random.randint(2, min(4, max(2, (w - 12) // 3 + 1)))
    cols = _spaced_positions(6, w - 7, n_vertical, 3)
    if cols is None:
        return None

    cross_row = random.randint(6, h - 7)
    bend_left = 3
    bend_right = w - 4
    bg, conflict, *path_colors = random.sample(range(10), n_vertical + 3)

    paths: list[dict] = []
    primary_color = path_colors[0]
    primary_squares = [(0, 0, 1, 1), (0, w - 2, 1, w - 1)]
    primary_order = _polyline(
        [
            (1, 2),
            (1, bend_left),
            (cross_row, bend_left),
            (cross_row, bend_right),
            (1, bend_right),
            (1, w - 3),
        ]
    )
    paths.append(
        {
            "color": primary_color,
            "squares": primary_squares,
            "start_square": primary_squares[0],
            "order": primary_order,
        }
    )

    for color, c in zip(path_colors[1:], cols):
        squares = [(0, c - 1, 1, c), (h - 2, c - 1, h - 1, c)]
        order = [(r, c) for r in range(2, h - 2)]
        paths.append({"color": color, "squares": squares, "start_square": squares[0], "order": order})

    return _render_from_paths((h, w), bg, conflict, paths)


def _many_horizontal() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_horizontal = random.randint(2, min(4, max(2, (h - 12) // 3 + 1)))
    rows = _spaced_positions(6, h - 7, n_horizontal, 3)
    if rows is None:
        return None

    cross_col = random.randint(6, w - 7)
    offset_choices = [d for d in (-2, -1, 0, 1, 2) if 2 <= cross_col + d <= w - 3]
    start_col = cross_col + random.choice(offset_choices)
    bend_top = 3
    bend_bottom = h - 4
    bg, conflict, *path_colors = random.sample(range(10), n_horizontal + 3)

    paths: list[dict] = []
    primary_color = path_colors[0]
    primary_squares = [(0, start_col - 1, 1, start_col), (h - 2, start_col - 1, h - 1, start_col)]
    primary_order = _polyline(
        [
            (2, start_col),
            (bend_top, start_col),
            (bend_top, cross_col),
            (bend_bottom, cross_col),
            (bend_bottom, start_col),
            (h - 3, start_col),
        ]
    )
    paths.append(
        {
            "color": primary_color,
            "squares": primary_squares,
            "start_square": primary_squares[0],
            "order": primary_order,
        }
    )

    for color, r in zip(path_colors[1:], rows):
        squares = [(r - 1, 0, r, 1), (r - 1, w - 2, r, w - 1)]
        order = [(r, c) for c in range(2, w - 2)]
        paths.append({"color": color, "squares": squares, "start_square": squares[0], "order": order})

    return _render_from_paths((h, w), bg, conflict, paths)


def _gen_attempt() -> np.ndarray | None:
    builder = _many_vertical if random.random() < 0.5 else _many_horizontal
    try:
        built = builder()
    except (AssertionError, ValueError):
        return None
    if built is None:
        return None
    inp, expected = built
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid woven-path input")
