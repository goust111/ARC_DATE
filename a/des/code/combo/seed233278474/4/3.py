from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PATH_COLORS = (1, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIRS4)}
START_SIDES = ("left", "right", "top", "bottom")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_separator_col(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col_color = int(grid[0, c])
        if col_color == BLACK:
            continue
        if not np.all(grid[:, c] == col_color):
            continue
        if not np.any(grid[:, :c] == RED):
            continue
        if not np.any(grid[:, :c] == GREEN):
            continue
        score = 0 if np.all(grid[:, c + 1 :] == BLACK) else 1
        candidates.append((score, c, col_color))
    assert candidates, "expected a full-height separator column"
    candidates.sort()
    _, col, color = candidates[0]
    return int(col), int(color)


def _extract_path(left: np.ndarray, sep_color: int) -> set[tuple[int, int]]:
    mask = np.ones_like(left, dtype=bool)
    for color in (BLACK, RED, GREEN, sep_color):
        mask &= left != color
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a path in the left panel"
    colors = {int(left[r, c]) for r, c in coords}
    assert len(colors) == 1, "expected a single path color"
    path = {(int(r), int(c)) for r, c in coords}

    start = next(iter(path))
    stack = [start]
    seen = {start}
    h, w = left.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) in path and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    assert seen == path, "path must be one connected 4-neighbor chain"
    return path


def _path_endpoints(path: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    endpoints = []
    for r, c in path:
        degree = sum((rr, cc) in path for rr, cc in _neighbors4(r, c, h, w))
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    return endpoints


def _closest_endpoint(marker: tuple[int, int], endpoints: list[tuple[int, int]]) -> tuple[int, int]:
    ranked = sorted(
        (
            abs(marker[0] - r) + abs(marker[1] - c),
            abs(marker[0] - r),
            abs(marker[1] - c),
            r,
            c,
        )
        for r, c in endpoints
    )
    assert ranked[0][0] < ranked[1][0], "marker must identify one endpoint unambiguously"
    return int(ranked[0][3]), int(ranked[0][4])


def _ordered_path(
    path: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int], h: int, w: int
) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    seen = {start}
    while cur != end:
        nxts = [(rr, cc) for rr, cc in _neighbors4(cur[0], cur[1], h, w) if (rr, cc) in path and (rr, cc) != prev]
        nxts = [pt for pt in nxts if pt not in seen or pt == end]
        assert len(nxts) == 1, "path traversal must be unambiguous"
        nxt = nxts[0]
        ordered.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(path), "path traversal must cover every path cell"
    return ordered


def _turn_sequence(ordered: list[tuple[int, int]]) -> list[str]:
    steps = []
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        step = (int(r1 - r0), int(c1 - c0))
        assert step in DIR_INDEX, "path must move orthogonally"
        steps.append(step)
    assert steps, "path must contain at least two cells"

    segments = [steps[0]]
    for step in steps[1:]:
        if step != segments[-1]:
            segments.append(step)

    turns = []
    for a, b in zip(segments, segments[1:]):
        delta = (DIR_INDEX[b] - DIR_INDEX[a]) % 4
        assert delta in (1, 3), "turns must be 90 degrees"
        turns.append("R" if delta == 1 else "L")
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the marked signal path and encode each turn into the right panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_col, sep_color = _find_separator_col(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size > 0 and right.size > 0, "separator must split two non-empty panels"

    red_pts = np.argwhere(left == RED)
    green_pts = np.argwhere(left == GREEN)
    assert len(red_pts) == 1 and len(green_pts) == 1, "expected one red start marker and one green end marker"
    red = tuple(map(int, red_pts[0]))
    green = tuple(map(int, green_pts[0]))

    path = _extract_path(left, sep_color)
    endpoints = _path_endpoints(path, *left.shape)
    start = _closest_endpoint(red, endpoints)
    end = _closest_endpoint(green, endpoints)
    assert start != end, "start and end endpoints must differ"

    ordered = _ordered_path(path, start, end, *left.shape)
    turns = _turn_sequence(ordered)
    assert len(turns) <= h, "right panel must have enough rows for the encoding"

    out = g.copy()
    out[:, sep_col + 1 :] = BLACK
    right_left_col = sep_col + 1
    right_right_col = w - 1
    for r, turn in enumerate(turns):
        if turn == "L":
            out[r, right_left_col] = RED
        else:
            out[r, right_right_col] = GREEN
    return out


def _extend_line(path: list[tuple[int, int]], target: tuple[int, int]) -> None:
    r, c = path[-1]
    tr, tc = target
    dr = 0 if tr == r else (1 if tr > r else -1)
    dc = 0 if tc == c else (1 if tc > c else -1)
    assert (dr == 0) ^ (dc == 0), "segments must be axis-aligned"
    while (r, c) != (tr, tc):
        r += dr
        c += dc
        path.append((r, c))


def _sample_sorted_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if lo > hi:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(80):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _next_row(h: int, current: int) -> int:
    rows = [r for r in range(1, h - 1) if r != current]
    strong = [r for r in rows if abs(r - current) >= 2]
    return random.choice(strong or rows)


def _build_base_path(h: int, w: int, max_turns: int) -> list[tuple[int, int]] | None:
    if h < 5 or w < 6:
        return None

    options = []
    max_right_jogs = min((w - 4 + 1) // 2, max_turns // 2)
    for jogs in range(1, max_right_jogs + 1):
        options.append(("right", jogs))

    max_edge_columns = (w - 4 + 1) // 2
    max_edge_jogs = min(max_edge_columns - 1, (max_turns - 1) // 2)
    for side in ("top", "bottom"):
        for jogs in range(0, max_edge_jogs + 1):
            options.append((side, jogs))
    if not options:
        return None

    final_side, jogs = random.choice(options)
    row = random.randint(1, h - 2)

    if final_side == "right":
        xs = _sample_sorted_positions(2, w - 3, jogs, 2)
        if xs is None:
            return None
        rows = [row]
        for _ in range(jogs):
            rows.append(_next_row(h, rows[-1]))
        path = [(rows[0], 0)]
        for i, x in enumerate(xs):
            _extend_line(path, (rows[i], x))
            _extend_line(path, (rows[i + 1], x))
        _extend_line(path, (rows[-1], w - 1))
        return path

    xs = _sample_sorted_positions(2, w - 3, jogs + 1, 2)
    if xs is None:
        return None
    rows = [row]
    for _ in range(jogs):
        rows.append(_next_row(h, rows[-1]))
    boundary_row = 0 if final_side == "top" else h - 1
    path = [(rows[0], 0)]
    for i in range(jogs):
        _extend_line(path, (rows[i], xs[i]))
        _extend_line(path, (rows[i + 1], xs[i]))
    _extend_line(path, (rows[-1], xs[-1]))
    _extend_line(path, (boundary_row, xs[-1]))
    return path


def _transform_path(
    path: list[tuple[int, int]], start_side: str, h: int, w: int
) -> list[tuple[int, int]]:
    if start_side == "left":
        return [(int(r), int(c)) for r, c in path]
    if start_side == "right":
        return [(int(r), int(w - 1 - c)) for r, c in path]
    if start_side == "top":
        return [(int(c), int(r)) for r, c in path]
    assert start_side == "bottom"
    return [(int(h - 1 - c), int(r)) for r, c in path]


def _marker_candidates(step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _pick_marker(
    endpoint: tuple[int, int],
    step: tuple[int, int],
    path: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int] | None:
    candidates = []
    for dr, dc in _marker_candidates(step):
        rr = endpoint[0] + dr
        cc = endpoint[1] + dc
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in path:
            candidates.append((rr, cc))
    if not candidates:
        return None
    random.shuffle(candidates)
    return candidates[0]


def _turn_count(path: list[tuple[int, int]]) -> int:
    return len(_turn_sequence(path))


def _build_instance() -> np.ndarray | None:
    h = random.randint(8, 16)
    left_w = random.randint(8, 16)
    right_w = random.randint(3, 7)
    start_side = random.choice(START_SIDES)

    if start_side in ("left", "right"):
        base_h, base_w = h, left_w
    else:
        base_h, base_w = left_w, h

    max_turns = min(h - 2, 7)
    path = _build_base_path(base_h, base_w, max_turns)
    if path is None:
        return None

    transformed = _transform_path(path, start_side, h, left_w)
    turns = _turn_count(transformed)
    if turns < 1 or turns > h:
        return None

    path_set = set(transformed)
    if len(path_set) != len(transformed):
        return None

    start = transformed[0]
    second = transformed[1]
    end = transformed[-1]
    before_end = transformed[-2]

    start_marker = _pick_marker(start, (second[0] - start[0], second[1] - start[1]), path_set, h, left_w)
    end_marker = _pick_marker(end, (end[0] - before_end[0], end[1] - before_end[1]), path_set, h, left_w)
    if start_marker is None or end_marker is None or start_marker == end_marker:
        return None

    endpoints = [start, end]
    start_dists = sorted(abs(start_marker[0] - r) + abs(start_marker[1] - c) for r, c in endpoints)
    end_dists = sorted(abs(end_marker[0] - r) + abs(end_marker[1] - c) for r, c in endpoints)
    if start_dists[0] == start_dists[1] or end_dists[0] == end_dists[1]:
        return None

    grid = np.full((h, left_w + 1 + right_w), BLACK, dtype=int)
    grid[:, left_w] = GREY
    path_color = random.choice(PATH_COLORS)
    for r, c in transformed:
        grid[r, c] = path_color
    grid[start_marker] = RED
    grid[end_marker] = GREEN

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[:, : left_w + 1], grid[:, : left_w + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
