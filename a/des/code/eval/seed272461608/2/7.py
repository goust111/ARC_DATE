import random

import numpy as np


BG = 8
PATH = 5
START = 4
RED = 2
GREEN = 3

PATH_COLORS = (PATH, START)
CODE_COLORS = (RED, GREEN)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors(r: int, c: int, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    path_mask = np.isin(grid, PATH_COLORS)
    start_cells = np.argwhere(grid == START)
    assert start_cells.shape[0] == 1, "expected one yellow start marker"

    start = tuple(map(int, start_cells[0]))
    assert len(_neighbors(start[0], start[1], path_mask)) == 1, "yellow cell must be a path endpoint"

    order = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        nxts = [cell for cell in _neighbors(cur[0], cur[1], path_mask) if cell != prev]
        assert len(nxts) <= 1, "path must not branch"
        if not nxts:
            break
        nxt = nxts[0]
        assert nxt not in seen, "path must not loop back onto itself"
        order.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert len(order) == int(path_mask.sum()), "path must be a single connected polyline"
    assert len(_neighbors(cur[0], cur[1], path_mask)) == 1, "path must end at one endpoint"
    return order


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    if len(path) < 3:
        return []

    dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return [i for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1]]


def _encoding_strip(grid: np.ndarray) -> list[int]:
    bottom = grid[-1, :]
    cols = np.flatnonzero(np.isin(bottom, CODE_COLORS))
    if len(cols) == 0:
        return []

    assert np.all(cols == np.arange(cols[0], cols[-1] + 1)), "code strip must be contiguous"
    code = [int(bottom[c]) for c in cols]
    assert all(code[i] != code[i + 1] for i in range(len(code) - 1)), "code strip must alternate"
    return code


def _colorized_output(grid: np.ndarray, path: list[tuple[int, int]], code: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[np.isin(out, PATH_COLORS)] = BG

    turns = _turn_indices(path)
    assert len(turns) == len(code), "turn count must match code strip length"

    boundaries = [0, *turns, len(path)]
    segment_colors = [START, *code]
    for color, begin, end in zip(segment_colors, boundaries, boundaries[1:]):
        for idx in range(begin, end):
            r, c = path[idx]
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each path segment according to the red/green strip indexed by turn order.

    Input structure:
    - Background is azure.
    - A single-cell-wide grey orthogonal path runs from one side to the other.
    - The yellow cell marks the start endpoint.
    - The bottom row contains a contiguous alternating red/green strip whose
      length equals the number of turns in the path.

    Rule:
    - Trace the path from the yellow endpoint.
    - Cells before the first turn become yellow.
    - At the k-th turn, read the k-th strip pixel from left to right and color
      the turn cell plus the following straight segment with that color.
    """
    g = np.asarray(grid, dtype=int)
    path = _trace_path(g)
    code = _encoding_strip(g)
    return _colorized_output(g, path, code)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    assert r0 == r1 or c0 == c1, "polyline segments must be orthogonal"

    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]

    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        segment = _line_cells(vertices[i], vertices[i + 1])
        if i:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    cells = set(path)
    if len(cells) != len(path) or len(path) < 2:
        return False

    endpoints = 0
    for r, c in path:
        degree = sum((r + dr, c + dc) in cells for dr, dc in DIRS4)
        if degree == 1:
            endpoints += 1
        elif degree != 2:
            return False
    return endpoints == 2


def _sample_spaced_positions(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if hi < lo:
        return None

    span = hi - lo + 1
    needed = count + (count - 1) * (min_gap - 1)
    if span < needed:
        return None

    compressed = span - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[i] + i * (min_gap - 1) for i in range(count)]


def _choose_row(current: int, rows: list[int], min_sep: int) -> int:
    far = [row for row in rows if abs(row - current) >= min_sep]
    if far:
        return int(random.choice(far))
    near = [row for row in rows if row != current]
    assert near, "need at least two usable rows"
    return int(random.choice(near))


def _build_vertices(
    height: int,
    width: int,
    *,
    start_vertical: bool,
    end_vertical: bool,
    internal_verticals: int,
) -> list[tuple[int, int]] | None:
    rows = list(range(height - 2))
    assert len(rows) >= 2

    left_pad = 2 if start_vertical else 1
    right_pad = 3 if end_vertical else 2
    xs = _sample_spaced_positions(
        internal_verticals,
        left_pad,
        width - right_pad,
        min_gap=2,
    )
    if xs is None:
        return None

    start_row = int(random.choice(rows))
    vertices = [(start_row, 0)]
    current_row = start_row

    if start_vertical:
        current_row = _choose_row(current_row, rows, min_sep=2)
        vertices.append((current_row, 0))

    for x in xs:
        vertices.append((current_row, int(x)))
        current_row = _choose_row(current_row, rows, min_sep=2)
        vertices.append((current_row, int(x)))

    vertices.append((current_row, width - 1))
    if end_vertical:
        current_row = _choose_row(current_row, rows, min_sep=2)
        vertices.append((current_row, width - 1))

    return vertices


def _render_input(height: int, width: int, vertices: list[tuple[int, int]], code: list[int], strip_col: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    path = _path_from_vertices(vertices)
    for r, c in path:
        grid[r, c] = PATH
    sr, sc = path[0]
    grid[sr, sc] = START
    if code:
        grid[height - 1, strip_col : strip_col + len(code)] = np.array(code, dtype=int)
    return grid


def _alternating_code(length: int) -> list[int]:
    first = random.choice(CODE_COLORS)
    second = GREEN if first == RED else RED
    return [first if i % 2 == 0 else second for i in range(length)]


GEN_HEIGHT_RANGE = (10, 18)
GEN_WIDTH_RANGE = (14, 24)
GEN_INTERNAL_VERTICAL_MAX = 3
GEN_MAX_ATTEMPTS = 300


def _candidate_modes(width: int) -> list[tuple[bool, bool, int]]:
    modes: list[tuple[bool, bool, int]] = []
    for start_vertical in (False, True):
        for end_vertical in (False, True):
            min_internal = 1 if not start_vertical and not end_vertical else 0
            for internal_verticals in range(min_internal, GEN_INTERNAL_VERTICAL_MAX + 1):
                left_pad = 2 if start_vertical else 1
                right_pad = 3 if end_vertical else 2
                if _sample_spaced_positions(
                    internal_verticals,
                    left_pad,
                    width - right_pad,
                    min_gap=2,
                ) is None:
                    continue
                modes.append((start_vertical, end_vertical, internal_verticals))
    return modes


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    modes = _candidate_modes(width)
    if not modes:
        return None

    start_vertical, end_vertical, internal_verticals = random.choice(modes)
    vertices = _build_vertices(
        height,
        width,
        start_vertical=start_vertical,
        end_vertical=end_vertical,
        internal_verticals=internal_verticals,
    )
    if vertices is None:
        return None

    path = _path_from_vertices(vertices)
    if not _is_simple_path(path):
        return None

    turn_count = int(start_vertical) + 2 * internal_verticals + int(end_vertical)
    assert turn_count >= 1
    code = _alternating_code(turn_count)
    strip_col = random.randint(0, width - len(code))
    grid = _render_input(height, width, vertices, code, strip_col)

    expected = _colorized_output(grid, path, code)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-strip ARC input")
